"""Running one active-learning round from inside the annotator.

The round is four phases and three places they run:

    Saving your labels        main thread, synchronous, sub-second
    Collecting the frames     worker thread (decodes video, writes JPEGs)
    Training the detector     child process (`yoake train stage=1`)
    Training the behaviors    child process (`yoake train stage=2`), keyframe
                              projects only: the clips around each keyframe
    Checking what it learned  main thread, reading two small files

Nothing here touches dearpygui off the main thread. The export worker reports
through ``self._post`` like every other worker in this app, and the child
process is watched by :meth:`TrainRoundMixin.tick_train_round`, which the
render loop calls once a frame and which reads ``results.csv`` at most once a
second. Deliberately **not** :class:`monitor.TrainingMonitor`: that polls from
a daemon thread and calls back from it, which is the cross-thread bug the hub's
training page already has.

The child's output goes to a **file**, not a pipe. A pipe that nobody drains
blocks the child once the OS buffer fills, and a cancelled round then cannot be
killed because it is stuck writing. The log is in the run folder afterwards,
which is where someone diagnosing a failed round would look anyway.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..annotation import keyframes as keyframes_mod
from ..annotation import round as round_mod
from ..annotation.layout import ProjectLayout
from ..annotation.storage import atomic_json
from . import auto_label, commands, monitor, round_export

__all__ = ["TrainRoundMixin", "RoundState"]

#: How often the render loop actually reads ``results.csv``. Two small file
#: reads a second is nothing; doing it every frame at 60 fps is a needless
#: stat() storm on a spinning disk for a number that changes once a minute.
POLL_SECONDS = 1.0

#: Seconds to wait for a stopped child to exit before killing it.
STOP_GRACE = 5.0


class RoundState:
    """Everything one round owns. ``None`` on the app means no round."""

    def __init__(self, plan, layout: ProjectLayout, options: Dict[str, Any]) -> None:
        self.plan = plan
        self.layout = layout
        self.options = dict(options)
        self.number = plan.number
        self.run_dir: Path = layout.round_run(plan.number)
        self.cancel = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.proc: Optional[subprocess.Popen] = None
        self.log: Optional[Any] = None
        self.started = 0.0
        self.polled = 0.0
        self.stopping = False
        self.holdout: Tuple[round_mod.FrameKey, ...] = ()
        self.digest = ""
        self.train_json: Optional[Path] = None
        self.val_json: Optional[Path] = None
        self.retried_small_batch = False
        self.last_epoch = -1
        #: The project this round was started for. A round outlives a project
        #: switch - the child keeps training - and its result must not land in
        #: whatever project happens to be open when it ends.
        self.project = None
        self.export_failures: list = []
        self.init_weights = ""
        #: Agreement on the frames this round is about to train on, measured
        #: **before** the training starts. It has to be taken here: training
        #: adopts a new checkpoint and the next Analyze overwrites the
        #: selection, so by the time the row is written the evidence is gone.
        self.agreement: Optional[float] = None
        #: ``detector`` (stage 1) then, for a keyframe project with enough
        #: behavior labels, ``behavior`` (stage 2 on the clips, warm-started
        #: from the detector). ``run_dir`` always points at the running phase.
        self.phase = "detector"
        self.detector_dir: Path = self.run_dir
        #: The detector's score, kept when the run folder moves to stage 2:
        #: adoption compares detectors across rounds, never a behavior F1
        #: against a detector loss.
        self.detector_score: Tuple[Optional[float], str] = (None, "")
        self.stage1_weights = ""
        self.behavior_epochs = 0
        #: Why the behavior phase did not run or did not finish, for the dialog.
        self.behavior_note = ""
        #: Set when the behavior phase starts, so a failed one that falls back
        #: to the detector is never started a second time.
        self.behavior_attempted = False


class TrainRoundMixin:
    """Mixed into ``AnnotationApp`` beside :class:`WorkspaceOperations`."""

    # ── the Activity log ─────────────────────────────────────────────────
    def _round_note(self, text: str, *, failed: bool = False) -> None:
        """Record a round's start or end where it can be read back.

        A round is the longest thing this window does and its dialog can be
        closed at any point - including by the user who then goes to lunch.
        The four sentences that make up its story therefore go somewhere that
        outlives the dialog, which is the Activity window. Everything else it
        says is already in ``round.log`` beside the checkpoint.

        Guarded by ``getattr``: the mixin is also exercised by tests that build
        the app far enough to run a round and no further.
        """
        note = getattr(self, "note_activity", None)
        if note is not None:
            note(text, failed=failed)

    # ── opening ──────────────────────────────────────────────────────────
    def open_train_round(self) -> None:
        """Show the round dialog, with this project's plan already quoted.

        A round already running gets its window back **as it is**. Re-showing
        it would reset the dialog to the form, which hides Stop - and Stop is
        the only way to reach a running trainer, so the user would be left
        watching a job they could not end.
        """
        if getattr(self, "_round", None) is not None or self.round_dialog.running:
            self.round_dialog.reopen()
            return
        blocked = self.train_round_blocked()
        if blocked:
            self._status(blocked)
            if self.project_dir is None:
                self.open_settings("project")
            return
        self.round_dialog.set_capability(gpu_reason=self._gpu_reason())
        self.round_dialog.show(plan=self.round_plan_sentence(self.round_dialog.options()))

    def _gpu_reason(self) -> str:
        """Why *Use the GPU* is off, or ``""``.

        Two different answers, and the second one is the one that was missing:
        torch can import perfectly well on a machine with no NVIDIA device and
        then train on the processor, about ten times slower. Deciding this on
        ``cap.torch`` alone quoted GPU speed to a CPU-only machine.
        """
        try:
            cap = self._capability()
        except Exception:  # noqa: BLE001
            return ""
        if not cap.torch:
            return (
                "PyTorch is not installed in this Python, so there is no GPU "
                "path and no training path either."
            )
        if not auto_label.has_gpu():
            return (
                "No CUDA device is visible on this machine, so training runs "
                "on the processor. It works; it is roughly ten times slower, "
                "and the estimate above already says so."
            )
        return ""

    def refresh_loop_gates(self) -> None:
        """Dim Train and Analyze when they cannot run, with the reason (P4).

        Only the **stable** reasons gate the button: no PyTorch, no package, no
        project folder, and - for Analyze - no checkpoint. The "too few frames"
        and "a box is labelled with a class that does not exist" checks stay as
        refusals at press time, because both change as the user draws and a
        button that flickers between live and dead while you label is worse
        than one that explains itself when pressed.
        """
        from . import widgets

        reason = self._gate_reason()
        if reason:
            widgets.locked("top_train", reason)
        else:
            widgets.unlock(
                "top_train",
                "Step 3. Train a detector on the labels you have so far, here, "
                "and make it this project's model.",
            )
        analyze = self.analyze_blocked()
        if analyze:
            widgets.locked("top_analyze", analyze)
        else:
            widgets.unlock(
                "top_analyze",
                "Step 4. Run this project's model over the videos and bring "
                "back what it was least sure about.",
            )

    def _gate_reason(self) -> str:
        """The reasons that do not change while the user is labelling."""
        if self.project_dir is None:
            return (
                "Training writes into the project folder, and this project has "
                "none yet. Settings > Project > Change folder... sets one."
            )
        try:
            cap = self._capability()
        except Exception:  # noqa: BLE001
            return ""
        if not cap.package:
            return cap.package_reason
        if not cap.torch:
            return cap.torch_reason
        return ""

    def train_round_blocked(self) -> str:
        """Why the round cannot run, as the sentence to show. ``""`` when it can.

        Every branch names the cause **and** what to do now, because a control
        that is merely off teaches that the feature is broken.
        """
        stable = self._gate_reason()
        if stable:
            return stable
        counts = self._round_counts()
        if counts.frames < round_mod.MIN_FRAMES:
            return (
                f"{counts.frames} labelled frame"
                f"{'s' if counts.frames != 1 else ''} is too few to train on. "
                f"Label about {round_mod.MIN_FRAMES} and this turns on - Pick "
                "frames to label will choose them for you."
            )
        if counts.max_class_id >= counts.classes:
            return (
                f"A box is labelled class {counts.max_class_id}, but this "
                f"project defines {counts.classes}. Add the missing class in "
                "Settings > Labels, or fix the box, then train."
            )
        return ""

    # ── the plan ─────────────────────────────────────────────────────────
    def _round_counts(self) -> round_mod.LabelCounts:
        return round_mod.count_labels(self._round_frames(), self.class_names)

    def _round_holdout(self, counts: round_mod.LabelCounts):
        """The frozen holdout, grown but never re-rolled."""
        data = self._round_frames()
        previous = [
            (str(row[0]), int(row[1]))
            for row in self._rounds_log().get("holdout", [])
            if isinstance(row, (list, tuple)) and len(row) == 2
        ]
        return round_mod.grow_holdout(data, previous, counts=counts)

    def round_plan(self, options: Dict[str, Any]):
        data = self._round_frames()
        counts = self._round_counts()
        holdout = self._round_holdout(counts)
        return self._derive(counts, holdout, options, data=data), holdout

    def _derive(self, counts, holdout, options: Dict[str, Any], *, data=None, number: int = 0):
        """One place the plan is made, so the quote and the run cannot differ.

        ``has_gpu`` asks whether a GPU **exists**, not whether torch imports:
        torch runs on the processor without complaint, and a forty-minute
        round quoted as four is a promise the window cannot keep.
        """
        gpu = (
            bool(options.get("use_gpu"))
            and self._capability().torch
            and auto_label.has_gpu()
        )
        number = number or self._next_round_number()
        start_from, classes_changed = self.warm_start_from(counts)
        warm = not round_mod.wants_cold(
            number, warm_available=bool(start_from), classes_changed=classes_changed
        )
        return round_mod.derive_plan(
            counts,
            number=number,
            has_gpu=gpu,
            effort=round_mod.EFFORT.get(str(options.get("effort", "Normal")), 1.0),
            holdout=len(holdout),
            val_kind=round_mod.val_kind(data if data is not None else self._round_frames(), holdout),
            warm=warm,
        )

    #: What a round-log row without a ``decoder_layers`` key was trained with.
    #: Rounds predating :data:`round.ROUND_DECODER_LAYERS` used the production
    #: default, so the first round after that change must start cold.
    LEGACY_DECODER_LAYERS = 4

    def warm_start_from(self, counts) -> Tuple[str, bool]:
        """``(checkpoint to carry on from, is it incompatible with this round)``.

        Two ways the previous checkpoint cannot be carried over, and neither is
        a nicety:

        * **The class list changed.** The detector head is sized by
          ``num_classes``, so loading old weights into a head of a different
          width raises even with ``strict=False``.
        * **The architecture changed.** ``init_weights`` loads with
          ``strict=False``, which does not raise on a decoder of a different
          depth - it silently leaves the mismatched half at its random
          initialisation. A round that quietly threw away its decoder and
          reported the result as a warm start is worse than one that says it
          is starting cold.
        """
        rows = [r for r in (self._rounds_log().get("rounds") or []) if isinstance(r, dict)]
        for row in reversed(rows):
            path = str(row.get("weights", "") or "")
            if not path or not Path(path).is_file():
                continue
            changed = int(row.get("classes", counts.classes) or 0) != int(counts.classes)
            layers = int(row.get("decoder_layers", self.LEGACY_DECODER_LAYERS) or 0)
            if layers != round_mod.ROUND_DECODER_LAYERS:
                changed = True
            return path, changed
        return "", False

    def _round_frames(self) -> Dict[str, Any]:
        """The labelled frames, as the shape :mod:`annotation.round` reads."""
        project = self._ensure_project()
        return {
            "videos": [
                {
                    "video_id": video.video_id,
                    "frames": [
                        {"frame_index": i, "objects": video.annotations[i]}
                        for i in sorted(video.annotations)
                        if video.annotations[i] is not None
                    ],
                }
                for video in project.videos.values()
                if video.annotations
            ]
        }

    def round_plan_sentence(self, options: Dict[str, Any]) -> str:
        plan, _holdout = self.round_plan(options)
        rows = [r for r in (self._rounds_log().get("rounds") or []) if isinstance(r, dict)]
        sentence = round_mod.plan_sentence(plan, rows)
        clip = self._round_clip()
        if clip is not None:
            # Said before the user commits (P17): a keyframe round is two runs.
            sentence += (
                " Then the behaviors train on the clip before each keyframe ("
                + keyframes_mod.describe_clip(clip)
                + "), which takes about as long again."
            )
        return sentence

    # ── the round log ────────────────────────────────────────────────────
    def _rounds_log(self) -> Dict[str, Any]:
        if self.project_dir is None:
            return {}
        path = ProjectLayout(self.project_dir).rounds_log
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_rounds_log(self, data: Dict[str, Any]) -> None:
        if self.project_dir is None:
            return
        try:
            atomic_json(ProjectLayout(self.project_dir).rounds_log, data)
        except OSError:
            pass

    def _next_round_number(self) -> int:
        """One past the highest number ever used, logged or not.

        The run folders are consulted as well as the log because a round that
        failed or was stopped writes no log row - and reusing its number would
        point the retry at the dead attempt's folder, where a stale
        ``results.csv`` and ``metrics_latest.json`` are waiting to be read as
        the new run's progress and score.
        """
        rows = self._rounds_log().get("rounds") or []
        highest = max((int(r.get("round", 0)) for r in rows if isinstance(r, dict)), default=0)
        if self.project_dir is not None:
            folder = ProjectLayout(self.project_dir).runs / "rounds"
            try:
                for path in folder.glob("round-*"):
                    try:
                        highest = max(highest, int(path.name.split("-")[-1]))
                    except ValueError:
                        continue
            except OSError:
                pass
        return highest + 1

    def _round_agreement(self) -> Optional[float]:
        """How much of the last round's proposals the human kept, ``0..1``.

        Measured over the frames the sampler chose, because those are the only
        frames where the model was asked a question and a human answered it.
        Everywhere else in the project the two agree by construction: the model
        never proposed anything, or the human never looked.

        The four counts, and why a box lands in each:

        * **accepted** - taken as drawn. The model was right.
        * **rejected** - thrown away. The model was wrong.
        * **edited**   - accepted after being moved or resized. The model found
          the animal and missed the box, which is not agreement.
        * **drawn**    - boxes on a chosen frame that no accepted candidate
          accounts for. The model missed them entirely. A miss has no proposal
          to disagree with, so it would vanish from a ratio that only counted
          verdicts - and a detector that finds one fly of two, perfectly, would
          score 100%.

        ``None`` when this round's frames have not been reviewed, which is the
        honest answer and not a zero.
        """
        project = self._ensure_project()
        accepted = rejected = edited = drawn = 0
        for video in project.videos.values():
            selection = getattr(video, "selection", None)
            if selection is None:
                continue
            session = video.review
            for frame in sorted(selection.frame_set()):
                taken = 0
                if session is not None:
                    for index, _cand in enumerate(session.candidates(frame)):
                        state = session.state(frame, index)
                        if state == auto_label.REJECTED:
                            rejected += 1
                        elif state == auto_label.ACCEPTED:
                            taken += 1
                            if session.is_edited(frame, index):
                                edited += 1
                            else:
                                accepted += 1
                objects = video.annotations.get(frame)
                if objects:
                    drawn += max(0, len(objects) - taken)
        return round_mod.agreement(accepted, rejected, edited, drawn)

    def _last_adopted(self) -> Tuple[Optional[float], str]:
        for row in reversed(self._rounds_log().get("rounds") or []):
            if isinstance(row, dict) and row.get("adopted"):
                score = row.get("score")
                return (float(score) if isinstance(score, (int, float)) else None,
                        str(row.get("holdout_digest", "")))
        return None, ""

    # ── phase 0-1: save, then collect the frames ─────────────────────────
    def start_train_round(self, options: Dict[str, Any]) -> None:
        blocked = self.train_round_blocked()
        if blocked:
            self.round_dialog.fail(blocked, step=0)
            return
        if getattr(self, "_round", None) is not None:
            self.round_dialog.fail("A round is already running.", step=0)
            return
        # A stopped round's export thread can still be inside a decode. Letting
        # a second round start beside it is how one round's late callback ends
        # up cancelling the other's - or orphaning its trainer.
        previous = getattr(self, "_round_thread", None)
        if previous is not None and previous.is_alive():
            self.round_dialog.fail(
                "The last round is still tidying up its frames. Try again in "
                "a moment.",
                step=0,
            )
            return

        plan, holdout = self.round_plan(options)
        layout = ProjectLayout(self.project_dir)
        state = RoundState(plan, layout, options)
        state.agreement = self._round_agreement()
        state.holdout = holdout
        state.digest = round_mod.holdout_digest(holdout)
        state.project = self._ensure_project()
        if plan.warm:
            state.init_weights, _changed = self.warm_start_from(plan.counts)
        self._round = state
        self._round_pending_weights = ""
        self._round_pending_project = None

        self.round_dialog.advance_to_step(0, "")
        if self._dirty:
            self.save_annotation()
        self.round_dialog.set_step(0, "done", "")

        self.round_dialog.advance_to_step(1, "")
        project = self._sync_project()
        clip = self._round_clip()
        inputs = [
            round_export.ExportInput(
                video_id=video.video_id,
                path=video.path,
                # A keyframe project trains on the clip that ends at each
                # keyframe, so the context frames are exported too (once - the
                # export skips any JPEG an earlier round already wrote).
                frames=tuple(
                    keyframes_mod.context_frames(video.annotations, clip, video.num_frames)
                    if clip is not None
                    else sorted(video.annotations)
                ),
            )
            for video in project.videos.values()
            if video.annotations
        ]
        dataset = project.dataset(layout.rounds)
        state.thread = threading.Thread(
            target=self._round_export_worker,
            args=(state, inputs, dataset),
            daemon=True,
            name="yoake-round-export",
        )
        # Kept past the state, so a later round can see that the previous
        # worker has not finished with the disk yet.
        self._round_thread = state.thread
        state.started = time.monotonic()
        state.thread.start()
        self._round_note(
            f"Round {state.number} started: {plan.counts.frames:,} labelled "
            f"frames, up to {plan.epochs} passes, batch {plan.batch_size}. "
            f"Log in {state.run_dir / 'round.log'}"
        )

    def _round_export_worker(self, state: RoundState, inputs, dataset) -> None:
        """Worker: decode the frames, write the dataset, verify it, then spawn.

        Every closure it posts carries ``state``. The main thread may have
        moved on - a stopped round, a second round, another project - and a
        callback that acts on ``self._round`` instead of on the round it came
        from will cancel or corrupt whichever one is current.
        """
        from ..data.annotation import save_annotation

        layout = state.layout
        try:
            layout.rounds.mkdir(parents=True, exist_ok=True)

            def progress(fraction: float, text: str) -> None:
                self._post(lambda f=fraction, t=text: self._round_ui(
                    state,
                    lambda: (self.round_dialog.set_progress(f, t),
                             self.round_dialog.set_step(1, "running", t)),
                ))

            result = round_export.export_frames(
                inputs, layout.rounds, cancel=state.cancel, progress=progress
            )
            state.export_failures = list(result.failed)
            if result.cancelled or state.cancel.is_set():
                self._post(lambda: self._round_stopped(
                    state, "Stopped before training. The frames collected so far are kept."
                ))
                return

            whole = layout.round_data(state.number)
            save_annotation(dataset, str(whole), preserve_derived=True)
            data = json.loads(whole.read_text(encoding="utf-8"))
            round_export.rewrite_image_paths(data)
            dropped = round_export.drop_frames(data, result.missing)
            atomic_json(whole, data)

            gone = round_export.missing_images(data, layout.rounds)
            if gone:
                sample = ", ".join(gone[:3])
                self._post(lambda n=len(gone), s=sample: self._round_fail(
                    state,
                    f"{n:,} of the frames could not be found on disk after "
                    f"exporting them (for example {s}). Training would run on "
                    "blank images and tell you nothing was wrong, so it has "
                    "not been started. Project Videos > Locate file fixes a "
                    "moved video.",
                    step=1,
                ))
                return

            # ── the plan is re-derived here, against what SURVIVED ───────
            # The first derivation was made from the project's label counts.
            # The export can drop frames - a video that moved, a frame that
            # will not decode - and every number in the plan depends on how
            # many are left. A stale batch size larger than the surviving
            # training set means zero optimizer steps, a 0.0 loss and a
            # checkpoint anyway; a holdout whose frames all failed means the
            # CLI silently validates on the training set while the dialog
            # calls the result a validation loss.
            counts = round_mod.count_labels(data, data.get("class_names") or [])
            if counts.frames < round_mod.MIN_FRAMES:
                detail = "; ".join(state.export_failures[:2])
                self._post(lambda n=counts.frames, d=detail: self._round_fail(
                    state,
                    f"Only {n} frame(s) survived the export, which is too few "
                    "to train on."
                    + (f" {d}." if d else "")
                    + " Project Videos > Locate file fixes a moved video.",
                    step=1,
                ))
                return

            survivors = set(round_mod.frame_keys(data))
            holdout = tuple(key for key in state.holdout if key in survivors)
            state.holdout = holdout
            state.digest = round_mod.holdout_digest(holdout)
            # Only the dataset size changes after export. Device and warm
            # start belong to the project that started this round, even if
            # the UI has since switched to another project.
            state.plan = round_mod.derive_plan(
                counts,
                number=state.number,
                has_gpu=state.plan.device == "cuda",
                effort=round_mod.EFFORT.get(str(state.options.get("effort", "Normal")), 1.0),
                holdout=len(holdout),
                val_kind=round_mod.val_kind(data, holdout),
                warm=state.plan.warm,
            )

            train, val = round_mod.split_dataset(data, holdout)
            state.train_json = layout.round_data(state.number, "train")
            if train is val:
                atomic_json(state.train_json, train)
                state.val_json = state.train_json
            else:
                state.val_json = layout.round_data(state.number, "val")
                atomic_json(state.train_json, train)
                atomic_json(state.val_json, val)

            note = f"{counts.frames:,} frames"
            if result.written:
                note += f", {result.written:,} newly decoded"
            if dropped:
                note += f", {dropped} unreadable and skipped"
            self._post(lambda n=note: self._round_ui(
                state,
                lambda: (self.round_dialog.set_step(1, "done", n),
                         self.round_dialog.set_plan(round_mod.plan_sentence(state.plan)),
                         self.round_dialog.set_progress(0.0, "")),
            ))
            self._post(lambda: self._round_spawn(state))
        except Exception as exc:  # noqa: BLE001 - a worker must not die silently
            self._post(lambda e=exc: self._round_fail(
                state, f"Could not build the training set: {e}", step=1
            ))

    # ── phase 2: the child ───────────────────────────────────────────────
    def _round_spawn(self, state: "RoundState", *, batch_size: int = 0) -> None:
        if getattr(self, "_round", None) is not state:
            return
        if state.cancel.is_set():
            # Stopped while the frames were being collected. The worker has
            # finished and nothing else will report this round, so the stop is
            # reported here rather than leaving the dialog saying "Stopping."
            # forever.
            self._round_stopped(
                state, "Stopped before training. The frames collected so far are kept."
            )
            return
        plan = state.plan
        batch = batch_size or plan.batch_size
        state.run_dir.mkdir(parents=True, exist_ok=True)
        if batch_size:
            # A retry inherits the dead attempt's run folder, and the trainer
            # appends to results.csv. Left in place, the first row the tick
            # reads back is the crashed attempt's, reported as the new one's
            # progress and then as its score.
            for stale in ("results.csv", "metrics_latest.json"):
                try:
                    (state.run_dir / stale).unlink()
                except OSError:
                    pass
        root = auto_label.repo_root_for(Path(__file__))
        behavior = state.phase == "behavior"
        if behavior and not batch_size:
            # A clip is ``length`` frames through the detector: half the batch.
            batch = max(1, plan.batch_size // 2)
        cmd = [sys.executable, "-u"] + commands.train_command(
            2 if behavior else 1,
            str(state.train_json),
            str(state.val_json),
            str(state.run_dir),
            epochs=state.behavior_epochs if behavior else plan.epochs,
            batch_size=batch,
            extra=round_mod.train_overrides(
                plan, init_weights=state.stage1_weights if behavior else state.init_weights
            ),
        )
        env = auto_label.device_env(plan.device, auto_label.child_env(root))
        try:
            state.log = open(state.run_dir / "round.log", "a", encoding="utf-8", errors="replace")
            state.log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
            state.log.flush()
            kwargs: Dict[str, Any] = {
                "cwd": str(root),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": state.log,
                "stderr": subprocess.STDOUT,
            }
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            state.proc = subprocess.Popen(cmd, **kwargs)
        except (OSError, ValueError) as exc:
            self._round_fail(state, f"Could not start training: {exc}", step=2)
            return
        state.started = time.monotonic()
        state.last_epoch = -1
        self.round_dialog.advance_to_step(2, f"batch {batch}, up to {plan.epochs} passes")

    def tick_train_round(self) -> None:
        """One line in the render loop: is the child still going, and where?"""
        state = getattr(self, "_round", None)
        if state is None or state.proc is None:
            return
        now = time.monotonic()
        if now - state.polled < POLL_SECONDS:
            return
        state.polled = now
        code = state.proc.poll()
        rows = monitor.read_results_csv(str(state.run_dir / "results.csv"))
        if rows:
            last = rows[-1]
            epoch = int(float(last.get("epoch", state.last_epoch) or 0))
            state.last_epoch = epoch
            value = last.get("primary_metric_value") or last.get("val_loss") or ""
            best = last.get("best_so_far") or ""
            if state.phase == "behavior":
                detail = f"behaviors: pass {epoch + 1} of {state.behavior_epochs}"
                if value:
                    detail += f" - behavior F1 {float(value):.3f}"
                self.round_dialog.set_step(2, "running", detail)
                self.round_dialog.set_progress(
                    min(1.0, (epoch + 1) / max(1, state.behavior_epochs)), ""
                )
                if code is not None:
                    self._round_finished(state, code)
                return
            detail = f"pass {epoch + 1} of {state.plan.epochs}"
            if value:
                # Stage 1 selects on validation loss, lower being better. The
                # CLI's own banner says AP50, which reads 0.0 for a long time
                # on a healthy run - a number the user would read as failure.
                detail += f" - validation loss {float(value):.4g}"
                if best:
                    detail += f" (best {float(best):.4g})"
            self.round_dialog.set_step(2, "running", detail)
            self.round_dialog.set_progress(
                min(1.0, (epoch + 1) / max(1, state.plan.epochs)), ""
            )
        elif code is None:
            waited = int(now - state.started)
            self.round_dialog.set_progress(
                None, f"Loading the model and the frames... {waited}s"
            )
        if code is not None:
            self._round_finished(state, code)

    def stop_train_round(self) -> None:
        state = getattr(self, "_round", None)
        if state is None:
            return
        # The flag goes up before the signal, so the non-zero exit that follows
        # is reported as a stop and never as an error.
        state.stopping = True
        state.cancel.set()
        proc = state.proc
        if proc is None:
            # Still collecting frames. The worker owns this round until it
            # notices the flag, so ownership stays where it is: clearing
            # ``_round`` here is what let a second round start beside a live
            # thread and have its callbacks stolen.
            self.round_dialog.set_progress(None, "Stopping.")
            return
        if proc.poll() is not None:
            self._round_stopped(state, "Stopped.")
            return

        def reap(p=proc) -> None:
            try:
                p.terminate()
                p.wait(timeout=STOP_GRACE)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=reap, daemon=True, name="yoake-round-stop").start()

    # ── phase 3: what it learned ─────────────────────────────────────────
    def _round_finished(self, state: "RoundState", code: int) -> None:
        if getattr(self, "_round", None) is not state:
            return
        self._round_close_log(state)
        if state.stopping:
            kept = (state.run_dir / "weights" / "best.pth").exists()
            self._round_stopped(
                state,
                "Stopped"
                + (f" after pass {state.last_epoch + 1}. The best checkpoint up to "
                   "there was kept in the round folder." if kept else "."),
            )
            return

        if code != 0 and state.phase == "behavior" and "out of memory" not in (
            self._round_log_tail(state).lower()
        ):
            # The detector trained fine; only the behaviors failed. Keep the
            # detector rather than throw the whole round away.
            tail = self._round_log_tail(state, lines=4)
            state.behavior_note = f"The behavior model stopped with an error (code {code}): {tail}"
            state.run_dir, state.phase = state.detector_dir, "detector"
            code = 0
        if code != 0:
            tail = self._round_log_tail(state)
            if "out of memory" in tail.lower() and not state.retried_small_batch:
                state.retried_small_batch = True
                smaller = max(1, state.plan.batch_size // 2)
                state.proc = None
                self.round_dialog.set_step(
                    2, "running", f"ran out of memory - retrying at batch {smaller}"
                )
                self._round_spawn(state, batch_size=smaller)
                return
            if "Aborted." in tail:
                self._round_fail(
                    state,
                    "The trainer could not read the training set at "
                    f"{state.train_json}. That is this window's bug, not "
                    "yours - please report the path.",
                    step=2,
                )
                return
            self._round_fail(
                state,
                f"Training stopped with an error (code {code}). The last lines "
                f"of its log:\n{tail}",
                step=2,
            )
            return

        if self._round_begin_behavior(state):
            return
        self.round_dialog.set_step(2, "done", f"{state.last_epoch + 1} passes")
        self.round_dialog.advance_to_step(3, "")
        self._round_adopt(state)

    # ── keyframe projects: the behavior phase ────────────────────────────
    def _round_clip(self):
        """The project's clip, or ``None`` for a track-labelled project."""
        if not getattr(self, "keyframe_mode", False):
            return None
        settings = self._ensure_project().settings
        return keyframes_mod.ClipSpec.from_dict(settings.get("clip") or {})

    def _round_begin_behavior(self, state: "RoundState") -> bool:
        """After the detector: train the behaviors on clips, if it is worth it.

        Warm-started from the detector just trained, which stage 2 keeps
        frozen, so the checkpoint it writes carries that same detector plus a
        behavior head. Too few behavior labels and it is skipped, said so, and
        the detector alone is adopted.
        """
        if state.phase != "detector" or state.behavior_attempted or self._round_clip() is None:
            return False
        weights = state.run_dir / "weights" / "best.pth"
        if not weights.exists():
            weights = state.run_dir / "weights" / "last.pth"
        if not weights.exists() or state.train_json is None:
            return False
        try:
            data = json.loads(Path(state.train_json).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        labelled = keyframes_mod.summarize(data, data.get("action_names") or []).with_behavior
        if labelled < round_mod.MIN_BEHAVIOR_LABELS:
            state.behavior_note = (
                f"Only {labelled} box(es) have a behavior, so this round trained the "
                f"detector only. The behavior model starts at {round_mod.MIN_BEHAVIOR_LABELS}."
            )
            return False
        state.detector_score = self._round_score(state)
        state.detector_dir = state.run_dir
        state.stage1_weights = str(weights)
        state.phase = "behavior"
        state.behavior_attempted = True
        state.run_dir = state.detector_dir.parent / "stage2"
        # Budgeted in optimizer steps, like the detector (annotation/round.py):
        # stage 2 skips keyframes with no behavior, and a clip takes half the
        # batch, so the step count per pass is its own - and the trainer's
        # 100-step warmup must not be the whole run.
        clips = sum(
            1 for video in data.get("videos") or [] for frame in video.get("frames") or []
            if any(int(o.get("action_id", -1)) >= 0 for o in frame.get("objects") or [])
        )
        steps = max(1, math.ceil(clips / max(1, state.plan.batch_size // 2)))
        state.behavior_epochs = min(
            max(4, state.plan.epochs * 2),
            max(4, state.plan.epochs // 2, math.ceil(round_mod.WARMUP_STEPS / steps)),
        )
        state.proc = None
        state.retried_small_batch = False
        self.round_dialog.set_step(
            2, "running", f"detector done ({state.last_epoch + 1} passes) - now the behaviors"
        )
        self._round_note(
            f"Round {state.number}: detector trained; training the behaviors on "
            f"{labelled} labelled boxes. Log in {state.run_dir / 'round.log'}"
        )
        self._round_spawn(state)
        return True

    def _round_adopt(self, state: RoundState) -> None:
        if state.project is not None and self._ensure_project() is not state.project:
            # The user opened another project while this was training. The
            # checkpoint belongs to the project that asked for it, and the one
            # on screen must not silently inherit it.
            self._round_close_log(state)
            self.round_dialog.fail(
                "This round finished after you opened another project, so "
                f"nothing was changed here. Its model is in {state.run_dir}.",
                step=3,
            )
            self._round = None
            return
        weights = state.run_dir / "weights" / "best.pth"
        fallback = state.run_dir / "weights" / "last.pth"
        chosen = weights if weights.exists() else (fallback if fallback.exists() else None)
        behavior_score: Tuple[Optional[float], str] = (None, "")
        if state.phase == "behavior":
            behavior_score = self._round_score(state)
            score, metric = state.detector_score
            if chosen is None and state.stage1_weights:
                chosen = Path(state.stage1_weights)
                state.behavior_note = "The behavior model wrote no checkpoint; the detector was kept."
        else:
            score, metric = self._round_score(state)
        previous_score, previous_digest = self._last_adopted()
        adopt, sentence = round_mod.adoption(
            checkpoint=chosen is not None,
            score=score,
            digest=state.digest if state.plan.comparable else "",
            previous_score=previous_score,
            previous_digest=previous_digest,
        )
        if adopt and chosen is not None:
            self._adopt_round_checkpoint(chosen)

        log = self._rounds_log()
        rows = list(log.get("rounds") or [])
        rows.append(
            {
                "round": state.number,
                "frames": state.plan.counts.frames,
                "videos": state.plan.counts.videos,
                "classes": state.plan.counts.classes,
                # Recorded so the next round can tell whether this checkpoint
                # will load into it at all. A row written before this key
                # existed is read as 4 - the production default, which is what
                # those rounds were trained with.
                "decoder_layers": round_mod.ROUND_DECODER_LAYERS,
                "warm": bool(state.plan.warm),
                "epochs": state.last_epoch + 1,
                "metric": metric,
                "score": score,
                "holdout_digest": state.digest if state.plan.comparable else "",
                "comparable": bool(state.plan.comparable),
                "run_dir": str(state.run_dir),
                "weights": str(chosen) if chosen else "",
                "adopted": bool(adopt and chosen is not None),
                # What the human kept of what the previous model proposed, on
                # the frames this round is about to train on. The one number in
                # this row that is comparable across rounds - the score beside
                # it is not, which is what ``comparable`` above records - and
                # therefore the one the series advice is read off.
                "agreement": state.agreement,
                "behavior_metric": behavior_score[1],
                "behavior_score": behavior_score[0],
                "behavior_note": state.behavior_note,
                "at": time.time(),
            }
        )
        log["rounds"] = rows
        log["holdout"] = [list(key) for key in state.holdout]
        self._write_rounds_log(log)

        body = sentence
        if behavior_score[0] is not None:
            body += f"\nBehaviors: F1 {behavior_score[0]:.3f} on the held-out keyframes."
        if state.behavior_note:
            body += "\n" + state.behavior_note
        if not state.plan.comparable and score is not None:
            body += (
                "\nThe score is measured on the frames it trained on, so it is "
                "not comparable with other rounds."
            )
        self.round_dialog.set_step(3, "done", "")
        self._round_note(f"Round {state.number} finished. {sentence} Kept in {state.run_dir}")
        self.round_dialog.finish(
            title=f"Round {state.number} finished.",
            body=body,
            where=f"Kept in {state.run_dir}",
            offer_adopt=bool(chosen is not None and not adopt),
        )
        self._round_pending_weights = str(chosen) if chosen and not adopt else ""
        self._round_pending_project = self._ensure_project() if chosen and not adopt else None
        self._round = None

    def _round_score(self, state: RoundState) -> Tuple[Optional[float], str]:
        """``best_so_far``, not the last epoch's value.

        ``metrics_latest.json`` is rewritten every validating epoch, so its
        ``primary_metric_value`` is wherever the run happened to end. The
        number worth recording is the best one it reached.
        """
        data = monitor.read_metrics_latest(str(state.run_dir / "metrics_latest.json"))
        if not isinstance(data, dict):
            return None, ""
        name = str(data.get("primary_metric", "") or "")
        for key in ("best_so_far", "primary_metric_value"):
            value = data.get(key)
            # NaN and inf are real outcomes of a diverged run, and they are
            # also what ``atomic_json`` refuses to write (allow_nan=False).
            # Letting one through would take the whole annotator down on the
            # line that records the round.
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value), name
        return None, name

    def adopt_pending_round_model(self) -> None:
        """*Use this model anyway* - for the round that scored worse."""
        path = getattr(self, "_round_pending_weights", "")
        if not path:
            return
        owner = getattr(self, "_round_pending_project", None)
        self._round_pending_weights = ""
        self._round_pending_project = None
        if owner is not self._ensure_project():
            self._status("This model belongs to another project. Open that project to use it.")
            return
        if not Path(path).is_file():
            self._status(f"The checkpoint is missing: {path}")
            return
        self._adopt_round_checkpoint(Path(path))
        log = self._rounds_log()
        for row in reversed(log.get("rounds") or []):
            if isinstance(row, dict) and row.get("weights") == path:
                row["adopted"] = True
                self._write_rounds_log(log)
                break
        self._status(f"This project now uses {Path(path).name}.")

    def _adopt_round_checkpoint(self, path: Path) -> None:
        project = self._ensure_project()
        project.settings["weights"] = str(path)
        # A round's checkpoint carries its own config, so predict needs no
        # config= argument and a stale one would only disagree with it.
        project.settings["config"] = ""
        self._mark_dirty()
        self._refresh_model_ui()

    # ── failure plumbing ─────────────────────────────────────────────────
    def _round_ui(self, state: "RoundState", action) -> None:
        """Run a progress update, but only for the round that is current."""
        if getattr(self, "_round", None) is not state:
            return
        try:
            action()
        except Exception:  # noqa: BLE001 - a progress update never fails a run
            pass

    def _round_log_tail(self, state: RoundState, lines: int = 12) -> str:
        try:
            text = (state.run_dir / "round.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log was written)"
        return "\n".join(text.strip().splitlines()[-lines:])

    def _round_close_log(self, state: RoundState) -> None:
        if state.log is not None:
            try:
                state.log.close()
            except Exception:  # noqa: BLE001
                pass
            state.log = None

    def _round_fail(self, state: "RoundState", message: str, *, step: int) -> None:
        """End ``state`` badly - and only ``state``.

        The identity check is the whole point. These are reached from closures
        a worker thread posted, and by the time one runs the main thread may
        have stopped that round, started another, or opened another project.
        Acting on ``self._round`` instead of on the round the closure came from
        is how one round's late callback cancelled the next one.
        """
        if getattr(self, "_round", None) is not state:
            return
        self._round_close_log(state)
        self._round_note(f"Round {state.number} failed: {message}", failed=True)
        self.round_dialog.fail(message, step=step)
        self._round = None

    def _round_stopped(self, state: "RoundState", message: str) -> None:
        if getattr(self, "_round", None) is not state:
            return
        self._round_close_log(state)
        self._round_note(f"Round {state.number} stopped: {message}")
        self.round_dialog.stopped(message)
        self._round = None

    def _shutdown_round(self) -> None:
        """Never leave a trainer holding the GPU after the window closes."""
        state = getattr(self, "_round", None)
        thread = getattr(self, "_round_thread", None)
        if state is None:
            # A round whose state was already cleared can still have a worker
            # writing JPEGs. It is a daemon thread and will die with the
            # process, but giving it its cancel flag first means it stops at
            # the next frame instead of mid-file.
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
            return
        state.cancel.set()
        if state.proc is not None and state.proc.poll() is None:
            try:
                state.proc.terminate()
                state.proc.wait(timeout=STOP_GRACE)
            except Exception:  # noqa: BLE001
                try:
                    state.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if state.thread is not None and state.thread.is_alive():
            state.thread.join(timeout=2.0)
        self._round_close_log(state)
        self._round = None
