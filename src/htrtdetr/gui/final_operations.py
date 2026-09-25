"""Running the final pass, and signing off what it found.

Mixed into ``AnnotationApp`` beside :class:`~htrtdetr.gui.train_operations.TrainRoundMixin`,
and split from it for the same reason that one is split from the round dialog:
the dialog is a view and this is the thing that spawns a child process, holds a
worker thread and writes to disk.

Three jobs, and the order matters:

1. **The pass.** One video at a time, ``yoake predict`` over every frame,
   detections linked by IoU, written to that video's ledger. Two hours for
   twenty hour-long videos, and resumable at video granularity - a video is
   either scanned or it is not, and re-running skips the ones that are.
2. **The walk.** The human is taken to the places worth looking at: every span
   the quality checks flagged, plus forty evenly-spaced spot checks in the
   spans that looked clean. Not every frame - 2.16M frames is 600 hours at a
   second each, and the number nobody quoted is the reason the first version of
   this design was wrong.
3. **The sign-off.** Per video, once. Only signed-off videos are exported, and
   a video that is not signed off is *named* rather than quietly dropped.

**The one rule that cannot bend.** The ledger is not ``annotations.json`` and
not ``video.review``. An hour of video with two animals in it is 216,000
boxes; the project manifest is rewritten whole on every autosave and the review
session is deep-copied on every undo, so routing the pass through either turns
a keystroke into a second of work. Sign-off moves a watermark in a small dict.
It never promotes a box into the human's labels.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..annotation import finalpass
from ..annotation.batch import BatchOptions, VideoInput, connect_detections, predict_video
from ..annotation.cache import video_fingerprint
from ..annotation.layout import ProjectLayout
from . import auto_label

__all__ = ["FinalPassMixin"]

#: Frames a second the detector manages end to end through ``yoake predict``
#: on this project's hardware, detect-only at batch 16 (measured: 350.6 fps on
#: the GPU, 308.0 fps end to end against the streaming path's 36.3). Used only
#: to quote the wait before the user commits to it.
FRAMES_PER_SECOND = 300.0


class FinalPassMixin:
    """Mixed into ``AnnotationApp``."""

    # ── can it run at all ────────────────────────────────────────────────
    def final_pass_blocked(self) -> str:
        """Why *Check every frame* cannot run. ``""`` when it can.

        The same three preconditions as a round, named in the order a user
        hits them, and each one naming the thing that fixes it rather than the
        subsystem that noticed.
        """
        if self.project_dir is None:
            return (
                "Checking every frame writes a file per video, and this "
                "project has no folder yet. Settings > Project > Change "
                "folder... sets one."
            )
        if not self._ensure_project().settings.get("weights"):
            return (
                "No model yet, so there is nothing to find animals with. "
                "Train makes one from the labels you already have."
            )
        try:
            cap = self._capability()
        except Exception:  # noqa: BLE001
            return ""
        return "" if cap.torch else cap.torch_reason

    # ── the quote ────────────────────────────────────────────────────────
    def final_pass_quote(self) -> str:
        """What the user is agreeing to, before they agree to it.

        Frames, hours, and - the part that makes the operation usable at all -
        that it can be stopped. A two-hour modal nobody may close is a two-hour
        modal nobody starts.
        """
        project = self._ensure_project()
        videos = list(project.videos.values())
        total = sum(int(v.num_frames or 0) for v in videos)
        done = sum(
            int(v.num_frames or 0)
            for v in videos
            if finalpass.PassState.from_dict(v.final_pass).scanned
        )
        left = max(0, total - done)
        hours = left / FRAMES_PER_SECOND / 3600.0
        if not total:
            return "No videos in this project yet."
        cost = (
            "about {:.0f} minutes".format(max(1.0, left / FRAMES_PER_SECOND / 60.0))
            if hours < 1.5
            else "about {:.1f} hours".format(hours)
        )
        lines = [
            f"{len(videos)} video{'s' if len(videos) != 1 else ''}, {total:,} frames. "
            + (
                f"{cost} on your GPU."
                if left
                else "Every video has already been looked at."
            )
        ]
        if done and left:
            scanned = sum(
                1 for v in videos if finalpass.PassState.from_dict(v.final_pass).scanned
            )
            lines.append(
                f"{scanned} of {len(videos)} have been done already and are skipped."
            )
        lines.append(
            "It runs one video at a time and picks up where it stopped, so you "
            "can close this window, or the machine, and start it again."
        )
        return " ".join(lines)

    # ── opening ──────────────────────────────────────────────────────────
    def open_final_pass(self) -> None:
        blocked = self.final_pass_blocked()
        project = self._ensure_project()
        names = [
            (v.path.name if v.path else v.video_id) for v in project.videos.values()
        ]
        self.final_dialog.show(videos=names, quote=self.final_pass_quote())
        if blocked:
            self.final_dialog.fail(blocked, index=None)

    # ── running it ───────────────────────────────────────────────────────
    def start_final_pass(self, _options: Dict[str, Any]) -> None:
        blocked = self.final_pass_blocked()
        if blocked:
            self.final_dialog.fail(blocked)
            return
        thread = getattr(self, "_final_thread", None)
        if thread is not None and thread.is_alive():
            self.final_dialog.fail("The pass is already running.")
            return
        if self._dirty:
            self.save_annotation()

        project = self._ensure_project()
        layout = ProjectLayout(self.project_dir)
        layout.final.mkdir(parents=True, exist_ok=True)
        weights = str(project.settings.get("weights", ""))
        options = BatchOptions(
            detect=True,
            weights=weights,
            config=str(project.settings.get("config", "")),
            device="cuda" if auto_label.has_gpu() else "cpu",
            score_threshold=float(project.settings.get("score_threshold", 0.3) or 0.3),
            connect_tracks=bool(project.settings.get("use_track_ids", True)),
            sample=False,
            # Dense, and this is the one place in the annotator where dense is
            # right. A round scans one frame in sixty because the sampler can
            # only pick a scanned frame; here every frame is the deliverable.
            detect_stride=1,
            detect_only=True,
            detect_batch=16,
        )
        order = list(project.videos.values())
        fingerprint = video_fingerprint(weights)
        self._final_cancel = threading.Event()
        self._final_started = time.time()
        self.note_activity(f"Final pass started over {len(order)} videos.")

        def work() -> None:
            self._run_final_pass(project, order, options, layout, fingerprint)

        self._final_thread = threading.Thread(
            target=work, daemon=True, name="yoake-final-pass"
        )
        self._final_thread.start()

    def stop_final_pass(self) -> None:
        cancel = getattr(self, "_final_cancel", None)
        if cancel is not None:
            cancel.set()

    def _run_final_pass(self, project, order, options, layout, fingerprint) -> None:
        """The worker. Never touches a widget; everything goes through ``_post``."""
        from ..annotation.batch import Cancelled

        cancel = self._final_cancel
        root = auto_label.repo_root_for(Path(__file__))
        done = 0
        for index, video in enumerate(order):
            if cancel.is_set():
                break
            state = finalpass.PassState.from_dict(video.final_pass)
            ledger = layout.final_ledger(video.video_id)
            # Already done by this same checkpoint: skip it without opening the
            # file. This is the whole of the resume story, and it is at video
            # granularity on purpose - a part-scanned video cannot be trusted
            # to have a consistent set of track ids, because they are assigned
            # by linking the video's detections to each other.
            if state.scanned and state.weights == fingerprint and ledger.is_file():
                self._post(
                    lambda i=index: self.final_dialog.mark(i, "skipped", "already done")
                )
                done += 1
                continue

            def progress(fraction, operation, i=index, v=video):
                self._post(
                    lambda: (
                        self.final_dialog.advance_to(i, operation),
                        self.final_dialog.set_progress(
                            (i + max(0.0, min(1.0, fraction))) / max(1, len(order)),
                            f"{v.path.name if v.path else v.video_id}: {operation}",
                        ),
                    )
                )

            try:
                rows = self._scan_one_video(video, options, root, cancel, progress)
            except Cancelled:
                break
            except Exception as exc:  # noqa: BLE001 - one video must not lose the rest
                message = f"{type(exc).__name__}: {exc}"
                self._post(
                    lambda i=index, m=message: self.final_dialog.mark(i, "failed", m)
                )
                self._post(
                    lambda v=video, m=message: self.note_activity(
                        f"Final pass failed on {v.video_id}: {m}", failed=True
                    )
                )
                continue

            finalpass.write_header(
                ledger,
                video=video.video_id,
                weights=fingerprint,
                num_frames=int(video.num_frames or 0),
                fps=float(video.fps or 25.0),
            )
            written = finalpass.append_rows(ledger, rows)
            boxes = sum(len(r) for r in rows.values())
            done += 1
            self._post(
                lambda i=index, w=written, b=boxes, v=video, f=fingerprint: (
                    self._adopt_final_scan(project, v, f, w, b),
                    self.final_dialog.mark(
                        i, "done", f"{v.num_frames:,} frames, {b:,} boxes"
                    ),
                )
            )

        self._post(lambda n=done: self._finish_final_pass(project, order, n))

    def _scan_one_video(self, video, options, root, cancel, progress) -> Dict[int, list]:
        """Every box in one video, as ``{frame: [Candidate, ...]}``.

        The detections come back without identities - ``detect_only`` has no
        tracker - so they are linked here by IoU. That is valid **because this
        pass is dense**: consecutive detections are one frame apart, which is
        the assumption :func:`connect_detections` is built on and the one a
        strided round deliberately does not make.
        """
        source = VideoInput(
            video.video_id, video.path, {}, {}, float(video.fps or 25.0),
            int(video.num_frames or 0),
        )
        with tempfile.TemporaryDirectory(prefix="yoake-final-") as temp:
            rows = predict_video(source, options, root, Path(temp), cancel, progress)
        for row in rows.values():
            for candidate in row:
                candidate.provenance.update(
                    model=Path(options.weights).name,
                    model_version=video_fingerprint(options.weights),
                    operation="final_pass",
                    human_review="pending",
                )
        if options.connect_tracks:
            rows = connect_detections(rows)
        return rows

    def _adopt_final_scan(self, project, video, fingerprint, written, boxes) -> None:
        """Record that this video has been scanned. Data first, marker second.

        The ledger is written before this runs, so a crash between the two
        leaves a complete file that nothing points at - which the next run
        overwrites - rather than a marker pointing at half a file, which the
        next run would trust.
        """
        if self.project is not project:
            return
        state = finalpass.PassState.from_dict(video.final_pass)
        video.final_pass = finalpass.PassState(
            weights=fingerprint,
            scanned=int(video.num_frames or written),
            verified=state.verified if state.weights == fingerprint else (),
            signed_off=state.signed_off if state.weights == fingerprint else None,
        ).to_dict()
        video.touch()
        self._dirty = True
        self._save_due = time.time() + 4
        self.note_activity(
            f"{video.video_id}: {boxes:,} boxes in {written:,} frames."
        )

    def _finish_final_pass(self, project, order, done) -> None:
        if self.project is not project:
            return
        cancelled = self._final_cancel.is_set()
        elapsed = time.time() - getattr(self, "_final_started", time.time())
        if cancelled:
            self.final_dialog.stopped(
                f"{done} of {len(order)} videos were looked at, and what they "
                "found is kept. Starting again carries on from the next one."
            )
            self.note_activity(f"Final pass stopped after {done} videos.")
            self._status(f"Final pass stopped after {done} of {len(order)} videos.")
            return
        minutes = max(1, int(round(elapsed / 60.0)))
        self.final_dialog.finish(
            title=f"{done} of {len(order)} videos done in {minutes} minutes.",
            body=(
                "Now check what it found. You will be taken to the places the "
                "checks flagged and to a sample of the places they did not, "
                "and you sign off one video at a time. Only what you sign off "
                "is exported."
            ),
        )
        self.note_activity(f"Final pass finished: {done} videos.")
        self.save_annotation()

    # ── the walk, and the sign-off ───────────────────────────────────────
    def final_stops(self, video=None) -> Tuple[Any, Tuple[int, ...]]:
        """``(segments, stops)`` for one video, derived from its ledger.

        Derived on every call rather than stored. The alternative is a second
        source of truth for "where should I look", which goes stale the moment
        the checks are re-run or the pass is repeated with a better model - and
        a stale work list is worse than none, because it is believed.
        """
        video = video or self._video_session
        layout = ProjectLayout(self.project_dir) if self.project_dir else None
        if layout is None or not video.num_frames:
            return (), ()
        ledger = layout.final_ledger(video.video_id)
        if not ledger.is_file():
            return (), ()
        cache = video.transient.get("final_stops")
        if cache is not None and cache[0] == video.final_pass.get("weights", ""):
            return cache[1], cache[2]
        rows = dict(finalpass.iter_rows(ledger))
        issues = [i for i in video.queue.items] if video.queue is not None else []
        segments = finalpass.segments(
            rows,
            issues,
            num_frames=int(video.num_frames),
            expected=self.quality_config.expected_animals,
            min_score=float(self.quality_config.low_score)
            if hasattr(self.quality_config, "low_score")
            else 0.0,
        )
        spots = finalpass.spot_check_frames(segments)
        stops = finalpass.stops(segments, spots)
        video.transient["final_stops"] = (
            video.final_pass.get("weights", ""),
            segments,
            stops,
        )
        return segments, stops

    def final_progress_sentence(self, video=None) -> str:
        video = video or self._video_session
        if not video.final_pass:
            return ""
        _segments, stops = self.final_stops(video)
        state = finalpass.PassState.from_dict(video.final_pass)
        return finalpass.progress_sentence(
            state, stops, num_frames=int(video.num_frames or 0),
            fps=float(video.fps or 25.0),
        )

    def goto_next_final_stop(self) -> None:
        """Take the user to the next place in this video worth looking at."""
        video = self._video_session
        _segments, stops = self.final_stops(video)
        if not stops:
            self._status(
                "This video has not been looked at yet. "
                "Project > Check every frame... runs the model over all of it."
            )
            return
        state = finalpass.PassState.from_dict(video.final_pass)
        left = [f for f in stops if not finalpass.covers(state, f)]
        if not left:
            self._status(
                "Every place in this video has been checked. Sign it off to "
                "include it in the exported dataset."
            )
            return
        nxt = next((f for f in left if f > self.current_frame_idx), left[0])
        self.goto_frame(int(nxt))
        self._status(
            f"{finalpass.visited(state, stops)} of {len(stops)} places checked "
            f"in this video. {self.final_progress_sentence(video)}"
        )

    def mark_final_stop_checked(self, span: int = 0) -> None:
        """Sign off the stretch around the current frame.

        ``span`` defaults to the segment the current frame is in, which is the
        unit the user actually looked at: they were shown a flagged span, they
        judged it, and the judgement is about the span rather than about the
        one frame the playhead happens to be on.
        """
        video = self._video_session
        segments, stops = self.final_stops(video)
        if not stops:
            return
        frame = self.current_frame_idx
        start, end = frame, frame
        for segment in segments:
            if segment.start <= frame <= segment.end:
                start, end = segment.start, segment.end
                break
        if span:
            start, end = max(0, frame - span), frame + span
        state = finalpass.mark_verified(
            finalpass.PassState.from_dict(video.final_pass), start, end
        )
        video.final_pass = state.to_dict()
        video.touch()
        self._dirty = True
        self._render_timeline()
        self._status(
            f"Checked {start:,}-{end:,}. "
            f"{finalpass.visited(state, stops)} of {len(stops)} places in this video."
        )

    def sign_off_video(self, video=None) -> None:
        """Make one video's boxes part of the exported dataset.

        The one place machine output becomes deliverable data, so it is the one
        place that asks rather than acts. It still writes nothing into
        ``annotations.json``: what it moves is a timestamp, and what it unlocks
        is the exporter.
        """
        video = video or self._video_session
        _segments, stops = self.final_stops(video)
        state = finalpass.PassState.from_dict(video.final_pass)
        ready, why = finalpass.can_sign_off(state, stops)
        if not ready:
            self._status(why)
            return
        video.final_pass = finalpass.sign_off(state, time.time()).to_dict()
        video.touch()
        self._dirty = True
        self._save_due = time.time() + 4
        self.note_activity(f"{video.video_id} signed off.")
        self._status(
            f"{video.video_id} is signed off. Its boxes are part of the "
            "exported dataset. File > Export the verified dataset... writes it."
        )

    def unsign_video(self, video=None) -> None:
        video = video or self._video_session
        state = finalpass.PassState.from_dict(video.final_pass)
        video.final_pass = finalpass.unsign(state).to_dict()
        video.touch()
        self._dirty = True
        self._status(f"{video.video_id} is no longer signed off, and is not exported.")

    # ── the review walk ──────────────────────────────────────────────────
    def start_final_review(self) -> None:
        """Begin checking what the final pass found.

        Opens the video with work left rather than whichever one happens to be
        on screen. Twenty videos is enough that "which one was I on" is a
        question the user should not have to answer, and the answer is always
        the same: the first one that is scanned and not signed off.
        """
        project = self._ensure_project()
        self.final_dialog.hide()
        for video in project.videos.values():
            state = finalpass.PassState.from_dict(video.final_pass)
            if not state.scanned or state.signed_off:
                continue
            if video is not self._video_session:
                self.switch_video(video.video_id)
            self.goto_next_final_stop()
            return
        scanned = any(
            finalpass.PassState.from_dict(v.final_pass).scanned
            for v in project.videos.values()
        )
        self._status(
            "Every video that has been looked at is signed off. "
            "File > Export the verified dataset... writes them out."
            if scanned
            else "Nothing has been looked at yet. "
            "Project > Check every frame... runs the model over all of it."
        )

    # ── the export ───────────────────────────────────────────────────────
    def _release_inputs(self):
        """``(plans, excluded)`` for the project as it stands.

        ``video_path`` is written **relative to the release folder**, the same
        way ``ProjectSession.dataset`` writes it. It is the deliverable's only
        locator - there is no image tree - so an absolute
        ``C:/Users/hayam/...`` in it would make the dataset unreadable on any
        machine but this one, which for a dataset is the whole point lost.

        ``width`` and ``height`` travel too: the annotation format's video row
        carries them, ``validate_annotation`` warns for every video without
        them, and a consumer normalising boxes has nothing to divide by.
        """
        from ..annotation import release
        from ..annotation.layout import portable_path

        project = self._ensure_project()
        base = ProjectLayout(self.project_dir).release if self.project_dir else Path(".")
        videos = [
            {
                "video_id": v.video_id,
                "video_path": portable_path(v.path, base) if v.path else "",
                "num_frames": int(v.num_frames or 0),
                "fps": float(v.fps or 25.0),
                "width": int(v.width or 0),
                "height": int(v.height or 0),
            }
            for v in project.videos.values()
        ]
        states = {v.video_id: dict(v.final_pass) for v in project.videos.values()}
        return release.release_plan(videos, states)

    def open_release_export(self) -> None:
        from ..annotation import release

        if self.project_dir is None:
            self._status(
                "Exporting writes into the project folder, and this project "
                "has none yet. Settings > Project > Change folder... sets one."
            )
            return
        plans, excluded = self._release_inputs()
        named = ", ".join(vid for vid, _why in excluded[:6])
        if len(excluded) > 6:
            named += f", and {len(excluded) - 6} more"
        self.release_dialog.show(
            summary=release.summary_sentence(plans, excluded),
            excluded=(
                f"{len(excluded)} video{'s' if len(excluded) != 1 else ''} "
                f"left out: {named}"
                if excluded
                else ""
            ),
            what=(
                "Writes datasets/release/annotations.json, "
                f"{release.size_estimate(plans)}, and a path to each source "
                "video beside it. No image files: two million frames as JPEGs "
                "is about 86 GB and twenty hours of decoding, and everything "
                "in this project reads the video directly."
            ),
        )

    def start_release_export(self, _options: Dict[str, Any]) -> None:
        from ..annotation import release

        thread = getattr(self, "_release_thread", None)
        if thread is not None and thread.is_alive():
            self.release_dialog.fail("An export is already running.")
            return
        if self._dirty:
            self.save_annotation()
        project = self._ensure_project()
        layout = ProjectLayout(self.project_dir)
        plans, excluded = self._release_inputs()
        if not plans:
            self.release_dialog.fail(
                "No video is signed off yet, so there is nothing to export. "
                "Check what the final pass found, then sign a video off."
            )
            return
        ledgers = {
            p.video_id: layout.final_ledger(p.video_id) for p in plans
        }
        annotations = {
            v.video_id: dict(v.annotations) for v in project.videos.values()
        }
        names = list(project.class_names), list(project.action_names)

        out = layout.release / "annotations.json"

        def work() -> None:
            def tick(done, total, video_id):
                self._post(
                    lambda: self.release_dialog.set_progress(
                        done / max(1, total), f"{video_id} ({done + 1} of {total})..."
                    )
                )

            try:
                # ``write_release``, not ``dataset_dict`` + ``atomic_json``.
                # The dict form holds the whole release in memory - measured
                # over 2 GB of peak for a 1.4 GB file - and ``atomic_json``
                # writes it indented, which was 37% of that file. This streams
                # one video at a time and writes it compact.
                stats = release.write_release(
                    out,
                    plans,
                    lambda vid: finalpass.iter_rows(ledgers[vid]),
                    lambda vid: annotations.get(vid, {}),
                    class_names=names[0],
                    action_names=names[1],
                    progress=tick,
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                message = f"{type(exc).__name__}: {exc}"
                self._post(lambda: self.release_dialog.fail(message))
                self._post(
                    lambda: self.note_activity(f"Export failed: {message}", failed=True)
                )
                return
            self._post(
                lambda: self._finish_release_export(project, stats, excluded, out)
            )

        self._release_thread = threading.Thread(
            target=work, daemon=True, name="yoake-release-export"
        )
        self._release_thread.start()
        self.release_dialog.set_progress(0.3, "Reading the boxes back...")

    def _finish_release_export(self, project, stats, excluded, out) -> None:
        if self.project is not project:
            return
        frames = int(stats.get("frames", 0))
        notes = []
        if excluded:
            notes.append(
                f"{len(excluded)} video{'s' if len(excluded) != 1 else ''} were "
                "left out because they are not signed off."
            )
        # A video whose ledger has been deleted or moved yields no rows, and
        # the export would otherwise report it as present with nothing in it.
        # The row count per video is the only place that can be noticed.
        empty = [vid for vid, rows in (stats.get("videos") or {}).items() if not rows]
        if empty:
            notes.append(
                f"{len(empty)} signed-off video{'s' if len(empty) != 1 else ''} "
                f"produced no rows ({', '.join(empty[:4])}) - their files under "
                "datasets/final are missing or empty."
            )
        if stats.get("dropped"):
            notes.append(
                f"{stats['dropped']:,} boxes fell outside their video's length "
                "and were left out."
            )
        self.release_dialog.finish(
            f"{frames:,} frames written. " + " ".join(notes),
            f"{out}  ({stats.get('bytes', 0) / 1_000_000:.0f} MB)",
        )
        self.note_activity(f"Exported {frames:,} frames to {out}.")
        self._status(f"Exported {frames:,} frames to {out}.")

    def _open_release_folder(self) -> None:
        if self.project_dir is None:
            return
        folder = ProjectLayout(self.project_dir).release
        folder.mkdir(parents=True, exist_ok=True)
        self._open_folder(folder)

    # ── construction ─────────────────────────────────────────────────────
    def build_final_dialogs(self, final_pass_mod, release_export_mod) -> None:
        """Create the two windows this mixin drives.

        Built with the rest of the app's dialogs rather than lazily inside the
        menu callback, so "is a dialog open" has one answer and a test can find
        them without opening a menu. The two modules are passed in rather than
        imported here because ``annotator.py`` owns the dual-import dance that
        lets this package be run both as a package and as a script.
        """
        self.final_dialog = final_pass_mod.FinalPassDialog(
            on_start=self.start_final_pass,
            on_stop=self.stop_final_pass,
            on_review=self.start_final_review,
            on_export=self.open_release_export,
        )
        self.release_dialog = release_export_mod.ReleaseExportDialog(
            on_export=self.start_release_export,
            on_open_folder=self._open_release_folder,
        )
