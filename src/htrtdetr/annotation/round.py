"""One active-learning round: what to train, for how long, and on what.

A *round* is the half of the loop the annotator could not do: take the labels
that exist right now, train a detector on them, and hand the result back so the
next pass through *Pick frames to label* has a model to be uncertain with.

Everything in this module is a **decision**, and every decision is here rather
than in the dialog, because each one of them is a trap with a silent failure
mode on the other side:

``num_classes``
    ``yoake train`` reads the class list out of the annotation file and throws
    it away (``cli.py``'s ``videos, _, _ = load_annotations(...)``); the model's
    class count comes from ``model.detector.head.num_classes``, which defaults
    to **1**. The loss pre-fills its target tensor with ``num_classes`` as the
    *background* index, so in a two-class project every ``class_id=1`` object
    would train as "no object" - with no error, no warning, and a checkpoint at
    the end.

``batch_size``
    The train loader is ``drop_last=True``. A batch larger than the number of
    training frames therefore yields **zero** optimizer steps per epoch: the
    run completes, logs ``train_loss`` 0.0, and still writes a checkpoint. A
    first round on 30 frames is exactly the size where that happens.

``epochs``
    Warmup is ``max(warmup_epochs * steps_per_epoch, 100)`` **steps**, and that
    100 cannot be lowered from config. On a tiny dataset with four steps an
    epoch, a "5 minute" 20-epoch run would spend all 80 of its steps warming
    up. So the epoch count has a floor derived from the step count, and when
    the floor wins :func:`plan_sentence` says the true number out loud.

``num_workers``
    0, not the default 8. Eight worker processes spawned per epoch for a few
    hundred images is slower than loading them inline, and on Windows the
    orphaned children outlive a cancelled round.

No torch, no cv2, no dearpygui: this module is arithmetic over a dataset dict,
so it is unit-testable from either of this project's interpreters.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "LabelCounts",
    "RoundPlan",
    "FrameKey",
    "MIN_FRAMES",
    "count_labels",
    "frame_keys",
    "grow_holdout",
    "wants_cold",
    "val_kind",
    "holdout_digest",
    "split_dataset",
    "derive_plan",
    "plan_sentence",
    "agreement",
    "series_note",
    "PLATEAU_WINDOW",
    "PLATEAU_BAND",
    "PLATEAU_AGREEMENT",
    "SERIES_HINT_FROM",
    "train_overrides",
    "adoption",
]

#: ``(video_id, frame_index)``. The identity of one labelled frame, stable
#: across rounds and across a project that has been moved.
FrameKey = Tuple[str, int]

#: Below this many labelled frames a round trains on noise. The dialog gates on
#: it rather than refusing at the end, so the user is told before they wait.
MIN_FRAMES = 10

#: Fraction of labelled frames held back to check the model against frames it
#: never saw, and the floor under that fraction.
HOLDOUT_FRACTION = 0.2
HOLDOUT_MIN = 4

#: Below this, holding frames back leaves too few to learn from, so the round
#: trains and validates on the same frames **and says so**. The alternative is
#: what the CLI does on its own: fall back silently, and then select the best
#: checkpoint on a number the UI is calling a validation loss.
HOLDOUT_VIABLE = 12

#: Seconds per optimizer step, **measured** on this project's own hardware
#: (RTX 5070 Ti, 150 frames at 640x640, batch 8): 0.6-0.8 s/step across repeated
#: runs, with about 20% variance between runs of identical settings.
#:
#: The first value here was 0.20, guessed. A round quoted at "about 12 minutes"
#: therefore took closer to 45, which is the kind of lie that stops a user
#: pressing the button a second time. The CPU figure is scaled from the GPU one
#: rather than measured, and is deliberately pessimistic: quoting too long and
#: finishing early is a good surprise.
SEC_PER_STEP = {"cuda": 0.65, "cpu": 5.20}

#: Batch size before the clamps below. Not read from VRAM - nothing in the GUI
#: process can, without importing torch - so an out-of-memory failure is
#: handled by retrying at half, which is cheap and honest.
BASE_BATCH = {"cuda": 8, "cpu": 2}

#: Steps of warmup to clear before a run is worth calling training. Four times
#: the hard 100-step floor the trainer applies.
WARMUP_STEPS = 400

#: **The round is budgeted in optimizer steps, not minutes.**
#:
#: A minute budget sounds friendlier and is the wrong unit: the number of steps
#: a minute buys depends on the label count, so the same "12 minutes" meant 400
#: steps on a small project and 3,600 on a large one - the large one silently
#: becoming a 45-minute job. Budgeting the steps makes a round cost the same
#: whatever the project has grown to, which is what "just keep the cycle
#: turning" needs.
#:
#: Cold is four times the warmup floor, so a from-scratch round leaves warmup
#: with something to spare. Warm rounds start from the last round's weights and
#: only have to move, not learn the species again.
COLD_STEPS = 400
WARM_STEPS = 150

#: What *How long* multiplies the step budget by. The user picks effort, not
#: minutes; the quote converts it back to minutes with the measured rate above.
EFFORT: Dict[str, float] = {"Quick": 0.5, "Normal": 1.0, "Thorough": 2.5}

#: Nothing a round does is worth more epochs than this; past here it is a
#: production run and belongs in the workflow window.
MAX_EPOCHS = 400

#: How often a warm round is forced cold anyway, so a chain of fine-tunes
#: cannot lock in the first round's mistakes.
COLD_EVERY = 5


@dataclass(frozen=True)
class LabelCounts:
    """What the project actually holds right now."""

    frames: int = 0
    videos: int = 0
    boxes: int = 0
    classes: int = 1
    #: The largest ``class_id`` seen. Compared against ``classes`` to catch the
    #: labels that would train as background or crash the matcher.
    max_class_id: int = 0

    @property
    def usable(self) -> bool:
        return self.frames >= MIN_FRAMES


@dataclass(frozen=True)
class RoundPlan:
    """Every number the round runs with, derived once and quoted to the user."""

    number: int = 1
    counts: LabelCounts = LabelCounts()
    device: str = "cpu"
    batch_size: int = 1
    epochs: int = 1
    patience: int = 20
    steps_per_epoch: int = 1
    n_train: int = 0
    n_val: int = 0
    #: ``"video"`` (whole videos held back), ``"frames"`` (a spread of frames),
    #: or ``"none"`` (too few labels; val is the training set).
    val_kind: str = "none"
    #: True when :data:`WARMUP_STEPS` set the epoch count rather than the time
    #: budget, which is the case the estimate must not round away.
    warmup_bound: bool = False
    #: Started from the previous round's weights rather than from ImageNet.
    #: A warm round needs a fraction of the steps, which is what makes the
    #: cycle cheap enough to turn while labelling.
    warm: bool = False
    #: How often validation runs, in epochs. Every epoch writes a 348 MB
    #: checkpoint and a validation pass, so a 100-epoch round spent more time
    #: saving and validating than training.
    val_interval: int = 1

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def est_seconds(self) -> float:
        return self.total_steps * SEC_PER_STEP.get(self.device, SEC_PER_STEP["cpu"])

    @property
    def comparable(self) -> bool:
        """Does this round produce a score worth comparing with the last one?"""
        return self.val_kind != "none"


# ── reading the dataset ──────────────────────────────────────────────────
def _videos(dataset: Any) -> List[dict]:
    """The video rows, from either the dict form or a ``DatasetAnno``."""
    if isinstance(dataset, dict):
        return list(dataset.get("videos") or [])
    return [
        {
            "video_id": getattr(v, "video_id", ""),
            "frames": [
                {
                    "frame_index": getattr(f, "frame_index", 0),
                    "objects": list(getattr(f, "objects", []) or []),
                }
                for f in getattr(v, "frames", []) or []
            ],
        }
        for v in getattr(dataset, "videos", []) or []
    ]


def _class_of(obj: Any) -> int:
    if isinstance(obj, dict):
        return int(obj.get("class_id", 0) or 0)
    return int(getattr(obj, "class_id", 0) or 0)


def count_labels(dataset: Any, class_names: Sequence[str] = ()) -> LabelCounts:
    """Count what a round would train on.

    A video with no labelled frames does not count as a video: it contributes
    nothing to training and it would otherwise inflate the holdout arithmetic
    into holding back a video that has nothing in it.
    """
    frames = boxes = 0
    videos = 0
    max_class = 0
    for video in _videos(dataset):
        rows = [f for f in video.get("frames") or []]
        if not rows:
            continue
        videos += 1
        frames += len(rows)
        for frame in rows:
            for obj in frame.get("objects") or []:
                boxes += 1
                max_class = max(max_class, _class_of(obj))
    return LabelCounts(
        frames=frames,
        videos=videos,
        boxes=boxes,
        classes=max(1, len(class_names)),
        max_class_id=max_class,
    )


def frame_keys(dataset: Any) -> List[FrameKey]:
    """Every labelled frame, in a deterministic order."""
    keys: List[FrameKey] = []
    for video in _videos(dataset):
        vid = str(video.get("video_id", ""))
        for frame in video.get("frames") or []:
            keys.append((vid, int(frame.get("frame_index", 0))))
    keys.sort()
    return keys


# ── the holdout, frozen across rounds ────────────────────────────────────
def grow_holdout(
    dataset: Any, previous: Iterable[FrameKey] = (), *, counts: LabelCounts | None = None
) -> Tuple[FrameKey, ...]:
    """The frames held back, keeping every one that was held back before.

    Freezing matters because the score is only worth showing if two rounds
    measured it with the same ruler. A holdout re-derived every round moves
    under the comparison: round 3 could score "better" than round 2 purely by
    being asked easier frames.

    It still **grows**, because a holdout frozen at the six frames a first
    round could spare stays a six-frame holdout at frame three hundred, and a
    six-frame validation loss is noise. New members are appended in key order
    and existing ones are never dropped while they are still labelled, so
    consecutive rounds overlap almost entirely - and
    :func:`holdout_digest` lets the caller say when they did not.
    """
    counts = counts or count_labels(dataset)
    available = frame_keys(dataset)
    if counts.frames < HOLDOUT_VIABLE:
        return ()
    present = set(available)
    kept = [key for key in previous if key in present]
    target = max(HOLDOUT_MIN, round(counts.frames * HOLDOUT_FRACTION))

    if not kept and counts.videos >= 3:
        # First round of a multi-video project: hold back the whole video with
        # the fewest labelled frames, so validation frames are from a recording
        # the model has never seen rather than from between frames it has.
        #
        # Only when that video is big enough to be a validation set. A project
        # with one video labelled twice would otherwise get a two-frame
        # holdout, and a validation loss over two frames is noise the adoption
        # rule would then treat as a measurement.
        by_video: Dict[str, List[FrameKey]] = {}
        for key in available:
            by_video.setdefault(key[0], []).append(key)
        smallest = min(by_video.values(), key=lambda rows: (len(rows), rows[0][0]))
        if HOLDOUT_MIN <= len(smallest) < counts.frames:
            return tuple(sorted(smallest))

    chosen = list(kept)
    if len(chosen) < target:
        held = set(chosen)
        # Every Nth remaining frame rather than the first N: a holdout taken
        # off the front of a video is a holdout of one behaviour.
        rest = [key for key in available if key not in held]
        need = min(target - len(chosen), max(0, len(available) - 1))
        if need > 0 and rest:
            stride = max(1, len(rest) // need)
            chosen.extend(rest[:: stride][:need])
    # Never hold back everything, whatever the arithmetic said.
    if len(chosen) >= len(available):
        chosen = chosen[: max(0, len(available) - 1)]
    return tuple(sorted(chosen))


def val_kind(dataset: Any, holdout: Iterable[FrameKey]) -> str:
    """What the holdout actually is, for the sentence that describes it.

    ``"video"`` only when every held-back frame belongs to a video **none** of
    whose frames are being trained on - which is the only case in which "a
    recording it never saw" is true. The first round of a multi-video project
    starts that way; by round three the holdout has usually grown into videos
    the model does train on, and the claim quietly stops being true.

    Deriving this from the holdout rather than from a count is the point. The
    first version compared the number of videos the holdout touched against the
    number in the project, which says "whole video" for any holdout that merely
    misses one video - including a spread of single frames taken from four
    videos out of five.
    """
    held = set(holdout)
    if not held:
        return "none"
    for video in _videos(dataset):
        vid = str(video.get("video_id", ""))
        rows = video.get("frames") or []
        if not rows:
            continue
        keys = {(vid, int(f.get("frame_index", 0))) for f in rows}
        if keys & held and not keys <= held:
            return "frames"
    return "video"


def holdout_digest(holdout: Iterable[FrameKey]) -> str:
    """A short, stable name for one holdout, so two rounds can be compared.

    Rounds whose digests differ measured different frames; saying "better than
    last round" across that boundary compares two rulers.
    """
    joined = "|".join(f"{vid}:{index}" for vid, index in sorted(holdout))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12] if joined else ""


def split_dataset(dataset: dict, holdout: Iterable[FrameKey]) -> Tuple[dict, dict]:
    """``(train, val)``, as two dataset dicts sharing one image tree.

    The frame rows are carried over **verbatim** - in particular their
    ``image_path`` - because the trainer resolves images against the directory
    holding the annotation file, so both halves must be written beside the same
    ``images/`` folder and must not have their paths rewritten.

    An empty holdout returns ``(dataset, dataset)``: the caller passes the same
    path twice on purpose and says so, rather than leaving the CLI to fall back
    silently and let a "validation" loss be the training loss.
    """
    held = set(holdout)
    if not held:
        return dataset, dataset

    def _side(keep: bool) -> dict:
        out = {k: v for k, v in dataset.items() if k != "videos"}
        videos = []
        for video in dataset.get("videos") or []:
            vid = str(video.get("video_id", ""))
            rows = [
                frame
                for frame in video.get("frames") or []
                if ((vid, int(frame.get("frame_index", 0))) in held) is keep
            ]
            if rows:
                videos.append({**{k: v for k, v in video.items() if k != "frames"}, "frames": rows})
        out["videos"] = videos
        return out

    return _side(False), _side(True)


# ── the plan ─────────────────────────────────────────────────────────────
def wants_cold(number: int, warm_available: bool, classes_changed: bool) -> bool:
    """Should this round start from ImageNet rather than from last time?

    Three reasons, and the third is the one that is easy to forget: a chain of
    fine-tunes each starting from the last can carry an early mistake forever,
    so every :data:`COLD_EVERY`-th round is forced cold whatever else is true.
    A changed class list is not optional - the detector head changes shape and
    loading the old weights into it raises.
    """
    if not warm_available or classes_changed:
        return True
    return int(number) % COLD_EVERY == 1 and int(number) > 1


def derive_plan(
    counts: LabelCounts,
    *,
    number: int = 1,
    has_gpu: bool = False,
    effort: float = 1.0,
    holdout: int = 0,
    val_kind: str = "none",
    warm: bool = False,
) -> RoundPlan:
    """Turn "what I have" and "how hard to try" into every other number.

    The budget is **steps**, not minutes (:data:`COLD_STEPS`). The duration is
    then quoted *from* the steps at the measured rate, so the number on screen
    and the number of steps that run are the same decision read two ways -
    rather than a minute budget that silently became 3,600 steps on a project
    that had grown.
    """
    device = "cuda" if has_gpu else "cpu"
    n_val = max(0, int(holdout))
    n_train = max(1, counts.frames - n_val)

    batch = BASE_BATCH[device]
    batch = min(batch, max(1, n_train // 4))
    batch = max(1, min(batch, n_train))
    steps_per_epoch = max(1, n_train // batch)

    base = WARM_STEPS if warm else COLD_STEPS
    steps = max(1, int(base * max(0.1, float(effort))))
    epochs = max(1, min(MAX_EPOCHS, math.ceil(steps / steps_per_epoch)))

    # Validate about four times a round, and never leave the last epoch
    # unvalidated: best.pth is only written on a validating epoch that
    # improved, so an unvalidated final epoch can end a round with no
    # checkpoint at all. The trainer validates when (epoch + 1) % interval
    # == 0, so the epoch count has to be a multiple of the interval.
    interval = max(1, epochs // 4)
    epochs = int(math.ceil(epochs / interval) * interval)

    return RoundPlan(
        number=max(1, int(number)),
        counts=counts,
        device=device,
        batch_size=batch,
        epochs=epochs,
        # In validating epochs, not raw ones: the trainer only counts a
        # no-improvement against patience on an epoch that validated.
        patience=max(3, (epochs // interval) // 2),
        steps_per_epoch=steps_per_epoch,
        n_train=n_train,
        n_val=n_val,
        val_kind=val_kind if n_val else "none",
        warmup_bound=(not warm) and epochs * steps_per_epoch < WARMUP_STEPS,
        warm=bool(warm),
        val_interval=interval,
    )


def _duration(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 1.5:
        return "about a minute"
    if minutes < 60:
        return f"about {round(minutes)} minutes"
    return f"about {minutes / 60.0:.1f} hours"


def plan_sentence(plan: RoundPlan, rows: Sequence[Dict[str, Any]] = ()) -> str:
    """What the user is agreeing to, in their units, before they press Start.

    Quoting the cost before the commitment is the rule this exists for; the
    part that is easy to get wrong is quoting it *honestly* when the machine
    overrode the request, which is what ``warmup_bound`` and ``val_kind`` are
    for.

    ``rows`` is the round log, and it buys the two facts that turn a button
    into a series: **which round this is**, and whether the rounds before it
    have stopped teaching (:func:`series_note`). A loop meant to be taken ten
    times has to say where in the ten you are, or the tenth is
    indistinguishable from the second and nobody knows when to stop.
    """
    counts = plan.counts
    plural = "s" if counts.videos != 1 else ""
    lines = [
        f"Round {plan.number}. "
        f"{counts.frames:,} labelled frame{'s' if counts.frames != 1 else ''} "
        f"in {counts.videos} video{plural}, {counts.classes} "
        f"class{'es' if counts.classes != 1 else ''}, {counts.boxes:,} boxes.",
    ]
    where = "on your GPU" if plan.device == "cuda" else "on the processor, which has no GPU to use"
    passes = f"{plan.total_steps:,} passes over batches of {plan.batch_size}"
    lines.append(f"{_duration(plan.est_seconds).capitalize()} {where} - {passes}.")
    if plan.warm:
        lines.append(
            "Carrying on from the last round's model, so it needs a third of "
            "the work: it only has to take in what you have labelled since."
        )
    if plan.warmup_bound:
        lines.append(
            "That is longer than you asked for: this many frames make for a "
            "short pass, and a shorter run would spend all of it warming up."
        )
    if plan.val_kind == "video":
        lines.append(
            f"{plan.n_val} frames from one video are held back, so the model is "
            "checked against a recording it never saw."
        )
    elif plan.val_kind == "frames":
        lines.append(
            f"{plan.n_val} frames are held back, spread through the videos, so "
            "the model is checked against frames it never saw."
        )
    else:
        lines.append(
            "Too few labels to hold any back, so this round is checked against "
            "the frames it trained on. The score will look better than it is."
        )
    lines.append(
        "Nothing you labelled changes, and nothing the model finds is written "
        "to annotations.json until you accept it."
    )
    note = series_note(rows, plan.number)
    if note:
        lines.append(note)
    return "\n".join(lines)


#: Decoder layers a round trains, against the four a production run uses.
#:
#: The decoder is where a round's time goes. Measured (RTX 5070 Ti, fp16,
#: channels_last, ResNet-18, 640x640, detector only) the four-layer default
#: spends 5.93 ms of its 12.95 ms there - 46%, against 2.98 ms in the backbone -
#: and halving it gives:
#:
#:     dec=4  18.81M   73.7 fps at batch 1    350.6 fps at batch 16
#:     dec=2  16.43M   99.9 fps at batch 1    473.0 fps at batch 16
#:
#: The backbone is **not** the lever people reach for it to be: ResNet-50 at
#: batch 1 measures 15.8 ms against ResNet-18's 15.0, and 384x384 is no faster
#: than 640x640, because at batch 1 the detector is bound by kernel launches
#: rather than by arithmetic. Two is the number, not one: a single-layer DETR
#: decoder has no iterative refinement left and loses more than the round can
#: afford to give.
#:
#: A round is not a model. It exists to decide what to label next, and it is
#: thrown away when the next round replaces it; the final model is made in the
#: workflow window's Train page, at the full four layers, from every label.
#: This is the one place where "fast enough to keep the loop turning" wins over
#: "as good as this data allows".
ROUND_DECODER_LAYERS = 2


def train_overrides(plan: RoundPlan, *, init_weights: str = "") -> Dict[str, Any]:
    """The config keys a round sets, and the reason each one is not a default.

    Passed through ``commands.train_command``'s ``extra``, which maps onto real
    config fields. Unknown keys are dropped in silence by the config merge, so
    every name here is one that was checked against the dataclass rather than
    guessed.
    """
    out: Dict[str, Any] = {
        # The single most important one: without it a multi-class project
        # trains every non-zero class as background.
        "model.detector.head.num_classes": plan.counts.classes,
        # Loading inline beats spawning workers for a few hundred images, and
        # a cancelled round leaves nothing behind to orphan. Measured: on this
        # machine 2 and 4 workers were no faster than 0 at this dataset size,
        # and on Windows terminate() does not reap them.
        "data.num_workers": 0,
        # Patience counts validating epochs, which is what the trainer counts.
        "train.early_stopping_patience": plan.patience,
        # Measured: last.pth is 348 MB and best.pth 249 MB, written every
        # epoch. A 100-epoch round therefore spent minutes writing ~35 GB of
        # checkpoints nobody reads - the round only ever adopts best.pth, and
        # falls back to last.pth only when no epoch improved.
        "train.val_interval": plan.val_interval,
        # And two thirds of each of those files is Adam's two moments per
        # parameter, for a run that is never resumed: the next round carries
        # the weights over with ``init_weights`` and builds a fresh optimizer
        # on purpose. 249 MB becomes about 94 MB.
        "train.save_optimizer_state": False,
        # A lighter decoder. See ROUND_DECODER_LAYERS - this is the round's
        # one deliberate departure from the production architecture, and
        # ``predict`` follows it without being told because the checkpoint
        # carries its own config (cli.load_runtime_config_from_checkpoint).
        "model.detector.head.num_decoder_layers": ROUND_DECODER_LAYERS,
    }
    if init_weights:
        # Weights only: a fresh optimizer, a fresh epoch counter and a fresh
        # schedule. NOT ``train.resume``, which also restores the scheduler
        # (so the learning rate resumes at its decayed floor) and the previous
        # run's best_metric (so the new round may never write best.pth).
        out["init_weights"] = init_weights
    return out


def adoption(
    *,
    checkpoint: bool,
    score: float | None,
    digest: str,
    previous_score: float | None,
    previous_digest: str,
    higher_is_better: bool = False,
) -> Tuple[bool, str]:
    """``(adopt, sentence)`` - should this round's model become the project's?

    Three cases, and the middle one is the reason this is a function rather
    than an ``if``:

    * No checkpoint on disk: never adopt. A run can exit 0 and write nothing -
      every checkpoint is written from inside the epoch loop, so a run that
      trained zero epochs reports success and leaves an empty ``weights/``.
      Adopting it would point the project at a file that does not exist.
    * Not comparable: adopt. Either there is nothing to compare against, or the
      two rounds were scored on different frames. More labels with the same
      recipe is the way to bet, and refusing to adopt would strand the loop on
      round one's model forever.
    * Comparable and worse: keep what the project has, and say so. Adopting a
      worse detector makes the *next* pass through the loop worse than having
      no detector at all, because the frames it picks are chosen by what the
      detector was unsure about.
    """
    if not checkpoint:
        return False, (
            "Training finished without writing a checkpoint, so the project's "
            "model is unchanged. The log in the round folder says why."
        )
    if score is None or not digest or digest != previous_digest or previous_score is None:
        return True, "This round's model is now the project's model."
    better = score > previous_score if higher_is_better else score < previous_score
    if better:
        return True, (
            f"Better than the last round ({score:.4g} against {previous_score:.4g}), "
            "so this round's model is now the project's model."
        )
    return False, (
        f"Not better than the last round ({score:.4g} against {previous_score:.4g}), "
        "so the project keeps the model it had. This round's checkpoint is kept "
        "in its own folder if you want it."
    )


# ══════════════════════════════════════════════════════════════════════════
# The series: when has the loop stopped teaching?
# ══════════════════════════════════════════════════════════════════════════
#
# A round is cheap and the loop is meant to be taken ten times, which raises
# the question nothing in this module could answer before: *when do I stop?*
#
# **Not on the holdout score.** It is the obvious candidate and it is the wrong
# one, for a reason written into this file already: the holdout is 20% of the
# labelled frames and it GROWS as the label count grows (:func:`grow_holdout`),
# so every round of this shape scores against a different set of frames. That
# is exactly the case :func:`adoption` refuses to compare, and ``rounds.json``
# records ``holdout_digest`` precisely so the comparison cannot be made by
# accident. A "plateau" read off those numbers would be reading noise.
#
# **On agreement instead** - the fraction of the model's proposals a human
# accepted without changing them. It is measured on new frames the model has
# never seen, one round's worth at a time, and it answers the question the user
# is actually asking, which is not "is the loss going down" but "is it still
# worth my afternoon to correct this thing". It needs no held-out set and no
# comparable ruler: the denominator is whatever this round proposed.

#: Rounds of agreement needed before anything is said about the series. Two
#: points make a line through anything; three is the fewest that can show a
#: trend flattening rather than a good round followed by a bad one.
PLATEAU_WINDOW = 3

#: How little the agreement has to move across :data:`PLATEAU_WINDOW` rounds
#: for the loop to have stopped teaching. Three percentage points is inside
#: the run-to-run variance of a round on this project's hardware, so a spread
#: below it is not a trend at all.
PLATEAU_BAND = 0.03

#: Agreement above this is a model whose mistakes are rare enough that finding
#: them by sampling has become inefficient - the sampler is spending 200 frames
#: to surface a handful of corrections.
PLATEAU_AGREEMENT = 0.9

#: Before this many rounds, say nothing. Early agreement swings wildly on a few
#: dozen frames, and a "you are done" after round 2 is advice that costs the
#: user their dataset.
SERIES_HINT_FROM = 5


def agreement(accepted: int, rejected: int, edited: int, drawn: int) -> float | None:
    """How much of what the model proposed survived the human, ``0..1``.

    ``accepted`` counts boxes taken as they were. ``rejected`` and ``edited``
    are the model's proposals the human threw away or moved, and ``drawn`` is
    the boxes the human had to add because the model missed them - a miss is a
    disagreement even though there was no proposal to disagree with, and
    leaving it out of the denominator would score a model that finds one fly
    out of two and finds it perfectly at 100%.

    ``None`` when nothing was proposed and nothing was drawn, which is not a
    score of zero: it is a round nobody has reviewed yet.
    """
    total = accepted + rejected + edited + drawn
    if total <= 0:
        return None
    return accepted / total


def _recent_agreements(rows: Iterable[Dict[str, Any]], window: int) -> List[float]:
    """The last ``window`` agreement figures, oldest first, skipping the gaps."""
    values = [
        float(row["agreement"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("agreement"), (int, float))
    ]
    return values[-window:]


def series_note(rows: Sequence[Dict[str, Any]], number: int) -> str:
    """One sentence about where this round sits in the series, or ``""``.

    Silent by default, and deliberately so: this is advice about when to stop,
    and advice that appears every round is noise by round three. It speaks only
    once there are enough rounds behind it to mean anything, and then only to
    say one of two things - carry on, or the loop has stopped teaching and the
    dataset is what is left to do.
    """
    if number <= SERIES_HINT_FROM:
        return ""
    recent = _recent_agreements(rows, PLATEAU_WINDOW)
    if len(recent) < PLATEAU_WINDOW:
        return ""
    spread = max(recent) - min(recent)
    latest = recent[-1]
    if spread <= PLATEAU_BAND and latest >= PLATEAU_AGREEMENT:
        return (
            f"You have kept {latest:.0%} of what it proposed for "
            f"{PLATEAU_WINDOW} rounds running. The loop has stopped teaching "
            "it much; what is left is to run it over every frame and check "
            "that."
        )
    if spread <= PLATEAU_BAND:
        return (
            f"Agreement has sat near {latest:.0%} for {PLATEAU_WINDOW} rounds. "
            "More rounds of the same size are unlikely to move it - try a "
            "larger budget, or look at what it keeps getting wrong."
        )
    return f"You kept {latest:.0%} of what it proposed last round."
