"""
sampling.py — deciding *what a human should annotate*, which is the whole point.

A 30-minute video at 30 fps is 54,000 frames. The question "which ones should I
label?" has no good manual answer, and the usual answers are all bad: the first
200 frames (the animals have not started doing anything), every 100th frame
(most of them look identical), or the ones that happen to be on screen when the
scrolling stops.

This module answers it. Given frame features and a budget, it returns the frames
and clips worth a human's time, plus **why each was chosen**, because a selection
a user cannot interrogate is one they will not trust.

Two regimes, one interface
--------------------------

**Cold start** — no model exists yet, so there is nothing to be uncertain about.
The only signal is appearance, and the right strategy is *diversity*: cover the
space of what the video looks like, so the first training set is not 200 frames
of the same pose (§7.1).

**Active learning** — a model exists and has run. Now uncertainty, temporal
inconsistency, interaction and rarity are all available, and the right strategy
combines them with diversity so the budget is not spent on 200 near-identical
hard frames (§7.2).

Both go through :func:`plan`. The caller does not branch; it passes whatever
signals it has, and the strategies use what is there.

Detection samples frames, behavior samples clips
------------------------------------------------
A box is a property of a frame and can be annotated from that frame alone. A
behavior is not — "is this courtship?" is unanswerable without the seconds
either side. So :func:`plan` returns frames for detection and :class:`Clip` for
behavior, and never asks a human to label a behavior from a still (§8).

Reproducibility
---------------
Every strategy is a deterministic function of ``(features, k, seed)``. The seed,
the strategy and every weight are recorded on the :class:`Selection`, so a
project can store them and get the same frames back (§41). No strategy calls the
global ``random`` module.

Purity: no dearpygui, no torch, no cv2, no numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .features import FrameFeature, distance, farthest_point_order

__all__ = [
    "TARGET_DETECTION", "TARGET_BEHAVIOR", "TARGET_BOTH", "TARGETS",
    "RANDOM", "UNIFORM", "DIVERSITY", "UNCERTAINTY", "DIVERSITY_UNCERTAINTY",
    "INTERACTION", "STRATEGIES", "STRATEGY_LABELS", "STRATEGY_HELP",
    "ScoreWeights", "WorkloadModel", "Budget", "SamplingRequest",
    "Clip", "Selection", "SamplingContext",
    "plan", "select_frames", "select_clips",
    "rank_normalize", "rarity_scores", "component_scores",
]

# ── what is being annotated ───────────────────────────────────────────────
TARGET_DETECTION = "detection"
TARGET_BEHAVIOR = "behavior"
TARGET_BOTH = "both"
TARGETS: Tuple[str, ...] = (TARGET_DETECTION, TARGET_BEHAVIOR, TARGET_BOTH)

# ── strategies ────────────────────────────────────────────────────────────
RANDOM = "random"
UNIFORM = "uniform"
DIVERSITY = "diversity"
UNCERTAINTY = "uncertainty"
DIVERSITY_UNCERTAINTY = "diversity+uncertainty"
INTERACTION = "interaction"

STRATEGY_LABELS: Dict[str, str] = {
    RANDOM: "Random",
    UNIFORM: "Uniform",
    DIVERSITY: "Diversity",
    UNCERTAINTY: "Uncertainty",
    DIVERSITY_UNCERTAINTY: "Diversity + uncertainty",
    INTERACTION: "Interaction-focused",
}

STRATEGY_HELP: Dict[str, str] = {
    RANDOM: "Picks at random. The honest baseline - everything else should beat it.",
    UNIFORM: "Picks at even intervals. Predictable, and blind to what is in the frames.",
    DIVERSITY: "Covers the range of what the video looks like. Use this before a model exists.",
    UNCERTAINTY: "Picks where the model was least sure. Needs a model that has run.",
    DIVERSITY_UNCERTAINTY: "Spreads the picks across the frames the model found hardest. "
                           "The default once a model exists.",
    INTERACTION: "Picks where animals are close or touching. Use this for social behavior.",
}


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoreWeights:
    """How much each signal counts toward "this frame is worth annotating".

    A weighted sum over rank-normalised components, not a product. A product
    (the shape §7.2 sketches) has a property that reads well and behaves badly:
    one zero component zeroes the whole score, so a frame the interaction
    detector scores 0 is unpickable no matter how uncertain the model was about
    it. The sum degrades instead of vetoing.

    Set a weight to ``0.0`` to switch a signal off. The numbers are a starting
    point for *Drosophila* courtship assays; they are not tuned, and no result
    in this module depends on their exact values.
    """

    uncertainty: float = 1.0
    novelty: float = 0.6
    temporal_inconsistency: float = 0.8
    interaction: float = 0.5
    rarity: float = 0.4

    def as_dict(self) -> Dict[str, float]:
        return {"uncertainty": self.uncertainty, "novelty": self.novelty,
                "temporal_inconsistency": self.temporal_inconsistency,
                "interaction": self.interaction, "rarity": self.rarity}


@dataclass(frozen=True)
class WorkloadModel:
    """Minutes of human time, converted to a number of samples.

    **These numbers are guesses, and the UI must say so.** They were not
    measured — there is no instrumented annotation session to measure — so the
    estimate is labelled "estimated" everywhere it appears and is never
    presented to two significant figures (``docs/GUI_DESIGN.md`` P12). Their job
    is to turn "15 minutes" into a plausible sample count, not to predict the
    clock.

    Once the annotator logs real per-frame times, replace these from that log
    and the whole feature gets better with no code change.
    """

    #: Drawing every box on a frame from nothing.
    seconds_per_blank_frame: float = 25.0
    #: Checking, and occasionally fixing, boxes a model already drew. The ratio
    #: to the line above is the entire argument for auto pre-labelling.
    seconds_per_prelabelled_frame: float = 6.0
    #: Watching a 2-5 s clip and labelling the behavior in it from nothing.
    seconds_per_blank_clip: float = 40.0
    #: Confirming or nudging a behavior interval a model proposed.
    seconds_per_prelabelled_clip: float = 15.0

    def frame_cost(self, prelabelled: bool) -> float:
        return (self.seconds_per_prelabelled_frame if prelabelled
                else self.seconds_per_blank_frame)

    def clip_cost(self, prelabelled: bool) -> float:
        return (self.seconds_per_prelabelled_clip if prelabelled
                else self.seconds_per_blank_clip)

    def estimate_seconds(self, n_frames: int, n_clips: int,
                         prelabelled: bool = True) -> float:
        return (n_frames * self.frame_cost(prelabelled)
                + n_clips * self.clip_cost(prelabelled))

    def describe(self, n_frames: int, n_clips: int,
                 prelabelled: bool = True) -> str:
        """``~14 minutes`` — deliberately one significant figure under an hour."""
        seconds = self.estimate_seconds(n_frames, n_clips, prelabelled)
        if seconds < 90:
            return "under 2 minutes"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"~{int(round(minutes))} minutes"
        return f"~{minutes / 60.0:.1f} hours"


@dataclass(frozen=True)
class Budget:
    """How much work the user is willing to do (§28).

    Exactly one of :attr:`samples` and :attr:`minutes` is used; ``samples`` wins
    when both are set, because a count is exact and a duration is an estimate,
    and silently preferring the estimate would be the wrong way round.
    """

    samples: Optional[int] = None
    minutes: Optional[float] = None
    #: Share of the budget spent on behavior clips when the target is *both*.
    #: 0.35 because a clip costs more than a frame but teaches a whole
    #: behavior — the split is by *time*, not by count.
    behavior_share: float = 0.35

    def counts(
        self,
        *,
        target: str,
        model: WorkloadModel,
        prelabelled: bool = True,
        default_samples: int = 100,
    ) -> Tuple[int, int]:
        """``(n_frames, n_clips)`` for this budget and target.

        A budget that resolves to zero of something the target asked for is
        raised to one: a user who selected "Individual + Behavior" and got no
        clips at all would reasonably read that as a bug.
        """
        if self.samples is not None and self.samples > 0:
            total = int(self.samples)
            if target == TARGET_DETECTION:
                return total, 0
            if target == TARGET_BEHAVIOR:
                return 0, total
            clips = max(1, int(round(total * self.behavior_share)))
            return max(1, total - clips), clips

        if self.minutes is not None and self.minutes > 0:
            seconds = float(self.minutes) * 60.0
            if target == TARGET_DETECTION:
                return max(1, int(seconds / model.frame_cost(prelabelled))), 0
            if target == TARGET_BEHAVIOR:
                return 0, max(1, int(seconds / model.clip_cost(prelabelled)))
            clip_seconds = seconds * self.behavior_share
            frame_seconds = seconds - clip_seconds
            return (max(1, int(frame_seconds / model.frame_cost(prelabelled))),
                    max(1, int(clip_seconds / model.clip_cost(prelabelled))))

        return Budget(samples=default_samples,
                      behavior_share=self.behavior_share).counts(
            target=target, model=model, prelabelled=prelabelled)

    def describe(self) -> str:
        if self.samples:
            return f"{self.samples} samples"
        if self.minutes:
            minutes = float(self.minutes)
            return f"{minutes:g} minutes"
        return "default budget"


@dataclass(frozen=True)
class SamplingRequest:
    """Everything the Smart Annotate dialog collects, as one value.

    Stored verbatim in project metadata so a selection is reproducible (§41).
    """

    target: str = TARGET_BOTH
    budget: Budget = field(default_factory=Budget)
    strategy: str = DIVERSITY
    seed: int = 0
    fps: float = 25.0
    #: Behavior clip length. 3 s at 25 fps is 75 frames — long enough to see a
    #: courtship bout begin and end, short enough to judge in one viewing.
    clip_seconds: float = 3.0
    #: Are candidates being generated for these samples? Changes the workload
    #: estimate by about 4x, which is the difference between "15 minutes buys
    #: 36 frames" and "15 minutes buys 150".
    prelabelled: bool = True
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    workload: WorkloadModel = field(default_factory=WorkloadModel)

    @property
    def clip_frames(self) -> int:
        return max(2, int(round(self.clip_seconds * max(self.fps, 1.0))))

    def to_metadata(self) -> Dict[str, Any]:
        """A JSON-safe record for ``project.json``."""
        return {
            "target": self.target,
            "strategy": self.strategy,
            "seed": self.seed,
            "fps": self.fps,
            "clip_seconds": self.clip_seconds,
            "prelabelled": self.prelabelled,
            "budget": {"samples": self.budget.samples,
                       "minutes": self.budget.minutes,
                       "behavior_share": self.budget.behavior_share},
            "weights": self.weights.as_dict(),
        }


# ══════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Clip:
    """A span of frames to annotate a behavior in.

    ``end`` is inclusive, matching :class:`~htrtdetr.annotation.intervals.BehaviorInterval`
    and :class:`~htrtdetr.annotation.quality.ReviewItem`. One convention for a
    range across the whole package; the off-by-one that the mixed convention
    produces is invisible until it drops the last frame of every clip.
    """

    start: int
    end: int
    reason: str = "representative"
    score: float = 0.0

    @property
    def span(self) -> int:
        return self.end - self.start + 1

    @property
    def middle(self) -> int:
        return (self.start + self.end) // 2

    def seconds(self, fps: float) -> float:
        return self.span / fps if fps > 0 else 0.0

    def overlaps(self, other: "Clip") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class Selection:
    """What to annotate, why, and what it should cost."""

    frames: Tuple[int, ...] = ()
    clips: Tuple[Clip, ...] = ()
    #: ``{frame: reason}`` for the frames. The timeline shows it on hover and
    #: the Smart Annotate summary groups by it.
    reasons: Dict[int, str] = field(default_factory=dict)
    strategy: str = DIVERSITY
    seed: int = 0
    estimated_seconds: float = 0.0
    #: Frames that were available to choose from. Shown as the denominator, so
    #: "72 of 1,800 scanned" reads as a decision rather than as a total.
    pool_size: int = 0

    @property
    def total(self) -> int:
        return len(self.frames) + len(self.clips)

    def by_reason(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for reason in self.reasons.values():
            out[reason] = out.get(reason, 0) + 1
        for clip in self.clips:
            key = f"clip:{clip.reason}"
            out[key] = out.get(key, 0) + 1
        return out

    def describe(self, workload: Optional[WorkloadModel] = None) -> str:
        """The Smart Annotate summary block (§28)::

            64 representative frames
            8 interaction clips
            Estimated review workload: ~14 minutes
        """
        lines: List[str] = []
        counts = self.by_reason()
        for reason in sorted(counts):
            n = counts[reason]
            if reason.startswith("clip:"):
                lines.append(f"{n} {reason[5:]} clip{'s' if n != 1 else ''}")
            else:
                lines.append(f"{n} {reason} frame{'s' if n != 1 else ''}")
        if not lines:
            return "Nothing selected"
        if self.estimated_seconds > 0:
            model = workload or WorkloadModel()
            minutes = self.estimated_seconds / 60.0
            estimate = ("under 2 minutes" if self.estimated_seconds < 90
                        else f"~{int(round(minutes))} minutes" if minutes < 60
                        else f"~{minutes / 60.0:.1f} hours")
            lines.append(f"Estimated review workload: {estimate}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

def rank_normalize(values: Sequence[float]) -> List[float]:
    """Map values onto ``0..1`` by rank, ties sharing the mean rank.

    Rank rather than min-max because the components have wildly different
    distributions — uncertainty clusters near 0 with a long tail, interaction is
    bimodal — and min-max lets whichever one happens to have an outlier dominate
    the weighted sum. After ranking, a weight means what it says.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 / (n - 1)
        for pos in range(i, j + 1):
            out[order[pos]] = shared
        i = j + 1
    return out


def rarity_scores(
    vectors: Sequence[Sequence[float]],
    *,
    sample: int = 192,
    seed: int = 0,
) -> List[float]:
    """How unlike the rest of the video each frame looks, in ``0..1``.

    Mean cosine distance to a fixed random subsample rather than to everything:
    ``O(n * sample)`` instead of ``O(n^2)``, which is the difference between
    0.4 s and 40 s at n = 4,000, and the ranking is stable well before the
    subsample is large — the quantity being estimated is a mean.

    The subsample is drawn from a seeded RNG, so the scores are reproducible.
    """
    n = len(vectors)
    if n == 0:
        return []
    rng = random.Random(seed)
    idx = list(range(n))
    if n > sample:
        idx = rng.sample(idx, sample)
    reference = [vectors[i] for i in idx if vectors[i]]
    if not reference:
        return [0.0] * n
    out: List[float] = []
    for vector in vectors:
        if not vector:
            out.append(0.0)
            continue
        total = sum(distance(vector, other) for other in reference)
        out.append(total / len(reference))
    return out


@dataclass
class SamplingContext:
    """The signals a strategy may use, precomputed once.

    Built by :func:`plan`. A strategy receives this and a ``k``; it never reads
    a video, a model or a config file, which is why every strategy is a pure
    function that a test can call with five fake frames.
    """

    features: Sequence[FrameFeature]
    seed: int = 0
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    #: Per-frame temporal inconsistency, usually aggregated from
    #: :func:`htrtdetr.annotation.quality.analyze` severities. Frames absent
    #: from the mapping score 0.
    inconsistency: Mapping[int, float] = field(default_factory=dict)
    #: Appearance vectors of frames a human has already annotated. Novelty is
    #: distance to the nearest of these, so the selection avoids re-proposing
    #: what is already in the training set.
    known: Sequence[Sequence[float]] = ()

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.vectors: List[Tuple[float, ...]] = [
            tuple(f.embedding) for f in self.features]
        self._components: Optional[Dict[str, List[float]]] = None

    @property
    def size(self) -> int:
        return len(self.features)

    def components(self) -> Dict[str, List[float]]:
        """Every signal, rank-normalised to ``0..1``, computed once."""
        if self._components is None:
            self._components = component_scores(self)
        return self._components

    def combined(self) -> List[float]:
        """The weighted sum. Zero-weight components cost nothing to include."""
        comps = self.components()
        weights = self.weights.as_dict()
        total_weight = sum(w for w in weights.values() if w > 0) or 1.0
        out = [0.0] * self.size
        for name, weight in weights.items():
            if weight <= 0:
                continue
            column = comps.get(name)
            if not column:
                continue
            for i, value in enumerate(column):
                out[i] += weight * value
        return [v / total_weight for v in out]

    def dominant_reason(self, index: int) -> str:
        """Which signal chose this frame — the word shown next to it.

        The largest weighted component. With no signal above zero it is
        ``representative``, which is the honest word for a diversity pick.
        """
        comps = self.components()
        weights = self.weights.as_dict()
        best, best_value = "representative", 0.0
        for name, weight in weights.items():
            if weight <= 0:
                continue
            column = comps.get(name)
            if not column:
                continue
            value = weight * column[index]
            if value > best_value:
                best, best_value = name, value
        return {"uncertainty": "uncertain", "novelty": "novel",
                "temporal_inconsistency": "inconsistent",
                "interaction": "interaction", "rarity": "rare"}.get(best, best)


def component_scores(ctx: SamplingContext) -> Dict[str, List[float]]:
    """All five signals for a context, each rank-normalised to ``0..1``."""
    features = ctx.features
    n = len(features)
    if n == 0:
        return {}

    uncertainty = [f.uncertainty for f in features]
    interaction = [f.interaction for f in features]
    inconsistency = [float(ctx.inconsistency.get(f.frame, 0.0)) for f in features]

    if ctx.known:
        novelty = [min((distance(v, k) for k in ctx.known), default=1.0)
                   if v else 0.0 for v in ctx.vectors]
    else:
        # Nothing annotated yet, so every frame is equally novel. A constant
        # column rank-normalises to a constant and contributes nothing, which
        # is the correct behaviour — not an excuse to invent a signal.
        novelty = [0.0] * n

    rarity = rarity_scores(ctx.vectors, seed=ctx.seed)

    return {
        "uncertainty": rank_normalize(uncertainty),
        "novelty": rank_normalize(novelty),
        "temporal_inconsistency": rank_normalize(inconsistency),
        "interaction": rank_normalize(interaction),
        "rarity": rank_normalize(rarity),
    }


# ══════════════════════════════════════════════════════════════════════════
# Strategies — each returns indices into ctx.features
# ══════════════════════════════════════════════════════════════════════════

def _strategy_random(ctx: SamplingContext, k: int) -> List[int]:
    n = ctx.size
    if n == 0 or k <= 0:
        return []
    return sorted(random.Random(ctx.seed).sample(range(n), min(k, n)))


def _strategy_uniform(ctx: SamplingContext, k: int) -> List[int]:
    n = ctx.size
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [n // 2]
    step = (n - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def _strategy_diversity(ctx: SamplingContext, k: int) -> List[int]:
    if ctx.size == 0 or k <= 0:
        return []
    # The seed picks the traversal's starting frame, which is the only free
    # choice in an otherwise deterministic algorithm - and only while nothing
    # has been annotated yet. Once ``known`` is populated the traversal starts
    # from what is already labelled, so a second pass continues the first
    # instead of re-rolling it. This is the cold-start path, where no model
    # exists and appearance is the only signal there is, so it is the one that
    # has to iterate on its own.
    first = random.Random(ctx.seed).randrange(ctx.size)
    return sorted(farthest_point_order(ctx.vectors, k, first=first, seeds=ctx.known))


def _strategy_uncertainty(ctx: SamplingContext, k: int) -> List[int]:
    if ctx.size == 0 or k <= 0:
        return []
    scores = ctx.components().get("uncertainty") or [0.0] * ctx.size
    order = sorted(range(ctx.size), key=lambda i: (-scores[i], i))
    return sorted(order[:min(k, ctx.size)])


def _spread_over_top(ctx: SamplingContext, k: int, scores: Sequence[float],
                     pool_factor: int = 4) -> List[int]:
    """Take the best ``k * pool_factor`` by score, then spread ``k`` across them.

    This two-stage shape, not a single weighted ranking, because a pure ranking
    of hard frames returns ``k`` frames from the same two seconds of video —
    difficulty is temporally autocorrelated, and 200 consecutive frames of one
    hard moment teach a model almost nothing more than one of them does. The
    farthest-point pass over the hard pool is what buys back the coverage.
    """
    n = ctx.size
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    pool_size = min(n, max(k, k * max(1, pool_factor)))
    pool = sorted(range(n), key=lambda i: (-scores[i], i))[:pool_size]
    if len(pool) <= k:
        return sorted(pool)
    sub = [ctx.vectors[i] for i in pool]
    if not any(sub):
        return sorted(pool[:k])   # no appearance vectors: ranking is all there is
    first = random.Random(ctx.seed).randrange(len(pool))
    picked = farthest_point_order(sub, k, first=first, seeds=ctx.known)
    return sorted(pool[p] for p in picked)


def _strategy_diversity_uncertainty(ctx: SamplingContext, k: int) -> List[int]:
    return _spread_over_top(ctx, k, ctx.combined())


def _strategy_interaction(ctx: SamplingContext, k: int) -> List[int]:
    scores = ctx.components().get("interaction") or [0.0] * ctx.size
    return _spread_over_top(ctx, k, scores)


#: Strategy registry. A new strategy is added here and becomes available in the
#: Smart Annotate dialog with no further wiring (§7.2: switchable, not
#: hard-coded).
STRATEGIES: Dict[str, Callable[[SamplingContext, int], List[int]]] = {
    RANDOM: _strategy_random,
    UNIFORM: _strategy_uniform,
    DIVERSITY: _strategy_diversity,
    UNCERTAINTY: _strategy_uncertainty,
    DIVERSITY_UNCERTAINTY: _strategy_diversity_uncertainty,
    INTERACTION: _strategy_interaction,
}


# ══════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════

def select_frames(ctx: SamplingContext, k: int, strategy: str = DIVERSITY) -> List[int]:
    """Frame **indices into ``ctx.features``**, chosen by one strategy.

    Falls back to :data:`DIVERSITY` on an unknown name rather than raising: a
    stale strategy string in a saved project must not stop the video opening.
    """
    fn = STRATEGIES.get(strategy, _strategy_diversity)
    return fn(ctx, k)


def select_clips(
    ctx: SamplingContext,
    k: int,
    clip_frames: int,
    *,
    strategy: str = INTERACTION,
    min_separation: int = 0,
) -> List[Clip]:
    """Non-overlapping clips, centred on the highest-scoring moments (§8).

    Scored by the mean of the frames inside the window, then greedy non-maximum
    suppression so two clips never cover the same seconds. Overlapping clips
    would make a human watch the same moment twice and would report a budget
    that buys less coverage than it claims.
    """
    n = ctx.size
    if n == 0 or k <= 0 or clip_frames <= 0:
        return []

    if strategy == INTERACTION:
        scores = ctx.components().get("interaction") or [0.0] * n
    elif strategy == UNCERTAINTY:
        scores = ctx.components().get("uncertainty") or [0.0] * n
    elif strategy in (RANDOM, UNIFORM, DIVERSITY):
        # These have no per-frame preference, so the clip centres come from the
        # frame strategy itself and the score only breaks ties.
        chosen = set(select_frames(ctx, k, strategy))
        scores = [1.0 if i in chosen else 0.0 for i in range(n)]
    else:
        scores = ctx.combined()

    frames = [f.frame for f in ctx.features]
    half = clip_frames // 2

    # The scan is sparse: ``features`` may hold every 3rd frame, or every 15th.
    # A window of `half` *indices* is therefore not a window of `half` *frames*,
    # and averaging over the wrong one smears a 3-second clip's score across 15
    # seconds of video. Convert through the median step between scanned frames.
    step = _median_step(frames)
    index_half = max(1, int(round(half / step))) if step > 0 else half

    windows: List[Tuple[float, int]] = []
    for i in range(n):
        lo, hi = max(0, i - index_half), min(n, i + index_half + 1)
        window = scores[lo:hi]
        windows.append((sum(window) / len(window) if window else 0.0, i))
    windows.sort(key=lambda w: (-w[0], w[1]))

    gap = max(0, int(min_separation))
    out: List[Clip] = []
    for score, i in windows:
        if len(out) >= k:
            break
        centre = frames[i]
        clip = Clip(start=max(0, centre - half),
                    end=max(0, centre - half) + clip_frames - 1,
                    reason=_clip_reason(strategy, ctx, i),
                    score=score)
        if any(clip.start - gap <= c.end and c.start - gap <= clip.end for c in out):
            continue
        out.append(clip)

    out.sort(key=lambda c: c.start)
    return out


def _median_step(frames: Sequence[int]) -> float:
    """Median gap between consecutive scanned frames, or ``1.0`` below two.

    Median, not mean: a scan that skipped a run of undecodable frames has one
    huge gap, and the mean would report a stride the scan never used.
    """
    if len(frames) < 2:
        return 1.0
    steps = sorted(b - a for a, b in zip(frames, frames[1:]) if b > a)
    if not steps:
        return 1.0
    mid = len(steps) // 2
    return float(steps[mid] if len(steps) % 2 else (steps[mid - 1] + steps[mid]) / 2)


def _clip_reason(strategy: str, ctx: SamplingContext, index: int) -> str:
    if strategy == INTERACTION:
        return "interaction"
    if strategy == UNCERTAINTY:
        return "difficult"
    if strategy in (RANDOM, UNIFORM, DIVERSITY):
        return "representative"
    return ctx.dominant_reason(index)


def plan(
    request: SamplingRequest,
    features: Sequence[FrameFeature],
    *,
    inconsistency: Optional[Mapping[int, float]] = None,
    known: Sequence[Sequence[float]] = (),
) -> Selection:
    """The whole Smart Sampling decision: budget in, selection out.

    Args:
        request: What the user asked for in the Smart Annotate dialog.
        features: One per scanned frame, from
            :func:`htrtdetr.annotation.features.scan`.
        inconsistency: Per-frame temporal-inconsistency scores, typically
            aggregated from :func:`htrtdetr.annotation.quality.analyze`.
        known: Appearance vectors of already-annotated frames, so the selection
            does not re-propose them.

    Returns:
        A :class:`Selection` carrying the frames, the clips, the reason for
        each, the strategy and seed that produced them, and the estimated
        workload.

    Clips are chosen with the interaction strategy whenever the request is
    behavior-aware and the plain strategy has no opinion about behavior — a
    behavior clip picked for appearance diversity is a clip of an empty arena.
    """
    n_frames, n_clips = request.budget.counts(
        target=request.target,
        model=request.workload,
        prelabelled=request.prelabelled,
    )

    ctx = SamplingContext(
        features=features,
        seed=request.seed,
        weights=request.weights,
        inconsistency=dict(inconsistency or {}),
        known=list(known),
    )

    picked = select_frames(ctx, n_frames, request.strategy) if n_frames else []
    frames = tuple(features[i].frame for i in picked)
    reasons = {features[i].frame: _frame_reason(request.strategy, ctx, i)
               for i in picked}

    clips: Tuple[Clip, ...] = ()
    if n_clips:
        clip_strategy = request.strategy
        if clip_strategy in (DIVERSITY, DIVERSITY_UNCERTAINTY):
            clip_strategy = INTERACTION
        clips = tuple(select_clips(ctx, n_clips, request.clip_frames,
                                   strategy=clip_strategy))

    return Selection(
        frames=frames,
        clips=clips,
        reasons=reasons,
        strategy=request.strategy,
        seed=request.seed,
        estimated_seconds=request.workload.estimate_seconds(
            len(frames), len(clips), request.prelabelled),
        pool_size=len(features),
    )


def _frame_reason(strategy: str, ctx: SamplingContext, index: int) -> str:
    if strategy in (RANDOM, UNIFORM):
        return strategy
    if strategy == DIVERSITY:
        return "representative"
    if strategy == UNCERTAINTY:
        return "uncertain"
    if strategy == INTERACTION:
        return "interaction"
    return ctx.dominant_reason(index)
