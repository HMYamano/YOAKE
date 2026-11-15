"""
bouts.py — per-frame behavior probabilities, joined by tracking, into bouts.

The keyframe scheme (``docs/KEYFRAME_BEHAVIOR.md``) never labels an interval.
The model says, for every animal in every frame, how likely each behavior is;
the tracker says which detections are the same animal. This module is the step
that turns those two into what a behavioral study reports:

    predictions.json  (frame -> detections, track_id, action_probs)
        │  track_sequences   group by track, cut at long gaps
        │  viterbi           smooth along the track: switching costs, staying is cheap
        │  min bout          absorb runs shorter than a real behavior
        ▼
    bouts       track, behavior, start, end, duration, mean probability
    summary     per track: time budget, bout counts, mean duration, latency, transitions
    population  per time bin: how many animals are doing each behavior (identity-free)

Why Viterbi and not a majority filter: the argmax of a frame classifier
flickers exactly where the behavior is ambiguous, and a fixed window either
leaves the flicker or erases short real bouts. A two-parameter HMM (how sticky
behaviors are, what the model believes per frame) trades the two off with
probabilities instead of a window width, and uses the model's confidence: one
uncertain frame cannot break a confident bout, a run of confident frames can.

The population table needs no identity at all - it sums the probabilities of
everything detected in a time bin - so it is the number to trust when the
tracks have swapped. Per-animal statistics are only as good as identity
continuity; :func:`summarize` reports each track's coverage so that is visible.

numpy only, no torch: it runs wherever the predictions file can be read.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Bout",
    "TrackSequence",
    "load_predictions",
    "track_sequences",
    "split_on_gaps",
    "viterbi",
    "enforce_min_bout",
    "bouts_from_labels",
    "decode_tracks",
    "summarize",
    "population_timeline",
    "proximity",
    "write_outputs",
    "analyze_predictions",
    "bouts_as_dicts",
    "DEFAULT_STAY",
    "DEFAULT_MIN_BOUT_S",
    "DEFAULT_MAX_GAP",
]

#: Probability a behavior continues to the next frame. 0.98 at 30 fps is an
#: expected bout of ~1.7 s before a switch is even considered on prior grounds;
#: the per-frame evidence still decides where the switches actually are.
DEFAULT_STAY = 0.98
#: Shorter than this is a flicker, not a behavior.
DEFAULT_MIN_BOUT_S = 0.3
#: Frames an animal may go undetected inside one sequence. Longer gaps cut the
#: track into separate sequences, because nothing is known about what it did.
DEFAULT_MAX_GAP = 5

_EPS = 1e-6


@dataclass(frozen=True)
class Bout:
    """One behavior, one track, one run of frames (``end`` inclusive)."""

    track_id: int
    action: int
    start: int
    end: int
    mean_prob: float

    @property
    def frames(self) -> int:
        return self.end - self.start + 1

    def duration(self, fps: float) -> float:
        return self.frames / fps if fps and fps > 0 else float(self.frames)


@dataclass
class TrackSequence:
    """One track's frames, in order, with what the model said about each."""

    track_id: int
    frames: np.ndarray          # (n,) int
    probs: np.ndarray           # (n, A) float, rows sum to 1
    centers: np.ndarray         # (n, 2) float, pixels
    sizes: np.ndarray           # (n,) float, longer box side in pixels

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def take(self, sl: slice) -> "TrackSequence":
        return TrackSequence(self.track_id, self.frames[sl], self.probs[sl],
                             self.centers[sl], self.sizes[sl])


# ══════════════════════════════════════════════════════════════════════════
# Reading
# ══════════════════════════════════════════════════════════════════════════

def load_predictions(path: str) -> Tuple[Dict[int, Mapping[str, Any]], Dict[str, Any]]:
    """``(frames, meta)`` from a ``predictions.json``. Non-integer keys are metadata."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    frames: Dict[int, Mapping[str, Any]] = {}
    for key, value in raw.items():
        try:
            frames[int(key)] = value
        except (TypeError, ValueError):
            continue
    meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    return frames, dict(meta)


def _num_actions(frames: Mapping[int, Mapping[str, Any]]) -> int:
    width = 0
    for payload in frames.values():
        probs = payload.get("action_probs") or []
        if probs:
            width = max(width, len(probs[0]))
        ids = [a for a in (payload.get("action_ids") or []) if a is not None]
        if ids:
            width = max(width, max(int(a) for a in ids) + 1)
    return max(width, 1)


def track_sequences(
    frames: Mapping[int, Mapping[str, Any]],
    num_actions: Optional[int] = None,
) -> Dict[int, TrackSequence]:
    """Group detections by ``track_id``. Untracked detections (``-1``) are skipped.

    ``action_probs`` is used when present. An older file without it falls back
    to a one-hot of ``action_ids`` - decoding still works, it just cannot weigh
    a confident frame above an unsure one.
    """
    A = int(num_actions) if num_actions else _num_actions(frames)
    rows: Dict[int, List[Tuple[int, np.ndarray, Tuple[float, float], float]]] = {}
    for frame in sorted(frames):
        payload = frames[frame]
        boxes = payload.get("boxes") or []
        tids = payload.get("track_ids") or []
        aids = payload.get("action_ids") or []
        probs = payload.get("action_probs") or []
        for i, box in enumerate(boxes):
            tid = int(tids[i]) if i < len(tids) else -1
            if tid < 0:
                continue
            if i < len(probs) and probs[i] is not None and len(probs[i]) == A:
                p = np.asarray(probs[i], dtype=np.float64)
            else:
                p = np.full(A, 1.0 / A)
                a = int(aids[i]) if i < len(aids) else -1
                if 0 <= a < A:
                    p = np.zeros(A)
                    p[a] = 1.0
            total = float(p.sum())
            p = p / total if total > 0 else np.full(A, 1.0 / A)
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            rows.setdefault(tid, []).append(
                (int(frame), p, ((x1 + x2) / 2, (y1 + y2) / 2), max(x2 - x1, y2 - y1))
            )
    out: Dict[int, TrackSequence] = {}
    for tid, items in rows.items():
        out[tid] = TrackSequence(
            track_id=tid,
            frames=np.array([r[0] for r in items], dtype=np.int64),
            probs=np.stack([r[1] for r in items]),
            centers=np.array([r[2] for r in items], dtype=np.float64),
            sizes=np.array([r[3] for r in items], dtype=np.float64),
        )
    return out


def split_on_gaps(seq: TrackSequence, max_gap: int = DEFAULT_MAX_GAP) -> List[TrackSequence]:
    """Cut a track where it went undetected for more than ``max_gap`` frames."""
    if len(seq) == 0:
        return []
    steps = np.diff(seq.frames)
    cuts = np.nonzero(steps > int(max_gap) + 1)[0] + 1
    bounds = [0, *cuts.tolist(), len(seq)]
    return [seq.take(slice(a, b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# ══════════════════════════════════════════════════════════════════════════
# Decoding
# ══════════════════════════════════════════════════════════════════════════

def viterbi(
    probs: np.ndarray,
    stay: float = DEFAULT_STAY,
    frames: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Most likely behavior per frame under a sticky HMM. ``(n, A) -> (n,)``.

    Emission: the model's per-frame probability, divided by ``prior`` when one
    is given (turning a posterior into a scaled likelihood, so a frequent
    behavior is not favoured twice). Transition: stay with ``stay``, otherwise
    switch uniformly. Across a gap of ``g`` frames the chance of staying is
    ``stay ** g`` - the longer an animal was unseen, the less the past binds.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n, A = probs.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if A == 1:
        return np.zeros(n, dtype=np.int64)
    stay = float(min(max(stay, 1.0 / A), 1.0 - _EPS))
    emit = np.log(probs + _EPS)
    if prior is not None:
        emit = emit - np.log(np.asarray(prior, dtype=np.float64) + _EPS)
    gaps = np.ones(n, dtype=np.int64)
    if frames is not None and len(frames) == n:
        gaps[1:] = np.maximum(np.diff(np.asarray(frames, dtype=np.int64)), 1)

    score = emit[0].copy()
    back = np.zeros((n, A), dtype=np.int64)
    eye = np.eye(A, dtype=bool)
    for t in range(1, n):
        s = stay ** int(gaps[t])
        s = min(max(s, 1.0 / A), 1.0 - _EPS)
        trans = np.where(eye, math.log(s), math.log((1.0 - s) / (A - 1)))
        cand = score[:, None] + trans                  # (from, to)
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emit[t]
    labels = np.zeros(n, dtype=np.int64)
    labels[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def _runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """``[(start_idx, end_idx_inclusive, label)]`` over positions."""
    out: List[Tuple[int, int, int]] = []
    if len(labels) == 0:
        return out
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i - 1, int(labels[start])))
            start = i
    return out


def enforce_min_bout(
    labels: np.ndarray,
    probs: np.ndarray,
    frames: np.ndarray,
    min_frames: int,
) -> np.ndarray:
    """Absorb runs shorter than ``min_frames`` (in video frames) into a neighbour.

    The shortest run goes first, and joins whichever neighbouring behavior the
    model found more likely over the run's own frames. A sequence that is one
    short run altogether is left as it is - there is nothing to absorb it into.
    """
    labels = np.array(labels, dtype=np.int64, copy=True)
    if min_frames <= 1 or len(labels) == 0:
        return labels
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        lengths = [int(frames[e] - frames[s] + 1) for s, e, _ in runs]
        order = sorted(range(len(runs)), key=lambda k: lengths[k])
        k = order[0]
        if lengths[k] >= min_frames:
            return labels
        s, e, _ = runs[k]
        neighbours = []
        if k > 0:
            neighbours.append(runs[k - 1][2])
        if k + 1 < len(runs):
            neighbours.append(runs[k + 1][2])
        best = max(neighbours, key=lambda a: float(probs[s:e + 1, a].mean()))
        labels[s:e + 1] = best


def bouts_from_labels(
    track_id: int, frames: np.ndarray, labels: np.ndarray, probs: np.ndarray
) -> List[Bout]:
    """Run-length encode decoded labels into bouts, with the mean probability."""
    return [
        Bout(
            track_id=int(track_id),
            action=label,
            start=int(frames[s]),
            end=int(frames[e]),
            mean_prob=round(float(probs[s:e + 1, label].mean()), 4),
        )
        for s, e, label in _runs(labels)
    ]


def decode_tracks(
    sequences: Mapping[int, TrackSequence],
    *,
    fps: float,
    stay: float = DEFAULT_STAY,
    min_bout_s: float = DEFAULT_MIN_BOUT_S,
    max_gap: int = DEFAULT_MAX_GAP,
    use_prior: bool = False,
) -> Dict[int, List[Bout]]:
    """Decode every track into bouts. ``use_prior`` divides out the class frequency."""
    prior = None
    if use_prior and sequences:
        stacked = np.concatenate([s.probs for s in sequences.values() if len(s)], axis=0)
        prior = stacked.mean(axis=0)
    min_frames = max(1, int(round(float(min_bout_s) * float(fps)))) if fps and fps > 0 else 1
    out: Dict[int, List[Bout]] = {}
    for tid, seq in sorted(sequences.items()):
        bouts: List[Bout] = []
        for part in split_on_gaps(seq, max_gap):
            labels = viterbi(part.probs, stay=stay, frames=part.frames, prior=prior)
            labels = enforce_min_bout(labels, part.probs, part.frames, min_frames)
            bouts.extend(bouts_from_labels(tid, part.frames, labels, part.probs))
        out[tid] = bouts
    return out


# ══════════════════════════════════════════════════════════════════════════
# Summaries
# ══════════════════════════════════════════════════════════════════════════

def _name(action_names: Sequence[str], a: int) -> str:
    return str(action_names[a]) if 0 <= a < len(action_names) else str(a)


def summarize(
    bouts: Mapping[int, List[Bout]],
    sequences: Mapping[int, TrackSequence],
    *,
    fps: float,
    action_names: Sequence[str] = (),
) -> Dict[str, Any]:
    """Per-track and overall statistics, keyed by behavior name."""
    A = max([len(action_names)] + [s.probs.shape[1] for s in sequences.values() if len(s)] + [1])
    names = [_name(action_names, a) for a in range(A)]
    rate = float(fps) if fps and fps > 0 else 1.0
    tracks: Dict[str, Any] = {}
    total_frames = np.zeros(A)
    total_bouts = np.zeros(A, dtype=np.int64)
    transitions = np.zeros((A, A), dtype=np.int64)
    for tid, track_bouts in sorted(bouts.items()):
        seq = sequences.get(tid)
        frames_per = np.zeros(A)
        counts = np.zeros(A, dtype=np.int64)
        durations: Dict[int, List[float]] = {a: [] for a in range(A)}
        first: Dict[int, int] = {}
        for bout in track_bouts:
            frames_per[bout.action] += bout.frames
            counts[bout.action] += 1
            durations[bout.action].append(bout.frames / rate)
            first.setdefault(bout.action, bout.start)
        local = np.zeros((A, A), dtype=np.int64)
        for prev, nxt in zip(track_bouts, track_bouts[1:]):
            # A transition only within continuous tracking: across a gap the
            # animal may have done anything.
            if nxt.start - prev.end <= DEFAULT_MAX_GAP + 1 and prev.action != nxt.action:
                local[prev.action, nxt.action] += 1
        transitions += local
        covered = float(frames_per.sum())
        start = int(seq.frames[0]) if seq is not None and len(seq) else (
            track_bouts[0].start if track_bouts else 0)
        end = int(seq.frames[-1]) if seq is not None and len(seq) else (
            track_bouts[-1].end if track_bouts else 0)
        tracks[str(tid)] = {
            "first_frame": start,
            "last_frame": end,
            "tracked_seconds": round(covered / rate, 3),
            # How much of its own lifetime the track was actually seen. Low
            # coverage means fragmented identity: read its numbers with care.
            "coverage": round(covered / max(1, end - start + 1), 4),
            "time_budget": {names[a]: round(frames_per[a] / covered, 4) if covered else 0.0
                            for a in range(A)},
            "bouts": {names[a]: int(counts[a]) for a in range(A)},
            "mean_bout_s": {names[a]: round(float(np.mean(durations[a])), 3)
                            if durations[a] else 0.0 for a in range(A)},
            "latency_s": {names[a]: round((first[a] - start) / rate, 3)
                          for a in range(A) if a in first},
            "transitions": local.tolist(),
        }
        total_frames += frames_per
        total_bouts += counts
    covered_all = float(total_frames.sum())
    return {
        "fps": rate,
        "action_names": names,
        "tracks": tracks,
        "overall": {
            "tracks": len(tracks),
            "tracked_seconds": round(covered_all / rate, 3),
            "time_budget": {names[a]: round(total_frames[a] / covered_all, 4)
                            if covered_all else 0.0 for a in range(A)},
            "bouts": {names[a]: int(total_bouts[a]) for a in range(A)},
            "transitions": transitions.tolist(),
        },
    }


def population_timeline(
    frames: Mapping[int, Mapping[str, Any]],
    *,
    fps: float,
    bin_s: float = 60.0,
    num_actions: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Expected number of animals in each behavior, per time bin. No identity used.

    Each detection contributes its probability vector, so an animal 70% likely
    courting adds 0.7 to courtship. Averaged over the frames of the bin, this
    is "how many animals were doing X", robust to every identity error.
    """
    A = int(num_actions) if num_actions else _num_actions(frames)
    rate = float(fps) if fps and fps > 0 else 1.0
    width = max(1, int(round(float(bin_s) * rate)))
    bins: Dict[int, Dict[str, Any]] = {}
    for frame in sorted(frames):
        payload = frames[frame]
        b = int(frame) // width
        row = bins.setdefault(b, {"frames": 0, "animals": 0.0, "expected": np.zeros(A)})
        row["frames"] += 1
        boxes = payload.get("boxes") or []
        probs = payload.get("action_probs") or []
        aids = payload.get("action_ids") or []
        row["animals"] += len(boxes)
        for i in range(len(boxes)):
            if i < len(probs) and probs[i] is not None and len(probs[i]) == A:
                p = np.asarray(probs[i], dtype=np.float64)
                total = float(p.sum())
                row["expected"] += p / total if total > 0 else 0.0
            elif i < len(aids) and 0 <= int(aids[i]) < A:
                row["expected"][int(aids[i])] += 1.0
    out: List[Dict[str, float]] = []
    for b in sorted(bins):
        row = bins[b]
        n = max(1, row["frames"])
        entry: Dict[str, float] = {
            "bin_start_s": round(b * width / rate, 3),
            "frames": int(row["frames"]),
            "animals": round(row["animals"] / n, 4),
        }
        for a in range(A):
            entry[str(a)] = round(float(row["expected"][a]) / n, 4)
        out.append(entry)
    return out


def proximity(
    sequences: Mapping[int, TrackSequence],
    *,
    fps: float,
    contact_lengths: float = 1.5,
) -> List[Dict[str, Any]]:
    """For every pair of tracks seen together: how close, and how often in contact.

    Distance is between box centres; "contact" is closer than
    ``contact_lengths`` body lengths (the median longer box side of the pair).
    """
    rate = float(fps) if fps and fps > 0 else 1.0
    ids = sorted(t for t, s in sequences.items() if len(s))
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(ids):
        sa = sequences[a]
        for b in ids[i + 1:]:
            sb = sequences[b]
            common, ia, ib = np.intersect1d(sa.frames, sb.frames, return_indices=True)
            if len(common) == 0:
                continue
            dist = np.linalg.norm(sa.centers[ia] - sb.centers[ib], axis=1)
            body = float(np.median(np.concatenate([sa.sizes[ia], sb.sizes[ib]])))
            contact = dist < contact_lengths * max(body, _EPS)
            out.append({
                "track_a": int(a),
                "track_b": int(b),
                "together_s": round(len(common) / rate, 3),
                "mean_distance_px": round(float(dist.mean()), 3),
                "contact_fraction": round(float(contact.mean()), 4),
                "body_length_px": round(body, 3),
            })
    return out


# ══════════════════════════════════════════════════════════════════════════
# Files
# ══════════════════════════════════════════════════════════════════════════

def write_outputs(
    out_dir: str,
    bouts: Mapping[int, List[Bout]],
    summary: Mapping[str, Any],
    population: Sequence[Mapping[str, float]],
    pairs: Sequence[Mapping[str, Any]] = (),
    *,
    fps: float,
    action_names: Sequence[str] = (),
) -> Dict[str, str]:
    """Write ``bouts.csv``, ``summary.json``, ``population.csv``, ``proximity.csv``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rate = float(fps) if fps and fps > 0 else 1.0
    paths = {name: str(out / name) for name in
             ("bouts.csv", "summary.json", "population.csv", "proximity.csv")}

    with open(paths["bouts.csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "action", "action_name", "start_frame", "end_frame",
                         "start_s", "duration_s", "mean_prob"])
        for tid in sorted(bouts):
            for bout in bouts[tid]:
                writer.writerow([bout.track_id, bout.action, _name(action_names, bout.action),
                                 bout.start, bout.end, round(bout.start / rate, 3),
                                 round(bout.duration(rate), 3), bout.mean_prob])

    with open(paths["summary.json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(paths["population.csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        A = len([k for k in (population[0] if population else {}) if k.isdigit()])
        writer.writerow(["bin_start_s", "frames", "animals"]
                        + [_name(action_names, a) for a in range(A)])
        for row in population:
            writer.writerow([row["bin_start_s"], row["frames"], row["animals"]]
                            + [row[str(a)] for a in range(A)])

    with open(paths["proximity.csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fields = ["track_a", "track_b", "together_s", "mean_distance_px",
                  "contact_fraction", "body_length_px"]
        writer.writerow(fields)
        for row in pairs:
            writer.writerow([row[f] for f in fields])
    return paths


def analyze_predictions(
    predictions: str,
    output_dir: str,
    *,
    fps: Optional[float] = None,
    stay: float = DEFAULT_STAY,
    min_bout_s: float = DEFAULT_MIN_BOUT_S,
    max_gap: int = DEFAULT_MAX_GAP,
    bin_s: float = 60.0,
    action_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The whole analysis, from a ``predictions.json`` to files. Returns the summary.

    ``fps`` and ``action_names`` default to the file's ``_meta`` (written by
    ``yoake predict``), then to 30 fps and numbered behaviors.
    """
    frames, meta = load_predictions(predictions)
    rate = float(fps or meta.get("fps") or 30.0)
    names = list(action_names or meta.get("action_names") or [])
    A = max(len(names), _num_actions(frames)) if frames else max(len(names), 1)
    sequences = track_sequences(frames, A)
    bouts = decode_tracks(sequences, fps=rate, stay=stay, min_bout_s=min_bout_s, max_gap=max_gap)
    summary = summarize(bouts, sequences, fps=rate, action_names=names)
    summary["settings"] = {"stay": stay, "min_bout_s": min_bout_s, "max_gap": max_gap,
                           "bin_s": bin_s, "source": str(predictions)}
    population = population_timeline(frames, fps=rate, bin_s=bin_s, num_actions=A)
    pairs = proximity(sequences, fps=rate)
    summary["pairs"] = pairs
    summary["files"] = write_outputs(output_dir, bouts, summary, population, pairs,
                                     fps=rate, action_names=names)
    return summary


def bouts_as_dicts(bouts: Mapping[int, List[Bout]]) -> List[Dict[str, Any]]:
    """Plain dicts, for JSON or a GUI table."""
    return [asdict(b) for tid in sorted(bouts) for b in bouts[tid]]
