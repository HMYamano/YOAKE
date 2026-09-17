"""Cancellable project inference and sampling, independent of the GUI.

The caller supplies detached video inputs and consumes events on its UI thread.
One failed video is a result, not an exception that loses the rest of the batch.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Event

from ..gui import auto_label
from . import features, intervals, quality, sampling, tracklet
from .cache import CacheKey, DiskCache, video_fingerprint


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    path: Path | None
    candidates: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)
    fps: float = 25.0
    num_frames: int = 0


@dataclass(frozen=True)
class BatchOptions:
    detect: bool = False
    weights: str = ""
    config: str = ""
    device: str = "cpu"
    score_threshold: float = 0.3
    connect_tracks: bool = True
    sample: bool = True
    stride: int = 15
    balanced: bool = True
    request: sampling.SamplingRequest = field(default_factory=sampling.SamplingRequest)
    quality: quality.QualityConfig = field(default_factory=quality.QualityConfig)


@dataclass
class VideoResult:
    video_id: str
    status: str = "completed"
    error: str = ""
    candidates: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    selection: sampling.Selection | None = None
    fps: float = 25.0
    num_frames: int = 0
    width: int = 0
    height: int = 0


def connect_detections(candidates: dict, threshold=0.3) -> dict:
    """Class-aware greedy links for detector rows lacking IDs; IDs stay local.

    Existing model track IDs are preserved. The proposals remain reviewable,
    and gaps are left disconnected rather than inventing cross-gap identities.
    """
    result = {f: [c.copy() for c in cs] for f, cs in candidates.items()}
    next_id = 1 + max((c.track_id for cs in result.values() for c in cs), default=0)
    previous: list[auto_label.Candidate] = []
    previous_frame = -2
    for frame, row in sorted(result.items()):
        used = {c.track_id for c in row if c.track_id >= 0}
        for cand in row:
            if cand.track_id >= 0:
                continue
            matches = [
                (auto_label.iou_any(cand.bbox, other.bbox, cand.obb, other.obb), other)
                for other in previous
                if other.class_id == cand.class_id and other.track_id not in used
            ]
            score, match = max(matches, key=lambda entry: entry[0], default=(0, None))
            if frame == previous_frame + 1 and match is not None and score >= threshold:
                cand.track_id = match.track_id
            else:
                cand.track_id = next_id
                next_id += 1
            used.add(cand.track_id)
        previous, previous_frame = row, frame
    return result


def check_video(candidates, annotations, config, num_frames):
    combined = {f: list(cs) for f, cs in candidates.items()}
    for frame, objects in annotations.items():
        row = combined.setdefault(frame, [])
        tids = {o.track_id for o in objects if o.track_id >= 0}
        row[:] = [c for c in row if c.track_id not in tids]
        row.extend(
            auto_label.Candidate(
                list(o.bbox),
                obb=o.obb,
                score=1.0,
                track_id=o.track_id,
                class_id=o.class_id,
                action_id=o.action_id,
                source="human",
            )
            for o in objects
        )
    parts = tracklet.build_tracklets(combined, max_gap=config.fragment_gap)
    issues = quality.analyze(
        combined,
        config=config,
        frames=range(num_frames) if num_frames else sorted(combined),
        tracklets=parts,
        num_frames=num_frames,
    )
    behaviors = intervals.merge_to_intervals(combined)
    issues.extend(intervals.detect_interval_issues(behaviors, config))
    return issues


def predict_video(video, options, root, output, cancel, progress):
    req = auto_label.PredictRequest(
        source=str(video.path),
        weights=options.weights,
        output_dir=str(output),
        score_threshold=options.score_threshold,
        config=options.config,
    )
    argv = [sys.executable, *auto_label.predict_argv(req)]
    env = auto_label.device_env(options.device, auto_label.child_env(root))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    # A file avoids PIPE deadlocks and lets cancellation work even if inference
    # stops printing (including model loading and a blocked CUDA operation).
    log_path = output / "inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            while process.poll() is None:
                if cancel.wait(0.15):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise Cancelled()
                progress(0.2, "Detecting animals (model loading may take a while)")
            if process.returncode:
                log.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"Inference exited {process.returncode}: {tail}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return auto_label.load_candidates(
        output / "predictions.json", min_score=options.score_threshold
    )


def sample_project(results: list[VideoResult], request, *, balanced=True):
    """One total budget across videos, with a minimum share before global ranks.

    Each video's ordered proposals come from the existing sampling algorithms.
    Global ranking is computed from the combined feature pool, so a video's
    duration does not determine its score or override the explicit budget.
    """
    valid = [r for r in results if r.features and r.status != "failed"]
    frame_budget, clip_budget = request.budget.counts(
        target=request.target, model=request.workload, prelabelled=request.prelabelled
    )
    if request.budget.samples == 1 and request.target == sampling.TARGET_BOTH:
        frame_budget, clip_budget = 1, 0
    pools = {}
    scores = {}
    all_features = [f for r in valid for f in r.features]
    if all_features:
        combined = sampling.SamplingContext(
            all_features, seed=request.seed, weights=request.weights
        ).combined()
        offset = 0
        for r in valid:
            scores[r.video_id] = {f.frame: combined[offset + i] for i, f in enumerate(r.features)}
            offset += len(r.features)
    for r in valid:
        ctx = sampling.SamplingContext(r.features, seed=request.seed, weights=request.weights)
        frames = (
            [
                r.features[i].frame
                for i in sampling.select_frames(ctx, frame_budget, request.strategy)
            ]
            if frame_budget
            else []
        )
        clips = (
            sampling.select_clips(
                ctx,
                clip_budget,
                max(2, round(request.clip_seconds * r.fps)),
                strategy=request.strategy,
            )
            if clip_budget
            else []
        )
        clips = (
            [replace(c, end=min(c.end, r.num_frames - 1)) for c in clips] if r.num_frames else clips
        )
        pools[r.video_id] = (frames, list(clips))
    chosen: dict[str, tuple[list, list]] = {r.video_id: ([], []) for r in valid}
    for kind, budget in enumerate((frame_budget, clip_budget)):
        remaining = budget
        if balanced:
            # Equal base allocation; short/exhausted videos release their slots
            # to the globally ranked remaining pool below.
            quota = budget // max(len(valid), 1)
            if budget >= len(valid):
                quota = max(1, quota)
            for r in valid:
                take = min(quota, len(pools[r.video_id][kind]), remaining)
                chosen[r.video_id][kind].extend(pools[r.video_id][kind][:take])
                del pools[r.video_id][kind][:take]
                remaining -= take
        ranked = []
        for r in valid:
            for order, item in enumerate(pools[r.video_id][kind]):
                index = item if kind == 0 else item.middle
                score = scores[r.video_id].get(index, 0.0)
                ranked.append((-score, order, r.video_id, item))
        ranked.sort(key=lambda v: v[:3])
        for _, _, vid, item in ranked[:remaining]:
            chosen[vid][kind].append(item)
    for r in valid:
        frames, clips = chosen[r.video_id]
        r.selection = sampling.Selection(
            frames=tuple(sorted(frames)),
            clips=tuple(sorted(clips, key=lambda c: c.start)),
            strategy=request.strategy,
            seed=request.seed,
            pool_size=len(r.features),
            reasons={f: request.strategy for f in frames},
            estimated_seconds=request.workload.estimate_seconds(
                len(frames), len(clips), request.prelabelled
            ),
        )


class BatchRunner:
    def __init__(
        self, root: Path, cache_dir: Path, *, predictor: Callable = predict_video, encoder=None
    ):
        self.root = Path(root)
        self.cache = DiskCache(cache_dir)
        self.predictor = predictor
        self.encoder = encoder
        self.cancel = Event()

    def run(self, videos, options, emit=lambda event: None):
        results = []
        for ordinal, video in enumerate(videos):
            if self.cancel.is_set():
                break
            last_progress = -1

            def progress(fraction, operation, video=video, ordinal=ordinal):
                nonlocal last_progress
                # Coalesce a long subprocess's heartbeat to keep the UI queue bounded.
                value = int(fraction * 100)
                if value == last_progress:
                    return
                last_progress = value
                emit(
                    {
                        "kind": "progress",
                        "video_id": video.video_id,
                        "index": ordinal + 1,
                        "total": len(videos),
                        "fraction": fraction,
                        "overall": (ordinal + fraction) / max(1, len(videos)),
                        "operation": operation,
                    }
                )

            try:
                result = self._video(video, options, progress)
            except Cancelled:
                break
            except Exception as exc:  # noqa: BLE001 - one failed video must not stop the batch.
                result = VideoResult(
                    video.video_id, status="failed", error=f"{type(exc).__name__}: {exc}"
                )
            results.append(result)
            emit({"kind": "video", "result": result})
        if options.sample and not self.cancel.is_set():
            sample_project(results, options.request, balanced=options.balanced)
        emit({"kind": "finished", "results": results, "cancelled": self.cancel.is_set()})
        return results

    def _video(self, video, options, progress):
        import cv2

        if video.path is None or not video.path.is_file():
            raise FileNotFoundError(f"Missing video: {video.path or video.video_id}")
        cap = cv2.VideoCapture(str(video.path))
        try:
            if not cap.isOpened():
                raise OSError("Video codec is unsupported or the file cannot be opened")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise OSError("The video contains no readable frames")
            result = VideoResult(
                video.video_id,
                fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                num_frames=count,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            candidates = {f: [c.copy() for c in cs] for f, cs in video.candidates.items()}
            if options.detect:
                key = CacheKey.for_video(
                    video.path,
                    model=video_fingerprint(options.weights),
                    path=str(video.path.resolve()),
                    weights=str(Path(options.weights).resolve()),
                    config=video_fingerprint(options.config),
                    threshold=options.score_threshold,
                    device=options.device,
                    operation="project_detection_v1",
                )
                cached = self.cache.get(key, "candidates")
                if cached is not None:
                    candidates = {
                        int(f): [auto_label.Candidate(**c) for c in cs] for f, cs in cached.items()
                    }
                    result.status = "cached"
                    progress(0.55, "Using cached predictions")
                else:
                    with tempfile.TemporaryDirectory(prefix="yoake-predict-") as temp:
                        candidates = self.predictor(
                            video, options, self.root, Path(temp), self.cancel, progress
                        )
                    if self.cancel.is_set():
                        raise Cancelled()
                    for row in candidates.values():
                        for c in row:
                            c.provenance.update(
                                model=Path(options.weights).name,
                                model_version=video_fingerprint(options.weights),
                                operation="auto_annotation",
                                human_review="pending",
                            )
                    self.cache.put(
                        key,
                        "candidates",
                        {str(f): [asdict(c) for c in cs] for f, cs in candidates.items()},
                    )
            if options.connect_tracks:
                candidates = connect_detections(candidates)
            result.candidates = candidates
            progress(0.6, "Checking tracks and behavior")
            result.issues = check_video(candidates, video.annotations, options.quality, count)
            if options.sample:
                encoder_key = (
                    asdict(self.encoder.info())
                    if self.encoder is not None
                    else {"model": "thumbnail"}
                )
                key = CacheKey.for_video(
                    video.path,
                    path=str(video.path.resolve()),
                    stride=options.stride,
                    encoder=encoder_key,
                    operation="frame_features_v1",
                )
                cached = self.cache.get(key, "features")
                if cached is None:

                    def read(index):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        return frame if ok else None

                    frames = list(range(0, count, max(1, options.stride)))
                    vectors = (
                        self.encoder.encode_frames(
                            read,
                            frames,
                            should_stop=self.cancel.is_set,
                            on_progress=lambda done, total: progress(
                                0.65 + 0.15 * done / max(1, total), "Encoding frames"
                            ),
                        )
                        if self.encoder is not None
                        else {}
                    )
                    scanned = features.scan(
                        read,
                        frames,
                        embeddings=vectors or None,
                        should_stop=self.cancel.is_set,
                        on_progress=lambda done, total: progress(
                            0.8 + 0.15 * done / max(1, total), "Choosing informative frames"
                        ),
                    )
                    if self.cancel.is_set():
                        raise Cancelled()
                    self.cache.put(key, "features", [asdict(f) for f in scanned])
                else:
                    scanned = [features.FrameFeature(**f) for f in cached]
                    if not options.detect:
                        result.status = "cached"
                result.features = [
                    replace(f, **features.detection_stats(candidates.get(f.frame, [])))
                    for f in scanned
                ]
            progress(1.0, "Completed")
            return result
        finally:
            cap.release()
