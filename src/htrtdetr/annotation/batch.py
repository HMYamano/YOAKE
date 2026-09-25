"""Cancellable project inference and sampling, independent of the GUI.

The caller supplies detached video inputs and consumes events on its UI thread.
One failed video is a result, not an exception that loses the rest of the batch.
"""

from __future__ import annotations

import bisect
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
    #: Run the detector on every ``detect_stride``-th frame instead of on all
    #: of them. ``1`` is the old behaviour and stays the default for anything
    #: that genuinely wants a dense pass.
    #:
    #: For active learning, dense is waste: the sampler can only ever choose a
    #: scanned frame, so detections on the 14 frames between two scan positions
    #: are computed, cached and never looked at. Measured on this machine at
    #: 121 frames/s, 20 one-hour videos cost 4.1 hours dense and 17 minutes at
    #: stride 15.
    detect_stride: int = 1
    #: Ask the child for **boxes only**, several frames at a time
    #: (:mod:`htrtdetr.inference.detect_only`).
    #:
    #: On by default, because it is what this loop actually needs and because
    #: the alternative is slower at everything it does. A round trains
    #: ``stage=1``; the identity, temporal and behavior heads that the
    #: streaming path runs on every frame are still at their random
    #: initialisation, and their output is discarded here. Dropping them also
    #: drops the constraint that made the detector run one frame at a time -
    #: the tracker's sequential memory - and batching it is measured 4.8x
    #: faster on the same GPU with the same weights (73.7 -> 350.6 fps).
    #:
    #: The trade is identity. Detections come back with ``track_id = -1``:
    #: :func:`connect_detections` links them by IoU on a dense pass, and a
    #: strided pass has no identity to give anyway - the online tracker used
    #: to join detections 15 frames apart as if they were neighbours, which is
    #: the defect ``detect_stride`` was introduced beside.
    detect_only: bool = True
    #: Frames per forward pass. 0 takes the runner's default (16, which is
    #: where the measured throughput curve flattens).
    detect_batch: int = 0
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
    #: Frames a human has already annotated, carried through from the input so
    #: :func:`sample_project` can score novelty against them. A frame counts as
    #: annotated when it is present, including with no boxes: an empty frame is
    #: a decision a human made and a sample the training set already has.
    annotated: tuple[int, ...] = ()


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


def check_video(candidates, annotations, config, num_frames, frames=None):
    """Run the quality checks over the frames that were actually examined.

    ``frames`` matters as soon as the detector runs sparsely. The checks treat
    every frame in this sequence as one they have an opinion about, so passing
    ``range(num_frames)`` after a stride-15 pass declares 14 frames out of
    every 15 to be places the detector found nothing - and the review inbox
    fills with tens of thousands of items about frames nobody ran a model on.
    """
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
    if frames is None:
        frames = range(num_frames) if num_frames else sorted(combined)
    issues = quality.analyze(
        combined,
        config=config,
        frames=frames,
        tracklets=parts,
        num_frames=num_frames,
    )
    behaviors = intervals.merge_to_intervals(combined)
    issues.extend(intervals.detect_interval_issues(behaviors, config))
    return issues


def write_scan_frames(video, plan, folder, cancel):
    """Write the planned frames as ``%06d.jpg``, numbered from zero.

    ``yoake predict`` re-indexes an image directory from 0 and has no
    start-frame option, which is why the local numbering and
    :func:`auto_label.remap_frame_indices` exist at all.

    Sequential ``grab()`` with a selective ``retrieve()``, never
    ``cap.set(CAP_PROP_POS_FRAMES)``: seeking is off-by-one at non-integer fps
    and agrees at frame 0, so the error is near-impossible to notice. At
    stride 15 a grab costs about a quarter of a millisecond, so walking an
    hour-long video to collect 6,000 frames is seconds, against the minutes
    the detector would have spent on the 84,000 frames in between.
    """
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    wanted = list(plan.frames)
    if not wanted:
        return 0
    cap = cv2.VideoCapture(str(video.path))
    written = 0
    try:
        if not cap.isOpened():
            raise OSError("Video codec is unsupported or the file cannot be opened")
        # ``position`` is the index of the frame the next read/grab consumes.
        #
        # Grab up to the target and then ``read()`` it. Not ``grab()`` through
        # the target and ``retrieve()``: retrieve returns the frame the LAST
        # grab consumed, which is target-1, and the only symptom is that every
        # exported frame is one early - invisible at frame 0, where it agrees.
        # The first version of this loop had exactly that bug, and the index
        # test did not catch it because the frame numbers come from the index
        # map and were right; only the pixels were wrong.
        position = 0
        for local, target in enumerate(wanted):
            if cancel.is_set():
                raise Cancelled()
            while position < target:
                if cancel.is_set():
                    raise Cancelled()
                if not cap.grab():
                    raise OSError(f"Could not read frame {position} while collecting frame {target}")
                position += 1
            ok, frame = cap.read()
            position += 1
            if not ok or frame is None:
                # predict enumerates the remaining files, so skipping one
                # would shift every later prediction onto the wrong frame.
                raise OSError(f"Could not read frame {target}")
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise OSError(f"Could not encode frame {target}")
            # imencode + write_bytes, not imwrite: imwrite silently returns
            # False for any path outside the active code page on Windows, and
            # a project folder named in Japanese is the normal case here.
            (folder / f"{local:06d}.jpg").write_bytes(buffer.tobytes())
            written += 1
    finally:
        cap.release()
    return written


def predict_video(video, options, root, output, cancel, progress):
    stride = max(1, int(getattr(options, "detect_stride", 1)))
    index_map = {}
    source = str(video.path)
    if stride > 1:
        plan = auto_label.export_plan(0, -1, video.num_frames, "scan", stride=stride)
        folder = output / "frames"
        progress(0.1, f"Collecting every {stride}th frame")
        written = write_scan_frames(video, plan, folder, cancel)
        if not written:
            raise OSError("No frames could be read for detection")
        index_map = plan.index_map
        source = str(folder)
    req = auto_label.PredictRequest(
        source=source,
        weights=options.weights,
        output_dir=str(output),
        score_threshold=options.score_threshold,
        config=options.config,
        detect_only=bool(getattr(options, "detect_only", True)),
        batch_size=int(getattr(options, "detect_batch", 0)),
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
    found = auto_label.load_candidates(
        output / "predictions.json", min_score=options.score_threshold
    )
    # Local index -> real video frame. Without this every candidate from a
    # strided run lands on the wrong frame; an empty map is the identity, so
    # the dense path is unchanged.
    return auto_label.remap_frame_indices(found, index_map)



def known_vectors(results: list[VideoResult]) -> list[tuple[float, ...]]:
    """Appearance vectors for the parts of the project a human has already done.

    Novelty is distance to the nearest of these, so without them
    :class:`~htrtdetr.annotation.sampling.SamplingContext` takes its constant
    branch and the signal contributes nothing - which is what used to happen at
    every call site, making a second Smart Annotate run a re-roll rather than a
    step.

    An annotated frame usually has no vector of its own: features are scanned
    every ``stride`` frames, so frame 17 is absent when the stride is 15. Each
    annotated frame is snapped to the nearest *scanned* frame instead, at most
    ``stride / 2`` away.

    That is the right resolution rather than a compromise. The sampler can only
    ever select a scanned frame, so "already covered" has to be expressed in the
    units the selection is made in. Snapping says *this part of the video is
    done*, which is what novelty needs; a frame-exact answer would be a
    precision the scan cannot support and would change no decision.
    """
    out: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for r in results:
        if not r.annotated or not r.features:
            continue
        frames = [f.frame for f in r.features]
        for target in r.annotated:
            i = bisect.bisect_left(frames, target)
            # The two scanned frames bracketing the annotated one; whichever is
            # closer wins, and a tie takes the earlier so the result is stable.
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(frames):
                    gap = abs(frames[j] - target)
                    if best is None or gap < best[0]:
                        best = (gap, j)
            if best is None:
                continue
            vector = tuple(r.features[best[1]].embedding)
            if vector and vector not in seen:
                seen.add(vector)
                out.append(vector)
    return out


def sample_project(results: list[VideoResult], request, *, balanced=True):
    """One total budget across videos, with a minimum share before global ranks.

    Each video's ordered proposals come from the existing sampling algorithms.
    Global ranking is computed from the combined feature pool, so a video's
    duration does not determine its score or override the explicit budget.
    """
    valid = [r for r in results if r.features and r.status != "failed"]
    # Project-wide, not per-video: a frame is worth annotating when it is
    # unlike anything already labelled *anywhere* in the study, and the budget
    # this function spends is project-wide too.
    known = known_vectors(valid)
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
            all_features, seed=request.seed, weights=request.weights, known=known
        ).combined()
        offset = 0
        for r in valid:
            scores[r.video_id] = {f.frame: combined[offset + i] for i, f in enumerate(r.features)}
            offset += len(r.features)
    for r in valid:
        ctx = sampling.SamplingContext(
            r.features, seed=request.seed, weights=request.weights, known=known
        )
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


def keep_chosen(results: list[VideoResult]) -> None:
    """Throw away every candidate outside the frames the sampler chose.

    **This is the difference between a round and an analysis**, and without it
    the two are the same operation with a different button.

    A round asks for 50-200 frames. To find them the detector scans the whole
    scope - 36,000 positions across twenty one-hour videos - and every one of
    those positions comes back carrying boxes. Merged whole, that is roughly
    72,000 candidates arriving in a Review Inbox that was asked for a hundred
    frames: the user is handed the entire project to check, the timeline fills
    end to end, and the one number they wanted - *how much of this round is
    left* - cannot be read off anything. The scan is scaffolding. Only the
    picks are the job.

    What survives: the chosen frames, and every frame inside a chosen clip
    (:meth:`~htrtdetr.annotation.sampling.Selection.frame_set`).

    What does not survive is still *cached*. The detections are on disk under
    their model fingerprint, so nothing here is recomputed if the same round is
    run again, and a dense pre-label pass - which does not sample, and does not
    come through this function - still gets all of them.

    A video the sampler gave nothing is emptied rather than left alone. It was
    scanned, it was ranked, and it lost; carrying its boxes into Review would
    say the opposite of what the ranking decided.
    """
    for result in results:
        if result.status == "failed":
            continue
        wanted = result.selection.frame_set() if result.selection is not None else set()
        result.candidates = {
            frame: row for frame, row in result.candidates.items() if frame in wanted
        }


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
        """Every video in turn, then - if this was a round - the choosing.

        **A sampling run's candidates are not final until the last video is
        done.** The budget is spent project-wide, so no video's picks are known
        until every video has been scanned, and :func:`keep_chosen` cannot trim
        anything before that. The ``video`` event therefore carries a full
        scan's worth of boxes and the ``finished`` event carries the trimmed
        ones; a caller that merges on ``video`` merges the scan.

        The consequence on cancel is the reason this is written down: a stopped
        run never reaches :func:`sample_project`, so nothing is trimmed and the
        caller must merge **nothing**. What was scanned is not lost - it is in
        the cache, and the next run gets it back without the detector.
        """
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
            keep_chosen(results)
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
            result.annotated = tuple(sorted(video.annotations))
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
                    stride=int(getattr(options, "detect_stride", 1)),
                    # Part of the key because it changes what comes back, not
                    # only how fast: a detect-only pass has no track ids in it.
                    # Without this, a project scanned before this option
                    # existed would keep serving its old rows for ever.
                    detect_only=bool(getattr(options, "detect_only", True)),
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
            detect_stride = max(1, int(getattr(options, "detect_stride", 1)))
            # Linking detections into tracklets means "these two are the same
            # animal a frame apart". On a strided pass consecutive detections
            # are ``detect_stride`` frames apart, where an animal can have
            # crossed the dish, so the greedy link would invent identities
            # rather than find them.
            if options.connect_tracks and detect_stride == 1:
                candidates = connect_detections(candidates)
            result.candidates = candidates
            progress(0.6, "Checking tracks and behavior")
            # Only the frames a detector actually looked at. Anything else is
            # the checks reporting on frames nobody ran a model on.
            examined = None
            if detect_stride > 1 and options.detect:
                examined = sorted(set(range(0, count, detect_stride)) | set(video.annotations))
            result.issues = check_video(
                candidates, video.annotations, options.quality, count, frames=examined
            )
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

                    # Grab forward instead of seeking. ``cap.grab()`` demuxes a
                    # frame without decoding it, so stepping over the stride is
                    # far cheaper than asking the container to seek - and the
                    # frame list is ascending, which is the case this wins on.
                    #
                    # Measured on a 640x480 30fps H.264 hour (the videos this
                    # was written for), per scanned frame:
                    #
                    #   stride  seek     grab-skip
                    #   15      19.5 ms  4.0 ms
                    #   30      15.7 ms  8.2 ms
                    #   60      ~16 ms   16.8 ms
                    #
                    # So it is a five-fold win at the default stride and a
                    # losing trade past about 60, which is where the cap
                    # below hands the work back to the seek. At the default,
                    # one pass over a twenty-hour project goes from 32 minutes
                    # to under 7 - and this ``read`` is called twice over the
                    # same list, once to encode and once to scan.
                    max_skip = 48
                    cursor = {"at": None}

                    def read(index):
                        here = cursor["at"]
                        if here is not None and here < index <= here + max_skip:
                            for _ in range(index - here - 1):
                                if not cap.grab():
                                    here = None
                                    break
                        else:
                            here = None
                        if here is None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        # A failed read leaves the position unusable, so the
                        # next call has to seek rather than trust the cursor.
                        cursor["at"] = index if ok else None
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
