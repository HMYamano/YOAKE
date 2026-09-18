"""The active-learning loop: does a second pass continue the first?

`sampling.SamplingContext.known` has always been able to answer "do not
re-propose what is already labelled", but no caller filled it, so the novelty
signal took its constant branch and `diversity` - the cold-start strategy, the
one used when no model exists - started its traversal from a seeded random
frame every time. Running Smart Annotate again therefore re-rolled the
selection instead of advancing it. These tests pin the loop shut.
"""

from __future__ import annotations

import math

from htrtdetr.annotation import batch, sampling
from htrtdetr.annotation.features import FrameFeature, distance, farthest_point_order


def unit(degrees: float) -> tuple[float, float]:
    """An L2-normalised 2-D vector, so cosine distance is a known function."""
    r = math.radians(degrees)
    return (math.cos(r), math.sin(r))


def fan(angles, *, start: int = 0, stride: int = 10) -> list[FrameFeature]:
    """Frames whose appearance fans out over an arc, scanned every `stride`."""
    return [
        FrameFeature(frame=start + i * stride, embedding=unit(a))
        for i, a in enumerate(angles)
    ]


ARC = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]


# ── farthest_point_order(seeds=...) ──────────────────────────────────────

def test_seeds_replace_the_arbitrary_start():
    vectors = [unit(a) for a in ARC]
    # first=0 would start at the 0 degree end; the seed sits there, so the
    # traversal must begin at the other end instead.
    picked = farthest_point_order(vectors, 1, first=0, seeds=[unit(0.0)])
    assert picked == [len(ARC) - 1]


def test_without_seeds_first_still_decides_the_start():
    vectors = [unit(a) for a in ARC]
    assert farthest_point_order(vectors, 1, first=3) == [3]


def test_seeding_keeps_the_prefix_property():
    """A k-pick run must stay a prefix of a 2k-pick run, seeded or not.

    This is what lets the dialog change the budget without recomputing, and
    seeding only rewrites the initial nearest-distance array, so it holds.
    """
    vectors = [unit(a) for a in ARC]
    seeds = [unit(0.0)]
    short = farthest_point_order(vectors, 3, seeds=seeds)
    long = farthest_point_order(vectors, 6, seeds=seeds)
    assert long[:3] == short


def test_an_empty_seed_vector_is_not_a_seed():
    """A frame with no embedding must not silently anchor the traversal."""
    vectors = [unit(a) for a in ARC]
    assert farthest_point_order(vectors, 1, first=2, seeds=[()]) == [2]


# ── the strategy that has to iterate on a cold start ─────────────────────

def test_diversity_avoids_what_is_already_labelled():
    feats = fan(ARC)
    plain = sampling.SamplingContext(feats, seed=1)
    aware = sampling.SamplingContext(feats, seed=1, known=[unit(0.0), unit(20.0)])

    picked_plain = sampling.select_frames(plain, 3, sampling.DIVERSITY)
    picked_aware = sampling.select_frames(aware, 3, sampling.DIVERSITY)

    assert picked_plain != picked_aware
    # The two labelled frames sit at indices 0 and 1; a selection that still
    # offers them is the bug this test exists for.
    assert 0 not in picked_aware and 1 not in picked_aware
    # And it should reach for the far end of the arc.
    assert len(ARC) - 1 in picked_aware


def test_novelty_is_dead_without_known_and_alive_with_it():
    feats = fan(ARC)
    plain = sampling.SamplingContext(feats, seed=0).components()["novelty"]
    aware = sampling.SamplingContext(
        feats, seed=0, known=[unit(0.0)]
    ).components()["novelty"]

    assert len(set(plain)) == 1, "no known frames: novelty must be a constant"
    assert len(set(aware)) > 1, "known frames given: novelty must discriminate"
    # Nearest to the labelled frame is least novel, furthest is most.
    assert aware[0] < aware[-1]


# ── known_vectors: annotated frames are not scanned frames ───────────────

def test_known_vectors_snap_to_the_nearest_scanned_frame():
    feats = fan(ARC, stride=10)          # frames 0, 10, 20, ... 70
    result = batch.VideoResult("v", features=feats, annotated=(12, 38))
    known = batch.known_vectors([result])
    # 12 -> 10 (index 1), 38 -> 40 (index 4). Neither frame was scanned.
    assert known == [feats[1].embedding, feats[4].embedding]


def test_known_vectors_is_empty_before_anything_is_annotated():
    result = batch.VideoResult("v", features=fan(ARC), annotated=())
    assert batch.known_vectors([result]) == []


def test_known_vectors_deduplicates_across_videos():
    feats = fan(ARC)
    a = batch.VideoResult("a", features=feats, annotated=(0, 1, 2))
    b = batch.VideoResult("b", features=feats, annotated=(0,))
    # 0, 1 and 2 all snap to scanned frame 0 in a 10-frame stride, and video b
    # repeats it, so one vector should come back rather than four.
    assert batch.known_vectors([a, b]) == [feats[0].embedding]


def test_a_failed_video_contributes_no_known_vectors():
    ok = batch.VideoResult("ok", features=fan(ARC), annotated=(0,))
    broken = batch.VideoResult("broken", features=[], annotated=(0, 10))
    assert len(batch.known_vectors([ok, broken])) == 1


# ── end to end: the second pass over one project ─────────────────────────

def test_sample_project_moves_on_after_the_first_pass():
    request = sampling.SamplingRequest(
        target=sampling.TARGET_DETECTION,
        budget=sampling.Budget(samples=3),
        strategy=sampling.DIVERSITY,
        seed=7,
    )

    def run(annotated):
        result = batch.VideoResult(
            "v", features=fan(ARC), annotated=annotated, num_frames=80, fps=25.0
        )
        batch.sample_project([result], request, balanced=False)
        return list(result.selection.frames)

    first = run(())
    second = run(tuple(first))

    assert first, "the first pass selected nothing"
    assert not set(first) & set(second), (
        "the second pass re-offered frames the human just annotated"
    )


# ── the wiring: does the annotator's dict reach the sampler? ─────────────

def test_batch_runner_carries_annotated_frames_onto_the_result(tmp_path):
    """The link that makes all of the above reachable from the GUI.

    `_batch_inputs` already deep-copies `video.annotations` into `VideoInput`,
    but nothing used to read it for sampling. If this assertion breaks, novelty
    is dead again at the one place a user can reach.
    """
    import pytest

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from htrtdetr.annotation.objects import AnnObject

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for i in range(12):
        image = np.full((48, 64, 3), 30, np.uint8)
        cv2.rectangle(image, (i + 5, 5), (i + 15, 15), (180, 180, 180), -1)
        writer.write(image)
    writer.release()

    video = batch.VideoInput(
        "clip",
        path,
        {},
        {3: [AnnObject([5, 5, 15, 15], track_id=0)], 7: []},
        10.0,
        12,
    )
    runner = batch.BatchRunner(tmp_path, tmp_path, predictor=lambda *a: {})
    results = runner.run([video], batch.BatchOptions(detect=False, sample=False))

    assert [r.status for r in results] == ["completed"]
    # Frame 7 has no boxes and still counts: an empty frame is a decision a
    # human made, and a sample the training set already contains.
    assert results[0].annotated == (3, 7)
