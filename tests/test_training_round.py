"""The arithmetic of one active-learning round.

Every assertion here corresponds to a way the training CLI fails *silently* -
a run that exits 0, writes a checkpoint, and produces a model that is wrong or
untrained. None of them raises, none of them logs a warning, and none of them
is visible in the loss curve, which is why the decisions live in a pure module
with tests rather than inside a dialog callback:

- a batch larger than the training set trains on zero optimizer steps
  (``drop_last=True`` on the train loader);
- ``model.detector.head.num_classes`` defaults to 1, and the loss uses
  ``num_classes`` as the *background* index, so class 1 trains as "no object";
- a run shorter than the 100-step warmup floor never leaves warmup;
- an empty holdout makes the CLI validate on the training set without saying
  so, and stage 1 selects its best checkpoint on exactly that number;
- a checkpoint that was never written leaves the project pointing at a path
  that does not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import round as rnd  # noqa: E402


def dataset(videos):
    """``{"video 1": [frame indices], ...}`` as a dataset dict."""
    return {
        "class_names": ["fly"],
        "videos": [
            {
                "video_id": vid,
                "frames": [
                    {"frame_index": i, "image_path": f"images/{vid}/{i:06d}.jpg",
                     "objects": [{"class_id": 0, "bbox": [0, 0, 1, 1]}]}
                    for i in frames
                ],
            }
            for vid, frames in videos.items()
        ],
    }


# ── counting ─────────────────────────────────────────────────────────────

def test_a_video_with_no_labels_is_not_a_video():
    """Otherwise the holdout arithmetic can hold back an empty recording."""
    data = dataset({"a": [1, 2, 3], "b": []})
    counts = rnd.count_labels(data, ["fly"])
    assert counts.videos == 1
    assert counts.frames == 3
    assert counts.boxes == 3


def test_the_largest_class_id_is_reported_so_the_caller_can_refuse():
    data = dataset({"a": [1]})
    data["videos"][0]["frames"][0]["objects"] = [{"class_id": 3, "bbox": [0, 0, 1, 1]}]
    assert rnd.count_labels(data, ["fly"]).max_class_id == 3


# ── the batch clamp ──────────────────────────────────────────────────────

def test_the_batch_never_exceeds_the_training_set():
    """drop_last=True means batch > frames is zero steps and a 0.0 loss."""
    for frames in (1, 2, 3, 7, 11):
        counts = rnd.LabelCounts(frames=frames, videos=1, boxes=frames, classes=1)
        plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0)
        assert plan.batch_size <= plan.n_train, (frames, plan)
        assert plan.batch_size >= 1
        assert plan.steps_per_epoch >= 1


def test_a_healthy_dataset_keeps_several_steps_per_epoch():
    counts = rnd.LabelCounts(frames=312, videos=6, boxes=1104, classes=2)
    plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0, holdout=60, val_kind="video")
    assert plan.steps_per_epoch >= 4
    assert plan.n_train == 252


# ── the epoch floor ──────────────────────────────────────────────────────

def test_a_cold_round_costs_the_same_whatever_the_project_grew_to():
    """The point of a step budget. A minute budget bought 400 steps on a small
    project and 3,600 on a large one - the same button, a ten-fold difference
    in what it actually did."""
    small = rnd.derive_plan(
        rnd.LabelCounts(frames=60, videos=2, boxes=120, classes=1),
        has_gpu=True, effort=1.0, holdout=12, val_kind="frames")
    large = rnd.derive_plan(
        rnd.LabelCounts(frames=4000, videos=20, boxes=8000, classes=1),
        has_gpu=True, effort=1.0, holdout=800, val_kind="video")
    assert large.total_steps <= small.total_steps * 1.5
    assert large.est_seconds < 10 * 60, "a round is never a coffee break"
    assert large.epochs <= rnd.MAX_EPOCHS


def test_a_warm_round_is_a_fraction_of_a_cold_one():
    counts = rnd.LabelCounts(frames=200, videos=4, boxes=400, classes=1)
    cold = rnd.derive_plan(counts, has_gpu=True, effort=1.0, holdout=40, val_kind="video")
    warm = rnd.derive_plan(counts, has_gpu=True, effort=1.0, holdout=40,
                           val_kind="video", warm=True)
    assert warm.total_steps < cold.total_steps / 2
    assert "carrying on from the last round" in rnd.plan_sentence(warm).lower()


def test_effort_scales_the_budget_and_the_quote_follows():
    counts = rnd.LabelCounts(frames=200, videos=4, boxes=400, classes=1)
    quick, normal, thorough = (
        rnd.derive_plan(counts, has_gpu=True, effort=rnd.EFFORT[name], holdout=40,
                        val_kind="video")
        for name in ("Quick", "Normal", "Thorough")
    )
    assert quick.total_steps < normal.total_steps < thorough.total_steps
    assert quick.est_seconds < normal.est_seconds < thorough.est_seconds


def test_the_quote_is_built_from_the_steps_that_will_actually_run():
    """The estimate was a guessed 0.20 s/step against a measured 0.6-0.8, so a
    round advertised as 12 minutes took closer to 45."""
    counts = rnd.LabelCounts(frames=200, videos=4, boxes=400, classes=1)
    plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0, holdout=40, val_kind="video")
    assert plan.est_seconds == plan.total_steps * rnd.SEC_PER_STEP["cuda"]
    assert rnd.SEC_PER_STEP["cuda"] >= 0.5, "measured on this machine, not guessed"


def test_the_last_epoch_always_validates():
    """best.pth is only written on a validating epoch that improved, so an
    unvalidated final epoch can end a round with no checkpoint at all."""
    for frames in (30, 60, 137, 200, 1000):
        counts = rnd.LabelCounts(frames=frames, videos=3, boxes=frames, classes=1)
        plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0,
                               holdout=frames // 5, val_kind="video")
        assert plan.epochs % plan.val_interval == 0, (frames, plan)


# ── the holdout ──────────────────────────────────────────────────────────

def test_too_few_labels_hold_nothing_back_and_the_quote_says_so():
    data = dataset({"a": list(range(8))})
    assert rnd.grow_holdout(data) == ()
    plan = rnd.derive_plan(rnd.count_labels(data, ["fly"]), has_gpu=True, effort=1.0)
    assert plan.val_kind == "none"
    assert "trained on" in rnd.plan_sentence(plan)


def test_three_videos_hold_back_the_smallest_whole_video():
    data = dataset({"a": list(range(40)), "b": list(range(40)), "c": list(range(10))})
    holdout = rnd.grow_holdout(data)
    assert {key[0] for key in holdout} == {"c"}
    assert len(holdout) == 10


def test_one_video_holds_back_a_spread_not_a_suffix():
    """A holdout taken off the end of a video is a holdout of one behaviour."""
    data = dataset({"a": list(range(100))})
    holdout = rnd.grow_holdout(data)
    indices = sorted(key[1] for key in holdout)
    assert len(indices) == 20
    assert max(indices) - min(indices) > 50, indices


def test_the_holdout_never_swallows_the_whole_dataset():
    data = dataset({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
    holdout = rnd.grow_holdout(data)
    assert 0 < len(holdout) < 12


def test_a_frozen_holdout_keeps_every_frame_it_held_before():
    """Round N+1 must measure on the same frames, or 'better' means nothing."""
    first = dataset({"a": list(range(40))})
    held = rnd.grow_holdout(first)
    later = dataset({"a": list(range(200))})
    grown = rnd.grow_holdout(later, held)
    assert set(held) <= set(grown)
    assert len(grown) > len(held), "the holdout grows with the project"


def test_a_frame_that_lost_its_labels_leaves_the_holdout():
    first = dataset({"a": list(range(40))})
    held = rnd.grow_holdout(first)
    smaller = dataset({"a": list(range(20))})
    grown = rnd.grow_holdout(smaller, held)
    assert all(key[1] < 20 for key in grown)


def test_the_digest_says_when_two_rounds_used_different_rulers():
    a = rnd.grow_holdout(dataset({"a": list(range(40))}))
    b = rnd.grow_holdout(dataset({"a": list(range(200))}), a)
    assert rnd.holdout_digest(a) != rnd.holdout_digest(b)
    assert rnd.holdout_digest(a) == rnd.holdout_digest(reversed(list(a)))


# ── the split ────────────────────────────────────────────────────────────

def test_the_split_is_disjoint_and_covers_everything():
    data = dataset({"a": list(range(30)), "b": list(range(20))})
    holdout = rnd.grow_holdout(data)
    train, val = rnd.split_dataset(data, holdout)
    train_keys = set(rnd.frame_keys(train))
    val_keys = set(rnd.frame_keys(val))
    assert not (train_keys & val_keys)
    assert train_keys | val_keys == set(rnd.frame_keys(data))
    assert val_keys == set(holdout)


def test_the_split_never_rewrites_an_image_path():
    """Both halves are written beside one images/ tree; the trainer resolves
    each path against the folder holding its annotation file."""
    data = dataset({"a": list(range(30))})
    train, val = rnd.split_dataset(data, rnd.grow_holdout(data))
    for side in (train, val):
        for video in side["videos"]:
            for frame in video["frames"]:
                assert frame["image_path"].startswith("images/a/")


def test_an_empty_holdout_returns_the_same_dataset_twice():
    """The caller then passes one path as both roots, deliberately."""
    data = dataset({"a": [1, 2, 3]})
    train, val = rnd.split_dataset(data, ())
    assert train is data and val is data


def test_the_split_keeps_the_class_names():
    data = dataset({"a": list(range(30))})
    train, val = rnd.split_dataset(data, rnd.grow_holdout(data))
    assert train["class_names"] == ["fly"]
    assert val["class_names"] == ["fly"]


# ── the overrides ────────────────────────────────────────────────────────

def test_the_class_count_is_always_passed():
    """Nothing else in the GUI sets it, and the default trains class 1 as
    background with no error."""
    counts = rnd.LabelCounts(frames=100, videos=2, boxes=200, classes=3)
    plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0)
    over = rnd.train_overrides(plan)
    assert over["model.detector.head.num_classes"] == 3
    assert over["data.num_workers"] == 0
    assert over["train.val_interval"] == plan.val_interval
    assert "init_weights" not in over, "a cold round starts from ImageNet"


def test_a_warm_round_passes_weights_only_never_resume():
    """train.resume also restores the scheduler - so the learning rate resumes
    at its decayed floor - and the previous run's best_metric."""
    counts = rnd.LabelCounts(frames=200, videos=4, boxes=400, classes=1)
    plan = rnd.derive_plan(counts, has_gpu=True, effort=1.0, warm=True)
    over = rnd.train_overrides(plan, init_weights="C:/p/runs/rounds/round-001/stage1/weights/best.pth")
    assert over["init_weights"].endswith("best.pth")
    assert "train.resume" not in over


def test_a_round_trains_a_lighter_decoder_than_production():
    """The round's one deliberate departure from the production architecture.

    Measured on this machine (RTX 5070 Ti, fp16, channels_last, ResNet-18,
    640x640, detector only): the four-layer default spends 5.93 ms of its
    12.95 ms in the decoder and 2.98 ms in the backbone, so halving the
    decoder is where the time is. Two, not one: a single-layer DETR decoder
    has no iterative refinement left.
    """
    plan = rnd.derive_plan(rnd.LabelCounts(frames=100, videos=2, boxes=200, classes=1))
    over = rnd.train_overrides(plan)
    assert over["model.detector.head.num_decoder_layers"] == rnd.ROUND_DECODER_LAYERS
    assert rnd.ROUND_DECODER_LAYERS == 2
    assert "model.detector.backbone.name" not in over, (
        "the backbone is not the lever: ResNet-50 measures 15.8 ms against "
        "ResNet-18's 15.0 at batch 1, because the detector is bound by kernel "
        "launches there rather than by arithmetic"
    )


def test_a_round_writes_weights_without_the_optimizer():
    """A round is never resumed, so two thirds of every checkpoint is dead.

    The next round carries the weights over with ``init_weights`` and builds a
    fresh optimizer on purpose. Adam keeps two moments per parameter: measured
    249 MB against 94 MB for the same model.
    """
    plan = rnd.derive_plan(rnd.LabelCounts(frames=100, videos=2, boxes=200, classes=1))
    assert rnd.train_overrides(plan)["train.save_optimizer_state"] is False


def test_the_production_default_is_untouched():
    """Nothing here may change what the workflow window's Train page does."""
    from htrtdetr.config import get_stage1_config

    cfg = get_stage1_config()
    assert cfg.model.detector.head.num_decoder_layers == 4
    assert cfg.train.save_optimizer_state is True


def test_every_fifth_round_is_forced_cold():
    """A chain of fine-tunes each starting from the last can carry an early
    mistake forever."""
    assert rnd.wants_cold(1, warm_available=False, classes_changed=False)
    assert not rnd.wants_cold(2, warm_available=True, classes_changed=False)
    assert rnd.wants_cold(6, warm_available=True, classes_changed=False), "every 5th"
    assert rnd.wants_cold(3, warm_available=True, classes_changed=True), "head shape changed"


# ── adoption ─────────────────────────────────────────────────────────────

def test_a_run_that_wrote_no_checkpoint_is_never_adopted():
    adopt, said = rnd.adoption(
        checkpoint=False, score=0.1, digest="abc", previous_score=None, previous_digest=""
    )
    assert not adopt
    assert "without writing a checkpoint" in said


def test_the_first_round_is_adopted():
    adopt, _ = rnd.adoption(
        checkpoint=True, score=3.0, digest="abc", previous_score=None, previous_digest=""
    )
    assert adopt


def test_a_better_round_is_adopted_and_a_worse_one_is_not():
    better, said = rnd.adoption(
        checkpoint=True, score=2.0, digest="abc", previous_score=3.0, previous_digest="abc"
    )
    assert better and "Better" in said
    worse, said = rnd.adoption(
        checkpoint=True, score=4.0, digest="abc", previous_score=3.0, previous_digest="abc"
    )
    assert not worse
    assert "keeps the model it had" in said


def test_two_different_rulers_are_not_compared():
    """A grown holdout makes the two scores incomparable, so the newer model -
    trained on strictly more labels - is taken rather than judged."""
    adopt, said = rnd.adoption(
        checkpoint=True, score=9.0, digest="def", previous_score=1.0, previous_digest="abc"
    )
    assert adopt
    assert "Better" not in said


# ── what the holdout actually is ─────────────────────────────────────────

def test_a_whole_video_is_only_claimed_when_it_is_one():
    """"A recording it never saw" is a claim about the model, and it is only
    true when no frame of that video is in the training set."""
    data = dataset({"a": list(range(40)), "b": list(range(40)), "c": list(range(10))})
    whole = rnd.grow_holdout(data)
    assert rnd.val_kind(data, whole) == "video"

    # The same holdout plus one frame from a trained video is not that any more.
    spread = tuple(sorted(set(whole) | {("a", 3)}))
    assert rnd.val_kind(data, spread) == "frames"
    assert rnd.val_kind(data, ()) == "none"


def test_a_holdout_grown_into_trained_videos_stops_claiming_a_whole_video():
    """From round two the holdout grows, and the sentence has to follow it."""
    first = dataset({"a": list(range(40)), "b": list(range(40)), "c": list(range(10))})
    held = rnd.grow_holdout(first)
    assert rnd.val_kind(first, held) == "video"

    later = dataset({"a": list(range(200)), "b": list(range(200)), "c": list(range(10))})
    grown = rnd.grow_holdout(later, held)
    assert rnd.val_kind(later, grown) == "frames", "it is a spread now, and says so"


def test_a_tiny_video_is_not_used_as_a_whole_video_holdout():
    """Two frames is not a validation set; the adoption rule would treat its
    loss as a measurement."""
    data = dataset({"a": list(range(40)), "b": list(range(40)), "c": [0, 1]})
    holdout = rnd.grow_holdout(data)
    assert len(holdout) >= rnd.HOLDOUT_MIN
    assert {k[0] for k in holdout} != {"c"}
