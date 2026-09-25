"""The loop taken ten times: what one round scans, keeps, and counts.

A round is meant to be cheap enough to take twenty times and legible enough
that the twentieth is distinguishable from the second. Three decisions carry
that, and all three are failures that look like successes if they go wrong:

- **what is scanned.** A fixed stride of 15 scans 144,000 positions across a
  20x1h project to choose 100 frames, and pays for it again every round. The
  run still finishes; it just costs an afternoon.
- **what comes back.** Without :func:`batch.keep_chosen` the whole scan is
  merged into the Review Inbox - roughly 72,000 candidates in answer to a
  request for a hundred frames. Nothing errors; the round simply becomes
  indistinguishable from an analysis of the entire project.
- **what counts as done.** A chosen frame with nothing in it is a decision a
  human made. If it does not count, the round's counter stops below its target
  forever and the user has no way to tell a finished round from a stalled one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import batch as batch_mod  # noqa: E402
from htrtdetr.annotation import round as rnd  # noqa: E402
from htrtdetr.annotation import sampling  # noqa: E402
from htrtdetr.gui import auto_label  # noqa: E402


HOUR_30FPS = 108_000
TWENTY_HOURS = HOUR_30FPS * 20


# ── what is scanned ──────────────────────────────────────────────────────
def test_one_hour_still_scans_at_the_old_default():
    """The floor wins on a single video, so nothing existing changes.

    Including, and this is the point of the floor, the frame-feature cache:
    it is keyed on the stride, so a project whose stride moved would re-walk
    every video once for nothing.
    """
    assert sampling.scan_stride(HOUR_30FPS, fps=30.0) == 15


def test_a_twenty_hour_project_scans_four_times_more_coarsely():
    assert sampling.scan_stride(TWENTY_HOURS, fps=30.0) == 60


def test_the_stride_never_leaves_its_band():
    assert sampling.scan_stride(0) == sampling.SCAN_STRIDE_FLOOR
    assert sampling.scan_stride(10) == sampling.SCAN_STRIDE_FLOOR
    # Two hours of video at 25 fps: the ceiling follows the frame rate.
    assert sampling.scan_stride(10_000_000, fps=25.0) == 50
    assert sampling.scan_stride(10_000_000, fps=30.0) == 60


def test_the_stride_never_goes_down_as_the_project_grows():
    previous = 0
    for total in (0, 50_000, 200_000, 700_000, 2_160_000, 20_000_000):
        stride = sampling.scan_stride(total, fps=30.0)
        assert stride >= previous
        previous = stride


def test_the_stride_does_not_depend_on_the_budget():
    """The signature has no budget in it, and that is the contract.

    A stride that followed the budget would change when the user moved from
    100 samples to 200 between rounds, invalidating the frame features cached
    under the old one - a cache miss that costs a full re-walk of every video
    in the project and produces exactly the same selection.
    """
    import inspect

    params = inspect.signature(sampling.scan_stride).parameters
    assert set(params) == {"total_frames", "fps"}


# ── what comes back ──────────────────────────────────────────────────────
def candidate(frame: int) -> auto_label.Candidate:
    return auto_label.Candidate([0, 0, 10, 10], track_id=-1, class_id=0)


def scanned(video_id: str, frames, selection=None) -> batch_mod.VideoResult:
    result = batch_mod.VideoResult(video_id, num_frames=1000)
    result.candidates = {f: [candidate(f)] for f in frames}
    result.selection = selection
    return result


def test_only_the_chosen_frames_survive():
    result = scanned(
        "a",
        range(0, 600, 15),
        sampling.Selection(frames=(15, 300)),
    )
    batch_mod.keep_chosen([result])
    assert sorted(result.candidates) == [15, 300]


def test_a_chosen_clip_keeps_every_frame_in_it():
    """A clip is a span of work, not an anchor with context around it."""
    result = scanned(
        "a",
        range(0, 200, 5),
        sampling.Selection(frames=(), clips=(sampling.Clip(start=100, end=110),)),
    )
    batch_mod.keep_chosen([result])
    assert sorted(result.candidates) == [100, 105, 110]


def test_a_video_the_sampler_passed_over_brings_nothing_back():
    result = scanned("a", range(0, 600, 15), None)
    batch_mod.keep_chosen([result])
    assert result.candidates == {}


def test_a_failed_video_is_left_exactly_as_it_was():
    result = scanned("a", [0, 15], None)
    result.status = "failed"
    batch_mod.keep_chosen([result])
    assert sorted(result.candidates) == [0, 15]


def test_trimming_is_per_video():
    first = scanned("a", [0, 15, 30], sampling.Selection(frames=(15,)))
    second = scanned("b", [0, 15, 30], sampling.Selection(frames=(30,)))
    batch_mod.keep_chosen([first, second])
    assert sorted(first.candidates) == [15]
    assert sorted(second.candidates) == [30]


# ── what counts as done ──────────────────────────────────────────────────
def test_an_empty_frame_counts_as_checked():
    """The assertion the round counter rests on.

    ``annotations[frame] = []`` is a human saying "there is nothing here",
    which is both a completed piece of work and a training sample. Counting
    only frames with boxes on them would leave the counter stuck below its
    target on exactly the frames the sampler works hardest to find.
    """
    selection = sampling.Selection(frames=(10, 20, 30))
    done = sampling.progress(selection, {10: [], 20: ["a box"]})
    assert (done.chosen, done.checked, done.remaining) == (3, 2, 1)
    assert not done.complete


def test_progress_counts_clip_frames_too():
    selection = sampling.Selection(frames=(0,), clips=(sampling.Clip(start=5, end=7),))
    done = sampling.progress(selection, {0: [], 5: [], 6: [], 7: []})
    assert (done.chosen, done.checked) == (4, 4)
    assert done.complete


def test_no_selection_is_no_work_rather_than_finished_work():
    done = sampling.progress(None, {1: [], 2: []})
    assert (done.chosen, done.checked) == (0, 0)
    assert not done.complete


# ── agreement, and when to stop ──────────────────────────────────────────
def test_agreement_is_what_survived_the_human():
    assert rnd.agreement(accepted=9, rejected=1, edited=0, drawn=0) == 0.9


def test_a_missed_animal_counts_against_the_model():
    """Boxes the human had to draw are in the denominator.

    Without them a detector that finds one fly of two, and finds it perfectly,
    scores 100% - and the series advice would then tell the user to stop
    exactly when the model is failing half the job.
    """
    assert rnd.agreement(accepted=10, rejected=0, edited=0, drawn=10) == 0.5


def test_an_edited_box_is_not_an_agreement():
    assert rnd.agreement(accepted=0, rejected=0, edited=4, drawn=0) == 0.0


def test_nothing_reviewed_is_not_a_score_of_zero():
    assert rnd.agreement(0, 0, 0, 0) is None


def rows(values):
    return [{"round": i + 1, "agreement": v} for i, v in enumerate(values)]


def test_the_series_says_nothing_early():
    assert rnd.series_note(rows([0.95, 0.95, 0.95]), 3) == ""


def test_the_series_says_nothing_without_enough_rounds_behind_it():
    assert rnd.series_note(rows([0.95]), 9) == ""


def test_a_flat_high_series_sends_the_user_to_the_whole_video():
    note = rnd.series_note(rows([0.90, 0.91, 0.92]), 9)
    assert "every frame" in note


def test_a_flat_low_series_does_not_say_the_dataset_is_ready():
    note = rnd.series_note(rows([0.60, 0.61, 0.62]), 9)
    assert "every frame" not in note
    assert "60%" in note or "62%" in note


def test_a_series_that_is_still_moving_just_reports_the_number():
    note = rnd.series_note(rows([0.50, 0.70, 0.90]), 9)
    assert note.startswith("You kept 90%")


def test_rounds_without_an_agreement_are_skipped_not_counted_as_zero():
    log = [{"round": 1, "agreement": 0.90}, {"round": 2}, {"round": 3, "agreement": 0.91}]
    assert rnd.series_note(log, 9) == ""  # only two figures: not a window yet


# ── the plan sentence carries the series ─────────────────────────────────
def plan_for(number: int) -> rnd.RoundPlan:
    counts = rnd.count_labels(
        {
            "class_names": ["fly"],
            "videos": [
                {
                    "video_id": "a",
                    "frames": [
                        {"frame_index": i, "objects": [{"class_id": 0, "bbox": [0, 0, 1, 1]}]}
                        for i in range(40)
                    ],
                }
            ],
        }
    )
    return rnd.derive_plan(counts, number=number, has_gpu=True, holdout=8, val_kind="frames")


def test_the_plan_says_which_round_this_is():
    assert rnd.plan_sentence(plan_for(7)).startswith("Round 7.")


def test_the_plan_carries_the_stopping_advice():
    text = rnd.plan_sentence(plan_for(9), rows([0.90, 0.91, 0.92]))
    assert "every frame" in text


def test_the_plan_is_silent_about_a_series_it_was_not_given():
    text = rnd.plan_sentence(plan_for(9))
    assert "stopped teaching" not in text
