"""The export at the end of the study: the ways it could ship a wrong dataset.

Every failure guarded here is silent. The release is written once, at the end of
about eight hours of sign-off and ten rounds of labelling, and nothing
downstream diffs it against a previous version - so a mistake in it is not
caught by a test failing, it is caught months later by a result that will not
reproduce. Four shapes of that:

- **The model's boxes leak back into a frame a human fixed.** A per-box merge
  re-inserts exactly the detections the human had just deleted, and the file
  still looks right: same format, more boxes, no error.
- **An empty human frame falls back to the model.** ``[]`` is a human saying
  "nothing here" and ``None`` is a human who never looked; treating them as one
  value deletes every negative example in the dataset, and the ones it deletes
  are precisely the frames the model was wrong about.
- **A video drops out without being named.** Video 18 was never signed off, the
  export skips it, and a dataset described as twenty videos ships with
  nineteen.
- **Frames nobody checked ship as signed off.** A pass re-run further than the
  human verified would otherwise put unchecked model output into the deliverable
  under a signed-off video's name.

Everything here is built from dicts and four ``AnnObject``s. No ledger is
written, no project is opened, and no video is decoded, which is what lets this
file run from either of this project's interpreters in well under a second.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htrtdetr.annotation import release  # noqa: E402
from htrtdetr.annotation.objects import AnnObject  # noqa: E402
from htrtdetr.data.annotation import load_annotation, save_annotation  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def drawn(x=10, y=10, *, track_id=1, class_id=0, source="human", score=None):
    """One object as the annotator holds it, in the project's own class."""
    return AnnObject(
        bbox=[x, y, x + 20, y + 8],
        track_id=track_id,
        class_id=class_id,
        source=source,
        score=score,
    )


def ledger_box(x=100.0, *, class_id=0, score=0.91, track_id=2):
    """One row of the final-pass ledger: [x1, y1, x2, y2, class, score, track]."""
    return [x, 50.0, x + 20.0, 58.0, class_id, score, track_id]


def video(video_id="vid_001", *, num_frames=1000, annotations=None, path="videos/a.mp4"):
    return {
        "video_id": video_id,
        "video_path": path,
        "num_frames": num_frames,
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "annotations": dict(annotations or {}),
    }


def state(*, scanned=1000, verified=((0, 999),), signed_off=1_748_999_999.0):
    return {
        "weights": "best.pth:249000000:170000000000000000",
        "scanned": scanned,
        "verified": [list(span) for span in verified],
        "signed_off": signed_off,
    }


def export(videos, states, ledger, *, class_names=("fly",), action_names=("idle", "court")):
    """``release_plan`` then ``dataset_dict``, the way a caller uses them."""
    plans, excluded = release.release_plan(videos, states)
    data = release.dataset_dict(
        plans,
        lambda vid: sorted(ledger.get(vid, {}).items()),
        lambda vid: next(v["annotations"] for v in videos if v["video_id"] == vid),
        class_names=list(class_names),
        action_names=list(action_names),
    )
    return plans, excluded, data


def frames_of(data, video_id="vid_001"):
    return {
        row["frame_index"]: row
        for v in data["videos"]
        if v["video_id"] == video_id
        for row in v["frames"]
    }


# ══════════════════════════════════════════════════════════════════════════
# The merge rule: a human frame wins whole
# ══════════════════════════════════════════════════════════════════════════

def test_a_hand_drawn_frame_keeps_none_of_the_models_boxes():
    """The rule this module exists for, at its smallest.

    The model found a second animal on this frame and the human did not draw
    it. Per-box merging would keep it; per-frame merging does not, because the
    human's frame is a decision about the whole frame including its gaps.
    """
    out = release.merge_frame([drawn()], [ledger_box(100.0), ledger_box(300.0)])
    assert len(out) == 1
    assert all(obj["source"] == "human" for obj in out)
    assert all(obj["provenance"]["human_review"] == release.HUMAN_DRAWN for obj in out)


def test_a_frame_the_human_cleared_exports_empty_rather_than_falling_back():
    """``[]`` is a label. ``None`` is an absence. They are not the same value."""
    assert release.merge_frame([], [ledger_box()]) == []
    assert len(release.merge_frame(None, [ledger_box()])) == 1


def test_an_untouched_frame_takes_the_models_boxes_and_says_so():
    out = release.merge_frame(None, [ledger_box(100.0, score=0.77, track_id=4)])
    assert out[0]["source"] == "model"
    assert out[0]["score"] == pytest.approx(0.77)
    assert out[0]["track_id"] == 4
    assert out[0]["provenance"]["human_review"] == release.MODEL_SIGNED_OFF
    # The final pass detects; it does not classify behaviour.
    assert out[0]["action_id"] == -1


def test_a_machine_box_the_human_kept_is_accepted_not_drawn():
    """Three grades of evidence, not two.

    ``source`` stays "auto" because that is where the box came from; what the
    human did about it is a different field, and flattening the two loses the
    ability to audit the dataset later.
    """
    out = release.merge_frame([drawn(source="auto", score=0.88)], None)
    assert out[0]["source"] == "auto"
    assert out[0]["provenance"]["human_review"] == release.HUMAN_ACCEPTED


def test_the_live_projects_objects_are_not_touched_by_an_export():
    """The export runs against the session the user is still editing."""
    obj = drawn()
    obj.provenance["tool"] = "wand"
    before = dict(obj.provenance)
    bbox_before = list(obj.bbox)

    record = release.merge_frame([obj], None)[0]
    record["provenance"]["human_review"] = "tampered"
    record["bbox"][0] = -999

    assert obj.provenance == before
    assert "human_review" not in obj.provenance
    assert obj.bbox == bbox_before


def test_a_short_ledger_row_is_defaulted_rather_than_raised_on():
    """Two hours of scanning must not be lost to one truncated line."""
    out = release.merge_frame(None, [[0.0, 0.0, 10.0, 10.0]])
    assert out[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert out[0]["class_id"] == 0
    assert out[0]["track_id"] == -1
    assert out[0]["score"] is None


# ══════════════════════════════════════════════════════════════════════════
# The plan: what is in, and what is named for being out
# ══════════════════════════════════════════════════════════════════════════

def test_a_video_that_is_not_signed_off_is_excluded_and_named():
    """Silently dropping it is the failure the return shape exists to prevent."""
    videos = [video("vid_001"), video("vid_002")]
    states = {"vid_001": state(), "vid_002": state(signed_off=None)}

    plans, excluded = release.release_plan(videos, states)

    assert [plan.video_id for plan in plans] == ["vid_001"]
    assert excluded == [("vid_002", release.NOT_SIGNED_OFF)]


def test_a_video_the_pass_never_scanned_is_named_for_its_own_reason():
    """"Not scanned" and "not signed off" are different jobs for the user."""
    _plans, excluded = release.release_plan([video("vid_003")], {"vid_003": state(scanned=0)})
    assert excluded == [("vid_003", release.NOT_SCANNED)]


def test_a_signed_off_video_with_no_source_path_cannot_be_released():
    """In a labels-only dataset, labels with no video point at nothing."""
    _plans, excluded = release.release_plan(
        [video("vid_004", path="")], {"vid_004": state()}
    )
    assert excluded == [("vid_004", release.NO_VIDEO_PATH)]


def test_the_state_may_ride_on_the_video_when_no_mapping_is_given():
    row = video("vid_001")
    row["final_pass"] = state()
    plans, excluded = release.release_plan([row], None)
    assert not excluded and plans[0].frames == 1000


def test_the_plan_counts_the_whole_signed_off_video():
    """``frames`` is the video the release covers, not the rows in the file.

    A frame the model found nothing in is absent from the ledger and so from
    the file, but it is still a frame inside a video a human signed off.
    Counting rows would shrink the headline number every time the detector
    improved.

    And it is the **whole** video, not the spans the human was taken to. Signing
    off is a statement about the recording; the spans say which parts of it were
    looked at directly, and that distinction is carried on each object rather
    than by dropping the other 99% of the frames.
    """
    videos = [video("vid_001", annotations={0: [drawn()], 10: [], 900: [drawn()]})]
    plans, _excluded = release.release_plan(videos, {"vid_001": state(verified=((0, 499),))})

    plan = plans[0]
    assert plan.frames == 1000
    assert plan.human == 3
    assert plan.model == 997
    assert plan.human + plan.model == plan.frames


def test_a_span_running_past_the_end_of_the_video_is_clipped():
    """A decoder that counted one frame more than the manifest did."""
    plans, _excluded = release.release_plan(
        [video("vid_001", num_frames=100)], {"vid_001": state(verified=((0, 999),))}
    )
    assert plans[0].frames == 100
    assert plans[0].verified == ((0, 99),)


def test_overlapping_spans_are_merged_before_they_are_read():
    """Overlapping spans must collapse, or "how much did you check" double-counts.

    The frame count is the video's, not the spans' - but the spans are what the
    grading reads, and two spans that overlap would otherwise make a frame both
    checked and checked again.
    """
    plans, _excluded = release.release_plan(
        [video("vid_001")], {"vid_001": state(verified=((0, 99), (50, 199), (200, 209)))}
    )
    assert plans[0].verified == ((0, 209),)
    assert plans[0].frames == 1000


# ══════════════════════════════════════════════════════════════════════════
# The dataset
# ══════════════════════════════════════════════════════════════════════════

def test_frames_outside_the_checked_spans_ship_but_say_so():
    """The whole video ships; only the boxes outside the spans are graded down.

    This is the honest half of the sampled review model. The human was taken to
    ~134 places in 108,000 frames and said the video was good. Dropping the rest
    would throw away the dataset; calling the rest "signed_off" would flatten a
    sampled guarantee into a per-frame one that nobody auditing the file later
    could tell apart from a frame a human actually opened.
    """
    videos = [video("vid_001", annotations={700: [drawn()]})]
    ledger = {"vid_001": {300: [ledger_box()], 700: [ledger_box()], 800: [ledger_box()]}}

    _plans, _excluded, data = export(videos, {"vid_001": state(verified=((0, 499),))}, ledger)

    assert sorted(frames_of(data)) == [300, 700, 800]
    grades = {
        frame["frame_index"]: frame["objects"][0]["provenance"]["human_review"]
        for frame in data["videos"][0]["frames"]
    }
    assert grades[300] == release.MODEL_SIGNED_OFF   # inside the checked span
    assert grades[800] == release.MODEL_SAMPLED      # outside it, and says so
    assert grades[700] == release.HUMAN_DRAWN        # the human's own box wins


def test_a_human_frame_beats_the_ledger_row_on_the_same_frame():
    videos = [video("vid_001", annotations={42: [drawn()]})]
    ledger = {"vid_001": {42: [ledger_box(100.0), ledger_box(300.0)]}}

    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    objects = frames_of(data)[42]["objects"]
    assert len(objects) == 1
    assert objects[0]["provenance"]["human_review"] == release.HUMAN_DRAWN


def test_a_frame_the_human_cleared_survives_as_an_empty_row():
    """Not dropped: an empty row is the negative example the training needs."""
    videos = [video("vid_001", annotations={42: []})]
    ledger = {"vid_001": {42: [ledger_box()]}}

    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    rows = frames_of(data)
    assert 42 in rows
    assert rows[42]["objects"] == []


def test_ledger_frames_with_no_boxes_never_become_rows():
    """~1.9M empty rows is what turns a 300 MB file into a 3 GB one."""
    videos = [video("vid_001")]
    ledger = {"vid_001": {10: [], 11: [ledger_box()]}}

    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    assert sorted(frames_of(data)) == [11]


def test_the_rows_come_out_ascending_however_the_two_sources_interleave():
    videos = [video("vid_001", annotations={50: [drawn()], 5: [drawn()]})]
    ledger = {"vid_001": {30: [ledger_box()], 1: [ledger_box()]}}

    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    assert [row["frame_index"] for row in data["videos"][0]["frames"]] == [1, 5, 30, 50]


def test_video_path_survives_into_the_dataset():
    """With no image tree, this path and ``frame_index`` are the only locator."""
    videos = [video("vid_001", path="videos/copulation_01.mp4", annotations={0: [drawn()]})]

    _plans, _excluded, data = export(videos, {"vid_001": state()}, {})

    assert data["videos"][0]["video_path"] == "videos/copulation_01.mp4"


def test_no_frame_claims_an_image_that_was_never_written():
    """``image_path`` is empty on purpose.

    ``load_annotation`` invents ``images/<video_id>/<frame>.jpg`` for a frame
    that has no ``image_path``, and that file does not exist - the 86 GB image
    tree is the thing this export was designed not to produce. A consumer
    reading black frames for a whole dataset is worse than one that says it
    cannot find an image.
    """
    videos = [video("vid_001", annotations={0: [drawn()]})]

    _plans, _excluded, data = export(videos, {"vid_001": state()}, {})

    assert data["meta"]["image_root"] == ""
    assert all(
        row["image_path"] == "" for v in data["videos"] for row in v["frames"]
    )


# ══════════════════════════════════════════════════════════════════════════
# The round trip: the release must be readable by the tools that exist
# ══════════════════════════════════════════════════════════════════════════

def test_the_dict_round_trips_through_save_and_load(tmp_path):
    """Written as JSON, read by ``load_annotation``, written again, read again.

    The release is the deliverable of the whole study, and a release in a
    format only the exporter can read is a release nobody can use. Both hops
    are here because they fail differently: the first catches a dict
    ``load_annotation`` cannot parse, the second catches a field this repo's own
    writer drops on the way back out.
    """
    videos = [video("vid_001", annotations={0: [drawn()], 7: []})]
    ledger = {"vid_001": {3: [ledger_box(100.0), ledger_box(300.0)]}}

    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    first = tmp_path / "release.json"
    first.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_annotation(str(first))

    assert [v.video_id for v in loaded.videos] == ["vid_001"]
    assert loaded.class_names == ["fly"]
    assert loaded.videos[0].video_path == "videos/a.mp4"
    assert [f.frame_index for f in loaded.videos[0].frames] == [0, 3, 7]
    assert [len(f.objects) for f in loaded.videos[0].frames] == [1, 2, 0]

    second = tmp_path / "again.json"
    save_annotation(loaded, str(second))
    again = load_annotation(str(second))

    assert [f.frame_index for f in again.videos[0].frames] == [0, 3, 7]
    assert again.videos[0].video_path == "videos/a.mp4"
    assert again.videos[0].fps == pytest.approx(30.0)
    model_objects = again.videos[0].frames[1].objects
    assert {o.source for o in model_objects} == {"model"}
    assert all(o.provenance["human_review"] == release.MODEL_SIGNED_OFF for o in model_objects)
    assert again.videos[0].frames[0].objects[0].provenance["human_review"] == release.HUMAN_DRAWN


def test_the_release_validates_as_an_annotation_file(tmp_path):
    """The repo's own validator, which is what the trainer's users run first."""
    from htrtdetr.data.annotation import validate_annotation

    videos = [video("vid_001", annotations={0: [drawn()]})]
    ledger = {"vid_001": {3: [ledger_box()]}}
    _plans, _excluded, data = export(videos, {"vid_001": state()}, ledger)

    path = tmp_path / "release.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = validate_annotation(load_annotation(str(path)))

    assert result.valid, result.errors


# ══════════════════════════════════════════════════════════════════════════
# What the user is told
# ══════════════════════════════════════════════════════════════════════════

def test_the_summary_says_what_was_left_out_and_how_much():
    plans = [
        release.VideoPlan("v%02d" % i, "videos/a.mp4", 108_023, 30.0, 108_023, 100, 107_923)
        for i in range(17)
    ]
    excluded = [("v17", release.NOT_SIGNED_OFF), ("v18", release.NOT_SIGNED_OFF),
                ("v19", release.NOT_SIGNED_OFF)]

    sentence = release.summary_sentence(plans, excluded)

    assert sentence == (
        "1,836,391 frames from 17 videos. "
        "3 videos are not signed off and were left out."
    )


def test_one_excluded_video_is_a_grammatical_sentence():
    plans = [release.VideoPlan("v01", "videos/a.mp4", 100, 30.0, 100, 1, 99)]
    sentence = release.summary_sentence(plans, [("v02", release.NOT_SCANNED)])
    assert sentence == (
        "100 frames from 1 video. "
        "1 video is not scanned by the final pass and was left out."
    )


def test_an_export_with_nothing_in_it_says_so_rather_than_printing_zero():
    assert release.summary_sentence([], []) == "No videos are ready to export."


def test_the_size_is_quoted_before_the_file_is_written():
    """Hundreds of MB against the 86 GB the user declined - the number is why."""
    plans = [
        release.VideoPlan("v%02d" % i, "videos/a.mp4", 108_000, 30.0, 108_000, 100, 107_900)
        for i in range(20)
    ]
    text = release.size_estimate(plans)
    assert "MB" in text
    assert "4,320,000 boxes" in text
    # Twenty hours of labels, and not one image.
    assert "No images are written" in text


# ══════════════════════════════════════════════════════════════════════════
# Regressions: what an adversarial read found after the module was green
# ══════════════════════════════════════════════════════════════════════════

def test_the_quoted_size_is_within_a_factor_of_two_of_the_file(tmp_path):
    """The estimate sits next to the Export button, so being 6.7x under is a lie.

    It was. ``BYTES_PER_BOX`` was counted by adding up field names, which missed
    the nested ``provenance`` object and the indentation ``atomic_json`` writes
    - "about 330 MB" for a release that put 2.2 GB on disk. The release is
    written compact by ``write_release`` now, and the constant is measured
    against that writer rather than guessed.
    """
    frames = 5_000
    plan = release.VideoPlan(
        video_id="vid_001", video_path="../videos/a.mp4", num_frames=frames,
        fps=30.0, frames=frames, human=0, model=frames, verified=((0, frames - 1),),
        width=640, height=480,
    )
    rows = lambda _v: ((f, [ledger_box(), ledger_box()]) for f in range(frames))
    out = tmp_path / "annotations.json"
    stats = release.write_release(
        out, [plan], rows, lambda _v: {}, class_names=["fly"], action_names=[]
    )

    quoted = release.size_estimate([plan])
    actual = stats["bytes"]
    # The estimate prints MB; recover the number it claimed and compare.
    claimed_mb = float(quoted.split("about ")[1].split(" ")[0].replace(",", ""))
    assert 0.5 <= (claimed_mb * 1_000_000) / actual <= 2.0, (quoted, actual)


def test_the_writer_never_holds_the_whole_release(tmp_path):
    """A gigabyte of dict, a gigabyte of JSON string and a gigabyte of file at once.

    ``dataset_dict`` builds the release as one object; the real export cannot,
    and the docstring that said it never holds two videos at once was written
    about a function that holds all twenty.
    """
    import tracemalloc

    frames = 20_000
    plan = release.VideoPlan(
        video_id="vid_001", video_path="../videos/a.mp4", num_frames=frames,
        fps=30.0, frames=frames, human=0, model=frames, verified=((0, frames - 1),),
        width=640, height=480,
    )
    rows = lambda _v: ((f, [ledger_box(), ledger_box()]) for f in range(frames))

    tracemalloc.start()
    stats = release.write_release(
        tmp_path / "a.json", [plan], rows, lambda _v: {},
        class_names=["fly"], action_names=[],
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert stats["frames"] == frames
    # Well under the size of the file it just wrote, which is the claim.
    assert peak < stats["bytes"] / 4, (peak, stats["bytes"])


def test_a_frame_past_the_end_of_the_video_is_dropped_and_counted(tmp_path):
    """A row nothing can seek to must not ship under a video row that says otherwise."""
    plan = release.VideoPlan(
        video_id="vid_001", video_path="../videos/a.mp4", num_frames=100,
        fps=30.0, frames=100, human=0, model=100, verified=((0, 99),),
    )
    rows = lambda _v: [(50, [ledger_box()]), (150, [ledger_box()]), (999_999, [ledger_box()])]
    out = tmp_path / "a.json"
    stats = release.write_release(
        out, [plan], rows, lambda _v: {}, class_names=["fly"], action_names=[]
    )

    assert stats["dropped"] == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [f["frame_index"] for f in data["videos"][0]["frames"]] == [50]


def test_a_video_whose_ledger_vanished_is_countable_afterwards(tmp_path):
    """``iter_rows`` is silent about a missing file, so the count is the only signal.

    A ledger deleted, or left behind when the project folder moved, otherwise
    exports as a video that claims 108,000 frames and holds none - and nothing
    in the summary would say so.
    """
    present = release.VideoPlan(
        video_id="here", video_path="../videos/a.mp4", num_frames=10, fps=30.0,
        frames=10, human=0, model=10, verified=((0, 9),),
    )
    gone = release.VideoPlan(
        video_id="gone", video_path="../videos/b.mp4", num_frames=10, fps=30.0,
        frames=10, human=0, model=10, verified=((0, 9),),
    )
    rows = lambda vid: ([(1, [ledger_box()])] if vid == "here" else [])
    stats = release.write_release(
        tmp_path / "a.json", [present, gone], rows, lambda _v: {},
        class_names=["fly"], action_names=[],
    )

    assert stats["videos"] == {"here": 1, "gone": 0}


def test_the_streamed_file_and_the_dict_agree(tmp_path):
    """Two writers, one format. They must not drift."""
    plan = release.VideoPlan(
        video_id="vid_001", video_path="../videos/a.mp4", num_frames=10, fps=30.0,
        frames=10, human=1, model=9, verified=((0, 4),), width=640, height=480,
    )
    rows = lambda _v: [(1, [ledger_box()]), (7, [ledger_box()])]
    human = {3: [drawn()]}
    out = tmp_path / "a.json"
    release.write_release(
        out, [plan], rows, lambda _v: human, class_names=["fly"], action_names=[]
    )
    streamed = json.loads(out.read_text(encoding="utf-8"))
    built = release.dataset_dict(
        [plan], rows, lambda _v: human, class_names=["fly"], action_names=[]
    )
    assert streamed == json.loads(json.dumps(built))


def test_a_human_frame_past_the_end_cannot_make_the_model_count_negative(tmp_path):
    """``model`` is the remainder, so an unbounded ``human`` count goes below zero."""
    videos = [video("vid_001", num_frames=100, annotations={5: [], 500: [drawn()]})]
    plans, _excluded = release.release_plan(
        videos, {"vid_001": state(scanned=100, verified=((0, 99),))}
    )
    plan = plans[0]
    assert plan.human == 1
    assert plan.model == 99
    assert plan.human + plan.model == plan.frames


def test_a_duplicated_ledger_frame_does_not_poison_the_whole_file(tmp_path):
    """``validate_annotation`` rejects the entire file for one repeated frame_index.

    ``append_rows`` cannot write one - the watermark rule forbids it - but a
    ledger concatenated by hand, or copied from two runs, can hold one, and the
    cost of not checking is the whole release rather than the one frame.
    """
    plan = release.VideoPlan(
        video_id="vid_001", video_path="../videos/a.mp4", num_frames=10, fps=30.0,
        frames=10, human=0, model=10, verified=((0, 9),),
    )
    out = tmp_path / "a.json"
    stats = release.write_release(
        out, [plan],
        lambda _v: [(3, [ledger_box()]), (3, [ledger_box(200.0)])],
        lambda _v: {}, class_names=["fly"], action_names=[],
    )
    indices = [f["frame_index"] for f in json.loads(out.read_text(encoding="utf-8"))["videos"][0]["frames"]]
    assert indices == [3]
    assert stats["dropped"] == 1


def test_the_release_names_the_checkpoint_that_drew_it(tmp_path):
    """Most of this file is model output; a dataset that cannot name its model
    cannot be reproduced, re-scored, or withdrawn if that model has a defect."""
    videos = [video("vid_001")]
    plans, _excluded = release.release_plan(videos, {"vid_001": state()})
    data = release.dataset_dict(
        plans, lambda _v: (), lambda _v: {}, class_names=["fly"], action_names=[]
    )
    assert data["meta"]["weights"] == state()["weights"]
