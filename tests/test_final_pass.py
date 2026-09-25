"""The last pass over 2.16M frames, and the four ways it fails quietly.

This is the end of the project: the model has stopped learning, it is run over
every frame of twenty one-hour videos, and what a human signs off here becomes
the dataset the study is made of. Nothing in that sentence raises an exception
when it goes wrong, which is what this file is for:

- **a flagged place merged away.** The queue is shortened so a review is 25
  minutes rather than 8 years, and the shortening runs over the output of the
  quality checks. A merge rule that can drop a two-frame ID switch produces a
  shorter queue, a faster review and a signed-off video with a swapped
  identity in it. So: the tiling covers ``[0, num_frames)`` exactly, and every
  frame the checks flagged is still inside a flagged segment afterwards.

- **a model that is confidently wrong.** Every other signal in the module comes
  from the model's own doubt. An animal it never detects produces no doubt at
  all, so the spot checks are the only thing standing between that and a
  sign-off. They must land in the *clean* spans, spread through the video.

- **an interrupted pass.** The scan is two hours of GPU time and will be
  stopped in the middle. A resume that re-appends the overlap must not
  duplicate a frame, the watermark must not go backwards, and a half-written
  final line must cost the frame in flight rather than the ledger.

- **a window read that is not a window.** ``read_window`` runs once per stop,
  134 times a video. If it scans the file instead of seeking into it, nothing
  breaks and nothing is wrong - the dock is just slow enough that nobody
  finishes the review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import finalpass as fp  # noqa: E402
from htrtdetr.annotation import quality  # noqa: E402
from htrtdetr.gui import auto_label  # noqa: E402


HOUR_30FPS = 108_000


def box(x=10, y=10, *, tid=0, score=0.9, cls=0):
    return auto_label.Candidate([x, y, x + 20, y + 20], score=score, track_id=tid, class_id=cls)


def two_animals(frames, *, score=0.9):
    """A boring video: two tracked animals, every frame, nothing happening."""
    return {
        f: [box(10 + f, 10, tid=1, score=score), box(200 - f, 50, tid=2, score=score)]
        for f in range(frames)
    }


def ledger(tmp_path, name="cam1", *, num_frames=1000, fps=30.0):
    path = tmp_path / f"{name}.jsonl"
    fp.write_header(path, video=name, weights="cam1.mp4:1:2", num_frames=num_frames, fps=fps)
    return path


# ══════════════════════════════════════════════════════════════════════════
# The ledger
# ══════════════════════════════════════════════════════════════════════════

def test_the_header_names_the_checkpoint_that_produced_the_file(tmp_path):
    """Without it, two models' boxes can be appended into one file unnoticed."""
    path = ledger(tmp_path, num_frames=HOUR_30FPS)
    header = fp.read_header(path)
    assert header["video"] == "cam1"
    assert header["weights"] == "cam1.mp4:1:2"
    assert header["num_frames"] == HOUR_30FPS
    assert header["fps"] == 30.0


def test_a_missing_ledger_reads_as_an_empty_header_rather_than_raising(tmp_path):
    """Every video asks this once, and on a fresh project none of them has one."""
    assert fp.read_header(tmp_path / "nothing.jsonl") == {}
    assert fp.frames_written(tmp_path / "nothing.jsonl") == 0
    assert fp.last_frame(tmp_path / "nothing.jsonl") == -1
    assert fp.read_window(tmp_path / "nothing.jsonl", 0, 100) == {}


def test_a_frame_with_no_boxes_is_absent_from_the_file(tmp_path):
    """The absence is the size decision: writing empty rows triples the file."""
    path = ledger(tmp_path)
    written = fp.append_rows(path, {0: [box()], 1: [], 2: [], 3: [box()]})
    assert written == 2
    assert fp.frames_written(path) == 2
    assert sorted(fp.read_window(path, 0, 10)) == [0, 3]


def test_append_and_read_window_round_trip_the_boxes(tmp_path):
    path = ledger(tmp_path)
    rows = {f: [box(f, 2 * f, tid=f % 3, score=0.5 + f / 2000.0)] for f in range(500)}
    fp.append_rows(path, rows)

    window = fp.read_window(path, 100, 109)
    assert sorted(window) == list(range(100, 110))
    x1, y1, x2, y2, class_id, score, track_id = window[104][0]
    assert (x1, y1, x2, y2) == (104, 208, 124, 228)
    assert class_id == 0
    assert track_id == 104 % 3
    # Rounded to three places on purpose; see _SCORE_DIGITS.
    assert score == pytest.approx(0.552, abs=5e-4)


def test_read_window_is_inclusive_at_both_ends(tmp_path):
    """The convention the whole package uses; an exclusive end here would drop
    the last frame of every span a human signs off."""
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(100)})
    assert sorted(fp.read_window(path, 10, 12)) == [10, 11, 12]
    assert fp.read_window(path, 7, 7) == {7: fp.read_window(path, 7, 7)[7]}
    assert fp.read_window(path, 12, 10) == {}


def test_a_truncated_last_line_is_dropped_rather_than_raised_on(tmp_path):
    """The machine was turned off during a two-hour pass.

    The cost of that must be the one frame that was in flight, not the 90,000
    frames in front of it.
    """
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(200)})
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 12])  # cut the last row in half

    assert sorted(fp.read_window(path, 190, 199)) == list(range(190, 199))
    assert len(list(fp.iter_rows(path))) == 199
    # frames_written notices the file changed under the index and recounts,
    # rather than reporting the 200 the index cached.
    assert fp.frames_written(path) == 199


def test_iter_rows_walks_the_whole_ledger_in_order(tmp_path):
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(0, 300, 3)})
    frames = [f for f, _ in fp.iter_rows(path)]
    assert frames == list(range(0, 300, 3))
    assert frames == sorted(frames)


# ── resuming an interrupted pass ─────────────────────────────────────────

def test_the_watermark_only_ever_grows(tmp_path):
    """A resume that replays the overlap must not rewind or duplicate."""
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(100)})
    assert fp.last_frame(path) == 99

    replayed = fp.append_rows(path, {f: [box(f + 1000)] for f in range(90, 120)})
    assert replayed == 20  # only 100..119 were new
    assert fp.last_frame(path) == 119
    assert fp.frames_written(path) == 120
    # And the replayed frames kept what was already written for them.
    assert fp.read_window(path, 95, 95)[95][0][0] == 95


def test_a_pass_that_ends_in_empty_frames_does_not_rescan_them(tmp_path):
    """The watermark counts frames the pass *reached*, not frames it wrote."""
    path = ledger(tmp_path)
    fp.append_rows(path, {0: [box()], 1: [], 2: [], 3: []})
    assert fp.last_frame(path) == 3
    assert fp.frames_written(path) == 1


def test_extending_a_ledger_twice_gives_the_same_file_as_writing_it_once(tmp_path):
    rows = {f: [box(f, tid=f % 2)] for f in range(0, 2500) if f % 7}
    # The same name in two folders, so the headers are byte-identical and the
    # comparison below is about the rows and the offsets into them.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    once = ledger(tmp_path / "a")
    fp.append_rows(once, rows)

    twice = ledger(tmp_path / "b")
    fp.append_rows(twice, {f: b for f, b in rows.items() if f < 1200})
    fp.append_rows(twice, {f: b for f, b in rows.items() if f >= 1200})

    assert twice.read_bytes() == once.read_bytes()
    assert json.loads(fp._idx_path(twice).read_text()) == json.loads(
        fp._idx_path(once).read_text()
    )
    assert list(fp.iter_rows(twice)) == list(fp.iter_rows(once))


def test_appending_the_same_rows_again_is_a_no_op(tmp_path):
    """So a caller that cannot tell whether its last write landed can repeat it."""
    path = ledger(tmp_path)
    rows = {f: [box(f)] for f in range(400)}
    fp.append_rows(path, rows)
    before = path.read_bytes()
    assert fp.append_rows(path, rows) == 0
    assert path.read_bytes() == before


def test_rows_may_arrive_as_pairs_or_out_of_order(tmp_path):
    """The scan yields batches; the caller should not have to sort them."""
    path = ledger(tmp_path)
    fp.append_rows(path, [(3, [box(3)]), (1, [box(1)]), (2, [box(2)])])
    assert [f for f, _ in fp.iter_rows(path)] == [1, 2, 3]


# ── the window is a window ───────────────────────────────────────────────

class _CountingFile:
    """A binary file that remembers how many bytes were actually read."""

    def __init__(self, handle, counter):
        self._handle = handle
        self._counter = counter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __iter__(self):
        for line in self._handle:
            self._counter[0] += len(line)
            yield line

    def seek(self, *a):
        return self._handle.seek(*a)

    def readline(self):
        data = self._handle.readline()
        self._counter[0] += len(data)
        return data

    def read(self, *a):
        data = self._handle.read(*a)
        self._counter[0] += len(data)
        return data


def test_read_window_seeks_instead_of_reading_the_whole_file(tmp_path, monkeypatch):
    """134 stops a video; a scan per stop is what makes the dock unusable.

    Counted in bytes rather than timed, because a timing assertion on a 2 MB
    file passes on any machine and would not have caught the scan.
    """
    path = ledger(tmp_path, num_frames=30_000)
    fp.append_rows(path, two_animals(30_000))
    size = path.stat().st_size
    assert size > 1_000_000  # a real file, not a toy one

    counter = [0]
    real_open = open
    monkeypatch.setattr(
        fp, "open", lambda *a, **kw: _CountingFile(real_open(*a, **kw), counter), raising=False
    )

    window = fp.read_window(path, 29_000, 29_010)
    assert sorted(window) == list(range(29_000, 29_011))
    # At most the one INDEX_STRIDE block the window starts in.
    assert counter[0] < size / 10.0


def test_the_index_offset_points_exactly_at_the_row_it_names(tmp_path):
    """The seek is only safe if the offsets are byte positions.

    On Windows a ledger written in text mode gets CRLF line endings and
    ``tell()`` returns an opaque cookie; either one lands the seek in the
    middle of a line, and only on the machine that wrote the file.
    """
    path = ledger(tmp_path, num_frames=5000)
    fp.append_rows(path, two_animals(5000))
    index = json.loads(fp._idx_path(path).read_text())

    assert index["stride"] == fp.INDEX_STRIDE
    assert [f for f, _ in index["offsets"]] == [0, 1000, 2000, 3000, 4000]
    with open(path, "rb") as fh:
        for frame, offset in index["offsets"]:
            fh.seek(offset)
            assert json.loads(fh.readline())["f"] == frame


def test_read_window_still_works_without_an_index(tmp_path):
    """The index is derived data. Losing it must cost speed, not the pass."""
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(300)})
    fp._idx_path(path).unlink()
    assert sorted(fp.read_window(path, 100, 104)) == [100, 101, 102, 103, 104]
    assert fp.frames_written(path) == 300


# ══════════════════════════════════════════════════════════════════════════
# The stops
# ══════════════════════════════════════════════════════════════════════════

def assert_tiles(segs, num_frames):
    """The invariant everything downstream depends on."""
    assert segs, "a video is always at least one segment"
    assert segs[0].start == 0
    assert segs[-1].end == num_frames - 1
    for previous, nxt in zip(segs, segs[1:]):
        assert nxt.start == previous.end + 1, "a gap or an overlap"
    assert sum(s.span for s in segs) == num_frames


def test_segments_tile_the_video_with_no_gap_and_no_overlap():
    """A frame in no segment is reviewed by nothing and sampled by nothing."""
    rows = two_animals(4000)
    rows[1500] = [box(10, tid=1)]              # an animal vanishes
    rows[2200] = [box(10, tid=1, score=0.1)]   # and comes back unsure
    issues = [
        quality.ReviewItem(kind=quality.ID_SWITCH_RISK, start=800, end=812, track_ids=(1, 2)),
        quality.ReviewItem(kind=quality.POSITION_JUMP, start=3999, end=3999, track_ids=(1,)),
    ]
    segs = fp.segments(rows, issues, num_frames=4000, expected=2, min_score=0.3)
    assert_tiles(segs, 4000)
    assert any(s.flagged for s in segs)
    assert any(not s.flagged for s in segs)


def test_an_empty_video_tiles_as_one_clean_segment():
    segs = fp.segments({}, [], num_frames=600)
    assert_tiles(segs, 600)
    assert [s.kind for s in segs] == ["clean"]


def test_a_video_shorter_than_the_minimum_segment_is_still_covered():
    segs = fp.segments({}, [quality.ReviewItem(kind=quality.OVERLAP, start=4, end=5)],
                       num_frames=10)
    assert_tiles(segs, 10)


def test_no_frames_at_all_produces_no_segments():
    assert fp.segments({}, [], num_frames=0) == []


def test_a_short_flagged_interval_is_never_swallowed_by_a_merge():
    """The failure this whole file exists for.

    A two-frame ID-switch risk in an otherwise quiet video is exactly the
    length that a "merge anything short into its neighbour" rule deletes, and
    deleting it produces a shorter queue, a faster review, and a dataset with a
    swapped identity in it.
    """
    item = quality.ReviewItem(kind=quality.ID_SWITCH_RISK, start=2000, end=2001, track_ids=(1, 2))
    segs = fp.segments(two_animals(5000), [item], num_frames=5000)
    assert_tiles(segs, 5000)

    flagged = [s for s in segs if s.flagged]
    assert len(flagged) == 1
    assert flagged[0].start <= 2000 and flagged[0].end >= 2001
    # Grown to a length a human can judge, not to half the video.
    assert flagged[0].span >= fp.MIN_SEGMENT
    assert flagged[0].span < 200
    assert "ID switch" in flagged[0].reason


def test_every_flagged_frame_stays_inside_a_flagged_segment():
    """Said as the general property, over a spread of awkward lengths."""
    items = [
        quality.ReviewItem(kind=quality.POSITION_JUMP, start=0, end=0),
        quality.ReviewItem(kind=quality.SIZE_JUMP, start=17, end=18),
        quality.ReviewItem(kind=quality.OVERLAP, start=900, end=1400),
        quality.ReviewItem(kind=quality.LOW_CONFIDENCE, start=2999, end=2999),
    ]
    segs = fp.segments(two_animals(3000), items, num_frames=3000)
    assert_tiles(segs, 3000)
    covered = {f for s in segs if s.flagged for f in range(s.start, s.end + 1)}
    for item in items:
        assert set(range(item.start, item.end + 1)) <= covered


def test_a_change_in_the_box_count_is_flagged_without_any_review_item():
    """The cheapest signal there is, and it fires where no per-frame check does."""
    rows = two_animals(2000)
    for f in range(1000, 1004):
        rows[f] = [box(10, tid=1)]
    segs = fp.segments(rows, [], num_frames=2000)
    assert_tiles(segs, 2000)
    flagged = [s for s in segs if s.flagged]
    assert flagged and any(s.start <= 1000 <= s.end for s in flagged)
    assert "number of boxes" in flagged[0].reason


def test_a_swapped_track_set_at_the_same_count_is_flagged():
    """The error that quietly ruins a trajectory dataset: same two boxes, new ids."""
    rows = two_animals(1000)
    for f in range(500, 1000):
        rows[f] = [box(10 + f, 10, tid=7), box(200 - f, 50, tid=8)]
    segs = fp.segments(rows, [], num_frames=1000)
    assert_tiles(segs, 1000)
    assert any(s.flagged and s.start <= 500 <= s.end for s in segs)


def test_a_missing_frame_counts_as_a_frame_with_no_boxes():
    """A frame the detector wrote nothing for is the loudest count change there
    is, and it is invisible to anything that iterates the ledger's keys."""
    rows = two_animals(800)
    for f in range(400, 420):
        del rows[f]
    segs = fp.segments(rows, [], num_frames=800)
    assert_tiles(segs, 800)
    assert any(s.flagged and s.start <= 400 <= s.end for s in segs)


def test_low_scores_and_the_expected_animal_count_flag_on_their_own():
    weak = two_animals(600, score=0.2)
    segs = fp.segments(weak, [], num_frames=600, min_score=0.5)
    assert [s.kind for s in segs] == ["flagged"]
    assert "0.50" in segs[0].reason

    one = {f: [box(10 + f, tid=1)] for f in range(600)}
    segs = fp.segments(one, [], num_frames=600, expected=2)
    assert [s.kind for s in segs] == ["flagged"]
    assert "not 2 animals" in segs[0].reason


def test_a_clean_sliver_between_two_flagged_spans_is_absorbed():
    """Sending someone back for a fifteen-frame gap buys nothing."""
    items = [
        quality.ReviewItem(kind=quality.OVERLAP, start=1000, end=1099),
        quality.ReviewItem(kind=quality.OVERLAP, start=1110, end=1200),
    ]
    segs = fp.segments(two_animals(3000), items, num_frames=3000)
    assert_tiles(segs, 3000)
    assert len([s for s in segs if s.flagged]) == 1


# ── spot checks ──────────────────────────────────────────────────────────

def test_spot_checks_land_only_inside_clean_segments():
    items = [quality.ReviewItem(kind=quality.OVERLAP, start=a, end=a + 40)
             for a in range(500, 9000, 1500)]
    segs = fp.segments(two_animals(10_000), items, num_frames=10_000)
    clean = {f for s in segs if not s.flagged for f in range(s.start, s.end + 1)}
    spots = fp.spot_check_frames(segs)
    assert spots
    assert set(spots) <= clean
    assert list(spots) == sorted(set(spots))


def test_a_confidently_wrong_model_is_caught_only_by_the_spot_checks():
    """The justification for the function, stated as a test.

    A detector that has decided the second fly is background is perfectly
    self-consistent: one box, one track id, high confidence, for all 108,000
    frames. It produces no low score, no count change and no broken track, so
    the quality checks produce nothing and the flagged queue is empty. Without
    spot checks the human is taken to zero places and signs off a video missing
    half its data in under a minute.
    """
    rows = {f: [box(10 + f % 300, tid=1, score=0.97)] for f in range(HOUR_30FPS)}
    segs = fp.segments(rows, [], num_frames=HOUR_30FPS)
    assert [s.kind for s in segs] == ["clean"], "nothing objected - that is the point"

    spots = fp.spot_check_frames(segs)
    assert len(spots) == fp.SPOT_CHECKS
    assert fp.stops(segs, spots) == spots  # every stop is a spot check
    # Spread through the whole video, not clustered at one end.
    assert spots[0] < HOUR_30FPS * 0.05
    assert spots[-1] > HOUR_30FPS * 0.95


def test_spot_checks_are_spaced_by_clean_frames_not_by_clean_segments():
    """A 40,000-frame clean stretch is where a confident error hides; it must
    not get the same single look as a 30-frame one."""
    items = [quality.ReviewItem(kind=quality.OVERLAP, start=a, end=a + 20)
             for a in range(100, 3000, 100)]
    segs = fp.segments(two_animals(50_000), items, num_frames=50_000)
    spots = fp.spot_check_frames(segs, n=20)
    assert sum(1 for f in spots if f > 10_000) >= 15


def test_a_video_with_nothing_clean_has_no_spot_checks():
    segs = fp.segments(two_animals(400, score=0.1), [], num_frames=400, min_score=0.5)
    assert [s.kind for s in segs] == ["flagged"]
    assert fp.spot_check_frames(segs) == ()


def test_the_stops_are_the_flagged_starts_plus_the_spot_checks():
    """The start of a flagged span, not its middle: the cause is at the front."""
    items = [quality.ReviewItem(kind=quality.OVERLAP, start=a, end=a + 60)
             for a in (1000, 4000, 7000)]
    segs = fp.segments(two_animals(10_000), items, num_frames=10_000)
    spots = fp.spot_check_frames(segs, n=10)
    got = fp.stops(segs, spots)

    assert list(got) == sorted(set(got))
    assert {s.start for s in segs if s.flagged} <= set(got)
    assert set(spots) <= set(got)
    assert len(got) == len([s for s in segs if s.flagged]) + len(spots)


# ══════════════════════════════════════════════════════════════════════════
# The state
# ══════════════════════════════════════════════════════════════════════════

def test_the_state_round_trips_through_the_manifest():
    """It goes into project.json as JSON, so tuples come back as lists."""
    state = fp.PassState(weights="w:1:2", scanned=108_000,
                         verified=((0, 4200), (9000, 9600)), signed_off=1748999999.0)
    back = fp.PassState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert back == state


def test_a_malformed_state_degrades_instead_of_refusing_to_open_the_project():
    """A manifest that will not open is worse than a pass that must be re-run."""
    assert fp.PassState.from_dict(None) == fp.PassState()
    assert fp.PassState.from_dict({"scanned": "many", "verified": [[5], "x", [9, 4]],
                                   "signed_off": "yesterday"}) == fp.PassState()


def test_verified_spans_stay_sorted_merged_and_non_overlapping():
    state = fp.PassState(scanned=1000)
    state = fp.mark_verified(state, 500, 599)
    state = fp.mark_verified(state, 0, 99)
    state = fp.mark_verified(state, 50, 200)
    assert state.verified == ((0, 200), (500, 599))
    # Touching spans join: with an inclusive end they are one stretch, and
    # leaving them apart makes every count of "how much is done" wrong.
    state = fp.mark_verified(state, 201, 499)
    assert state.verified == ((0, 599),)
    assert state.verified_frames == 600


def test_verifying_the_same_span_twice_changes_nothing():
    """An eight-hour review across twenty sittings must not depend on order."""
    once = fp.mark_verified(fp.PassState(scanned=10), 100, 200)
    assert fp.mark_verified(once, 100, 200) == once
    assert fp.mark_verified(once, 200, 100) == once


def test_covers_and_visited_count_the_stops_inside_verified_spans():
    state = fp.mark_verified(fp.PassState(scanned=1000), 100, 199)
    assert fp.covers(state, 100) and fp.covers(state, 199)
    assert not fp.covers(state, 99) and not fp.covers(state, 200)
    assert fp.visited(state, (50, 100, 150, 199, 500)) == 3


def test_can_sign_off_refuses_with_a_countable_reason():
    """"Not ready" is a refusal a user cannot plan an afternoon around."""
    stops = tuple(range(0, 1340, 10))  # 134 places
    state = fp.PassState(weights="w", scanned=108_000)

    ready, why = fp.can_sign_off(state, stops)
    assert not ready
    assert "134 of 134" in why

    for start in range(0, 940, 10):
        state = fp.mark_verified(state, start, start + 9)
    ready, why = fp.can_sign_off(state, stops)
    assert not ready
    assert why == "40 of 134 places still to check."

    state = fp.mark_verified(state, 0, 1400)
    assert fp.can_sign_off(state, stops) == (True, "")


def test_a_pass_that_has_not_run_cannot_be_signed_off():
    ready, why = fp.can_sign_off(fp.PassState(), (1, 2, 3))
    assert not ready
    assert "has not run" in why


def test_reopening_a_signed_off_video_keeps_the_work_already_done():
    """Clearing the spans would price a second look at a fresh 25 minutes."""
    state = fp.mark_verified(fp.PassState(scanned=100), 0, 99)
    signed = fp.sign_off(state, 1748999999.0)
    assert signed.signed_off == 1748999999.0
    reopened = fp.unsign(signed)
    assert reopened.signed_off is None
    assert reopened.verified == state.verified


def test_the_progress_sentence_counts_places_not_frames():
    """A percentage hides whether the last 13% is two stops or two hundred."""
    stops = tuple(range(0, 1340, 10))
    state = fp.PassState(weights="w", scanned=HOUR_30FPS)
    for start in range(0, 940, 10):
        state = fp.mark_verified(state, start, start + 9)
    assert (
        fp.progress_sentence(state, stops, num_frames=HOUR_30FPS, fps=30.0)
        == "94 of 134 places checked in this video."
    )

    fresh = fp.progress_sentence(fp.PassState(), stops, num_frames=HOUR_30FPS, fps=30.0)
    assert "has not run" in fresh and "108,000 frames" in fresh

    done = fp.mark_verified(state, 0, HOUR_30FPS - 1)
    assert "Ready to sign off" in fp.progress_sentence(
        done, stops, num_frames=HOUR_30FPS, fps=30.0
    )
    assert "Signed off" in fp.progress_sentence(
        fp.sign_off(done, 1.0), stops, num_frames=HOUR_30FPS, fps=30.0
    )


# ══════════════════════════════════════════════════════════════════════════
# End to end: one video's worth of the real shape
# ══════════════════════════════════════════════════════════════════════════

def test_one_video_is_about_a_hundred_stops_and_covers_every_frame(tmp_path):
    """The claim the whole design rests on: 25 minutes a video, not 8 years."""
    num_frames = 20_000
    rows = two_animals(num_frames)
    for start in range(1000, num_frames, 1200):          # something happens
        for f in range(start, start + 15):
            rows[f] = [box(10 + f, 10, tid=1, score=0.2)]

    path = ledger(tmp_path, num_frames=num_frames)
    assert fp.append_rows(path, rows) == num_frames

    window = fp.read_window(path, 0, num_frames - 1)
    issues = quality.analyze(
        {f: [_as_candidate(b) for b in bs] for f, bs in window.items()},
        frames=range(num_frames),
        num_frames=num_frames,
        config=quality.QualityConfig(expected_animals=2),
    )
    segs = fp.segments(window, issues, num_frames=num_frames, expected=2, min_score=0.45)
    assert_tiles(segs, num_frames)

    places = fp.stops(segs, fp.spot_check_frames(segs))
    assert 20 <= len(places) <= 200, len(places)

    state = fp.PassState(weights="w", scanned=num_frames)
    assert not fp.can_sign_off(state, places)[0]
    state = fp.mark_verified(state, 0, num_frames - 1)
    assert fp.can_sign_off(state, places) == (True, "")


def _as_candidate(row):
    x1, y1, x2, y2, class_id, score, track_id = row
    return auto_label.Candidate([x1, y1, x2, y2], score=score, track_id=track_id,
                                class_id=class_id)


# ══════════════════════════════════════════════════════════════════════════
# Regressions: three bugs an adversarial read found after the module was green
# ══════════════════════════════════════════════════════════════════════════

def test_a_box_with_no_score_is_certain_not_worthless(tmp_path):
    """``AnnObject.score`` is ``None``, and ``None`` must not read as 0.0.

    The expression was ``float(getattr(box, "score", 1.0) or 0.0)``. The
    default never fired - the attribute exists on a hand-drawn box and holds
    ``None`` - so the ``or`` did, and every box a human drew went into the
    ledger as the least confident thing in the file. ``segments`` then flagged
    exactly the frames the human had already finished.
    """
    class Drawn:
        bbox = [1.0, 1.0, 9.0, 9.0]
        class_id = 0
        track_id = 1
        score = None

    path = ledger(tmp_path)
    fp.append_rows(path, {f: [Drawn()] for f in range(200)})

    assert {row[5] for _f, rows in fp.iter_rows(path) for row in rows} == {1.0}
    segs = fp.segments(dict(fp.iter_rows(path)), [], num_frames=200, min_score=0.45)
    assert not [s for s in segs if s.flagged]


def test_resuming_after_an_interrupted_write_does_not_fuse_two_rows(tmp_path):
    """A killed pass leaves half a line; appending onto it destroys a frame.

    ``open(path, "ab")`` starts at the end of the fragment, so the fragment and
    the first new row become one line that parses as neither. The frame in
    flight is lost - which is survivable - and ``frames_written`` and
    ``iter_rows`` disagree for the rest of the file's life, which is not: the
    ledger is the thing a human is about to sign off.
    """
    path = ledger(tmp_path)
    fp.append_rows(path, {f: [box(f)] for f in range(200)})
    raw = path.read_bytes()
    path.write_bytes(raw[:-12])                      # power cut, mid-row

    resume = fp.last_frame(path) + 1
    fp.append_rows(path, {f: [box(f)] for f in range(resume, 260)})

    frames = [f for f, _boxes in fp.iter_rows(path)]
    assert frames == sorted(set(frames)), "a row was written out of order or twice"
    assert set(frames) == set(range(260)), "a frame was lost across the resume"
    assert fp.frames_written(path) == len(frames)


def test_a_flicker_heavy_video_still_ends_in_a_finishable_list(tmp_path):
    """Sign-off is gated on every stop, so an unbounded list is an unfinishable video.

    One of two flies briefly lost every couple of seconds is routine when they
    climb over each other, and it produces ~1,800 flagged segments in an hour.
    At ten seconds a stop that is five hours for one video of twenty.
    """
    total = 108_000
    both = [[0, 0, 10, 10, 0, 0.9, 1], [20, 20, 30, 30, 0, 0.9, 2]]
    rows = {f: (both[:1] if f % 60 == 0 else both) for f in range(total)}

    segs = fp.segments(rows, [], num_frames=total, expected=2)
    spots = fp.spot_check_frames(segs)
    walk = fp.stops(segs, spots)

    assert len([s for s in segs if s.flagged]) > 1000, "the repro did not reproduce"
    assert len(walk) <= fp.STOP_BUDGET
    # Every spot check survives the cut: they are the only guard against a
    # model that is stably and confidently wrong, and a flickering model is
    # not evidence that it never misses anything.
    assert set(spots) <= set(walk)
    assert "not in this list" in fp.stop_budget_note(segs, spots)


def test_a_budget_that_drops_nothing_says_nothing(tmp_path):
    segs = fp.segments({0: [box(0)]}, [], num_frames=600)
    spots = fp.spot_check_frames(segs)
    assert fp.stop_budget_note(segs, spots) == ""


def test_unsorted_spans_cannot_reach_the_span_walker(tmp_path):
    """``covers`` stops at the first span past the frame, so order is an invariant.

    ``from_dict`` and ``mark_verified`` both normalised; direct construction did
    not, and a frozen dataclass has exactly one place where an invariant can be
    made true for every path.
    """
    state = fp.PassState(scanned=1000, verified=((500, 599), (0, 99)))
    assert state.verified == ((0, 99), (500, 599))
    assert fp.covers(state, 50)
    assert fp.covers(state, 550)
    assert not fp.covers(state, 300)


def test_an_untracked_box_does_not_lose_the_whole_append(tmp_path):
    """``track_id`` is the one field whose ``None`` is routine, and it raised.

    A project that does not follow individuals draws boxes with no track, and
    ``int(None)`` raises from inside the loop that writes a whole chunk - so one
    such box costs every frame in that append, during a two-hour pass.
    """
    class Untracked:
        bbox = [1.0, 1.0, 9.0, 9.0]
        class_id = 0
        track_id = None
        score = None

    path = ledger(tmp_path)
    assert fp.append_rows(path, {f: [Untracked()] for f in range(5)}) == 5
    assert {row[6] for _f, rows in fp.iter_rows(path) for row in rows} == {-1}
