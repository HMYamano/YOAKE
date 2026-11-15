"""
Tests for :mod:`htrtdetr.analysis.bouts` — per-frame probabilities + tracks -> bouts.

Synthetic predictions only (numpy, no torch, no video), so this runs anywhere.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from htrtdetr.analysis import bouts as B


def _det(tid, probs, x=10.0):
    return {"box": [x, 10.0, x + 20.0, 18.0], "tid": tid, "probs": probs}


def _frames(rows):
    """``{frame: [det, ...]}`` -> the predictions.json shape."""
    out = {}
    for frame, dets in rows.items():
        out[frame] = {
            "boxes": [d["box"] for d in dets],
            "scores": [0.9] * len(dets),
            "track_ids": [d["tid"] for d in dets],
            "action_ids": [int(np.argmax(d["probs"])) for d in dets],
            "action_probs": [list(d["probs"]) for d in dets],
        }
    return out


def _flickery_track(n=90, switch=45, flicker=(20, 60), seed=0):
    """Behavior 0 then 1, with confident switch, and single-frame flickers."""
    rng = np.random.default_rng(seed)
    rows = {}
    for f in range(n):
        p = np.array([0.8, 0.15, 0.05]) if f < switch else np.array([0.1, 0.85, 0.05])
        if f in flicker:
            p = p[[1, 0, 2]] * 0.9 + 0.033       # one frame says the other thing, weakly
        p = np.clip(p + rng.normal(0, 0.01, 3), 0.001, None)
        rows[f] = [_det(0, p / p.sum())]
    return rows


def test_viterbi_removes_flicker_but_keeps_the_real_switch():
    seqs = B.track_sequences(_frames(_flickery_track()))
    seq = seqs[0]
    raw = seq.probs.argmax(axis=1)
    assert raw[20] == 1 and raw[60] == 0          # the flickers are there in the argmax
    labels = B.viterbi(seq.probs, stay=0.98, frames=seq.frames)
    assert labels[:45].tolist() == [0] * 45
    assert labels[45:].tolist() == [1] * 45


def test_min_bout_absorbs_short_runs_into_the_likelier_neighbour():
    frames = np.arange(10)
    labels = np.array([0, 0, 0, 0, 2, 2, 1, 1, 1, 1])
    probs = np.full((10, 3), 0.1)
    probs[4:6, 1] = 0.6                           # the short run looks more like 1 than 0
    out = B.enforce_min_bout(labels, probs, frames, min_frames=3)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    # one run alone is never removed
    assert B.enforce_min_bout(np.array([2, 2]), probs[:2], frames[:2], 5).tolist() == [2, 2]


def test_gaps_cut_a_track_and_bouts_do_not_span_them():
    rows = {f: [_det(3, [0.9, 0.1])] for f in range(0, 20)}
    rows.update({f: [_det(3, [0.9, 0.1])] for f in range(40, 60)})
    seqs = B.track_sequences(_frames(rows))
    parts = B.split_on_gaps(seqs[3], max_gap=5)
    assert [(int(p.frames[0]), int(p.frames[-1])) for p in parts] == [(0, 19), (40, 59)]
    bouts = B.decode_tracks(seqs, fps=10.0, min_bout_s=0.0)[3]
    assert [(b.start, b.end) for b in bouts] == [(0, 19), (40, 59)]


def test_untracked_detections_are_ignored_and_old_files_fall_back_to_argmax():
    frames = _frames({0: [_det(-1, [0.5, 0.5]), _det(1, [0.2, 0.8])]})
    for payload in frames.values():
        del payload["action_probs"]
    seqs = B.track_sequences(frames, num_actions=2)
    assert list(seqs) == [1]
    assert seqs[1].probs.tolist() == [[0.0, 1.0]]


def test_summary_time_budget_bouts_latency_and_transitions():
    seqs = B.track_sequences(_frames(_flickery_track()))
    bouts = B.decode_tracks(seqs, fps=30.0, min_bout_s=0.2)
    s = B.summarize(bouts, seqs, fps=30.0, action_names=["still", "walk", "groom"])
    track = s["tracks"]["0"]
    assert track["time_budget"]["still"] == pytest.approx(0.5)
    assert track["time_budget"]["walk"] == pytest.approx(0.5)
    assert sum(track["time_budget"].values()) == pytest.approx(1.0)
    assert track["bouts"] == {"still": 1, "walk": 1, "groom": 0}
    assert track["latency_s"]["walk"] == pytest.approx(45 / 30)
    assert track["transitions"][0][1] == 1
    assert track["coverage"] == pytest.approx(1.0)
    assert s["overall"]["tracks"] == 1


def test_population_counts_need_no_identity():
    rows = {f: [_det(0, [1.0, 0.0]), _det(1, [0.25, 0.75], x=60.0)] for f in range(30)}
    pop = B.population_timeline(_frames(rows), fps=30.0, bin_s=0.5)
    assert len(pop) == 2
    first = pop[0]
    assert first["animals"] == pytest.approx(2.0)
    assert first["0"] == pytest.approx(1.25) and first["1"] == pytest.approx(0.75)


def test_proximity_reports_contact_between_tracks_seen_together():
    rows = {f: [_det(0, [1.0]), _det(1, [1.0], x=15.0 if f < 10 else 200.0)] for f in range(20)}
    pairs = B.proximity(B.track_sequences(_frames(rows)), fps=10.0)
    assert len(pairs) == 1
    assert pairs[0]["contact_fraction"] == pytest.approx(0.5)
    assert pairs[0]["together_s"] == pytest.approx(2.0)


def test_analyze_predictions_writes_every_file_and_reads_meta(tmp_path):
    payload = _frames(_flickery_track())
    payload = {str(k): v for k, v in payload.items()}
    payload["_meta"] = {"fps": 30.0, "action_names": ["still", "walk", "groom"]}
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = B.analyze_predictions(str(path), str(tmp_path / "out"), bin_s=1.0)
    assert summary["fps"] == 30.0
    assert summary["action_names"] == ["still", "walk", "groom"]
    for name in ("bouts.csv", "summary.json", "population.csv", "proximity.csv"):
        assert (tmp_path / "out" / name).is_file()
    with open(tmp_path / "out" / "bouts.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["action_name"] for r in rows] == ["still", "walk"]
    assert float(rows[1]["start_s"]) == pytest.approx(1.5)
    with open(tmp_path / "out" / "population.csv", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["bin_start_s", "frames", "animals", "still", "walk", "groom"]


def test_the_analysis_cli_mode(tmp_path, capsys):
    from htrtdetr.cli import run_analyze

    payload = {str(k): v for k, v in _frames(_flickery_track()).items()}
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    run_analyze(["mode=bouts", f"predictions={path}", f"output_dir={tmp_path / 'o'}",
                 "fps=30", "action_names=still,walk,groom"])
    out = capsys.readouterr().out
    assert "Bouts: 2 across 1 track(s)" in out
    assert (tmp_path / "o" / "summary.json").is_file()
