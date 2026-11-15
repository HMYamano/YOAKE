"""
Multi-head behaviors: one answer per behavior group (docs/KEYFRAME_BEHAVIOR.md §15).

Default is one behavior per animal. A keyframe project may group its behaviors
- exclusive inside a group, simultaneous across groups - and these tests follow
that choice through every layer that has to honour it:

* the vocabulary and a box's answer (pure, no torch)
* the file format, the project, the keyframe counts
* the dataset's targets, GT matching, the loss, the model's output order
* the validation metric, the CLI's resolution, streaming output
* bout decoding per group

and pin the three bugs the old ``multi_head`` path had (§15.6).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest

from htrtdetr.annotation import behaviors as B
from htrtdetr.annotation import keyframes as K
from htrtdetr.annotation.objects import AnnObject

GROUPS = {"locomotion": ["still", "walk", "run"], "social": ["none", "courtship", "copulation"]}
TEXT = "locomotion: still, walk, run\nsocial: none, courtship, copulation"


def _vocab() -> B.BehaviorVocabulary:
    return B.BehaviorVocabulary.from_groups(GROUPS)


# ══════════════════════════════════════════════════════════════════════════
# The vocabulary
# ══════════════════════════════════════════════════════════════════════════

def test_groups_are_numbered_through_in_order():
    v = _vocab()
    assert v.multi and v.heads == ("locomotion", "social")
    assert v.names == ("still", "walk", "run", "none", "courtship", "copulation")
    assert v.sizes == (3, 3) and v.offsets == (0, 3) and len(v) == 6
    assert [v.group_of(a) for a in (0, 2, 3, 5, 6, -1)] == [0, 0, 1, 1, -1, -1]
    assert v.local_of(4) == 1 and v.flat_id(1, 1) == 4
    assert v.find("social", "courtship") == 4 and v.find("social", "walk") == -1
    assert v.as_groups() == GROUPS


def test_parse_and_print_round_trip_and_one_line_form():
    v = B.parse_vocabulary(TEXT, multi_head=True)
    assert v == _vocab()
    assert B.parse_vocabulary(v.to_text(), multi_head=True) == v
    same = B.parse_vocabulary("locomotion: still, walk, run; social: none, courtship, copulation",
                              multi_head=True)
    assert same == v
    single = B.parse_vocabulary("still, walk,\n groom", multi_head=False)
    assert not single.multi and single.names == ("still", "walk", "groom")
    assert single.to_text() == "still, walk, groom"


def test_a_name_used_in_two_groups_is_told_apart_in_the_flat_list():
    v = B.parse_vocabulary("social: none, courtship\ngrooming: none, groom", multi_head=True)
    assert v.names == ("social:none", "courtship", "grooming:none", "groom")
    assert v.name_of(2) == "none" and v.name_of(2, short=False) == "grooming:none"


@pytest.mark.parametrize("text, words", [
    ("locomotion: still", "at least two"),
    ("a: x, y\na: p, q", "both called"),
    ("still, walk", "group name"),
    (": x, y", "needs a name"),
    ("a: x, x", "twice"),
])
def test_bad_groups_say_what_to_fix(text, words):
    with pytest.raises(ValueError, match=words):
        B.parse_vocabulary(text, multi_head=True)


# ══════════════════════════════════════════════════════════════════════════
# One box's answer
# ══════════════════════════════════════════════════════════════════════════

def test_a_key_answers_its_own_group_and_keeps_the_others():
    v = _vocab()
    box = AnnObject(bbox=[0, 0, 10, 10])
    assert B.labels_of(box, v) == (-1, -1) and B.describe(box, v) == "? + ?"
    assert B.set_behavior(box, v, 1)                     # walk
    assert B.set_behavior(box, v, 4)                     # courtship
    assert (box.action_id, box.action_ids) == (1, [1, 4])
    assert B.describe(box, v) == "walk + courtship"
    assert not B.set_behavior(box, v, 4), "the same answer twice changes nothing"
    assert B.set_behavior(box, v, 2)                     # run replaces walk only
    assert box.action_ids == [2, 4] and box.action_id == 2
    assert not B.set_behavior(box, v, 9), "a number with no behavior changes nothing"
    assert B.group_targets(box, v) == (2, 1)
    assert B.is_complete(box, v)
    assert B.clear_group(box, v, 0) and box.action_ids == [-1, 4] and box.action_id == -1
    assert B.missing_groups(box, v) == (0,) and B.has_behavior(box, v)


def test_a_box_labelled_before_groups_keeps_its_answer():
    v = _vocab()
    old = AnnObject(bbox=[0, 0, 1, 1], action_id=1)       # single-behavior era
    assert B.labels_of(old, v) == (1, -1)
    social_only = {"action_id": 4}
    assert B.labels_of(social_only, v) == (-1, 4)


def test_answers_follow_their_names_when_the_vocabulary_is_edited():
    v = _vocab()
    box = AnnObject(bbox=[0, 0, 1, 1])
    B.set_labels(box, v, [1, 4])                          # walk, courtship
    wider = B.parse_vocabulary(
        "locomotion: still, walk, run, jump\nsocial: none, courtship, copulation", True)
    assert B.remap([box], v, wider) == 1
    assert box.action_ids == [1, 5], "courtship moved from 4 to 5 and the answer with it"
    renamed = B.parse_vocabulary(
        "locomotion: still, walking, run, jump\nsocial: none, courtship, copulation", True)
    B.remap([box], wider, renamed)
    assert box.action_ids == [-1, 5], "a renamed behavior is not guessed at"
    flat = B.BehaviorVocabulary.single(["still", "walk", "courtship"])
    B.set_labels(box, renamed, [renamed.find("locomotion", "still"), 5])
    B.remap([box], renamed, flat)
    assert box.action_id == 0 and box.action_ids == [], "the first group wins in one list"
    regrouped = _vocab()
    B.remap([box], flat, regrouped)
    assert box.action_ids == [0, -1]


def test_dicts_and_dataset_objects_work_too():
    from htrtdetr.data.annotation import ObjectAnno

    v = _vocab()
    row = {"action_id": -1}
    B.set_behavior(row, v, 5)
    assert row == {"action_id": -1, "action_ids": [-1, 5]}
    anno = ObjectAnno(object_id=1, bbox=[0, 0, 1, 1])
    B.set_behavior(anno, v, 0)
    assert anno.action_ids == [0, -1] and anno.action_id == 0


# ══════════════════════════════════════════════════════════════════════════
# Files, projects, counts
# ══════════════════════════════════════════════════════════════════════════

def _multi_dataset(path: Path) -> Path:
    data = {
        "meta": {"labeling": K.labeling_meta(K.ClipSpec(4, 3), B.MULTI_HEAD)},
        "class_names": ["fly"],
        "action_names": list(_vocab().names),
        "action_groups": GROUPS,
        "videos": [{"video_id": "v", "fps": 30.0, "width": 96, "height": 64,
                    "num_frames": 80, "frames": [
                        {"frame_index": 20, "objects": [
                            {"object_id": 1, "bbox": [1, 1, 9, 9], "action_id": 1,
                             "action_ids": [1, 4]},
                            {"object_id": 2, "bbox": [20, 20, 30, 30], "action_id": -1,
                             "action_ids": [-1, 3]},
                        ]},
                        {"frame_index": 40, "objects": [
                            {"object_id": 1, "bbox": [1, 1, 9, 9], "action_id": -1},
                        ]},
                    ]}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_declaration_not_the_groups_field_makes_a_file_multi_head():
    assert B.behavior_mode_of({"meta": {}, "action_groups": GROUPS}) == B.SINGLE
    assert not B.vocabulary_of({"action_names": ["a", "b"], "action_groups": GROUPS}).multi
    declared = {"meta": {"labeling": {"mode": "keyframe", "behaviors": "multi_head"}},
                "action_groups": GROUPS}
    assert B.vocabulary_of(declared) == _vocab()
    with pytest.raises(ValueError, match="action_groups"):
        B.vocabulary_of({"meta": declared["meta"]})
    assert "behaviors" not in K.labeling_meta(K.ClipSpec())


def test_action_ids_survive_a_save_and_single_files_do_not_grow_them(tmp_path):
    from htrtdetr.data.annotation import load_annotation, save_annotation, validate_annotation

    anno = load_annotation(str(_multi_dataset(tmp_path / "a.json")))
    assert anno.videos[0].frames[0].objects[0].action_ids == [1, 4]
    save_annotation(anno, str(tmp_path / "b.json"))
    raw = json.loads((tmp_path / "b.json").read_text("utf-8"))
    rows = raw["videos"][0]["frames"]
    assert rows[0]["objects"][1]["action_ids"] == [-1, 3]
    assert "action_ids" not in rows[1]["objects"][0], "nothing to write, nothing written"
    assert validate_annotation(anno).valid

    anno.videos[0].frames[0].objects[0].action_ids = [4, 1]       # swapped groups
    result = validate_annotation(anno)
    assert not result.valid and any("behavior groups" in e for e in result.errors)


def test_a_multi_head_project_writes_its_groups_and_reopens_with_them(tmp_path):
    from htrtdetr.annotation.project import ProjectSession
    from htrtdetr.annotation.session import VideoSession

    project = ProjectSession(name="p", action_names=list(_vocab().names),
                             action_groups=dict(GROUPS))
    project.settings.update(label_mode="keyframe", behaviors=B.MULTI_HEAD,
                            clip={"length": 4, "stride": 3, "anchor": "end"})
    video = VideoSession(video_id="v", num_frames=50, width=96, height=64)
    box = AnnObject(bbox=[1, 1, 9, 9])
    B.set_labels(box, project.behavior_vocabulary(), [2, 5])
    video.annotations[10] = [box]
    project.videos["v"] = video
    data = project.dataset(tmp_path)
    assert data.meta["labeling"]["behaviors"] == "multi_head"
    assert data.action_groups == GROUPS and data.action_names == list(_vocab().names)
    assert data.videos[0].frames[0].objects[0].action_ids == [2, 5]

    from htrtdetr.data.annotation import save_annotation
    save_annotation(data, str(tmp_path / "annotations.json"))
    again = ProjectSession.from_annotation(tmp_path / "annotations.json")
    assert again.settings["behaviors"] == B.MULTI_HEAD
    assert again.behavior_vocabulary() == _vocab()
    assert again.videos["v"].annotations[10][0].action_ids == [2, 5]

    project.settings["label_mode"] = "tracks"
    assert not project.behavior_vocabulary().multi, "groups are a keyframe-only choice"
    assert "labeling" not in project.dataset(tmp_path).meta


def test_counts_and_sentences_name_the_open_groups(tmp_path):
    data = json.loads(_multi_dataset(tmp_path / "a.json").read_text("utf-8"))
    summary = K.summarize(data, data["action_names"])
    assert summary.instances == 3 and summary.with_behavior == 2 and summary.complete == 1
    assert summary.per_action == (0, 1, 0, 1, 1, 0)
    warnings = K.check(data, data["action_names"])
    assert any("groups unanswered" in w for w in warnings)
    v = _vocab()
    boxes = [AnnObject(bbox=[0, 0, 1, 1]), AnnObject(bbox=[0, 0, 1, 1])]
    B.set_labels(boxes[0], v, [1, 4])
    B.set_labels(boxes[1], v, [1, -1])
    assert K.keyframe_sentence(boxes, v) == "2 animals - 1 still needs a behavior (social)."
    assert K.frame_status(boxes, v) == (2, 1)
    assert K.keyframe_marks({5: boxes}, v) == [(5, 1), (5, 4), (5, 1)]
    assert K.keyframe_sentence(boxes[:1], v) == "1 animal - every one has a behavior."


# ══════════════════════════════════════════════════════════════════════════
# Training: targets, matching, loss, model, validation, CLI
# ══════════════════════════════════════════════════════════════════════════

torch = pytest.importorskip("torch")


def _video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (96, 64))
    for i in range(80):
        writer.write(np.full((64, 96, 3), 40 + i, dtype=np.uint8))
    writer.release()
    return path


def _clip_dataset(tmp_path: Path) -> Path:
    path = _multi_dataset(tmp_path / "annotations.json")
    data = json.loads(path.read_text("utf-8"))
    data["videos"][0]["video_path"] = str(_video(tmp_path))
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_dataset_gives_each_group_its_own_target(tmp_path):
    from htrtdetr.data.keyframe_clip import KeyframeClipDataset, collate_keyframe_clips
    from htrtdetr.training.targets import last_frame_targets

    ds = KeyframeClipDataset(str(_clip_dataset(tmp_path)), image_size=(64, 64), augment=False)
    assert ds.vocab == _vocab()
    sample = ds[0]
    key = sample["head_action_ids"][-1]
    assert key.tolist() == [[1, 1], [-1, 0]], "positions inside each group, -1 = open"
    assert sample["action_ids"][-1].tolist() == [1, -1], "the first group's flat answer"
    assert sample["head_action_ids"][0].shape == (0, 2), "context frames carry no labels"
    batch = collate_keyframe_clips([sample, ds[1]])
    last = last_frame_targets(batch["targets"])
    assert last[1]["head_action_ids"].tolist() == [[-1, -1]]

    only = KeyframeClipDataset(str(_clip_dataset(tmp_path)), image_size=(64, 64),
                               augment=False, require_behavior=True)
    assert len(only) == 1, "the keyframe with no answer in any group is skipped"


def test_matching_copies_group_answers_and_the_old_per_head_keys():
    """§15.6 bug 1: ``action_<head>`` never reached the detections."""
    from htrtdetr.training.matching import assign_gt_to_detections, collect_matched_gt

    target = {
        "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]]),
        "head_action_ids": torch.tensor([[1, 2], [0, -1]]),
        "action_social": torch.tensor([2, 0]),
    }
    preds = torch.tensor([[0.7, 0.7, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.05, 0.9, 0.05, 0.05]])
    out = assign_gt_to_detections(preds, target)
    assert out["matched_head_action_ids"].tolist() == [[0, -1], [1, 2], [-1, -1]]
    assert out["matched_action_social"].tolist() == [0, 2, -1]
    gt = collect_matched_gt([{"boxes": preds}], [target], key="action_social")
    assert gt.tolist() == [0, 2, -1]
    empty = assign_gt_to_detections(preds, {"boxes": torch.zeros(0, 4),
                                            "head_action_ids": torch.zeros(0, 2).long()})
    assert empty["matched_head_action_ids"].shape == (3, 2)


class _Out:
    """The parts of HTRTDETROutput the stage-2 loss reads."""

    def __init__(self, det_boxes, head_logits):
        b, q = 1, 3
        self.pred_logits = torch.zeros(b, q, 2)
        self.pred_boxes = torch.rand(b, q, 4) * 0.2 + 0.2
        self.det_results = [{"boxes": det_boxes}]
        self.action_head_logits = head_logits
        self.action_logits = torch.cat(list(head_logits.values()), dim=-1)
        self.behavior = None
        self.id_logits = None
        self.aux_outputs = None


def test_the_loss_trains_every_group_with_an_answer():
    from htrtdetr.config.config import LossConfig
    from htrtdetr.training.losses import CombinedLoss

    loss = CombinedLoss(LossConfig(), num_actions=6)
    boxes = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]])
    logits = {"locomotion": torch.randn(2, 3, requires_grad=True),
              "social": torch.randn(2, 3, requires_grad=True)}
    target = {"boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]]),
              "class_ids": torch.zeros(2, dtype=torch.long),
              "action_ids": torch.tensor([1, -1]),
              "head_action_ids": torch.tensor([[1, 2], [-1, 0]])}
    out = loss(_Out(boxes, logits), [target], stage=2)
    assert {"loss_action_locomotion", "loss_action_social"} <= set(out)
    assert float(out["loss_action"].detach()) > 0
    out["loss_action"].backward()
    assert logits["social"].grad.abs().sum() > 0
    assert logits["locomotion"].grad[1].abs().sum() == 0, "an open group teaches nothing"


@pytest.fixture(scope="module")
def multi_model():
    from htrtdetr.config.config import build_model_config
    from htrtdetr.models import build_model

    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 8
    cfg.detector.score_threshold = 0.0
    cfg.behavior.use_clip = True
    cfg.behavior.clip_channels = 16
    cfg.behavior.classifier_type = "multi_head"
    cfg.behavior.heads = {k: list(v) for k, v in GROUPS.items()}
    cfg.action_head.num_actions = 6
    m = build_model(cfg)
    m.set_stage(2)
    return m


def test_the_model_returns_group_logits_in_detection_order(multi_model):
    """§15.6 bugs 2 and 3: row order, and the old ActionHead standing in."""
    multi_model.eval()
    with torch.no_grad():
        out = multi_model(torch.randn(2, 4, 3, 64, 64))
    heads = out.action_head_logits
    assert heads is not None and list(heads) == ["locomotion", "social"]
    n = sum(int(d["boxes"].shape[0]) for d in out.det_results)
    assert heads["locomotion"].shape == (n, 3) and heads["social"].shape == (n, 3)
    assert torch.equal(out.action_logits, torch.cat([heads["locomotion"], heads["social"]], -1)), \
        "action_logits is the behavior branch's, not the legacy ActionHead's"
    sums = out.action_probs.split([3, 3], dim=-1)
    for part in sums:
        assert torch.allclose(part.sum(-1), torch.ones(n), atol=1e-5)
    # Row order: the detection-order logits equal the track rows gathered.
    rows = [r for b in out.behavior_rows for r in b]
    raw = out.behavior.head_logits["social"]
    for i, r in enumerate(rows):
        if r >= 0:
            assert torch.allclose(heads["social"][i], raw[r])


def test_streaming_answers_every_group(multi_model):
    import numpy as np

    from htrtdetr.config.config import InferenceConfig
    from htrtdetr.inference import StreamingInferencer

    si = StreamingInferencer(multi_model, InferenceConfig(use_amp=False, use_channels_last=False),
                             action_names=list(_vocab().names), window_size=4,
                             image_size=(64, 64), clip_stride=2)
    rng = np.random.default_rng(0)
    last = None
    for i in range(8):
        last = si.process_frame(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), i)
    payload = last.to_json()
    assert len(payload["head_action_ids"]) == len(payload["boxes"]) > 0
    for row, flat, probs in zip(payload["head_action_ids"], payload["action_ids"],
                                payload["action_probs"]):
        if row[0] < 0:
            continue
        assert 0 <= row[0] < 3 <= row[1] < 6, "one answer inside each group"
        assert flat == row[0]
        assert sum(probs[:3]) == pytest.approx(1.0, abs=1e-3)
        assert sum(probs[3:]) == pytest.approx(1.0, abs=1e-3)
    named = payload["head_predictions"][0]
    assert set(named) == {"locomotion", "social"}


def test_validation_scores_each_group_and_averages_them(tmp_path):
    """macro_f1 is the mean of the groups' macro-F1 - what rounds record."""
    import torch.nn as nn

    from htrtdetr.config.config import LossConfig, TrainConfig
    from htrtdetr.models.ht_rtdetr import HTRTDETROutput
    from htrtdetr.training.trainer import Trainer

    boxes = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]])
    target = {"boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]]),
              "class_ids": torch.zeros(2, dtype=torch.long),
              "action_ids": torch.tensor([1, 0]),
              "head_action_ids": torch.tensor([[1, 2], [0, 2]])}

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))
            self.cfg = type("C", (), {"behavior": type("B", (), {"heads": GROUPS})()})()

        def forward(self, images, **_):
            loco = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]) + self.w   # both right
            social = torch.tensor([[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]]) + self.w  # one right
            return HTRTDETROutput(
                pred_logits=torch.zeros(1, 2, 2) + self.w, pred_boxes=boxes[None],
                query_features=torch.zeros(1, 2, 4),
                action_logits=torch.cat([loco, social], -1),
                det_results=[{"boxes": boxes}],
                action_head_logits={"locomotion": loco, "social": social},
            )

    batch = {"images": torch.zeros(1, 2, 3, 8, 8), "targets": [[{}, target]]}
    cfg = TrainConfig(stage=2, max_epochs=1, output_dir=str(tmp_path), use_amp=False,
                      use_ema=False)
    trainer = Trainer(Fake(), [batch], [batch], cfg, LossConfig(), num_actions=6)
    metrics = trainer._val_epoch(0)
    assert metrics["macro_f1_locomotion"] == pytest.approx(1.0)
    # social: both are copulation, one predicted courtship -> F1 2/3 for the one
    # class with examples (classes nobody labelled do not enter a macro-F1)
    assert metrics["macro_f1_social"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert metrics["macro_f1"] == pytest.approx((1.0 + 2.0 / 3.0) / 2, abs=1e-3)
    assert set(trainer._last_head_evals) == {"locomotion", "social"}


def test_the_cli_builds_the_heads_the_file_declares(tmp_path):
    from htrtdetr.cli import _resolve_label_mode
    from htrtdetr.config.config import get_variant_config

    path = _multi_dataset(tmp_path / "annotations.json")
    cfg = get_variant_config("small", stage=2)
    log = logging.getLogger("test.behavior_groups")
    assert _resolve_label_mode(cfg, path, 2, {}, log) == "keyframe"
    assert cfg.model.behavior.classifier_type == "multi_head"
    assert cfg.model.behavior.heads == GROUPS
    assert cfg.model.action_head.num_actions == 6
    assert cfg.eval.action_names == list(_vocab().names)

    with pytest.raises(SystemExit) as stop:
        _resolve_label_mode(get_variant_config("small", stage=2), path, 2,
                            {"model": {"behavior": {"classifier_type": "single"}}}, log)
    assert stop.value.code == 2

    single = json.loads(path.read_text("utf-8"))
    single["meta"]["labeling"].pop("behaviors")
    (tmp_path / "single.json").write_text(json.dumps(single), encoding="utf-8")
    cfg = get_variant_config("small", stage=2)
    cfg.model.behavior.classifier_type = "multi_head"
    cfg.model.behavior.heads = dict(GROUPS)
    _resolve_label_mode(cfg, tmp_path / "single.json", 2, {}, log)
    assert cfg.model.behavior.classifier_type == "single", "the labels decide"


# ══════════════════════════════════════════════════════════════════════════
# Bouts, per group
# ══════════════════════════════════════════════════════════════════════════

def _predictions(path: Path) -> Path:
    """One fly: still then walking, courting the whole time."""
    rows = {"_meta": {"fps": 30.0, "action_names": list(_vocab().names),
                      "action_groups": GROUPS}}
    for f in range(60):
        loco = [0.9, 0.05, 0.05] if f < 30 else [0.05, 0.9, 0.05]
        rows[str(f)] = {"boxes": [[10.0, 10.0, 20.0, 20.0]], "track_ids": [3],
                        "action_ids": [0 if f < 30 else 1],
                        "head_action_ids": [[0 if f < 30 else 1, 4]],
                        "action_probs": [loco + [0.1, 0.8, 0.1]]}
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_each_group_decodes_on_its_own(tmp_path):
    from htrtdetr.analysis.bouts import analyze_predictions

    summary = analyze_predictions(str(_predictions(tmp_path / "p.json")), str(tmp_path / "out"),
                                  min_bout_s=0.1)
    assert set(summary["groups"]) == {"locomotion", "social"}
    loco = summary["groups"]["locomotion"]["tracks"]["3"]
    assert loco["time_budget"]["still"] == pytest.approx(0.5)
    assert loco["time_budget"]["walk"] == pytest.approx(0.5)
    social = summary["groups"]["social"]["tracks"]["3"]
    assert social["time_budget"]["courtship"] == pytest.approx(1.0), \
        "courting through the walk: groups do not compete"
    assert summary["overall"]["bouts"] == {"still": 1, "walk": 1, "run": 0,
                                           "none": 0, "courtship": 1, "copulation": 0}
    with open(tmp_path / "out" / "bouts.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["group"], r["action_name"]) for r in rows] == [
        ("locomotion", "still"), ("social", "courtship"), ("locomotion", "walk")]
    with open(tmp_path / "out" / "population.csv", encoding="utf-8") as handle:
        pop = list(csv.DictReader(handle))
    first = pop[0]
    total_loco = sum(float(first[n]) for n in ("still", "walk", "run"))
    total_social = sum(float(first[n]) for n in ("none", "courtship", "copulation"))
    assert total_loco == pytest.approx(1.0) and total_social == pytest.approx(1.0), \
        "one animal counts once in every group"


def test_without_probabilities_the_group_answers_are_used(tmp_path):
    from htrtdetr.analysis.bouts import track_sequences

    frames = {f: {"boxes": [[0, 0, 4, 4]], "track_ids": [1], "head_action_ids": [[2, 5]]}
              for f in range(4)}
    seq = track_sequences(frames, 6, head_sizes=(3, 3))[1]
    assert seq.probs[0].tolist() == [0, 0, 1, 0, 0, 1]
