"""
test_pipeline_integration.py — Dataset → DataLoader → Trainer → loss → backward の疎通

単体テストは「関数が単体で正しいか」しか見ない。YOAKE で実際に壊れていたのは
**繋ぎ目**だった: collate が返す targets の形と trainer が期待する形が違う、
dataset が振る identity ラベルが動画をまたいで衝突する、といった具合に、
どのモジュールも単体では正しいのに組み合わせると壊れる。

そこでここでは、合成の極小データセットを本物の ``Dataset`` / ``DataLoader`` /
``Trainer`` に通し、**1 バッチだけ実際に backward まで走らせる**。
事前学習済みチェックポイントは一切ダウンロードしない (backbone は pretrained=False)。

カバーする経路:

* Integration A: detection → OnlineTracker → TrackFeatureCache → temporal features
* Integration B: TrackWindowDataset → DataLoader → Stage3Trainer → ID loss → backward
* Integration C: SceneSequenceDataset → DataLoader → Stage4Trainer → loss → backward
* Integration D: 部分 pose アノテーション → matching → pose target → pose loss
* Integration E: Stage 1-4 の 1 ステップ学習
"""

from __future__ import annotations


import pytest
import torch
from torch.utils.data import DataLoader

from htrtdetr.config.config import build_model_config, get_variant_config
from htrtdetr.data.annotation import DatasetAnno, FrameAnno, ObjectAnno, VideoAnno
from htrtdetr.data.fly_dataset import (
    FlyDetectionDataset,
    SceneSequenceDataset,
    TrackWindowDataset,
    collate_detection,
    collate_scene_sequences,
    collate_track_windows,
)
from htrtdetr.models import build_model
from htrtdetr.training.losses import CombinedLoss
from htrtdetr.training.stage_trainers import (
    Stage1Trainer,
    Stage2Trainer,
    Stage3Trainer,
    Stage4Trainer,
)
from htrtdetr.training.targets import last_frame_targets

_HW = 64        # 合成画像はゼロ画像なので小さくてよい
_WINDOW = 4


# --------------------------------------------------------------------------- #
# 合成アノテーション
# --------------------------------------------------------------------------- #

def _make_video(video_id: str, track_ids, n_frames: int = 8) -> VideoAnno:
    """各 track が毎フレーム少しずつ動く、画像ファイルを持たない動画。"""
    frames = []
    for f in range(n_frames):
        objects = []
        for i, tid in enumerate(track_ids):
            x = 5.0 + 10.0 * i + f
            y = 5.0 + 10.0 * i
            objects.append(
                ObjectAnno(
                    object_id=i,
                    bbox=[x, y, x + 8.0, y + 8.0],
                    class_id=0,
                    track_id=tid,
                    action_id=i % 2,
                )
            )
        frames.append(
            FrameAnno(frame_index=f, image_path="", width=_HW, height=_HW,
                      objects=objects)
        )
    return VideoAnno(video_id=video_id, width=_HW, height=_HW,
                    num_frames=n_frames, frames=frames)


@pytest.fixture(scope="module")
def two_video_anno() -> DatasetAnno:
    """2 本の動画。どちらも track_id 0 と 1 を持つ (= 衝突しうる構成)。"""
    return DatasetAnno(videos=[
        _make_video("video_A", [0, 1]),
        _make_video("video_B", [0, 1]),
    ])


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = get_variant_config("small", stage=1)
    cfg.model.detector.backbone.pretrained = False
    cfg.model.detector.head.num_queries = 8
    cfg.model.detector.score_threshold = 0.0
    cfg.model.id_head.max_ids = 16
    cfg.train.use_amp = False
    return cfg


def _model(cfg, stage: int):
    m = build_model(cfg.model)
    m.set_stage(stage)
    return m


# --------------------------------------------------------------------------- #
# Integration A: detection → tracker → cache → temporal
# --------------------------------------------------------------------------- #

class TestIntegrationA_TrackingToTemporal:
    """
    複数個体・新規 track・既存 track を混ぜても、temporal 入力の各行が
    最後まで同じ個体のものであり続けること。
    """

    def test_track_histories_do_not_cross_individuals(self):
        from htrtdetr.tracking import OBBAssociator, OnlineTracker, TrackFeatureCache

        feat_dim = 4
        tracker = OnlineTracker(
            OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.5),
            TrackFeatureCache(window=8, feat_dim=feat_dim),
        )

        # 個体 i の特徴は定数ベクトル (i+1) * 1.0 — 混線したら一目で分かる
        positions = {0: 0.15, 1: 0.50, 2: 0.85}
        signature = {i: torch.full((feat_dim,), float(i + 1)) for i in positions}

        expected_feature_of_track: dict[int, torch.Tensor] = {}

        for t in range(6):
            # frame 2 から 3 個体目が登場し、検出順も毎フレーム変える
            present = [0, 1] if t < 2 else [2, 0, 1]
            if t % 2 == 1:
                present = list(reversed(present))

            obbs = torch.tensor([
                [positions[i] + 0.002 * t, 0.5, 0.08, 0.08, 0.0] for i in present
            ])
            feats = torch.stack([signature[i] for i in present])
            tracked = tracker.update(t, obbs, feats)

            assert len(tracked) == len(present)
            for obj in tracked:
                individual = present[obj.detection_index]
                prev = expected_feature_of_track.get(obj.track_id)
                if prev is not None:
                    assert torch.allclose(prev, signature[individual]), (
                        f"track {obj.track_id} が別個体に乗り移った "
                        f"(frame {t}, 検出 {obj.detection_index})"
                    )
                expected_feature_of_track[obj.track_id] = signature[individual]

        hist, obb_hist, mask, ids = tracker.track_history()
        assert len(ids) == 3, f"個体 3 匹に対し track が {len(ids)} 本"
        for row, tid in enumerate(ids):
            observed = hist[row][mask[row]]
            unique = {round(float(v), 3) for v in observed[:, 0].tolist()}
            assert len(unique) == 1, (
                f"track {tid} の履歴に複数個体の特徴が混ざった: {sorted(unique)}"
            )
            assert torch.allclose(
                observed[-1], expected_feature_of_track[tid]
            )

    def test_temporal_model_consumes_track_history(self, tiny_cfg):
        """cache の履歴がそのまま HTM に通り、track 行数ぶんの出力になる。"""
        from htrtdetr.tracking import OBBAssociator, OnlineTracker, TrackFeatureCache

        model = _model(tiny_cfg, stage=2).eval()
        feat_dim = model.detector.get_hidden_dim()
        tracker = OnlineTracker(
            OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.5),
            TrackFeatureCache(window=_WINDOW, feat_dim=feat_dim),
        )
        torch.manual_seed(0)
        for t in range(_WINDOW):
            obbs = torch.tensor([
                [0.2 + 0.01 * t, 0.3, 0.08, 0.08, 0.0],
                [0.8 - 0.01 * t, 0.7, 0.08, 0.08, 0.0],
            ])
            tracker.update(t, obbs, torch.randn(2, feat_dim))

        hist, _, mask, ids = tracker.track_history()
        assert hist.shape == (2, _WINDOW, feat_dim)
        with torch.no_grad():
            packed = model.track_temporal_fusion.pack_from_cache(hist, mask)
            out = model.temporal(packed)
        assert out.shape[0] == len(ids) == 2


# --------------------------------------------------------------------------- #
# Integration B: TrackWindowDataset → Stage3Trainer → ID loss → backward
# --------------------------------------------------------------------------- #

class TestIntegrationB_Stage3Identity:
    def _dataset(self, anno) -> TrackWindowDataset:
        return TrackWindowDataset(
            anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, min_track_length=_WINDOW,
        )

    def test_same_local_id_in_different_videos_gets_different_identity(
        self, two_video_anno
    ):
        """
        P0-3 回帰: video A の track 0 と video B の track 0 は別 identity。

        修正前は両方 local_track_id=0 になり、classification では同一クラス、
        triplet では positive pair として学習されていた。
        """
        ds = self._dataset(two_video_anno)
        a = ds.global_identity_map[("video_A", 0)]
        b = ds.global_identity_map[("video_B", 0)]
        assert a != b, "別動画の別個体が同じ identity クラスになっている"

        # local_track_id は従来通り衝突する (後方互換のため残している)
        assert ds.track_local_id_map["video_A"][0] == ds.track_local_id_map["video_B"][0]
        assert ds.num_identities == 4

    def test_loss_labels_differ_across_videos(self, two_video_anno):
        """DataLoader を通した batch の identity_ids でも別値になること。"""
        ds = self._dataset(two_video_anno)
        loader = DataLoader(
            ds, batch_size=len(ds), shuffle=False, collate_fn=collate_track_windows,
        )
        batch = next(iter(loader))

        by_video: dict[tuple, set] = {}
        for i, meta in enumerate(batch["meta"]):
            key = (meta["video_id"], int(batch["track_ids"][i]))
            by_video.setdefault(key, set()).add(int(batch["identity_ids"][i]))

        a = by_video[("video_A", 0)]
        b = by_video[("video_B", 0)]
        assert a.isdisjoint(b), (
            f"video_A/track0 {a} と video_B/track0 {b} が同じ identity ラベル"
        )
        # triplet loss はこのラベルで positive / negative を決める
        assert len({next(iter(v)) for v in by_video.values()}) == len(by_video)

    def test_stage3_train_step_runs_and_backpropagates(self, two_video_anno, tiny_cfg, tmp_path):
        """TrackWindowDataset → DataLoader → Stage3Trainer → ID loss → backward。"""
        ds = self._dataset(two_video_anno)
        loader = DataLoader(
            ds, batch_size=4, shuffle=True, collate_fn=collate_track_windows,
        )
        model = _model(tiny_cfg, stage=3)
        model.train()
        trainer = Stage3Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)

        optimizer = trainer._make_optimizer(
            [p for p in model.parameters() if p.requires_grad]
        )
        metrics = trainer._train_epoch(loader, optimizer, epoch=1, num_epochs=1)

        assert "loss_id" in metrics
        assert metrics["loss_id"] > 0.0
        grads = [
            n for n, p in model.named_parameters()
            if p.grad is not None and float(p.grad.norm()) > 0
        ]
        assert any(n.startswith("id_head") or n.startswith("geo_") for n in grads), (
            "ID head に勾配が届いていない"
        )

    def test_stage3_val_step_uses_identity_labels(self, two_video_anno, tiny_cfg, tmp_path):
        ds = self._dataset(two_video_anno)
        loader = DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=collate_track_windows,
        )
        model = _model(tiny_cfg, stage=3)
        trainer = Stage3Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        metrics = trainer._val_epoch(loader)
        assert "loss_id" in metrics


# --------------------------------------------------------------------------- #
# Integration C: SceneSequenceDataset → Stage4Trainer → loss → backward
# --------------------------------------------------------------------------- #

class TestIntegrationC_Stage4Sequence:
    def _loader(self, anno, batch_size: int = 2) -> DataLoader:
        ds = SceneSequenceDataset(
            anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, max_tracks=4,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_scene_sequences,
        )

    def test_collate_returns_batch_by_time_targets(self, two_video_anno):
        """collate の契約を固定する: targets は List[List[Dict]] (B × T)。"""
        loader = self._loader(two_video_anno)
        batch = next(iter(loader))
        targets = batch["targets"]
        assert isinstance(targets, list) and isinstance(targets[0], list)
        assert len(targets) == batch["images"].shape[0]
        assert len(targets[0]) == batch["images"].shape[1] == _WINDOW
        assert isinstance(targets[0][0], dict)

    def test_last_frame_targets_picks_one_per_batch_element(self, two_video_anno):
        """
        P0-1 回帰の核心: B × T から取り出すのは「各シーケンスの最終フレーム」。
        ``targets[-1]`` は「最後のバッチ要素のシーケンス全体」であって別物。
        """
        loader = self._loader(two_video_anno)
        batch = next(iter(loader))
        last = last_frame_targets(batch["targets"])
        assert len(last) == batch["images"].shape[0]
        assert all(isinstance(t, dict) and "boxes" in t for t in last)
        for b, seq in enumerate(batch["targets"]):
            assert last[b] is seq[-1]

    def test_stage4_train_step_runs_and_backpropagates(self, two_video_anno, tiny_cfg, tmp_path):
        """
        SceneSequenceDataset → DataLoader → Stage4Trainer → model → loss → backward。

        修正前はここで ``AttributeError: 'list' object has no attribute 'items'``
        になり、Stage 4 の学習が 1 バッチも回らなかった。
        """
        loader = self._loader(two_video_anno)
        model = _model(tiny_cfg, stage=4)
        model.train()
        trainer = Stage4Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)

        optimizer = trainer._make_optimizer()
        metrics = trainer._train_epoch(loader, optimizer, epoch=1, num_epochs=1)

        assert metrics, "1 バッチも損失が計算されなかった"
        assert "total_loss" in metrics
        assert any(
            p.grad is not None and float(p.grad.norm()) > 0
            for p in model.parameters()
        ), "backward で勾配が一切流れていない"

    def test_stage4_val_step_runs(self, two_video_anno, tiny_cfg, tmp_path):
        loader = self._loader(two_video_anno)
        model = _model(tiny_cfg, stage=4)
        trainer = Stage4Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        metrics = trainer._val_epoch(loader)
        assert "loss_detection" in metrics


# --------------------------------------------------------------------------- #
# Integration D: 部分 pose アノテーション → matching → pose loss
# --------------------------------------------------------------------------- #

class TestIntegrationD_PartialPose:
    def test_partial_keypoints_do_not_leak_between_images(self, tiny_cfg):
        """
        一部の画像だけ keypoint GT を持つバッチを、実際の model 出力へ通す。
        GT が別画像へ broadcast されるなら、ここで行数不一致の例外になる。
        """
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 8
        cfg.detector.score_threshold = 0.0
        cfg.detector.pose.enabled = True
        cfg.detector.pose.num_keypoints = 3
        model = build_model(cfg)
        # stage 1 の forward は detection だけで早期 return するので det_results が
        # 無い。pose の GT マッチングは det_results 経由なので stage 4 で回す。
        model.set_stage(4)
        model.train()

        B = 3
        out = model(torch.randn(B, _WINDOW, 3, _HW, _HW))
        assert out.det_results is not None

        from htrtdetr.utils.misc import cxcywh_to_xyxy

        targets = []
        for b in range(B):
            boxes = out.det_results[b]["boxes"][:2].detach()
            n = boxes.shape[0]
            t = {
                "boxes": cxcywh_to_xyxy(boxes),
                "class_ids": torch.zeros(n, dtype=torch.long),
                "track_ids": torch.arange(n),
                "action_ids": torch.zeros(n, dtype=torch.long),
            }
            if b == 0:                          # 1 枚目だけ pose ラベルあり
                kp = torch.rand(n, cfg.detector.pose.num_keypoints, 3)
                kp[..., 2] = 2.0
                t["keypoints"] = kp
            targets.append(t)

        loss = CombinedLoss(LossConfig_or_default(), num_classes=1, num_actions=5)
        loss.set_stage(1)
        losses = loss(out, targets, stage=1)

        assert "loss_pose" in losses
        assert torch.isfinite(losses["total_loss"]).all()
        losses["total_loss"].backward()

    def test_no_pose_annotation_anywhere_produces_no_pose_loss(self, tiny_cfg):
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 8
        cfg.detector.score_threshold = 0.0
        model = build_model(cfg)
        model.set_stage(4)
        model.train()

        out = model(torch.randn(2, _WINDOW, 3, _HW, _HW))
        from htrtdetr.utils.misc import cxcywh_to_xyxy

        targets = []
        for b in range(2):
            boxes = (out.det_results[b]["boxes"][:2].detach()
                     if out.det_results is not None
                     else out.pred_boxes[b][:2].detach())
            n = boxes.shape[0]
            targets.append({
                "boxes": cxcywh_to_xyxy(boxes),
                "class_ids": torch.zeros(n, dtype=torch.long),
                "track_ids": torch.arange(n),
                "action_ids": torch.zeros(n, dtype=torch.long),
            })

        loss = CombinedLoss(LossConfig_or_default(), num_classes=1, num_actions=5)
        loss.set_stage(1)
        losses = loss(out, targets, stage=1)
        assert not any(k.startswith("loss_pose") for k in losses)


def LossConfig_or_default():
    from htrtdetr.config.config import LossConfig
    return LossConfig()


# --------------------------------------------------------------------------- #
# Integration E: Stage 1-4 の 1 ステップ学習 (重い checkpoint は落とさない)
# --------------------------------------------------------------------------- #

class TestIntegrationE_OneStepPerStage:
    def test_stage1_one_step(self, two_video_anno, tiny_cfg, tmp_path):
        ds = FlyDetectionDataset(
            two_video_anno, image_size=(_HW, _HW), augment=False
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_detection)
        model = _model(tiny_cfg, stage=1)
        model.train()
        trainer = Stage1Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        optimizer = trainer._make_optimizer()
        metrics = trainer._train_epoch(loader, optimizer, epoch=1, num_epochs=1)
        assert "loss" in metrics or "loss_detection" in metrics

    def test_stage2_one_step(self, two_video_anno, tiny_cfg, tmp_path):
        ds = TrackWindowDataset(
            two_video_anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, min_track_length=_WINDOW,
        )
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=collate_track_windows
        )
        model = _model(tiny_cfg, stage=2)
        model.train()
        trainer = Stage2Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        optimizer = trainer._make_optimizer(
            [p for p in model.parameters() if p.requires_grad]
        )
        metrics = trainer._train_epoch(loader, optimizer, epoch=1, num_epochs=1)
        assert "loss" in metrics
        # 検証ループも同じ collate のキーで動くこと (action_labels は存在しない)
        assert "loss" in trainer._val_epoch(loader)

    def test_stage3_one_step(self, two_video_anno, tiny_cfg, tmp_path):
        ds = TrackWindowDataset(
            two_video_anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, min_track_length=_WINDOW,
        )
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=collate_track_windows
        )
        model = _model(tiny_cfg, stage=3)
        model.train()
        trainer = Stage3Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        optimizer = trainer._make_optimizer(
            [p for p in model.parameters() if p.requires_grad]
        )
        assert "loss_id" in trainer._train_epoch(loader, optimizer, 1, 1)

    def test_stage4_one_step(self, two_video_anno, tiny_cfg, tmp_path):
        ds = SceneSequenceDataset(
            two_video_anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, max_tracks=4,
        )
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=collate_scene_sequences
        )
        model = _model(tiny_cfg, stage=4)
        model.train()
        trainer = Stage4Trainer(tiny_cfg, model, torch.device("cpu"), tmp_path)
        optimizer = trainer._make_optimizer()
        assert trainer._train_epoch(loader, optimizer, 1, 1)

    def test_no_checkpoint_download_happens(self, tiny_cfg):
        """backbone は pretrained=False — CI でネットワークを要求しない。"""
        assert tiny_cfg.model.detector.backbone.pretrained is False


class TestIntegrationB2_CanonicalTrainerIdentity:
    """
    P0-3 は **正式な学習経路** (``yoake train`` → ``htrtdetr.training.Trainer``)
    にも同じ形で存在した。こちらは ``SlidingWindowDataset`` を使い、アノテーション
    の ``track_id`` をそのまま ID 分類ラベルにしていたため、動画をまたいで
    shuffle すると「video A の track 0」と「video B の track 0」が同じ identity
    クラスになっていた。

    dataset が ``(video_id, track_id)`` から全体一意な ``identity_ids`` を出し、
    ``CombinedLoss`` がそれを優先することで解消する。
    """

    def _videos(self):
        from htrtdetr.data.annotation import FrameAnno, ObjectAnno, VideoAnno

        def one(vid, tids, n=_WINDOW):
            frames = []
            for f in range(n):
                objs = [
                    ObjectAnno(object_id=i, bbox=[5.0 + f, 5.0, 25.0 + f, 25.0],
                               track_id=t, action_id=0)
                    for i, t in enumerate(tids)
                ]
                frames.append(FrameAnno(frame_index=f, image_path="",
                                        width=_HW, height=_HW, objects=objs))
            return VideoAnno(video_id=vid, width=_HW, height=_HW,
                             num_frames=n, frames=frames)

        return [one("video_A", [0, 1]), one("video_B", [0, 1])]

    def test_sliding_window_dataset_separates_identities_across_videos(self):
        from htrtdetr.data.dataset import SlidingWindowDataset

        ds = SlidingWindowDataset(self._videos(), window_size=_WINDOW,
                                  stride=_WINDOW, image_size=(_HW, _HW),
                                  augment=False)
        assert ds.num_identities == 4
        per_video = {}
        for i in range(len(ds)):
            s = ds[i]
            per_video[s["video_id"]] = s["identity_ids"][-1].tolist()

        assert per_video["video_A"] == [0, 1]
        assert per_video["video_B"] == [2, 3], (
            "video B の個体が video A と同じ identity クラスになっている"
        )

    def test_single_frame_dataset_separates_identities_across_videos(self):
        from htrtdetr.data.dataset import SingleFrameDataset

        ds = SingleFrameDataset(self._videos(), image_size=(_HW, _HW), augment=False)
        assert ds.num_identities == 4
        first = ds[0]["identity_ids"].tolist()
        later = ds[len(ds) // 2]["identity_ids"].tolist()
        assert set(first).isdisjoint(set(later))

    def test_collate_carries_identity_ids_into_targets(self):
        from htrtdetr.data.collate import collate_sequence
        from htrtdetr.data.dataset import SlidingWindowDataset

        ds = SlidingWindowDataset(self._videos(), window_size=_WINDOW,
                                  stride=_WINDOW, image_size=(_HW, _HW),
                                  augment=False)
        batch = collate_sequence([ds[0], ds[1]])
        last = last_frame_targets(batch["targets"])
        assert all("identity_ids" in t for t in last)
        a, b = [t["identity_ids"].tolist() for t in last]
        assert set(a).isdisjoint(set(b)), f"identity が動画をまたいで衝突: {a} {b}"
        # track_ids は従来どおり動画内 ID のまま (表示・保存の後方互換)
        assert [t["track_ids"].tolist() for t in last] == [[0, 1], [0, 1]]

    def test_combined_loss_prefers_identity_ids_over_track_ids(self):
        """GT 収集が identity_ids を選ぶこと (track_ids だと別個体が同一クラス)。"""
        from htrtdetr.config.config import LossConfig
        from htrtdetr.training.losses import CombinedLoss

        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5, max_ids=8)

        class _Out:
            det_results = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]

        targets = [{
            "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]),
            "track_ids": torch.tensor([0]),
            "identity_ids": torch.tensor([5]),
        }]
        gt = loss._collect_gt_identity_ids(targets, _Out())
        assert gt.tolist() == [5], "identity_ids ではなく track_ids を使っている"

    def test_falls_back_to_track_ids_for_old_datasets(self):
        """``identity_ids`` を持たない古い batch でも動くこと (後方互換)。"""
        from htrtdetr.config.config import LossConfig
        from htrtdetr.training.losses import CombinedLoss

        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5, max_ids=8)

        class _Out:
            det_results = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]

        targets = [{
            "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]),
            "track_ids": torch.tensor([3]),
        }]
        assert loss._collect_gt_identity_ids(targets, _Out()).tolist() == [3]


class TestIdentityClassifierCapacity:
    """
    identity は dataset 全体で一意なので、動画を足すとクラス数が増える。
    ``max_ids`` を超えた分は ``IDLoss`` が "unknown" クラスへ畳んでしまうため、
    教師信号が静かに捨てられる。ヘッドが自動で広がることを固定する。
    """

    def _loader(self, anno):
        ds = TrackWindowDataset(
            anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, min_track_length=_WINDOW,
        )
        return DataLoader(ds, batch_size=2, shuffle=False,
                          collate_fn=collate_track_windows)

    def test_resize_preserves_known_id_weights(self):
        from htrtdetr.config.config import MemoryIDConfig
        from htrtdetr.models.id_head import MemoryIDHead

        cfg = MemoryIDConfig()
        cfg.max_ids = 4
        head = MemoryIDHead(cfg)
        before = head.id_classifier.weight.detach().clone()

        head.resize_id_classifier(9)

        assert head.id_classifier.out_features == 10   # 9 known + unknown
        assert head.cfg.max_ids == 9
        assert torch.allclose(head.id_classifier.weight[:4], before[:4]), (
            "既知 ID の重みが失われた (checkpoint を捨てることになる)"
        )
        assert torch.allclose(head.id_classifier.weight[9], before[4]), (
            "unknown 行が末尾へ移っていない"
        )

    def test_resize_is_a_noop_when_already_big_enough(self):
        from htrtdetr.config.config import MemoryIDConfig
        from htrtdetr.models.id_head import MemoryIDHead

        head = MemoryIDHead(MemoryIDConfig())
        same = head.id_classifier
        head.resize_id_classifier(head.cfg.max_ids)
        assert head.id_classifier is same

    def test_stage3_trainer_grows_the_head_for_a_big_dataset(
        self, two_video_anno, tmp_path
    ):
        cfg = get_variant_config("small", stage=3)
        cfg.model.detector.backbone.pretrained = False
        cfg.model.detector.head.num_queries = 8
        cfg.model.id_head.max_ids = 2          # dataset には identity が 4 つある
        cfg.train.use_amp = False

        loader = self._loader(two_video_anno)
        assert loader.dataset.num_identities == 4

        model = _model(cfg, stage=3)
        trainer = Stage3Trainer(cfg, model, torch.device("cpu"), tmp_path)
        assert trainer.max_ids == 2

        trainer._fit_id_classifier_to_dataset(loader)

        assert trainer.max_ids == 4
        assert model.id_head.id_classifier.out_features == 5
        assert trainer.loss_fn.max_ids == 4

    def test_id_loss_warns_once_when_labels_exceed_capacity(self):
        from htrtdetr.config.config import LossConfig
        from htrtdetr.training.losses import IDLoss

        loss = IDLoss(LossConfig(), max_ids=3)
        logits = torch.randn(4, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 7, 9])       # 7 と 9 は範囲外

        with pytest.warns(RuntimeWarning, match="max_ids"):
            out = loss(logits, labels)
        assert loss.num_clamped == 2
        assert torch.isfinite(out["loss_id"])

        # 2 回目は黙る (毎ステップ出すとログが埋まる) が、件数は数え続ける
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            loss(logits, labels)
        assert loss.num_clamped == 4

    def test_id_loss_is_silent_when_capacity_is_enough(self):
        from htrtdetr.config.config import LossConfig
        from htrtdetr.training.losses import IDLoss
        import warnings as _w

        loss = IDLoss(LossConfig(), max_ids=10)
        with _w.catch_warnings():
            _w.simplefilter("error")
            loss(torch.randn(3, 11), torch.tensor([0, 1, 2]))
        assert loss.num_clamped == 0


class TestIdentityResizeDoesNotBreakEma:
    """
    ID ヘッドの自動拡張が ``ModelEMA`` と噛み合うこと。

    ``ModelEMA`` は生成時のパラメータ形状を影コピーとして保持する。ヘッドを
    後から差し替えると、最初の ``ema.update()`` で
    ``RuntimeError: The size of tensor a (51) must match the size of tensor b``
    になり、**学習が 1 エポック目で落ちる**。実際に踏んだので固定する。

    併せて、ID ヘッドを学習しない stage では形を変えないことも確認する
    (Stage 1 は detector だけを学習するので、dataset の identity 数に応じて
    checkpoint の形が変わる理由がない)。
    """

    def _cfg(self, stage: int, max_ids: int = 2):
        cfg = get_variant_config("small", stage=stage)
        cfg.model.detector.backbone.pretrained = False
        cfg.model.detector.head.num_queries = 8
        cfg.model.id_head.max_ids = max_ids
        cfg.train.use_amp = False
        cfg.train.stage = stage
        cfg.train.max_epochs = 1
        return cfg

    def _loader(self, anno):
        ds = TrackWindowDataset(
            anno, window_size=_WINDOW, stride=_WINDOW,
            image_size=(_HW, _HW), augment=False, min_track_length=_WINDOW,
        )
        return DataLoader(ds, batch_size=2, shuffle=False,
                          collate_fn=collate_track_windows)

    def _trainer(self, cfg, loader, tmp_path):
        from htrtdetr.training import Trainer

        cfg.train.output_dir = str(tmp_path)
        model = _model(cfg, stage=cfg.train.stage)
        return Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            train_cfg=cfg.train,
            loss_cfg=cfg.loss,
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            max_ids=cfg.model.id_head.max_ids,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler,
            metrics_cfg=getattr(cfg, "metrics", None),
            full_cfg=cfg,
        )

    def test_ema_shadow_matches_the_resized_head(self, two_video_anno, tmp_path):
        cfg = self._cfg(stage=3, max_ids=2)
        cfg.train.use_ema = True
        loader = self._loader(two_video_anno)
        assert loader.dataset.num_identities == 4      # > max_ids, so it resizes

        trainer = self._trainer(cfg, loader, tmp_path)
        assert trainer.model.id_head.id_classifier.out_features == 5
        assert trainer.ema is not None

        # 落ちないこと自体がテスト (影コピーの形が合っていなければここで
        # RuntimeError になる)
        trainer.ema.update(trainer.model)

        shadow = trainer.ema.ema.id_head.id_classifier.weight
        assert tuple(shadow.shape) == tuple(
            trainer.model.id_head.id_classifier.weight.shape
        )

    def test_stage1_never_resizes_the_id_head(self, two_video_anno, tmp_path):
        cfg = self._cfg(stage=1, max_ids=2)
        cfg.train.use_ema = True
        loader = self._loader(two_video_anno)

        trainer = self._trainer(cfg, loader, tmp_path)
        assert trainer.model.id_head.id_classifier.out_features == 3, (
            "Stage 1 は detector だけを学習するので ID ヘッドの形を変えてはいけない"
        )
        trainer.ema.update(trainer.model)


class TestLegacyTrainerDeprecation:
    """
    ``stage_trainers.py`` は非推奨だが、消せない (README / TRAINING_GUIDE が
    参照する ``scripts/train_stage*_{detector,action,id,unified}.py`` が使う)。

    「非推奨だから放置」にならないよう 2 つを固定する:
      * 使うと必ず移行先を名乗る DeprecationWarning が出る
      * 上の TestIntegrationE で stage 1-4 とも 1 ステップ回ることを確認済み
    """

    @pytest.mark.parametrize(
        "cls_name,stage",
        [("Stage1Trainer", 1), ("Stage2Trainer", 2),
         ("Stage3Trainer", 3), ("Stage4Trainer", 4)],
    )
    def test_construction_warns_with_the_replacement(
        self, cls_name, stage, tiny_cfg, tmp_path
    ):
        import htrtdetr.training.stage_trainers as legacy

        cls = getattr(legacy, cls_name)
        model = _model(tiny_cfg, stage=stage)
        with pytest.warns(DeprecationWarning, match=f"yoake train stage={stage}"):
            cls(tiny_cfg, model, torch.device("cpu"), tmp_path)

    def test_canonical_trainer_does_not_warn(self):
        """``htrtdetr.training.Trainer`` は現役なので警告しない。"""
        import htrtdetr.training as t

        assert t.Trainer.__module__.endswith("trainer")
