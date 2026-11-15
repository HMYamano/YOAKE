"""
test_tracking.py — OBB 対応 tracking と track 特徴リングバッファのテスト

検証項目:
  * association (幾何 / 外観 / 運動 / 向き)
  * track の生成・消失・再出現 (re-ID)
  * リングバッファの欠損マスクと時系列順
  * track-aligned 履歴の組み立て
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from htrtdetr.tracking import (
    FrameDetections,
    OBBAssociator,
    OnlineTracker,
    TrackFeatureCache,
    build_track_aligned_history,
)


def _obb(cx, cy, w=0.1, h=0.05, th=0.0):
    return torch.tensor([[cx, cy, w, h, th]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

class TestAssociator:
    def test_perfect_overlap_matches(self):
        a = OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.5)
        det = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.1, 0.1, 0.2, 0.1, 0.0]])
        trk = torch.tensor([[0.1, 0.1, 0.2, 0.1, 0.0], [0.5, 0.5, 0.2, 0.1, 0.3]])
        res = a.associate(det, trk)
        assert sorted(res.matches) == [(0, 1), (1, 0)]
        assert all(s > 0.99 for s in res.scores)

    def test_below_threshold_is_unmatched(self):
        a = OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.9)
        det = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        trk = torch.tensor([[0.9, 0.9, 0.2, 0.1, 0.0]])   # 重なりなし
        res = a.associate(det, trk)
        assert res.matches == []
        assert res.unmatched_detections == [0]
        assert res.unmatched_tracks == [0]

    def test_appearance_breaks_geometric_tie(self):
        """
        幾何的に区別できない 2 個体を外観で正しく分けられること。
        (接近して重なる個体で ID が入れ替わるのを防ぐ核心)
        """
        a = OBBAssociator(iou_weight=0.0, appearance_weight=1.0, match_threshold=0.6)
        det_obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0], [0.5, 0.5, 0.2, 0.1, 0.0]])
        trk_obb = det_obb.clone()
        det_emb = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
        trk_emb = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=-1)
        res = a.associate(det_obb, trk_obb, det_embed=det_emb, track_embed=trk_emb)
        assert sorted(res.matches) == [(0, 1), (1, 0)]

    def test_orientation_weight_prefers_same_heading(self):
        a = OBBAssociator(
            iou_weight=0.0, appearance_weight=0.0, orientation_weight=1.0,
            match_threshold=0.5,
        )
        det = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        trk = torch.tensor([
            [0.5, 0.5, 0.2, 0.1, math.pi / 2],   # 直交 → 不一致
            [0.5, 0.5, 0.2, 0.1, 0.02],          # ほぼ同方向 → 一致
        ])
        res = a.associate(det, trk)
        assert res.matches == [(0, 1)]

    def test_motion_weight_uses_velocity_prediction(self):
        a = OBBAssociator(
            iou_weight=0.0, appearance_weight=0.0, motion_weight=1.0,
            match_threshold=0.5,
        )
        det = torch.tensor([[0.60, 0.5, 0.1, 0.05, 0.0]])
        trk = torch.tensor([
            [0.50, 0.5, 0.1, 0.05, 0.0],    # 速度 +0.1 なら 0.60 に来る
            [0.30, 0.5, 0.1, 0.05, 0.0],    # 速度 0 なら 0.30 のまま
        ])
        vel = torch.tensor([[0.10, 0.0], [0.0, 0.0]])
        res = a.associate(det, trk, track_velocity=vel)
        assert res.matches == [(0, 0)]

    def test_center_distance_gate(self):
        a = OBBAssociator(
            iou_weight=1.0, appearance_weight=0.0,
            match_threshold=0.0, max_center_distance=0.05,
        )
        det = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        trk = torch.tensor([[0.9, 0.9, 0.2, 0.1, 0.0]])
        assert a.associate(det, trk).matches == []

    def test_empty_inputs(self):
        a = OBBAssociator()
        res = a.associate(torch.zeros(0, 5), torch.rand(3, 5))
        assert res.matches == []
        assert res.unmatched_tracks == [0, 1, 2]

    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValueError, match="重み"):
            OBBAssociator(iou_weight=0.0, appearance_weight=0.0)

    def test_similarity_averages_only_the_cues_it_could_compute(self):
        """
        embedding の無い呼び出しで appearance の重みを分母に残すと、同一の箱でも
        類似度が 0.5 で頭打ちになり閾値を超えない (学習時の内部 tracker が
        一度も連結しなかった原因)。計算できた手がかりだけで平均する。
        """
        a = OBBAssociator()   # 既定: iou 0.5 + appearance 0.5
        box = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])
        sim, terms = a.similarity(box, box)
        assert "appearance" not in terms
        assert float(sim[0, 0]) == pytest.approx(1.0, abs=1e-4)

    def test_similarity_with_every_cue_is_unchanged(self):
        """全手がかりが揃う streaming 推論の値は修正前と同じ重み付き平均。"""
        a = OBBAssociator(iou_weight=0.5, appearance_weight=0.5)
        box = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])
        e1 = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        e2 = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)     # cos 0 -> 0.5
        sim, _ = a.similarity(box, box, det_embed=e1, track_embed=e2)
        assert float(sim[0, 0]) == pytest.approx((0.5 * 1.0 + 0.5 * 0.5) / 1.0, abs=1e-4)

    def test_no_usable_cue_means_no_match_not_a_crash(self):
        a = OBBAssociator(iou_weight=0.0, appearance_weight=1.0)
        box = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])
        res = a.associate(box, box)     # embedding が無い -> 手がかりゼロ
        assert res.matches == []


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class TestTrackFeatureCache:
    def test_history_is_chronological(self):
        c = TrackFeatureCache(window=4, feat_dim=2)
        for t in range(4):
            c.update(t, [0], torch.full((1, 2), float(t)), torch.zeros(1, 5))
        feats, _, mask, ids = c.stacked_history()
        assert ids == [0]
        assert feats[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert mask[0].tolist() == [True] * 4

    def test_ring_buffer_wraps(self):
        """window を超えたら最古が落ちて最新が残る。"""
        c = TrackFeatureCache(window=3, feat_dim=1)
        for t in range(6):
            c.update(t, [0], torch.full((1, 1), float(t)), torch.zeros(1, 5))
        feats, _, _, _ = c.stacked_history()
        assert feats[0, :, 0].tolist() == [3.0, 4.0, 5.0]

    def test_missing_frames_are_masked_not_filled(self):
        """未観測フレームは mask=False で明示され、前フレームの複製にはならない。"""
        c = TrackFeatureCache(window=4, feat_dim=1)
        c.update(0, [0], torch.ones(1, 1), torch.zeros(1, 5))
        c.update(1, [], torch.zeros(0, 1), torch.zeros(0, 5))
        c.update(2, [0], torch.full((1, 1), 5.0), torch.zeros(1, 5))
        feats, _, mask, _ = c.stacked_history()
        assert mask[0].tolist()[-3:] == [True, False, True]
        assert feats[0, -2, 0].item() == 0.0, "欠損は複製せずゼロのまま"

    def test_track_expires_after_max_age(self):
        c = TrackFeatureCache(window=4, feat_dim=1, max_age=2, reid_max_age=5)
        c.update(0, [7], torch.ones(1, 1), torch.zeros(1, 5))
        assert c.active_ids() == [7]
        for t in range(1, 4):
            c.update(t, [], torch.zeros(0, 1), torch.zeros(0, 5))
        assert c.active_ids() == []
        assert 7 in c.reid_ids(), "max_age 超過後も re-ID 候補として残る"

    def test_track_evicted_after_reid_max_age(self):
        c = TrackFeatureCache(window=4, feat_dim=1, max_age=1, reid_max_age=2)
        c.update(0, [7], torch.ones(1, 1), torch.zeros(1, 5))
        for t in range(1, 5):
            c.update(t, [], torch.zeros(0, 1), torch.zeros(0, 5))
        assert 7 not in c

    def test_velocity_estimation(self):
        c = TrackFeatureCache(window=4, feat_dim=1)
        for t in range(3):
            obb = torch.tensor([[0.1 * t, 0.0, 0.1, 0.1, 0.0]])
            c.update(t, [0], torch.ones(1, 1), obb)
        v = c.velocities([0])
        assert v[0, 0].item() == pytest.approx(0.1, abs=1e-5)

    def test_count_mismatch_raises(self):
        c = TrackFeatureCache(window=2, feat_dim=3)
        with pytest.raises(ValueError, match="件数"):
            c.update(0, [1, 2], torch.zeros(1, 3), torch.zeros(1, 5))

    def test_feature_dim_mismatch_raises(self):
        c = TrackFeatureCache(window=2, feat_dim=3)
        with pytest.raises(ValueError, match="feature dim"):
            c.update(0, [1], torch.zeros(1, 9), torch.zeros(1, 5))

    def test_reid_max_age_must_exceed_max_age(self):
        with pytest.raises(ValueError, match="reid_max_age"):
            TrackFeatureCache(window=4, feat_dim=1, max_age=10, reid_max_age=2)

    def test_empty_history(self):
        c = TrackFeatureCache(window=4, feat_dim=2)
        feats, obb, mask, ids = c.stacked_history()
        assert feats.shape == (0, 4, 2) and ids == []


# ---------------------------------------------------------------------------
# OnlineTracker
# ---------------------------------------------------------------------------

class TestOnlineTracker:
    def _tracker(self, **kw):
        assoc = OBBAssociator(
            iou_weight=kw.pop("iou_w", 0.6),
            appearance_weight=kw.pop("app_w", 0.4),
            match_threshold=kw.pop("thr", 0.5),
        )
        return OnlineTracker(
            assoc, TrackFeatureCache(window=8, feat_dim=4, **kw)
        )

    def test_ids_are_stable_for_moving_objects(self):
        tr = self._tracker()
        emb = F.normalize(torch.tensor([[1.0, 0, 0, 0], [0.0, 1, 0, 0]]), dim=-1)
        seen = []
        for t in range(10):
            obb = torch.tensor([
                [0.2 + 0.01 * t, 0.3, 0.10, 0.05, 0.2],
                [0.7 - 0.01 * t, 0.6, 0.10, 0.05, -0.4],
            ])
            out = tr.update(t, obb, torch.randn(2, 4), det_embed=emb)
            seen.append(tuple(sorted(o.track_id for o in out)))
        assert len(set(seen)) == 1, f"ID が揺れた: {set(seen)}"
        assert seen[0] == (0, 1)

    def test_default_config_links_detections_without_embeddings(self):
        """
        学習時の内部 tracker (``HTRTDETR._build_track_aligned``) と同じ呼び方:
        既定の TrackingConfig、embedding なし。以前は毎フレーム新しい ID になり、
        track 単位の履歴が 1 フレームしか無かった。
        """
        from htrtdetr.config.config import TrackingConfig

        tr = OnlineTracker.from_config(TrackingConfig(), feat_dim=4, window=8)
        seen = []
        for t in range(6):
            obb = torch.tensor([
                [0.30 + 0.004 * t, 0.40, 0.08, 0.03, 0.3],
                [0.70 - 0.004 * t, 0.60, 0.08, 0.03, -0.5],
            ])
            out = tr.update(t, obb, torch.randn(2, 4), has_obb=True)
            seen.append([o.track_id for o in out])
        assert seen == [[0, 1]] * 6, seen
        _feats, _obbs, mask, ids = tr.track_history()
        assert sorted(ids) == [0, 1]
        assert int(mask.sum(dim=1).min()) == 6     # 6 フレームすべてが同じ track に積まれる

    def test_new_track_created_for_new_object(self):
        tr = self._tracker()
        tr.update(0, _obb(0.2, 0.2), torch.randn(1, 4))
        out = tr.update(1, torch.cat([_obb(0.2, 0.2), _obb(0.8, 0.8)]), torch.randn(2, 4))
        assert len(out) == 2
        assert {o.track_id for o in out} == {0, 1}
        assert any(o.is_new for o in out)

    def test_lost_track_then_reappears_gets_new_id_without_embeddings(self):
        """
        embedding が無い構成では re-ID しない (幾何だけで復元すると別個体に
        同じ ID を与える危険があるため)。
        """
        tr = OnlineTracker(
            OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.5),
            TrackFeatureCache(window=8, feat_dim=4, max_age=1, reid_max_age=6),
        )
        tr.update(0, _obb(0.2, 0.2), torch.randn(1, 4))
        for t in range(1, 5):
            tr.update(t, torch.zeros(0, 5), torch.zeros(0, 4))
        out = tr.update(5, _obb(0.2, 0.2), torch.randn(1, 4))
        assert out[0].track_id != 0
        assert out[0].is_new

    def test_reid_restores_id_with_embeddings(self):
        """外観が一致すれば消失後に戻ってきた個体へ同じ ID を返す。"""
        tr = OnlineTracker(
            OBBAssociator(iou_weight=1.0, appearance_weight=0.0, match_threshold=0.5),
            TrackFeatureCache(window=8, feat_dim=4, max_age=1, reid_max_age=8),
            enable_reid=True, reid_threshold=0.9,
        )
        emb = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), dim=-1)
        tr.update(0, _obb(0.2, 0.2), torch.randn(1, 4), det_embed=emb)
        for t in range(1, 4):
            tr.update(t, torch.zeros(0, 5), torch.zeros(0, 4),
                      det_embed=torch.zeros(0, 4))
        out = tr.update(4, _obb(0.9, 0.9), torch.randn(1, 4), det_embed=emb)
        assert out[0].track_id == 0
        assert out[0].is_reid

    def test_reset_clears_state(self):
        tr = self._tracker()
        tr.update(0, _obb(0.2, 0.2), torch.randn(1, 4))
        tr.reset()
        assert tr.cache.active_ids() == []
        out = tr.update(0, _obb(0.2, 0.2), torch.randn(1, 4))
        assert out[0].track_id == 0

    def test_empty_frame_is_safe(self):
        tr = self._tracker()
        assert tr.update(0, torch.zeros(0, 5), torch.zeros(0, 4)) == []


# ---------------------------------------------------------------------------
# track-aligned 履歴の組み立て
# ---------------------------------------------------------------------------

class TestTrackAlignedAssembly:
    def test_history_groups_by_track_not_position(self):
        """
        同じ個体が検出リスト内で別の位置に来ても、履歴は track_id で揃う。
        (query index で揃えると壊れるケースを直接検証する)
        """
        T = 3
        frames = []
        for t in range(T):
            # 検出の順番を毎フレーム入れ替える
            order = [0, 1] if t % 2 == 0 else [1, 0]
            feats = torch.tensor([[float(10 + t)], [float(20 + t)]])[order]
            obb = torch.tensor([
                [0.1, 0.1, 0.1, 0.1, 0.0], [0.9, 0.9, 0.1, 0.1, 0.0],
            ])[order]
            tids = [0, 1] if t % 2 == 0 else [1, 0]
            frames.append(FrameDetections(feats, obb, tids))

        batch = build_track_aligned_history([frames], window=T, feat_dim=1)
        assert batch.num_tracks == 2
        row = batch.track_ids.index(0)
        # track 0 の履歴は 10, 11, 12 と一貫していなければならない
        assert batch.features[row, :, 0].tolist() == [10.0, 11.0, 12.0]

    def test_batch_index_separates_sequences(self):
        f = lambda v: FrameDetections(                        # noqa: E731
            torch.tensor([[v]]), torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0]]), [0]
        )
        batch = build_track_aligned_history([[f(1.0)], [f(2.0)]], window=1, feat_dim=1)
        assert batch.num_tracks == 2
        assert batch.batch_index.tolist() == [0, 1]

    def test_current_rows_map_back_to_detections(self):
        frames = [FrameDetections(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.0], [0.9, 0.9, 0.1, 0.1, 0.0]]),
            [5, 9],
        )]
        batch = build_track_aligned_history([frames], window=1, feat_dim=1)
        rows = batch.current_rows[0]
        assert len(rows) == 2
        assert batch.track_ids[rows[0]] == 5
        assert batch.track_ids[rows[1]] == 9

    def test_missing_track_in_some_frames(self):
        """フレーム t にしか居ない track も、他フレームは mask=False で埋まる。"""
        empty = FrameDetections(torch.zeros(0, 1), torch.zeros(0, 5), [])
        present = FrameDetections(
            torch.tensor([[7.0]]), torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0]]), [3]
        )
        batch = build_track_aligned_history([[empty, present, empty]], window=3, feat_dim=1)
        assert batch.num_tracks == 1
        assert batch.mask[0].tolist() == [False, True, False]

    def test_empty_input(self):
        batch = build_track_aligned_history([[]], window=4, feat_dim=2)
        assert batch.num_tracks == 0
        assert batch.features.shape == (0, 4, 2)

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="件数"):
            FrameDetections(torch.zeros(2, 1), torch.zeros(1, 5), [0, 1])
