"""
test_tracking_alignment.py — 検出順と track の対応が壊れないことの回帰テスト

ここで守る不変条件はひとつだけ:

    OnlineTracker.update に渡した per-detection 配列の行 i と、
    その検出に付いた track_id / 特徴履歴 / OBB 履歴は、常に同じ個体を指す。

これが崩れると tracking の表示は正常なまま TrackFeatureCache の中で個体履歴が
入れ替わり、temporal feature → behavior → identity まで静かに汚染される。
解析は最後まで正常終了するので、テストが無いと気付けない種類の壊れ方になる。

修正前の実装は track_id を「マッチ → re-ID → 新規」という**処理順**で append し、
features / obbs は検出順のまま渡していたため、この順序が食い違うケース
(= 新規個体が既存個体より前の index に来るケース) で必ず入れ替わっていた。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from htrtdetr.tracking import OBBAssociator, OnlineTracker, TrackFeatureCache


def _tracker(feat_dim: int = 1, *, app_w: float = 0.0, iou_w: float = 1.0,
             window: int = 8, max_age: int = 30, reid_max_age: int = 90,
             reid_threshold: float = 0.9, enable_reid: bool = True) -> OnlineTracker:
    return OnlineTracker(
        OBBAssociator(
            iou_weight=iou_w, appearance_weight=app_w, match_threshold=0.5,
        ),
        TrackFeatureCache(
            window=window, feat_dim=feat_dim,
            max_age=max_age, reid_max_age=reid_max_age,
        ),
        enable_reid=enable_reid,
        reid_threshold=reid_threshold,
    )


def _obb(cx: float, cy: float) -> list:
    return [cx, cy, 0.1, 0.1, 0.0]


def _assert_cache_matches_detections(tracker, tracked, feats, obbs):
    """検出 i に付いた track の履歴末尾が、検出 i の feature / OBB であること。"""
    for obj in tracked:
        state = tracker.cache.get(obj.track_id)
        assert state is not None, f"track {obj.track_id} が cache に無い"
        i = obj.detection_index
        assert torch.allclose(state.last_feature(), feats[i]), (
            f"track {obj.track_id} の特徴が検出 {i} のものと違う "
            f"(got {state.last_feature().tolist()}, want {feats[i].tolist()})"
        )
        assert torch.allclose(state.last_obb(), obbs[i]), (
            f"track {obj.track_id} の OBB が検出 {i} のものと違う"
        )


class TestDetectionOrderAlignment:
    """P0-2 回帰: track_id 配列を処理順で組むと検出順とずれる。"""

    def test_case1_existing_then_new(self):
        """Detection 0 = 既存 track, Detection 1 = 新規。"""
        tr = _tracker()
        tr.update(0, torch.tensor([_obb(0.2, 0.2)]), torch.tensor([[20.0]]))

        obbs = torch.tensor([_obb(0.2, 0.2), _obb(0.8, 0.8)])
        feats = torch.tensor([[20.0], [80.0]])
        tracked = tr.update(1, obbs, feats)

        by_det = {o.detection_index: o for o in tracked}
        assert by_det[0].track_id == 0 and not by_det[0].is_new
        assert by_det[1].is_new
        _assert_cache_matches_detections(tr, tracked, feats, obbs)

    def test_case2_new_then_existing(self):
        """
        Detection 0 = 新規, Detection 1 = 既存 track。

        修正前はここで track 0 に検出 0 (別個体) の特徴が入っていた。
        """
        tr = _tracker()
        tr.update(0, torch.tensor([_obb(0.8, 0.8)]), torch.tensor([[80.0]]))

        obbs = torch.tensor([_obb(0.2, 0.2), _obb(0.8, 0.8)])
        feats = torch.tensor([[20.0], [80.0]])
        tracked = tr.update(1, obbs, feats)

        by_det = {o.detection_index: o for o in tracked}
        assert by_det[0].is_new, "検出 0 は新規個体のはず"
        assert by_det[1].track_id == 0, "検出 1 は既存 track 0 のはず"

        assert tr.cache.get(0).last_feature().item() == 80.0
        assert tr.cache.get(by_det[0].track_id).last_feature().item() == 20.0
        _assert_cache_matches_detections(tr, tracked, feats, obbs)

    def test_case3_matched_reid_and_new_in_one_frame(self):
        """matched + re-ID + 新規が 1 フレームに共存しても対応が崩れない。"""
        emb_keep = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
        emb_lost = F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
        emb_new = F.normalize(torch.tensor([[0.0, 0.0, 1.0]]), dim=-1)

        tr = _tracker(feat_dim=1, max_age=1, reid_max_age=20, reid_threshold=0.9)

        # frame 0: track A (0.2) と track B (0.8) を作る
        tr.update(
            0,
            torch.tensor([_obb(0.2, 0.2), _obb(0.8, 0.8)]),
            torch.tensor([[20.0], [80.0]]),
            det_embed=torch.cat([emb_keep, emb_lost]),
        )
        assert tr.cache.active_ids() == [0, 1]

        # frame 1-2: B を見失わせる (A だけ観測 → B が re-ID 候補へ落ちる)
        for t in (1, 2):
            tr.update(t, torch.tensor([_obb(0.2, 0.2)]), torch.tensor([[20.0]]),
                      det_embed=emb_keep)
        assert 1 in tr.cache.reid_ids(), "track 1 が re-ID 候補になっていない"

        # frame 3: 検出順を「新規 → re-ID → matched」にして最悪ケースを作る
        obbs = torch.tensor([_obb(0.5, 0.1), _obb(0.9, 0.9), _obb(0.2, 0.2)])
        feats = torch.tensor([[50.0], [80.0], [20.0]])
        embs = torch.cat([emb_new, emb_lost, emb_keep])
        tracked = tr.update(3, obbs, feats, det_embed=embs)

        by_det = {o.detection_index: o for o in tracked}
        assert by_det[0].is_new, "検出 0 は新規のはず"
        assert by_det[1].is_reid and by_det[1].track_id == 1, "検出 1 は track 1 の復元"
        assert by_det[2].track_id == 0 and not by_det[2].is_new, "検出 2 は track 0"

        _assert_cache_matches_detections(tr, tracked, feats, obbs)

    def test_track_history_stays_with_one_individual(self):
        """
        検出順を毎フレーム入れ替えても、各 track の履歴は一貫した個体のまま。
        """
        tr = _tracker()
        # 個体 A は特徴 10 番台、個体 B は 20 番台
        for t in range(6):
            flip = t % 2 == 1
            a_obb, b_obb = _obb(0.2, 0.2), _obb(0.8, 0.8)
            a_feat, b_feat = [10.0 + t], [20.0 + t]
            if flip:
                obbs = torch.tensor([b_obb, a_obb])
                feats = torch.tensor([b_feat, a_feat])
            else:
                obbs = torch.tensor([a_obb, b_obb])
                feats = torch.tensor([a_feat, b_feat])
            tracked = tr.update(t, obbs, feats)
            _assert_cache_matches_detections(tr, tracked, feats, obbs)

        hist, _, mask, ids = tr.cache.stacked_history()
        for row, tid in enumerate(ids):
            values = hist[row][mask[row]][:, 0].tolist()
            decades = {int(v) // 10 for v in values}
            assert len(decades) == 1, (
                f"track {tid} の履歴に別個体の特徴が混ざった: {values}"
            )

    def test_every_detection_receives_a_track_id(self):
        """未割当の検出を黙って捨てない (件数が合わなければ例外)。"""
        tr = _tracker()
        obbs = torch.tensor([_obb(0.2, 0.2), _obb(0.8, 0.8)])
        tracked = tr.update(0, obbs, torch.tensor([[1.0], [2.0]]))
        assert sorted(o.detection_index for o in tracked) == [0, 1]
        assert len({o.track_id for o in tracked}) == 2


class TestVelocityUsesRealFrameIndex:
    """P1-5 回帰: リングバッファ上の位置差ではなく実フレーム番号で dt を取る。"""

    def _cache(self, window: int = 8) -> TrackFeatureCache:
        return TrackFeatureCache(window=window, feat_dim=1)

    def _push(self, cache, frame: int, x: float) -> None:
        cache.update(
            frame, [0], torch.ones(1, 1),
            torch.tensor([[x, 0.0, 0.1, 0.1, 0.0]]),
        )

    def test_consecutive_frames(self):
        c = self._cache()
        self._push(c, 0, 0.0)
        self._push(c, 1, 0.1)
        assert c.velocities([0])[0, 0].item() == pytest.approx(0.1, abs=1e-6)

    def test_frame_gap_of_ten(self):
        """frame 0 → frame 10 で 1.0 動いたなら速度は 0.1 / frame。"""
        c = self._cache()
        self._push(c, 0, 0.0)
        self._push(c, 10, 1.0)
        assert c.velocities([0])[0, 0].item() == pytest.approx(0.1, abs=1e-6)

    def test_irregular_intervals(self):
        c = self._cache()
        self._push(c, 0, 0.0)
        self._push(c, 3, 0.3)
        self._push(c, 8, 0.8)   # 直近 2 点は frame 3→8 で 0.5 動いた
        assert c.velocities([0])[0, 0].item() == pytest.approx(0.1, abs=1e-6)

    def test_ring_buffer_wraparound(self):
        """バッファを一周しても、実フレーム番号から正しい dt が取れる。"""
        c = self._cache(window=3)
        for f in range(0, 25, 5):      # 0, 5, 10, 15, 20
            self._push(c, f, 0.02 * f)
        assert c.velocities([0])[0, 0].item() == pytest.approx(0.02, abs=1e-6)

    def test_missing_frames_are_skipped_not_counted_as_observations(self):
        c = self._cache()
        self._push(c, 0, 0.0)
        c.update(1, [], torch.zeros(0, 1), torch.zeros(0, 5))  # 欠損
        c.update(2, [], torch.zeros(0, 1), torch.zeros(0, 5))  # 欠損
        self._push(c, 3, 0.3)
        assert c.velocities([0])[0, 0].item() == pytest.approx(0.1, abs=1e-6)

    def test_duplicate_frame_index_gives_zero_not_infinity(self):
        """dt <= 0 は速度ゼロ (0 除算や負の速度で association を壊さない)。"""
        c = self._cache()
        cache_state_pushes = [(0, 0.0), (0, 0.5)]
        for frame, x in cache_state_pushes:
            state = c.get(0)
            if state is None:
                self._push(c, frame, x)
            else:
                state.push(
                    torch.ones(1), torch.tensor([x, 0.0, 0.1, 0.1, 0.0]), frame
                )
        assert c.velocities([0])[0].tolist() == [0.0, 0.0]

    def test_single_observation_is_zero(self):
        c = self._cache()
        self._push(c, 4, 0.5)
        assert c.velocities([0])[0].tolist() == [0.0, 0.0]


class TestCacheRejectsDuplicateTrackIds:
    """1 フレームに同じ track_id が 2 回来ると時間軸がずれるので拒否する。"""

    def test_duplicate_ids_raise(self):
        c = TrackFeatureCache(window=4, feat_dim=1)
        with pytest.raises(ValueError, match="重複"):
            c.update(0, [3, 3], torch.ones(2, 1), torch.zeros(2, 5))
