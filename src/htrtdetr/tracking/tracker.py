"""
tracker.py — OnlineTracker: association + ring buffer を束ねたオンライン追跡器

streaming 推論の 1 フレーム分の流れ:

    detections (OBB + embedding)
        ↓  OBBAssociator          既存 track との最適割当
        ↓  未マッチ検出 → re-ID    消失中 track の特徴と照合して ID を復元
        ↓  それでも未マッチ → 新規 track
        ↓  TrackFeatureCache      track ごとのリングバッファへ push
    track_id 付き detections + track 履歴

``MemoryIDHead`` (GRU メモリ) を置き換えるものではなく、**その外側の幾何的な追跡**
を担う。GRU による appearance embedding を ``det_embed`` として渡せば両者は協調し、
embedding を渡さなければ純粋な幾何追跡として単体で動く。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .associator import OBBAssociator
from .track_buffer import TrackFeatureCache

__all__ = ["OnlineTracker", "TrackedObject"]


@dataclass
class TrackedObject:
    """1 フレーム分の追跡結果 (検出 1 件に対応)。"""
    detection_index: int
    track_id: int
    obb: torch.Tensor          # (5,) [cx, cy, w, h, theta]
    score: float
    is_new: bool = False       # 新規に採番された ID か
    is_reid: bool = False      # 消失していた track を復元したか
    match_score: float = 0.0   # association の類似度


class OnlineTracker:
    """
    OBB 対応のオンライン追跡器。

    Args:
        associator: 関連付け器
        cache: track 特徴のリングバッファ
        enable_reid: 消失 track の再同定を行うか
        reid_threshold: 再同定成立に必要な類似度。通常の match_threshold より
            厳しめにする (誤った ID 復元は ID switch より有害なため)。
    """

    def __init__(
        self,
        associator: OBBAssociator,
        cache: TrackFeatureCache,
        enable_reid: bool = True,
        reid_threshold: Optional[float] = None,
    ):
        self.associator = associator
        self.cache = cache
        self.enable_reid = bool(enable_reid)
        self.reid_threshold = (
            float(reid_threshold) if reid_threshold is not None
            else min(0.95, associator.match_threshold + 0.15)
        )
        self._next_id = 0
        self._frame_index = -1
        # track_id -> 最後の appearance embedding (re-ID 用)
        self._embeddings: Dict[int, torch.Tensor] = {}

    @classmethod
    def from_config(cls, cfg, feat_dim: int, window: int,
                    device: Optional[torch.device] = None) -> "OnlineTracker":
        """``TrackingConfig`` から一式を組み立てる。"""
        return cls(
            associator=OBBAssociator.from_config(cfg),
            cache=TrackFeatureCache(
                window=window, feat_dim=feat_dim,
                max_age=cfg.max_age, reid_max_age=cfg.reid_max_age,
                device=device,
            ),
            enable_reid=cfg.enable_reid,
        )

    def reset(self) -> None:
        """新しい動画の開始時に状態を捨てる。"""
        self.cache.reset()
        self._embeddings.clear()
        self._next_id = 0
        self._frame_index = -1

    # ------------------------------------------------------------------ #
    # 1 フレームの更新
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_index: int,
        det_obb: torch.Tensor,                     # (D, 5)
        det_features: torch.Tensor,                # (D, feat_dim)
        det_scores: Optional[torch.Tensor] = None,  # (D,)
        det_embed: Optional[torch.Tensor] = None,   # (D, E) L2 正規化済み
        has_obb: bool = True,
    ) -> List[TrackedObject]:
        """
        1 フレーム分の検出を取り込み、track_id を割り当てる。

        Returns:
            検出順に並んだ TrackedObject のリスト
        """
        D = int(det_obb.shape[0])
        if det_scores is None:
            det_scores = torch.ones(D, device=det_obb.device)

        results: List[Optional[TrackedObject]] = [None] * D
        assigned_ids: List[int] = []

        active = self.cache.active_ids()
        unmatched = list(range(D))

        # --- 1. アクティブ track との関連付け ---
        if active and D > 0:
            track_obb = self._track_obb(active, det_obb)
            track_embed = self._track_embed(active, det_embed)
            track_vel = (
                self.cache.velocities(active)
                if self.associator.motion_weight > 0 else None
            )
            res = self.associator.associate(
                det_obb, track_obb,
                det_embed=det_embed, track_embed=track_embed,
                track_velocity=track_vel, has_obb=has_obb,
            )
            for (di, ti), score in zip(res.matches, res.scores):
                tid = active[ti]
                results[di] = TrackedObject(
                    detection_index=di, track_id=tid,
                    obb=det_obb[di], score=float(det_scores[di]),
                    match_score=score,
                )
                assigned_ids.append(tid)
            unmatched = res.unmatched_detections

        # --- 2. 消失 track との再同定 ---
        if self.enable_reid and unmatched:
            lost = self.cache.reid_ids()
            if lost:
                unmatched = self._try_reid(
                    unmatched, lost, det_obb, det_scores, det_embed,
                    has_obb, results, assigned_ids,
                )

        # --- 3. 残りは新規 track ---
        for di in unmatched:
            tid = self._next_id
            self._next_id += 1
            results[di] = TrackedObject(
                detection_index=di, track_id=tid,
                obb=det_obb[di], score=float(det_scores[di]), is_new=True,
            )
            assigned_ids.append(tid)

        # --- 4. リングバッファを更新 ---
        self.cache.update(
            frame_index=frame_index,
            track_ids=assigned_ids,
            features=det_features,
            obbs=det_obb,
        )
        if det_embed is not None:
            for obj in results:
                if obj is not None:
                    self._embeddings[obj.track_id] = det_embed[obj.detection_index].detach()

        self._frame_index = frame_index
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------ #
    # 内部ヘルパー
    # ------------------------------------------------------------------ #

    def _track_obb(self, ids: List[int], like: torch.Tensor) -> torch.Tensor:
        """指定 track の最後に観測された OBB を (T, 5) で返す。"""
        rows = []
        for tid in ids:
            state = self.cache.get(tid)
            obb = state.last_obb() if state is not None else None
            rows.append(obb if obb is not None else torch.zeros(5, device=like.device))
        return torch.stack(rows, dim=0).to(like.dtype)

    def _track_embed(
        self, ids: List[int], like: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """指定 track の最後の appearance embedding を (T, E) で返す。"""
        if like is None or not self._embeddings:
            return None
        E = like.shape[1]
        rows = [
            self._embeddings.get(tid, torch.zeros(E, device=like.device))
            for tid in ids
        ]
        return torch.stack(rows, dim=0).to(like.dtype)

    def _try_reid(
        self,
        unmatched: List[int],
        lost: List[int],
        det_obb: torch.Tensor,
        det_scores: torch.Tensor,
        det_embed: Optional[torch.Tensor],
        has_obb: bool,
        results: List[Optional[TrackedObject]],
        assigned_ids: List[int],
    ) -> List[int]:
        """
        未マッチ検出を消失 track と照合して ID を復元する。

        消失中は位置が信用できない (その間に動いている) ので、**appearance のみ**で
        判定する。embedding が無い構成では re-ID を行わない (幾何だけで復元すると
        別個体に同じ ID を与える危険が高い)。
        """
        if det_embed is None:
            return unmatched

        sub_idx = torch.tensor(unmatched, device=det_obb.device, dtype=torch.long)
        sub_embed = det_embed[sub_idx]
        lost_embed = self._track_embed(lost, det_embed)
        if lost_embed is None:
            return unmatched

        cos = torch.mm(sub_embed, lost_embed.t())
        sim = (cos + 1.0) * 0.5                       # [0, 1]

        still: List[int] = []
        used_lost: set = set()
        for k, di in enumerate(unmatched):
            best = int(sim[k].argmax())
            if float(sim[k, best]) >= self.reid_threshold and best not in used_lost:
                tid = lost[best]
                used_lost.add(best)
                results[di] = TrackedObject(
                    detection_index=di, track_id=tid,
                    obb=det_obb[di], score=float(det_scores[di]),
                    is_reid=True, match_score=float(sim[k, best]),
                )
                assigned_ids.append(tid)
            else:
                still.append(di)
        return still

    # ------------------------------------------------------------------ #
    # 読み出し
    # ------------------------------------------------------------------ #

    def track_history(self):
        """
        アクティブ track の履歴を (N, T, D) で返す (HTM への入力)。

        Returns:
            feats, obbs, mask, ids — ``TrackFeatureCache.stacked_history`` と同じ
        """
        return self.cache.stacked_history()

    def stats(self) -> Dict[str, int]:
        s = self.cache.stats()
        s["next_id"] = self._next_id
        return s
