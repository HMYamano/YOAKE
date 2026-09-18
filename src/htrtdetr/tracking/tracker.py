"""
tracker.py — OnlineTracker: association + ring buffer を束ねたオンライン追跡器

streaming 推論の 1 フレーム分の流れ:

    detections (OBB + embedding)
        ↓  OBBAssociator          既存 track との最適割当
        ↓  未マッチ検出 → re-ID    消失中 track の特徴と照合して ID を復元
        ↓  それでも未マッチ → 新規 track
        ↓  TrackFeatureCache      track ごとのリングバッファへ push
    track_id 付き detections + track 履歴

=== 責務の分担 (ID assignment authority) ===

``OnlineTracker`` が **final track ID の唯一の決定者**である。``MemoryIDHead`` は
その外側ではなく内側 — appearance / identity embedding の生成器 — として働き、
ID を決めない。streaming 推論の流れは:

    detector → MemoryIDHead.embed()  (identity embedding, ID は決めない)
             → OnlineTracker         (association → final track ID)
             → MemoryIDHead.commit_memory()  (確定した ID で GRU memory を更新)

``MemoryIDHead.forward_inference()`` は自前で ID を決めるが、それは tracker を
持たない構成 (``HTRTDETR._run_id_head`` / ``memory_id_stage``) 専用の経路である。
両方を同じフレームに適用して ID を二重に決めてはならない。

=== 配列の不変条件 ===

All per-detection arrays are aligned by original detection index:
``det_obb[i]`` / ``det_features[i]`` / ``det_scores[i]`` / ``det_embed[i]`` と、
このモジュールが組み立てる track_id 配列の要素 i は、常に同じ検出 i を指す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .associator import OBBAssociator, _optimal_assignment
from .track_buffer import TrackFeatureCache

__all__ = ["OnlineTracker", "TrackedObject"]


@dataclass
class TrackedObject:
    """
    1 フレーム分の追跡結果 (検出 1 件に対応)。

    ``detection_index`` は ``OnlineTracker.update`` に渡した ``det_obb`` /
    ``det_features`` / ``det_scores`` / ``det_embed`` の**行番号**である。
    この対応は追跡器の出力全体を通じて保たれる不変条件で、下流 (特徴キャッシュ・
    時間モデル・行動推定) が「どの行がどの個体か」を知る唯一の手がかりになる。
    """
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

        **不変条件**: 引数の per-detection 配列 (``det_obb`` / ``det_features`` /
        ``det_scores`` / ``det_embed``) はすべて行 i が同じ検出 i を指す。
        割り当てた track ID も ``track_ids_by_detection[i]`` という検出 index 順の
        配列で持ち回り、そのままリングバッファへ渡す。
        マッチ → re-ID → 新規という**処理順で ID を append すると、features /
        obbs の行順とずれて別個体の履歴を混ぜてしまう**ので、処理順ではなく
        検出 index で書き込むこと。

        Returns:
            検出 index の昇順に並んだ TrackedObject のリスト
        """
        D = int(det_obb.shape[0])
        if det_scores is None:
            det_scores = torch.ones(D, device=det_obb.device)

        results: List[Optional[TrackedObject]] = [None] * D
        # 検出 index -> track_id。-1 は「まだ未割当」。
        track_ids_by_detection: List[int] = [-1] * D

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
                track_ids_by_detection[di] = tid
            unmatched = res.unmatched_detections

        # --- 2. 消失 track との再同定 ---
        if self.enable_reid and unmatched:
            lost = self.cache.reid_ids()
            if lost:
                unmatched = self._try_reid(
                    unmatched, lost, det_obb, det_scores, det_embed,
                    has_obb, results, track_ids_by_detection,
                )

        # --- 3. 残りは新規 track ---
        for di in unmatched:
            tid = self._next_id
            self._next_id += 1
            results[di] = TrackedObject(
                detection_index=di, track_id=tid,
                obb=det_obb[di], score=float(det_scores[di]), is_new=True,
            )
            track_ids_by_detection[di] = tid

        # --- 4. リングバッファを更新 ---
        # track_ids / features / obbs はいずれも検出 index 順なので、
        # cache の行 i は常に同じ個体を指す。
        if D and min(track_ids_by_detection) < 0:
            missing = [i for i, t in enumerate(track_ids_by_detection) if t < 0]
            raise RuntimeError(
                f"検出 {missing} に track_id が割り当てられませんでした "
                "(tracking failure を無言で捨てないための不変条件チェック)"
            )
        self.cache.update(
            frame_index=frame_index,
            track_ids=track_ids_by_detection,
            features=det_features,
            obbs=det_obb,
        )
        # Appearance memory has the same lifetime as its track, including when
        # an empty frame expires the last re-identification candidate.
        self._embeddings = {
            tid: embedding for tid, embedding in self._embeddings.items()
            if tid in self.cache
        }
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
        track_ids_by_detection: List[int],
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

        # Resolve competing candidates jointly. A row-wise argmax discards
        # valid alternatives whenever an earlier detection claimed its best ID.
        # Penalize invalid pairs before assignment so they cannot displace a
        # valid match merely to improve the sum of similarities.
        cost = (1.0 - sim).masked_fill(
            sim < self.reid_threshold, float(min(sim.shape) + 1)
        )
        rows, cols = _optimal_assignment(cost)
        matched = set()
        for k, best in zip(rows, cols):
            score = float(sim[k, best])
            if score < self.reid_threshold:
                continue
            di, tid = unmatched[k], lost[best]
            results[di] = TrackedObject(
                detection_index=di, track_id=tid,
                obb=det_obb[di], score=float(det_scores[di]),
                is_reid=True, match_score=score,
            )
            track_ids_by_detection[di] = tid
            matched.add(di)
        return [di for di in unmatched if di not in matched]

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
