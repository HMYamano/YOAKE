"""
fusion.py — Feature Fusion Module

設計方針:
- Detector output / Temporal feature / ID embedding を統合する
- 各モジュールの出力次元を揃えて concat / attention で融合する
- 将来的に optical flow / graph feature を追加しやすい設計

主な役割:
- detector の query feature を temporal feature に渡す前に適切に reshape する
- temporal feature を action head と ID head に分配する
- 各 head への入力 feature を組み立てる
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackTemporalFusion(nn.Module):
    """
    **track 単位**で特徴履歴を束ね、Hierarchical Temporal Module に渡す adapter。

    === なぜ track 単位なのか ===

    DETR の query index はフレーム間で同一個体を指すことを**保証しない**。
    ある個体を frame t で query #17 が捉えても、frame t+1 では query #42 が
    捉えているかもしれない。したがって query index を時間方向に連結すると、
    「異なる個体の特徴を 1 本の時系列として HTM に食わせる」ことになり、
    速度・加速度・行動遷移といった時間特徴がすべて無意味になる。

    正しい整列軸は tracking が与える track_id:

        Detection → Tracking → track_id → track-aligned history → HTM

    欠損フレーム (その track が検出されなかったフレーム) は mask で明示し、
    無言のゼロ埋めや前フレーム複製で「動きが止まった」ように見せない。

    ``QueryTemporalFusion`` は旧挙動の互換ラッパとして残してある (非推奨)。
    """

    def __init__(self, feature_dim: int, missing_fill: str = "interpolate"):
        """
        Args:
            feature_dim: 特徴次元
            missing_fill: 欠損フレームの埋め方
                "interpolate" — 前後の有効フレームから線形補間 (既定)
                "zero"        — ゼロ埋め (mask で下流に伝える)
                "hold"        — 直前の有効フレームを保持
        """
        super().__init__()
        if missing_fill not in ("interpolate", "zero", "hold"):
            raise ValueError(
                f"missing_fill は 'interpolate' | 'zero' | 'hold': {missing_fill!r}"
            )
        self.feature_dim = feature_dim
        self.missing_fill = missing_fill
        # 将来の前処理拡張点 (現状は恒等)
        self.proj = nn.Identity()

    def pack_from_cache(
        self,
        history: torch.Tensor,   # (N, T, D) — track ごとの履歴
        mask: torch.Tensor,      # (N, T) bool — True が実観測
    ) -> torch.Tensor:
        """
        ``TrackFeatureCache.stacked_history()`` の出力を HTM 入力へ整える。

        Args:
            history: (N_tracks, T, D)
            mask: (N_tracks, T) — False の位置は未観測
        Returns:
            (N_tracks, T, D) — 欠損を missing_fill に従って埋めたもの
        """
        if history.numel() == 0:
            return history
        return self.proj(self._fill_missing(history, mask))

    def _fill_missing(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """欠損フレームを missing_fill 方針で埋める。"""
        if self.missing_fill == "zero":
            return x * mask.unsqueeze(-1).to(x.dtype)

        N, T, D = x.shape
        m = mask.to(x.dtype).unsqueeze(-1)            # (N, T, 1)

        # 直前の有効フレームの index を前方伝播で求める
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(N, T)
        valid_idx = torch.where(mask, idx, torch.full_like(idx, -1))
        prev_idx = torch.cummax(valid_idx, dim=1).values        # (N, T)

        held = torch.gather(
            x, 1, prev_idx.clamp(min=0).unsqueeze(-1).expand(N, T, D)
        )
        # 先頭が未観測 (prev が無い) 区間はゼロのまま
        has_prev = (prev_idx >= 0).unsqueeze(-1).to(x.dtype)
        held = held * has_prev

        if self.missing_fill == "hold":
            return x * m + held * (1 - m)

        # --- interpolate: 直後の有効フレームも取り、線形に混ぜる ---
        rev_valid = torch.where(
            mask.flip(1), idx, torch.full_like(idx, -1)
        )
        next_rev = torch.cummax(rev_valid, dim=1).values
        next_idx = (T - 1) - next_rev.flip(1)                   # (N, T)
        has_next = (next_rev.flip(1) >= 0).unsqueeze(-1).to(x.dtype)

        nxt = torch.gather(
            x, 1, next_idx.clamp(min=0, max=T - 1).unsqueeze(-1).expand(N, T, D)
        ) * has_next

        span = (next_idx - prev_idx).clamp(min=1).unsqueeze(-1).to(x.dtype)
        alpha = ((idx - prev_idx).unsqueeze(-1).to(x.dtype) / span).clamp(0.0, 1.0)

        both = has_prev * has_next
        interp = held * (1 - alpha) + nxt * alpha
        # 片側しか無い区間は存在する側を保持する
        filled = both * interp + (1 - both) * (held * has_prev + nxt * (1 - has_prev))
        return x * m + filled * (1 - m)

    @staticmethod
    def pack_windows(
        track_feats: List[torch.Tensor],  # List of (T, D)
    ) -> torch.Tensor:
        """学習時に track ごとの窓を積んで (N, T, D) にする補助。"""
        return torch.stack(track_feats, dim=0)

    @staticmethod
    def last_frame(temporal_feats: torch.Tensor) -> torch.Tensor:
        """(N, T, out_dim) → 最終フレーム (N, out_dim)。"""
        return temporal_feats[:, -1]


class QueryTemporalFusion(TrackTemporalFusion):
    """
    **非推奨** — DETR query index を時間方向に連結する旧 adapter。

    query index はフレーム間で同一個体を保証しないため、これで作った時系列は
    異なる個体の特徴が混ざりうる (詳細は ``TrackTemporalFusion`` の docstring)。
    後方互換 (旧 checkpoint のロード・旧 ablation の再現) のためだけに残している。
    新しいコードは ``TrackTemporalFusion`` を使うこと。

    ``ModelConfig.track_aligned_temporal=False`` を指定した場合のみ使われる。
    """

    def __init__(self, feature_dim: int):
        super().__init__(feature_dim, missing_fill="zero")

    def pack_for_temporal(
        self,
        query_feats: List[torch.Tensor],  # List of (B, Q, D), length T
    ) -> torch.Tensor:
        """(B, Q, D) × T → (B*Q, T, D)  — query index を時間軸に連結する。"""
        warnings.warn(
            "QueryTemporalFusion は非推奨です。DETR の query index はフレーム間で"
            "同一個体を保証しないため、時間特徴が別個体の混合になります。"
            "TrackTemporalFusion (track_aligned_temporal=True) を使ってください。",
            DeprecationWarning,
            stacklevel=2,
        )
        B, Q, D = query_feats[0].shape
        T = len(query_feats)
        stacked = torch.stack(query_feats, dim=1)          # (B, T, Q, D)
        return stacked.permute(0, 2, 1, 3).reshape(B * Q, T, D)

    def unpack_from_temporal(
        self,
        temporal_feats: torch.Tensor,  # (B*Q, T, out_dim)
        B: int,
        Q: int,
    ) -> torch.Tensor:
        """(B*Q, T, out_dim) → (B, T, Q, out_dim)"""
        BQ, T, D = temporal_feats.shape
        return temporal_feats.reshape(B, Q, T, D).permute(0, 2, 1, 3)


class DetectionFeatureAdapter(nn.Module):
    """
    Detector の query feature を各 head が期待する形式に変換する。

    - 検出結果と query feature の対応を管理
    - score threshold で有効な検出だけを抽出
    - bbox feature と query feature を結合
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def extract_valid_detections(
        self,
        pred_logits: torch.Tensor,   # (B, Q, num_classes+1)
        pred_boxes: torch.Tensor,    # (B, Q, 4) [cx, cy, w, h]
        query_feats: torch.Tensor,   # (B, Q, D)
        score_threshold: float = 0.3,
        pred_obb: Optional[torch.Tensor] = None,        # (B, Q, 5)
        pred_keypoints: Optional[torch.Tensor] = None,  # (B, Q, K, 3)
        pred_polarity: Optional[torch.Tensor] = None,   # (B, Q, 2)
    ) -> List[Dict[str, torch.Tensor]]:
        """
        バッチ内の各画像について、score threshold を超えた検出を抽出する。

        OBB / pose は渡されたときだけ結果 dict に追加されるので、AABB 構成の
        呼び出し側 (既存コード) は戻り値の形が変わらない。

        Returns:
            List[Dict]: B 個の辞書
              - boxes: (N_i, 4) [cx, cy, w, h] — OBB 有効時は外接矩形
              - scores: (N_i,)
              - class_ids: (N_i,)
              - features: (N_i, D)
              - query_indices: (N_i,) — **同一性ではなく由来の記録用**。
                フレーム間で同じ個体を指す保証は無いので、時間方向の整列には
                使わないこと (track_id を使う)。
              - obb: (N_i, 5) — pred_obb が渡された場合のみ
              - keypoints / polarity — 同上
        """
        B = pred_logits.shape[0]
        results = []

        for b in range(B):
            # background (最後のクラス) を除く最大クラスのスコア
            logits_b = pred_logits[b]  # (Q, C+1)
            probs = F.softmax(logits_b, dim=-1)
            # background を除いた最大スコア
            scores, class_ids = probs[:, :-1].max(dim=-1)  # (Q,)

            mask = scores > score_threshold  # (Q,)
            idx = mask.nonzero(as_tuple=False).squeeze(1)

            entry: Dict[str, torch.Tensor] = {
                "boxes": pred_boxes[b][idx],           # (N_i, 4)
                "scores": scores[idx],                 # (N_i,)
                "class_ids": class_ids[idx],           # (N_i,)
                "features": query_feats[b][idx],       # (N_i, D)
                "query_indices": idx,                  # (N_i,)
            }
            if pred_obb is not None:
                entry["obb"] = pred_obb[b][idx]        # (N_i, 5)
            if pred_keypoints is not None:
                entry["keypoints"] = pred_keypoints[b][idx]
            if pred_polarity is not None:
                entry["polarity"] = pred_polarity[b][idx]
            results.append(entry)

        return results


class MultiHeadFeatureRouter(nn.Module):
    """
    HTM の出力と detection feature を
    ID head / Action head に適切に配布する router。

    将来的に optical flow, graph feature を追加する場合も
    このクラスに入力を追加するだけで対応できる。
    """

    def __init__(
        self,
        feature_dim: int,
        temporal_dim: int,
        output_dim: int = 256,
    ):
        super().__init__()
        # ID head 用 feature projection
        self.id_proj = nn.Sequential(
            nn.Linear(feature_dim + temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
        # Action head 用 feature projection (spatial のみ)
        self.action_spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,    # (N, feature_dim)
        temporal_feat: torch.Tensor,   # (N, temporal_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            id_feat: (N, output_dim) — ID head 用
            action_spatial_feat: (N, output_dim) — Action head の spatial input
        """
        id_feat = self.id_proj(
            torch.cat([spatial_feat, temporal_feat], dim=-1)
        )
        action_spatial_feat = self.action_spatial_proj(spatial_feat)

        return {
            "id_feat": id_feat,
            "action_spatial_feat": action_spatial_feat,
        }
