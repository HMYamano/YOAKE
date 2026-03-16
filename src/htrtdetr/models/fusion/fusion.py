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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTemporalFusion(nn.Module):
    """
    Detector の query feature シーケンスを
    Hierarchical Temporal Module に渡すための adapter。

    - detector は毎フレーム (B, Q, D) の query feature を出力
    - 複数フレーム分を (B*Q, T, D) に変換して HTM に渡す
    - HTM の出力 (B*Q, T, D) を (B, T, Q, D) に戻す
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        # query feature の前処理 (optional projection)
        self.proj = nn.Identity()

    def pack_for_temporal(
        self,
        query_feats: List[torch.Tensor],  # List of (B, Q, D), length T
    ) -> torch.Tensor:
        """
        (B, Q, D) × T → (B*Q, T, D)
        """
        B, Q, D = query_feats[0].shape
        T = len(query_feats)
        # Stack: (B, T, Q, D)
        stacked = torch.stack(query_feats, dim=1)
        # → (B, Q, T, D) → (B*Q, T, D)
        return stacked.permute(0, 2, 1, 3).reshape(B * Q, T, D)

    def unpack_from_temporal(
        self,
        temporal_feats: torch.Tensor,  # (B*Q, T, out_dim)
        B: int,
        Q: int,
    ) -> torch.Tensor:
        """
        (B*Q, T, out_dim) → (B, T, Q, out_dim)
        """
        BQ, T, D = temporal_feats.shape
        # (B, Q, T, D) → (B, T, Q, D)
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
    ) -> List[Dict[str, torch.Tensor]]:
        """
        バッチ内の各画像について、score threshold を超えた検出を抽出する。

        Returns:
            List[Dict]: B 個の辞書
              - boxes: (N_i, 4)
              - scores: (N_i,)
              - class_ids: (N_i,)
              - features: (N_i, D)
              - query_indices: (N_i,)
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

            results.append({
                "boxes": pred_boxes[b][idx],           # (N_i, 4)
                "scores": scores[idx],                 # (N_i,)
                "class_ids": class_ids[idx],           # (N_i,)
                "features": query_feats[b][idx],       # (N_i, D)
                "query_indices": idx,                  # (N_i,)
            })

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
