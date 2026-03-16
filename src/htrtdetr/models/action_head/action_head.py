"""
action_head.py — Action Classification Head

設計方針:
- bbox feature + temporal hierarchy feature + interaction feature から行動を分類
- frame-wise classification がベース (segment-level は将来拡張)
- 入力の各ソースは optional にして ablation しやすくする
- focal loss 対応のため logits を返す (softmax は外部で)

入力:
- spatial_feat: bbox query feature (B*N, feature_dim) — detector の query output
- temporal_feat: HTM の出力 (B*N, temporal_dim)
- interaction_feat: 個体間 interaction feature (B*N, interaction_dim)

出力:
- action_logits: (B*N, num_actions) — 各行動の logit
- action_probs: (B*N, num_actions) — softmax 済み確率
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config.config import ActionHeadConfig


# ---------------------------------------------------------------------------
# Feature Gating (optional: どの特徴を使うか動的に調整)
# ---------------------------------------------------------------------------

class FeatureGating(nn.Module):
    """
    複数の feature ソースをゲーティングで重み付けする。
    ablation 時に特定の feature を無効化したときの補償として機能する。
    """

    def __init__(self, num_sources: int, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_sources, num_sources),
            nn.Sigmoid(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List[(N, D)] — 各 feature ソース
        Returns:
            gated_sum: (N, D)
        """
        concat = torch.cat(features, dim=-1)
        weights = self.gate(concat)  # (N, num_sources)
        stacked = torch.stack(features, dim=1)  # (N, num_sources, D)
        out = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (N, D)
        return out


# ---------------------------------------------------------------------------
# Action Head (メインクラス)
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """
    行動分類ヘッド。

    アーキテクチャ:
      [spatial_feat, temporal_feat, interaction_feat]
        → concat (or gating)
        → MLP (3 層)
        → action_logits
    """

    def __init__(self, cfg: ActionHeadConfig):
        super().__init__()
        self.cfg = cfg

        # 入力次元の計算
        in_dim = cfg.feature_dim
        if cfg.use_interaction:
            in_dim += cfg.interaction_dim
        # temporal feature は常に使う (HTM の出力)
        in_dim += cfg.temporal_dim

        # 入力 projection (次元を揃える)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dims[0]),
            nn.LayerNorm(cfg.hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # MLP 本体
        layers = []
        prev_dim = cfg.hidden_dims[0]
        for h_dim in cfg.hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            ])
            prev_dim = h_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # 分類器
        self.classifier = nn.Linear(prev_dim, cfg.num_actions)

        # 将来のセグメントレベル拡張のためのテンポラルアグリゲータ (placeholder)
        # 現状は identity (frame-wise)
        self.temporal_aggregator = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """分類器の bias を uniform prior で初期化"""
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        spatial_feat: torch.Tensor,                          # (N, feature_dim)
        temporal_feat: torch.Tensor,                         # (N, temporal_dim)
        interaction_feat: Optional[torch.Tensor] = None,     # (N, interaction_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spatial_feat: detector の query feature
            temporal_feat: HTM の temporal feature
            interaction_feat: 個体間 interaction feature (optional)

        Returns:
            action_logits: (N, num_actions)
            action_probs: (N, num_actions)
        """
        parts = [spatial_feat, temporal_feat]

        if self.cfg.use_interaction and interaction_feat is not None:
            parts.append(interaction_feat)
        elif self.cfg.use_interaction:
            # interaction feature がない場合はゼロで補完
            zero_int = torch.zeros(
                spatial_feat.shape[0], self.cfg.interaction_dim,
                device=spatial_feat.device, dtype=spatial_feat.dtype
            )
            parts.append(zero_int)

        x = torch.cat(parts, dim=-1)  # (N, in_dim)

        # MLP
        x = self.input_proj(x)
        x = self.mlp(x)

        # 将来のセグメントアグリゲーション (現状は identity)
        x = self.temporal_aggregator(x)

        # 分類
        logits = self.classifier(x)  # (N, num_actions)
        probs = F.softmax(logits, dim=-1)

        return {
            "action_logits": logits,
            "action_probs": probs,
        }

    def forward_sequence(
        self,
        spatial_feats: torch.Tensor,                          # (B, T, N, feature_dim)
        temporal_feats: torch.Tensor,                         # (B, T, N, temporal_dim)
        interaction_feats: Optional[torch.Tensor] = None,     # (B, T, N, interaction_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        シーケンス全体 (B, T, N) を一括処理する版。
        実際には (B*T*N, ...) に reshape して forward() を呼び出す。

        Returns:
            action_logits: (B, T, N, num_actions)
            action_probs:  (B, T, N, num_actions)
        """
        B, T, N, D = spatial_feats.shape

        sf = spatial_feats.reshape(B * T * N, -1)
        tf = temporal_feats.reshape(B * T * N, -1)
        intf = None
        if interaction_feats is not None:
            intf = interaction_feats.reshape(B * T * N, -1)

        out = self.forward(sf, tf, intf)

        logits = out["action_logits"].reshape(B, T, N, -1)
        probs = out["action_probs"].reshape(B, T, N, -1)

        return {"action_logits": logits, "action_probs": probs}


# ---------------------------------------------------------------------------
# Interaction Feature Computer
# ---------------------------------------------------------------------------

class InteractionFeatureComputer(nn.Module):
    """
    frame 内の個体間 interaction feature を計算する。
    ActionHead の外部から呼び出して interaction_feat を作成する。

    Config の use_* フラグに応じて使う特徴を選択する。
    """

    def __init__(self, cfg: ActionHeadConfig):
        super().__init__()
        self.cfg = cfg

        # 使う特徴の次元を計算
        raw_dim = 0
        if cfg.use_nn_distance:
            raw_dim += 1
        if cfg.use_relative_angle:
            raw_dim += 1
        if cfg.use_relative_velocity:
            raw_dim += 2
        if cfg.use_overlap:
            raw_dim += 1

        if raw_dim == 0:
            raw_dim = 1  # 最低でも 1 次元

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, cfg.interaction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.interaction_dim, cfg.interaction_dim),
        )

    def forward(
        self,
        bboxes: torch.Tensor,        # (N, 4) [cx, cy, w, h] normalized
        velocities: torch.Tensor,    # (N, 2) [dx, dy]
    ) -> torch.Tensor:
        """
        Returns: (N, interaction_dim)
        """
        N = bboxes.shape[0]
        device = bboxes.device

        if N == 0:
            return torch.zeros(0, self.cfg.interaction_dim, device=device)

        parts = []
        centers = bboxes[:, :2]  # (N, 2)

        if self.cfg.use_nn_distance or self.cfg.use_relative_angle or self.cfg.use_overlap:
            if N == 1:
                dists = torch.zeros(1, device=device)
                nn_idx = torch.zeros(1, dtype=torch.long, device=device)
            else:
                dist_mat = torch.cdist(centers, centers)  # (N, N)
                dist_mat.fill_diagonal_(float("inf"))
                dists, nn_idx = dist_mat.min(dim=1)

        if self.cfg.use_nn_distance:
            parts.append(dists.unsqueeze(-1))

        if self.cfg.use_relative_angle:
            if N == 1:
                angle = torch.zeros(1, device=device)
            else:
                nn_centers = centers[nn_idx]
                delta = nn_centers - centers
                angle = torch.atan2(delta[:, 1], delta[:, 0])
            parts.append(angle.unsqueeze(-1))

        if self.cfg.use_relative_velocity:
            if N == 1:
                rel_vel = torch.zeros(1, 2, device=device)
            else:
                rel_vel = velocities - velocities[nn_idx]
            parts.append(rel_vel)

        if self.cfg.use_overlap:
            if N == 1:
                iou_vals = torch.zeros(1, device=device)
            else:
                from ...utils.misc import box_iou, cxcywh_to_xyxy
                boxes_xyxy = cxcywh_to_xyxy(bboxes)
                nn_boxes = boxes_xyxy[nn_idx]
                # 各検出と最近傍の IoU
                iou_mat = box_iou(boxes_xyxy, nn_boxes)  # (N, N)
                iou_vals = iou_mat.diagonal()
            parts.append(iou_vals.unsqueeze(-1))

        if not parts:
            parts.append(torch.zeros(N, 1, device=device))

        x = torch.cat(parts, dim=-1)  # (N, raw_dim)
        x = x.nan_to_num(0.0).clamp(-10.0, 10.0)
        return self.proj(x)  # (N, interaction_dim)
