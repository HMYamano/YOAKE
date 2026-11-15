"""
branch.py — BehaviorBranch: 行動認識ブランチ全体

個体検出 (tight OBB) と tracking (track_id) の結果を受け取り、行動を推定する。

    track-aligned history + backbone feature
        ├─ actor    : OBB に沿った回転 crop      (細粒度 appearance / 姿勢)
        ├─ context  : 拡大 AABB crop             (他個体 / 周辺状況)
        ├─ detector : query 特徴                 (検出器が既に抽出した情報)
        ├─ motion   : 速度 / 加速度 / 経路        (kinematics)
        ├─ orient   : 体軸と移動方向の関係
        ├─ interact : 最近傍距離 / 相対速度 / facing
        ├─ temporal : HTM の時間特徴
        ├─ clip     : context box をクリップ全体に固定した tube (tracking 非依存, 任意)
        └─ pose     : キーポイント (任意)
                ↓
        EvidenceBehaviorHead  → 行動 + 寄与分解

``use_*`` フラグはどれも「その group を作らない」ことを意味する (ゼロ埋めではない)。
したがって ablation で切ると本当にパラメータと計算が消え、公平な比較になる。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .clip import ClipTubeEncoder
from .evidence_head import BehaviorPrediction, EvidenceBehaviorHead
from .kinematics import KinematicsEncoder
from .roi import ContextROIExtractor, RotatedROIExtractor

__all__ = ["BehaviorBranch"]


class BehaviorBranch(nn.Module):
    """
    行動認識ブランチ。

    Args:
        cfg: ``BehaviorConfig``
        num_actions: 行動クラス数
        feature_channels: backbone / FPN 特徴のチャネル数
        detector_dim: detector の query 特徴次元
        temporal_dim: HTM 出力次元
        num_keypoints: pose を使う場合のキーポイント数
        action_names: 行動名 (hierarchical の葉の対応づけに使う)
        explain_cfg: ``ExplainabilityConfig``
    """

    def __init__(
        self,
        cfg,
        num_actions: int,
        feature_channels: int = 256,
        detector_dim: int = 256,
        temporal_dim: int = 256,
        num_keypoints: int = 0,
        action_names: Optional[List[str]] = None,
        explain_cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_actions = int(num_actions)
        self.feature_channels = int(feature_channels)

        d = int(cfg.feature_dim)
        use_evidence = True
        use_gating = True
        if explain_cfg is not None:
            use_evidence = bool(explain_cfg.use_evidence_decomposition)
            use_gating = bool(explain_cfg.use_gating)
        self.return_gates = bool(explain_cfg.return_gates) if explain_cfg else True

        group_dims: Dict[str, int] = {}

        # --- Actor stream (回転 crop) ---
        if cfg.use_actor:
            self.actor_roi = RotatedROIExtractor(
                output_size=cfg.actor_roi_size,
                orientation_normalize=cfg.use_orientation,
            )
            self.actor_encoder = _make_roi_encoder(
                feature_channels, cfg.actor_roi_size, d
            )
            group_dims["actor"] = d

        # --- Context stream (拡大 AABB crop) ---
        if cfg.use_context:
            self.context_roi = ContextROIExtractor(
                output_size=cfg.context_roi_size,
                context_scale=cfg.context_scale,
                square=cfg.context_square,
            )
            self.context_encoder = _make_roi_encoder(
                feature_channels, cfg.context_roi_size, d
            )
            group_dims["context"] = d

        # --- Detector query 特徴 ---
        if cfg.use_detector_feat:
            self.detector_proj = nn.Sequential(
                nn.Linear(detector_dim, d), nn.LayerNorm(d), nn.ReLU(inplace=True)
            )
            group_dims["detector"] = d

        # --- Kinematics (motion / orientation / interaction) ---
        self.kinematics = KinematicsEncoder(
            output_dim=d,
            use_motion=cfg.use_motion,
            use_orientation=cfg.use_orientation,
            use_interaction=cfg.use_interaction,
            dropout=cfg.dropout,
        )
        for name, flag in (
            ("motion", cfg.use_motion),
            ("orientation", cfg.use_orientation),
            ("interaction", cfg.use_interaction),
        ):
            if flag:
                group_dims[name] = d

        # --- Temporal (HTM 出力) ---
        if cfg.use_temporal:
            self.temporal_proj = nn.Sequential(
                nn.Linear(temporal_dim, d), nn.LayerNorm(d), nn.ReLU(inplace=True)
            )
            group_dims["temporal"] = d

        # --- Clip tube (YOWO の 3D 分岐に相当。None は「自動」= ここでは無効) ---
        self.use_clip = bool(getattr(cfg, "use_clip", False))
        if self.use_clip:
            self.clip_encoder = ClipTubeEncoder(
                in_channels=feature_channels,
                out_dim=d,
                reduced_channels=int(getattr(cfg, "clip_channels", 64)),
                roi_size=int(getattr(cfg, "clip_roi_size", 7)),
                context_scale=cfg.context_scale,
                square=cfg.context_square,
            )
            group_dims["clip"] = d

        # --- Pose (任意) ---
        if cfg.use_pose and num_keypoints > 0:
            self.pose_proj = nn.Sequential(
                nn.Linear(num_keypoints * 3, d), nn.LayerNorm(d), nn.ReLU(inplace=True)
            )
            group_dims["pose"] = d

        if not group_dims:
            raise ValueError(
                "BehaviorBranch: 有効な feature group がありません。"
                "behavior.use_* のいずれかを true にしてください。"
            )
        self.group_dims = group_dims

        self.head = EvidenceBehaviorHead(
            group_dims=group_dims,
            num_actions=num_actions,
            hidden_dim=d,
            classifier_type=cfg.classifier_type,
            use_evidence=use_evidence,
            use_gating=use_gating,
            dropout=cfg.dropout,
            hierarchy=cfg.hierarchy or None,
            heads=cfg.heads or None,
            action_names=action_names,
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obb_history: torch.Tensor,                    # (N, T, 5) 正規化座標
        history_mask: torch.Tensor,                   # (N, T) bool
        spatial_features: Optional[torch.Tensor] = None,   # (B, C, H, W)
        batch_index: Optional[torch.Tensor] = None,        # (N,)
        detector_features: Optional[torch.Tensor] = None,  # (N, detector_dim)
        temporal_features: Optional[torch.Tensor] = None,  # (N, temporal_dim)
        keypoints: Optional[torch.Tensor] = None,          # (N, K, 3)
        clip_frames: Optional[List[torch.Tensor]] = None,  # T x (B, c, H, W)
    ) -> BehaviorPrediction:
        """
        Args:
            obb_history: track ごとの OBB 履歴 (最終フレームが現在)
            history_mask: True が実観測のフレーム
            spatial_features: actor / context crop を取る特徴マップ
            batch_index: 各 track がどのバッチ要素に属するか
            clip_frames: clip tube 用に ``clip_encoder.reduce_frame`` を通した
                各フレームの特徴マップ (古い → 新しい、最後がキーフレーム)
        Returns:
            BehaviorPrediction
        """
        N = obb_history.shape[0]
        device = obb_history.device
        current_obb = obb_history[:, -1]                    # (N, 5)

        if batch_index is None:
            batch_index = torch.zeros(N, dtype=torch.long, device=device)

        features: Dict[str, torch.Tensor] = {}

        if self.cfg.use_actor:
            if spatial_features is None:
                raise ValueError(
                    "behavior.use_actor=true には spatial_features が必要です"
                )
            crop = self.actor_roi(spatial_features, current_obb, batch_index)
            features["actor"] = self.actor_encoder(crop)

        if self.cfg.use_context:
            if spatial_features is None:
                raise ValueError(
                    "behavior.use_context=true には spatial_features が必要です"
                )
            crop = self.context_roi(spatial_features, current_obb, batch_index)
            features["context"] = self.context_encoder(crop)

        if self.cfg.use_detector_feat:
            if detector_features is None:
                raise ValueError(
                    "behavior.use_detector_feat=true には detector_features が必要です"
                )
            features["detector"] = self.detector_proj(detector_features)

        kin = self.kinematics(obb_history, history_mask, group_index=batch_index)
        features.update(kin)

        if self.cfg.use_temporal:
            if temporal_features is None:
                raise ValueError(
                    "behavior.use_temporal=true には temporal_features が必要です"
                )
            features["temporal"] = self.temporal_proj(temporal_features)

        if self.use_clip:
            if not clip_frames:
                raise ValueError(
                    "behavior.use_clip=true には clip_frames (各フレームの縮約特徴) が必要です"
                )
            features["clip"] = self.clip_encoder(clip_frames, current_obb, batch_index)

        if "pose" in self.group_dims:
            if keypoints is None:
                raise ValueError("behavior.use_pose=true には keypoints が必要です")
            features["pose"] = self.pose_proj(keypoints.flatten(1))

        return self.head(features)

    # ------------------------------------------------------------------ #
    # 付随情報
    # ------------------------------------------------------------------ #

    def context_boxes(self, obb: torch.Tensor) -> Optional[torch.Tensor]:
        """GUI 表示用に、実際に使った context AABB を返す。"""
        if not self.cfg.use_context:
            return None
        return self.context_roi.context_boxes(obb)

    def active_groups(self) -> List[str]:
        """有効な feature group 名 (ablation 設定の確認用)。"""
        return list(self.group_dims.keys())


def _num_groups(channels: int, preferred: int = 32) -> int:
    """
    GroupNorm の num_groups を選ぶ。

    num_groups は num_channels を割り切る必要があるので、preferred 以下で
    最大の約数を返す (任意の feature_dim を設定できるようにするため)。
    """
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _make_roi_encoder(in_channels: int, roi_size: int, out_dim: int) -> nn.Module:
    """
    ROI crop (C, S, S) → (out_dim,) の符号化器。

    軽量な 2 層 conv で空間構造を潰しすぎずに集約し、平均プーリングで
    ROI サイズに依存しない固定長にする。
    """
    g = _num_groups(out_dim)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
    )
