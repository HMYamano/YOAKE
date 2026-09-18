"""
base.py — Detector 抽象基底クラス

設計方針:
- 具体的な detector (RT-DETR, YOLO 等) を交換できるよう、
  共通インタフェースを abc で定義する
- forward() は画像 tensor を受け取り、DetectionOutput を返す
- detector は内部でロスを計算しない (ロスは losses.py で一元管理)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 出力型定義
# ---------------------------------------------------------------------------

@dataclass
class DetectionOutput:
    """
    Detector の *学習/内部* 出力をまとめる dataclass (RT-DETR デコーダの生出力)。

    OBB / pose のフィールドはすべて Optional で、既定 (AABB detector) では None に
    なる。よって既存の消費側 (損失 / 評価 / 推論) は変更なしで動き続ける。

    Attributes:
        pred_logits: (B, num_queries, num_classes+1) — class logits
        pred_boxes:  (B, num_queries, 4) — [cx, cy, w, h] in [0, 1]
                     OBB 有効時も**外接矩形として常に埋まる** (AABB 互換の保証)。
        query_features: (B, num_queries, hidden_dim) — per-query features
        encoder_features: List[(B, C, H_i, W_i)] — multi-scale encoder outputs
        pred_angles: (B, num_queries, 2) — [sin(2t), cos(2t)] 未正規化。OBB 時のみ。
        pred_obb:    (B, num_queries, 5) — [cx, cy, w, h, theta] 正規化座標系。
                     w/h は**回転前の tight なサイズ**で、pred_boxes の外接矩形とは異なる。
        pred_keypoints: (B, num_queries, K, 3) — [x, y, logit] pose head 有効時のみ。
        pred_polarity:  (B, num_queries, 2) — head-tail 反転の 2 値 logit。
    """
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    query_features: torch.Tensor
    encoder_features: Optional[List[torch.Tensor]] = None
    aux_outputs: Optional[List[Dict]] = None  # 各中間デコーダ層の予測 (auxiliary loss 用)
    # --- OBB (detector.obb.enabled=True のときのみ) ---
    pred_angles: Optional[torch.Tensor] = None
    pred_obb: Optional[torch.Tensor] = None
    # --- Pose (detector.pose.enabled=True のときのみ) ---
    pred_keypoints: Optional[torch.Tensor] = None
    pred_polarity: Optional[torch.Tensor] = None

    @property
    def has_obb(self) -> bool:
        return self.pred_obb is not None


# ---------------------------------------------------------------------------
# 抽象基底クラス
# ---------------------------------------------------------------------------

class BaseDetector(nn.Module, ABC):
    """
    Detector の抽象基底クラス。
    すべての detector 実装はこのクラスを継承すること。
    """

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
    ) -> DetectionOutput:
        """
        Args:
            images: (B, 3, H, W) 正規化済み画像

        Returns:
            DetectionOutput
        """
        ...

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """query feature の次元数を返す"""
        ...

    @abstractmethod
    def get_num_queries(self) -> int:
        """detection query の数を返す"""
        ...

    def get_encoder_channels(self) -> List[int]:
        """encoder_features の各スケールのチャネル数を返す"""
        return []
