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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 出力型定義
# ---------------------------------------------------------------------------

@dataclass
class DetectionOutput:
    """
    Detector の出力をまとめる dataclass

    Attributes:
        pred_logits: (B, num_queries, num_classes) — class logits
        pred_boxes:  (B, num_queries, 4) — [cx, cy, w, h] in [0, 1]
        query_features: (B, num_queries, hidden_dim) — per-query features
        encoder_features: List[(B, H*W, C)] — multi-scale encoder outputs
                          (Hierarchical Temporal Module に渡す)
    """
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    query_features: torch.Tensor
    encoder_features: Optional[List[torch.Tensor]] = None


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
