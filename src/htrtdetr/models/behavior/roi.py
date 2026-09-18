"""
roi.py — 行動認識用の 2-stream ROI 抽出

YOAKE の設計原則: **個体検出と行動認識で同じ箱を使わない**。

  Actor stream   — 個体 OBB に沿って切り出し、体軸が一定方向を向くように回転
                   正規化した ROI。個体自身の細かい appearance と姿勢を見る。
  Context stream — 個体 OBB の外接矩形を context_scale 倍に広げた**軸平行** ROI。
                   他個体・相対位置・周辺状況 (social context) を見る。

    OBB ──┬─→ rotated ROIAlign ─→ actor feature   (細粒度・orientation-normalized)
          └─→ AABB → x scale ──→ context feature (粗粒度・広い文脈)

回転 crop は ``affine_grid`` + ``grid_sample`` で実装する。追加依存が無く、
CPU / CUDA / MPS すべてで動き、勾配も流れる (学習時に backbone まで誤差が届く)。
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from ...geometry.obb import obb_to_context_aabb

__all__ = ["RotatedROIExtractor", "ContextROIExtractor", "build_roi_affine"]


def build_roi_affine(
    obb: torch.Tensor,        # (N, 5) [cx, cy, w, h, theta] 正規化座標 [0,1]
    feat_h: int,
    feat_w: int,
    orientation_normalize: bool = True,
) -> torch.Tensor:
    """
    OBB から ``affine_grid`` 用の 2x3 変換行列を作る。

    出力の正方グリッド [-1,1]^2 が入力特徴マップ上の回転矩形に写るようにする。
    ``align_corners=False`` の grid_sample 規約 (y 軸は下向き) に合わせてあるので、
    theta の符号は ``obb_to_corners`` と一致する。

    非正方の特徴マップでは正規化座標が異方的になり、素朴な回転行列ではせん断が
    生じる。``feat_h != feat_w`` の場合はアスペクト比で補正して**真の回転**にする。

    Args:
        obb: (N, 5) 正規化座標
        feat_h / feat_w: 入力特徴マップの高さ・幅
        orientation_normalize: False なら回転を無視 (theta=0 として切り出す)。
            ablation で「actor crop の向き正規化」の効果を測るために使う。
    Returns:
        (N, 2, 3) affine 行列
    """
    cx, cy, w, h, theta = obb.unbind(-1)
    if not orientation_normalize:
        theta = torch.zeros_like(theta)

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # 正規化中心を [0,1] → [-1,1] へ
    tx = 2.0 * cx - 1.0
    ty = 2.0 * cy - 1.0
    # アスペクト比補正 (正方なら 1.0)
    ar_x = feat_h / max(feat_w, 1)
    ar_y = feat_w / max(feat_h, 1)

    a11 = w * cos_t
    a12 = -h * ar_x * sin_t
    a21 = w * ar_y * sin_t
    a22 = h * cos_t

    row1 = torch.stack([a11, a12, tx], dim=-1)
    row2 = torch.stack([a21, a22, ty], dim=-1)
    return torch.stack([row1, row2], dim=-2)     # (N, 2, 3)


class RotatedROIExtractor(nn.Module):
    """
    Actor stream: OBB に沿った回転 ROI を切り出す。

    体軸 (OBB の w 方向) が常に crop の横軸になるので、同じ姿勢の個体は向きに
    よらず同じ特徴に写る。これが個体の細かい appearance・姿勢の解析を容易にする。

    Args:
        output_size: 出力 ROI の一辺 (正方)
        orientation_normalize: False で向き正規化を無効化 (ablation 用)
        expand: OBB を少し膨らませてから切り出す倍率 (境界の情報を落とさないため)
    """

    def __init__(
        self,
        output_size: int = 16,
        orientation_normalize: bool = True,
        expand: float = 1.1,
    ):
        super().__init__()
        self.output_size = int(output_size)
        self.orientation_normalize = bool(orientation_normalize)
        self.expand = float(expand)

    def forward(
        self,
        features: torch.Tensor,     # (B, C, H, W)
        obb: torch.Tensor,          # (N, 5) 正規化座標
        batch_index: torch.Tensor,  # (N,) 各 ROI が属するバッチ要素
    ) -> torch.Tensor:
        """
        Returns:
            (N, C, output_size, output_size)
        """
        N = obb.shape[0]
        B, C, H, W = features.shape
        if N == 0:
            return features.new_zeros(0, C, self.output_size, self.output_size)

        obb_e = obb.clone()
        obb_e[:, 2:4] = obb_e[:, 2:4] * self.expand

        theta = build_roi_affine(
            obb_e, H, W, orientation_normalize=self.orientation_normalize
        )                                                    # (N, 2, 3)
        grid = F.affine_grid(
            theta, (N, C, self.output_size, self.output_size), align_corners=False
        )
        # 各 ROI に対応するバッチ要素の特徴マップを集める
        src = features[batch_index]                          # (N, C, H, W)
        return F.grid_sample(
            src, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )


class ContextROIExtractor(nn.Module):
    """
    Context stream: 個体周辺を含む広い**軸平行** ROI を切り出す。

    個体 OBB → 外接 AABB → context_scale 倍に拡大 → ROIAlign。
    他個体との相対位置や周辺状況を保つため、回転はかけない (回転させると
    「相手がどちら側にいるか」という social context が失われる)。

    Args:
        output_size: 出力 ROI の一辺
        context_scale: AABB の拡大率
        square: 正方形に揃えるか (ROI の歪みを避ける)
    """

    def __init__(
        self,
        output_size: int = 16,
        context_scale: float = 2.0,
        square: bool = True,
    ):
        super().__init__()
        self.output_size = int(output_size)
        self.context_scale = float(context_scale)
        self.square = bool(square)

    def forward(
        self,
        features: torch.Tensor,     # (B, C, H, W)
        obb: torch.Tensor,          # (N, 5) 正規化座標
        batch_index: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        Returns:
            (N, C, output_size, output_size)
        """
        N = obb.shape[0]
        B, C, H, W = features.shape
        if N == 0:
            return features.new_zeros(0, C, self.output_size, self.output_size)

        aabb = obb_to_context_aabb(
            obb, scale=self.context_scale, square=self.square
        )                                                    # (N, 4) 正規化 xyxy

        # AABB を回転 0 の OBB とみなして同じ affine 経路に載せる
        x1, y1, x2, y2 = aabb.unbind(-1)
        ctx = torch.stack([
            (x1 + x2) * 0.5, (y1 + y2) * 0.5,
            (x2 - x1).clamp(min=1e-6), (y2 - y1).clamp(min=1e-6),
            torch.zeros_like(x1),
        ], dim=-1)

        theta = build_roi_affine(ctx, H, W, orientation_normalize=False)
        grid = F.affine_grid(
            theta, (N, C, self.output_size, self.output_size), align_corners=False
        )
        src = features[batch_index]
        return F.grid_sample(
            src, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    def context_boxes(self, obb: torch.Tensor) -> torch.Tensor:
        """
        GUI 表示用に context AABB (N, 4) を返す。

        「個体検出 = tight rotated box / 行動認識 = larger normal box」の違いを
        可視化するために、モデルが実際に使った箱をそのまま出す。
        """
        return obb_to_context_aabb(obb, scale=self.context_scale, square=self.square)
