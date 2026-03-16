"""
hierarchical_temporal.py — Hierarchical Temporal Module (HTM)

設計方針:
- 3 branch (short / mid / long) で異なる時間スケールを捉える
- 各 branch は depthwise separable 1D temporal conv で実装
- dilation で receptive field を branch ごとに変える
- residual connection + optional channel attention gate
- branch 出力を concat して projection する fusion module

入出力:
- 入力: (B, T, D) — B: batch, T: frames, D: feature dim
- 出力: (B, T, out_dim) — temporal-enriched feature

Note: 実際の使用では各 detection query ごとに独立して適用する。
      つまり (B*Q, T, D) の形で入力することが多い。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config.config import HierarchicalTemporalConfig, TemporalBranchConfig


# ---------------------------------------------------------------------------
# Depthwise Separable 1D Temporal Conv
# ---------------------------------------------------------------------------

class DepthwiseSeparableTemporalConv(nn.Module):
    """
    Depthwise separable 1D convolution for temporal processing.
    通常の 1D conv に比べてパラメータ数・計算量を削減する。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: Optional[int] = None,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        # Depthwise conv: チャネルごとに独立して時間方向を畳み込む
        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,  # depthwise
            bias=False,
        )
        # Pointwise conv: チャネル間の情報を混合
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T)"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# Standard Dilated 1D Temporal Conv (代替実装)
# ---------------------------------------------------------------------------

class DilatedTemporalConv(nn.Module):
    """通常の dilated 1D conv (depthwise なし)"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Channel Attention Gate (SE-like)
# ---------------------------------------------------------------------------

class ChannelAttentionGate(nn.Module):
    """
    Squeeze-and-Excitation 風のチャネルアテンションゲート。
    時間方向の global average pooling → fc → sigmoid で各チャネルの重みを計算。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T)"""
        # Global average over time → (B, C)
        gap = x.mean(dim=2)
        weight = self.gate(gap)  # (B, C)
        return x * weight.unsqueeze(2)


# ---------------------------------------------------------------------------
# 単一 Temporal Branch
# ---------------------------------------------------------------------------

class TemporalBranch(nn.Module):
    """
    Hierarchical Temporal Module の 1 branch。

    処理フロー:
      入力 (B, C, T)
        → temporal conv (depthwise separable or standard)
        → [optional] channel attention
        → [optional] residual add
      出力 (B, C, T)
    """

    def __init__(self, feature_dim: int, cfg: TemporalBranchConfig):
        super().__init__()
        self.num_frames = cfg.num_frames

        # Temporal conv の選択
        if cfg.use_depthwise:
            self.conv = DepthwiseSeparableTemporalConv(
                feature_dim, cfg.kernel_size, cfg.dilation
            )
        else:
            self.conv = DilatedTemporalConv(
                feature_dim, cfg.kernel_size, cfg.dilation
            )

        # Channel attention gate (optional)
        self.channel_attn = (
            ChannelAttentionGate(feature_dim) if cfg.use_channel_attention else None
        )

        self.use_residual = cfg.use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — T はこの branch の num_frames
        Returns:
            out: (B, C, T)
        """
        residual = x
        out = self.conv(x)

        if self.channel_attn is not None:
            out = self.channel_attn(out)

        if self.use_residual:
            out = out + residual

        return out


# ---------------------------------------------------------------------------
# Branch Fusion
# ---------------------------------------------------------------------------

class BranchFusion(nn.Module):
    """
    3 branch の出力を融合する。
    fusion_method:
      - "concat_proj": 3 branch を concat して線形変換
      - "sum": 単純加算 (次元が同じ場合)
      - "attention": attention-weighted sum
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        fusion_method: str = "concat_proj",
    ):
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == "concat_proj":
            # 3 branch を concat → projection
            self.proj = nn.Sequential(
                nn.Linear(feature_dim * 3, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_method == "sum":
            assert feature_dim == output_dim, \
                "sum fusion requires feature_dim == output_dim"
            self.proj = None
        elif fusion_method == "attention":
            # 学習可能な attention weight で branch を重み付け
            self.attn_proj = nn.Linear(feature_dim, 1)
            self.out_proj = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

    def forward(
        self,
        short: torch.Tensor,   # (B, C, T)
        mid: torch.Tensor,     # (B, C, T)
        long: torch.Tensor,    # (B, C, T)
    ) -> torch.Tensor:
        """
        Returns: (B, out_dim, T)
        """
        if self.fusion_method == "concat_proj":
            # (B, 3C, T) → permute → (B, T, 3C) → linear → (B, T, out_dim)
            cat = torch.cat([short, mid, long], dim=1)  # (B, 3C, T)
            out = cat.permute(0, 2, 1)                  # (B, T, 3C)
            out = self.proj(out)                        # (B, T, out_dim)
            return out.permute(0, 2, 1)                 # (B, out_dim, T)

        elif self.fusion_method == "sum":
            return short + mid + long

        elif self.fusion_method == "attention":
            # (B, C, T) → (B, T, C) で attention
            s = short.permute(0, 2, 1)  # (B, T, C)
            m = mid.permute(0, 2, 1)
            l = long.permute(0, 2, 1)
            stack = torch.stack([s, m, l], dim=2)  # (B, T, 3, C)

            scores = self.attn_proj(stack).squeeze(-1)  # (B, T, 3)
            weights = F.softmax(scores, dim=2).unsqueeze(-1)  # (B, T, 3, 1)

            fused = (stack * weights).sum(dim=2)  # (B, T, C)
            out = self.out_proj(fused)             # (B, T, out_dim)
            return out.permute(0, 2, 1)            # (B, out_dim, T)


# ---------------------------------------------------------------------------
# Hierarchical Temporal Module (メインクラス)
# ---------------------------------------------------------------------------

class HierarchicalTemporalModule(nn.Module):
    """
    Hierarchical Temporal Module (HTM)

    3 つの時間スケール branch で特徴を処理し、融合する。

    設計:
    - 入力は (B, T, D) の feature sequence
    - 各 branch で異なる T (num_frames) に対応:
      短い方のブランチは最近の T_short フレームのみ使い、
      長い方は T_long フレームを使う
    - 出力は (B, T, out_dim)

    使用例:
        htm = HierarchicalTemporalModule(cfg)
        # query features の時系列: (B*Q, T, D)
        temporal_feat = htm(query_feat_seq)  # (B*Q, T, out_dim)
    """

    def __init__(self, cfg: HierarchicalTemporalConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.feature_dim

        # 入力 projection (次元を統一)
        self.input_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
        )

        # 3 branches
        self.short_branch = TemporalBranch(D, cfg.short_branch)
        self.mid_branch = TemporalBranch(D, cfg.mid_branch)
        self.long_branch = TemporalBranch(D, cfg.long_branch)

        # Fusion
        self.fusion = BranchFusion(D, cfg.output_dim, cfg.fusion_method)

        # 出力 Layer Norm
        self.output_norm = nn.LayerNorm(cfg.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — T はシーケンス長 (window_size と同じ)

        Returns:
            out: (B, T, out_dim)
        """
        B, T, D = x.shape

        # 入力 projection: (B, T, D) → (B, D, T)
        x_proj = self.input_proj(x)  # (B, T, D)
        x_t = x_proj.permute(0, 2, 1)  # (B, D, T)

        # --- Short branch ---
        # 最新の T_short フレームを使用
        T_short = min(self.cfg.short_branch.num_frames, T)
        x_short = self._pad_or_truncate(x_t, T_short)
        short_out = self.short_branch(x_short)  # (B, D, T_short)
        short_out = self._pad_to_T(short_out, T)  # (B, D, T)

        # --- Mid branch ---
        T_mid = min(self.cfg.mid_branch.num_frames, T)
        x_mid = self._pad_or_truncate(x_t, T_mid)
        mid_out = self.mid_branch(x_mid)  # (B, D, T_mid)
        mid_out = self._pad_to_T(mid_out, T)   # (B, D, T)

        # --- Long branch ---
        T_long = min(self.cfg.long_branch.num_frames, T)
        x_long = self._pad_or_truncate(x_t, T_long)
        long_out = self.long_branch(x_long)  # (B, D, T_long)
        long_out = self._pad_to_T(long_out, T)  # (B, D, T)

        # --- Fusion ---
        fused = self.fusion(short_out, mid_out, long_out)  # (B, out_dim, T)

        # (B, out_dim, T) → (B, T, out_dim)
        out = fused.permute(0, 2, 1)
        out = self.output_norm(out)

        return out

    @staticmethod
    def _pad_or_truncate(x: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        x: (B, D, T) を target_T の長さに調整する。
        - T > target_T: 最新の target_T フレームを取る
        - T < target_T: 先頭を zero-pad する
        """
        T = x.shape[2]
        if T >= target_T:
            return x[:, :, -target_T:]  # 最新フレームを取る
        else:
            pad = torch.zeros(x.shape[0], x.shape[1], target_T - T,
                              device=x.device, dtype=x.dtype)
            return torch.cat([pad, x], dim=2)

    @staticmethod
    def _pad_to_T(x: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        x: (B, D, T_branch) を target_T にリサイズ (補間 or pad)。
        時間軸方向に interpolate して元の T に合わせる。
        """
        T = x.shape[2]
        if T == target_T:
            return x
        # 線形補間でリサイズ
        return F.interpolate(x.unsqueeze(0), size=(x.shape[1], target_T),
                             mode="nearest").squeeze(0)

    def forward_ablation(
        self,
        x: torch.Tensor,
        use_short: bool = True,
        use_mid: bool = True,
        use_long: bool = True,
    ) -> torch.Tensor:
        """
        Ablation 用: 使用する branch を選択できる forward。
        無効化した branch には zero tensor を使う。
        """
        B, T, D = x.shape
        x_proj = self.input_proj(x)
        x_t = x_proj.permute(0, 2, 1)

        def _run_branch(branch, num_frames):
            T_b = min(num_frames, T)
            x_b = self._pad_or_truncate(x_t, T_b)
            out = branch(x_b)
            return self._pad_to_T(out, T)

        zero = torch.zeros(B, D, T, device=x.device, dtype=x.dtype)

        short_out = _run_branch(self.short_branch, self.cfg.short_branch.num_frames) \
            if use_short else zero
        mid_out   = _run_branch(self.mid_branch,   self.cfg.mid_branch.num_frames)   \
            if use_mid   else zero
        long_out  = _run_branch(self.long_branch,  self.cfg.long_branch.num_frames)  \
            if use_long  else zero

        fused = self.fusion(short_out, mid_out, long_out)
        out = fused.permute(0, 2, 1)
        return self.output_norm(out)
