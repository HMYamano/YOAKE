"""
clip.py — the clip tube: temporal evidence that does not depend on tracking.

YOWO reads a keyframe with a 2D network and the clip before it with a 3D
network. YOAKE already has its 2D side (actor and context crops of the
keyframe) and a *track-aligned* temporal side (kinematics and the HTM over each
animal's own history). What it lacked is YOWO's other half: motion read
straight from the pixels, with no tracker in between.

That half matters exactly where tracking is weakest. Two flies in copula look
like one blob; the detector may see one box, the tracker may cut the track, and
then the animal's history is a single frame. The tube still sees the motion.

    every frame t  : FPN map (B, C, H, W) --1x1 conv--> (B, c, H, W)    once per frame
    keyframe OBB   : context AABB (the same box the context stream uses)
    tube           : that box, ROI-sampled on every frame  -> (N, c, T, S, S)
    motion         : [x_t, x_t - x_{t-1}]                  -> (N, 2c, T, S, S)
    encoder        : Conv3d x2 + GroupNorm + ReLU, pooled  -> (N, out_dim)

The box is held fixed over the clip (a "static tube", as AVA-style detectors
pool over the keyframe box), so the tube needs no association at all. The
per-frame reduction is separate from the encoder on purpose: streaming runs it
once per frame and keeps the small maps in a ring buffer, preserving the
invariant that every frame passes the backbone exactly once.

It is one more evidence group, added like the others (``Σ_g gate_g · head_g``),
so how much a prediction leaned on tube motion versus the animal's tracked
motion is part of the prediction's own decomposition.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .roi import ContextROIExtractor

__all__ = ["ClipTubeEncoder"]


def _groups(channels: int, preferred: int = 16) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ClipTubeEncoder(nn.Module):
    """Encode the keyframe's context box across the whole clip.

    Args:
        in_channels: channels of the FPN level the crops come from.
        out_dim: size of the evidence feature (the branch's common ``d``).
        reduced_channels: channels kept per frame. The ring buffer in streaming
            holds ``span`` of these maps, so this is the memory knob.
        roi_size: side of the per-frame ROI grid.
        context_scale / square: the context box, identical to the context
            stream's so the two read the same region.
        hidden: width of the 3D encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        reduced_channels: int = 64,
        roi_size: int = 7,
        context_scale: float = 2.0,
        square: bool = True,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        c = int(reduced_channels)
        h = int(hidden)
        self.out_dim = int(out_dim)
        self.reduced_channels = c
        self.reduce = nn.Sequential(
            nn.Conv2d(int(in_channels), c, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(c), c),
            nn.ReLU(inplace=True),
        )
        self.roi = ContextROIExtractor(
            output_size=int(roi_size), context_scale=float(context_scale), square=bool(square)
        )
        self.encoder = nn.Sequential(
            nn.Conv3d(2 * c, h, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(h), h),
            nn.ReLU(inplace=True),
            nn.Conv3d(h, h, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(h), h),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(h, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )

    def reduce_frame(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, c, H, W)``. Run once per frame."""
        return self.reduce(features)

    def forward(
        self,
        frames: Sequence[torch.Tensor],   # T x (B, c, H, W), oldest first, reduced
        obb: torch.Tensor,                # (N, 5) keyframe OBBs, normalised
        batch_index: torch.Tensor,        # (N,)
    ) -> torch.Tensor:
        """``(N, out_dim)`` — one feature per animal on the keyframe."""
        n = int(obb.shape[0])
        if n == 0 or not frames:
            return obb.new_zeros(0, self.out_dim)
        tube = torch.stack(
            [self.roi(frame, obb, batch_index) for frame in frames], dim=2
        )                                                    # (N, c, T, S, S)
        motion = torch.cat(
            [torch.zeros_like(tube[:, :, :1]), tube[:, :, 1:] - tube[:, :, :-1]], dim=2
        )
        return self.encoder(torch.cat([tube, motion], dim=1))
