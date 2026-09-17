"""
kinematics.py — track 履歴から運動 / 向き / 個体間関係の特徴を作る

行動は「見た目」だけでは決まらない。静止画では groom と idle が区別できず、
courtship と chase は「誰が誰をどう追っているか」で決まる。ここでは
track-aligned な履歴から、そうした判断に必要な量を明示的に取り出す。

=== 特徴グループ ===

MOTION (運動)           速度 / 加速度 / 変位 / 移動方向 / 角速度
ORIENTATION (向き)      体軸方向 / 体軸と移動方向のなす角 / 体軸の回転速度
INTERACTION (個体間)    最近傍距離 / 相対速度 / 接近率 / 相手を向いているか / 重なり

すべて微分可能な torch 演算で、勾配は (bbox 回帰経由で) detector まで届く。
特徴は行動ごとに意味が読めるように**名前付き**で管理し、説明可能性の
evidence group にそのまま対応させる。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...geometry.obb import normalize_angle, obb_to_aabb
from ...utils.misc import box_iou

__all__ = [
    "MOTION_FEATURE_NAMES",
    "ORIENTATION_FEATURE_NAMES",
    "INTERACTION_FEATURE_NAMES",
    "MOTION_DIM",
    "ORIENTATION_DIM",
    "INTERACTION_DIM",
    "compute_motion_features",
    "compute_orientation_features",
    "compute_interaction_features",
    "KinematicsEncoder",
]

MOTION_FEATURE_NAMES: List[str] = [
    "vx", "vy",                    # 速度 (正規化座標 / frame)
    "speed",                       # 速さ
    "ax", "ay",                    # 加速度
    "accel_magnitude",             # 加速度の大きさ
    "displacement",                # 窓全体の正味変位
    "path_length",                 # 窓全体の移動距離 (経路長)
    "straightness",                # displacement / path_length ∈ [0,1]
    "speed_std",                   # 速さの分散 (断続的な動きの指標)
    "movement_dir_sin", "movement_dir_cos",  # 移動方向
]
MOTION_DIM = len(MOTION_FEATURE_NAMES)

ORIENTATION_FEATURE_NAMES: List[str] = [
    "body_sin", "body_cos",        # 体軸方向 (2 倍角でなく実角度; 向きの連続表現)
    "angular_velocity",            # 体軸の回転速度
    "heading_vs_body",             # 移動方向と体軸のなす角 (前進 / 横歩き / 後退)
    "elongation",                  # w/h — 伸長度 (姿勢変化の指標)
    "area_change",                 # 面積の変化率 (立ち上がり / 伏せ)
]
ORIENTATION_DIM = len(ORIENTATION_FEATURE_NAMES)

INTERACTION_FEATURE_NAMES: List[str] = [
    "nn_distance",                 # 最近傍個体との距離
    "nn_distance_norm",            # 自身のサイズで正規化した距離
    "rel_vx", "rel_vy",            # 最近傍との相対速度
    "approach_rate",               # 接近率 (>0 で近づいている)
    "facing_to_nn",                # 自分が相手を向いているか
    "faced_by_nn",                 # 相手が自分を向いているか
    "nn_iou",                      # 最近傍との重なり
    "neighbor_count",              # 近傍個体数 (密度)
]
INTERACTION_DIM = len(INTERACTION_FEATURE_NAMES)

_EPS = 1e-6


def _masked_diff(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    時系列の 1 階差分。無効フレームの寄与は 0 にする。

    Args:
        seq: (N, T, K)
        mask: (N, T) bool
    Returns:
        (N, T-1, K)
    """
    d = seq[:, 1:] - seq[:, :-1]
    both = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(seq.dtype)
    return d * both


def compute_motion_features(
    obb_hist: torch.Tensor,   # (N, T, 5) [cx, cy, w, h, theta] 正規化座標
    mask: torch.Tensor,       # (N, T) bool
) -> torch.Tensor:
    """
    運動特徴 (N, MOTION_DIM) を計算する。窓の**最終フレーム時点**の値を返す。

    無効 (未観測) フレームは差分から除外されるので、検出が飛んでも速度が
    偽の急変を起こさない。
    """
    N, T, _ = obb_hist.shape
    device, dtype = obb_hist.device, obb_hist.dtype
    if N == 0:
        return torch.zeros(0, MOTION_DIM, device=device, dtype=dtype)

    centers = obb_hist[..., :2]                               # (N, T, 2)
    vel = _masked_diff(centers, mask)                         # (N, T-1, 2)

    if T >= 2:
        last_v = vel[:, -1]                                   # (N, 2)
        speeds = vel.norm(dim=-1)                             # (N, T-1)
    else:
        last_v = torch.zeros(N, 2, device=device, dtype=dtype)
        speeds = torch.zeros(N, 1, device=device, dtype=dtype)

    speed = last_v.norm(dim=-1)                               # (N,)

    if T >= 3:
        acc = vel[:, 1:] - vel[:, :-1]                        # (N, T-2, 2)
        last_a = acc[:, -1]
    else:
        last_a = torch.zeros(N, 2, device=device, dtype=dtype)

    valid_cnt = mask.sum(dim=1).clamp(min=1).to(dtype)
    # 正味変位: 最初と最後の有効フレーム間の直線距離
    first_idx = mask.float().argmax(dim=1)                    # (N,)
    last_idx = (T - 1) - mask.flip(1).float().argmax(dim=1)
    gather = lambda idx: centers[torch.arange(N, device=device), idx]   # noqa: E731
    displacement = (gather(last_idx) - gather(first_idx)).norm(dim=-1)

    path_length = speeds.sum(dim=-1)
    straightness = displacement / (path_length + _EPS)

    speed_mean = path_length / valid_cnt
    speed_std = ((speeds - speed_mean.unsqueeze(1)) ** 2).mean(dim=-1).sqrt()

    dir_norm = last_v / (speed.unsqueeze(-1) + _EPS)
    move_cos, move_sin = dir_norm[:, 0], dir_norm[:, 1]

    return torch.stack([
        last_v[:, 0], last_v[:, 1], speed,
        last_a[:, 0], last_a[:, 1], last_a.norm(dim=-1),
        displacement, path_length, straightness.clamp(0.0, 1.0),
        speed_std, move_sin, move_cos,
    ], dim=-1)


def compute_orientation_features(
    obb_hist: torch.Tensor,   # (N, T, 5)
    mask: torch.Tensor,       # (N, T)
) -> torch.Tensor:
    """
    向き・姿勢特徴 (N, ORIENTATION_DIM) を計算する。

    体軸方向そのものに加え、**移動方向との関係**を明示的に持たせる。
    前進 / 横歩き / 後退は同じ速度でも行動として全く異なるため。
    """
    N, T, _ = obb_hist.shape
    device, dtype = obb_hist.device, obb_hist.dtype
    if N == 0:
        return torch.zeros(0, ORIENTATION_DIM, device=device, dtype=dtype)

    theta = obb_hist[..., 4]                                  # (N, T)
    last_theta = theta[:, -1]

    # 角速度: 周期 pi を考慮して差を畳んでから取る
    if T >= 2:
        dtheta = normalize_angle(theta[:, 1:] - theta[:, :-1])
        both = (mask[:, 1:] & mask[:, :-1]).to(dtype)
        ang_vel = (dtheta * both)[:, -1]
    else:
        ang_vel = torch.zeros(N, device=device, dtype=dtype)

    # 移動方向と体軸のなす角
    centers = obb_hist[..., :2]
    vel = _masked_diff(centers, mask)
    last_v = vel[:, -1] if T >= 2 else torch.zeros(N, 2, device=device, dtype=dtype)
    heading = torch.atan2(last_v[:, 1], last_v[:, 0])
    heading_vs_body = normalize_angle(heading - last_theta).abs()
    # 速度がほぼ 0 のときは向きの関係が定義できないので 0 にする
    moving = (last_v.norm(dim=-1) > _EPS).to(dtype)
    heading_vs_body = heading_vs_body * moving

    w = obb_hist[:, -1, 2].clamp(min=_EPS)
    h = obb_hist[:, -1, 3].clamp(min=_EPS)
    elongation = w / h

    area = obb_hist[..., 2] * obb_hist[..., 3]                # (N, T)
    if T >= 2:
        area_change = (area[:, -1] - area[:, -2]) / (area[:, -2] + _EPS)
        area_change = area_change * (mask[:, -1] & mask[:, -2]).to(dtype)
    else:
        area_change = torch.zeros(N, device=device, dtype=dtype)

    return torch.stack([
        torch.sin(last_theta), torch.cos(last_theta),
        ang_vel, heading_vs_body,
        elongation.clamp(max=20.0), area_change.clamp(-5.0, 5.0),
    ], dim=-1)


def compute_interaction_features(
    obb: torch.Tensor,             # (N, 5) 現フレームの OBB
    velocity: torch.Tensor,        # (N, 2) 現フレームの速度
    group_index: Optional[torch.Tensor] = None,  # (N,) 同一シーンの識別子
    neighbor_radius: float = 0.15,
) -> torch.Tensor:
    """
    個体間 interaction 特徴 (N, INTERACTION_DIM) を計算する。

    既存の ``InteractionFeatureComputer`` が「自分→最近傍」の単一角度しか持たない
    のに対し、ここでは **facing_to_nn と faced_by_nn を別次元**にする。
    「A が B を追う」と「B が A を追う」は行動として非対称なので、対称な特徴では
    courtship / chase / aggression を原理的に区別できない。

    Args:
        group_index: 同じ値を持つ個体だけを近傍候補にする (バッチやシーンの分離)。
            None なら全個体が同一シーンとみなされる。
        neighbor_radius: neighbor_count を数える半径 (正規化座標)
    """
    N = obb.shape[0]
    device, dtype = obb.device, obb.dtype
    if N == 0:
        return torch.zeros(0, INTERACTION_DIM, device=device, dtype=dtype)
    if N == 1:
        # 相手がいない: 距離は最大、その他は 0。行動側が「単独」と読めるようにする。
        out = torch.zeros(1, INTERACTION_DIM, device=device, dtype=dtype)
        out[0, 0] = 1.0
        out[0, 1] = 1.0
        return out

    centers = obb[:, :2]                                       # (N, 2)
    dist = torch.cdist(centers, centers)                       # (N, N)

    # 自分自身と、別グループの個体を候補から外す
    invalid = torch.eye(N, dtype=torch.bool, device=device)
    if group_index is not None:
        invalid = invalid | (group_index.unsqueeze(1) != group_index.unsqueeze(0))
    dist_masked = dist.masked_fill(invalid, float("inf"))

    nn_dist, nn_idx = dist_masked.min(dim=1)                   # (N,)
    # 候補が皆無の個体 (グループ内で単独) は距離 1.0 として扱う
    lonely = torch.isinf(nn_dist)
    nn_dist = torch.where(lonely, torch.ones_like(nn_dist), nn_dist)
    nn_idx = torch.where(lonely, torch.arange(N, device=device), nn_idx)

    size = obb[:, 2:4].mean(dim=1).clamp(min=_EPS)
    nn_dist_norm = nn_dist / size

    nn_centers = centers[nn_idx]
    los = nn_centers - centers                                 # (N, 2) 自分→相手
    los_norm = los / (los.norm(dim=-1, keepdim=True) + _EPS)

    rel_vel = velocity - velocity[nn_idx]                      # (N, 2)
    # 接近率 (正で接近)。距離の時間変化は d|los|/dt = (v_nn - v_self)·los_norm
    # なので、接近 (距離が縮む) は rel_vel·los_norm > 0 に対応する。
    approach = (rel_vel * los_norm).sum(dim=-1)

    # facing: 体軸と視線のなす角 (小さいほど相手を向いている)。
    # OBB の theta は周期 pi なので値域は [0, pi/2] で、「前を向いているか後ろか」
    # までは決まらない (頭尾の曖昧さ)。その 180 度は pose head が解消する。
    theta = obb[:, 4]
    los_angle = torch.atan2(los[:, 1], los[:, 0])
    facing_to_nn = normalize_angle(theta - los_angle).abs()
    faced_by_nn = normalize_angle(theta[nn_idx] - (los_angle + torch.pi)).abs()

    aabb = obb_to_aabb(obb)
    iou_full = box_iou(aabb, aabb)
    nn_iou = iou_full[torch.arange(N, device=device), nn_idx]

    neighbor_count = (dist_masked < neighbor_radius).sum(dim=1).to(dtype)

    out = torch.stack([
        nn_dist, nn_dist_norm.clamp(max=50.0),
        rel_vel[:, 0], rel_vel[:, 1],
        approach, facing_to_nn, faced_by_nn,
        nn_iou, neighbor_count,
    ], dim=-1)
    # 単独個体は相互作用が未定義なので距離以外を 0 に落とす
    out = torch.where(lonely.unsqueeze(-1), torch.zeros_like(out), out)
    out[lonely, 0] = 1.0
    out[lonely, 1] = 1.0
    return out


class KinematicsEncoder(nn.Module):
    """
    生の kinematics 特徴を学習可能な埋め込みに変換する。

    各グループ (motion / orientation / interaction) を**別々に**符号化するのが要点。
    evidence decomposition でグループごとの寄与を分離するには、入口から混ぜずに
    持っておく必要がある。

    Args:
        output_dim: 各グループの出力次元
        use_motion / use_orientation / use_interaction: ablation スイッチ
    """

    def __init__(
        self,
        output_dim: int = 64,
        use_motion: bool = True,
        use_orientation: bool = True,
        use_interaction: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.use_motion = bool(use_motion)
        self.use_orientation = bool(use_orientation)
        self.use_interaction = bool(use_interaction)

        def _mlp(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(output_dim, output_dim),
            )

        if self.use_motion:
            self.motion_proj = _mlp(MOTION_DIM)
        if self.use_orientation:
            self.orientation_proj = _mlp(ORIENTATION_DIM)
        if self.use_interaction:
            self.interaction_proj = _mlp(INTERACTION_DIM)

    def forward(
        self,
        obb_hist: torch.Tensor,     # (N, T, 5)
        mask: torch.Tensor,         # (N, T)
        group_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            有効なグループ名 → (N, output_dim) の dict。
            無効化したグループはキー自体が現れない (ゼロ埋めしない)。
        """
        out: Dict[str, torch.Tensor] = {}
        N = obb_hist.shape[0]

        motion_raw: Optional[torch.Tensor] = None
        if self.use_motion or self.use_interaction:
            motion_raw = compute_motion_features(obb_hist, mask)

        if self.use_motion:
            out["motion"] = self.motion_proj(motion_raw)

        if self.use_orientation:
            orient_raw = compute_orientation_features(obb_hist, mask)
            out["orientation"] = self.orientation_proj(orient_raw)

        if self.use_interaction:
            vel = motion_raw[:, :2] if motion_raw is not None \
                else torch.zeros(N, 2, device=obb_hist.device, dtype=obb_hist.dtype)
            inter_raw = compute_interaction_features(
                obb_hist[:, -1], vel, group_index=group_index
            )
            out["interaction"] = self.interaction_proj(inter_raw)

        return out

    def raw_features(
        self,
        obb_hist: torch.Tensor,
        mask: torch.Tensor,
        group_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[torch.Tensor, List[str]]]:
        """
        符号化前の生の特徴と名前を返す (GUI / デバッグ / 論文の図表用)。

        Returns:
            {"motion": (値 (N, D), 名前リスト), ...}
        """
        motion_raw = compute_motion_features(obb_hist, mask)
        vel = motion_raw[:, :2]
        return {
            "motion": (motion_raw, MOTION_FEATURE_NAMES),
            "orientation": (
                compute_orientation_features(obb_hist, mask),
                ORIENTATION_FEATURE_NAMES,
            ),
            "interaction": (
                compute_interaction_features(obb_hist[:, -1], vel, group_index),
                INTERACTION_FEATURE_NAMES,
            ),
        }
