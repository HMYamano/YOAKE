"""
pair_features.py — 順序付きペア (A→B) の非対称特徴 (L5 共通)

L5 の肝は「A は B を向いているか」と「B は A を向いているか」を別次元で持つこと
(既存 InteractionFeatureComputer の自→最近傍の単一角度とは決定的に異なる)。A を基準
(egocentric) にした非対称特徴を作る。pose があれば向きを体軸ベースに置換できる。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ...pipeline.schema import Track

PAIR_FEAT_DIM = 10
PAIR_FEAT_NAMES = [
    "dist", "facing_AB", "facing_BA", "range_rate",
    "approach_A", "approach_B", "speed_A", "speed_B",
    "speed_diff", "behind",
]


def _wrap(a: float) -> float:
    """角度を [-pi, pi] に畳む。"""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def angle_between(heading: float, vec: np.ndarray) -> float:
    """heading(rad) と 2D ベクトル vec のなす角の絶対値 [0, pi]。"""
    if np.linalg.norm(vec) < 1e-6:
        return np.pi
    va = np.arctan2(vec[1], vec[0])
    return abs(_wrap(heading - va))


def _pos(track: Track, f: int, frame_pos: Dict[int, int]):
    i = frame_pos.get(f)
    if i is None:
        return None
    c = track.centers[i] if track.centers is not None else None
    v = track.velocities[i] if track.velocities is not None else np.zeros(2, np.float32)
    h = float(track.headings[i]) if track.headings is not None else 0.0
    s = float(track.speeds[i]) if track.speeds is not None else float(np.linalg.norm(v))
    return c, v, h, s


def build_pair_feature(
    A: Track, B: Track, f: int,
    posA: Dict[int, int], posB: Dict[int, int],
    prev_dist: Optional[float] = None,
) -> Optional[np.ndarray]:
    """frame f における A→B の非対称ペア特徴 (PAIR_FEAT_DIM,) を返す。無効なら None。"""
    a = _pos(A, f, posA)
    b = _pos(B, f, posB)
    if a is None or b is None or a[0] is None or b[0] is None:
        return None
    (ca, va, ha, sa) = a
    (cb, vb, hb, sb) = b

    los = (cb - ca).astype(np.float32)          # A→B 視線
    d = float(np.linalg.norm(los))
    los_u = los / (d + 1e-6)

    facing_AB = angle_between(ha, los)          # 小さいほど A は B を向く
    facing_BA = angle_between(hb, -los)         # 小さいほど B は A を向く
    range_rate = (d - prev_dist) if prev_dist is not None else 0.0
    approach_A = float(np.dot(va, los_u))       # A の視線方向速度成分 (>0: 接近)
    approach_B = float(np.dot(vb, -los_u))
    # A が B の後方 (B の進行方向の逆側) にいるか → chase 判定に使う。
    # 「A が B の後方」= B→A ベクトル (cb - ca の逆 = ca - cb ではなく B の進行方向と同側)。
    # B の heading と (A→B = cb - ca) が同方向なら A は B の後方にいる。
    behind = 1.0 if angle_between(hb, cb - ca) < (np.pi / 3) else 0.0

    return np.array([
        d, facing_AB, facing_BA, range_rate,
        approach_A, approach_B, sa, sb,
        sa - sb, behind,
    ], dtype=np.float32)
