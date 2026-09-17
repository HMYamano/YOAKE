"""
_geometry.py — 段が共有する numpy 幾何ユーティリティ (torch 非依存)

古典アルゴリズム段 (安定化 / 幾何関係 / bytetrack / 運動 / heuristic) はここだけに
依存し、モデルや GPU なしで完結する。
"""

from __future__ import annotations

from typing import List

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """2つの xyxy 箱の IoU。"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def linear_interp_box(a: np.ndarray, b: np.ndarray, frac: float) -> np.ndarray:
    """xyxy 箱 a→b を frac (0..1) で線形補間する。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a * (1.0 - frac) + b * frac).astype(np.float32)


def greedy_match(cost: np.ndarray, max_cost: float):
    """
    コスト最小のペアから貪欲に割り当てる (scipy 非依存)。

    Returns:
        matches: List[(row, col)] — cost <= max_cost のペアのみ
        unmatched_rows, unmatched_cols
    """
    nr, nc = cost.shape
    matches: List = []
    used_r, used_c = set(), set()
    if nr and nc:
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if r in used_r or c in used_c:
                continue
            if cost[r, c] > max_cost:
                break
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)
    unmatched_rows = [r for r in range(nr) if r not in used_r]
    unmatched_cols = [c for c in range(nc) if c not in used_c]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# OBB (Oriented Bounding Box) — numpy 版
# ---------------------------------------------------------------------------
# geometry/obb.py (torch 版) と同じ定義・同じ数値を返す numpy サブセット。
# 段プレーンを torch 非依存に保つために意図的に複製している。
# 両者の一致は tests/test_geometry_obb.py::TestTorchNumpyParity が固定する。
#
#   obb     : [cx, cy, w, h, theta]  theta は rad、周期 pi
#   corners : (4, 2) 反時計回り
#   aabb    : [x1, y1, x2, y2]

_OBB_EPS = 1e-7
_OBB_ANGLE_SNAP = 1e-6


def normalize_angle_np(theta: float) -> float:
    """theta を半開区間 [-pi/2, pi/2) に畳む (矩形の周期は pi)。"""
    out = (theta + np.pi / 2) % np.pi - np.pi / 2
    if out >= np.pi / 2 - _OBB_ANGLE_SNAP:
        out -= np.pi
    return float(out)


def obb_to_corners_np(obb: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h, theta] → (4, 2) 頂点 (反時計回り)。"""
    cx, cy, w, h, theta = (float(v) for v in obb[:5])
    hw, hh = w * 0.5, h * 0.5
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return (local @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def obb_to_aabb_np(obb: np.ndarray) -> np.ndarray:
    """OBB → それを囲う最小の軸平行矩形 [x1, y1, x2, y2]。"""
    c = obb_to_corners_np(obb)
    return np.array(
        [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], dtype=np.float32
    )


def aabb_to_obb_np(aabb: np.ndarray) -> np.ndarray:
    """軸平行矩形 → theta=0 の OBB (真の回転アノテーションではない fallback)。"""
    x1, y1, x2, y2 = (float(v) for v in aabb[:4])
    return np.array(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 0.0), max(y2 - y1, 0.0), 0.0],
        dtype=np.float32,
    )


def _poly_area_np(poly: np.ndarray) -> float:
    """shoelace 公式による凸多角形の面積。"""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _clip_poly_np(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman により凸多角形 subject を凸多角形 clip で切り取る。"""
    out = subject
    for i in range(len(clip)):
        if len(out) == 0:
            return out
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a

        def _inside(p, _a=a, _e=edge):
            return _e[0] * (p[1] - _a[1]) - _e[1] * (p[0] - _a[0]) >= -_OBB_EPS

        new: list = []
        for j in range(len(out)):
            cur, prv = out[j], out[j - 1]
            cur_in, prv_in = _inside(cur), _inside(prv)
            if cur_in != prv_in:
                d = cur - prv
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > _OBB_EPS:
                    t = (edge[0] * (prv[1] - a[1]) - edge[1] * (prv[0] - a[0])) / -denom
                    new.append(prv + t * d)
            if cur_in:
                new.append(cur)
        out = np.array(new, dtype=np.float32) if new else np.zeros((0, 2), np.float32)
    return out


def rotated_iou_np(obb_a: np.ndarray, obb_b: np.ndarray) -> float:
    """2 つの OBB の厳密な rotated IoU。"""
    ca, cb = obb_to_corners_np(obb_a), obb_to_corners_np(obb_b)
    # 反時計回りを保証 (clip 側の巻き方向に依存するため)
    if _signed_area_np(ca) < 0:
        ca = ca[::-1].copy()
    if _signed_area_np(cb) < 0:
        cb = cb[::-1].copy()
    inter = _poly_area_np(_clip_poly_np(ca, cb))
    area_a = max(float(obb_a[2]), 0.0) * max(float(obb_a[3]), 0.0)
    area_b = max(float(obb_b[2]), 0.0) * max(float(obb_b[3]), 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > _OBB_EPS else 0.0


def _signed_area_np(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def expand_aabb_np(
    aabb: np.ndarray,
    scale: float,
    width: int = 0,
    height: int = 0,
    square: bool = False,
) -> np.ndarray:
    """AABB を中心を保ったまま scale 倍に広げる (行動認識の context ROI)。"""
    x1, y1, x2, y2 = (float(v) for v in aabb[:4])
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w = max(x2 - x1, 0.0) * scale
    h = max(y2 - y1, 0.0) * scale
    if square:
        w = h = max(w, h)
    out = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
    if width > 0 and height > 0:
        out[0] = max(out[0], 0.0)
        out[1] = max(out[1], 0.0)
        out[2] = min(out[2], float(width))
        out[3] = min(out[3], float(height))
    return out


def orientation_consistency_np(theta_a: float, theta_b: float) -> float:
    """
    2 つの OBB 向きの一致度 ∈ [0, 1] (1 が完全一致)。

    矩形の周期は pi なので |Δtheta| を [0, pi/2] に畳んでから正規化する。
    追跡の association コストに使う。
    """
    d = abs(normalize_angle_np(theta_a - theta_b))
    return float(1.0 - d / (np.pi / 2))
