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
