"""
labeling_ops.py — 範囲ラベリング用の純粋関数 (dearpygui 非依存)
===============================================================

アノテータ (``htrtdetr.annotator``) の「範囲ラベリング」機能で使う bbox 補間などを
GUI から切り離して実装する。dearpygui / torch を import しないため、そのまま単体
テストできる (``tests/test_labeling_range.py``)。

bbox は ``[x1, y1, x2, y2]`` (ピクセル座標・左上/右下)。YOAKE JSON v1.1 の
``objects[].bbox`` と同じ表現。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Bbox = List[float]


def lerp(a: float, b: float, t: float) -> float:
    """線形補間。``t=0`` で ``a``、``t=1`` で ``b``。"""
    return a + (b - a) * t


def interpolate_bbox(box_a: Sequence[float], box_b: Sequence[float], t: float) -> Bbox:
    """2 つの bbox を割合 ``t`` (0..1) で線形補間する。"""
    return [lerp(float(a), float(b), t) for a, b in zip(box_a, box_b)]


def interpolate_between(
    f_start: int,
    box_start: Sequence[float],
    f_end: int,
    box_end: Sequence[float],
) -> List[Tuple[int, Bbox]]:
    """区間 ``[f_start, f_end]`` の各整数フレームに対する ``(frame_index, bbox)`` を返す。

    両端は ``box_start`` / ``box_end`` を厳密に再現する。``f_end < f_start`` の場合は
    自動的に入れ替える。単一フレーム (``f_start == f_end``) の場合は 1 要素のみ。
    """
    if f_end < f_start:
        f_start, f_end = f_end, f_start
        box_start, box_end = box_end, box_start
    span = f_end - f_start
    out: List[Tuple[int, Bbox]] = []
    for f in range(f_start, f_end + 1):
        t = 0.0 if span == 0 else (f - f_start) / span
        out.append((f, interpolate_bbox(box_start, box_end, t)))
    return out


def resolve_range_boxes(
    anchors: List[Tuple[int, Sequence[float]]],
    f_start: int,
    f_end: int,
) -> dict:
    """区間 ``[f_start, f_end]`` の各フレームに割り当てる bbox を返す ``{frame: bbox}``。

    ``anchors`` はキーフレーム ``(frame, bbox)`` のリスト。隣接アンカー間は線形補間し、
    アンカー範囲の外側 (最初のアンカーより前 / 最後のアンカーより後) は最も近いアンカーの
    box を保持する。アンカーが空なら空 dict (幾何情報なし)。

    範囲ラベリングの中核ロジック。GUI から切り離して単体テストできる。
    """
    if not anchors:
        return {}
    anchors = sorted(anchors, key=lambda a: a[0])
    first_f, first_b = anchors[0]
    last_f, last_b = anchors[-1]
    out: dict = {}
    for f in range(f_start, f_end + 1):
        if f <= first_f:
            out[f] = list(first_b)
        elif f >= last_f:
            out[f] = list(last_b)
        else:
            for i in range(len(anchors) - 1):
                fa, ba = anchors[i]
                fb, bb = anchors[i + 1]
                if fa <= f <= fb:
                    t = (f - fa) / (fb - fa) if fb > fa else 0.0
                    out[f] = interpolate_bbox(ba, bb, t)
                    break
    return out


def clamp_bbox(box: Sequence[float], width: int, height: int) -> Bbox:
    """bbox を画像範囲 ``[0, width] x [0, height]`` に丸める。順序も正規化する。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, x2, y2]


def round_bbox(box: Sequence[float]) -> List[int]:
    """bbox を最も近い整数ピクセルに丸める (保存用)。"""
    return [int(round(float(v))) for v in box]
