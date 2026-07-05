"""
predictions.py — YOAKE 推論出力 (predictions.json) ローダ (dearpygui 非依存)
==========================================================================

``yoake predict`` / ``yoake pipeline`` が書き出す ``predictions.json`` を読み込む。
Video Analysis GUI とテストの両方から使う純モジュール。

スキーマ (Inferencer.run が出力):
    { "<frame_idx>": {
        "boxes":      [[x1, y1, x2, y2], ...],   # 元画像ピクセル座標
        "scores":     [float, ...],
        "track_ids":  [int, ...],
        "action_ids": [int, ...],
        "class_ids":  [int, ...]                  # 省略される場合あり
    }, ... }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class FrameDetections:
    """1 フレーム分の検出結果 (並列リスト、全て同じ長さ)。"""

    __slots__ = ("boxes", "scores", "track_ids", "action_ids", "class_ids")

    def __init__(
        self,
        boxes: List[List[float]],
        scores: List[float],
        track_ids: List[int],
        action_ids: List[int],
        class_ids: List[int],
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.track_ids = track_ids
        self.action_ids = action_ids
        self.class_ids = class_ids

    def __len__(self) -> int:
        return len(self.boxes)


def load_predictions(path: str) -> Dict[int, FrameDetections]:
    """``predictions.json`` を読み込み ``{frame_index: FrameDetections}`` を返す。

    欠損フィールドは box 数に合わせて既定値 (score=1.0, track_id=-1, action_id=-1,
    class_id=0) で補完する。ファイルが読めない/壊れている場合は例外を送出する。
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[int, FrameDetections] = {}
    for key, val in raw.items():
        try:
            fi = int(key)
        except (TypeError, ValueError):
            continue
        boxes = [[float(v) for v in b] for b in (val.get("boxes") or [])]
        n = len(boxes)
        out[fi] = FrameDetections(
            boxes=boxes,
            scores=_as_list(val.get("scores"), n, 1.0, float),
            track_ids=_as_list(val.get("track_ids"), n, -1, int),
            action_ids=_as_list(val.get("action_ids"), n, -1, int),
            class_ids=_as_list(val.get("class_ids"), n, 0, int),
        )
    return out


def _as_list(value, n: int, default, cast):
    if value is None:
        return [default] * n
    out = [cast(v) for v in value]
    # box 数と食い違う場合は既定値で埋める/切り詰める
    if len(out) < n:
        out = out + [default] * (n - len(out))
    elif len(out) > n:
        out = out[:n]
    return out


def frame_range(preds: Dict[int, FrameDetections]) -> Tuple[int, int]:
    """予測が存在するフレーム番号の (最小, 最大)。空なら (0, 0)。"""
    if not preds:
        return (0, 0)
    keys = preds.keys()
    return (min(keys), max(keys))


def build_track_timelines(preds: Dict[int, FrameDetections]) -> Dict[int, Dict[int, int]]:
    """個体別の行動タイムライン ``{track_id: {frame_index: action_id}}`` を作る。"""
    timelines: Dict[int, Dict[int, int]] = {}
    for fi, fd in preds.items():
        for tid, aid in zip(fd.track_ids, fd.action_ids):
            timelines.setdefault(int(tid), {})[fi] = int(aid)
    return timelines
