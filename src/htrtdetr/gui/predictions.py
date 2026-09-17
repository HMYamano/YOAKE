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
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FrameDetections:
    """
    1 フレーム分の検出結果 (並列リスト、全て同じ長さ)。

    v2.0 で ``obb`` / ``context_boxes`` / ``evidence`` / ``gates`` /
    ``head_predictions`` を任意フィールドとして追加した。いずれも無いファイル
    (旧 predictions.json) は空リストとして読まれるので、GUI は分岐なしで扱える。
    """

    __slots__ = (
        "boxes", "scores", "track_ids", "action_ids", "class_ids",
        "obb", "context_boxes", "evidence", "gates", "head_predictions",
    )

    def __init__(
        self,
        boxes: List[List[float]],
        scores: List[float],
        track_ids: List[int],
        action_ids: List[int],
        class_ids: List[int],
        obb: Optional[List[List[float]]] = None,
        context_boxes: Optional[List[List[float]]] = None,
        evidence: Optional[List[Dict[str, float]]] = None,
        gates: Optional[List[Dict[str, float]]] = None,
        head_predictions: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.track_ids = track_ids
        self.action_ids = action_ids
        self.class_ids = class_ids
        self.obb = obb or []
        self.context_boxes = context_boxes or []
        self.evidence = evidence or []
        self.gates = gates or []
        self.head_predictions = head_predictions or []

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def has_obb(self) -> bool:
        return bool(self.obb)

    def obb_of(self, i: int) -> Optional[List[float]]:
        """i 番目の OBB [cx, cy, w, h, theta]。無ければ None。"""
        return self.obb[i] if i < len(self.obb) else None

    def evidence_of(self, i: int) -> Dict[str, float]:
        """i 番目の判断根拠 {group: logit 寄与}。無ければ空。"""
        return self.evidence[i] if i < len(self.evidence) else {}

    def gates_of(self, i: int) -> Dict[str, float]:
        return self.gates[i] if i < len(self.gates) else {}

    def heads_of(self, i: int) -> Dict[str, str]:
        return self.head_predictions[i] if i < len(self.head_predictions) else {}


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
            obb=[[float(v) for v in o] for o in (val.get("obb") or [])],
            context_boxes=[
                [float(v) for v in b] for b in (val.get("context_boxes") or [])
            ],
            evidence=list(val.get("evidence") or []),
            gates=list(val.get("gates") or []),
            head_predictions=list(val.get("head_predictions") or []),
        )
    return out


def obb_corners(obb: List[float]) -> List[Tuple[float, float]]:
    """
    [cx, cy, w, h, theta] → 4 頂点 (反時計回り)。GUI 描画用の純関数。

    ``geometry/obb.py`` の ``obb_to_corners`` と同じ定義 (torch 非依存の純 Python 版)。
    """
    cx, cy, w, h, th = (float(v) for v in obb[:5])
    hw, hh = w / 2, h / 2
    c, s = math.cos(th), math.sin(th)
    return [
        (cx + lx * c - ly * s, cy + lx * s + ly * c)
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def build_track_trajectories(
    preds: Dict[int, FrameDetections],
    max_points: int = 120,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    個体ごとの軌跡 ``{track_id: [(frame_index, cx, cy), ...]}`` を作る。

    Video Analysis GUI の trajectory 表示に使う。長すぎる軌跡は末尾
    ``max_points`` 点に切る (描画負荷を抑えるため)。
    """
    traj: Dict[int, List[Tuple[int, float, float]]] = {}
    for fi in sorted(preds):
        fd = preds[fi]
        for i, (tid, box) in enumerate(zip(fd.track_ids, fd.boxes)):
            o = fd.obb_of(i)
            if o is not None:
                cx, cy = float(o[0]), float(o[1])
            else:
                cx = (float(box[0]) + float(box[2])) / 2
                cy = (float(box[1]) + float(box[3])) / 2
            traj.setdefault(int(tid), []).append((fi, cx, cy))
    return {k: v[-max_points:] for k, v in traj.items()}


def build_head_timelines(
    preds: Dict[int, FrameDetections],
) -> Dict[str, Dict[int, Dict[int, str]]]:
    """
    multi-head 用のタイムライン ``{head: {track_id: {frame: class_name}}}``。

    locomotion / social / self-directed を別レーンとして表示するために使う。
    """
    out: Dict[str, Dict[int, Dict[int, str]]] = {}
    for fi, fd in preds.items():
        for i, tid in enumerate(fd.track_ids):
            for head, cls in fd.heads_of(i).items():
                out.setdefault(head, {}).setdefault(int(tid), {})[fi] = cls
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
