"""
htrtdetr.stages — 各レイヤーのバックエンド実装 (段の「中身」)

このパッケージを import すると全段が pipeline.registry に登録される。
古典アルゴリズム段 (numpy 完結: 安定化 / 幾何 / bytetrack / 運動 / heuristic) は
依存無しで動く。神経系段 (rtdetr / fused / memory_id / learned_pair) は torch と
checkpoint を必要とし、遅延 import で setup 時に初めて重い依存を触る。
"""

from __future__ import annotations

# --- 検出 (L1) ---
from .detection import rtdetr_stage, fused_stage, yolo_stage  # noqa: F401
# --- 安定化 (L1b) ---
from .stabilization import temporal_nms_stage  # noqa: F401
# --- 静的関係 (L2) ---
from .relation import geometric_stage  # noqa: F401
# --- 追跡 (L3) ---
from .tracking import bytetrack_stage, memory_id_stage  # noqa: F401
# --- 運動 (L4) ---
from .motion import kinematic_stage  # noqa: F401
# --- pose (任意) ---
from .pose import keypoint_stage  # noqa: F401
# --- 相互作用 (L5) ---
from .interaction import heuristic_stage, learned_pair_stage, gnn_stage  # noqa: F401
