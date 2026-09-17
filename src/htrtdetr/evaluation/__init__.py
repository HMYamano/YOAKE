"""
htrtdetr.evaluation — 評価指標

``evaluator`` は既存の指標 (AP / IDF1 / action accuracy / runtime) をそのまま維持し、
``advanced`` が OBB・追跡・multi-head・性能の指標を追加する。既存の import は
変わらないので、旧スクリプトはそのまま動く。
"""

from .advanced import (
    BehaviorEvaluator,
    MOTEvaluator,
    MultiHeadActionEvaluator,
    PerformanceEvaluator,
    RotatedDetectionEvaluator,
)
from .evaluator import (
    ActionEvaluator,
    DetectionEvaluator,
    HTRTDETREvaluator,
    RuntimeEvaluator,
    SimpleTrackingEvaluator,
    TrackingEvaluator,
)

__all__ = [
    # 既存
    "DetectionEvaluator",
    "TrackingEvaluator",
    "SimpleTrackingEvaluator",
    "ActionEvaluator",
    "RuntimeEvaluator",
    "HTRTDETREvaluator",
    # 追加
    "RotatedDetectionEvaluator",
    "MOTEvaluator",
    "BehaviorEvaluator",
    "MultiHeadActionEvaluator",
    "PerformanceEvaluator",
]
