from .ht_rtdetr import HTRTDETR, HTRTDETROutput
from .detector import BaseDetector, DetectionOutput, RTDETRDetector, build_detector
from .temporal import HierarchicalTemporalModule
from .id_head import MemoryIDHead, IdentityMemory
from .action_head import ActionHead, InteractionFeatureComputer
from .fusion import QueryTemporalFusion, DetectionFeatureAdapter, MultiHeadFeatureRouter


def build_model(cfg) -> HTRTDETR:
    """config から統合モデルを構築するファクトリ関数"""
    return HTRTDETR(cfg)


__all__ = [
    "HTRTDETR",
    "HTRTDETROutput",
    "BaseDetector",
    "DetectionOutput",
    "RTDETRDetector",
    "build_detector",
    "HierarchicalTemporalModule",
    "MemoryIDHead",
    "IdentityMemory",
    "ActionHead",
    "InteractionFeatureComputer",
    "QueryTemporalFusion",
    "DetectionFeatureAdapter",
    "MultiHeadFeatureRouter",
    "build_model",
]
