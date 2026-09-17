from .ht_rtdetr import HTRTDETR, HTRTDETROutput
from .detector import BaseDetector, DetectionOutput, RTDETRDetector, build_detector
from .temporal import HierarchicalTemporalModule
from .id_head import MemoryIDHead, IdentityMemory
from .action_head import ActionHead, InteractionFeatureComputer
from .fusion import (
    DetectionFeatureAdapter,
    MultiHeadFeatureRouter,
    QueryTemporalFusion,
    TrackTemporalFusion,
)
from .behavior import (
    BehaviorBranch,
    BehaviorPrediction,
    EvidenceBehaviorHead,
    KinematicsEncoder,
)


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
    "TrackTemporalFusion",
    # 非推奨 (後方互換のため export は維持する)
    "QueryTemporalFusion",
    "DetectionFeatureAdapter",
    "MultiHeadFeatureRouter",
    "BehaviorBranch",
    "BehaviorPrediction",
    "EvidenceBehaviorHead",
    "KinematicsEncoder",
    "build_model",
]
