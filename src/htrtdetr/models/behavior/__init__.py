"""
htrtdetr.models.behavior — 行動認識ブランチ

個体検出 (tight OBB, 細粒度) とは役割を分け、広い context・時間情報・社会情報を
使う粗粒度認識を担う。

  * ``roi``            — actor 回転 crop / context 拡大 AABB crop の 2-stream ROI
  * ``kinematics``     — 速度 / 向き / 個体間関係の明示的な特徴
  * ``evidence_head``  — evidence 分解つき single / hierarchical / multi-head 分類器
  * ``branch``         — 上記を束ねた BehaviorBranch
"""

from .branch import BehaviorBranch  # noqa: F401
from .evidence_head import (  # noqa: F401
    BehaviorPrediction,
    EvidenceBehaviorHead,
    FeatureGating,
    HierarchicalClassifier,
    MultiHeadClassifier,
)
from .kinematics import (  # noqa: F401
    INTERACTION_DIM,
    INTERACTION_FEATURE_NAMES,
    MOTION_DIM,
    MOTION_FEATURE_NAMES,
    ORIENTATION_DIM,
    ORIENTATION_FEATURE_NAMES,
    KinematicsEncoder,
    compute_interaction_features,
    compute_motion_features,
    compute_orientation_features,
)
from .roi import ContextROIExtractor, RotatedROIExtractor  # noqa: F401

__all__ = [
    "BehaviorBranch",
    "BehaviorPrediction",
    "EvidenceBehaviorHead",
    "FeatureGating",
    "HierarchicalClassifier",
    "MultiHeadClassifier",
    "KinematicsEncoder",
    "RotatedROIExtractor",
    "ContextROIExtractor",
    "compute_motion_features",
    "compute_orientation_features",
    "compute_interaction_features",
    "MOTION_FEATURE_NAMES",
    "ORIENTATION_FEATURE_NAMES",
    "INTERACTION_FEATURE_NAMES",
    "MOTION_DIM",
    "ORIENTATION_DIM",
    "INTERACTION_DIM",
]
