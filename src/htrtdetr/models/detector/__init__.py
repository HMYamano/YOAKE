from .base import BaseDetector, DetectionOutput
from .rtdetr_wrapper import RTDETRDetector, ResNetBackbone, FPN


def build_detector(cfg) -> BaseDetector:
    """config から detector を構築するファクトリ関数"""
    return RTDETRDetector(cfg)


__all__ = [
    "BaseDetector",
    "DetectionOutput",
    "RTDETRDetector",
    "ResNetBackbone",
    "FPN",
    "build_detector",
]
