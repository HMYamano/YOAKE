"""
htrtdetr.models.detector — detector backend の登録と生成

``DetectorConfig.type`` で backend を選ぶ:

    DetectorBackend
     ├─ rtdetr_aabb  — 従来の軸平行 RT-DETR (既定・完全後方互換)
     ├─ rtdetr_obb   — RT-DETR + 角度ヘッド (Oriented Bounding Box)
     ├─ yolo_aabb    — 外部 YOLO (stages プレーン経由)
     └─ yolo_obb     — 外部 YOLO-OBB (stages プレーン経由)

rtdetr_aabb と rtdetr_obb は同じ ``RTDETRDetector`` クラスで、``cfg.obb.enabled``
だけが違う。transformer query / detector feature / downstream の ID・temporal・
behavior との統合性を保つため、OBB 化で別クラスへ分岐させない設計にしている。

yolo_* は torch モデルとして本体に組み込むのではなく、orchestration プレーン
(``stages/detection/yolo_stage.py``) の段として差し替える。YOLO は query 特徴を
出さないので、dense_features を要求する下流段とは resolver が自動で整合を取る。
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import BaseDetector, DetectionOutput
from .rtdetr_wrapper import RTDETRDetector, ResNetBackbone, FPN

# backend 名 → ファクトリ
_DETECTOR_BACKENDS: Dict[str, Callable[..., BaseDetector]] = {}


def register_detector(name: str):
    """detector backend を名前付きで登録するデコレータ。"""
    def _wrap(fn: Callable[..., BaseDetector]) -> Callable[..., BaseDetector]:
        _DETECTOR_BACKENDS[name] = fn
        return fn
    return _wrap


@register_detector("rtdetr_aabb")
def _build_rtdetr_aabb(cfg) -> BaseDetector:
    return RTDETRDetector(cfg)


@register_detector("rtdetr_obb")
def _build_rtdetr_obb(cfg) -> BaseDetector:
    if not (getattr(cfg, "obb", None) and cfg.obb.enabled):
        raise ValueError(
            "detector.type='rtdetr_obb' には detector.obb.enabled=true が必要です "
            "(config.enable_obb() を使うと関連フラグをまとめて整合させられます)"
        )
    return RTDETRDetector(cfg)


def available_detectors() -> list:
    """登録済み backend 名の一覧。"""
    return sorted(_DETECTOR_BACKENDS)


def build_detector(cfg) -> BaseDetector:
    """
    config から detector を構築するファクトリ関数。

    ``cfg.type`` が無い古い config (dataclass 更新前に pickle された等) は
    ``rtdetr_aabb`` として扱うので、後方互換が保たれる。
    """
    backend = getattr(cfg, "type", "rtdetr_aabb") or "rtdetr_aabb"

    if backend.startswith("yolo"):
        raise ValueError(
            f"detector.type='{backend}' は本体モデルに埋め込む backend ではありません。"
            "YOLO 系は orchestration プレーンの段として使ってください "
            "(configs/pipeline/yolo_bytetrack.yaml を参照)。"
        )
    if backend not in _DETECTOR_BACKENDS:
        raise ValueError(
            f"未知の detector backend '{backend}'。利用可能: {available_detectors()}"
        )
    return _DETECTOR_BACKENDS[backend](cfg)


__all__ = [
    "BaseDetector",
    "DetectionOutput",
    "RTDETRDetector",
    "ResNetBackbone",
    "FPN",
    "build_detector",
    "register_detector",
    "available_detectors",
]
