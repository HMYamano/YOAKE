"""
htrtdetr.pipeline — Orchestration プレーン (宣言的 DAG)

L1〜L5 の各段を requires/provides の capability トークンで結び、strict/lenient で
依存を検証しつつ窓単位で実行する軽量パイプライン。各段の「中身」は
`htrtdetr.stages` 以下のバックエンドが担い、config から差し替え可能。

構成を config (YAML/dict) から組み立てるには `build_pipeline` / `build_from_yaml`
を使う (これらは import 時に htrtdetr.stages を読み込んで registry を満たす)。
"""

from __future__ import annotations

from .schema import (
    CAP_APPEARANCE_EMB,
    CAP_BOXES,
    CAP_CLASS,
    CAP_DENSE_FEATURES,
    CAP_DET_SCORE,
    CAP_DIRECTED_INTER,
    CAP_IMAGE,
    CAP_KINEMATICS,
    CAP_POSE,
    CAP_STATIC_RELATION,
    CAP_TEMPORAL_STABLE,
    CAP_TRACK_ID,
    KNOWN_CAPABILITIES,
    Detection,
    DirectedInteraction,
    FrameResult,
    FrameWindow,
    Relation,
    Track,
)
from .stage import (
    TIER_DETECTION,
    TIER_INTERACTION,
    TIER_MOTION,
    TIER_STATIC,
    TIER_TRACKING,
    PipelineContext,
    Stage,
    StageBase,
)
from .registry import (
    UnknownBackendError,
    available_backends,
    create,
    get_class,
    is_registered,
    register,
)
from .resolver import InvalidPipelineError, resolve
from .runner import PipelineRunner
from .builder import build_from_yaml, build_pipeline

__all__ = [
    # schema
    "Detection", "Relation", "FrameResult", "DirectedInteraction", "Track", "FrameWindow",
    "KNOWN_CAPABILITIES",
    "CAP_IMAGE", "CAP_BOXES", "CAP_CLASS", "CAP_DET_SCORE", "CAP_TEMPORAL_STABLE",
    "CAP_DENSE_FEATURES", "CAP_STATIC_RELATION", "CAP_TRACK_ID", "CAP_APPEARANCE_EMB",
    "CAP_KINEMATICS", "CAP_POSE", "CAP_DIRECTED_INTER",
    # stage
    "Stage", "StageBase", "PipelineContext",
    "TIER_DETECTION", "TIER_STATIC", "TIER_TRACKING", "TIER_MOTION", "TIER_INTERACTION",
    # registry
    "register", "create", "is_registered", "available_backends", "get_class",
    "UnknownBackendError",
    # resolver / runner
    "resolve", "InvalidPipelineError", "PipelineRunner",
    # builder
    "build_pipeline", "build_from_yaml",
]
