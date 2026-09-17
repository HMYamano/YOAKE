"""
htrtdetr.tracking — OBB 対応の追跡と track 単位の特徴キャッシュ

YOAKE のデータフローで「検出」と「行動認識」を繋ぐ層。ここで付いた track_id が
時間モデルの整列軸になる (DETR の query index ではなく)。

  * ``OBBAssociator``    — rotated IoU / appearance / motion / orientation の
                           重み付き和による detection ↔ track 関連付け
  * ``TrackFeatureCache``— track ごとのリングバッファ。streaming 推論で過去フレームの
                           再計算を不要にする
  * ``OnlineTracker``    — 上記 2 つを束ねた最小のオンライン追跡器
"""

from .assembly import (  # noqa: F401
    FrameDetections,
    TrackAlignedBatch,
    build_track_aligned_history,
)
from .associator import AssociationResult, OBBAssociator  # noqa: F401
from .track_buffer import TrackFeatureCache, TrackState  # noqa: F401
from .tracker import OnlineTracker, TrackedObject  # noqa: F401

__all__ = [
    "OBBAssociator",
    "AssociationResult",
    "TrackFeatureCache",
    "TrackState",
    "OnlineTracker",
    "TrackedObject",
    "FrameDetections",
    "TrackAlignedBatch",
    "build_track_aligned_history",
]
