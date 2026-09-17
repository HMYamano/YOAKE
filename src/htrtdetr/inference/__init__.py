"""
htrtdetr.inference — 推論エントリポイント

2 つのモードを用意し、``InferenceConfig.mode`` で選ぶ:

  * ``StreamingInferencer`` (mode="streaming", 既定) — 各フレームを detector に
    1 回だけ通し、過去は track ごとの ring buffer から再利用する。窓長に依存しない。
  * ``Inferencer`` (mode="batch") — 毎フレーム窓全体を通す従来実装。
    数値比較・回帰検証のために残してある。
"""

from .inferencer import Inferencer, VideoWriter, draw_detections
from .streaming import LatencyBreakdown, StreamingInferencer, StreamingResult

__all__ = [
    "Inferencer",
    "StreamingInferencer",
    "StreamingResult",
    "LatencyBreakdown",
    "draw_detections",
    "VideoWriter",
]
