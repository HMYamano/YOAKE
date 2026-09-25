"""
htrtdetr.inference — 推論エントリポイント

3 つのモードを用意する:

  * ``StreamingInferencer`` (mode="streaming", 既定) — 各フレームを detector に
    1 回だけ通し、過去は track ごとの ring buffer から再利用する。窓長に依存しない。
  * ``Inferencer`` (mode="batch") — 毎フレーム窓全体を通す従来実装。
    数値比較・回帰検証のために残してある。
  * ``BatchedDetector`` / ``run_detect_only`` (``detect_only=true``) — detector
    だけを複数フレームまとめて通す。ID・行動・注釈動画を作らない代わりに、
    同じ GPU・同じ重みで実測 4.8 倍 (73.7 → 350.6 fps)。アノテーションの
    候補生成が欲しいのは箱だけなので、ループはこれを使う。
"""

from .detect_only import DEFAULT_BATCH, BatchedDetector, run_detect_only
from .inferencer import Inferencer, VideoWriter, draw_detections
from .streaming import LatencyBreakdown, StreamingInferencer, StreamingResult

__all__ = [
    "Inferencer",
    "BatchedDetector",
    "run_detect_only",
    "DEFAULT_BATCH",
    "StreamingInferencer",
    "StreamingResult",
    "LatencyBreakdown",
    "draw_detections",
    "VideoWriter",
]
