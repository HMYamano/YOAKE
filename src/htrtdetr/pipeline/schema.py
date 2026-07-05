"""
schema.py — パイプラインを流れる展開データ形式 (Orchestration プレーン)

設計原則:
- `track_id` / `timestamp` などは「あれば付く」オプション項目。空欄は `Optional`
  で表しつつ、どの capability が埋まっているかは **文字列トークン集合**
  (`FrameWindow.available`) で管理する (null 判定に依存しない)。
- per-frame 段 (L1, L2) は各フレームに map する。temporal 段 (L3, L4, L5) は
  窓 (FrameWindow) 全体を見る。処理単位は常に窓に統一する。

capability トークン語彙は requires/provides の共通語彙になる。文字列で緩く扱い、
既知トークン集合は `registry` 側で検証して打鍵ミスに早期に気づけるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# capability トークン語彙
# ---------------------------------------------------------------------------
CAP_IMAGE = "image"
CAP_BOXES = "boxes"
CAP_CLASS = "class_id"
CAP_DET_SCORE = "det_score"
CAP_TEMPORAL_STABLE = "temporal_stable"     # L1b: 窓内で時系列安定化済みの検出 (IDは付かない)
CAP_DENSE_FEATURES = "dense_features"       # DETR系の query/encoder 特徴 (YOLOは出さない)
CAP_STATIC_RELATION = "static_relation"     # L2
CAP_TRACK_ID = "track_id"                   # L3
CAP_APPEARANCE_EMB = "appearance_embedding" # L3 (任意)
CAP_KINEMATICS = "kinematics"               # L4: 速度/向き/軌跡
CAP_POSE = "pose"                           # 姿勢/向き (任意段)
CAP_DIRECTED_INTER = "directed_interaction" # L5

# 既知 capability トークン (registry / resolver がタイポ検出に使う)
KNOWN_CAPABILITIES: Set[str] = {
    CAP_IMAGE, CAP_BOXES, CAP_CLASS, CAP_DET_SCORE, CAP_TEMPORAL_STABLE,
    CAP_DENSE_FEATURES, CAP_STATIC_RELATION, CAP_TRACK_ID, CAP_APPEARANCE_EMB,
    CAP_KINEMATICS, CAP_POSE, CAP_DIRECTED_INTER,
}


# ---------------------------------------------------------------------------
# 展開データレコード
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """1フレーム内の1物体。段を進めるごとにフィールドが埋まっていく。"""
    # --- L1 が必ず埋める ---
    bbox: np.ndarray                                 # [x1, y1, x2, y2] (画素)
    score: float
    class_id: int
    # --- 以降は「あれば付く」オプション ---
    interpolated: bool = False                       # L1b: 欠損補間で復元した箱なら True
    raw_score: Optional[float] = None                # L1b: 平滑化前の元スコア (監査用)
    track_id: Optional[int] = None                   # L3 が付与
    velocity: Optional[np.ndarray] = None            # L4: [dx, dy] (px/frame)
    heading: Optional[float] = None                  # L4: 進行方向 rad (速度 or pose 由来)
    speed: Optional[float] = None                    # L4
    keypoints: Optional[np.ndarray] = None           # pose段: (K, 3) = x,y,conf
    orientation: Optional[float] = None              # pose段: 体軸方向 rad
    embedding: Optional[np.ndarray] = None           # L3: 見た目埋め込み
    dense_feature: Optional[np.ndarray] = None       # L1(DETR): query特徴 (L3/L5 神経系が使う)

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) * 0.5)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) * 0.5)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float32)


@dataclass
class Relation:
    """L2 の(準)対称な位置関係。近接/重なり/上下 など。"""
    src: int                        # Detection index (または track_id)
    dst: int
    kind: str                       # "near" | "overlap" | "above" | "below" | ...
    value: float                    # 距離や IoU などの連続値
    directed: bool = False          # 上下は有向。近接/重なりは対称 (src<dst で1本に正規化可)


@dataclass
class FrameResult:
    """1フレームの解析結果。per-frame段(L1,L2)の入出力単位。"""
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)      # L2
    timestamp: Optional[float] = None                            # あれば付く
    image: Optional[np.ndarray] = None                           # 生フレーム (任意保持)
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class DirectedInteraction:
    """L5 の出力: 有向ペア (A→B ≠ B→A) の役割付き相互作用。"""
    agent_track_id: int
    patient_track_id: int
    relation: str                   # "court" | "chase" | "mount" | "investigate" | ...
    score: float
    t_start: int                    # frame_index
    t_end: int


@dataclass
class Track:
    """L4/L5 が扱う「窓内でのトラック集約」。"""
    track_id: int
    frames: List[int] = field(default_factory=list)              # 観測フレーム
    centers: Optional[np.ndarray] = None                         # (T, 2)
    velocities: Optional[np.ndarray] = None                      # (T, 2)
    headings: Optional[np.ndarray] = None                        # (T,)
    speeds: Optional[np.ndarray] = None                          # (T,)


@dataclass
class FrameWindow:
    """段を流れる主単位。per-frame段は各フレームにmap、temporal段は窓全体を見る。"""
    frames: List[FrameResult]
    tracks: dict = field(default_factory=dict)                       # {track_id: Track}
    interactions: List[DirectedInteraction] = field(default_factory=list)
    # この窓で現在利用可能な capability (runner が更新)
    available: Set[str] = field(default_factory=set)

    # ---- 便利アクセサ ---------------------------------------------------
    def iter_detections(self):
        """(frame_index, Detection) を順に yield する。"""
        for fr in self.frames:
            for det in fr.detections:
                yield fr.frame_index, det

    def num_detections(self) -> int:
        return sum(len(fr.detections) for fr in self.frames)

    def frame_by_index(self, frame_index: int) -> Optional[FrameResult]:
        for fr in self.frames:
            if fr.frame_index == frame_index:
                return fr
        return None
