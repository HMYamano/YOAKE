"""
feature_builder.py — Geometric / Motion Feature Builder

各個体の bbox 軌跡から temporal feature を構築する。

特徴ベクトル (per frame, per track):
  [0] cx_norm       : center x (normalized to [0,1])
  [1] cy_norm       : center y
  [2] w_norm        : width (normalized)
  [3] h_norm        : height (normalized)
  [4] area_norm     : area = w*h (normalized to max area 1.0)
  [5] dx            : cx displacement from prev frame (0 if first)
  [6] dy            : cy displacement from prev frame
  [7] speed         : sqrt(dx^2 + dy^2)
  [8] aspect_ratio  : w / (h + 1e-6)
  [9] log_area      : log(area + 1e-6) (log scale でサイズ変化を捉える)

合計: 10 次元 (GEO_FEAT_DIM)

これらの特徴は、視覚的特徴が使えない場合でも動作の時系列パターンを捉えられる。
Stage 2/3 の初期学習で安定した gradient を与える。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .annotation import ObjectAnno, FrameAnno

# Feature dimension (変更する場合は train config の feature_dim も要調整)
GEO_FEAT_DIM = 10


@dataclass
class FeatureBuilderConfig:
    """GeometricFeatureBuilder の設定"""
    # normalize coordinates by image size
    img_width: int = 640
    img_height: int = 640
    # displacement を clamp する最大値 (pixel / frame)
    max_disp: float = 100.0
    # 速度を clamp する最大値
    max_speed: float = 141.0  # sqrt(100^2 + 100^2)
    # area を normalize する基準 (full image area = 1)
    max_area_ratio: float = 1.0


class GeometricFeatureBuilder:
    """
    bbox シーケンスから geometric / motion feature を構築する。

    使い方:
        builder = GeometricFeatureBuilder(cfg)

        # 単一 track のシーケンス
        feat = builder.build_track_sequence(bbox_seq)  # (T, GEO_FEAT_DIM)

        # 複数 track を含む FrameAnno シーケンス全体から一括変換
        track_feats = builder.build_from_frames(frames, video_width, video_height)
        # → Dict[track_id, (T, GEO_FEAT_DIM)]
    """

    def __init__(self, cfg: Optional[FeatureBuilderConfig] = None):
        self.cfg = cfg or FeatureBuilderConfig()

    def build_track_sequence(
        self,
        bbox_seq: List[Optional[List[float]]],
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> np.ndarray:
        """
        1 track の bbox シーケンスから feature matrix を構築する。

        Args:
            bbox_seq: List of [x1, y1, x2, y2] in pixels.
                      None はその frame で track が欠損している場合。
            video_width: 動画幅 (px)。None なら cfg のデフォルト値を使う。
            video_height: 動画高さ (px)。

        Returns:
            features: (T, GEO_FEAT_DIM) float32 ndarray
        """
        W = video_width or self.cfg.img_width
        H = video_height or self.cfg.img_height
        T = len(bbox_seq)
        features = np.zeros((T, GEO_FEAT_DIM), dtype=np.float32)

        prev_cx: Optional[float] = None
        prev_cy: Optional[float] = None

        for t, bbox in enumerate(bbox_seq):
            if bbox is None:
                # 欠損フレーム: 前フレームを引き継ぐ (ゼロのまま)
                if prev_cx is not None:
                    features[t, 0] = prev_cx / W
                    features[t, 1] = prev_cy / H
                continue

            x1, y1, x2, y2 = bbox
            bw = max(x2 - x1, 1e-3)
            bh = max(y2 - y1, 1e-3)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = bw * bh

            # Normalize
            cx_n = cx / W
            cy_n = cy / H
            w_n = bw / W
            h_n = bh / H
            area_n = area / (W * H)

            # Displacement
            if prev_cx is not None:
                dx = (cx - prev_cx) / W
                dy = (cy - prev_cy) / H
            else:
                dx, dy = 0.0, 0.0

            speed = math.sqrt(dx * dx + dy * dy)
            aspect = bw / (bh + 1e-6)
            log_area = math.log(area_n + 1e-6)

            features[t, 0] = cx_n
            features[t, 1] = cy_n
            features[t, 2] = w_n
            features[t, 3] = h_n
            features[t, 4] = area_n
            features[t, 5] = np.clip(dx, -1.0, 1.0)
            features[t, 6] = np.clip(dy, -1.0, 1.0)
            features[t, 7] = np.clip(speed, 0.0, 2.0)
            features[t, 8] = np.clip(aspect, 0.1, 10.0)
            features[t, 9] = np.clip(log_area, -15.0, 0.0)

            prev_cx = cx
            prev_cy = cy

        return features

    def build_from_frames(
        self,
        frames: List[FrameAnno],
        video_width: int,
        video_height: int,
    ) -> Dict[int, np.ndarray]:
        """
        FrameAnno シーケンスから全 track の feature sequence を構築する。

        Returns:
            Dict[track_id, np.ndarray(T, GEO_FEAT_DIM)]
            - track_id が存在しないフレームでは None でパディングして補間する
        """
        # 各 track_id がどのフレームで観測されているか収集
        track_bbox: Dict[int, Dict[int, List[float]]] = {}  # track_id → {frame_idx → bbox}

        for frame in frames:
            for obj in frame.objects:
                if obj.track_id < 0:
                    continue
                if obj.track_id not in track_bbox:
                    track_bbox[obj.track_id] = {}
                track_bbox[obj.track_id][frame.frame_index] = obj.bbox

        if not frames:
            return {}

        frame_indices = [fr.frame_index for fr in frames]
        T = len(frame_indices)

        result: Dict[int, np.ndarray] = {}
        for track_id, bbox_by_frame in track_bbox.items():
            bbox_seq: List[Optional[List[float]]] = []
            for fi in frame_indices:
                bbox_seq.append(bbox_by_frame.get(fi, None))
            result[track_id] = self.build_track_sequence(
                bbox_seq, video_width, video_height
            )

        return result

    def build_from_frames_padded(
        self,
        frames: List[FrameAnno],
        video_width: int,
        video_height: int,
        max_tracks: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        padding 版。バッチ処理向け。

        Returns:
            track_features: (max_tracks, T, GEO_FEAT_DIM)
            track_ids: (max_tracks,) — -1 は padding
            track_mask: (max_tracks,) bool — True: valid track
        """
        track_feats = self.build_from_frames(frames, video_width, video_height)
        T = len(frames)

        features_arr = np.zeros((max_tracks, T, GEO_FEAT_DIM), dtype=np.float32)
        track_ids_arr = np.full(max_tracks, -1, dtype=np.int64)
        mask = np.zeros(max_tracks, dtype=bool)

        for i, (tid, feats) in enumerate(sorted(track_feats.items())):
            if i >= max_tracks:
                break
            t_len = min(feats.shape[0], T)
            features_arr[i, :t_len] = feats[:t_len]
            track_ids_arr[i] = tid
            mask[i] = True

        return features_arr, track_ids_arr, mask

    def get_action_labels_for_tracks(
        self,
        frames: List[FrameAnno],
        track_ids: List[int],
        method: str = "majority",
    ) -> Dict[int, int]:
        """
        指定 track_ids のアクションラベルを収集する。

        Args:
            frames: フレームリスト
            track_ids: ラベルを取得したい track ID のリスト
            method: "majority" (多数決), "last" (最終フレーム), "first"

        Returns:
            Dict[track_id, action_id]  — -1 は未アノテーション
        """
        from collections import Counter

        track_actions: Dict[int, List[int]] = {tid: [] for tid in track_ids}

        for frame in frames:
            for obj in frame.objects:
                if obj.track_id in track_actions and obj.action_id >= 0:
                    track_actions[obj.track_id].append(obj.action_id)

        result: Dict[int, int] = {}
        for tid in track_ids:
            actions = track_actions[tid]
            if not actions:
                result[tid] = -1
            elif method == "majority":
                result[tid] = Counter(actions).most_common(1)[0][0]
            elif method == "last":
                result[tid] = actions[-1]
            elif method == "first":
                result[tid] = actions[0]
            else:
                result[tid] = Counter(actions).most_common(1)[0][0]

        return result


# ---------------------------------------------------------------------------
# Feature 統計の計算 (正規化パラメータ推定用)
# ---------------------------------------------------------------------------

def compute_feature_stats(
    dataset_anno,
    builder: Optional[GeometricFeatureBuilder] = None,
    max_frames: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    データセット全体から feature の mean / std を計算する。
    データセット正規化に使用できる。

    Returns:
        {"mean": (GEO_FEAT_DIM,), "std": (GEO_FEAT_DIM,)}
    """
    if builder is None:
        builder = GeometricFeatureBuilder()

    all_features = []
    frame_count = 0

    for video in dataset_anno.videos:
        if frame_count >= max_frames:
            break
        W, H = video.width or 640, video.height or 640
        track_feats = builder.build_from_frames(video.frames[:50], W, H)
        for feats in track_feats.values():
            all_features.append(feats)
        frame_count += len(video.frames[:50])

    if not all_features:
        return {
            "mean": np.zeros(GEO_FEAT_DIM, dtype=np.float32),
            "std": np.ones(GEO_FEAT_DIM, dtype=np.float32),
        }

    all_arr = np.concatenate(all_features, axis=0)  # (N_total, GEO_FEAT_DIM)
    return {
        "mean": all_arr.mean(axis=0).astype(np.float32),
        "std": (all_arr.std(axis=0) + 1e-6).astype(np.float32),
    }
