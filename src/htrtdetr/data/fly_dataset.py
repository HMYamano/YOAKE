"""
fly_dataset.py — ショウジョウバエ行動解析用 Dataset クラス群

クラス一覧:
- FlyDetectionDataset   : Stage 1 用。単フレーム + bbox GT。
- TrackWindowDataset    : Stage 2/3 用。1 track × T フレームのウィンドウ。
- SceneSequenceDataset  : Stage 4 用。シーン全体のシーケンス。

設計方針:
- 可変個体数は pad せず list で返す (collate_fn で処理)
- 視覚的特徴 (image tensor) と幾何学的特徴 (geo_features) を両方返す
- annotation が存在しない場合は DummyMode で代替データを返す
- image_path が存在しない場合はゼロ画像で代替 (開発時のフォールバック)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .annotation import DatasetAnno, FrameAnno, ObjectAnno, load_annotation
from .feature_builder import GeometricFeatureBuilder, FeatureBuilderConfig, GEO_FEAT_DIM


# ---------------------------------------------------------------------------
# 画像読み込み helper
# ---------------------------------------------------------------------------

def _load_image_rgb(path: str, fallback_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    画像を (H, W, 3) RGB uint8 で読み込む。
    ファイルが存在しない場合はゼロ画像を返す。
    """
    if not path or not Path(path).exists():
        return np.zeros((*fallback_size, 3), dtype=np.uint8)
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros((*fallback_size, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        return np.zeros((*fallback_size, 3), dtype=np.uint8)


def _resize_image(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """(H, W, 3) → target_hw にリサイズ"""
    tH, tW = target_hw
    if image.shape[:2] == (tH, tW):
        return image
    try:
        from PIL import Image
        pil = Image.fromarray(image)
        return np.array(pil.resize((tW, tH), Image.BILINEAR), dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2
        return cv2.resize(image, (tW, tH), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from skimage.transform import resize as sk_resize
        return (sk_resize(image, (tH, tW)) * 255).astype(np.uint8)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32, ImageNet normalized"""
    img = image.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Simple color jitter augmentation (PIL optional)
# ---------------------------------------------------------------------------

def _augment_image(image: np.ndarray) -> np.ndarray:
    """軽量な色拡張。PIL があれば適用、なければそのまま返す。"""
    try:
        from PIL import Image, ImageEnhance
        pil = Image.fromarray(image)
        if random.random() < 0.5:
            pil = ImageEnhance.Brightness(pil).enhance(1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            pil = ImageEnhance.Contrast(pil).enhance(1.0 + random.uniform(-0.2, 0.2))
        return np.array(pil)
    except ImportError:
        return image


# ---------------------------------------------------------------------------
# Stage 1: FlyDetectionDataset
# ---------------------------------------------------------------------------

class FlyDetectionDataset(Dataset):
    """
    単フレーム bbox 検出 Dataset (Stage 1)。

    返り値 (1 サンプル):
        image:      (3, H, W) float32 tensor
        boxes:      (N, 4) float32 tensor — [x1,y1,x2,y2] normalized [0,1]
        class_ids:  (N,) int64 tensor
        track_ids:  (N,) int64 tensor  (optional, -1 if unknown)
        action_ids: (N,) int64 tensor  (optional, -1 if unknown)
        geo_features: (N, GEO_FEAT_DIM) float32 tensor — per-object geometric features
        meta: dict  (video_id, frame_index, image_path, orig_size)
    """

    def __init__(
        self,
        anno: DatasetAnno,
        image_size: Tuple[int, int] = (640, 640),
        data_root: str = "",
        augment: bool = True,
        hflip_prob: float = 0.5,
        min_bbox_area: float = 4.0,    # pixel^2 以下の bbox は除外
    ):
        self.anno = anno
        self.image_size = image_size  # (H, W)
        self.data_root = Path(data_root)
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.min_bbox_area = min_bbox_area

        # フラットなフレームリスト (video_id, FrameAnno, video_width, video_height)
        self.samples: List[Tuple[str, FrameAnno, int, int]] = []
        for video in anno.videos:
            W, H = video.width or image_size[1], video.height or image_size[0]
            for frame in video.frames:
                # GT box がないフレームはスキップしない (negative sample として使う)
                self.samples.append((video.video_id, frame, W, H))

        self.geo_builder = GeometricFeatureBuilder(
            FeatureBuilderConfig(img_width=image_size[1], img_height=image_size[0])
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, frame, v_width, v_height = self.samples[idx]
        H_tgt, W_tgt = self.image_size

        # 画像読み込み
        img_path = str(self.data_root / frame.image_path) if self.data_root.name else frame.image_path
        image = _load_image_rgb(img_path, (H_tgt, W_tgt))
        orig_h, orig_w = image.shape[:2]

        # scale factor (pixel → normalized)
        sx = W_tgt / (v_width or orig_w)
        sy = H_tgt / (v_height or orig_h)

        # GT bbox を image_size にスケール
        boxes = []
        class_ids = []
        track_ids = []
        action_ids = []

        for obj in frame.objects:
            x1, y1, x2, y2 = obj.bbox
            # スケール
            x1s, y1s, x2s, y2s = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            bw = x2s - x1s
            bh = y2s - y1s
            if bw * bh < self.min_bbox_area:
                continue
            # クランプ
            x1s = max(0.0, min(W_tgt, x1s))
            y1s = max(0.0, min(H_tgt, y1s))
            x2s = max(0.0, min(W_tgt, x2s))
            y2s = max(0.0, min(H_tgt, y2s))
            if x2s <= x1s or y2s <= y1s:
                continue
            boxes.append([x1s, y1s, x2s, y2s])
            class_ids.append(obj.class_id)
            track_ids.append(obj.track_id)
            action_ids.append(obj.action_id)

        # Augmentation
        do_hflip = self.augment and random.random() < self.hflip_prob
        if do_hflip:
            image = image[:, ::-1].copy()
            for i, b in enumerate(boxes):
                boxes[i] = [W_tgt - b[2], b[1], W_tgt - b[0], b[3]]

        if self.augment:
            image = _augment_image(image)

        # リサイズ
        image = _resize_image(image, self.image_size)

        # Normalize bbox
        boxes_norm = np.array(boxes, dtype=np.float32) / np.array(
            [W_tgt, H_tgt, W_tgt, H_tgt], dtype=np.float32
        ) if boxes else np.zeros((0, 4), dtype=np.float32)

        # Geometric features (単フレームなので dx=0, speed=0)
        geo_feats = np.zeros((len(boxes_norm), GEO_FEAT_DIM), dtype=np.float32)
        for i, b in enumerate(boxes_norm):
            x1n, y1n, x2n, y2n = b
            cx_n = (x1n + x2n) / 2
            cy_n = (y1n + y2n) / 2
            w_n = x2n - x1n
            h_n = y2n - y1n
            area_n = w_n * h_n
            import math
            geo_feats[i, 0] = cx_n
            geo_feats[i, 1] = cy_n
            geo_feats[i, 2] = w_n
            geo_feats[i, 3] = h_n
            geo_feats[i, 4] = area_n
            # dx, dy, speed = 0 (frame i のみ)
            geo_feats[i, 8] = w_n / (h_n + 1e-6)
            geo_feats[i, 9] = math.log(area_n + 1e-6)

        return {
            "image": _to_tensor(image),                               # (3, H, W)
            "boxes": torch.from_numpy(boxes_norm),                    # (N, 4)
            "class_ids": torch.tensor(class_ids, dtype=torch.long),  # (N,)
            "track_ids": torch.tensor(track_ids, dtype=torch.long),  # (N,)
            "action_ids": torch.tensor(action_ids, dtype=torch.long),# (N,)
            "geo_features": torch.from_numpy(geo_feats),              # (N, GEO_FEAT_DIM)
            "meta": {
                "video_id": video_id,
                "frame_index": frame.frame_index,
                "image_path": frame.image_path,
                "orig_size": (orig_h, orig_w),
            },
        }


# ---------------------------------------------------------------------------
# Stage 2/3: TrackWindowDataset
# ---------------------------------------------------------------------------

class TrackWindowDataset(Dataset):
    """
    per-track スライディングウィンドウ Dataset (Stage 2 / Stage 3)。

    各サンプルは「1 個体 × T フレーム」のウィンドウ。
    - Stage 2: action 分類学習
    - Stage 3: ID 識別学習

    返り値 (1 サンプル):
        images:        (T, 3, H, W) float32 tensor
        geo_features:  (T, GEO_FEAT_DIM) float32 tensor
        boxes_seq:     (T, 4) float32 tensor — normalized [x1,y1,x2,y2]
        action_id:     int — majority action in window (-1 if unannotated)
        track_id:      int — (global) track ID
        local_track_id: int — within-video remapped track ID [0, N_tracks)
        video_id:      str
        frame_indices: List[int]
    """

    def __init__(
        self,
        anno: DatasetAnno,
        window_size: int = 16,
        stride: int = 8,
        image_size: Tuple[int, int] = (640, 640),
        data_root: str = "",
        augment: bool = True,
        min_track_length: int = 4,      # window_size 未満の track はスキップ
        require_action_label: bool = False,
    ):
        self.anno = anno
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.data_root = Path(data_root)
        self.augment = augment

        self.geo_builder = GeometricFeatureBuilder(
            FeatureBuilderConfig(img_width=image_size[1], img_height=image_size[0])
        )

        # サンプルインデックスを構築
        # サンプル = (video_id, track_id, local_track_id, start_frame_idx, frames_with_obj)
        self.samples: List[Dict[str, Any]] = []

        for video in anno.videos:
            W, H = video.width or image_size[1], video.height or image_size[0]

            # track_id → [(frame_index, frame_anno, object_anno)] の対応を構築
            track_data: Dict[int, List[Tuple[int, FrameAnno, ObjectAnno]]] = defaultdict(list)
            for frame in video.frames:
                for obj in frame.objects:
                    if obj.track_id < 0:
                        continue
                    track_data[obj.track_id].append((frame.frame_index, frame, obj))

            # 各 track を frame_index でソート
            for tid, frames_objs in track_data.items():
                frames_objs.sort(key=lambda x: x[0])

                if len(frames_objs) < min_track_length:
                    continue

                # スライディングウィンドウ
                for start in range(0, len(frames_objs) - window_size + 1, stride):
                    window = frames_objs[start:start + window_size]

                    if require_action_label:
                        has_action = any(obj.action_id >= 0 for _, _, obj in window)
                        if not has_action:
                            continue

                    self.samples.append({
                        "video_id": video.video_id,
                        "track_id": tid,
                        "video_width": W,
                        "video_height": H,
                        "window": window,  # List[(frame_index, FrameAnno, ObjectAnno)]
                    })

        # ビデオ内の track_id を [0, N_tracks) に remap (ID 分類用)
        self._build_track_id_map()

    def _build_track_id_map(self) -> None:
        """video_id ごとの track_id → local_id マッピングを構築"""
        # {video_id: {track_id: local_id}}
        self.track_local_id_map: Dict[str, Dict[int, int]] = {}
        for s in self.samples:
            vid = s["video_id"]
            tid = s["track_id"]
            if vid not in self.track_local_id_map:
                self.track_local_id_map[vid] = {}
            if tid not in self.track_local_id_map[vid]:
                n = len(self.track_local_id_map[vid])
                self.track_local_id_map[vid][tid] = n

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        window = sample["window"]
        T = len(window)
        W = sample["video_width"]
        H = sample["video_height"]
        H_tgt, W_tgt = self.image_size

        # bbox シーケンス構築
        bbox_seq = [list(obj.bbox) for _, _, obj in window]  # List[(4,)]

        # 水平 flip 拡張 (T フレーム全体で一貫させる)
        do_hflip = self.augment and random.random() < 0.5
        if do_hflip:
            for i, b in enumerate(bbox_seq):
                x1, y1, x2, y2 = b
                bbox_seq[i] = [W - x2, y1, W - x1, y2]

        # Geometric features
        geo_feats = self.geo_builder.build_track_sequence(bbox_seq, W, H)  # (T, GEO_FEAT_DIM)

        # normalized boxes (xyxy)
        boxes_seq = np.array(bbox_seq, dtype=np.float32)
        sx = W_tgt / W
        sy = H_tgt / H
        boxes_seq[:, [0, 2]] *= sx / W_tgt  # pixel → normalized
        boxes_seq[:, [1, 3]] *= sy / H_tgt
        boxes_seq = np.clip(boxes_seq, 0, 1)

        # 画像読み込み
        images = []
        for _, frame, _ in window:
            img_path = str(self.data_root / frame.image_path) \
                if self.data_root.name else frame.image_path
            img = _load_image_rgb(img_path, (H_tgt, W_tgt))
            if do_hflip:
                img = img[:, ::-1].copy()
            if self.augment:
                img = _augment_image(img)
            img = _resize_image(img, self.image_size)
            images.append(_to_tensor(img))

        # Action label (majority)
        action_ids_in_window = [obj.action_id for _, _, obj in window if obj.action_id >= 0]
        if action_ids_in_window:
            from collections import Counter
            action_id = Counter(action_ids_in_window).most_common(1)[0][0]
        else:
            action_id = -1

        # Local track ID
        vid = sample["video_id"]
        tid = sample["track_id"]
        local_id = self.track_local_id_map.get(vid, {}).get(tid, -1)

        return {
            "images": torch.stack(images, dim=0),                     # (T, 3, H, W)
            "geo_features": torch.from_numpy(geo_feats),              # (T, GEO_FEAT_DIM)
            "boxes_seq": torch.from_numpy(boxes_seq),                 # (T, 4)
            "action_id": action_id,
            "track_id": tid,
            "local_track_id": local_id,
            "video_id": vid,
            "frame_indices": [fi for fi, _, _ in window],
        }

    @property
    def num_local_tracks(self) -> Dict[str, int]:
        """ビデオごとの track 数 (ID 分類の class 数)"""
        return {vid: len(mapping) for vid, mapping in self.track_local_id_map.items()}


# ---------------------------------------------------------------------------
# Stage 4: SceneSequenceDataset
# ---------------------------------------------------------------------------

class SceneSequenceDataset(Dataset):
    """
    シーン全体のシーケンス Dataset (Stage 4 統合学習)。

    返り値 (1 サンプル):
        images:       (T, 3, H, W)
        boxes_per_frame:   List[(N_t, 4)] — per frame, variable N
        class_ids_per_frame: List[(N_t,)]
        track_ids_per_frame: List[(N_t,)]
        action_ids_per_frame: List[(N_t,)]
        geo_all:      (max_tracks, T, GEO_FEAT_DIM) — padded
        track_ids_padded: (max_tracks,) — -1 is padding
        track_mask:   (max_tracks,) bool
        video_id:     str
        window_start: int
    """

    def __init__(
        self,
        anno: DatasetAnno,
        window_size: int = 16,
        stride: int = 8,
        image_size: Tuple[int, int] = (640, 640),
        data_root: str = "",
        augment: bool = True,
        max_tracks: int = 20,
    ):
        self.anno = anno
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.data_root = Path(data_root)
        self.augment = augment
        self.max_tracks = max_tracks

        self.geo_builder = GeometricFeatureBuilder(
            FeatureBuilderConfig(img_width=image_size[1], img_height=image_size[0])
        )

        # ウィンドウインデックス: (video, start_frame_list_idx)
        self.windows: List[Tuple[Any, int]] = []  # (VideoAnno, start)
        for video in anno.videos:
            nf = len(video.frames)
            if nf < window_size:
                continue
            for start in range(0, nf - window_size + 1, stride):
                self.windows.append((video, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video, start = self.windows[idx]
        frames = video.frames[start:start + self.window_size]
        T = len(frames)
        W = video.width or self.image_size[1]
        H = video.height or self.image_size[0]
        H_tgt, W_tgt = self.image_size

        do_hflip = self.augment and random.random() < 0.5

        # 画像読み込み
        images = []
        for frame in frames:
            img_path = str(self.data_root / frame.image_path) \
                if self.data_root.name else frame.image_path
            img = _load_image_rgb(img_path, (H_tgt, W_tgt))
            if do_hflip:
                img = img[:, ::-1].copy()
            if self.augment:
                img = _augment_image(img)
            img = _resize_image(img, self.image_size)
            images.append(_to_tensor(img))

        # Per-frame GT
        boxes_per_frame = []
        class_ids_per_frame = []
        track_ids_per_frame = []
        action_ids_per_frame = []
        sx = W_tgt / W
        sy = H_tgt / H

        for frame in frames:
            boxes_f = []
            cids, tids, aids = [], [], []
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox
                if do_hflip:
                    x1, x2 = W - x2, W - x1
                x1n = max(0., min(1., x1 * sx / W_tgt))
                y1n = max(0., min(1., y1 * sy / H_tgt))
                x2n = max(0., min(1., x2 * sx / W_tgt))
                y2n = max(0., min(1., y2 * sy / H_tgt))
                if x2n <= x1n or y2n <= y1n:
                    continue
                boxes_f.append([x1n, y1n, x2n, y2n])
                cids.append(obj.class_id)
                tids.append(obj.track_id)
                aids.append(obj.action_id)
            boxes_per_frame.append(torch.tensor(boxes_f, dtype=torch.float32))
            class_ids_per_frame.append(torch.tensor(cids, dtype=torch.long))
            track_ids_per_frame.append(torch.tensor(tids, dtype=torch.long))
            action_ids_per_frame.append(torch.tensor(aids, dtype=torch.long))

        # Geometric features (per track, padded)
        geo_arr, tid_arr, mask_arr = self.geo_builder.build_from_frames_padded(
            frames, W, H, self.max_tracks
        )

        return {
            "images": torch.stack(images, dim=0),                     # (T, 3, H, W)
            "boxes_per_frame": boxes_per_frame,                       # List[(N_t, 4)]
            "class_ids_per_frame": class_ids_per_frame,               # List[(N_t,)]
            "track_ids_per_frame": track_ids_per_frame,               # List[(N_t,)]
            "action_ids_per_frame": action_ids_per_frame,             # List[(N_t,)]
            "geo_all": torch.from_numpy(geo_arr),                     # (max_tracks, T, GEO_FEAT_DIM)
            "track_ids_padded": torch.from_numpy(tid_arr),            # (max_tracks,)
            "track_mask": torch.from_numpy(mask_arr),                 # (max_tracks,)
            "video_id": video.video_id,
            "window_start": start,
        }


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_detection(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FlyDetectionDataset 用 collate_fn。
    可変長 object 数は list のまま targets に格納する。
    """
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 3, H, W)

    targets = []
    for b in batch:
        targets.append({
            "boxes":      b["boxes"],       # (N, 4)
            "class_ids":  b["class_ids"],   # (N,)
            "track_ids":  b["track_ids"],   # (N,)
            "action_ids": b["action_ids"],  # (N,)
            "geo_features": b["geo_features"],  # (N, GEO_FEAT_DIM)
        })

    meta = [b["meta"] for b in batch]
    return {"images": images, "targets": targets, "meta": meta}


def collate_track_windows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TrackWindowDataset 用 collate_fn。
    各サンプルは 1 track × T frames なので images は stack できる。
    """
    images = torch.stack([b["images"] for b in batch], dim=0)    # (B, T, 3, H, W)
    geo    = torch.stack([b["geo_features"] for b in batch], dim=0)  # (B, T, GEO_FEAT_DIM)
    boxes  = torch.stack([b["boxes_seq"] for b in batch], dim=0)  # (B, T, 4)

    action_ids = torch.tensor([b["action_id"] for b in batch], dtype=torch.long)  # (B,)
    track_ids  = torch.tensor([b["track_id"]  for b in batch], dtype=torch.long)  # (B,)
    local_ids  = torch.tensor([b["local_track_id"] for b in batch], dtype=torch.long)

    meta = [{
        "video_id": b["video_id"],
        "frame_indices": b["frame_indices"],
    } for b in batch]

    return {
        "images": images,
        "geo_features": geo,
        "boxes_seq": boxes,
        "action_ids": action_ids,
        "track_ids": track_ids,
        "local_track_ids": local_ids,
        "meta": meta,
    }


def collate_scene_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SceneSequenceDataset 用 collate_fn。
    per-frame の GT は list のまま保持する。
    """
    images = torch.stack([b["images"] for b in batch], dim=0)    # (B, T, 3, H, W)
    geo_all = torch.stack([b["geo_all"] for b in batch], dim=0)  # (B, max_tracks, T, GEO_FEAT_DIM)
    track_ids_padded = torch.stack([b["track_ids_padded"] for b in batch], dim=0)  # (B, max_tracks)
    track_mask = torch.stack([b["track_mask"] for b in batch], dim=0)              # (B, max_tracks)

    # per-frame GT は可変長なので List[List[...]] のまま
    B = len(batch)
    T = batch[0]["images"].shape[0]
    all_targets = []  # B × T
    for b in batch:
        frame_targets = []
        for t in range(T):
            frame_targets.append({
                "boxes":      b["boxes_per_frame"][t],
                "class_ids":  b["class_ids_per_frame"][t],
                "track_ids":  b["track_ids_per_frame"][t],
                "action_ids": b["action_ids_per_frame"][t],
            })
        all_targets.append(frame_targets)

    meta = [{"video_id": b["video_id"], "window_start": b["window_start"]} for b in batch]

    return {
        "images": images,
        "geo_all": geo_all,
        "track_ids_padded": track_ids_padded,
        "track_mask": track_mask,
        "targets": all_targets,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# DataModule: Dataset + DataLoader の組み合わせを管理
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_anno: DatasetAnno,
    val_anno: Optional[DatasetAnno],
    stage: int,
    cfg=None,  # DataConfig (optional) — overrides individual params below
    image_size: Tuple[int, int] = (640, 640),
    window_size: int = 16,
    stride: int = 8,
    batch_size: int = 4,
    num_workers: int = 4,
    data_root: str = "",
    max_tracks: int = 20,
) -> Dict[str, Any]:
    """
    stage に応じた train / val DataLoader を構築して dict で返す。

    Args:
        cfg: DataConfig object (overrides image_size / window_size / batch_size / etc.)

    Returns:
        {"train": DataLoader, "val": DataLoader or None}
    """
    from torch.utils.data import DataLoader

    # DataConfig が渡された場合は各パラメータを上書き
    if cfg is not None:
        image_size = tuple(cfg.image_size) if hasattr(cfg, "image_size") else image_size
        window_size = getattr(cfg, "window_size", window_size)
        stride = getattr(cfg, "window_stride", stride)
        batch_size = getattr(cfg, "batch_size", batch_size)
        num_workers = getattr(cfg, "num_workers", num_workers)

    if stage == 1:
        train_ds = FlyDetectionDataset(
            train_anno, image_size=image_size, data_root=data_root, augment=True
        )
        val_ds = FlyDetectionDataset(
            val_anno, image_size=image_size, data_root=data_root, augment=False
        ) if val_anno is not None else None
        collate = collate_detection

    elif stage in (2, 3):
        train_ds = TrackWindowDataset(
            train_anno, window_size=window_size, stride=stride,
            image_size=image_size, data_root=data_root, augment=True,
            require_action_label=(stage == 2),
        )
        val_ds = TrackWindowDataset(
            val_anno, window_size=window_size, stride=window_size,
            image_size=image_size, data_root=data_root, augment=False,
            require_action_label=(stage == 2),
        ) if val_anno is not None else None
        collate = collate_track_windows

    else:  # stage 4
        train_ds = SceneSequenceDataset(
            train_anno, window_size=window_size, stride=stride,
            image_size=image_size, data_root=data_root, augment=True,
            max_tracks=max_tracks,
        )
        val_ds = SceneSequenceDataset(
            val_anno, window_size=window_size, stride=window_size,
            image_size=image_size, data_root=data_root, augment=False,
            max_tracks=max_tracks,
        ) if val_anno is not None else None
        collate = collate_scene_sequences

    nw = min(num_workers, 0) if num_workers == 0 else num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_ds) >= batch_size,
        persistent_workers=(nw > 0),
    )

    result: Dict[str, Any] = {"train": train_loader}

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=False,
            persistent_workers=(nw > 0),
        )
        result["val"] = val_loader

    return result
