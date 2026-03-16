"""
dataset.py — YOAKE データセットクラス

設計方針:
- 内部表現は video_id / frame_index / image_path / objects[] に統一
- objects は bbox / class_id / track_id / action_id を持つ
- SlidingWindowDataset でシーケンスウィンドウを切り出す
- 将来的にCOCO形式、MOT形式、AVA形式にも対応しやすい基底クラス

データ形式 (JSON annotation):
{
  "videos": [
    {
      "video_id": "vid_001",
      "fps": 30,
      "frames": [
        {
          "frame_index": 0,
          "image_path": "images/vid_001/000000.jpg",
          "objects": [
            {
              "bbox": [x1, y1, x2, y2],   # pixel 座標
              "class_id": 0,
              "track_id": 1,
              "action_id": 2
            }
          ]
        }
      ]
    }
  ],
  "class_names": ["fly"],
  "action_names": ["idle", "walk", "groom", "interact", "other"]
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 内部データ表現
# ---------------------------------------------------------------------------

@dataclass
class ObjectAnnotation:
    """1 フレーム内の 1 個体アノテーション"""
    bbox: List[float]          # [x1, y1, x2, y2] in pixels
    class_id: int = 0
    track_id: int = -1         # -1: unknown
    action_id: int = -1        # -1: unannotated


@dataclass
class FrameAnnotation:
    """1 フレームのアノテーション"""
    frame_index: int
    image_path: str
    objects: List[ObjectAnnotation] = field(default_factory=list)
    width: int = 0
    height: int = 0


@dataclass
class VideoAnnotation:
    """1 動画のアノテーション"""
    video_id: str
    fps: float = 30.0
    frames: List[FrameAnnotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON アノテーション読み込み
# ---------------------------------------------------------------------------

def load_annotations(json_path: str) -> Tuple[List[VideoAnnotation], List[str], List[str]]:
    """
    JSON ファイルからアノテーションを読み込む

    Returns:
        videos: VideoAnnotation のリスト
        class_names: クラス名リスト
        action_names: 行動名リスト
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_names = data.get("class_names", ["animal"])
    action_names = data.get("action_names", ["unknown"])

    videos = []
    for v in data.get("videos", []):
        frames = []
        for fr in v.get("frames", []):
            objects = []
            for obj in fr.get("objects", []):
                objects.append(ObjectAnnotation(
                    bbox=obj.get("bbox", [0, 0, 1, 1]),
                    class_id=obj.get("class_id", 0),
                    track_id=obj.get("track_id", -1),
                    action_id=obj.get("action_id", -1),
                ))
            frames.append(FrameAnnotation(
                frame_index=fr["frame_index"],
                image_path=fr["image_path"],
                objects=objects,
                width=fr.get("width", 0),
                height=fr.get("height", 0),
            ))
        videos.append(VideoAnnotation(
            video_id=v["video_id"],
            fps=v.get("fps", 30.0),
            frames=frames,
        ))
    return videos, class_names, action_names


# ---------------------------------------------------------------------------
# 画像読み込み
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    PIL または OpenCV で画像を読み込む。
    Returns: (H, W, 3) RGB uint8 array
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        pass
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        raise ImportError("PIL (Pillow) または OpenCV が必要です")


# ---------------------------------------------------------------------------
# データ拡張
# ---------------------------------------------------------------------------

class FrameAugmentation:
    """フレーム単体のデータ拡張。空間変換を行い、bbox を同期して変換する。"""

    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.scale_range = scale_range

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,  # (N, 4) [x1, y1, x2, y2]
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        # Horizontal flip
        if self.hflip and random.random() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # Vertical flip
        if self.vflip and random.random() < 0.5:
            image = image[::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        # Color jitter (PIL ベース)
        try:
            from PIL import Image, ImageEnhance
            pil = Image.fromarray(image)
            if self.brightness > 0 and random.random() < 0.5:
                factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                pil = ImageEnhance.Brightness(pil).enhance(factor)
            if self.contrast > 0 and random.random() < 0.5:
                factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                pil = ImageEnhance.Contrast(pil).enhance(factor)
            if self.saturation > 0 and random.random() < 0.5:
                factor = 1.0 + random.uniform(-self.saturation, self.saturation)
                pil = ImageEnhance.Color(pil).enhance(factor)
            image = np.array(pil)
        except ImportError:
            pass  # PIL なしの場合は色拡張をスキップ

        return image, boxes


# ---------------------------------------------------------------------------
# 単フレーム Dataset (Stage 1: detection 学習用)
# ---------------------------------------------------------------------------

class SingleFrameDataset(Dataset):
    """
    1 フレームを 1 サンプルとして返す Dataset。
    Stage 1 (detector pretraining) で使用する。
    """

    def __init__(
        self,
        videos: List[VideoAnnotation],
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        augment_cfg: Optional[Dict] = None,
        data_root: str = "",
    ):
        self.image_size = image_size  # (H, W)
        self.data_root = Path(data_root)
        self.augment = augment

        aug_cfg = augment_cfg or {}
        self.augmentation = FrameAugmentation(**aug_cfg) if augment else None

        # 全フレームをフラットなリストに展開
        self.samples: List[FrameAnnotation] = []
        for video in videos:
            self.samples.extend(video.frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.samples[idx]
        image_path = str(self.data_root / frame.image_path) if self.data_root.name else frame.image_path

        # 画像読み込み
        try:
            image = load_image(image_path)
        except Exception:
            # ダミー画像 (テスト用フォールバック)
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        h_orig, w_orig = image.shape[:2]

        # bbox を numpy に変換
        boxes = np.array([obj.bbox for obj in frame.objects], dtype=np.float32) \
            if frame.objects else np.zeros((0, 4), dtype=np.float32)
        class_ids = np.array([obj.class_id for obj in frame.objects], dtype=np.int64) \
            if frame.objects else np.zeros(0, dtype=np.int64)
        track_ids = np.array([obj.track_id for obj in frame.objects], dtype=np.int64) \
            if frame.objects else np.zeros(0, dtype=np.int64)
        action_ids = np.array([obj.action_id for obj in frame.objects], dtype=np.int64) \
            if frame.objects else np.zeros(0, dtype=np.int64)

        # データ拡張
        if self.augmentation is not None:
            image, boxes = self.augmentation(image, boxes)

        # リサイズ
        image, boxes = self._resize(image, boxes, self.image_size)

        # Normalize image: [0, 255] → [0, 1], (H, W, C) → (C, H, W)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        # ImageNet 正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # bbox を [0, 1] に正規化
        H, W = self.image_size
        if len(boxes):
            boxes_norm = boxes / np.array([W, H, W, H], dtype=np.float32)
        else:
            boxes_norm = boxes

        return {
            "image": image_tensor,                    # (3, H, W)
            "boxes": torch.from_numpy(boxes_norm),    # (N, 4) [x1, y1, x2, y2] normalized
            "class_ids": torch.from_numpy(class_ids), # (N,)
            "track_ids": torch.from_numpy(track_ids), # (N,)
            "action_ids": torch.from_numpy(action_ids),# (N,)
            "frame_index": frame.frame_index,
            "image_path": frame.image_path,
            "orig_size": (h_orig, w_orig),
        }

    @staticmethod
    def _resize(
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """画像をリサイズし、bbox を対応してスケールする"""
        h, w = image.shape[:2]
        tH, tW = target_size
        try:
            from PIL import Image
            pil = Image.fromarray(image)
            pil = pil.resize((tW, tH), Image.BILINEAR)
            image = np.array(pil)
        except ImportError:
            try:
                import cv2
                image = cv2.resize(image, (tW, tH), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # 粗いリサイズ (fallback)
                from skimage.transform import resize as sk_resize
                image = (sk_resize(image, (tH, tW)) * 255).astype(np.uint8)

        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= tW / w
            boxes[:, [1, 3]] *= tH / h

        return image, boxes


# ---------------------------------------------------------------------------
# シーケンス Dataset (Stage 2, 3, 4: temporal / action / unified 学習用)
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Sliding window でシーケンスを切り出す Dataset。
    各サンプルは window_size フレームのシーケンスを返す。
    """

    def __init__(
        self,
        videos: List[VideoAnnotation],
        window_size: int = 16,
        stride: int = 8,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        augment_cfg: Optional[Dict] = None,
        data_root: str = "",
        require_action: bool = False,  # action annotation が必須かどうか
        require_track: bool = False,   # track annotation が必須かどうか
    ):
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.data_root = Path(data_root)
        self.augment = augment
        self.require_action = require_action
        self.require_track = require_track

        aug_cfg = augment_cfg or {}
        self.augmentation = FrameAugmentation(**aug_cfg) if augment else None

        # 動画ごとに sliding window でインデックスを生成
        self.windows: List[Tuple[VideoAnnotation, int]] = []  # (video, start_idx)
        for video in videos:
            n = len(video.frames)
            if n < window_size:
                continue
            for start in range(0, n - window_size + 1, stride):
                frames = video.frames[start:start + window_size]
                # annotation の有無でフィルタリング
                if require_action and not self._has_action(frames):
                    continue
                if require_track and not self._has_track(frames):
                    continue
                self.windows.append((video, start))

        # 拡張時に一貫した flip を適用するためのフラグ (シーケンス内で統一)
        self._single_frame_ds = SingleFrameDataset.__new__(SingleFrameDataset)
        self._single_frame_ds.image_size = image_size
        self._single_frame_ds.data_root = self.data_root
        self._single_frame_ds.augmentation = None  # 個別に処理

    @staticmethod
    def _has_action(frames: List[FrameAnnotation]) -> bool:
        return any(
            obj.action_id >= 0
            for fr in frames
            for obj in fr.objects
        )

    @staticmethod
    def _has_track(frames: List[FrameAnnotation]) -> bool:
        return any(
            obj.track_id >= 0
            for fr in frames
            for obj in fr.objects
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video, start = self.windows[idx]
        frames = video.frames[start:start + self.window_size]

        # シーケンス全体で一貫したデータ拡張を適用するため
        # flip フラグをここで決める
        do_hflip = self.augment and self.augmentation is not None and random.random() < 0.5
        do_vflip = self.augment and self.augmentation is not None and \
            self.augmentation.vflip and random.random() < 0.5

        images = []
        all_boxes = []
        all_class_ids = []
        all_track_ids = []
        all_action_ids = []
        frame_indices = []

        for frame in frames:
            image_path = str(self.data_root / frame.image_path) \
                if self.data_root.name else frame.image_path

            try:
                image = load_image(image_path)
            except Exception:
                image = np.zeros((*self.image_size, 3), dtype=np.uint8)

            boxes = np.array([obj.bbox for obj in frame.objects], dtype=np.float32) \
                if frame.objects else np.zeros((0, 4), dtype=np.float32)
            class_ids = np.array([obj.class_id for obj in frame.objects], dtype=np.int64) \
                if frame.objects else np.zeros(0, dtype=np.int64)
            track_ids = np.array([obj.track_id for obj in frame.objects], dtype=np.int64) \
                if frame.objects else np.zeros(0, dtype=np.int64)
            action_ids = np.array([obj.action_id for obj in frame.objects], dtype=np.int64) \
                if frame.objects else np.zeros(0, dtype=np.int64)

            # 空間拡張を適用 (flip のみシーケンス全体で統一)
            h, w = image.shape[:2]
            if do_hflip:
                image = image[:, ::-1].copy()
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if do_vflip:
                image = image[::-1].copy()
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

            # 色拡張 (シーケンス全体で一貫させる必要はないので個別に OK)
            if self.augment and self.augmentation:
                image, _ = self.augmentation(image, np.zeros((0, 4)))

            # リサイズ
            image, boxes = SingleFrameDataset._resize(image, boxes, self.image_size)

            # Normalize
            H, W = self.image_size
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

            if len(boxes):
                boxes_norm = boxes / np.array([W, H, W, H], dtype=np.float32)
            else:
                boxes_norm = boxes

            images.append(image_tensor)
            all_boxes.append(torch.from_numpy(boxes_norm))
            all_class_ids.append(torch.from_numpy(class_ids))
            all_track_ids.append(torch.from_numpy(track_ids))
            all_action_ids.append(torch.from_numpy(action_ids))
            frame_indices.append(frame.frame_index)

        return {
            "images": torch.stack(images, dim=0),     # (T, 3, H, W)
            "boxes": all_boxes,                        # List[(N_t, 4)]
            "class_ids": all_class_ids,                # List[(N_t,)]
            "track_ids": all_track_ids,                # List[(N_t,)]
            "action_ids": all_action_ids,              # List[(N_t,)]
            "frame_indices": frame_indices,            # List[int]
            "video_id": video.video_id,
            "window_start": start,
        }


# ---------------------------------------------------------------------------
# ダミーデータセット (テスト・デバッグ用)
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    """
    実データなしで動作確認するためのダミーデータセット。
    すべてランダムなテンソルを返す。
    """

    def __init__(
        self,
        n_samples: int = 100,
        window_size: int = 16,
        image_size: Tuple[int, int] = (640, 640),
        max_objects: int = 10,
        num_classes: int = 1,
        num_actions: int = 5,
        max_ids: int = 20,
        mode: str = "sequence",  # "single" | "sequence"
    ):
        self.n_samples = n_samples
        self.window_size = window_size
        self.image_size = image_size
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.max_ids = max_ids
        self.mode = mode

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        H, W = self.image_size
        n_obj = random.randint(1, self.max_objects)

        if self.mode == "single":
            image = torch.randn(3, H, W)
            boxes = torch.rand(n_obj, 4)
            # [x1, y1, x2, y2] を valid にする
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:].abs().clamp(min=0.01)
            boxes = boxes.clamp(0, 1)
            return {
                "image": image,
                "boxes": boxes,
                "class_ids": torch.randint(0, self.num_classes, (n_obj,)),
                "track_ids": torch.randint(0, self.max_ids, (n_obj,)),
                "action_ids": torch.randint(0, self.num_actions, (n_obj,)),
                "frame_index": idx,
                "image_path": f"dummy/{idx:06d}.jpg",
                "orig_size": (H, W),
            }
        else:
            T = self.window_size
            images = torch.randn(T, 3, H, W)
            boxes = []
            class_ids = []
            track_ids = []
            action_ids = []
            for _ in range(T):
                b = torch.rand(n_obj, 4)
                b[:, 2:] = b[:, :2] + b[:, 2:].abs().clamp(min=0.01)
                b = b.clamp(0, 1)
                boxes.append(b)
                class_ids.append(torch.randint(0, self.num_classes, (n_obj,)))
                track_ids.append(torch.randint(0, self.max_ids, (n_obj,)))
                action_ids.append(torch.randint(0, self.num_actions, (n_obj,)))

            return {
                "images": images,
                "boxes": boxes,
                "class_ids": class_ids,
                "track_ids": track_ids,
                "action_ids": action_ids,
                "frame_indices": list(range(T)),
                "video_id": f"dummy_{idx}",
                "window_start": 0,
            }


# ---------------------------------------------------------------------------
# Helper: annotation 生成ユーティリティ (実データ変換用)
# ---------------------------------------------------------------------------

def create_annotation_from_mot(
    mot_root: str,
    image_root: str,
    action_csv: Optional[str] = None,
    fps: float = 30.0,
) -> Dict:
    """
    MOT 形式 (MOT17/MOT20 等) のアノテーションを内部 JSON 形式に変換する。

    mot_root/
      <seq>/
        gt/gt.txt   # frame, id, x, y, w, h, conf, class, visibility

    これを内部形式の dict に変換して返す。
    """
    import glob
    import os

    class_names = ["animal"]
    action_names = ["unknown"]
    videos = []

    action_map: Dict[str, Dict[int, Dict[int, int]]] = {}  # vid -> frame -> track -> action
    if action_csv:
        import csv
        with open(action_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_id"]
                fi = int(row["frame_index"])
                tid = int(row["track_id"])
                aid = int(row["action_id"])
                if vid not in action_map:
                    action_map[vid] = {}
                if fi not in action_map[vid]:
                    action_map[vid][fi] = {}
                action_map[vid][fi][tid] = aid

    for seq_dir in sorted(glob.glob(os.path.join(mot_root, "*"))):
        if not os.path.isdir(seq_dir):
            continue
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue

        seq_name = os.path.basename(seq_dir)
        frame_data: Dict[int, List] = {}

        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                fi, tid, x, y, w, h = (
                    int(parts[0]), int(parts[1]),
                    float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5])
                )
                x1, y1, x2, y2 = x, y, x + w, y + h
                aid = action_map.get(seq_name, {}).get(fi, {}).get(tid, -1)
                if fi not in frame_data:
                    frame_data[fi] = []
                frame_data[fi].append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": 0,
                    "track_id": tid,
                    "action_id": aid,
                })

        frames = []
        for fi in sorted(frame_data.keys()):
            img_path = os.path.join(image_root, seq_name, f"{fi:06d}.jpg")
            frames.append({
                "frame_index": fi,
                "image_path": img_path,
                "objects": frame_data[fi],
            })

        videos.append({
            "video_id": seq_name,
            "fps": fps,
            "frames": frames,
        })

    return {
        "class_names": class_names,
        "action_names": action_names,
        "videos": videos,
    }
