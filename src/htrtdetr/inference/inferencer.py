"""
inferencer.py — Video Inference

機能:
- 動画ファイル / 画像ディレクトリ入力に対応
- sliding window でシーケンスを処理
- 各フレームに bbox / ID / action をオーバーレイ
- 結果を JSON で保存

入出力:
  Input:  video file (.mp4, .avi等) or images dir
  Output: annotated video + JSON predictions
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..config.config import InferenceConfig, ModelConfig
from ..models.ht_rtdetr import HTRTDETR
from ..models.id_head import IdentityMemory
from ..utils.misc import (
    load_model_weights, get_device,
    cxcywh_to_xyxy, Timer, GPUMemoryTracker
)
from ..utils.logging import get_logger


# ---------------------------------------------------------------------------
# 動画読み込み / 書き出し helper
# ---------------------------------------------------------------------------

def read_video_frames(
    path: str,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    """
    動画を 1 フレームずつ読み込む generator。
    Yields: (frame_index, frame_rgb: np.ndarray(H,W,3), fps: float)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV が必要です: pip install opencv-python")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_idx, frame_rgb, fps
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()


def read_image_dir_frames(
    dir_path: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    fps: float = 30.0,
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    """画像ディレクトリを動画のように読み込む"""
    images = sorted([
        p for p in Path(dir_path).iterdir()
        if p.suffix.lower() in extensions
    ])
    for idx, img_path in enumerate(images):
        try:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            import cv2
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        yield idx, img, fps


class VideoWriter:
    """アノテーション済み動画を書き出す helper"""

    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        """size: (W, H)"""
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV が必要です: pip install opencv-python")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.enabled = self.writer.isOpened()

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.enabled:
            bgr = self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)

    def release(self) -> None:
        if self.enabled:
            self.writer.release()


# ---------------------------------------------------------------------------
# 描画ユーティリティ
# ---------------------------------------------------------------------------

# 色テーブル (ID に対応した色)
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


def _get_color(track_id: int) -> Tuple[int, int, int]:
    return _COLORS[track_id % len(_COLORS)]


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,           # (N, 4) pixel coords
    scores: np.ndarray,               # (N,)
    class_ids: np.ndarray,            # (N,)
    track_ids: np.ndarray,            # (N,)
    action_ids: np.ndarray,           # (N,)
    action_names: List[str],
    class_names: List[str],
    show_bbox: bool = True,
    show_id: bool = True,
    show_action: bool = True,
    show_confidence: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    フレームに検出結果をオーバーレイして返す。
    OpenCV を使う。
    """
    try:
        import cv2
    except ImportError:
        return frame  # OpenCV なしの場合はそのまま返す

    frame_vis = frame.copy()
    H, W = frame.shape[:2]

    for i, (box, score, cls_id, track_id, action_id) in enumerate(
        zip(boxes_xyxy, scores, class_ids, track_ids, action_ids)
    ):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # クランプ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        color = _get_color(int(track_id))

        if show_bbox:
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)

        # ラベル文字列
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{int(track_id)}")
        if show_confidence:
            label_parts.append(f"{score:.2f}")
        if show_action and 0 <= action_id < len(action_names):
            label_parts.append(action_names[int(action_id)])

        if label_parts:
            label = " ".join(label_parts)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(
                frame_vis,
                (x1, label_y - th - baseline),
                (x1 + tw, label_y + baseline),
                color, -1
            )
            cv2.putText(
                frame_vis, label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    return frame_vis


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

class Inferencer:
    """
    YOAKE を使ってビデオ推論を行うクラス。

    sliding window で過去フレームをバッファに保持し、
    各フレームに対してシーケンス入力を作成して推論する。
    """

    def __init__(
        self,
        model: HTRTDETR,
        cfg: InferenceConfig,
        action_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        window_size: int = 16,
    ):
        self.model = model
        self.cfg = cfg
        self.action_names = action_names or ["unknown"]
        self.class_names = class_names or ["animal"]
        self.window_size = window_size

        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.logger = get_logger("inferencer")

        # ID memory (動画全体で保持)
        self.memory: Optional[IdentityMemory] = None

    def reset_memory(self) -> None:
        """メモリをリセットする (シーン切り替え等)"""
        self.memory = self.model.id_head.create_memory(self.device)

    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
    ) -> torch.Tensor:
        """
        np.ndarray (H, W, 3) RGB → normalized tensor (1, 3, H, W)
        """
        H, W = target_size
        try:
            from PIL import Image
            pil = Image.fromarray(frame).resize((W, H), Image.BILINEAR)
            img = np.array(pil, dtype=np.float32) / 255.0
        except ImportError:
            import cv2
            img = cv2.resize(frame, (W, H)).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    @torch.no_grad()
    def infer_frame(
        self,
        frame_buffer: List[torch.Tensor],  # List of (1, 3, H, W)
        orig_size: Tuple[int, int],         # (H_orig, W_orig)
    ) -> Dict:
        """
        フレームバッファを使って 1 フレームを推論する。

        Returns:
            結果 dict:
              boxes_xyxy: (N, 4) in original pixel coords
              scores: (N,)
              class_ids: (N,)
              track_ids: (N,)
              id_scores: (N,)
              action_ids: (N,)
              action_probs: (N, num_actions)
        """
        T = len(frame_buffer)
        # (1, T, 3, H, W)
        images = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(self.device)

        if self.memory is None:
            self.reset_memory()

        with autocast(enabled=torch.cuda.is_available()):
            output = self.model(images, memory_list=[self.memory])

        # 結果を整形
        H_orig, W_orig = orig_size
        H_proc, W_proc = images.shape[-2], images.shape[-1]

        det_results = output.det_results[0] if output.det_results else {}

        if not det_results or det_results.get("boxes") is None:
            return {
                "boxes_xyxy": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "class_ids": np.zeros(0, dtype=int),
                "track_ids": np.zeros(0, dtype=int),
                "id_scores": np.zeros(0),
                "action_ids": np.zeros(0, dtype=int),
                "action_probs": np.zeros((0, len(self.action_names))),
            }

        boxes = det_results["boxes"].cpu()      # (N, 4) [cx, cy, w, h] normalized
        scores = det_results["scores"].cpu()
        class_ids = det_results["class_ids"].cpu()
        track_ids = det_results.get("track_ids", torch.zeros(len(boxes), dtype=torch.long)).cpu()
        id_scores = det_results.get("id_scores", torch.ones(len(boxes))).cpu()

        # cxcywh → xyxy → pixel coords
        boxes_xyxy = cxcywh_to_xyxy(boxes)
        scale = boxes.new_tensor([W_orig, H_orig, W_orig, H_orig])
        boxes_pixel = (boxes_xyxy * scale).numpy()

        # Action IDs
        if output.action_probs is not None:
            action_probs_all = output.action_probs.cpu().numpy()  # (N_total, A)
            # det_results のインデックスに対応する部分を取る
            action_probs = action_probs_all[:len(boxes)]
            action_ids = action_probs.argmax(axis=-1)
        else:
            action_probs = np.zeros((len(boxes), len(self.action_names)))
            action_ids = np.zeros(len(boxes), dtype=int)

        return {
            "boxes_xyxy": boxes_pixel,
            "scores": scores.numpy(),
            "class_ids": class_ids.numpy(),
            "track_ids": track_ids.numpy(),
            "id_scores": id_scores.numpy(),
            "action_ids": action_ids,
            "action_probs": action_probs,
        }

    def run(
        self,
        input_path: str,
        output_dir: str,
        image_size: Tuple[int, int] = (640, 640),
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        動画全体を推論して結果を保存する。

        Returns:
            results: {frame_idx: {boxes, scores, ...}}
        """
        input_path = Path(input_path)
        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # 入力ソースの選択
        if input_path.is_dir():
            frame_gen = read_image_dir_frames(str(input_path))
        elif input_path.exists():
            frame_gen = read_video_frames(str(input_path), max_frames=max_frames)
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

        self.reset_memory()
        frame_buffer: List[torch.Tensor] = []
        all_results: Dict[int, Dict] = {}
        video_writer: Optional[VideoWriter] = None
        fps = 30.0

        from ..evaluation.evaluator import RuntimeEvaluator
        runtime_eval = RuntimeEvaluator()

        for frame_idx, frame_rgb, fps in frame_gen:
            orig_h, orig_w = frame_rgb.shape[:2]
            tensor = self.preprocess_frame(frame_rgb, image_size)
            frame_buffer.append(tensor)

            # window size に達したら推論
            if len(frame_buffer) > self.window_size:
                frame_buffer.pop(0)

            if len(frame_buffer) < 1:
                continue

            # タイマー開始
            start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            result = self.infer_frame(frame_buffer, (orig_h, orig_w))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            gpu_mb = GPUMemoryTracker.current_mb()
            runtime_eval.update(elapsed, gpu_mb)

            all_results[frame_idx] = result

            # Video writer の初期化
            if video_writer is None:
                out_video_path = str(output_dir_p / "output.mp4")
                video_writer = VideoWriter(
                    out_video_path, fps, (orig_w, orig_h)
                )

            # オーバーレイ描画
            annotated = draw_detections(
                frame_rgb,
                result["boxes_xyxy"],
                result["scores"],
                result["class_ids"],
                result["track_ids"],
                result["action_ids"],
                self.action_names,
                self.class_names,
                show_bbox=self.cfg.show_bbox,
                show_id=self.cfg.show_id,
                show_action=self.cfg.show_action,
                show_confidence=self.cfg.show_confidence,
                thickness=self.cfg.bbox_thickness,
                font_scale=self.cfg.font_scale,
            )
            if video_writer:
                video_writer.write(annotated)

            if frame_idx % 30 == 0:
                self.logger.info(
                    f"Frame {frame_idx} | FPS: {1/max(elapsed, 1e-6):.1f} | "
                    f"Objects: {len(result['boxes_xyxy'])}"
                )

        if video_writer:
            video_writer.release()

        # 結果を JSON 保存
        json_results = {
            str(k): {
                "boxes": v["boxes_xyxy"].tolist(),
                "scores": v["scores"].tolist(),
                "track_ids": v["track_ids"].tolist(),
                "action_ids": v["action_ids"].tolist(),
            }
            for k, v in all_results.items()
        }
        json_path = output_dir_p / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        # Runtime 統計
        runtime_stats = runtime_eval.compute()
        runtime_path = output_dir_p / "runtime_stats.json"
        with open(runtime_path, "w") as f:
            json.dump(runtime_stats, f, indent=2)

        self.logger.info(
            f"Inference complete. "
            f"Mean FPS: {runtime_stats.get('mean_fps', 0):.1f} | "
            f"Results: {json_path}"
        )

        return all_results
