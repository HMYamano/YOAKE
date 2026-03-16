"""
infer_video.py — Video Inference with YOAKE

入力動画に対して YOAKE を実行し、
bbox / track_id / action_id をオーバーレイした mp4 と
JSON 予測ファイルを出力する。

使い方:
  python tools/infer_video.py \
      input=data/videos/sample.mp4 \
      checkpoint=outputs/stage4/checkpoint_best.pth \
      output_dir=outputs/inference \
      score_threshold=0.3 \
      window_size=16

オプション:
  input           : 入力動画 (.mp4) または画像ディレクトリ
  checkpoint      : モデル checkpoint パス
  output_dir      : 出力ディレクトリ
  score_threshold : 検出スコア閾値 (default: 0.3)
  window_size     : 推論ウィンドウサイズ (default: 16)
  img_size        : リサイズ後の画像サイズ (default: 640)
  action_names    : カンマ区切りのアクション名 (default: 自動)
  no_video        : 動画出力を省略 (flag, e.g. no_video=true)
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from htrtdetr.config.config import get_stage4_config
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video I/O disabled.")


# ---------------------------------------------------------------------------
# Color palette for track IDs
# ---------------------------------------------------------------------------

_PALETTE = [
    (255, 80, 80),  (80, 255, 80),  (80, 80, 255),  (255, 255, 80),
    (255, 80, 255), (80, 255, 255), (255, 160, 80),  (80, 160, 255),
    (160, 255, 80), (160, 80, 255), (255, 80, 160),  (80, 255, 160),
    (200, 200, 80), (200, 80, 200), (80, 200, 200),  (255, 120, 50),
]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,            # (N, 4) xyxy pixels
    scores: np.ndarray,           # (N,)
    track_ids: Optional[np.ndarray] = None,
    action_ids: Optional[np.ndarray] = None,
    action_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """frame に bbox / track_id / action_id をオーバーレイして返す (BGR)"""
    out = frame.copy()
    N = len(boxes)

    for i in range(N):
        if scores[i] < score_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        tid = int(track_ids[i]) if track_ids is not None else -1
        color = _track_color(tid) if tid >= 0 else (200, 200, 200)
        bgr = (color[2], color[1], color[0])

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        parts = []
        if tid >= 0:
            parts.append(f"ID:{tid}")
        parts.append(f"{scores[i]:.2f}")
        if action_ids is not None:
            aid = int(action_ids[i])
            if action_names and 0 <= aid < len(action_names):
                parts.append(action_names[aid])
            else:
                parts.append(f"A:{aid}")

        label = " ".join(parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw, y1), bgr, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 2), font, scale,
                    (255, 255, 255), thickness)
    return out


# ---------------------------------------------------------------------------
# Video Reader / Writer
# ---------------------------------------------------------------------------

class VideoReader:
    def __init__(self, path: str, img_size: int = 640):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.img_size = img_size
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
        """Returns (bgr_frame, tensor (1,3,H,W)) or None at EOF"""
        ret, bgr = self.cap.read()
        if not ret:
            return None
        bgr_r = cv2.resize(bgr, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return bgr_r, tensor

    def release(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    video_path: str,
    output_dir: Path,
    device: torch.device,
    score_threshold: float = 0.3,
    window_size: int = 16,
    img_size: int = 640,
    action_names: Optional[List[str]] = None,
    write_video: bool = True,
) -> Dict:
    """
    動画全体に推論を実行し、mp4 + JSON を保存する。
    Returns summary dict.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for video inference.")

    reader = VideoReader(video_path, img_size=img_size)
    fps = reader.fps
    video_name = Path(video_path).stem
    out_video_path = str(output_dir / f"{video_name}_pred.mp4")
    out_json_path = str(output_dir / f"{video_name}_pred.json")

    writer = VideoWriter(out_video_path, fps, img_size, img_size) if write_video else None

    frame_buffer: deque = deque(maxlen=window_size)
    bgr_buffer: deque = deque(maxlen=window_size)

    memory_list = model.create_memory_list(1, device)

    all_frame_preds = []
    frame_idx = 0
    total_dets = 0
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        while True:
            result = reader.read_frame()
            if result is None:
                break

            bgr_frame, frame_tensor = result
            frame_buffer.append(frame_tensor)
            bgr_buffer.append(bgr_frame)

            # Wait until buffer is full
            if len(frame_buffer) < window_size:
                if writer:
                    writer.write(bgr_frame)
                frame_idx += 1
                continue

            # (1, T, 3, H, W)
            images = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)
            out = model(images, memory_list)

            det = out.det_results[0] if out.det_results else {}
            pred_boxes_cxcywh = det.get("boxes", torch.zeros(0, 4))
            pred_scores = det.get("scores", torch.zeros(0))
            pred_track_ids = det.get("track_ids", None)

            pred_action_ids = None
            if out.action_logits is not None and out.action_logits.shape[0] > 0:
                pred_action_ids = out.action_logits.argmax(dim=-1).cpu().numpy()

            pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes_cxcywh)
            pred_boxes_pixel = pred_boxes_xyxy.cpu().numpy() * img_size

            n_det = len(pred_boxes_pixel)
            total_dets += n_det

            frame_pred = {"frame_idx": frame_idx, "detections": []}
            scores_np = pred_scores.cpu().numpy() if len(pred_scores) > 0 else np.zeros(n_det)
            tids_np = pred_track_ids.cpu().numpy() if pred_track_ids is not None else None

            for i in range(n_det):
                frame_pred["detections"].append({
                    "bbox": pred_boxes_pixel[i].tolist(),
                    "score": float(scores_np[i]),
                    "track_id": int(tids_np[i]) if tids_np is not None else -1,
                    "action_id": int(pred_action_ids[i]) if pred_action_ids is not None else -1,
                })
            all_frame_preds.append(frame_pred)

            if writer:
                annotated = draw_detections(
                    bgr_buffer[-1].copy(),
                    pred_boxes_pixel,
                    scores_np,
                    tids_np,
                    pred_action_ids,
                    action_names=action_names,
                    score_threshold=score_threshold,
                )
                cv2.putText(annotated, f"F{frame_idx}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Frame {frame_idx}/{reader.total_frames}"
                      f"  ({frame_idx / max(elapsed, 1e-3):.1f} FPS)")

    reader.release()
    if writer:
        writer.release()

    elapsed = time.time() - t0
    summary = {
        "video": video_path,
        "total_frames": frame_idx,
        "total_detections": total_dets,
        "inference_fps": frame_idx / max(elapsed, 1e-6),
        "output_video": out_video_path if write_video else None,
        "output_json": out_json_path,
        "predictions": all_frame_preds,
    }

    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_overrides(argv) -> dict:
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v
    return overrides


def main():
    overrides = parse_overrides(sys.argv[1:])

    input_path = overrides.get("input", None)
    checkpoint_path = overrides.get("checkpoint", "outputs/stage4/checkpoint_best.pth")
    output_dir = Path(overrides.get("output_dir", "outputs/inference"))
    score_threshold = float(overrides.get("score_threshold", "0.3"))
    window_size = int(overrides.get("window_size", "16"))
    img_size = int(overrides.get("img_size", "640"))
    action_names_str = overrides.get("action_names", "")
    no_video = overrides.get("no_video", "false").lower() == "true"

    if input_path is None:
        print("Usage: python tools/infer_video.py input=<video.mp4> checkpoint=<path>")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = [s.strip() for s in action_names_str.split(",") if s.strip()] or None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input: {input_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = get_stage4_config()
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(4)

    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: checkpoint not found, using random weights.")

    summary = run_inference(
        model=model,
        video_path=input_path,
        output_dir=output_dir,
        device=device,
        score_threshold=score_threshold,
        window_size=window_size,
        img_size=img_size,
        action_names=action_names,
        write_video=not no_video,
    )

    print(f"\nDone.")
    print(f"  Frames: {summary['total_frames']}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Inference FPS: {summary['inference_fps']:.1f}")
    if summary["output_video"]:
        print(f"  Video: {summary['output_video']}")
    print(f"  JSON:  {summary['output_json']}")


if __name__ == "__main__":
    main()
