# Inference Guide — YOAKE

This guide covers running inference on video files, interpreting the output format, loading predictions for downstream analysis, exporting to ONNX, and tuning inference performance.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Single Video Inference](#2-single-video-inference)
3. [Batch Inference Pattern](#3-batch-inference-pattern)
4. [Output Format](#4-output-format)
5. [Loading Predictions for Downstream Analysis](#5-loading-predictions-for-downstream-analysis)
6. [ONNX Export and Inference](#6-onnx-export-and-inference)
7. [Performance Optimization](#7-performance-optimization)
8. [Visualization Options](#8-visualization-options)

---

## 1. Prerequisites

- A trained Stage 4 checkpoint: `outputs/stage4/checkpoint_best.pth`
- Python environment with `opencv-python` installed (required for video I/O)
- Optional: `matplotlib` for analysis plots

Install if not already present:

```bash
pip install opencv-python matplotlib
```

Confirm the checkpoint exists and is loadable:

```python
import torch
ckpt = torch.load("outputs/stage4/checkpoint_best.pth", map_location="cpu")
print("Keys:", list(ckpt.keys()))
# Expected: ['model_state_dict', 'epoch', 'best_metric', ...]
```

---

## 2. Single Video Inference

`tools/infer_video.py` is the primary inference script. It reads a video file frame-by-frame using a sliding window of 16 frames, runs the full model forward pass, overlays bounding boxes and labels on each frame, and writes an annotated MP4 alongside a JSON predictions file.

### Basic usage

```bash
python tools/infer_video.py \
    input=data/videos/experiment_01.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference
```

This produces:
- `outputs/inference/experiment_01_pred.mp4` — annotated video
- `outputs/inference/experiment_01_pred.json` — structured predictions (see Output Format)

### Full argument reference

```bash
python tools/infer_video.py \
    input=data/videos/experiment_01.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference \
    score_threshold=0.3 \
    window_size=16 \
    img_size=640 \
    action_names=idle,walk,groom,court,other \
    no_video=false
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Path to input MP4 or image directory |
| `checkpoint` | `outputs/stage4/checkpoint_best.pth` | Path to trained model checkpoint |
| `output_dir` | `outputs/inference` | Directory for output files |
| `score_threshold` | `0.3` | Minimum detection confidence; detections below this are not visualized or saved |
| `window_size` | `16` | Number of frames per inference window (must match training config) |
| `img_size` | `640` | Resize input frames to this square resolution before inference |
| `action_names` | (auto) | Comma-separated action class names for visualization labels |
| `no_video` | `false` | Set to `true` to skip writing the annotated MP4 (faster; only JSON is written) |

### Inference from an image directory

Pass a directory containing sequentially numbered JPEG/PNG frames instead of an MP4:

```bash
python tools/infer_video.py \
    input=data/images/experiment_01 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference
```

Frames are read in lexicographic filename order.

---

## 3. Batch Inference Pattern

For processing multiple videos in sequence, use a shell loop or a simple Python wrapper:

### Shell loop (Linux/macOS/WSL2)

```bash
for VIDEO in data/videos/*.mp4; do
    STEM=$(basename "$VIDEO" .mp4)
    python tools/infer_video.py \
        input="$VIDEO" \
        checkpoint=outputs/stage4/checkpoint_best.pth \
        output_dir="outputs/inference/$STEM" \
        no_video=false
done
```

### Python wrapper for batch processing

```python
from pathlib import Path
import subprocess
import sys

checkpoint = "outputs/stage4/checkpoint_best.pth"
video_dir = Path("data/videos")
out_base = Path("outputs/inference")

for video_path in sorted(video_dir.glob("*.mp4")):
    out_dir = out_base / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "tools/infer_video.py",
        f"input={video_path}",
        f"checkpoint={checkpoint}",
        f"output_dir={out_dir}",
        "score_threshold=0.3",
    ]
    print(f"Processing: {video_path.name}")
    subprocess.run(cmd, check=True)
    print(f"  -> {out_dir}")
```

For large datasets, consider GPU batching within the inference script. The current implementation processes one video at a time with `batch_size=1` during inference.

---

## 4. Output Format

### Annotated video (MP4)

The output MP4 has the same frame rate and resolution as the input (after rescaling to `img_size`). Each detected animal is annotated with:

- A colored bounding box (color is fixed per `track_id` using a 16-color palette)
- A label string: `ID:<track_id> <detection_score> <action_name>` in the top-left corner of the bounding box
- A frame counter in the top-left corner of the frame

### JSON predictions file

```json
{
  "video": "data/videos/experiment_01.mp4",
  "total_frames": 1800,
  "total_detections": 9423,
  "inference_fps": 28.4,
  "output_video": "outputs/inference/experiment_01_pred.mp4",
  "output_json": "outputs/inference/experiment_01_pred.json",
  "predictions": [
    {
      "frame_idx": 16,
      "detections": [
        {
          "bbox": [112.3, 87.1, 132.7, 109.4],
          "score": 0.92,
          "track_id": 1,
          "action_id": 1
        },
        {
          "bbox": [450.1, 210.3, 468.9, 230.8],
          "score": 0.88,
          "track_id": 2,
          "action_id": 0
        }
      ]
    },
    {
      "frame_idx": 17,
      "detections": [...]
    }
  ]
}
```

**Top-level fields:**

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Input video path |
| `total_frames` | int | Total number of frames processed |
| `total_detections` | int | Total number of detections across all frames |
| `inference_fps` | float | Wall-clock throughput in frames per second |
| `output_video` | string or null | Path to annotated video (null if `no_video=true`) |
| `output_json` | string | Path to this JSON file |
| `predictions` | array | Per-frame prediction list (see below) |

**Per-frame prediction fields:**

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Zero-based frame index in the source video |
| `detections` | array | List of detection objects for this frame |

**Per-detection fields:**

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | float[4] | Bounding box `[x1, y1, x2, y2]` in pixels at `img_size` resolution |
| `score` | float | Detection confidence in [0, 1] |
| `track_id` | int | Persistent identity; `-1` if ID assignment failed |
| `action_id` | int | Predicted action class index; `-1` if action head was not run |

Note: The first `window_size - 1` frames (frames 0–14 for `window_size=16`) are passed through without inference because the temporal window is not yet full. These frames are written to the output video without annotations.

---

## 5. Loading Predictions for Downstream Analysis

### Basic loading

```python
import json

with open("outputs/inference/experiment_01_pred.json") as f:
    result = json.load(f)

# Build a dict: frame_idx -> list of detections
frames = {pred["frame_idx"]: pred["detections"] for pred in result["predictions"]}

# Example: get all detections at frame 100
dets = frames.get(100, [])
for det in dets:
    print(f"  ID={det['track_id']}  action={det['action_id']}  score={det['score']:.2f}  bbox={det['bbox']}")
```

### Building per-track action timelines

```python
import json
from collections import defaultdict

ACTION_NAMES = ["idle", "walk", "groom", "court", "other"]

with open("outputs/inference/experiment_01_pred.json") as f:
    result = json.load(f)

# {track_id: [(frame_idx, action_id), ...]}
track_timelines = defaultdict(list)

for pred in result["predictions"]:
    for det in pred["detections"]:
        if det["track_id"] >= 0 and det["action_id"] >= 0:
            track_timelines[det["track_id"]].append(
                (pred["frame_idx"], det["action_id"])
            )

# Print a summary for each track
for tid, timeline in sorted(track_timelines.items()):
    action_counts = defaultdict(int)
    for _, aid in timeline:
        action_counts[ACTION_NAMES[aid]] += 1
    print(f"Track {tid}: {len(timeline)} frames, actions={dict(action_counts)}")
```

### Visualizing action timelines

```bash
python tools/analyze.py mode=timeline \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

This saves `outputs/analysis/experiment_01/action_timeline.png`.

### Analyzing ID switches

```bash
python tools/analyze.py mode=id_switch \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

This saves `outputs/analysis/experiment_01/id_switch_analysis.json` with the total switch count and per-track switch events.

---

## 6. ONNX Export and Inference

### Exporting to ONNX

YOAKE uses standard PyTorch operations (MultiheadAttention, GRUCell, standard convolutions) that are supported by the ONNX opset. The export is performed with `torch.onnx.export`.

When `tools/export_onnx.py` is available, run:

```bash
python tools/export_onnx.py \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output=weights/ht_rtdetr.onnx \
    opset=17 \
    img_size=640 \
    window_size=16
```

If the export script is not yet present, use the Python API directly:

```python
import torch
from htrtdetr.config.config import get_stage4_config
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_checkpoint

cfg = get_stage4_config()
model = build_model(cfg.model)
load_checkpoint(model, "outputs/stage4/checkpoint_best.pth")
model.eval()
model.set_stage(4)

# Dummy input: batch=1, T=16, 3, 640, 640
dummy_images = torch.randn(1, 16, 3, 640, 640)
memory_list = model.create_memory_list(1, torch.device("cpu"))

torch.onnx.export(
    model,
    (dummy_images, memory_list),
    "weights/ht_rtdetr.onnx",
    opset_version=17,
    input_names=["images", "memory"],
    output_names=["pred_logits", "pred_boxes", "action_logits"],
    dynamic_axes={
        "images": {0: "batch"},
    },
)
print("ONNX export complete.")
```

**Note on dynamic shapes**: The number of detected objects per frame is dynamic. If your ONNX runtime does not support dynamic axes well, fix the output tensor to `num_queries=100` and apply score thresholding in post-processing.

### Running ONNX inference

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("weights/ht_rtdetr.onnx", providers=["CUDAExecutionProvider"])

images = np.random.randn(1, 16, 3, 640, 640).astype(np.float32)
# Provide memory as zero tensors for the first window
# Consult model.create_memory_list() for the exact structure

outputs = sess.run(None, {"images": images})
pred_logits, pred_boxes, action_logits = outputs
```

---

## 7. Performance Optimization

### Tuning `score_threshold`

The detection score threshold controls the recall/precision trade-off and directly affects processing time (fewer detections → faster ID and action heads).

- **Default**: `0.3` — balanced for general use
- **Higher** (`0.5`–`0.7`): Fewer false positives; faster; may miss low-confidence animals
- **Lower** (`0.1`–`0.2`): More complete detections; slower; higher false positive rate

Tune by examining precision-recall curves from `scripts/eval_unified.py`.

### Tuning `window_size`

The temporal window size trades accuracy for speed. Smaller windows are faster but reduce the effective temporal context.

- **Default**: `16` frames — uses all three HTM branches at full capacity
- **Reduced** (`8`): Mid and long branches receive fewer frames; action recognition accuracy degrades for long-duration behaviors
- **Increased** (`32`): More temporal context; may improve long-behavior recall; requires more VRAM and is slower

### Disabling video output

Writing an annotated MP4 with OpenCV typically costs 20–30% of total inference time. For pure analysis workflows, disable it:

```bash
python tools/infer_video.py \
    input=video.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference \
    no_video=true
```

### Mixed precision inference

The model respects PyTorch's automatic mixed precision (AMP). On Ampere or newer GPUs, enable `torch.cuda.amp.autocast` for 1.5–2x speedup:

```python
with torch.cuda.amp.autocast():
    outputs = model(images, memory_list)
```

The inference script does not enable AMP by default. Add this context manager to `run_inference()` in `tools/infer_video.py` for faster throughput.

### Input resolution scaling

For very high-resolution source video (e.g., 4K), inference at `img_size=640` already rescales internally. If detection quality is poor at 640 px, try `img_size=960` (note: quadratic memory scaling):

```bash
python tools/infer_video.py \
    input=video.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference \
    img_size=960
```

---

## 8. Visualization Options

### Action timeline plot

Produces a raster-style timeline chart with one row per tracked individual and time on the x-axis, colored by action class.

```bash
python tools/analyze.py mode=timeline \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

Output: `outputs/analysis/experiment_01/action_timeline.png`

### Dataset distribution plot

Before training, inspect the distribution of action labels and detection counts in your annotation file:

```bash
python tools/analyze.py mode=distribution \
    annotation=data/train/annotations.json \
    output=outputs/analysis/train_distribution
```

Outputs:
- `action_distribution.png` — bar chart of action label counts
- `class_distribution.png` — bar chart of class label counts
- `distribution.json` — raw counts in JSON format

### ID switch analysis

```bash
python tools/analyze.py mode=id_switch \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

Output: `outputs/analysis/experiment_01/id_switch_analysis.json` with total switches and per-track events.

### Confidence histogram

Visualize the distribution of detection or action confidence scores to calibrate thresholds:

```bash
python tools/analyze.py mode=confidence \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

Output: `outputs/analysis/experiment_01/confidence_histogram.png`

### Custom visualization using the Python API

```python
from htrtdetr.analysis import ActionTimelineVisualizer

action_names = ["idle", "walk", "groom", "court", "other"]
vis = ActionTimelineVisualizer(action_names)

# Build timelines dict: {track_id: [action_id, action_id, ...]}
timelines = {
    1: [0, 0, 1, 1, 1, 2, 2, 1, 0],
    2: [1, 1, 1, 3, 3, 3, 1, 0, 0],
}
vis.plot(timelines, save_path="outputs/custom_timeline.png")
```
