![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)

# YOAKE: Hierarchical Temporal RT-DETR

YOAKE is a PyTorch model for multi-animal tracking and behavior classification in laboratory video recordings, with a primary focus on *Drosophila melanogaster*. It combines a ResNet-18-based RT-DETR spatial detector with a three-branch dilated temporal module (HTM), a GRU memory-based identity tracking head, and an MLP action classification head — producing bounding boxes, persistent track IDs, and per-frame behavior labels from a single forward pass over a 16-frame sliding window. The model is trained in four sequential stages that progressively unlock module complexity, ensuring stable convergence without requiring end-to-end training from scratch.

---

## Architecture

```
Input frames (B, T, 3, 640, 640)
        |
+-------v---------------------+
|   ResNet-18 Backbone        |  C3, C4, C5 feature maps
+-------+---------------------+
        |
+-------v---------------------+
|   FPN Neck                  |  P3 + P4 + P5  (256 ch each)
+-------+---------------------+
        |
+-------v---------------------+
|   AIFI Encoder              |  Self-attention on P5
+-------+---------------------+
        |
+-------v---------------------+
|   DETR Decoder              |  4-layer transformer, 100 queries
|   -> (B, T, Q, 256)         |  query features per frame
+-------+---------------------+
        |
+-------v---------------------------------------------+
|   Hierarchical Temporal Module (HTM)                |
|                                                     |
|  +----------------+  +------------+  +-----------+ |
|  | Short branch   |  | Mid branch |  | Long      | |
|  | dilation=1     |  | dilation=2 |  | dilation=4| |
|  | receptive=3 fr |  | recept=8fr |  | recept=16f| |
|  | depthwise conv |  | dw conv    |  | dw conv   | |
|  | + residual     |  | + residual |  | + residual| |
|  +-------+--------+  +-----+------+  +-----+-----+ |
|          |                 |               |        |
|          +--------+--------+---------------+        |
|                   | concat + linear projection      |
|                   | -> (B, T, Q, 256)               |
+-------------------+---------------------------------+
                    |
        +-----------+----------+
        |                      |
+-------v-------+    +---------v-----------+
| Memory ID Head|    |  Action Head        |
|               |    |                     |
| GRU memory    |    | spatial feat        |
| per track_id  |    | + temporal feat     |
| -> embedding  |    | + interaction feat  |
| cosine match  |    | (nn-dist, angle,    |
| + IoU match   |    |  velocity, overlap) |
| Hungarian     |    | MLP: 512->256->5    |
+-------+-------+    +---------+-----------+
        |                      |
   track_id               action_id
   id_confidence          action_confidence

Output per detection: [bbox, det_score, class_id, track_id, id_conf, action_id, action_conf]
```

---

## Key Features

- **Single-pass inference**: bounding box detection, identity tracking, and behavior classification in one forward pass
- **Hierarchical temporal reasoning**: three parallel dilated 1D convolution branches capture motion at 3, 8, and 16-frame scales without 3D convolutions
- **GRU memory tracking**: per-track hidden states enable persistent identity assignment across frames with configurable TTL
- **Interaction-aware action classification**: nearest-neighbor distance, relative angle, relative velocity, and pair IoU are fed to the action head alongside visual and temporal features
- **4-stage training**: each stage freezes previously trained modules to stabilize learning of new components
- **Config-first design**: all hyperparameters are managed by typed dataclasses and YAML files; no `argparse` in the main API
- **Mixed precision support**: native PyTorch AMP for 1.5–2x training speedup on Ampere+ GPUs
- **Three model size variants**: `small` (ResNet-18, ~15–20M), `medium` (ResNet-34, ~30–40M), `large` (ResNet-50, ~55–70M) via `get_variant_config()`

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- numpy >= 1.24.0
- scipy >= 1.10.0 (Hungarian matching)
- Pillow >= 9.0.0
- pyyaml >= 6.0
- opencv-python >= 4.8.0 (video I/O, optional)
- matplotlib >= 3.7.0 (visualization, optional)
- wandb >= 0.16.0 (experiment tracking, optional)

**GPU**: NVIDIA RTX 8000 or equivalent recommended. Minimum 16 GB VRAM at default batch size 4. Tested with CUDA 11.8 and CUDA 12.1.

---

## Installation

```bash
# Create and activate the Conda environment
conda create -n yoake python=3.10 -y
conda activate yoake

# Install PyTorch with CUDA (adjust CUDA version to match your driver)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install the package with all optional dependencies
pip install -e ".[all]"
```

For a minimal install without visualization or experiment tracking:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

If your shell cannot find `python` or `pytest`, use the repository CI workflow as the reference environment.

---

## Quick Start

### Verify the installation

```bash
python examples/smoke_test.py
```

This runs a full forward pass and a two-epoch training loop on synthetic data. No GPU or real data required. Expected output: `Smoke test passed.`

### Run inference on a video

```bash
python tools/infer_video.py \
    input=data/videos/experiment_01.mp4 \
    checkpoint=weights/full_model_best.pth \
    output_dir=outputs/inference \
    score_threshold=0.3
```

Outputs:
- `outputs/inference/experiment_01_pred.mp4` — annotated video
- `outputs/inference/experiment_01_pred.json` — structured predictions

### Python API

```python
from htrtdetr.config import HTRTDETRConfig
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_checkpoint
import torch

cfg = HTRTDETRConfig.from_yaml("configs/default.yaml")
model = build_model(cfg.model).cuda()
load_checkpoint(model, "weights/full_model_best.pth")
model.eval()

memory_list = model.create_memory_list(batch_size=1, device=torch.device("cuda"))
images = torch.randn(1, 16, 3, 640, 640).cuda()  # (B, T, C, H, W)

with torch.no_grad():
    outputs = model(images, memory_list)

# outputs.det_results[0]: dict with 'boxes', 'scores', 'track_ids'
# outputs.action_logits: (N_det, num_actions) action classification logits
```

---

## Dataset Preparation

YOAKE uses a custom JSON annotation format (v1.1) described in full in [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md).

### Annotation structure (summary)

```json
{
  "meta": { "version": "1.1", "fps_default": 25.0, "image_root": "images/" },
  "class_names": ["fly"],
  "action_names": ["idle", "walk", "groom", "interact", "other"],
  "videos": [
    {
      "video_id": "vid_001", "fps": 25.0, "width": 1024, "height": 1024,
      "frames": [
        {
          "frame_index": 0, "image_path": "images/vid_001/000000.jpg",
          "objects": [
            { "object_id": 1, "bbox": [112, 87, 132, 109], "class_id": 0,
              "track_id": 1, "action_id": 1 }
          ]
        }
      ]
    }
  ]
}
```

Bounding boxes use `[x1, y1, x2, y2]` pixel coordinates. `action_id: -1` marks unannotated frames.

### Prepare a dataset

```bash
# Step 1: Convert CSV annotations to JSON (when convert_annotations.py is available)
python tools/convert_annotations.py \
    input=data/raw/annotations.csv \
    output=data/raw/annotations.json \
    fps=25.0 width=1024 height=1024

# Step 2: Validate annotations
python -c "
from htrtdetr.data.annotation import load_annotation, validate_annotation, print_annotation_stats
anno = load_annotation('data/raw/annotations.json')
print(validate_annotation(anno))
print_annotation_stats(anno)
"

# Step 3: Split into train/val/test at the video level
# See docs/DATASET_FORMAT.md for a Python snippet

# Step 4: Inspect class and action distributions
python tools/analyze.py mode=distribution \
    annotation=data/train/annotations.json \
    output=outputs/analysis/train_dist
```

---

## Staged Training

Full details, hyperparameter explanations, and troubleshooting are in [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

### Stage 1 — Spatial Detector

Trains ResNet-18 + FPN + AIFI + DETR decoder to detect animals. All other modules are inactive.

```bash
python scripts/train_stage1_detector.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    output_dir=outputs/stage1 \
    num_epochs=100 \
    batch_size=4
```

### Stage 2 — Action Head

Freezes the detector. Trains the HTM and MLP action head using geometric track features.

```bash
python scripts/train_stage2_action.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/stage2 \
    num_epochs=30
```

### Stage 3 — ID Head

Freezes the detector. Trains the HTM and GRU memory ID head. Runs in parallel with Stage 2.

```bash
python scripts/train_stage3_id.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/stage3 \
    num_epochs=30
```

### Stage 4 — Unified Fine-Tuning

Unlocks all modules. Initializes from Stages 1–3. Uses combined detection + action + ID + temporal smoothness loss.

```bash
python scripts/train_stage4_unified.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    stage2_ckpt=outputs/stage2/checkpoint_best.pth \
    stage3_ckpt=outputs/stage3/checkpoint_best.pth \
    output_dir=outputs/stage4 \
    num_epochs=30
```

---

## Evaluation

### Stage 1 — Detection metrics (AP50, AP75)

These are class-aware aggregate detection metrics. The current evaluator does not emit per-class AP tables.

```bash
python scripts/eval_stage1.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/eval_stage1
```

### Stage 2 — Action classification metrics (per-class F1)

```bash
python scripts/eval_stage2.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage2/checkpoint_best.pth \
    output_dir=outputs/eval_stage2
```

### Stage 3 — Tracking metrics (IDF1, ID switches)

```bash
python scripts/eval_stage3.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage3/checkpoint_best.pth \
    output_dir=outputs/eval_stage3
```

### Unified evaluation — all metrics together

```bash
python scripts/eval_unified.py \
    anno=data/test/annotations.json \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/eval_unified \
    score_threshold=0.3
```

Results are written to `outputs/eval_unified/results_unified.json`.

---

## Inference

Full inference guide: [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md)

### Single video

```bash
python tools/infer_video.py \
    input=data/videos/experiment_01.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference \
    score_threshold=0.3 \
    window_size=16 \
    action_names=idle,walk,groom,interact,other
```

### JSON-only output (no annotated video)

```bash
python tools/infer_video.py \
    input=data/videos/experiment_01.mp4 \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/inference \
    no_video=true
```

### Output format

Each detection in the output JSON contains:

```json
{
  "frame_idx": 42,
  "detections": [
    { "bbox": [112.3, 87.1, 132.7, 109.4], "score": 0.92,
      "track_id": 1, "action_id": 1 }
  ]
}
```

---

## Visualization and Analysis

```bash
# Per-track action timeline chart
python tools/analyze.py mode=timeline \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01

# Action and class distribution of a dataset split
python tools/analyze.py mode=distribution \
    annotation=data/train/annotations.json \
    output=outputs/analysis/train_dist

# ID switch analysis
python tools/analyze.py mode=id_switch \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01

# Detection confidence histogram
python tools/analyze.py mode=confidence \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

---

## Ablation Studies

Run variants with different configurations and use the ablation aggregator to compare results:

```bash
# Example: ablate the long temporal branch
python scripts/train_stage2_action.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/ablation/no_long_branch \
    model.temporal.long_branch.num_frames=3

python scripts/eval_stage2.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/ablation/no_long_branch/checkpoint_best.pth \
    output_dir=outputs/ablation/no_long_branch

# Aggregate all ablation results
python tools/analyze.py mode=ablation \
    results_dir=outputs/ablation
```

---

## Export

### ONNX export

```python
import torch
from htrtdetr.config.config import get_stage4_config
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_checkpoint

cfg = get_stage4_config()
model = build_model(cfg.model)
load_checkpoint(model, "outputs/stage4/checkpoint_best.pth")
model.eval()

dummy = torch.randn(1, 16, 3, 640, 640)
memory = model.create_memory_list(1, torch.device("cpu"))

torch.onnx.export(
    model, (dummy, memory), "weights/ht_rtdetr.onnx",
    opset_version=17,
    input_names=["images", "memory"],
    output_names=["pred_logits", "pred_boxes", "action_logits"],
)
```

---

## Benchmark

| Stage | Metric | Value |
|-------|--------|-------|
| Stage 1 | AP50 (val) | — |
| Stage 1 | AP75 (val) | — |
| Stage 2 | Action Macro F1 (val) | — |
| Stage 3 | IDF1 (val) | — |
| Stage 4 | AP50 (test) | — |
| Stage 4 | IDF1 (test) | — |
| Stage 4 | Action Macro F1 (test) | — |
| Runtime | Inference FPS (RTX 8000) | — |

Fill in these values after running the evaluation scripts on your trained model. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the full metrics table including per-class action F1 and runtime memory.

---

## Results

Quantitative results will be reported here after the full training pipeline is run on the target Drosophila behavior dataset. Placeholder — see [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the evaluation methodology and metric definitions.

---

## Project Structure

```
YOAKE/
├── src/
│   └── htrtdetr/
│       ├── config/              # HTRTDETRConfig dataclass + YAML loader
│       ├── data/
│       │   ├── annotation.py    # JSON v1.1 format: load, save, validate
│       │   ├── fly_dataset.py   # FlyDataset, build_dataloaders
│       │   ├── collate.py       # batch collation
│       │   └── feature_builder.py
│       ├── models/
│       │   ├── detector/        # ResNet-18 + FPN + AIFI + DETR decoder
│       │   ├── temporal/        # HierarchicalTemporalModule (HTM)
│       │   ├── id_head/         # MemoryIDHead (GRU)
│       │   ├── action_head/     # ActionHead (MLP)
│       │   ├── fusion/          # MultiHeadFeatureRouter
│       │   └── ht_rtdetr.py     # Unified HTRTDETRModel
│       ├── training/
│       │   ├── stage_trainers.py  # Stage1–4 Trainer classes
│       │   ├── losses.py          # Combined loss functions
│       │   ├── matching.py        # Hungarian matcher
│       │   └── optimizer.py       # AdamW + cosine scheduler
│       ├── evaluation/
│       │   └── evaluator.py     # DetectionEvaluator, ActionEvaluator, TrackingEvaluator
│       ├── inference/
│       │   └── inferencer.py    # Inferencer class
│       ├── analysis/
│       │   └── analyzer.py      # Timeline, distribution, ablation tools
│       └── utils/
│           ├── misc.py          # load_checkpoint, set_seed, cxcywh_to_xyxy, ...
│           └── logging.py       # get_logger
├── scripts/
│   ├── train_stage1_detector.py
│   ├── train_stage2_action.py
│   ├── train_stage3_id.py
│   ├── train_stage4_unified.py
│   ├── eval_stage1.py
│   ├── eval_stage2.py
│   ├── eval_stage3.py
│   └── eval_unified.py
├── tools/
│   ├── infer_video.py           # Video inference with MP4 + JSON output
│   └── analyze.py               # Timeline / distribution / ablation / ID-switch analysis
├── configs/
│   ├── default.yaml             # All hyperparameter defaults
│   ├── fly_dataset.yaml         # Dataset-specific overrides
│   ├── stage1_detector.yaml     # Stage 1 config overrides
│   └── stage4_unified.yaml      # Stage 4 config overrides
├── docs/
│   ├── MODEL_CARD.md            # HuggingFace-style model card
│   ├── DATASET_FORMAT.md        # Annotation format specification v1.1
│   ├── TRAINING_GUIDE.md        # Stage-by-stage training instructions
│   ├── INFERENCE_GUIDE.md       # Inference, ONNX, and analysis guide
│   ├── RELEASE_CHECKLIST.md     # Pre-release verification checklist
│   ├── architecture.md          # Architecture diagrams and module details
│   └── assumptions.md           # Design decisions and known limitations
├── examples/
│   └── smoke_test.py            # Dependency-free integration test
├── weights/                     # Pre-trained checkpoints (not tracked in git)
├── data/                        # Training data (not tracked in git)
├── outputs/                     # Training and inference outputs (not tracked in git)
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Citation

If you use YOAKE in your research, please cite:

```bibtex
@software{htrtdetr2026,
  title   = {YOAKE: Hierarchical Temporal RT-DETR for Drosophila Behavior Analysis},
  author  = {TODO},
  year    = {2026},
  url     = {TODO},
  version = {0.1.0},
  license = {MIT},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
