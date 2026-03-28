# YOAKE Manual

> **YOAKE** — Hierarchical Temporal RT-DETR for Multi-Animal Tracking and Behavior Analysis

---

## Table of Contents / 目次

- [English Manual](#english-manual)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Installation](#installation)
  - [Dataset Format](#dataset-format)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Configuration Reference](#configuration-reference)
  - [Output Formats](#output-formats)
- [日本語マニュアル](#日本語マニュアル)
  - [概要](#概要)
  - [アーキテクチャ](#アーキテクチャ)
  - [インストール](#インストール)
  - [データセット形式](#データセット形式)
  - [学習](#学習)
  - [評価](#評価)
  - [推論](#推論)
  - [設定リファレンス](#設定リファレンス)
  - [出力形式](#出力形式)

---

# English Manual

## Overview

**YOAKE** is a PyTorch-based deep learning framework for **multi-animal detection, identity tracking, and behavior classification** from laboratory video recordings. It was developed primarily for *Drosophila melanogaster* (fruit fly) experiments but is adaptable to other small animal species.

### Key Features

| Feature | Description |
|---------|-------------|
| **Single-pass inference** | Simultaneous detection, ID tracking, and action classification |
| **Hierarchical temporal reasoning** | 3 parallel dilated-convolution branches (3 / 8 / 16-frame receptive fields) |
| **Persistent identity tracking** | GRU-based memory head with Hungarian matching |
| **End-to-end trainable** | 4-stage progressive training converges to a unified model |
| **Efficient** | No 3D convolutions — lightweight temporal module on top of 2D detector |

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1.0 |
| torchvision | ≥ 0.16.0 |
| CUDA | 11.8+ (recommended) |
| GPU VRAM | 16 GB+ (tested on RTX 8000) |

---

## Architecture

YOAKE consists of four main modules that are trained progressively.

```
Input Video (B, T, 3, 640, 640)
        │
        ▼
┌────────────────────────────┐
│  Stage 1: Spatial Detector │  ResNet-18 → FPN → AIFI → DETR Decoder
│  bbox + class + queries    │  (B, 100, 4) + (B, 100, C) + (B, 100, 256)
└────────────────┬───────────┘
                 │  frozen after Stage 1
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│  Stage 2      │  │  Stage 3         │  ← can train in parallel
│  Action Head  │  │  ID Head (GRU)   │
│  MLP + focal  │  │  embedding + HAM │
└───────┬───────┘  └────────┬─────────┘
        └──────────┬────────┘
                   ▼
        ┌──────────────────────┐
        │  Stage 4: Unified    │  all modules fine-tuned jointly
        │  Fine-Tuning         │  combined loss: det + action + ID + smooth
        └──────────────────────┘
```

### Module Details

#### Spatial Detector (Stage 1)

| Sub-module | Details |
|-----------|---------|
| Backbone | ResNet-18 (ImageNet pretrained), extracts C3/C4/C5 |
| Neck (FPN) | Feature Pyramid Network, 256-ch outputs P3/P4/P5 |
| Encoder (AIFI) | Self-attention on P5 for cross-scale interaction |
| Decoder | 4-layer DETR transformer, 100 learnable queries |
| Output | 100 bbox proposals + class logits + 256-d query features |
| Loss | Focal classification + L1 bbox + GIoU |

#### Hierarchical Temporal Module (HTM)

Three parallel 1D dilated convolution branches:

| Branch | Dilation | Effective Receptive Field | Purpose |
|--------|----------|--------------------------|---------|
| Short  | 1        | 3 frames                 | Frame-to-frame fine motion |
| Mid    | 2        | 8 frames                 | Medium-term behavioral patterns |
| Long   | 4        | 16 frames                | Long-range locomotion trends |

- Depthwise separable convolutions for efficiency
- Residual connections within each branch
- Outputs fused and projected to 256-d

#### Memory-based ID Head (Stage 3)

```
Detection feature (N, 256)
      ↓ GRU memory update (per track)
ID embedding (N, 128)  [L2 normalized]
      ↓ Cosine similarity (0.5) + IoU matching (0.5)
      ↓ Hungarian assignment
track_id  +  id_confidence
```

Key parameters: `embedding_dim=128`, `max_ids=50`, `memory_ttl=30` frames, `new_id_threshold=0.5`

> **Note (Stage 2 & 3 Training):** During training of Stages 2 and 3, the visual detector is bypassed entirely. Instead, pre-computed per-track geometric features (bounding box trajectory, normalized position, velocity) are fed directly into the HTM via `forward_geo_sequence()`. This avoids wasting GPU time running the frozen detector and allows larger effective batch sizes.

#### Action Head (Stage 2)

```
[spatial_feat | temporal_feat | interaction_feat]  ← concat
         ↓ Linear → LayerNorm → ReLU
         ↓ MLP  512 → 256
         ↓ Linear → num_actions logits
```

Default actions: `idle`, `walk`, `groom`, `interact`, `other`

Interaction features (optional): nearest-neighbor distance, relative angle, relative velocity, pairwise IoU

---

## Installation

### 1. Create Environment

```bash
conda create -n yoake python=3.10 -y
conda activate yoake
```

### 2. Install PyTorch

```bash
# Adjust CUDA version as needed (cu118 / cu121 / cpu)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install YOAKE

```bash
cd /path/to/YOAKE

# Full install (includes visualization tools + W&B logging)
pip install -e ".[all]"

# Minimal install
pip install -e .
```

### 4. Verify Installation

```bash
python examples/smoke_test.py
# Expected: "Smoke test passed."
```

The smoke test runs a full forward pass and 2-epoch mini-training on synthetic data without requiring a GPU.

---

## Dataset Format

YOAKE uses **JSON Annotation Format v1.1**.

### File Structure

```json
{
  "meta": {
    "version": "1.1",
    "fps_default": 25.0,
    "image_root": "images/"
  },
  "class_names": ["fly"],
  "action_names": ["idle", "walk", "groom", "interact", "other"],
  "videos": [
    {
      "video_id": "vid_001",
      "fps": 25.0,
      "width": 1024,
      "height": 1024,
      "num_frames": 300,
      "frames": [
        {
          "frame_index": 0,
          "image_path": "images/vid_001/000000.jpg",
          "objects": [
            {
              "object_id": 1,
              "bbox": [112.0, 87.5, 132.0, 109.0],
              "class_id": 0,
              "track_id": 1,
              "action_id": 1,
              "occluded": false
            }
          ]
        }
      ]
    }
  ]
}
```

### Coordinate System

- Format: `[x1, y1, x2, y2]` in pixel coordinates (xyxy)
- Origin: top-left `(0, 0)`
- Constraint: `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`
- Internally converted to normalized `cxcywh` for loss computation

### Special Values

| Value | Meaning |
|-------|---------|
| `track_id: -1` | Untracked object |
| `action_id: -1` | Unannotated frame (ignored in loss) |
| `occluded: true` | Partially visible animal |
| `is_crowd: true` | Overlapping group (excluded from metrics) |

### Data Preparation Workflow

```bash
# 1. Extract frames from video
ffmpeg -i raw_videos/vid_001.mp4 -q:v 2 data/images/vid_001/%06d.jpg

# 2. Annotate with CVAT, VGG, or custom tool
#    → bounding boxes, track IDs, action labels

# 3. Convert CSV annotations to YOAKE JSON (if needed)
python tools/convert_annotations.py \
    input=data/raw/annotations.csv \
    output=data/raw/annotations.json \
    fps=25.0 width=1024 height=1024

# 4. Validate annotations
python -c "
from htrtdetr.data.annotation import load_annotation, validate_annotation
anno = load_annotation('data/raw/annotations.json')
print(validate_annotation(anno))
"

# 5. Split into train / val / test (video-level split to prevent leakage)
#    → data/train/annotations.json  (~70%)
#    → data/val/annotations.json    (~15%)
#    → data/test/annotations.json   (~15%)
```

### Annotation Quality Guidelines

| Requirement | Recommended | Minimum |
|-------------|-------------|---------|
| Annotation density | 100% frames | 80% frames |
| Minimum track length | — | 16 frames (one temporal window) |
| Bounding box tightness | Tight body contour | — |
| Video metadata (fps, w, h) | Must be accurate | — |
| Image format | JPEG quality ≥ 90 | — |

---

## Training

Training proceeds in four stages. Stages 2 and 3 are independent and can run on separate GPUs simultaneously.

### Quick Start — `yoake` CLI (recommended)

After installing (`pip install -e .`), use the single `yoake` command:

```bash
# Stage 1: Spatial Detector
yoake train stage=1 data.train_root=data/train data.val_root=data/val

# Stage 2: Action Head  (loads stage1 checkpoint automatically)
yoake train stage=2

# Stage 3: ID Head  (can run in parallel with Stage 2 on a separate GPU)
yoake train stage=3

# Stage 4: Unified Fine-Tuning
yoake train stage=4
```

Outputs go to `runs/train/stage{N}/` by default.

Override any config value with `key=value`:

```bash
yoake train stage=1 train.max_epochs=100 data.batch_size=8 model.variant=small
```

### Stage 1: Spatial Detector

**Goal:** Learn to detect and localize animals in single frames.
**Typical epochs:** 50–500 | **Time/epoch:** ~8 min (RTX 8000)

```bash
yoake train stage=1 \
    data.train_root=data/train \
    data.val_root=data/val \
    train.max_epochs=100 \
    data.batch_size=4
```

### Stage 2: Action Head

**Goal:** Learn to classify behavior from temporal query features.
**Typical epochs:** 20–80 | **Time/epoch:** ~3 min

```bash
yoake train stage=2 \
    data.train_root=data/train \
    data.val_root=data/val
```

### Stage 3: ID Head

**Goal:** Learn persistent identity embeddings for re-identification.
**Typical epochs:** 20–80 | **Time/epoch:** ~3 min
*(Can run in parallel with Stage 2 on a separate GPU)*

```bash
yoake train stage=3
```

### Stage 4: Unified Fine-Tuning

**Goal:** Joint fine-tuning of all modules with combined loss.
**Typical epochs:** 20–50 | **Time/epoch:** ~10 min

```bash
yoake train stage=4
```

### Base Directory (`root=`)

Scripts accept `root=<path>` to set the working root. The default is the repository root (auto-detected). Checkpoints from previous stages are searched under `{root}/runs/train/`.

```bash
yoake train stage=2 root=/data/myproject
```

### Using a config file

```bash
yoake train stage=1 config=configs/stage1_small.yaml train.max_epochs=200
```

### Legacy script interface (deprecated)

The old per-stage scripts still work but are deprecated:

```bash
python scripts/train_stage1.py data.train_root=data/train
python scripts/train_stage2.py
python scripts/train_stage3.py
python scripts/train_stage4.py
```

> ⚠️ `scripts/train_stage1_detector.py`, `train_stage2_action.py`, `train_stage3_id.py`, `train_stage4_unified.py` use the deprecated `stage_trainers.py` backend. Prefer the scripts above or the `yoake` CLI.

### Key Hyperparameters

| Category | Parameter | Default | Notes |
|----------|-----------|---------|-------|
| Optimizer | `optimizer.lr` | `1e-4` | Base learning rate (Stage 4 uses `1e-5`) |
| | `backbone_lr_factor` | `0.1` | Backbone LR multiplier (set to `0.0` in Stage 2 to fully freeze detector via zero LR) |
| | `grad_clip_norm` | `0.1` | Gradient clipping |
| Scheduler | `warmup_epochs` | `5` | Cosine warm-up |
| Data | `batch_size` | `4` | Per-GPU |
| | `window_size` | `16` | Frames per sample |
| | `window_stride` | `8` | Sliding window stride |
| Loss | `w_bbox_l1` | `5.0` | L1 bbox weight |
| | `w_bbox_giou` | `2.0` | GIoU bbox weight |
| | `w_action` | `1.0` | Action loss weight |
| | `w_id_cls` | `1.0` | ID classification loss weight |
| | `focal_gamma` | `2.0` | Focal loss gamma |
| | `focal_alpha` | `0.25` | Focal loss alpha |
| | `w_id_metric` | `0.5` | Metric learning (triplet) loss weight |
| | `w_temporal_smooth` | `0.1` | Temporal smoothness regularization weight |

### Estimated Total Training Time

| Stage | Epochs | Time |
|-------|--------|------|
| Stage 1 | 100 | ~13 h |
| Stage 2 + 3 (parallel) | 30 | ~1.5 h |
| Stage 4 | 30 | ~5 h |
| **Total** | — | **~20 h** |

*Measured on a single RTX 8000 with `batch_size=4`, `num_workers=8`.*

---

## Evaluation

### `yoake val` CLI (recommended)

```bash
yoake val stage=1
yoake val stage=2
yoake val stage=3
yoake val stage=4
```

If `checkpoint=` is omitted, `runs/train/stage{N}/weights/best.pth` is searched automatically.

### Primary Metrics per Stage

| Stage | Primary metric | Direction | Notes |
|-------|----------------|-----------|-------|
| Stage 1 | AP50 | higher | COCO-style detection AP at IoU=0.5 |
| Stage 2 | macro_F1 | higher | Unweighted mean of per-class F1 |
| Stage 3 | IDF1 | higher | Identity F1 (harmonic mean of IDP / IDR) |
| Stage 4 | composite | higher | `0.4*AP50 + 0.3*macro_F1 + 0.3*IDF1` |

### Stage 1: Detection

```bash
yoake val stage=1 \
    data.val_root=data/val \
    checkpoint=runs/train/stage1/weights/best.pth
```

Outputs: `runs/val/stage1/val_results.json`, `metrics_latest.json`

Metrics: **AP50**, **AP75**, class-aware aggregate AP / recall. The current evaluator is not a per-class AP table exporter.

### Stage 2: Action Classification

```bash
yoake val stage=2 \
    data.val_root=data/val \
    checkpoint=runs/train/stage2/weights/best.pth
```

Outputs: `runs/val/stage2/val_results.json`, `confusion_matrix.png`, `per_class_metrics.json`, `per_class_metrics.csv`

Metrics: **macro F1**, weighted F1, per-class F1 / precision / recall, confusion matrix

### Stage 3: Identity Tracking

```bash
yoake val stage=3 \
    data.val_root=data/val \
    checkpoint=runs/train/stage3/weights/best.pth
```

Metrics: **IDF1**, **IDP**, **IDR**, ID switch count (IDSW). These are lightweight matched-track validation metrics used consistently across Stage 3 / 4.

### Unified Evaluation (Stage 4)

```bash
yoake val stage=4 \
    data.val_root=data/test \
    checkpoint=runs/train/stage4/weights/best.pth
```

Outputs: `runs/val/stage4/val_results.json`, `confusion_matrix.png`, `per_class_metrics.json`

Metrics: **composite** (AP50 + macro_F1 + IDF1 weighted sum), plus all individual stage metrics.

---

## Inference

```bash
yoake predict \
    source=path/to/video.mp4 \
    weights=runs/train/stage4/weights/best.pth \
    score_threshold=0.3
```

Output goes to `runs/predict/` by default. Override with `output_dir=`.

```bash
# Run on a directory of videos
yoake predict \
    source=data/videos/ \
    weights=runs/train/stage4/weights/best.pth \
    output_dir=runs/predict/myexp
```

### Inference Output

| File | Description |
|------|-------------|
| `<stem>_pred.mp4` | Annotated video with bbox, track ID, action overlaid |
| `<stem>_pred.json` | Per-frame detection results (see [Output Formats](#output-formats)) |

---

## Configuration Reference

All configuration is managed by typed dataclasses and can be overridden on the command line with `key=value` syntax or via YAML files.

### Model

```yaml
model:
  detector:
    backbone:
      name: resnet18          # resnet18 | resnet34 | resnet50
      pretrained: true
    num_queries: 100
    num_decoder_layers: 4
    num_heads: 8
    score_threshold: 0.3
  temporal:
    short_branch:
      num_frames: 3
    mid_branch:
      num_frames: 8
    long_branch:
      num_frames: 16
    fusion_method: concat_proj   # concat_proj | sum | attention
  id_head:
    embedding_dim: 128
    memory_dim: 256
    max_ids: 50
    new_id_threshold: 0.5
    memory_ttl: 30
  action_head:
    num_actions: 5
    hidden_dims: [512, 256]
    dropout: 0.1
    use_interaction: true
```

### Model Size Variants

Three pre-defined model size variants are available via `build_model_config()` / `get_variant_config()`:

| Variant | Backbone | hidden_dim | Approx. params |
|---------|----------|------------|----------------|
| `small` | ResNet-18 | 128 | ~15–20M |
| `medium` (default) | ResNet-34 | 256 | ~30–40M |
| `large` | ResNet-50 | 512 | ~55–70M |

Python API:

```python
from htrtdetr.config.config import get_variant_config
cfg = get_variant_config("small", stage=1)
cfg = get_variant_config("large", stage=4, overrides={"data.batch_size": 2})
```

### Data

```yaml
data:
  batch_size: 4
  image_size: [640, 640]
  window_size: 16
  window_stride: 8
  num_workers: 8
  augment_train: true
```

### Training

```yaml
train:
  stage: 1                       # 1 | 2 | 3 | 4
  max_epochs: 100
  early_stopping_patience: 20
  use_amp: true                  # Automatic Mixed Precision
  use_wandb: false               # W&B experiment tracking
```

### Class Imbalance (Stage 2 / 4)

```yaml
loss:
  imbalance_strategy: class_weight   # none | class_weight | focal
  # class_weight: scans up to 200 training batches, computes Laplace-smoothed
  #               inverse-frequency weights, calls ActionLoss.set_class_weights()
  # focal:        uses focal loss (gamma=2.0) without per-class weighting
  # none:         plain cross-entropy
```

CLI override:

```bash
yoake train stage=2 loss.imbalance_strategy=class_weight
```

> **Note:** `imbalance_strategy=sampler` is not implemented and will fall back to `class_weight` with a warning.

Computed class weights and per-class sample counts are saved to `args.yaml` after the first scan.

### Metrics Configuration

```yaml
metrics:
  stage2_primary: macro_f1        # macro_f1 | val_loss
  stage3_primary: idf1            # idf1 | val_loss
  stage4_primary: composite       # composite | ap50 | val_loss
  composite_ap50: 0.4             # weight for AP50 in Stage 4 composite score
  composite_macro_f1: 0.3         # weight for macro_F1
  composite_idf1: 0.3             # weight for IDF1
```

CLI overrides:

```bash
yoake train stage=4 metrics.stage4_primary=composite
yoake train stage=4 metrics.composite_ap50=0.5 metrics.composite_macro_f1=0.25 metrics.composite_idf1=0.25
```

---

## Output Formats

### Run Directory Structure

```
runs/
  train/stage{N}/
    weights/
      best.pth            # Best checkpoint by primary metric (EMA weights when available)
      last.pth            # Checkpoint from last epoch
    args.yaml             # Full config snapshot with class_counts, class_weights
    results.csv           # Per-epoch metrics (unified columns across all stages)
    val/
      epoch_NNN.json      # Per-epoch detailed evaluation results
    plots/
      loss_history.png    # Train/val loss curve
      f1_history.png      # macro_F1 over epochs (Stage 2/4)
      idf1_history.png    # IDF1 over epochs (Stage 3/4)
      composite_history.png  # Composite score over epochs (Stage 4)
      confusion_matrix.png   # Confusion matrix PNG (Stage 2/4, requires matplotlib)
      confusion_matrix.json  # Confusion matrix JSON (always saved)
    metrics_latest.json   # Latest val metrics + best-so-far tracking info
    train.log
  val/stage{N}/
    val_results.json
    confusion_matrix.png  # Stage 2/4
    per_class_metrics.json
    per_class_metrics.csv
    metrics_latest.json
  predict/                # Inference results
  analyze/                # Analysis results

```

### results.csv Columns

All stages share a unified column set; columns not applicable to a stage are empty strings.
`val_loss` is stored as an unweighted validation objective so runs stay comparable even when train-time class weighting is enabled.

| Column | Stage | Description |
|--------|-------|-------------|
| `epoch` | all | Epoch number |
| `train_loss` | all | Training loss |
| `val_loss` | all | Unweighted validation loss |
| `val_loss_kind` | all | Current validation loss definition (`unweighted`) |
| `ap50` | 1, 4 | AP @ IoU=0.5 |
| `macro_f1` | 2, 4 | Macro-averaged F1 |
| `weighted_f1` | 2, 4 | Sample-weighted F1 |
| `idf1` | 3, 4 | Identity F1 |
| `idp` | 3, 4 | Identity Precision |
| `idr` | 3, 4 | Identity Recall |
| `idsw` | 3, 4 | ID switch count |
| `composite_score` | 4 | 0.4*AP50 + 0.3*macro_F1 + 0.3*IDF1 |
| `lr` | all | Learning rate at epoch end |
| `primary_metric` | all | Name of the primary metric for this stage |
| `primary_metric_value` | all | Value of the primary metric |
| `best_so_far` | all | Best primary metric value seen so far |

### Validation Metric JSONs

`val/epoch_NNN.json` and `metrics_latest.json` share the same core validation keys:

- `val_loss`, `val_loss_kind`
- `val_AP50` when detection metrics are active
- `macro_f1`, `weighted_f1` when action metrics are active
- `idf1`, `idp`, `idr`, `IDSW` when tracking metrics are active
- `primary_metric`, `primary_metric_value`, `best_so_far`

For Stage 3 / 4, `idp` / `idr` come from the same lightweight validation-time tracking proxy as `idf1`.

---

## Testing and CI

Development/test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest -q
```

CI:

- GitHub Actions workflow: `.github/workflows/tests.yml`
- Python: 3.10
- Install step: `pip install -r requirements-dev.txt`
- Test step: `pytest -q`

### Inference JSON

```json
{
  "metadata": {
    "model_version": "0.1.0",
    "fps": 25.0,
    "width": 1024,
    "height": 1024,
    "num_frames": 300
  },
  "frames": [
    {
      "frame_idx": 0,
      "detections": [
        {
          "bbox": [112.3, 87.1, 132.7, 109.4],
          "score": 0.92,
          "class_id": 0,
          "track_id": 1,
          "id_confidence": 0.87,
          "action_id": 1,
          "action_confidence": 0.78
        }
      ]
    }
  ]
}
```

### Unified Evaluation JSON

```json
{
  "detection": {
    "ap50": 0.85,
    "ap75": 0.72,
    "per_class": { "fly": { "ap50": 0.85, "ap75": 0.72 } }
  },
  "action": {
    "macro_f1": 0.78,
    "per_class": {
      "idle":  { "f1": 0.82, "precision": 0.80, "recall": 0.84 },
      "walk":  { "f1": 0.75, "precision": 0.74, "recall": 0.76 },
      "groom": { "f1": 0.79, "precision": 0.81, "recall": 0.77 },
      "interact": { "f1": 0.76, "precision": 0.77, "recall": 0.75 },
      "other": { "f1": 0.71, "precision": 0.70, "recall": 0.72 }
    }
  },
  "tracking": {
    "idf1": 0.88,
    "id_switches": 42,
    "mota": 0.79
  }
}
```

---

---

# 日本語マニュアル

## 概要

**YOAKE** は、実験室での映像から**多個体の検出・ID追跡・行動分類**を同時に行うPyTorchベースの深層学習フレームワークです。主に*Drosophila melanogaster*（ショウジョウバエ）向けに開発されていますが、他の小型動物にも適用可能です。

### 主な特徴

| 特徴 | 説明 |
|------|------|
| **シングルパス推論** | 検出・ID追跡・行動分類を1回の推論で同時実行 |
| **階層的時間推論** | 3本の並列拡張畳み込みブランチ（3/8/16フレームの受容野） |
| **持続的ID追跡** | GRUメモリヘッド + ハンガリアンマッチング |
| **エンドツーエンド学習** | 4段階プログレッシブ学習で統合モデルに収束 |
| **軽量設計** | 3D畳み込み不使用 — 2D検出器上の軽量時間モジュール |

### 動作環境

| コンポーネント | 要件 |
|--------------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1.0 |
| torchvision | ≥ 0.16.0 |
| CUDA | 11.8以上（推奨） |
| GPU VRAM | 16 GB以上（RTX 8000 で検証済み） |

---

## アーキテクチャ

YOAKEは4つの主要モジュールから構成され、段階的に学習します。

```
入力映像 (B, T, 3, 640, 640)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: 空間検出器          │  ResNet-18 → FPN → AIFI → DETR Decoder
│  bbox + クラス + クエリ特徴量   │  (B,100,4) + (B,100,C) + (B,100,256)
└───────────────┬─────────────┘
                │  Stage 1 以降はフリーズ
       ┌────────┴────────┐
       ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│  Stage 2      │  │  Stage 3         │  ← 並列学習可能
│  行動ヘッド    │  │  IDヘッド (GRU)   │
│  MLP + focal  │  │  埋め込み + HAM   │
└───────┬───────┘  └────────┬─────────┘
        └──────────┬────────┘
                   ▼
        ┌──────────────────────┐
        │  Stage 4: 統合微調整  │  全モジュール共同学習
        │                      │  損失 = 検出 + 行動 + ID + 平滑化
        └──────────────────────┘
```

### モジュール詳細

#### 空間検出器（Stage 1）

| サブモジュール | 詳細 |
|-------------|------|
| バックボーン | ResNet-18（ImageNet事前学習済み）、C3/C4/C5特徴量を抽出 |
| ネック (FPN) | 特徴ピラミッドネットワーク、256ch出力 P3/P4/P5 |
| エンコーダ (AIFI) | P5 上のセルフアテンションでスケール間相互作用を学習 |
| デコーダ | 4層DETRトランスフォーマー、100個の学習可能クエリ |
| 出力 | 100個のbbox候補 + クラスロジット + 256次元クエリ特徴量 |
| 損失 | Focal分類損失 + L1 bbox + GIoU |

#### 階層的時間モジュール（HTM）

3本の並列1D拡張畳み込みブランチ:

| ブランチ | 拡張率 | 実効受容野 | 目的 |
|--------|-------|----------|------|
| Short  | 1     | 3フレーム  | フレーム間の細かい動き |
| Mid    | 2     | 8フレーム  | 中期的な行動パターン |
| Long   | 4     | 16フレーム | 長期的な移動トレンド |

- 効率化のためのDepthwise Separable畳み込み
- 各ブランチ内のResidual接続
- 出力を結合して256次元に射影

#### メモリベースIDヘッド（Stage 3）

```
検出特徴量 (N, 256)
      ↓ GRUメモリ更新（トラックごと）
ID埋め込み (N, 128)  [L2正規化]
      ↓ コサイン類似度 (0.5) + IoUマッチング (0.5)
      ↓ ハンガリアン割り当て
track_id  +  id_confidence
```

主要パラメータ: `embedding_dim=128`, `max_ids=50`, `memory_ttl=30`フレーム, `new_id_threshold=0.5`

> **注意（Stage 2・3 学習時）:** Stage 2/3 の学習では視覚的な検出器を迂回し、トラックごとのジオメトリ特徴量（bbox軌跡・正規化座標・速度）を直接 HTM に入力する `forward_geo_sequence()` を使用します。これにより frozen 検出器の計算を省いて GPU 効率を高めます。

#### 行動ヘッド（Stage 2）

```
[spatial_feat | temporal_feat | interaction_feat]  ← 結合
         ↓ Linear → LayerNorm → ReLU
         ↓ MLP  512 → 256
         ↓ Linear → num_actions ロジット
```

デフォルト行動クラス: `idle`（静止）, `walk`（歩行）, `groom`（グルーミング）, `interact`（インタラクション）, `other`（その他）

インタラクション特徴量（オプション）: 最近傍距離、相対角度、相対速度、ペアIoU

---

## インストール

### 1. 仮想環境の作成

```bash
conda create -n yoake python=3.10 -y
conda activate yoake
```

### 2. PyTorchのインストール

```bash
# CUDAバージョンに合わせて変更してください（cu118 / cu121 / cpu）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. YOAKEのインストール

```bash
cd /path/to/YOAKE

# フルインストール（可視化ツール + W&Bログ含む）
pip install -e ".[all]"

# 最小インストール
pip install -e .
```

### 4. インストール確認

```bash
python examples/smoke_test.py
# 期待される出力: "Smoke test passed."
```

スモークテストはGPU不要で、合成データによるフォワードパスと2エポックのミニ学習を実行します。

---

## データセット形式

YOAKEは独自の **JSON アノテーション形式 v1.1** を使用します。

### ファイル構造

```json
{
  "meta": {
    "version": "1.1",
    "fps_default": 25.0,
    "image_root": "images/"
  },
  "class_names": ["fly"],
  "action_names": ["idle", "walk", "groom", "interact", "other"],
  "videos": [
    {
      "video_id": "vid_001",
      "fps": 25.0,
      "width": 1024,
      "height": 1024,
      "num_frames": 300,
      "frames": [
        {
          "frame_index": 0,
          "image_path": "images/vid_001/000000.jpg",
          "objects": [
            {
              "object_id": 1,
              "bbox": [112.0, 87.5, 132.0, 109.0],
              "class_id": 0,
              "track_id": 1,
              "action_id": 1,
              "occluded": false
            }
          ]
        }
      ]
    }
  ]
}
```

### 座標系

- 形式: ピクセル座標での `[x1, y1, x2, y2]`（xyxy形式）
- 原点: 左上 `(0, 0)`
- 制約: `0 ≤ x1 < x2 ≤ width` かつ `0 ≤ y1 < y2 ≤ height`
- 損失計算時に正規化された `cxcywh` 形式に変換されます

### 特殊値

| 値 | 意味 |
|----|------|
| `track_id: -1` | 追跡対象外の個体 |
| `action_id: -1` | 行動未アノテーション（損失計算から除外） |
| `occluded: true` | 部分的に隠れている個体 |
| `is_crowd: true` | 重複グループ（メトリクスから除外） |

### データ準備ワークフロー

```bash
# 1. 動画からフレームを抽出
ffmpeg -i raw_videos/vid_001.mp4 -q:v 2 data/images/vid_001/%06d.jpg

# 2. CVAT・VGGなどのツールでアノテーション
#    → バウンディングボックス、トラックID、行動ラベルを付与

# 3. CSVアノテーションをYOAKE JSON形式に変換（必要な場合）
python tools/convert_annotations.py \
    input=data/raw/annotations.csv \
    output=data/raw/annotations.json \
    fps=25.0 width=1024 height=1024

# 4. アノテーションの検証
python -c "
from htrtdetr.data.annotation import load_annotation, validate_annotation
anno = load_annotation('data/raw/annotations.json')
print(validate_annotation(anno))
"

# 5. 動画単位で train/val/test に分割（時系列リークを防ぐため）
#    → data/train/annotations.json  (~70%)
#    → data/val/annotations.json    (~15%)
#    → data/test/annotations.json   (~15%)
```

### アノテーション品質基準

| 要件 | 推奨 | 最低限 |
|------|------|--------|
| アノテーション密度 | 全フレーム100% | 80%以上 |
| トラック最小長 | — | 16フレーム（時間ウィンドウ1枚分） |
| バウンディングボックスの精度 | 体輪郭に密着 | — |
| 動画メタデータ（fps・幅・高さ） | 正確に設定必須 | — |
| 画像形式 | JPEG 品質90以上 | — |

---

## 学習

学習は4段階で進めます。Stage 2 と Stage 3 は互いに独立しており、別々のGPUで並列実行できます。

### クイックスタート — `yoake` CLI（推奨）

`pip install -e .` インストール後、`yoake` コマンド一本で実行できます：

```bash
# Stage 1: 空間検出器
yoake train stage=1 data.train_root=data/train data.val_root=data/val

# Stage 2: 行動ヘッド（Stage 1 のチェックポイントを自動ロード）
yoake train stage=2

# Stage 3: IDヘッド（Stage 2 と別GPUで並列実行可能）
yoake train stage=3

# Stage 4: 統合微調整
yoake train stage=4
```

出力は `runs/train/stage{N}/` に保存されます。

`key=value` でパラメータを自由に上書きできます：

```bash
yoake train stage=1 train.max_epochs=100 data.batch_size=8 model.variant=small
```

### Stage 1: 空間検出器

**目的:** 単一フレームで動物を検出・局在化する能力を学習する。
**推奨エポック数:** 50〜500 | **1エポック所要時間:** 約8分（RTX 8000）

```bash
yoake train stage=1 \
    data.train_root=data/train \
    data.val_root=data/val \
    train.max_epochs=100 \
    data.batch_size=4
```

### Stage 2: 行動ヘッド

**目的:** 時間的クエリ特徴量から行動を分類する能力を学習する。
**推奨エポック数:** 20〜80 | **1エポック所要時間:** 約3分

```bash
yoake train stage=2 data.train_root=data/train data.val_root=data/val
```

### Stage 3: IDヘッド

**目的:** 個体再識別のための持続的ID埋め込みを学習する。
**推奨エポック数:** 20〜80 | **1エポック所要時間:** 約3分
*（Stage 2 と別のGPUで並列実行可能）*

```bash
yoake train stage=3
```

### Stage 4: 統合微調整

**目的:** 全モジュールを複合損失で共同微調整する。
**推奨エポック数:** 20〜50 | **1エポック所要時間:** 約10分

```bash
yoake train stage=4
```

### ベースディレクトリ（`root=`）

スクリプト・CLIとも `root=<path>` でワーキングルートを指定できます。デフォルトはリポジトリルート（自動検出）です。前ステージのチェックポイントは `{root}/runs/train/` から自動探索されます。

```bash
yoake train stage=2 root=/data/myproject
```

### 旧スクリプトインターフェース（非推奨）

従来のスクリプトも引き続き動作しますが非推奨です：

```bash
python scripts/train_stage1.py data.train_root=data/train
python scripts/train_stage2.py
python scripts/train_stage3.py
python scripts/train_stage4.py
```

> ⚠️ `scripts/train_stage1_detector.py` 等 `_detector/_action/_id/_unified` の付くスクリプトは旧 `stage_trainers.py` 系統です。上記 CLI またはスクリプトを推奨します。

### 主要ハイパーパラメータ

| カテゴリ | パラメータ | デフォルト | 説明 |
|---------|-----------|-----------|------|
| オプティマイザ | `optimizer.lr` | `1e-4` | ベース学習率（Stage 4 では `1e-5` を使用） |
| | `backbone_lr_factor` | `0.1` | バックボーン学習率の倍率（Stage 2 では `0.0` に設定してゼロLRで検出器を完全フリーズ） |
| | `grad_clip_norm` | `0.1` | 勾配クリッピング |
| スケジューラ | `warmup_epochs` | `5` | コサインウォームアップエポック数 |
| データ | `batch_size` | `4` | GPU当たりバッチサイズ |
| | `window_size` | `16` | 時間ウィンドウサイズ（フレーム数） |
| | `window_stride` | `8` | スライディングウィンドウのストライド |
| 損失重み | `w_bbox_l1` | `5.0` | L1 bbox 損失の重み |
| | `w_bbox_giou` | `2.0` | GIoU bbox 損失の重み |
| | `w_action` | `1.0` | 行動分類損失の重み |
| | `w_id_cls` | `1.0` | ID分類損失の重み |
| | `focal_gamma` | `2.0` | Focal損失のガンマ値 |
| | `focal_alpha` | `0.25` | Focal損失のアルファ値 |
| | `w_id_metric` | `0.5` | 計量学習（トリプレット）損失の重み |
| | `w_temporal_smooth` | `0.1` | 時間的平滑化正則化の重み |

### 推定学習時間の目安

| Stage | エポック数 | 所要時間 |
|-------|----------|--------|
| Stage 1 | 100 | 約13時間 |
| Stage 2 + 3（並列） | 30 | 約1.5時間 |
| Stage 4 | 30 | 約5時間 |
| **合計** | — | **約20時間** |

*RTX 8000、`batch_size=4`、`num_workers=8` での計測値。*

---

## 評価

### `yoake val` CLI（推奨）

```bash
yoake val stage=1
yoake val stage=2
yoake val stage=3
yoake val stage=4
```

`checkpoint=` を省略すると `runs/train/stage{N}/weights/best.pth` が自動で探索されます。

### ステージ別プライマリ指標

| Stage | プライマリ指標 | 方向 | 説明 |
|-------|-------------|------|------|
| Stage 1 | AP50 | 高いほど良い | IoU=0.5 の COCO スタイル検出 AP |
| Stage 2 | macro_F1 | 高いほど良い | クラス非加重平均 F1 |
| Stage 3 | IDF1 | 高いほど良い | Identity F1 (IDP・IDR の調和平均) |
| Stage 4 | composite | 高いほど良い | `0.4*AP50 + 0.3*macro_F1 + 0.3*IDF1` |

### Stage 1: 検出評価

```bash
yoake val stage=1 \
    data.val_root=data/val \
    checkpoint=runs/train/stage1/weights/best.pth
```

出力: `runs/val/stage1/val_results.json`、`metrics_latest.json`

メトリクス: **AP50**、**AP75**、class-aware の集約 AP / recall です。現状はクラス別 AP 表の出力ではありません。

### Stage 2: 行動分類評価

```bash
yoake val stage=2 \
    data.val_root=data/val \
    checkpoint=runs/train/stage2/weights/best.pth
```

出力: `runs/val/stage2/val_results.json`、`confusion_matrix.png`、`per_class_metrics.json`、`per_class_metrics.csv`

メトリクス: **Macro F1**、Weighted F1、クラス別 F1 / Precision / Recall、混同行列

### Stage 3: ID追跡評価

```bash
yoake val stage=3 \
    data.val_root=data/val \
    checkpoint=runs/train/stage3/weights/best.pth
```

メトリクス: **IDF1**、**IDP**、**IDR**、ID スイッチ数（IDSW）です。Stage 3 / 4 では同じ lightweight validation proxy で一貫して算出されます。

### 統合評価（Stage 4）

```bash
yoake val stage=4 \
    data.val_root=data/test \
    checkpoint=runs/train/stage4/weights/best.pth
```

出力: `runs/val/stage4/val_results.json`、`confusion_matrix.png`、`per_class_metrics.json`

メトリクス: **composite**（AP50 + macro_F1 + IDF1 の加重和）、各ステージの個別指標も含む。

---

## 推論

```bash
yoake predict \
    source=path/to/video.mp4 \
    weights=runs/train/stage4/weights/best.pth \
    score_threshold=0.3
```

出力は `runs/predict/` に保存されます。`output_dir=` で変更できます。

```bash
# ディレクトリ内の全動画を処理
yoake predict \
    source=data/videos/ \
    weights=runs/train/stage4/weights/best.pth \
    output_dir=runs/predict/myexp
```

### 推論出力ファイル

| ファイル | 説明 |
|---------|------|
| `<stem>_pred.mp4` | bbox・トラックID・行動ラベルを重畳した動画 |
| `<stem>_pred.json` | フレームごとの検出結果JSON（[出力形式](#出力形式)参照） |

---

## 設定リファレンス

すべての設定は型付きデータクラスで管理され、コマンドライン引数 `key=value` またはYAMLファイルで上書きできます。

### モデル設定

```yaml
model:
  detector:
    backbone:
      name: resnet18          # resnet18 | resnet34 | resnet50
      pretrained: true
    num_queries: 100
    num_decoder_layers: 4
    num_heads: 8
    score_threshold: 0.3
  temporal:
    short_branch:
      num_frames: 3
    mid_branch:
      num_frames: 8
    long_branch:
      num_frames: 16
    fusion_method: concat_proj   # concat_proj | sum | attention
  id_head:
    embedding_dim: 128
    memory_dim: 256
    max_ids: 50
    new_id_threshold: 0.5
    memory_ttl: 30
  action_head:
    num_actions: 5
    hidden_dims: [512, 256]
    dropout: 0.1
    use_interaction: true
```

### モデルサイズバリアント

`build_model_config()` / `get_variant_config()` で3種類のモデルサイズバリアントが利用できます:

| バリアント | バックボーン | hidden_dim | パラメータ数目安 |
|-----------|------------|------------|----------------|
| `small` | ResNet-18 | 128 | ~15–20M |
| `medium`（デフォルト） | ResNet-34 | 256 | ~30–40M |
| `large` | ResNet-50 | 512 | ~55–70M |

Python API:

```python
from htrtdetr.config.config import get_variant_config
cfg = get_variant_config("small", stage=1)
cfg = get_variant_config("large", stage=4, overrides={"data.batch_size": 2})
```

### データ設定

```yaml
data:
  batch_size: 4
  image_size: [640, 640]
  window_size: 16
  window_stride: 8
  num_workers: 8
  augment_train: true
```

### 学習設定

```yaml
train:
  stage: 1                       # 1 | 2 | 3 | 4
  max_epochs: 100
  early_stopping_patience: 20
  use_amp: true                  # 自動混合精度
  use_wandb: false               # W&B 実験追跡
```

### クラス不均衡対策（Stage 2 / 4）

```yaml
loss:
  imbalance_strategy: class_weight   # none | class_weight | focal
  # class_weight: 最大200バッチをスキャンし、ラプラス平滑化した逆頻度重みを計算
  #               ActionLoss.set_class_weights() に自動で渡される
  # focal:        Focal損失（gamma=2.0）を使用、クラスごとの重みなし
  # none:         通常のクロスエントロピー
```

CLI での指定:

```bash
yoake train stage=2 loss.imbalance_strategy=class_weight
```

> **注意:** `imbalance_strategy=sampler` は未実装です。`class_weight` にフォールバックし、警告を出力します。

計算されたクラス重みとクラスごとのサンプル数は、最初のスキャン後に `args.yaml` に保存されます。

### メトリクス設定

```yaml
metrics:
  stage2_primary: macro_f1        # macro_f1 | val_loss
  stage3_primary: idf1            # idf1 | val_loss
  stage4_primary: composite       # composite | ap50 | val_loss
  composite_ap50: 0.4             # Stage 4 複合スコアにおける AP50 の重み
  composite_macro_f1: 0.3         # macro_F1 の重み
  composite_idf1: 0.3             # IDF1 の重み
```

CLI での指定:

```bash
yoake train stage=4 metrics.stage4_primary=composite
yoake train stage=4 metrics.composite_ap50=0.5 metrics.composite_macro_f1=0.25 metrics.composite_idf1=0.25
```

---

## 出力形式

### 実行ディレクトリ構成

```
runs/
  train/stage{N}/
    weights/
      best.pth            # プライマリ指標が最良のチェックポイント（EMA重みを優先）
      last.pth            # 最終エポックのチェックポイント
    args.yaml             # 実行時設定の完全スナップショット（class_counts・class_weights含む）
    results.csv           # エポック別メトリクス（全ステージ共通列）
    val/
      epoch_NNN.json      # エポック別詳細評価結果
    plots/
      loss_history.png    # Train/val ロス推移グラフ
      f1_history.png      # macro_F1 推移（Stage 2/4）
      idf1_history.png    # IDF1 推移（Stage 3/4）
      composite_history.png  # 複合スコア推移（Stage 4）
      confusion_matrix.png   # 混同行列 PNG（Stage 2/4、matplotlib が必要）
      confusion_matrix.json  # 混同行列 JSON（常に保存）
    metrics_latest.json   # 最新バリデーション指標 + ベスト追跡情報
    train.log
  val/stage{N}/
    val_results.json
    confusion_matrix.png  # Stage 2/4
    per_class_metrics.json
    per_class_metrics.csv
    metrics_latest.json
  predict/                # 推論結果
  analyze/                # 解析結果
```

### 推論結果 JSON

```json
{
  "metadata": {
    "model_version": "0.1.0",
    "fps": 25.0,
    "width": 1024,
    "height": 1024,
    "num_frames": 300
  },
  "frames": [
    {
      "frame_idx": 0,
      "detections": [
        {
          "bbox": [112.3, 87.1, 132.7, 109.4],
          "score": 0.92,
          "class_id": 0,
          "track_id": 1,
          "id_confidence": 0.87,
          "action_id": 1,
          "action_confidence": 0.78
        }
      ]
    }
  ]
}
```

### 統合評価結果 JSON

```json
{
  "detection": {
    "ap50": 0.85,
    "ap75": 0.72,
    "per_class": { "fly": { "ap50": 0.85, "ap75": 0.72 } }
  },
  "action": {
    "macro_f1": 0.78,
    "per_class": {
      "idle":  { "f1": 0.82, "precision": 0.80, "recall": 0.84 },
      "walk":  { "f1": 0.75, "precision": 0.74, "recall": 0.76 },
      "groom": { "f1": 0.79, "precision": 0.81, "recall": 0.77 },
      "interact": { "f1": 0.76, "precision": 0.77, "recall": 0.75 },
      "other": { "f1": 0.71, "precision": 0.70, "recall": 0.72 }
    }
  },
  "tracking": {
    "idf1": 0.88,
    "id_switches": 42,
    "mota": 0.79
  }
}
```

---

*YOAKE Manual — last updated 2026-03-28*
