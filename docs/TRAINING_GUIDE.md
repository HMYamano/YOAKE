# Training Guide — YOAKE

This guide covers environment setup, data preparation, and the four-stage training procedure for YOAKE. Each stage progressively increases model complexity, improving training stability compared to end-to-end training from scratch.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Data Preparation Pipeline](#3-data-preparation-pipeline)
4. [Stage-by-Stage Training](#4-stage-by-stage-training)
   - [Stage 1: Spatial Detector](#stage-1-spatial-detector)
   - [Stage 2: Action Head](#stage-2-action-head)
   - [Stage 3: ID Head](#stage-3-id-head)
   - [Stage 4: Unified Fine-Tuning](#stage-4-unified-fine-tuning)
5. [Key Hyperparameters](#5-key-hyperparameters)
6. [Resuming from Checkpoints](#6-resuming-from-checkpoints)
7. [Ablation Studies](#7-ablation-studies)
8. [Monitoring Training Outputs](#8-monitoring-training-outputs)
9. [Expected Training Times](#9-expected-training-times)
10. [Common Issues and Fixes](#10-common-issues-and-fixes)

---

## 1. Prerequisites

### Hardware

- **GPU**: NVIDIA RTX 8000 (48 GB VRAM) or equivalent. The minimum recommended VRAM for default batch size is **16 GB**. An RTX 3090, A5000, or A6000 will work at `batch_size=4`. Reduce to `batch_size=2` on 12 GB cards.
- **CPU**: 8+ cores recommended for data loading (`num_workers=4` default).
- **Storage**: Allow approximately 50 GB for a medium-sized dataset (100 videos, 300 frames each) plus checkpoints.

### Software

- OS: Linux (recommended) or Windows 10/11 with WSL2 or native CUDA support
- CUDA >= 11.8
- cuDNN >= 8.6
- Python >= 3.10
- Conda (Miniconda or Anaconda)

---

## 2. Installation

### Create and activate the Conda environment

```bash
conda create -n yoake python=3.10 -y
conda activate yoake
```

### Install PyTorch with CUDA support

Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the exact command matching your CUDA version. Example for CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Install the package and all dependencies

```bash
cd /path/to/YOAKE
pip install -e ".[all]"
```

The `[all]` extra includes `matplotlib`, `opencv-python`, and `wandb`. For a minimal install (no visualization):

```bash
pip install -e .
```

### Verify the installation

```bash
python examples/smoke_test.py
```

The smoke test runs a forward pass and a short training loop on synthetic data with no GPU required. A passing run prints `Smoke test passed.` with no errors.

---

## 3. Data Preparation Pipeline

### 3.1 Organize raw data

Place source videos in `data/raw/videos/` and extracted frame images in `data/raw/images/<video_id>/`. If your frames are already extracted, skip to 3.2.

```
data/
  raw/
    videos/
      vid_001.mp4
      vid_002.mp4
    images/
      vid_001/
        000000.jpg
        000001.jpg
        ...
```

### 3.2 Convert annotations to YOAKE JSON format

If your annotations are in CSV format, convert them using `tools/convert_annotations.py` (when available):

```bash
python tools/convert_annotations.py \
    input=data/raw/annotations.csv \
    output=data/raw/annotations.json \
    image_root=data/raw/images \
    fps=25.0 \
    width=1024 \
    height=1024
```

If you are creating annotations manually, follow the JSON v1.1 specification in `docs/DATASET_FORMAT.md`.

Validate the result:

```python
from htrtdetr.data.annotation import load_annotation, validate_annotation, print_annotation_stats

anno = load_annotation("data/raw/annotations.json")
result = validate_annotation(anno)
print(result)
print_annotation_stats(anno)
```

### 3.3 Split into train / val / test

Split at the video level to prevent temporal leakage. A helper script `tools/build_splits.py` (when available) automates this. Manual approach:

```python
from htrtdetr.data.annotation import load_annotation, save_annotation, DatasetAnno
import random

anno = load_annotation("data/raw/annotations.json")
videos = anno.videos[:]
random.seed(42)
random.shuffle(videos)

n = len(videos)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

def subset(videos_list):
    return DatasetAnno(
        meta=anno.meta,
        class_names=anno.class_names,
        action_names=anno.action_names,
        videos=videos_list,
    )

save_annotation(subset(videos[:n_train]),         "data/train/annotations.json")
save_annotation(subset(videos[n_train:n_train+n_val]), "data/val/annotations.json")
save_annotation(subset(videos[n_train+n_val:]),    "data/test/annotations.json")
```

### 3.4 Summarize and verify splits

```bash
python tools/analyze.py mode=distribution \
    annotation=data/train/annotations.json \
    output=outputs/analysis/train

python tools/analyze.py mode=distribution \
    annotation=data/val/annotations.json \
    output=outputs/analysis/val
```

Inspect `outputs/analysis/train/action_distribution.png` to check class balance before training.

---

## 4. Stage-by-Stage Training

### Stage 1: Spatial Detector

**Goal**: Train the ResNet-18 + FPN + AIFI + DETR decoder to detect animals and produce high-quality query features. No temporal or ID components are active.

**Frozen modules**: None
**Active modules**: ResNet-18 backbone, FPN, AIFI encoder, DETR decoder
**Loss**: Classification loss (focal) + bounding box L1 loss + GIoU loss
**Checkpoint produced**: `outputs/stage1/checkpoint_best.pth`

```bash
python scripts/train_stage1_detector.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    output_dir=outputs/stage1 \
    num_epochs=100 \
    batch_size=4
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `train_anno` | `data/sample/annotations_train.json` | Path to training annotation JSON |
| `val_anno` | `data/sample/annotations_val.json` | Path to validation annotation JSON |
| `output_dir` | `outputs/stage1` | Directory for checkpoints and logs |
| `num_epochs` | `50` | Number of training epochs |
| `batch_size` | `4` | Batch size (reduce if OOM) |
| `resume` | `null` | Path to a checkpoint to resume from |
| `seed` | `42` | Random seed |

Monitor validation AP50. Training typically converges within 50–100 epochs. The checkpoint is saved whenever validation AP50 improves.

Evaluate Stage 1:

```bash
python scripts/eval_stage1.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/eval_stage1
```

---

### Stage 2: Action Head

**Goal**: Train the Hierarchical Temporal Module (HTM) and the Action Head MLP using geometric (per-track) features, with the detector frozen.

**Frozen modules**: ResNet-18 backbone, FPN, AIFI, DETR decoder
**Active modules**: HTM (all three branches), Action Head MLP
**Loss**: Focal cross-entropy action classification loss
**Checkpoint produced**: `outputs/stage2/checkpoint_best.pth`

```bash
python scripts/train_stage2_action.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/stage2 \
    num_epochs=30
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `stage1_ckpt` | `null` | Checkpoint from Stage 1 (initializes detector weights) |
| `num_epochs` | `30` | Typically 20–40 epochs are sufficient |
| `batch_size` | `4` | Same as Stage 1 |

Because the detector is frozen, GPU memory is lower in this stage. Larger batches are feasible (`batch_size=8` on 16 GB).

Evaluate Stage 2:

```bash
python scripts/eval_stage2.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage2/checkpoint_best.pth \
    output_dir=outputs/eval_stage2
```

---

### Stage 3: ID Head

**Goal**: Train the Hierarchical Temporal Module (HTM) and the Memory-based ID Head (GRU) using geometric features, with the detector frozen. Track IDs within each video are remapped to contiguous local IDs [0..N-1] and treated as a classification problem.

**Frozen modules**: ResNet-18 backbone, FPN, AIFI, DETR decoder
**Active modules**: HTM (all three branches), GRU memory, ID embedding head
**Loss**: ID classification cross-entropy + optional triplet metric loss
**Checkpoint produced**: `outputs/stage3/checkpoint_best.pth`

```bash
python scripts/train_stage3_id.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/stage3 \
    num_epochs=30
```

Note: Stage 2 and Stage 3 are **independent** — both initialize from the Stage 1 detector checkpoint. They can be run in parallel if you have multiple GPUs.

Evaluate Stage 3:

```bash
python scripts/eval_stage3.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage3/checkpoint_best.pth \
    output_dir=outputs/eval_stage3
```

---

### Stage 4: Unified Fine-Tuning

**Goal**: Fine-tune all modules jointly using the combined detection + action + ID + temporal smoothness loss. Weights from Stages 1–3 are used to initialize each respective module.

**Frozen modules**: None (all unfrozen, but detector uses a reduced learning rate)
**Active modules**: All (backbone LR factor = 0.1 relative to head LR)
**Loss**: Detection loss + action loss + ID loss + temporal smoothness regularization
**Checkpoint produced**: `outputs/stage4/checkpoint_best.pth`

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

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `stage1_ckpt` | `null` | Path to Stage 1 checkpoint (detector weights) |
| `stage2_ckpt` | `null` | Path to Stage 2 checkpoint (HTM + action head weights) |
| `stage3_ckpt` | `null` | Path to Stage 3 checkpoint (HTM + ID head weights) |
| `num_epochs` | `30` | 20–40 epochs is typical for convergence |

Stage 4 is the most memory-intensive stage because all modules are active. If you encounter out-of-memory errors, reduce `batch_size` first.

Evaluate the unified model:

```bash
python scripts/eval_unified.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/stage4/checkpoint_best.pth \
    output_dir=outputs/eval_unified \
    score_threshold=0.3
```

This script reports AP50, AP75, IDF1, and per-class action F1 scores.

---

## 5. Key Hyperparameters

All hyperparameters are defined in `configs/default.yaml` and can be overridden with `key=value` arguments on the command line. The most impactful parameters for each stage are:

### Detector (Stage 1)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_epochs` | 100 | More epochs improve AP but risk overfitting on small datasets |
| `batch_size` | 4 | Larger batches stabilize gradient estimates |
| `optimizer.lr` | 1e-4 | Base learning rate for non-backbone parameters |
| `optimizer.backbone_lr_factor` | 0.1 | Backbone LR = `lr * factor`; lower values preserve ImageNet features |
| `scheduler.warmup_epochs` | 5 | Cosine warmup prevents early instability |
| `model.detector.score_threshold` | 0.3 | Detection confidence threshold for evaluation |
| `model.detector.head.num_queries` | 100 | Increase if >100 animals can appear simultaneously |
| `loss.w_bbox_l1` | 5.0 | Weight for L1 bounding box regression loss |
| `loss.w_bbox_giou` | 2.0 | Weight for GIoU bounding box regression loss |

### Action Head (Stage 2)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model.temporal.short_branch.num_frames` | 3 | Temporal context for short branch |
| `model.temporal.mid_branch.num_frames` | 8 | Temporal context for mid branch |
| `model.temporal.long_branch.num_frames` | 16 | Temporal context for long branch |
| `loss.w_action` | 1.0 | Weight for action classification loss |
| `loss.focal_gamma` | 2.0 | Focal loss gamma; increase to focus harder on difficult examples |
| `loss.focal_alpha` | 0.25 | Focal loss alpha; decrease for highly imbalanced classes |

### ID Head (Stage 3)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model.id_head.embedding_dim` | 128 | Dimensionality of L2-normalized ID embeddings |
| `model.id_head.memory_dim` | 256 | GRU hidden state dimension |
| `model.id_head.new_id_threshold` | 0.5 | Cosine similarity below this creates a new track |
| `model.id_head.memory_ttl` | 30 | Frames before an unseen track is discarded |
| `loss.w_id_cls` | 1.0 | Weight for ID classification loss |
| `loss.w_id_metric` | 0.5 | Weight for triplet metric loss (set to 0 to disable) |

### Unified Fine-Tuning (Stage 4)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `optimizer.lr` | 1e-4 | Use a lower LR (1e-5) if Stage 1–3 weights diverge |
| `loss.w_temporal_smooth` | 0.1 | Temporal smoothness regularization weight |
| `train.early_stopping_patience` | 20 | Stop if val loss does not improve for this many epochs |

---

## 6. Resuming from Checkpoints

Every stage trainer saves two checkpoints after each epoch that improves validation metrics:

- `<output_dir>/checkpoint_best.pth` — best validation metric so far
- `<output_dir>/checkpoint_last.pth` — most recent epoch (for resuming)

To resume an interrupted training run, pass the last checkpoint via `resume=`:

```bash
python scripts/train_stage1_detector.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    output_dir=outputs/stage1 \
    num_epochs=100 \
    resume=outputs/stage1/checkpoint_last.pth
```

The trainer restores the model weights, optimizer state, scheduler state, and epoch counter. Resuming is supported for all four stages.

---

## 7. Ablation Studies

Use `tools/analyze.py` in `ablation` mode to compare experiment results.

**Step 1**: Run each ablation variant with a distinct `output_dir`, e.g.:

```bash
# Baseline: all three temporal branches
python scripts/train_stage2_action.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/ablation/full_htm

# Ablation: short branch only
python scripts/train_stage2_action.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    stage1_ckpt=outputs/stage1/checkpoint_best.pth \
    output_dir=outputs/ablation/short_only \
    model.temporal.mid_branch.num_frames=3 \
    model.temporal.long_branch.num_frames=3
```

**Step 2**: Save evaluation JSON files to a shared ablation results directory:

```bash
python scripts/eval_stage2.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/ablation/full_htm/checkpoint_best.pth \
    output_dir=outputs/ablation/full_htm

python scripts/eval_stage2.py \
    anno=data/val/annotations.json \
    checkpoint=outputs/ablation/short_only/checkpoint_best.pth \
    output_dir=outputs/ablation/short_only
```

**Step 3**: Aggregate and compare:

```bash
python tools/analyze.py mode=ablation results_dir=outputs/ablation
```

This produces `outputs/ablation/ablation_comparison.json` and prints a side-by-side metric table.

---

## 8. Monitoring Training Outputs

Each stage trainer writes the following to its `output_dir`:

```
outputs/stage1/
  checkpoint_best.pth      # Best checkpoint (by validation metric)
  checkpoint_last.pth      # Most recent checkpoint
  train_log.csv            # Per-epoch: loss, lr, val metrics
  config.yaml              # Copy of the config used for this run
```

To plot training curves from the CSV log:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/stage1/train_log.csv")
plt.plot(df["epoch"], df["val_ap50"], label="AP50")
plt.xlabel("Epoch")
plt.ylabel("AP50")
plt.title("Stage 1 Validation AP50")
plt.legend()
plt.savefig("outputs/stage1/ap50_curve.png")
```

### W&B integration (optional)

To enable Weights & Biases logging, install `wandb` and set `train.use_wandb=true`:

```bash
pip install wandb
wandb login

python scripts/train_stage1_detector.py \
    train_anno=data/train/annotations.json \
    val_anno=data/val/annotations.json \
    train.use_wandb=true \
    train.wandb_project=ht-rtdetr \
    train.wandb_run_name=stage1_baseline
```

---

## 9. Expected Training Times

These estimates are for a dataset of approximately 80 training videos × 300 frames × 5 animals on a single NVIDIA RTX 8000 with `batch_size=4` and `num_workers=4`.

| Stage | Epochs | Approx. time per epoch | Total |
|-------|--------|------------------------|-------|
| Stage 1 (detector) | 100 | ~8 min | ~13 h |
| Stage 2 (action) | 30 | ~3 min | ~1.5 h |
| Stage 3 (ID) | 30 | ~3 min | ~1.5 h |
| Stage 4 (unified) | 30 | ~10 min | ~5 h |

Stages 2 and 3 can run in parallel on separate GPUs. Total wall-clock time for the full pipeline assuming sequential execution is approximately 21 hours. With two GPUs (parallel Stage 2 and 3), this reduces to approximately 18 hours.

Training times scale roughly linearly with dataset size. On smaller datasets (20 videos), epochs are faster but more epochs may be needed.

---

## 10. Common Issues and Fixes

### Out-of-memory (CUDA OOM) error

The most common cause is `batch_size` being too large for the available VRAM.

**Fix**: Reduce `batch_size` by half and retry:

```bash
python scripts/train_stage1_detector.py \
    ... \
    batch_size=2
```

If OOM persists at `batch_size=1`, check that no other processes are occupying the GPU (`nvidia-smi`). Also consider reducing `model.detector.head.num_queries` from 100 to 50 for a smaller model.

### Slow convergence / loss not decreasing

**Diagnosis**: Check `train_log.csv` for steeply decreasing loss in the first few epochs. If loss is flat from epoch 1, the learning rate is likely too low or the data loader is returning malformed batches.

**Fix 1 — Learning rate**: Try increasing `optimizer.lr` by 5–10x:

```bash
python scripts/train_stage1_detector.py ... optimizer.lr=5e-4
```

**Fix 2 — Data issues**: Validate your annotation JSON with `validate_annotation()`. Common issues: negative-area bounding boxes, missing frames, or all-zero image tensors.

**Fix 3 — Warmup**: Extend the learning rate warmup to stabilize early training:

```bash
python scripts/train_stage1_detector.py ... scheduler.warmup_epochs=10
```

### NaN loss during Stage 4

Stage 4 uses multiple loss terms simultaneously. NaN can arise from exploding gradients, particularly in the GRU memory head.

**Fix 1**: The default gradient clipping norm is `0.1`. If NaN appears, try reducing to `0.01`:

```bash
python scripts/train_stage4_unified.py ... optimizer.grad_clip_norm=0.01
```

**Fix 2**: Verify that Stage 1–3 checkpoints are valid and not corrupt:

```python
import torch
ckpt = torch.load("outputs/stage1/checkpoint_best.pth", map_location="cpu")
print(ckpt.keys())
print({k: v.shape for k, v in ckpt["model_state_dict"].items() if "backbone" in k})
```

### Action loss is constant (no learning in Stage 2)

This usually means that Stage 1 weights were not loaded correctly, so the detector produces random features that carry no useful signal for the action head.

**Fix**: Confirm that `stage1_ckpt` points to an existing, non-trivial checkpoint (AP50 > 0.3 on validation). Use `print(result)` from `validate_annotation()` to confirm data integrity.

### "checkpoint not found" warning

The scripts print a warning but continue with random weights when a checkpoint file does not exist.

**Fix**: Use absolute paths for checkpoint arguments to avoid working-directory confusion:

```bash
python scripts/train_stage2_action.py \
    ... \
    stage1_ckpt=/absolute/path/to/YOAKE/outputs/stage1/checkpoint_best.pth
```
