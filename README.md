![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)

# YOAKE: Hierarchical Temporal RT-DETR

YOAKE is a PyTorch model for multi-animal tracking and behavior classification in laboratory video recordings, with a primary focus on *Drosophila melanogaster*. It combines an RT-DETR spatial detector with a three-branch dilated temporal module (HTM), a GRU memory-based identity tracking head, and a behavior branch — producing bounding boxes, persistent track IDs, and per-frame behavior labels. The model is trained in four sequential stages that progressively unlock module complexity, ensuring stable convergence without requiring end-to-end training from scratch.

The architecture reflects one central idea: **individual detection and behavior recognition need different spatial granularity**. Individuals are detected with tight, orientation-sensitive *rotated* boxes (OBB), which separate elongated animals that lie close together. Behavior is recognized from a *wider, axis-aligned context box* around each individual, which keeps the neighbours and surroundings that make a behavior what it is. Time is aligned by **track ID**, never by detector query index.

---

## Architecture

```
Frame
  |
  v
Shared Backbone (ResNet) + FPN
  |   P2(stride 4, optional) / P3 / P4 / P5
  |
  +---------------------------+
  |                           |
  v                           v
Fine detection            Spatial features
RT-DETR-OBB                    |
  | cx, cy, w, h, theta        |
  v                            |
OBB detections                 |
  |                            |
  v                            |
OBB-aware tracking             |
  rotated IoU + appearance     |
  + motion + orientation       |
  |                            |
  v                            |
track_id                       |
  |                            |
  v                            |
Per-track ring buffer          |
  (no detector recompute)      |
  |                            |
  +-------------+--------------+----------------+
  |             |              |                |
  v             v              v                v
Actor OBB   Context AABB   Kinematics      Interaction
rotated     expanded       velocity        nn-distance
crop        ROIAlign       acceleration    approach rate
(orientation-             angular vel      facing A->B
 normalized)              orientation      facing B->A
  |             |              |                |
  +-------------+------+-------+----------------+
                       |
                       v
       Hierarchical Temporal Module (HTM)
       short (3f) / mid (8f) / long (16f)
       --- input is TRACK-aligned history ---
                       |
                       v
         Behavior Evidence Model
         logits = SUM_g  gate_g * head_g(feature_g)
           appearance + context + motion
           + pose + interaction + temporal
                       |
                       v
     single | hierarchical | multi-head output
                       |
        +--------------+--------------+
        |              |              |
   Prediction     Confidence     Explanation
                                 (per-group logit
                                  contributions)
```

Two planes coexist and remain independent:

- **Model plane** (`models/`) — the unified `HTRTDETR` network above, trained in 4 stages.
- **Orchestration plane** (`pipeline/` + `stages/`) — a capability-token DAG that lets you
  swap backends (`rtdetr` / `yolo` / `bytetrack` / `obb`) from YAML without touching code.

---

## Key Features

- **Keyframe labels, YOWO-style**: a human labels single frames - a box and a behavior for every animal, no track IDs, no intervals. Training cuts the clip that ends at each keyframe out of the video for free, identity comes from tracking at analysis time, and behavior bouts come from decoding per-frame probabilities along each track (`yoake analyze mode=bouts`). One behavior per animal by default; optionally behavior groups (multi-head: one answer per group, e.g. locomotion + social), each trained, scored and decoded on its own. Design and prior-work comparison: [docs/KEYFRAME_BEHAVIOR.md](docs/KEYFRAME_BEHAVIOR.md).
- **Rotated (oriented) box detection**: individuals are detected as OBBs (`cx, cy, w, h, theta`), which separate elongated animals lying close together far better than axis-aligned boxes. Angle is regressed as `sin(2θ), cos(2θ)` so the period matches the box's own period of π — the boundary discontinuity disappears by construction.
- **Track-aligned temporal modeling**: the HTM consumes per-**track** history, not per-**query** history. DETR query indices do not identify the same individual across frames, so query-aligned sequences silently mix different animals.
- **True streaming inference**: every frame passes the detector exactly **once**. Past frames are served from per-track ring buffers instead of being re-fed, so cost no longer scales with window size (**~4.5–6.2× faster**, detector calls 78 → 12 at window=16).
- **Two-stream behavior recognition**: a rotated, orientation-normalized *actor* crop for fine appearance and pose, plus a wider axis-aligned *context* crop for neighbours and surroundings.
- **Explainability built into the model**: behavior logits are an additive decomposition, `logits = Σ_g gate_g · head_g(feature_g)`, so per-group contributions *are* the prediction rather than a post-hoc approximation.
- **Hierarchical / multi-head behavior classification**: `single` (backward compatible), `hierarchical` (coarse → fine), or `multi_head` (locomotion / social / self-directed simultaneously).
- **OBB-aware tracking**: association mixes rotated IoU, appearance embedding, motion prediction, and orientation consistency — each weight independently ablatable.
- **P2 (stride 4) feature level**: optional high-resolution level for small and densely packed individuals, with the VRAM/latency trade-off exposed in config.
- **Hierarchical temporal reasoning**: three parallel dilated 1D convolution branches capture motion at 3, 8, and 16-frame scales without 3D convolutions
- **GRU memory tracking**: per-track hidden states enable persistent identity assignment across frames with configurable TTL
- **4-stage training**: each stage freezes previously trained modules to stabilize learning of new components
- **Config-first design**: all hyperparameters are managed by typed dataclasses and YAML files; no `argparse` in the main API
- **Research-ready ablation**: P2, OBB, actor, context, motion, orientation, interaction, pose, HTM, and evidence gating can each be disabled from config — and disabling one really removes its parameters
- **Mixed precision support**: native PyTorch AMP for 1.5–2x training speedup on Ampere+ GPUs
- **Three model size variants**: `small` (ResNet-18, ~15–20M), `medium` (ResNet-34, ~30–40M), `large` (ResNet-50, ~55–70M) via `get_variant_config()`
- **Backward compatible**: v1.1 annotations, existing configs, and old checkpoints all still work (see [docs/MIGRATION.md](docs/MIGRATION.md))

---

## Requirements

Python >= 3.10. Everything else is grouped so that a labelling machine does not
have to install the training stack:

| Group | Install | Contents |
|---|---|---|
| base | `pip install -e .` | numpy >= 1.24, Pillow >= 9.0, pyyaml >= 6.0, tqdm >= 4.65 |
| `gui` | `pip install -e ".[gui]"` | opencv-python >= 4.8, dearpygui >= 1.11 |
| `ml` | `pip install -e ".[ml]"` | torch >= 2.1, torchvision >= 0.16, scipy >= 1.10 (Hungarian matching) |
| `vis` | `pip install -e ".[vis]"` | matplotlib >= 3.7 |
| `wandb` | `pip install -e ".[wandb]"` | wandb >= 0.16 (experiment tracking) |
| `concept` | `pip install -e ".[concept]"` | transformers >= 4.40 ("Find all in frame") |
| `all` | `pip install -e ".[all]"` | gui + ml + vis + wandb + concept |

`sam2` is intentionally **not** part of `all`: it needs a matching PyTorch
already present and builds CUDA extensions, so bundling it would turn one bad
wheel into a failed install of everything. Install it on its own, into the
Python that runs `tools/sam_sidecar.py` — see [Video propagation](#gui-labeling--training--video-analysis).

> **Changed:** `pip install -e .` used to install PyTorch. It no longer does, so
> that `".[gui]"` can stay torch-free. For training or inference use `".[ml]"`
> or `".[all]"`.

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

For training and inference without the GUI or experiment tracking:

```bash
pip install -e ".[ml]"
```

For a labelling-only machine, with no PyTorch and no CUDA wheels:

```bash
pip install -e ".[gui]"
```

For development and tests (this installs `.[ml,gui]`, which the suite needs):

```bash
pip install -r requirements-dev.txt
pytest -q
```

If your shell cannot find `python` or `pytest`, use the repository CI workflow as the reference environment.

---

## Quick Start

### GUI (Labeling / Training / Video Analysis)

A DearPyGui hub (inspired by [YORU](https://github.com/Kamikouchi-lab/YORU)) bundles
video **labeling**, staged **training with a live metric monitor**, and **interactive
video analysis** (overlayed playback + per-track action timeline). It drives the
`yoake` CLI under the hood, so the GUI itself is torch-free — `".[gui]"` installs
no PyTorch, and the hub tells you what is locked and how to unlock it. Training
and automatic detection need `".[ml]"` as well; everything else (labelling,
propagation, project management, analysis playback) works without it.

```bash
pip install -e ".[gui]"     # dearpygui + opencv-python, no PyTorch
yoake gui                    # or:  python -m htrtdetr.gui  /  gui
```

All three commands start the same hub.

Boxes are drawn per frame, carried forward with `Space` (propagate) or the
auto-tracker, and labeled with track ID and action. Output stays in the standard
JSON v1.1 format. See [docs/gui.md](docs/gui.md).

#### Projects are folders

A study is one folder: annotations, the training data built from them, every run
and every analysis result live inside it, and every path it stores is relative —
so the folder can be moved, copied or handed to someone else and still work.

```
<project>/  project.yoake.json  annotations.json
            videos/  datasets/  runs/  analysis/
```

```bash
yoake project new dir="D:/studies" name="courtship 2026" videos=D:/captures
yoake project info path="D:/studies/courtship 2026"     # what resolves, what does not
annotate --project "D:/studies/courtship 2026"
```

Full description, including what happens when the videos move too:
[docs/PROJECT_FOLDER.md](docs/PROJECT_FOLDER.md).

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

### Layered pipeline (detection → tracking → interaction)

The single-pass model can also be driven through a declarative, stage-based pipeline that
stacks temporal analysis (L1→L5) on top of detection. Each layer is toggled/swapped from a
config file — no code branching. See [docs/pipeline.md](docs/pipeline.md).

```bash
# ④ full analysis: detect → stabilize → static-relation → track → motion → interaction
yoake pipeline config=configs/pipeline/full_offline.yaml source=data/videos/experiment_01.mp4 \
      output_dir=runs/pipeline
```

Analysis modes are just config swaps: `detect_only.yaml` (①), `temporal_detect.yaml` (②),
`detect_track.yaml` (③), `full_offline.yaml` (④), `yolo_bytetrack.yaml` (backend swap example).

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

YOAKE uses a custom JSON annotation format described in full in [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md).

Schema **v2.0** adds optional `obb` (rotated box) and `keypoints` fields. It is fully
backward compatible: v1.1 files load unchanged, and a dataset without any OBB is saved
byte-identically to v1.1 so older tools can still read it. See
[docs/MIGRATION.md](docs/MIGRATION.md) for the (optional) migration step.

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

`yoake train` is the supported entry point for every stage. Full details,
hyperparameter explanations, and troubleshooting are in
[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

```bash
yoake train stage=1 config=configs/stage1_small.yaml
yoake train stage=2 data.train_root=data/train data.val_root=data/val
```

> The `scripts/train_stage*_{detector,action,id,unified}.py` scripts drive the
> legacy `stage_trainers.py` backend. They still work and are still tested, but
> they emit a `DeprecationWarning` and will be removed. Prefer `yoake train`.

### Stage 1 — Spatial Detector

Trains ResNet-18 + FPN + AIFI + DETR decoder to detect animals. All other modules are inactive.

```bash
yoake train stage=1 \
    data.train_root=data/train/annotations.json \
    data.val_root=data/val/annotations.json \
    train.output_dir=outputs/stage1 \
    train.max_epochs=100 \
    data.batch_size=4
```

### Stage 2 — Action Head

Freezes the detector. Trains the HTM and MLP action head using geometric track features.

```bash
yoake train stage=2 \
    data.train_root=data/train/annotations.json \
    data.val_root=data/val/annotations.json \
    train.output_dir=outputs/stage2 \
    train.max_epochs=30
```

### Stage 3 — ID Head

Freezes the detector. Trains the HTM and GRU memory ID head. Runs in parallel with Stage 2.

Identity classes are keyed by `(video_id, track_id)` across the whole dataset, so
track 0 of one video and track 0 of another are never learned as the same animal.
Set `model.id_head.max_ids` at or above the dataset's identity count — the
trainer prints a warning if it is too low, because the surplus would otherwise
collapse into a single "unknown" class.

```bash
yoake train stage=3 \
    data.train_root=data/train/annotations.json \
    data.val_root=data/val/annotations.json \
    train.output_dir=outputs/stage3 \
    train.max_epochs=30
```

### Stage 4 — Unified Fine-Tuning

Unlocks all modules. Initializes from Stages 1–3. Uses combined detection + action + ID + temporal smoothness loss.

```bash
yoake train stage=4 \
    data.train_root=data/train/annotations.json \
    data.val_root=data/val/annotations.json \
    train.output_dir=outputs/stage4 \
    train.max_epochs=30
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

### Streaming vs batch

Inference runs in **streaming** mode by default: each frame passes the detector exactly
once, and past frames are served from per-track ring buffers. The old behaviour — re-feeding
the whole sliding window every frame — is still available as `inference.mode=batch` for
numerical comparison.

```bash
# streaming (default)
yoake predict source=video.mp4 weights=best.pth

# batch (legacy, for A/B comparison)
yoake predict source=video.mp4 weights=best.pth inference.mode=batch
```

**Detections (bbox / score) are identical between the two modes** — streaming only skips
recomputation, never changes the per-frame detector result. Track IDs can differ, because
batch rebuilds the tracker per window while streaming keeps one tracker for the whole video.

Measure the difference on your own hardware:

```bash
python tools/benchmark.py mode=streaming windows=4,8,16 num_frames=24
```

```
window=  4: batch 120.20 ms/f | streaming 26.85 ms/f | speedup 4.48x | detector calls 42 -> 12
window=  8: batch 148.38 ms/f | streaming 25.24 ms/f | speedup 5.88x | detector calls 68 -> 12
window= 16: batch 173.23 ms/f | streaming 27.98 ms/f | speedup 6.19x | detector calls 78 -> 12
```

### Explaining a prediction

```python
out = model(images)
out.explain_detection(batch_index=0, det_index=2)
# {'actor': 0.32, 'context': 0.28, 'motion': 0.54, 'interaction': 1.10, 'temporal': 0.41}
# these sum to the action logit — the explanation IS the prediction, not an approximation
out.behavior_gates(batch_index=0, det_index=2)
# {'actor': 0.71, 'motion': 0.83, ...}   learned per-group importance
```

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

# Behavior bouts per track (keyframe-trained models): Viterbi decoding along each track,
# time budgets, latencies, transitions, identity-free population counts
yoake analyze mode=bouts predictions=outputs/inference/predictions.json output_dir=outputs/bouts

# Detection confidence histogram
python tools/analyze.py mode=confidence \
    predictions=outputs/inference/experiment_01_pred.json \
    output=outputs/analysis/experiment_01
```

---

## Ablation Studies

Every new component can be switched off from config, and switching one off genuinely
removes its parameters (not just zeroes its input) so comparisons stay fair. The full
list and the rationale for each is in [docs/OBB_PIPELINE.md](docs/OBB_PIPELINE.md#7-ablation).

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
│       ├── geometry/            # OBB geometry: the single source of truth
│       │   └── obb.py           #   obb<->corners<->aabb, rotated IoU, ProbIoU
│       ├── data/
│       │   ├── annotation.py    # JSON v1.1/v2.0: load, save, validate, migrate
│       │   ├── fly_dataset.py   # FlyDataset, build_dataloaders
│       │   ├── collate.py       # batch collation
│       │   └── feature_builder.py
│       ├── tracking/            # OBB-aware tracking + track feature cache
│       │   ├── associator.py    #   rotated IoU + appearance + motion + orientation
│       │   ├── track_buffer.py  #   per-track ring buffers (streaming's backbone)
│       │   ├── tracker.py       #   OnlineTracker (association + re-ID)
│       │   └── assembly.py      #   track-aligned history construction
│       ├── models/
│       │   ├── detector/        # ResNet + FPN(+P2) + AIFI + DETR decoder (+OBB/pose)
│       │   ├── temporal/        # HierarchicalTemporalModule (HTM)
│       │   ├── id_head/         # MemoryIDHead (GRU)
│       │   ├── action_head/     # ActionHead (MLP, legacy-compatible)
│       │   ├── behavior/        # 2-stream behavior branch
│       │   │   ├── roi.py       #   actor rotated crop + context expanded crop
│       │   │   ├── kinematics.py#   motion / orientation / interaction features
│       │   │   ├── evidence_head.py # additive evidence decomposition + classifiers
│       │   │   └── branch.py    #   BehaviorBranch
│       │   ├── fusion/          # TrackTemporalFusion, MultiHeadFeatureRouter
│       │   └── ht_rtdetr.py     # Unified HTRTDETRModel
│       ├── training/
│       │   ├── stage_trainers.py  # Stage1–4 Trainer classes
│       │   ├── losses.py          # Detection/OBB/pose/action/ID losses
│       │   ├── matching.py        # Hungarian matcher (rotated-IoU aware)
│       │   └── optimizer.py       # AdamW + cosine scheduler
│       ├── evaluation/
│       │   ├── evaluator.py     # DetectionEvaluator, ActionEvaluator, TrackingEvaluator
│       │   └── advanced.py      # rotated AP, MOTA/IDF1/HOTA, multi-head, latency
│       ├── inference/
│       │   ├── inferencer.py    # Inferencer (batch mode, legacy)
│       │   └── streaming.py     # StreamingInferencer (default; 1 detector pass/frame)
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
│   ├── PROJECT_FOLDER.md        # The project folder layout and why it survives a move
│   ├── DATASET_FORMAT.md        # Annotation format specification
│   ├── MIGRATION.md             # v1.1->v2.0, config, checkpoint migration
│   ├── OBB_PIPELINE.md          # OBB / track-aligned / streaming design rationale
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
