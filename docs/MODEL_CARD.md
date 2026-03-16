# Model Card: YOAKE (Hierarchical Temporal RT-DETR)

## Model Overview

| Field | Value |
|-------|-------|
| **Model name** | YOAKE-R18 |
| **Version** | v0.1.0 |
| **Task** | Multi-animal detection, identity tracking, and behavior classification |
| **Primary subject** | *Drosophila melanogaster* (fruit fly); adaptable to other small animals |
| **Input** | RGB video sequence — tensor shape `(B, T, 3, 640, 640)` |
| **Output** | Per-frame detections: `bbox`, `track_id`, `id_confidence`, `action_id`, `action_confidence`, `detection_score` |
| **License** | MIT |
| **Framework** | PyTorch >= 2.1.0 |

---

## Architecture Summary

YOAKE processes a sliding window of T=16 frames through four sequential stages:

**1. Spatial Detector (frame-wise)**

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-18, ImageNet pretrained (torchvision, BSD License) |
| Neck | Feature Pyramid Network (FPN): C3+C4+C5 → P3+P4+P5, 256 channels each |
| Encoder | AIFI (Attention-based Inter-scale Feature Interaction): self-attention on the P5 map |
| Decoder | DETR-style transformer decoder, 4 layers, 8 attention heads |
| Queries | 100 learnable object queries per frame |
| Output | Per-query bounding boxes (cxcywh, normalized), class logits, and 256-d query feature vectors |

**2. Hierarchical Temporal Module (HTM)**

The HTM enriches per-frame query features with multi-scale temporal context using three parallel 1D dilated convolution branches operating over the T-frame window:

| Branch | Dilation | Receptive Frames | Purpose |
|--------|----------|------------------|---------|
| Short | 1 | 3 | Fine-grained frame-to-frame movement |
| Mid | 2 | 8 | Medium-term behavioral patterns |
| Long | 4 | 16 | Long-range locomotion trends |

Each branch uses depthwise separable 1D temporal convolutions with residual connections. Branch outputs are fused via concatenation followed by a linear projection to 256 dimensions.

**3. Memory-based ID Head (GRU)**

```
Detection feature (N, 256)
        |
  [GRU Memory Update]  <- per-track hidden state (N, 256)
        |
  ID Embedding (N, 128)  <- L2-normalized
        |
  Cosine similarity against stored embeddings
        |
  Hungarian matching  (cost = 0.5 * cosine + 0.5 * IoU)
        |
  track_id assignment, id_confidence
```

New-ID threshold: 0.5 (configurable). Memory time-to-live: 30 frames (configurable). Maximum simultaneous identities: 50.

**4. Action Head (MLP)**

Concatenates spatial features, temporal features from HTM, and optional interaction features (nearest-neighbor distance, relative angle, relative velocity, pair IoU), then passes through:

```
Linear(feature_dim) -> LayerNorm -> ReLU
MLP: 512 -> 256
Linear -> 5 action logits
```

---

## Input / Output Specification

### Input

- Tensor shape: `(B, T, 3, H, W)` where B = batch size, T = 16 (window size), H = W = 640
- Pixel values: float32 in [0, 1], RGB channel order
- Each video is processed as a sliding window with stride 8 frames

### Output (per frame, per detection)

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `[x1, y1, x2, y2]` | Bounding box in pixel coordinates (xyxy format) |
| `detection_score` | float [0, 1] | Detector confidence from the DETR head |
| `class_id` | int | Animal class index (0 = fly) |
| `track_id` | int | Persistent individual identity across frames |
| `id_confidence` | float [0, 1] | Confidence of the ID assignment |
| `action_id` | int | Action class index (see table below) |
| `action_confidence` | float [0, 1] | Action classification confidence |

---

## Action Classes

| `action_id` | Name | Description |
|-------------|------|-------------|
| 0 | `stationary` / `idle` | Animal is not moving; body posture at rest |
| 1 | `walking` / `walk` | Directed locomotion across the arena floor |
| 2 | `grooming` / `groom` | Self-cleaning movements of legs, wings, or head |
| 3 | `courtship` / `court` | Courtship-specific behaviors (wing extension, following, singing) |
| 4 | `aggression` / `other` | Aggressive or unclassified interactions |

Action class names are configured via `action_names` in the YAML config and annotation JSON. The `-1` sentinel value in annotations indicates unannotated frames and is excluded from loss computation.

---

## Training Data

| Field | Detail |
|-------|--------|
| **Dataset** | Drosophila behavior videos (laboratory top-down arena view) |
| **Annotation** | Per-frame bounding boxes, track IDs, and action labels |
| **Format** | YOAKE JSON v1.1 (see `docs/DATASET_FORMAT.md`) |
| **Environment** | Controlled lighting, fixed top-down camera, circular or rectangular arena |
| **Resolution** | Frames resized to 640×640 for training |
| **Typical scale** | Individual fly occupies approximately 15–40 px in a 640-px image |

The model is trained in four sequential stages (see `docs/TRAINING_GUIDE.md`). Stages 2 and 3 operate on geometric features extracted from tracks rather than raw pixels, which reduces compute and data requirements for the action and ID heads.

---

## Evaluation Metrics

Results below are placeholder values intended to demonstrate the reporting format. Fill in actual numbers after running `scripts/eval_unified.py` on your held-out test split.

### Detection (validation set)

| Metric | Value |
|--------|-------|
| AP50 | — |
| AP75 | — |
| Recall @ IoU=0.5 | — |
| Center Error (px) | — |

### Tracking (validation set)

| Metric | Value |
|--------|-------|
| IDF1 | — |
| ID Switches | — |
| Fragments | — |
| MOTA | — |

### Action Classification (validation set)

| Metric | Value |
|--------|-------|
| Frame-wise Accuracy | — |
| Macro F1 | — |
| F1 — stationary | — |
| F1 — walking | — |
| F1 — grooming | — |
| F1 — courtship | — |
| F1 — aggression/other | — |

### Runtime (NVIDIA RTX 8000, mixed precision)

| Metric | Value |
|--------|-------|
| Inference FPS | — |
| Mean latency per window (ms) | — |
| GPU memory peak (MB) | — |

---

## Intended Use

### Primary use cases

- Academic research on *Drosophila melanogaster* social and individual behavior in laboratory settings
- Quantitative behavioral phenotyping in neuroscience and genetics studies
- Automated scoring of high-throughput behavioral assays (e.g., courtship assay, aggression assay)
- Adaptation to other small laboratory animal species (mice, *C. elegans*, zebrafish larvae) with retraining

### Suitable deployment context

The model is designed for offline or near-real-time analysis on a workstation with a GPU. It expects controlled laboratory conditions: uniform background, stable illumination, top-down or near-top-down camera angle, and animals of roughly known size.

---

## Limitations

- **Camera perspective**: Validated only on top-down arena recordings. Side-view or oblique angles will degrade detection quality.
- **Occlusion handling**: When two animals overlap substantially, the GRU memory may produce ID switches. Memory TTL of 30 frames limits re-identification after prolonged occlusion.
- **Action boundary ambiguity**: Behavior labels near class boundaries (e.g., slow walking vs. stationary) are inherently ambiguous. Accuracy drops near transitions.
- **Lighting sensitivity**: The ResNet-18 backbone is pretrained on ImageNet but not specifically robust to infrared or UV illumination used in some behavioral rigs. Fine-tuning under target lighting conditions is recommended.
- **Scale assumption**: The model is tuned for animals occupying 15–40 px in a 640-px image. Very small or very large subjects require adjusting anchor priors or input resolution.
- **Maximum individual count**: The ID head supports up to 50 simultaneous identities per video by default (`max_ids=50`). Exceeding this limit degrades tracking.

---

## Out-of-Scope Uses

- Real-time production systems without further profiling and optimization (ONNX export or TensorRT conversion recommended for latency-critical deployments)
- Non-laboratory, uncontrolled outdoor environments
- Human or large-animal tracking (architectural assumptions do not hold)
- Surveillance or any application requiring individual identification of humans

---

## Ethical Considerations

This model is intended exclusively for academic animal behavior research. All animal experiments using this tool should comply with institutional ethics requirements and applicable national regulations on animal research. The model does not collect, store, or transmit personal data.

---

## How to Use

```python
from htrtdetr.config import HTRTDETRConfig
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_checkpoint

cfg = HTRTDETRConfig.from_yaml("configs/default.yaml")
model = build_model(cfg.model)
load_checkpoint(model, "weights/full_model_best.pth")
model.eval()

# memory_list holds GRU hidden states across frames for one batch
memory_list = model.create_memory_list(batch_size=1, device="cuda")

# images: (1, 16, 3, 640, 640) float32 tensor
outputs = model(images, memory_list)
# outputs.det_results[0] contains boxes, scores, track_ids
# outputs.action_logits contains per-detection action logits
```

For video file inference, use `tools/infer_video.py` (see `docs/INFERENCE_GUIDE.md`).

---

## Citation

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

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| v0.1.0 | 2026-03 | Initial public release |
