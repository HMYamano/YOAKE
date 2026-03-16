# YOAKE Architecture

## Overview

**YOAKE (Hierarchical Temporal RT-DETR)** は、ショウジョウバエなどの小動物を対象とした
リアルタイム行動解析アルゴリズムです。
1回のオンライン推論で各個体の **bbox / ID / action** を同時出力します。

---

## Architecture Diagram

```
Input: T frames (B, T, 3, H, W)
                │
    ┌───────────▼──────────────┐
    │   Spatial Detector       │  ← ResNet-18 + FPN + AIFI + DETR Decoder
    │   (frame-wise)           │
    └─────┬──────────┬─────────┘
          │          │
    (B,T,Q,4)   (B,T,Q,D)
    pred_boxes  query_features
                     │
    ┌────────────────▼─────────────────┐
    │  Hierarchical Temporal Module    │
    │  ┌──────────┐┌────────┐┌───────┐│
    │  │ Short    ││  Mid   ││ Long  ││
    │  │ d=1, K=3 ││d=2, K=8││d=4,K=16
    │  └────┬─────┘└───┬────┘└──┬────┘│
    │       └──────────┼────────┘     │
    │            [concat + proj]       │
    └───────────────────┬──────────────┘
                        │ (B,T,Q,out_dim)
         ┌──────────────┴──────────────┐
         │      Feature Router         │
         └──────┬────────────┬─────────┘
                │            │
    ┌───────────▼───┐  ┌─────▼────────────────┐
    │ Memory-based  │  │     Action Head       │
    │   ID Head     │  │  spatial + temporal   │
    │  GRU memory   │  │  + interaction feats  │
    │  cosine match │  │  → MLP → action_logit │
    └───────┬───────┘  └──────────┬────────────┘
            │                     │
       track_id               action_label
       id_score               action_conf

Output per detection:
  [bbox, det_score, class_id, track_id, id_conf, action_label, action_conf]
```

---

## Modules

### 1. Spatial Detector

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-18 (torchvision, ImageNet pretrained) |
| Neck | FPN: C3+C4+C5 → P3+P4+P5 (256ch) |
| Encoder | AIFI: P5 に self-attention で特徴強化 |
| Decoder | DETR style: 4-layer transformer decoder |
| Queries | 100 learnable queries |
| Output | bbox (cxcywh, normalized), class logits, query features |

### 2. Hierarchical Temporal Module (HTM)

| Branch | Dilation | Frames | Purpose |
|--------|----------|--------|---------|
| Short | 1 | 3 | 直近のフレームの細かい動き |
| Mid | 2 | 8 | 中期的な行動パターン |
| Long | 4 | 16 | 長期的な行動・移動傾向 |

- Depthwise separable 1D temporal conv
- Residual connection
- Branch fusion: concat + linear projection

### 3. Memory-based ID Head

```
Detection feature (N, D)
       │
  [GRU Memory Update]  ← past hidden state (N, memory_dim)
       │
  ID Embedding (N, emb_dim)  ← L2 normalized
       │
  Cosine similarity with memory embeddings
       │
  Hungarian matching
       │
  track_id assignment
```

- Memory: GRU hidden states (per individual)
- Matching: cosine similarity × 0.5 + IoU × 0.5
- New ID threshold: configurable
- Memory TTL: 30 frames (configurable)

### 4. Action Head

```
[spatial_feat | temporal_feat | interaction_feat]
       │ concat
  Linear → LayerNorm → ReLU
       │
  MLP (512 → 256)
       │
  Linear → num_actions logits
```

Interaction features (optional):
- Nearest neighbor distance
- Relative angle
- Relative velocity
- Overlap (IoU)

---

## Staged Training

```
Stage 1: Detector pretraining
  → detector_best.pth
  Loss: class + bbox_L1 + GIoU

Stage 2: Action head pretraining
  → action_head_best.pth
  Freeze: detector
  Train: temporal + action_head
  Loss: action classification (focal)

Stage 3: ID head pretraining
  → id_head_best.pth
  Freeze: detector
  Train: temporal + id_head
  Loss: ID classification + optional triplet

Stage 4: Unified fine-tuning
  → full_model_best.pth
  Train: all modules (small lr)
  Loss: det + action + ID + temporal_smooth
```

---

## Input / Output Specification

### Input
- Video frames: (B, T, 3, H, W)
- Default image size: 640×640
- Window size: 16 frames
- Frame stride: 8

### Output (per frame, per detection)
| Field | Type | Description |
|-------|------|-------------|
| bbox | [x1,y1,x2,y2] | Bounding box (pixel coords) |
| detection_score | float | Detector confidence [0,1] |
| class_id | int | Animal class index |
| track_id | int | Individual identity |
| id_confidence | float | ID assignment confidence |
| action_id | int | Action class index |
| action_confidence | float | Action classification confidence |

---

## Configuration

設定は `HTRTDETRConfig` dataclass で管理。
YAML ファイルとの相互変換をサポート。

```python
from htrtdetr.config import HTRTDETRConfig

# デフォルト設定
cfg = HTRTDETRConfig()

# YAML から読み込み
cfg = HTRTDETRConfig.from_yaml("configs/default.yaml")

# 部分的に上書き
cfg = cfg.merge({"data": {"batch_size": 8}})
```

---

## Future Extensions

- **Optical Flow**: `fusion/fusion.py` の `MultiHeadFeatureRouter` に追加
- **Graph Module**: interaction feature を GNN で強化
- **Segment-level Action**: `action_head.py` の `temporal_aggregator` を実装
- **ONNX Export**: `torch.onnx.export()` で対応可能な設計
- **Multi-class / Multi-species**: `num_classes` を増やすだけで対応
