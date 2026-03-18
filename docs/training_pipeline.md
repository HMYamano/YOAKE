# YOAKE 学習パイプライン完全ガイド

> **対象GPU**: NVIDIA RTX 8000（VRAM 48GB）
> **モデルバリアント**: large（ResNet-50 backbone, hidden_dim=512）
> **リポジトリ**: `c:/Users/hayam/YOAKE`
> **データ・出力先**: `c:/Users/hayam/Desktop/YOAKE_tryal`

---

## 1. 全体パイプライン図

```
データ
├─ 単一フレームアノテーション (train/val/annotations.json)
└─ シーケンスアノテーション (action/track ラベル付き)

┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Detector Pretraining                                          │
│  ┌─────────────┐  ┌─────────┐  ┌──────────────────┐                   │
│  │  ResNet-50  │→ │   FPN   │→ │ DETR Decoder ×6  │  ← 学習           │
│  │  backbone   │  │ out=512 │  │  hidden=512       │                   │
│  └─────────────┘  └─────────┘  └──────────────────┘                   │
│  [凍結なし] / 使用データ: 単一フレーム / Loss: cls + bbox_l1 + giou    │
│  出力: stage1_best.pth (detector 全重み)                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ 重みロード (detector)
              ┌──────────────────┴──────────────────┐
              │                                      │
              ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  Stage 2: Action Head   │          │  Stage 3: ID Head       │
│  ┌──────────────┐       │          │  ┌──────────────┐       │
│  │  Detector    │ ←凍結 │          │  │  Detector    │ ←凍結 │
│  └──────┬───────┘       │          │  └──────┬───────┘       │
│  ┌──────▼───────────┐   │          │  ┌──────▼───────────┐   │
│  │ Hierarchical     │ ←学習        │  │ Hierarchical     │ ←学習
│  │ Temporal Module  │   │          │  │ Temporal Module  │   │
│  │ (short/mid/long) │   │          │  │ (short/mid/long) │   │
│  └──────┬───────────┘   │          │  └──────┬───────────┘   │
│  ┌──────▼───────┐       │          │  ┌──────▼───────┐       │
│  │  Action Head │ ←学習 │          │  │  Memory ID   │ ←学習 │
│  │  + Interact  │       │          │  │  Head (GRU)  │       │
│  └──────────────┘       │          │  └──────────────┘       │
│  Loss: action_ce        │          │  Loss: id_cls + triplet  │
│  データ: シーケンス      │          │  データ: シーケンス      │
│  出力: stage2_best.pth  │          │  出力: stage3_best.pth  │
└─────────────────────────┘          └─────────────────────────┘
              │                                      │
              └──────────────────┬───────────────────┘
                                 │ 重みロード (stage2 or 3)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 4: Unified Fine-tuning                                           │
│  ┌───────────┐ ┌─────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ ResNet-50 │→│ FPN │→│ DETR Dec │→│  HTM     │→│ ID Head  │ ←学習  │
│  └───────────┘ └─────┘ └──────────┘ └──────────┘ └──────────┘        │
│                                                   ↘ Action Head ←学習  │
│  [全モジュール学習 / 小 lr=1e-5]                                       │
│  Loss: cls + bbox + action + id_cls + temporal_smooth                   │
│  データ: シーケンス / 出力: stage4_best.pth (完成モデル)               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 環境セットアップ

### 2-1. パッケージインストール

```bash
# カレントディレクトリを YOAKE に設定
cd c:/Users/hayam/YOAKE

# PyTorch (CUDA 12.x 向け。環境に合わせて変更)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 主要依存パッケージ
pip install \
    scipy \
    pyyaml \
    tqdm \
    pillow \
    opencv-python \
    torchmetrics \
    wandb

# パッケージとして src をインストール（editable）
pip install -e .
```

### 2-2. ディレクトリ構成の作成

```bash
YOAKE_TRYAL="C:/Users/hayam/Desktop/YOAKE_tryal"

mkdir -p "${YOAKE_TRYAL}/data/train"
mkdir -p "${YOAKE_TRYAL}/data/val"
mkdir -p "${YOAKE_TRYAL}/data/test"
mkdir -p "${YOAKE_TRYAL}/outputs/stage1"
mkdir -p "${YOAKE_TRYAL}/outputs/stage2"
mkdir -p "${YOAKE_TRYAL}/outputs/stage3"
mkdir -p "${YOAKE_TRYAL}/outputs/stage4"
mkdir -p "${YOAKE_TRYAL}/outputs/large/stage1"
mkdir -p "${YOAKE_TRYAL}/outputs/large/stage2"
mkdir -p "${YOAKE_TRYAL}/outputs/large/stage3"
mkdir -p "${YOAKE_TRYAL}/outputs/large/stage4"
mkdir -p "${YOAKE_TRYAL}/outputs/inference"
```

---

## 3. データ準備

### 3-1. アノテーション JSON フォーマット仕様

```
annotations.json
├─ videos: [ VideoInfo, ... ]       # 動画リスト
│   ├─ video_id: str                # 一意な動画ID (例: "video_001")
│   ├─ video_path: str              # 動画ファイルパス (絶対パス推奨)
│   ├─ fps: float                   # フレームレート
│   ├─ width: int                   # フレーム幅 [px]
│   ├─ height: int                  # フレーム高 [px]
│   └─ frames: [ FrameInfo, ... ]   # フレームリスト
│       ├─ frame_id: int            # 0始まりのフレームインデックス
│       ├─ image_path: str          # 画像ファイルパス (絶対パス推奨)
│       └─ annotations: [ Ann, ... ]# このフレームのアノテーション
│           ├─ track_id: int        # 個体ID (動画内で一貫した整数)
│           ├─ class_id: int        # 種クラス (0始まり; 単種なら常に 0)
│           ├─ bbox: [cx, cy, w, h] # 正規化座標 [0,1] の中心+幅高さ
│           └─ action_id: int       # 行動クラス (0始まり; ラベルなし時は -1)
├─ action_names: [ str, ... ]       # 行動クラス名リスト (省略可)
└─ class_names: [ str, ... ]        # 種クラス名リスト (省略可)
```

**フィールド補足:**
| フィールド | 型 | 説明 |
|---|---|---|
| `bbox` | `[float×4]` | `[cx, cy, w, h]` — 画像幅・高さで正規化 (0〜1) |
| `track_id` | `int` | 同一個体は動画全体で同じ値。動画をまたいでリセット可 |
| `action_id` | `int` | Stage 2/4 でのみ使用。-1 は未アノテーション |
| `class_id` | `int` | 複数種混在シーン用。単種なら常に 0 |

### 3-2. サンプル JSON（最小例）

```json
{
  "action_names": ["idle", "walk", "groom", "interact", "other"],
  "class_names": ["drosophila"],
  "videos": [
    {
      "video_id": "video_001",
      "video_path": "C:/Users/hayam/Desktop/YOAKE_tryal/videos/video_001.mp4",
      "fps": 30.0,
      "width": 1280,
      "height": 720,
      "frames": [
        {
          "frame_id": 0,
          "image_path": "C:/Users/hayam/Desktop/YOAKE_tryal/frames/video_001/frame_000000.jpg",
          "annotations": [
            {"track_id": 0, "class_id": 0, "bbox": [0.3, 0.4, 0.05, 0.08], "action_id": 0},
            {"track_id": 1, "class_id": 0, "bbox": [0.6, 0.5, 0.06, 0.09], "action_id": 1}
          ]
        },
        {
          "frame_id": 1,
          "image_path": "C:/Users/hayam/Desktop/YOAKE_tryal/frames/video_001/frame_000001.jpg",
          "annotations": [
            {"track_id": 0, "class_id": 0, "bbox": [0.31, 0.41, 0.05, 0.08], "action_id": 0},
            {"track_id": 1, "class_id": 0, "bbox": [0.61, 0.51, 0.06, 0.09], "action_id": 2}
          ]
        },
        {
          "frame_id": 2,
          "image_path": "C:/Users/hayam/Desktop/YOAKE_tryal/frames/video_001/frame_000002.jpg",
          "annotations": [
            {"track_id": 0, "class_id": 0, "bbox": [0.32, 0.42, 0.05, 0.08], "action_id": 0},
            {"track_id": 1, "class_id": 0, "bbox": [0.62, 0.52, 0.06, 0.09], "action_id": 3}
          ]
        }
      ]
    }
  ]
}
```

### 3-3. データ配置の確認スクリプト

```python
# scripts/check_data.py
# 実行: python scripts/check_data.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path
from htrtdetr.data import load_annotations

ROOT = "C:/Users/hayam/Desktop/YOAKE_tryal"

def check_split(split: str):
    anno_path = f"{ROOT}/data/{split}/annotations.json"
    if not Path(anno_path).exists():
        print(f"[WARN] {split}: annotations.json not found at {anno_path}")
        return

    videos, action_names, class_names = load_annotations(anno_path)
    total_frames = sum(len(v.frames) for v in videos)
    total_anns   = sum(len(f.annotations) for v in videos for f in v.frames)
    missing_imgs = []
    for v in videos:
        for f in v.frames:
            if not Path(f.image_path).exists():
                missing_imgs.append(f.image_path)

    print(f"\n=== {split.upper()} ===")
    print(f"  動画数       : {len(videos)}")
    print(f"  総フレーム数 : {total_frames}")
    print(f"  総アノテーション数: {total_anns}")
    print(f"  行動クラス   : {action_names}")
    print(f"  種クラス     : {class_names}")
    if missing_imgs:
        print(f"  [WARN] 画像未発見: {len(missing_imgs)} 件 (例: {missing_imgs[:3]})")
    else:
        print(f"  [OK] 全画像ファイルが存在します")

for split in ["train", "val", "test"]:
    check_split(split)
```

---

## 4. 各ステージの詳細

---

### Stage 1: Detector Pretraining

#### (a) 目的と学習対象モジュール

| モジュール | 状態 | 説明 |
|---|---|---|
| ResNet-50 backbone | **学習** (backbone_lr × 0.1) | ImageNet 事前学習済みから fine-tune |
| FPN (out=512) | **学習** | |
| DETR Decoder ×6 | **学習** | Hungarian matching で bbox 回帰・分類 |
| Hierarchical Temporal Module | 凍結/未使用 | Stage 1 では forward されない |
| Memory ID Head | 凍結/未使用 | |
| Action Head | 凍結/未使用 | |

単一フレームの bbox 検出を安定させることが目的。
`SingleFrameDataset` を使用（シーケンス不要）。

#### (b) 実行コマンド（RTX 8000向け最適化）

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage1.py \
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/train/annotations.json \
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/val/annotations.json \
    train.max_epochs=500 \
    train.use_amp=true \
    data.batch_size=64 \
    data.num_workers=12 \
    data.pin_memory=true \
    data.persistent_workers=true \
    data.prefetch_factor=4 \
    data.image_size="[640,640]" \
    optimizer.lr=1e-4 \
    optimizer.backbone_lr_factor=0.1 \
    optimizer.weight_decay=1e-4 \
    optimizer.grad_clip_norm=0.1 \
    scheduler.warmup_epochs=5 \
    scheduler.eta_min=1e-6 \
    train.save_best=true \
    train.log_interval=10
```

> **VRAM 目安（large variant, batch=64, 640×640）**: ~38GB
> OOM が出た場合は `data.batch_size=32` に下げてください。

#### (c) Pythonコードによる設定（RTX 8000 最適化）

```python
import sys
sys.path.insert(0, "src")

from htrtdetr.config.config import get_variant_config, validate_config

cfg = get_variant_config(
    variant="large",
    stage=1,
    overrides={
        "data": {
            "batch_size": 64,
            "num_workers": 12,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "image_size": [640, 640],
            "window_size": 16,       # Stage 1 は単フレームだが整合のため設定
            "augment_train": True,
            "aug_hflip": True,
            "aug_brightness": 0.2,
            "aug_contrast": 0.2,
        },
        "train": {
            "max_epochs": 500,
            "early_stopping_patience": 200,
            "use_amp": True,         # AMP 常時有効
            "seed": 42,
            "log_interval": 10,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "backbone_lr_factor": 0.1,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "total_epochs": 500,
            "eta_min": 1e-6,
        },
        "loss": {
            "w_class": 2.0,
            "w_bbox_l1": 5.0,
            "w_bbox_giou": 2.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
        },
        "model": {
            "detector": {
                "head": {
                    "num_classes": 1,      # ショウジョウバエ単種
                    "num_queries": 100,    # 最大100個体
                }
            }
        }
    }
)

# large variant の実際の値を確認
print(f"backbone   : {cfg.model.detector.backbone.name}")          # resnet50
print(f"hidden_dim : {cfg.model.detector.head.hidden_dim}")        # 512
print(f"num_heads  : {cfg.model.detector.head.num_heads}")         # 16
print(f"dec_layers : {cfg.model.detector.head.num_decoder_layers}")# 6
print(f"ffn_dim    : {cfg.model.detector.head.ffn_dim}")           # 2048
print(f"batch_size : {cfg.data.batch_size}")                       # 64
print(f"use_amp    : {cfg.train.use_amp}")                         # True

validate_config(cfg)
print("Config validation: OK")

# YAML に保存
cfg.save_yaml("C:/Users/hayam/Desktop/YOAKE_tryal/configs/stage1_large.yaml")
```

**RTX 8000 向けパラメーター一覧（Stage 1）:**

| パラメーター | 値 | 理由 |
|---|---|---|
| `data.batch_size` | 64 | 48GB VRAM / 単フレームは軽量 |
| `data.num_workers` | 12 | CPU コア数に応じて調整 |
| `data.prefetch_factor` | 4 | DataLoader のプリフェッチ深度 |
| `train.use_amp` | True | fp16 で VRAM ほぼ半減、速度向上 |
| `model variant` | large | ResNet-50, hidden_dim=512, ~65M params |
| `optimizer.grad_clip_norm` | 0.1 | 勾配爆発防止 |
| `gradient_accumulation_steps` | 1 | batch=64 で十分; OOM時は batch=16 + accum=4 |

#### (d) 期待される学習ログ

```
[Stage1] Epoch  1/500 | loss=8.42 | cls=2.31 | bbox=4.67 | giou=1.44 | lr=1.0e-5
[Stage1] Epoch 10/500 | loss=4.13 | cls=1.02 | bbox=2.21 | giou=0.90 | AP50=0.41
[Stage1] Epoch 50/500 | loss=1.87 | cls=0.38 | bbox=1.12 | giou=0.37 | AP50=0.71
[Stage1] Epoch100/500 | loss=1.31 | cls=0.24 | bbox=0.77 | giou=0.30 | AP50=0.80
[Stage1] Epoch200/500 | loss=0.96 | cls=0.17 | bbox=0.56 | giou=0.23 | AP50=0.86
[Stage1] Epoch500/500 | loss=0.71 | cls=0.12 | bbox=0.41 | giou=0.18 | AP50=0.89
```

**収束の目安:**
- `AP50 >= 0.80` → Stage 2/3 へ進んでよい
- `AP50 >= 0.85` → 良好
- `loss < 1.0` かつ train/val の乖離が小さい → 過学習なし

#### (e) チェックポイントの確認方法

```python
import sys
sys.path.insert(0, "src")
import torch
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

ckpt_path = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage1/stage1_best.pth"
cfg = get_variant_config("large", stage=1)
model = build_model(cfg.model)

ckpt = torch.load(ckpt_path, map_location="cpu")
print("Keys in checkpoint:", list(ckpt.keys()))
# 通常: ['epoch', 'model_state_dict', 'optimizer_state_dict', 'best_metric', ...]

print(f"Best epoch   : {ckpt.get('epoch', 'N/A')}")
print(f"Best AP50    : {ckpt.get('best_metric', 'N/A'):.4f}")

# 重みロード
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"Missing keys  : {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
# Stage 1 モデルなので temporal/id/action 関連は missing で正常
```

#### (f) 評価の実行コマンドと出力例

```bash
python scripts/eval_stage1.py \
    anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/val/annotations.json \
    checkpoint=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage1/stage1_best.pth \
    output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/eval_stage1
```

期待される出力:
```
=== Detection Evaluation (Stage 1) ===
AP50             : 0.887
AP75             : 0.762
Recall@50        : 0.921
Mean center error: 3.42 px  (at 640×640)
Results saved to: C:/Users/hayam/Desktop/YOAKE_tryal/outputs/eval_stage1/
```

---

### Stage 2: Action Head Pretraining

#### (a) 目的と学習対象モジュール

| モジュール | 状態 |
|---|---|
| Detector (backbone + FPN + DETR) | **凍結** (`backbone_lr_factor=0.0`) |
| Hierarchical Temporal Module (short/mid/long) | **学習** |
| Action Head + Interaction | **学習** |
| Memory ID Head | 凍結/未使用 |

`SlidingWindowDataset`（`window_size=16`）を使用。
行動ラベル付きフレームが必要（`require_action=True`）。

#### (b) 実行コマンド

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage2.py \
    train.resume=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage1/stage1_best.pth \
    train.max_epochs=80 \
    train.use_amp=true \
    data.batch_size=16 \
    data.num_workers=12 \
    data.window_size=16 \
    data.window_stride=8 \
    data.pin_memory=true \
    data.persistent_workers=true \
    optimizer.lr=5e-5 \
    optimizer.backbone_lr_factor=0.0 \
    optimizer.weight_decay=1e-4 \
    scheduler.warmup_epochs=3 \
    scheduler.total_epochs=80 \
    loss.w_action=1.0 \
    loss.w_class=0.0 \
    loss.w_bbox_l1=0.0 \
    loss.w_bbox_giou=0.0
```

> **VRAM 目安（large variant, batch=16, window=16, 640×640）**: ~42GB
> OOM 時は `data.batch_size=8`（~22GB）

#### (c) Pythonコードによる設定

```python
import sys
sys.path.insert(0, "src")
from htrtdetr.config.config import get_variant_config

cfg = get_variant_config(
    variant="large",
    stage=2,
    overrides={
        "data": {
            "batch_size": 16,
            "num_workers": 12,
            "window_size": 16,
            "window_stride": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "image_size": [640, 640],
        },
        "train": {
            "max_epochs": 80,
            "use_amp": True,
            "early_stopping_patience": 30,
            "log_interval": 10,
        },
        "optimizer": {
            "lr": 5e-5,
            "backbone_lr_factor": 0.0,   # detector 完全凍結
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 80,
            "eta_min": 1e-6,
        },
        "loss": {
            "w_action": 1.0,
            "w_class": 0.0,     # detector は凍結するので検出 loss は不要
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
        },
        "model": {
            "action_head": {
                "num_actions": 5,        # idle/walk/groom/interact/other
                "dropout": 0.1,
                "use_interaction": True,
            }
        }
    }
)

# Temporal Module の channel_attention が有効なことを確認（large では True）
print(f"short_branch channel_attention: {cfg.model.temporal.short_branch.use_channel_attention}")
# True
```

**RTX 8000 向けパラメーター一覧（Stage 2）:**

| パラメーター | 値 | 理由 |
|---|---|---|
| `data.batch_size` | 16 | シーケンス(T=16フレーム)はメモリ重い |
| `data.window_size` | 16 | long_branch.num_frames=16 と一致必須 |
| `data.window_stride` | 8 | 50% オーバーラップ |
| `train.use_amp` | True | AMP 常時有効 |
| `model variant` | large | Stage 1 と同一バリアント |
| `optimizer.backbone_lr_factor` | 0.0 | detector 完全凍結 |
| `gradient_accumulation_steps` | 1 | batch=16 で十分; OOM時は batch=4+accum=4 |

#### (d) 期待される学習ログ

```
[Stage2] Epoch  1/80 | action_loss=1.61 | action_acc=0.22 | lr=5.0e-6
[Stage2] Epoch 10/80 | action_loss=1.12 | action_acc=0.51 | macro_F1=0.38
[Stage2] Epoch 30/80 | action_loss=0.74 | action_acc=0.68 | macro_F1=0.62
[Stage2] Epoch 50/80 | action_loss=0.56 | action_acc=0.75 | macro_F1=0.71
[Stage2] Epoch 80/80 | action_loss=0.44 | action_acc=0.80 | macro_F1=0.77
```

**収束の目安:**
- `macro_F1 >= 0.65` → Stage 4 へ進んでよい
- クラス間で F1 の乖離が大きい場合 → クラスウェイトの調整を検討

#### (e) チェックポイントの確認方法

```python
import sys
sys.path.insert(0, "src")
import torch
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

ckpt_path = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage2/stage2_best.pth"
cfg = get_variant_config("large", stage=2)
model = build_model(cfg.model)
model.set_stage(2)

ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Best epoch  : {ckpt.get('epoch')}")
print(f"Best macro_F1: {ckpt.get('best_metric', 0):.4f}")

missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"Missing keys (ID head 等は正常): {missing[:5]}")
```

#### (f) 評価の実行コマンドと出力例

```bash
python scripts/eval_stage2.py \
    checkpoint=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage2/stage2_best.pth \
    output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/eval_stage2
```

```
=== Action Classification Evaluation (Stage 2) ===
Overall Accuracy : 0.804
Macro F1-score   : 0.771
Per-class F1:
  idle     : 0.88
  walk     : 0.82
  groom    : 0.74
  interact : 0.69
  other    : 0.71
```

---

### Stage 3: ID Head Pretraining

#### (a) 目的と学習対象モジュール

| モジュール | 状態 |
|---|---|
| Detector (backbone + FPN + DETR) | **凍結** |
| Hierarchical Temporal Module | **学習** |
| Memory ID Head (GRU + embedding) | **学習** |
| Action Head | 凍結/未使用 |

`SlidingWindowDataset`（`require_track=True`）を使用。
`use_metric_loss=True` でトリプレット損失が追加される。

#### (b) 実行コマンド

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage3.py \
    train.max_epochs=80 \
    train.use_amp=true \
    data.batch_size=16 \
    data.num_workers=12 \
    data.window_size=16 \
    data.window_stride=8 \
    data.pin_memory=true \
    data.persistent_workers=true \
    optimizer.lr=5e-5 \
    optimizer.backbone_lr_factor=0.0 \
    optimizer.weight_decay=1e-4 \
    scheduler.warmup_epochs=3 \
    scheduler.total_epochs=80 \
    loss.w_id_cls=1.0 \
    loss.w_id_metric=0.5 \
    model.id_head.use_metric_loss=true \
    model.id_head.metric_loss_margin=0.3 \
    model.id_head.max_ids=50 \
    model.id_head.memory_ttl=30 \
    model.id_head.new_id_threshold=0.5
```

#### (c) Pythonコードによる設定

```python
import sys
sys.path.insert(0, "src")
from htrtdetr.config.config import get_variant_config

cfg = get_variant_config(
    variant="large",
    stage=3,
    overrides={
        "data": {
            "batch_size": 16,
            "num_workers": 12,
            "window_size": 16,
            "window_stride": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "max_epochs": 80,
            "use_amp": True,
            "early_stopping_patience": 30,
        },
        "optimizer": {
            "lr": 5e-5,
            "backbone_lr_factor": 0.0,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 80,
            "eta_min": 1e-6,
        },
        "loss": {
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,      # triplet loss の重み
        },
        "model": {
            "id_head": {
                "use_metric_loss": True,
                "metric_loss_margin": 0.3,
                "max_ids": 50,
                "memory_ttl": 30,
                "new_id_threshold": 0.5,
                "use_appearance": True,
                "use_geometry": True,
                "use_motion": True,
                # embedding_dim=256, memory_dim=512 は large から自動設定
            }
        }
    }
)

print(f"ID embedding_dim: {cfg.model.id_head.embedding_dim}")  # 256
print(f"ID memory_dim   : {cfg.model.id_head.memory_dim}")     # 512
print(f"metric_loss     : {cfg.model.id_head.use_metric_loss}")# True
```

**RTX 8000 向けパラメーター一覧（Stage 3）:**

| パラメーター | 値 |
|---|---|
| `data.batch_size` | 16 |
| `data.window_size` | 16 |
| `model.id_head.max_ids` | 50 |
| `model.id_head.memory_ttl` | 30 |
| `model.id_head.embedding_dim` | 256 (large 自動設定) |
| `model.id_head.memory_dim` | 512 (large 自動設定) |
| `train.use_amp` | True |

#### (d) 期待される学習ログ

```
[Stage3] Epoch  1/80 | id_cls_loss=3.91 | metric_loss=0.82 | MOTA=0.12
[Stage3] Epoch 10/80 | id_cls_loss=2.44 | metric_loss=0.55 | MOTA=0.41
[Stage3] Epoch 30/80 | id_cls_loss=1.53 | metric_loss=0.31 | MOTA=0.63
[Stage3] Epoch 50/80 | id_cls_loss=1.12 | metric_loss=0.22 | MOTA=0.72
[Stage3] Epoch 80/80 | id_cls_loss=0.87 | metric_loss=0.17 | MOTA=0.78
```

**収束の目安:**
- `MOTA >= 0.65` → Stage 4 へ進んでよい
- `metric_loss < 0.25` → embedding 空間の分離が十分

#### (e) チェックポイントの確認方法

```python
import sys
sys.path.insert(0, "src")
import torch
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

ckpt_path = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage3/stage3_best.pth"
cfg = get_variant_config("large", stage=3)
model = build_model(cfg.model)
model.set_stage(3)

ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Best MOTA: {ckpt.get('best_metric', 0):.4f}")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
print("Weights loaded OK")
```

#### (f) 評価の実行コマンド

```bash
python scripts/eval_stage3.py \
    checkpoint=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage3/stage3_best.pth \
    output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/eval_stage3
```

```
=== Tracking Evaluation (Stage 3) ===
MOTA  : 0.783
MOTP  : 0.812
IDF1  : 0.741
ID Switches: 42 / 1000 tracks
```

---

### Stage 4: Unified Fine-tuning

#### (a) 目的と学習対象モジュール

| モジュール | 状態 |
|---|---|
| 全モジュール | **学習**（小さい lr=1e-5） |
| Memory ID Head | `use_action_summary=True` に変更 |

Stage 2/3 で独立して学習した各ヘッドを統合し、相互に最適化する。

#### (b) 実行コマンド

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage4.py \
    train.max_epochs=50 \
    train.use_amp=true \
    data.batch_size=8 \
    data.num_workers=12 \
    data.window_size=16 \
    data.window_stride=8 \
    data.pin_memory=true \
    data.persistent_workers=true \
    optimizer.lr=1e-5 \
    optimizer.backbone_lr_factor=0.1 \
    optimizer.weight_decay=1e-4 \
    optimizer.grad_clip_norm=0.1 \
    scheduler.warmup_epochs=2 \
    scheduler.total_epochs=50 \
    loss.w_class=2.0 \
    loss.w_bbox_l1=5.0 \
    loss.w_bbox_giou=2.0 \
    loss.w_action=1.0 \
    loss.w_id_cls=1.0 \
    loss.w_id_metric=0.5 \
    loss.w_temporal_smooth=0.1 \
    model.id_head.use_action_summary=true
```

> **VRAM 目安（large variant, batch=8, window=16）**: ~44GB
> OOM 時は `data.batch_size=4`（~24GB）

#### (c) Pythonコードによる設定

```python
import sys
sys.path.insert(0, "src")
from htrtdetr.config.config import get_variant_config

cfg = get_variant_config(
    variant="large",
    stage=4,
    overrides={
        "data": {
            "batch_size": 8,
            "num_workers": 12,
            "window_size": 16,
            "window_stride": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "image_size": [640, 640],
        },
        "train": {
            "max_epochs": 50,
            "use_amp": True,
            "early_stopping_patience": 20,
            "log_interval": 5,
        },
        "optimizer": {
            "lr": 1e-5,               # 全体 lr は非常に小さく
            "backbone_lr_factor": 0.1,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 2,
            "total_epochs": 50,
            "eta_min": 1e-7,
        },
        "loss": {
            "w_class": 2.0,
            "w_bbox_l1": 5.0,
            "w_bbox_giou": 2.0,
            "w_action": 1.0,
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,
            "w_temporal_smooth": 0.1,
        },
        "model": {
            "id_head": {
                "use_action_summary": True,  # Stage 4 で有効化
                "max_ids": 50,
                "memory_ttl": 30,
            }
        }
    }
)

print(f"use_action_summary: {cfg.model.id_head.use_action_summary}")  # True
print(f"lr               : {cfg.optimizer.lr}")                        # 1e-5
```

**RTX 8000 向けパラメーター一覧（Stage 4）:**

| パラメーター | 値 | 理由 |
|---|---|---|
| `data.batch_size` | 8 | 全モジュール起動で最重量 |
| `data.window_size` | 16 | Stage 2/3 と同一 |
| `optimizer.lr` | 1e-5 | fine-tune は小 lr |
| `optimizer.backbone_lr_factor` | 0.1 | backbone も少し学習 |
| `train.use_amp` | True | AMP 常時有効 |
| `gradient_accumulation_steps` | 1 (batch=8で十分) | |

#### (d) 期待される学習ログ

```
[Stage4] Epoch  1/50 | total=3.21 | det=1.42 | action=0.61 | id=0.98 | smooth=0.20
[Stage4] Epoch 10/50 | total=2.18 | det=0.87 | action=0.44 | id=0.72 | smooth=0.15
[Stage4] Epoch 30/50 | total=1.67 | det=0.64 | action=0.34 | id=0.57 | smooth=0.12
[Stage4] Epoch 50/50 | total=1.41 | det=0.53 | action=0.29 | id=0.48 | smooth=0.11
                      AP50=0.90   macro_F1=0.80   MOTA=0.82
```

**収束の目安:**
- `AP50 >= 0.88`, `macro_F1 >= 0.77`, `MOTA >= 0.79` → 良好な収束

#### (e) チェックポイントの確認方法

```python
import sys
sys.path.insert(0, "src")
import torch
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

ckpt_path = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage4/stage4_best.pth"
cfg = get_variant_config("large", stage=4)
model = build_model(cfg.model)
model.set_stage(4)

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model_state_dict"]

# 全モジュールの重みがロードされることを確認
missing, unexpected = model.load_state_dict(state, strict=True)
assert len(missing) == 0 and len(unexpected) == 0, \
    f"Unexpected mismatch: missing={missing}, unexpected={unexpected}"
print(f"Stage 4 checkpoint fully loaded. Epoch: {ckpt['epoch']}")
```

#### (f) 評価の実行コマンドと出力例

```bash
python scripts/eval_stage4.py \
    checkpoint=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage4/stage4_best.pth \
    output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/eval_stage4
```

```
=== Unified Evaluation (Stage 4) ===
Detection  AP50 : 0.901
Detection  AP75 : 0.823
Tracking   MOTA : 0.824
Tracking   IDF1 : 0.789
Action macro_F1 : 0.804
```

---

## 5. Stage 間の重み引き継ぎ

### 重みロードのコード

```python
import sys
sys.path.insert(0, "src")
import torch
from pathlib import Path
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model
from htrtdetr.utils import load_model_weights

ROOT = "C:/Users/hayam/Desktop/YOAKE_tryal"

def load_stage_weights(target_stage: int, variant: str = "large") -> None:
    """指定ステージのモデルに前ステージの重みをロードする"""

    cfg = get_variant_config(variant, stage=target_stage)
    model = build_model(cfg.model)
    model.set_stage(target_stage)

    # 前ステージチェックポイントの優先順位
    priority = {
        2: [f"{ROOT}/outputs/stage1/stage1_best.pth"],
        3: [f"{ROOT}/outputs/stage2/stage2_best.pth",
            f"{ROOT}/outputs/stage1/stage1_best.pth"],
        4: [f"{ROOT}/outputs/stage3/stage3_best.pth",
            f"{ROOT}/outputs/stage2/stage2_best.pth",
            f"{ROOT}/outputs/stage1/stage1_best.pth"],
    }

    for ckpt_path in priority.get(target_stage, []):
        if Path(ckpt_path).exists():
            print(f"Loading: {ckpt_path}")
            load_model_weights(ckpt_path, model, strict=False)
            break
    else:
        print("No previous checkpoint found. Starting from scratch.")

    return model

# 使用例
model_for_stage2 = load_stage_weights(target_stage=2)
model_for_stage4 = load_stage_weights(target_stage=4)
```

### 各ステージで引き継がれる / 初期化されるレイヤー一覧

| レイヤー / モジュール | Stage 1 → 2 | Stage 1/2 → 3 | Stage 2/3 → 4 |
|---|---|---|---|
| `backbone` (ResNet-50) | 引き継ぎ | 引き継ぎ | 引き継ぎ |
| `fpn` (FPN) | 引き継ぎ | 引き継ぎ | 引き継ぎ |
| `decoder` (DETR Decoder) | 引き継ぎ | 引き継ぎ | 引き継ぎ |
| `temporal.short_branch` | **ランダム初期化** | 引き継ぎ (stage2から) | 引き継ぎ |
| `temporal.mid_branch` | **ランダム初期化** | 引き継ぎ | 引き継ぎ |
| `temporal.long_branch` | **ランダム初期化** | 引き継ぎ | 引き継ぎ |
| `action_head` | **ランダム初期化** | **ランダム初期化** | 引き継ぎ (stage2から) |
| `id_head.memory_net` | **ランダム初期化** | **ランダム初期化** | 引き継ぎ (stage3から) |
| `id_head.id_classifier` | **ランダム初期化** | **ランダム初期化** | 引き継ぎ |

> `load_model_weights(..., strict=False)` を使うため、キーが存在しないモジュールはランダム初期化のまま。

---

## 6. 複数種混在シーン向け設定

`use_species_separated_pools=True` を使用すると、種ごとに独立した ID メモリプールでマッチングが行われる。
異種間の ID 混同（例: ♂ を ♀ の ID として追跡してしまう）を防ぐ。

### 設定コード

```python
import sys
sys.path.insert(0, "src")
from htrtdetr.config.config import get_variant_config

# 2種混在シーン（例: ♂ ♀ 分離）
cfg = get_variant_config(
    variant="large",
    stage=4,
    overrides={
        "data": {
            "batch_size": 8,
            "num_workers": 12,
            "window_size": 16,
            "window_stride": 8,
        },
        "train": {
            "max_epochs": 50,
            "use_amp": True,
        },
        "optimizer": {
            "lr": 1e-5,
            "backbone_lr_factor": 0.1,
        },
        "model": {
            "detector": {
                "head": {
                    "num_classes": 2,    # ♂=0, ♀=1 など
                }
            },
            "id_head": {
                "use_species_separated_pools": True,
                "num_species": 2,        # num_classes と一致させる
                "max_ids": 50,
            }
        }
    }
)

print(f"num_classes              : {cfg.model.detector.head.num_classes}")     # 2
print(f"use_species_separated_pools: {cfg.model.id_head.use_species_separated_pools}")  # True
print(f"num_species              : {cfg.model.id_head.num_species}")            # 2
```

### 実行コマンド（複数種混在）

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage4.py \
    train.max_epochs=50 \
    train.use_amp=true \
    data.batch_size=8 \
    data.num_workers=12 \
    data.window_size=16 \
    data.window_stride=8 \
    optimizer.lr=1e-5 \
    model.detector.head.num_classes=2 \
    model.id_head.use_species_separated_pools=true \
    model.id_head.num_species=2 \
    model.id_head.max_ids=50
```

### 種分離マッチングの仕組み

```python
# memory_id_head.py の forward_inference 内（抜粋・説明用）
#
# use_species_separated_pools=True の場合:
#   種クラス 0 の検出 → 種クラス 0 の memory のみとマッチング
#   種クラス 1 の検出 → 種クラス 1 の memory のみとマッチング
#
# マッチングコスト:
#   cost = -(0.5 * cosine_similarity + 0.5 * IoU)
#   cost > -new_id_threshold (=0.5) → 新規 ID として割り当て
```

---

## 7. トラブルシューティング

### OOM（Out of Memory）が発生した場合

**症状**: `CUDA out of memory` エラー

| 対処法 | コマンド例 |
|---|---|
| batch_size を半減 | `data.batch_size=8` → `data.batch_size=4` |
| window_size を短縮 | `data.window_size=16` → `data.window_size=8` |
| 画像サイズを縮小 | `data.image_size="[480,480]"` |
| gradient accumulation を使う | batch=4 + `optimizer.grad_accumulation=4` を実装 |
| AMP を確認 | `train.use_amp=true` になっているか確認 |

**勾配蓄積の手動実装例**（現スクリプトにない場合）:
```python
# Trainer を修正する代わりに、一時的に batch_size を下げて
# 実効 batch を維持する方法:
# batch=16, accum=1 ≡ batch=4, accum=4 (勾配の合計は同じ)
# 現スクリプトは accum 非対応なので batch_size のみ下げる
```

### loss が発散した場合

**症状**: loss が NaN または急激に増加

| 原因 | 対処法 |
|---|---|
| 学習率が大きすぎる | lr を 1/10 に下げる (`optimizer.lr=1e-5`) |
| grad_clip が無効 | `optimizer.grad_clip_norm=0.1` を確認 |
| AMP のオーバーフロー | `train.use_amp=false` で一時的に無効化して確認 |
| bbox 座標が正規化されていない | アノテーション JSON の bbox を [0,1] に正規化 |
| loss weight が大きすぎる | `loss.w_bbox_l1=5.0` → `loss.w_bbox_l1=2.0` に下げる |

```bash
# AMP を無効化してデバッグ
python scripts/train_stage1.py train.use_amp=false optimizer.lr=1e-5
```

### ID スイッチが多発する場合

**症状**: MOTA は高いが ID Switches が多い

| 原因 | 対処法 |
|---|---|
| `new_id_threshold` が低すぎる | `model.id_head.new_id_threshold=0.5` → `0.6` に上げる |
| `memory_ttl` が短すぎる | `model.id_head.memory_ttl=30` → `50` に増やす |
| metric loss が弱い | `loss.w_id_metric=0.5` → `1.0` に上げる |
| Stage 3 の学習が不十分 | Stage 3 を epoch 80 → 150 に延ばす |

```bash
python scripts/train_stage3.py \
    train.max_epochs=150 \
    model.id_head.new_id_threshold=0.6 \
    model.id_head.memory_ttl=50 \
    loss.w_id_metric=1.0
```

### 行動分類の F1 が上がらない場合

**症状**: `macro_F1 < 0.5` で停滞

| 原因 | 対処法 |
|---|---|
| クラス不均衡 | focal loss の `focal_gamma=2.0`, `focal_alpha=0.25` を確認 |
| interaction feature が効いていない | `model.action_head.use_interaction=true` を確認 |
| window_size が短すぎる | `data.window_size=16` → `32` に増やす（メモリ注意）|
| 行動ラベルの質が低い | アノテーションの見直し |
| detector が不十分 | Stage 1 の AP50 が 0.75 未満なら Stage 2 前に再学習 |

```bash
# クラスウェイトを上げて再学習
python scripts/train_stage2.py \
    train.max_epochs=120 \
    loss.focal_gamma=3.0 \
    data.window_size=32 \
    data.batch_size=8
```

---

## 8. 学習完了後の推論実行

```python
# inference.py — 動画ファイルに対して推論を実行し、可視化付きで保存する
# 実行: python inference.py
#
# 前提: Stage 4 チェックポイントが存在すること

import sys
sys.path.insert(0, "src")

import cv2
import torch
import numpy as np
from pathlib import Path
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model
from htrtdetr.models.id_head import IdentityMemory

# ===== 設定 =====
CHECKPOINT  = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage4/stage4_best.pth"
INPUT_VIDEO = "C:/Users/hayam/Desktop/YOAKE_tryal/videos/test_video.mp4"
OUTPUT_VIDEO= "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/inference/result.mp4"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = (640, 640)   # (H, W)
SCORE_THR   = 0.3
ACTION_NAMES= ["idle", "walk", "groom", "interact", "other"]

# ID ごとに色を固定
COLORS = [(np.random.randint(100,255), np.random.randint(100,255), np.random.randint(100,255))
          for _ in range(200)]

# ===== モデルロード =====
cfg = get_variant_config("large", stage=4)
cfg.model.detector.score_threshold = SCORE_THR

model = build_model(cfg.model)
model.set_stage(4)
ckpt = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval().to(DEVICE)

memory = model.id_head.create_memory(DEVICE)

# ===== 動画処理 =====
cap = cv2.VideoCapture(INPUT_VIDEO)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"Processing: {INPUT_VIDEO}")
print(f"FPS={fps}, {width}x{height}")

frame_idx = 0
with torch.no_grad():
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # --- 前処理 ---
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(frame_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        tensor    = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor    = tensor.unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

        # --- 推論 ---
        output = model(tensor, memory=memory)

        # --- 結果の取り出し ---
        # output.det_results は List[Dict] (バッチサイズ=1)
        if output.det_results and len(output.det_results) > 0:
            det = output.det_results[0]
            boxes      = det.get("boxes", torch.zeros(0, 4))        # [cx,cy,w,h] 正規化
            scores     = det.get("scores", torch.zeros(0))
            track_ids  = det.get("track_ids", torch.zeros(0, dtype=torch.long))
            action_ids = det.get("action_ids", torch.zeros(0, dtype=torch.long))

            # --- 描画 ---
            vis = frame_bgr.copy()
            for i in range(len(scores)):
                if scores[i] < SCORE_THR:
                    continue
                cx, cy, bw, bh = boxes[i].tolist()
                x1 = int((cx - bw/2) * width)
                y1 = int((cy - bh/2) * height)
                x2 = int((cx + bw/2) * width)
                y2 = int((cy + bh/2) * height)
                tid = track_ids[i].item()
                act = ACTION_NAMES[action_ids[i].item()] if len(action_ids) > i else "?"
                color = COLORS[tid % len(COLORS)]

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {act} {scores[i]:.2f}"
                cv2.putText(vis, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            vis = frame_bgr.copy()

        out.write(vis)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} processed")

cap.release()
out.release()
print(f"Saved: {OUTPUT_VIDEO}")
```

### 推論時の注意事項

- **メモリリセット**: シーン切り替え時は `memory.reset()` を呼ぶ
- **batch推論**: 複数動画を並列処理する場合は動画ごとに独立した `IdentityMemory` を作成する
- **score_threshold**: `0.3`（デフォルト）。密集シーンでは `0.4〜0.5` に上げると誤検出が減る
- **window_size との整合**: 推論時は1フレームずつ処理するため window は不要

---

## 付録: モデルバリアント比較

| バリアント | Backbone | hidden_dim | decoder layers | num_heads | 総パラメーター | 推奨 GPU |
|---|---|---|---|---|---|---|
| small  | ResNet-18 | 128 | 2 | 4  | ~15-20M | RTX 3080 (10GB) |
| medium | ResNet-34 | 256 | 4 | 8  | ~30-40M | RTX 3090 (24GB) |
| **large** | **ResNet-50** | **512** | **6** | **16** | **~55-70M** | **RTX 8000 (48GB)** |

---

## 付録: ステージ別 VRAM 使用量目安（large, 640×640）

| Stage | batch_size | window_size | VRAM (AMP有効) | VRAM (AMP無効) |
|---|---|---|---|---|
| Stage 1 | 64 | 1 (単フレーム) | ~38 GB | ~46 GB+ |
| Stage 1 | 32 | 1 | ~20 GB | ~28 GB |
| Stage 2 | 16 | 16 | ~42 GB | OOM |
| Stage 2 | 8  | 16 | ~22 GB | ~38 GB |
| Stage 3 | 16 | 16 | ~42 GB | OOM |
| Stage 3 | 8  | 16 | ~22 GB | ~38 GB |
| Stage 4 | 8  | 16 | ~44 GB | OOM |
| Stage 4 | 4  | 16 | ~24 GB | ~42 GB |

> ※上記はあくまで目安。実際の値はデータセットのアノテーション密度・個体数に依存する。

---

## 付録: 公開データセットを使った Pre-training

自前アノテーションデータが少ない場合、公開データセットで Stage 1（検出器）を事前学習してから
自前データで fine-tune することで精度・収束速度が大幅に向上する。

### 推奨データセット一覧

| データセット | 用途 | 特徴 | 入手先 |
|---|---|---|---|
| **COCO 2017** | Stage 1 detector pre-train | 118k 画像、80 クラス、bbox アノテーション | cocodataset.org |
| **MOT17 / MOT20** | Stage 1 + Stage 3 (ID) | 歩行者追跡、bbox + track_id | motchallenge.net |
| **Cell Tracking Challenge** | Stage 1 + Stage 3 | 顕微鏡細胞追跡、生物画像に近いドメイン | celltrackingchallenge.net |
| **Drosophila Tracking (DART)** | Stage 1〜4 全般 | ハエ追跡専用、bbox + track_id | 論文付属または要問い合わせ |
| **FlyTracker データ** | 行動分類補助 | ショウジョウバエ行動ラベル付き | Caltech FlyTracker |

**推奨戦略:**
```
1. COCO で Stage 1 pre-train (backbone + FPN + decoder の汎用的な物体検出能力を獲得)
   ↓
2. MOT17 で Stage 1 続き / Stage 3 pre-train (追跡・外観 embedding 能力を追加)
   ↓
3. 自前ハエデータで Stage 1〜4 fine-tune (ドメイン適応)
```

---

### COCO 2017 → YOAKE JSON 変換スクリプト

```bash
# 1. COCO のダウンロード (約 25GB)
#    https://cocodataset.org/#download から以下を取得:
#    - train2017.zip (images)
#    - val2017.zip   (images)
#    - annotations_trainval2017.zip

# 解凍先の想定ディレクトリ構成:
# C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/coco/
#   images/train2017/
#   images/val2017/
#   annotations/instances_train2017.json
#   annotations/instances_val2017.json
```

```python
# scripts/convert_coco_to_yoake.py
# 実行: python scripts/convert_coco_to_yoake.py
#
# COCO の instances アノテーションを YOAKE の JSON フォーマットに変換する。
# ・各 COCO 画像を「1枚 = 1フレームの動画」として扱う
# ・track_id は画像内の bbox インデックス（追跡情報なし）
# ・action_id は -1（未アノテーション）

import json
from pathlib import Path
from tqdm import tqdm

COCO_ROOT  = "C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/coco"
OUT_ROOT   = "C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/coco_yoake"


def convert_split(split: str, max_images: int = None):
    """
    split: "train" or "val"
    max_images: デバッグ時は小さい値を指定 (None で全件)
    """
    anno_path  = f"{COCO_ROOT}/annotations/instances_{split}2017.json"
    image_dir  = f"{COCO_ROOT}/images/{split}2017"
    out_path   = f"{OUT_ROOT}/data/{split}/annotations.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO {split} annotations...")
    with open(anno_path, "r") as f:
        coco = json.load(f)

    # image_id → image info のマップ
    id2img = {img["id"]: img for img in coco["images"]}

    # image_id → annotations のマップ
    id2anns: dict = {}
    for ann in coco["annotations"]:
        id2anns.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(id2img.keys())
    if max_images:
        image_ids = image_ids[:max_images]

    videos = []
    for img_id in tqdm(image_ids, desc=f"Converting {split}"):
        img_info = id2img[img_id]
        W = img_info["width"]
        H = img_info["height"]
        img_path = str(Path(image_dir) / img_info["file_name"])

        anns = id2anns.get(img_id, [])
        frame_anns = []
        for local_idx, ann in enumerate(anns):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]   # COCO: [x_min, y_min, w, h] pixel 値
            # → YOAKE: [cx, cy, w, h] 正規化
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            bw = w / W
            bh = h / H
            # bbox が画像外にはみ出ていないかクリップ
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))
            frame_anns.append({
                "track_id": local_idx,  # COCO に track_id はないので連番
                "class_id": 0,          # 単一クラス（物体検出器の pre-train 用）
                "bbox": [cx, cy, bw, bh],
                "action_id": -1,        # 行動ラベルなし
            })

        if not frame_anns:
            continue  # アノテーションなし画像はスキップ

        videos.append({
            "video_id": f"coco_{split}_{img_id:012d}",
            "video_path": img_path,
            "fps": 1.0,
            "width": W,
            "height": H,
            "frames": [{
                "frame_id": 0,
                "image_path": img_path,
                "annotations": frame_anns,
            }],
        })

    result = {
        "action_names": [],
        "class_names": ["object"],
        "videos": videos,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} images → {out_path}")


if __name__ == "__main__":
    convert_split("train")   # ~118k 画像
    convert_split("val")     # ~5k 画像
    # デバッグ用: convert_split("train", max_images=1000)
```

**変換後の COCO pre-train 実行:**

```bash
cd c:/Users/hayam/YOAKE

python scripts/train_stage1.py \
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/coco_yoake/data/train/annotations.json \
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/coco_yoake/data/val/annotations.json \
    train.max_epochs=100 \
    train.use_amp=true \
    data.batch_size=64 \
    data.num_workers=12 \
    optimizer.lr=1e-4 \
    train.output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_coco
```

---

### MOT17 → YOAKE JSON 変換スクリプト

```bash
# MOT17 のダウンロード:
#   https://motchallenge.net/data/MOT17/ から MOT17.zip を取得
#
# 解凍後の想定ディレクトリ構成:
# C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/MOT17/
#   train/
#     MOT17-02-DPM/
#       img1/          ← フレーム画像 (000001.jpg, ...)
#       gt/gt.txt      ← ground truth (frame,id,x,y,w,h,conf,class,vis)
#     MOT17-04-DPM/
#     ...
#   test/
#     ...
```

```python
# scripts/convert_mot17_to_yoake.py
# 実行: python scripts/convert_mot17_to_yoake.py
#
# MOT17 の gt.txt を YOAKE フォーマットに変換する。
# track_id が付いているため Stage 3 (ID Head) の pre-train にも使用可。

import csv
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

MOT17_ROOT = "C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/MOT17"
OUT_ROOT   = "C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/mot17_yoake"


def parse_gt(gt_path: Path, img_dir: Path, seq_name: str):
    """
    gt.txt を読み込み、YOAKE の video dict を返す。

    gt.txt フォーマット (MOTChallenge):
      frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    # フレームサイズを最初の画像から取得
    first_img = sorted(img_dir.glob("*.jpg"))[0]
    from PIL import Image
    with Image.open(first_img) as im:
        W, H = im.size

    # フレームごとのアノテーションを収集
    frame_anns: dict = defaultdict(list)
    with open(gt_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            frame_id  = int(row[0]) - 1    # 0 始まりに変換
            track_id  = int(row[1]) - 1    # 0 始まりに変換
            bb_left   = float(row[2])
            bb_top    = float(row[3])
            bb_width  = float(row[4])
            bb_height = float(row[5])
            conf      = float(row[6]) if len(row) > 6 else 1.0
            cls       = int(row[7])   if len(row) > 7 else 1   # 1=pedestrian
            vis       = float(row[8]) if len(row) > 8 else 1.0

            # 低可視性・低信頼度はスキップ
            if conf < 0.5 or vis < 0.3:
                continue
            # 歩行者クラス (1) のみ使用
            if cls != 1:
                continue

            # YOAKE フォーマットへ変換
            cx = (bb_left + bb_width  / 2) / W
            cy = (bb_top  + bb_height / 2) / H
            bw = bb_width  / W
            bh = bb_height / H
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            frame_anns[frame_id].append({
                "track_id": track_id,
                "class_id": 0,
                "bbox": [cx, cy, bw, bh],
                "action_id": -1,
            })

    # フレームリストを構築
    img_files = sorted(img_dir.glob("*.jpg"))
    frames = []
    for img_file in img_files:
        fid = int(img_file.stem) - 1   # 000001.jpg → 0
        anns = frame_anns.get(fid, [])
        if not anns:
            continue
        frames.append({
            "frame_id": fid,
            "image_path": str(img_file),
            "annotations": anns,
        })

    return {
        "video_id": seq_name,
        "video_path": str(img_dir.parent),
        "fps": 25.0,
        "width": W,
        "height": H,
        "frames": frames,
    }


def convert_split(split: str = "train"):
    out_path = f"{OUT_ROOT}/data/{split}/annotations.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    mot_dir = Path(MOT17_ROOT) / split
    seq_dirs = sorted(mot_dir.iterdir())

    # DPM/FRCNN/SDP の重複を避けるため DPM のみ使用
    seq_dirs = [d for d in seq_dirs if d.is_dir() and "DPM" in d.name]

    videos = []
    for seq_dir in tqdm(seq_dirs, desc=f"Converting MOT17 {split}"):
        gt_path  = seq_dir / "gt" / "gt.txt"
        img_dir  = seq_dir / "img1"
        if not gt_path.exists() or not img_dir.exists():
            print(f"  Skipping {seq_dir.name} (gt or img not found)")
            continue
        video = parse_gt(gt_path, img_dir, seq_dir.name)
        if video["frames"]:
            videos.append(video)
            print(f"  {seq_dir.name}: {len(video['frames'])} frames")

    result = {
        "action_names": [],
        "class_names": ["pedestrian"],
        "videos": videos,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(videos)} sequences → {out_path}")


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install pillow が必要です")
    convert_split("train")


# --- 変換後の内容確認 ---
# python -c "
# import json
# with open('C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/mot17_yoake/data/train/annotations.json') as f:
#     d = json.load(f)
# print(f'sequences : {len(d[\"videos\"])}')
# print(f'frames    : {sum(len(v[\"frames\"]) for v in d[\"videos\"])}')
# print(f'tracks    : {len(set(a[\"track_id\"] for v in d[\"videos\"] for fr in v[\"frames\"] for a in fr[\"annotations\"]))}')
# "
```

**MOT17 で Stage 1 を pre-train し、Stage 3 を続けて実行する例:**

```bash
# Step 1: MOT17 で Stage 1 pre-train
python scripts/train_stage1.py \
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/mot17_yoake/data/train/annotations.json \
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/pretrain_data/mot17_yoake/data/train/annotations.json \
    train.max_epochs=150 \
    train.use_amp=true \
    data.batch_size=64 \
    data.num_workers=12 \
    train.output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_mot17_stage1

# Step 2: MOT17 で Stage 3 pre-train (上記の重みを使用)
python scripts/train_stage3.py \
    train.max_epochs=80 \
    train.use_amp=true \
    data.batch_size=16 \
    data.num_workers=12 \
    data.window_size=16 \
    train.resume=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_mot17_stage1/stage1_best.pth \
    train.output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_mot17_stage3
```

---

### 推奨 Pre-train → Fine-tune フロー

```
Phase A: 公開データで基盤を作る
──────────────────────────────────────────────
[COCO Stage 1]       100 epoch    batch=64
        ↓ 重みロード
[MOT17 Stage 1]       50 epoch    batch=64   (検出を動画ドメインに寄せる)
        ↓ 重みロード
[MOT17 Stage 3]       80 epoch    batch=16   (ID embedding を歩行者で初期化)

Phase B: 自前ハエデータで fine-tune
──────────────────────────────────────────────
[自前 Stage 1]       200 epoch    batch=64   resume=Phase A Stage 1 の重み
        ↓
[自前 Stage 2/3]      80 epoch    batch=16   (並列実行可)
        ↓
[自前 Stage 4]        50 epoch    batch=8
```

```python
# pre-train 重みをロードして自前 Stage 1 を開始する例
import sys
sys.path.insert(0, "src")

from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model
from htrtdetr.utils import load_model_weights

PRETRAIN_CKPT = "C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_mot17_stage1/stage1_best.pth"

cfg = get_variant_config("large", stage=1, overrides={
    "train": {
        "max_epochs": 200,
        "use_amp": True,
        "resume": PRETRAIN_CKPT,
    },
    "data": {"batch_size": 64, "num_workers": 12},
    "optimizer": {
        "lr": 5e-5,               # pre-train 済みなので lr を小さめに
        "backbone_lr_factor": 0.1,
    },
})

model = build_model(cfg.model)
load_model_weights(PRETRAIN_CKPT, model, strict=False)
print("Pre-train weights loaded. Starting fine-tune on fly data...")
```

```bash
# コマンドラインから直接 resume して fine-tune する例
python scripts/train_stage1.py \
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/train/annotations.json \
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/val/annotations.json \
    train.resume=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/pretrain_mot17_stage1/stage1_best.pth \
    train.max_epochs=200 \
    train.use_amp=true \
    data.batch_size=64 \
    data.num_workers=12 \
    optimizer.lr=5e-5 \
    optimizer.backbone_lr_factor=0.1 \
    train.output_dir=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/stage1
```

---

### データセット選択のガイドライン

| 自前データ量 | 推奨戦略 |
|---|---|
| フレーム数 < 5,000 | COCO → MOT17 で pre-train 後、fine-tune |
| フレーム数 5,000〜50,000 | COCO で backbone pre-train のみ後、fine-tune |
| フレーム数 > 50,000 | ImageNet backbone のみで十分な場合が多い |
| ハエ専用データ (DART等) が入手可能 | ドメインが近いため最優先で使用 |

> **注意**: COCO は 80 クラスの汎用物体が含まれるため、`num_classes=1`（ハエのみ）で pre-train すると
> 最終層（クラス分類ヘッド）は fine-tune 時に再初期化が必要になる場合がある。
> `strict=False` で重みをロードすれば、クラス数不一致のレイヤーは自動でスキップされる。
