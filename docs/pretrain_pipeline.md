# HT-RTDETR 汎用 Pre-trained モデル学習パイプライン

> **モデルバリアント**: large (ResNet-50 backbone, feature_dim=512, ~60M params)
> **GPU**: NVIDIA RTX 8000 (VRAM 48GB) × 1
> **リポジトリ**: `C:/Users/utopi/YOAKE`
> **データ・出力先**: `C:/Users/utopi/YOAKE_pre-train`

> **⚠️ パスのカスタマイズ**: `src/htrtdetr/config/config.py` と `scripts/train_stage2.py` 内の `_YOAKE_TRYAL` / `_DEFAULT_ROOT` 変数にハードコードされたパスがある。初回セットアップ時に自分の環境に合わせて変更すること。
> ```python
> # config.py (19行目付近)
> _YOAKE_TRYAL = "C:/Users/<your-name>/YOAKE_pre-train"
> # train_stage2.py (33行目付近)
> _YOAKE_TRYAL = "C:/Users/<your-name>/YOAKE_pre-train"
> ```

---

## 依頼1: データセット選定

### Stage 1 — 物体検出 (backbone + FPN + detector head)

| データセット | 採用 | 採用/不採用理由 | 推奨 epoch 数 | 推定学習時間 (RTX 8000) |
|---|---|---|---|---|
| **COCO** (330K, 80 classes) | ✅ 採用 | 高品質 bbox アノテーション・標準ベンチマーク。backbone 学習の基盤として必須 | 24 epoch | 約 14h |
| **AP-10K** (10K, 54 animal species) | ✅ 採用 | 54種の動物に特化した姿勢・bbox データ。多種生物検出の先行知識として有効 | COCO と混合で同時学習 | 追加コスト小 |
| **Objects365** (2M, 365 classes) | ⚠️ オプション | 大規模・多様性に優れるが、学習コストが高い (単独で +60h)。計算資源に余裕があれば追加 | 12 epoch | 約 60h (単独) |
| OpenImages v7 (9M) | ❌ 不採用 | サイズが巨大すぎて計算コスト非現実的。ラベル品質にばらつきあり | — | — |
| iNaturalist (2M+, 分類のみ) | ❌ 不採用 | **bbox アノテーションなし** (分類データ)。Stage 1 には使用不可 | — | — |
| CrowdHuman (15K) | ⚠️ オプション | 高密度シーン専用の人物データ。生物追跡への汎化性は限定的 | — | — |

**Stage 1 推奨構成**: `COCO + AP-10K` (必須) → 計算資源に余裕があれば `+ Objects365`

---

### Stage 2 — 行動分類 (action head + temporal module)

| データセット | 採用 | 採用/不採用理由 | 推奨 epoch 数 | 推定学習時間 (RTX 8000) |
|---|---|---|---|---|
| **Animal Kingdom** (140+種, 30K clips, 140 actions) | ✅ 採用 | 昆虫含む多種生物の行動ラベル付き動画。Pre-training の主力データ | 15 epoch | 約 25-35h |
| **CalMS21** (マウス社会行動, 7 actions) | ✅ 採用 | 哺乳類の社会行動。密集・相互作用シーンの学習に有効。HDF5 → YOAKE 変換必要 | Animal Kingdom と混合 | 追加コスト小 |
| **MABE22** (Kaggle, マウス行動) | ⚠️ オプション | CalMS21 と類似。既に CalMS21 があれば優先度低 | — | — |
| Fly-vs-Fly | ❌ 不採用 | 現時点で一般公開・入手が不安定。代替として Animal Kingdom の昆虫クリップを活用 | — | — |

**Stage 2 推奨構成**: `Animal Kingdom + CalMS21`
**注**: `SlidingWindowDataset` に `require_action=True` を指定するため、`action_id != -1` のフレームのみが学習に使用される。

---

### Stage 3 — 個体追跡・ReID (ID head + temporal module)

| データセット | 採用 | 採用/不採用理由 | 推奨 epoch 数 | 推定学習時間 (RTX 8000) |
|---|---|---|---|---|
| **DanceTrack** (100動画, 密集・外見類似) | ✅ 採用 | **外見が酷似した個体の追跡**。生物種追跡に最も近い難易度。Track ID 付き | 30 epoch | 約 20-25h |
| **BEE23** (蜂追跡, 昆虫特化) | ✅ 採用 | 昆虫追跡に直接対応。小型・外見均質な個体のReID学習に最適 | DanceTrack と混合 | 追加コスト小 |
| **AnimalTrack** (動物追跡) | ✅ 採用 | 魚・鳥・チーターなど多種の動物Track。汎用性向上に貢献 | DanceTrack と混合 | 追加コスト小 |
| MOT17 (歩行者追跡, 7動画) | ⚠️ オプション | データが少量 (7動画)。標準ベンチマークとして有用だが生物追跡との乖離あり | 混合で利用 | 追加コスト小 |
| MOT20 (高密度歩行者) | ⚠️ オプション | MOT17 の高密度版。追加すると密集シーン耐性が向上 | — | — |
| TAO (multi-category) | ❌ 不採用 | ラベルが疎 (sparse)。annotation品質にばらつきあり | — | — |

**Stage 3 推奨構成**: `DanceTrack + BEE23 + AnimalTrack` (+ MOT17 オプション)

---

### Stage 4 — 統合 Fine-tune (全モジュール)

`bbox + track_id + action_id` の3つすべてを持つ公開データセットはほぼ存在しない。以下の代替手段を推奨:

| 手段 | 内容 |
|---|---|
| **CalMS21 擬似ラベル** (推奨) | CalMS21 は bbox 列 + action ラベルを持つ。連続フレームの bbox から Hungarian matching で `track_id` を自動生成する。変換スクリプトを後述 |
| **Animal Kingdom サブセット** | 一部のクリップは個体追跡可能 (密集度が低いもの)。bbox の時系列から track_id を付与可能 |
| **ユーザー所有データ** | 少量 (数百フレーム) でも bbox + track_id + action が揃っていれば Stage 4 に使用可能 |

---

## 依頼2: 学習パイプライン全体ドキュメント

### 1. 全体フロー図

```
┌─────────────────────────────────────────────────────────────┐
│  公開データセット: COCO (330K) + AP-10K (10K)               │
│  オプション:  + Objects365 (2M)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Stage 1       │  単フレーム入力 (B, 3, H, W)
              │  Detection      │  学習: backbone / FPN / decoder
              │  Pre-training   │  凍結: temporal / ID head / action head
              │  (COCO+AP-10K)  │  損失: cls_loss + bbox_l1 + giou
              └────────┬────────┘
                       │  stage1_best.pth
          ┌────────────┴───────────────┐
          │                            │
┌─────────▼────────────┐  ┌───────────▼──────────────┐
│     Stage 2          │  │      Stage 3              │
│  Action Pre-training │  │  Tracking Pre-training    │
│ (Animal Kingdom      │  │ (DanceTrack + BEE23       │
│  + CalMS21)          │  │  + AnimalTrack)           │
│                      │  │                           │
│ 学習: temporal +     │  │ 学習: temporal + ID head  │
│       action head    │  │ 凍結: detector            │
│ 凍結: detector       │  │ 損失: id_cls + metric     │
│ 損失: action_loss    │  │                           │
│  → stage2_best.pth   │  │  → stage3_best.pth        │
└──────────────────────┘  └───────────────────────────┘
          │                            │
          └────────────┬───────────────┘
                       │  stage3_best.pth (優先), stage2 → stage1 フォールバック
              ┌────────▼────────┐
              │   Stage 4       │  シーケンス入力 (B, T, 3, H, W)
              │  Unified        │  学習: 全モジュール (unfreeze all)
              │  Fine-tune      │  凍結: なし
              │ (CalMS21 擬似   │  損失: cls + bbox + action + id + metric
              │  ラベル等)      │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 汎用 Pre-trained │
              │     モデル       │
              │ stage4_best.pth  │
              └─────────────────┘
                       │
          ┌────────────▼──────────────┐
          │   種固有 Fine-tune         │
          │   (少量データで高精度)     │
          └───────────────────────────┘
```

---

### 2. 環境セットアップ

#### 2-1. ディレクトリ構成の作成

```powershell
# 作業ディレクトリ
$WORK = "C:/Users/utopi/YOAKE_pre-train"

# Stage ごとのデータディレクトリ
New-Item -ItemType Directory -Force -Path "$WORK/data/stage1/train" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage1/val" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage2/train" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage2/val" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage3/train" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage3/val" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage4/train" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/data/stage4/val" | Out-Null

# 出力ディレクトリ
New-Item -ItemType Directory -Force -Path "$WORK/outputs/large/stage1" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/outputs/large/stage2" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/outputs/large/stage3" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/outputs/large/stage4" | Out-Null

# Config 保存先
New-Item -ItemType Directory -Force -Path "$WORK/configs" | Out-Null

# 変換スクリプト用
New-Item -ItemType Directory -Force -Path "$WORK/convert" | Out-Null

# 元データ保存先
New-Item -ItemType Directory -Force -Path "$WORK/raw/coco" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/ap10k" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/animal_kingdom" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/calms21" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/dancetrack" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/bee23" | Out-Null
New-Item -ItemType Directory -Force -Path "$WORK/raw/animaltrack" | Out-Null
```

#### 2-2. データセットのダウンロード

```powershell
# ---- COCO 2017 ----
cd "$WORK/raw/coco"
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile "train2017.zip"
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "val2017.zip"
Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "annotations_trainval2017.zip"
Expand-Archive -Path "train2017.zip" -DestinationPath "." -Force
Expand-Archive -Path "val2017.zip" -DestinationPath "." -Force
Expand-Archive -Path "annotations_trainval2017.zip" -DestinationPath "." -Force

# ---- AP-10K ----
# GitHub: GuanghuiHan/AnimalPose
git clone https://github.com/GuanghuiHan/AnimalPose "$WORK/raw/ap10k"
# またはリポジトリの指示に従ってダウンロード

# ---- Animal Kingdom ----
# 公式: https://sutdcv.github.io/Animal-Kingdom
gdown <FILE_ID> -O "$WORK/raw/animal_kingdom/animal_kingdom.zip"
Expand-Archive -Path "$WORK/raw/animal_kingdom/animal_kingdom.zip" -DestinationPath "$WORK/raw/animal_kingdom" -Force

# ---- CalMS21 ----
# Harvard Dataverse から入手: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FH92BI
# ダウンロード後: "$WORK/raw/calms21/calms21_task1_train.npy" 等

# ---- DanceTrack ----
# GitHub: DanceTrack/DanceTrack
Invoke-WebRequest -Uri "<official_link>" -OutFile "$WORK/raw/dancetrack/dancetrack.zip"
Expand-Archive -Path "$WORK/raw/dancetrack/dancetrack.zip" -DestinationPath "$WORK/raw/dancetrack" -Force

# ---- BEE23 ----
# 公式リポジトリからダウンロード
Invoke-WebRequest -Uri "<official_link>" -OutFile "$WORK/raw/bee23/bee23.zip"
Expand-Archive -Path "$WORK/raw/bee23/bee23.zip" -DestinationPath "$WORK/raw/bee23" -Force

# ---- AnimalTrack ----
# 論文著者配布 (arXiv: 2202.12561)
gdown <FILE_ID> -O "$WORK/raw/animaltrack/animaltrack.zip"
Expand-Archive -Path "$WORK/raw/animaltrack/animaltrack.zip" -DestinationPath "$WORK/raw/animaltrack" -Force
```

#### 2-3. Python 依存パッケージ

```bash
cd C:/Users/utopi/YOAKE
pip install pycocotools h5py scipy tqdm pyyaml
```

> **PyTorch バージョン要件**: Trainer は `torch.amp.GradScaler('cuda')` (PyTorch 2.0+ の API) を使用する。PyTorch 1.x では `torch.cuda.amp.GradScaler()` を使う必要があるため、**PyTorch 2.0 以上**を推奨。
> ```bash
> # バージョン確認
> python -c "import torch; print(torch.__version__)"
> ```

---

### 3. データセット変換スクリプト

> **YOAKE アノテーション形式 v1.1 仕様 (要約)**
>
> - `bbox`: pixel 座標 `[x1, y1, x2, y2]` (left-top, right-bottom)
> - `track_id`: 動画内ユニーク整数 (-1 = 未追跡)
> - `action_id`: 0-indexed (-1 = 未アノテーション)
> - `class_id`: 0-indexed (class_names のインデックス)

#### 3-1. COCO → YOAKE

```powershell
python convert/convert_coco_to_yoake.py `
    --coco_ann C:/Users/utopi/YOAKE_pre-train/raw/coco/annotations/instances_train2017.json `
    --image_root C:/Users/utopi/YOAKE_pre-train/raw/coco/train2017 `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage1/train/coco_annotations.json
```

#### 3-2. AP-10K → YOAKE

```powershell
python convert/convert_ap10k_to_yoake.py `
    --ann C:/Users/utopi/YOAKE_pre-train/raw/ap10k/annotations/ap10k-train-split1.json `
    --image_root C:/Users/utopi/YOAKE_pre-train/raw/ap10k/data `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage1/train/ap10k_annotations.json
```

#### 3-3. COCO + AP-10K のアノテーション統合

Stage 1 では COCO と AP-10K を1つの annotations.json にまとめる。

```powershell
python convert/merge_stage1_annotations.py `
    --inputs `
        C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations.json `
        C:/Users/utopi/YOAKE_pre-train/data/stage1/train/ap10k_annotations.json `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations_merged.json
```

#### 3-4. Animal Kingdom → YOAKE (Stage 2)

```powershell
python convert/convert_animal_kingdom_to_yoake.py `
    --ann_dir C:/Users/utopi/YOAKE_pre-train/raw/animal_kingdom/annotation/AR `
    --frame_dir C:/Users/utopi/YOAKE_pre-train/raw/animal_kingdom/frames `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage2/train/annotations.json `
    --window_size 16 `
    --split train
```

#### 3-5. MOT17/DanceTrack/AnimalTrack → YOAKE (Stage 3)

```powershell
python convert/convert_mot_to_yoake.py `
    --dataset_root C:/Users/utopi/YOAKE_pre-train/raw/dancetrack/train `
    --image_root C:/Users/utopi/YOAKE_pre-train/raw/dancetrack `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage3/train/annotations.json `
    --dataset_name dancetrack
```

#### 3-6. CalMS21 + 擬似 track_id 付与 → YOAKE (Stage 4)

```powershell
python convert/convert_calms21_to_yoake_stage4.py `
    --calms21_npy C:/Users/utopi/YOAKE_pre-train/raw/calms21/calms21_task1_train.npy `
    --output C:/Users/utopi/YOAKE_pre-train/data/stage4/train/annotations.json `
    --window_size 16
```

#### 3-7. 変換実行スクリプト（全 Stage 一括）

> **変換スクリプトの配置**: `convert/` ディレクトリはリポジトリに存在する。上記 3-1〜3-6 のスクリプトをそれぞれのファイル名で保存すること (`convert_coco_to_yoake.py` はすでに配置済み)。
> なお以下の既存ツールも活用できる:
> - `tools/convert_mot17.py` — MOT17 専用変換ツール（Stage 3 に最適）
> - `tools/convert_annotations.py` — CSV/JSON/動画 CSV の汎用変換ツール（独自データ向け）

```powershell
# run_all_conversions.ps1
# C:/Users/utopi/YOAKE から実行する

$REPO = "C:/Users/utopi/YOAKE"
$WORK = "C:/Users/utopi/YOAKE_pre-train"
$PY = "python"

# --- Stage 1: COCO train ---
& $PY "$REPO/convert/convert_coco_to_yoake.py" `
    --coco_ann "$WORK/raw/coco/annotations/instances_train2017.json" `
    --image_root "$WORK/raw/coco/train2017" `
    --output "$WORK/data/stage1/train/coco_annotations.json"

# --- Stage 1: COCO val ---
& $PY "$REPO/convert/convert_coco_to_yoake.py" `
    --coco_ann "$WORK/raw/coco/annotations/instances_val2017.json" `
    --image_root "$WORK/raw/coco/val2017" `
    --output "$WORK/data/stage1/val/annotations.json"

# --- Stage 1: AP-10K ---
& $PY "$REPO/convert/convert_ap10k_to_yoake.py" `
    --ann "$WORK/raw/ap10k/annotations/ap10k-train-split1.json" `
    --image_root "$WORK/raw/ap10k/data" `
    --output "$WORK/data/stage1/train/ap10k_annotations.json"

# --- Stage 1: 統合 ---
& $PY "$REPO/convert/merge_stage1_annotations.py" `
    --inputs `
        "$WORK/data/stage1/train/coco_annotations.json" `
        "$WORK/data/stage1/train/ap10k_annotations.json" `
    --output "$WORK/data/stage1/train/annotations.json"

# --- Stage 2: Animal Kingdom ---
& $PY "$REPO/convert/convert_animal_kingdom_to_yoake.py" `
    --ann_dir "$WORK/raw/animal_kingdom/annotation/AR" `
    --frame_dir "$WORK/raw/animal_kingdom/frames" `
    --output "$WORK/data/stage2/train/annotations.json" `
    --split train

# --- Stage 3: DanceTrack ---
& $PY "$REPO/convert/convert_mot_to_yoake.py" `
    --dataset_root "$WORK/raw/dancetrack/train" `
    --image_root "$WORK/raw/dancetrack" `
    --output "$WORK/data/stage3/train/dancetrack_annotations.json" `
    --dataset_name dancetrack --class_name person

# --- Stage 3: BEE23 ---
& $PY "$REPO/convert/convert_mot_to_yoake.py" `
    --dataset_root "$WORK/raw/bee23/train" `
    --image_root "$WORK/raw/bee23" `
    --output "$WORK/data/stage3/train/bee23_annotations.json" `
    --dataset_name bee23 --class_name bee

# --- Stage 3: AnimalTrack ---
& $PY "$REPO/convert/convert_mot_to_yoake.py" `
    --dataset_root "$WORK/raw/animaltrack/train" `
    --image_root "$WORK/raw/animaltrack" `
    --output "$WORK/data/stage3/train/animaltrack_annotations.json" `
    --dataset_name animaltrack --class_name animal

# --- Stage 3: 統合 ---
& $PY "$REPO/convert/merge_stage1_annotations.py" `
    --inputs `
        "$WORK/data/stage3/train/dancetrack_annotations.json" `
        "$WORK/data/stage3/train/bee23_annotations.json" `
        "$WORK/data/stage3/train/animaltrack_annotations.json" `
    --output "$WORK/data/stage3/train/annotations.json"

# --- Stage 4: CalMS21 ---
& $PY "$REPO/convert/convert_calms21_to_yoake_stage4.py" `
    --calms21_npy "$WORK/raw/calms21/calms21_task1_train.npy" `
    --output "$WORK/data/stage4/train/annotations.json" `
    --window_size 16

Write-Host "All conversions complete."
```

---

### 4. 各ステージの詳細

#### 4-0. YAML Config 生成スクリプト (large variant 共通)

各ステージの設定を YAML に書き出してから学習スクリプトに渡す方式を使う。これにより large variant の次元設定（`fpn_out_channels=512`, `hidden_dim=512` 等）が一貫して適用される。

> **Config 生成前の確認事項**:
> 1. `src/htrtdetr/config/config.py` の `_YOAKE_TRYAL` を自分の環境パスに変更する
> 2. `scripts/train_stage2.py` の `_YOAKE_TRYAL` も同様に変更する
> 3. `setup_pretrain_configs.py` の `WORK` 変数を自分のパスに変更する
>
> **`validate_config` / `clamp_temporal_branches` について**:
> - `validate_config(cfg)` — config の整合性チェック（dimension の一致、window_size の妥当性など）。学習スクリプト起動前に呼び出すと設定ミスを早期発見できる
> - `clamp_temporal_branches(cfg)` — `data.window_size` が small のとき `temporal.long_branch.num_frames` を自動クリップする。`validate_config` 前に呼ぶ
> ```python
> from htrtdetr.config import get_variant_config, validate_config, clamp_temporal_branches
> cfg = get_variant_config("large", stage=2)
> clamp_temporal_branches(cfg)   # window_size < 16 の場合に必要
> validate_config(cfg)           # エラーがあれば ConfigValidationError
> ```

```powershell
cd C:/Users/utopi/YOAKE
python setup_pretrain_configs.py
```

---

#### 4-1. Stage 1: Detection Pre-training

##### (a) 目的・学習対象・凍結モジュール

| 項目 | 内容 |
|---|---|
| **目的** | ResNet-50 backbone + FPN + DETR decoder を多様な物体検出データで鍛える |
| **学習モジュール** | `detector` (backbone + FPN + decoder head) |
| **凍結モジュール** | `temporal`, `id_head`, `action_head` |
| **入力形式** | 単フレーム `(B, 3, H, W)` — `forward_single_frame()` を使用 |
| **損失** | focal classification + L1 bbox + GIoU (Hungarian matching) |
| **データ** | COCO train2017 + AP-10K |
| **評価指標** | val_AP50 (高いほど良い) → `stage1_best.pth` |

##### (b) RTX 8000 最適化パラメーター

```python
# YAML 生成は setup_pretrain_configs.py の make_stage1_config() を参照
# 主要パラメーター (大モデル, RTX 8000 48GB, AMP):

cfg.data.batch_size = 32        # single frame 640x640, ~28-34GB VRAM使用
cfg.data.num_workers = 8
cfg.data.image_size = (640, 640)
cfg.train.use_amp = True
cfg.train.max_epochs = 24
cfg.optimizer.lr = 1e-4
cfg.optimizer.backbone_lr_factor = 0.1   # backbone: 1e-5
cfg.scheduler.warmup_epochs = 3
cfg.model.detector.head.num_queries = 300   # large input → queries を増やす
```

##### (c) 実行コマンド

```powershell
cd C:/Users/utopi/YOAKE

python scripts/train_stage1.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage1.yaml `
    train_anno=C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations.json `
    val_anno=C:/Users/utopi/YOAKE_pre-train/data/stage1/val/annotations.json `
    root=C:/Users/utopi/YOAKE_pre-train
```

> **VRAM OOM 時**: `data.batch_size=24` または `data.batch_size=16` に下げてリトライ

##### (d) 期待される loss 推移

| epoch | train cls_loss | train box_loss | val AP50 | 状態 |
|---|---|---|---|---|
| 1 | ~3.5 | ~2.0 | ~0.05 | 初期収束中 |
| 5 | ~2.2 | ~1.2 | ~0.25 | 急速改善 |
| 10 | ~1.5 | ~0.85 | ~0.40 | 安定改善 |
| 18 | ~1.1 | ~0.65 | ~0.50 | プラトー近く |
| 24 | ~1.0 | ~0.60 | ~0.52 | 収束 |

##### (e) 収束確認・早期終了の判断基準

- **正常収束**: val_AP50 が epoch ごとに改善し、epoch 10-15 で 0.4 超え
- **早期終了**: 連続 10 epoch 改善なし (`early_stopping_patience=10`)
- **異常サイン**:
  - epoch 5 で val_AP50 < 0.1 → lr を `1e-5` に下げて再起動
  - train loss が NaN → AMP の `grad_clip_norm=0.1` を確認、またはバッチサイズを半減

---

#### 4-2. Stage 2: Action Pre-training

##### (a) 目的・学習対象・凍結モジュール

| 項目 | 内容 |
|---|---|
| **目的** | Hierarchical Temporal Module + Action Head を多種生物行動データで鍛える |
| **学習モジュール** | `temporal`, `action_head`, `interaction`, 関連 adapters |
| **凍結モジュール** | `detector` (`freeze_module(self.detector)` で完全凍結) |
| **入力形式** | シーケンス `(B, T, 3, H, W)` — `forward()` を使用 |
| **損失** | cross entropy (action classification, `w_action=1.0`) |
| **データ** | Animal Kingdom + CalMS21 |
| **評価指標** | val_loss (低いほど良い) → `stage2_best.pth` |

> **注**: Stage 2 スクリプトは `clamp_temporal_branches(cfg)` を自動実行するため、`window_size` が `long_branch.num_frames` より小さい場合でも `validate_config` エラーにならない。

##### (b) RTX 8000 最適化パラメーター

```python
cfg.data.batch_size = 8         # シーケンス8本 * 16フレーム = 128フレーム
cfg.data.window_size = 16       # 長期・中期・短期 branch のカバー範囲
cfg.data.window_stride = 8      # オーバーラップありでデータ増強
cfg.data.num_workers = 8
cfg.train.use_amp = True
cfg.train.max_epochs = 15
```

##### (c) 実行コマンド

```powershell
cd C:/Users/utopi/YOAKE

python scripts/train_stage2.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage2.yaml `
    root=C:/Users/utopi/YOAKE_pre-train
```

> Stage 2 スクリプトは `_find_checkpoint()` で以下の順に自動探索する:
> 1. `{root}/outputs/*/stage1/stage1_best.pth` (バリアント別: large / medium / small)
> 2. `{root}/outputs/stage1/stage1_best.pth` (フラット構造・旧形式)
>
> 複数候補がある場合は最終更新日時が最新のものを使用する。見つからない場合やパスを明示したい場合は `train.resume` で直接指定:

```powershell
python scripts/train_stage2.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage2.yaml `
    root=C:/Users/utopi/YOAKE_pre-train `
    train.resume=C:/Users/utopi/YOAKE_pre-train/outputs/large/stage1/stage1_best.pth
```

##### (d) 期待される loss 推移

| epoch | train action_loss | val action_loss | 状態 |
|---|---|---|---|
| 1 | ~3.5 | ~3.3 | 初期 |
| 5 | ~2.1 | ~2.4 | 改善中 |
| 10 | ~1.5 | ~1.8 | 安定 |
| 15 | ~1.3 | ~1.6 | 収束 |

##### (e) 収束確認・早期終了の判断基準

- **正常**: epoch 5-8 で action accuracy (val) が 30-50%
- **要注意**: epoch 5 で val_loss > 3.8 → `w_action` を `0.5` に下げて再試行

---

#### 4-3. Stage 3: Tracking / ReID Pre-training

##### (a) 目的・学習対象・凍結モジュール

| 項目 | 内容 |
|---|---|
| **目的** | Memory-based ID Head + Temporal Module を追跡データで鍛える |
| **学習モジュール** | `temporal`, `id_head` (GRU memory + embedding + classifier) |
| **凍結モジュール** | `detector`, `action_head` |
| **入力形式** | シーケンス `(B, T, 3, H, W)` — `forward()` を使用 |
| **損失** | ID cross entropy (`w_id_cls=1.0`) + triplet metric loss (`w_id_metric=0.5`) |
| **データ** | DanceTrack + BEE23 + AnimalTrack |
| **評価指標** | val_loss (低いほど良い) → `stage3_best.pth` |

##### (b) RTX 8000 最適化パラメーター

```python
cfg.data.batch_size = 8
cfg.data.window_size = 16
cfg.data.window_stride = 8
cfg.data.num_workers = 8
cfg.train.use_amp = True
cfg.train.max_epochs = 30
cfg.model.id_head.max_ids = 100     # 追跡データは多数の track_id を含む
cfg.model.id_head.use_metric_loss = True
cfg.model.id_head.memory_ttl = 30
```

##### (c) 実行コマンド

```powershell
cd C:/Users/utopi/YOAKE

python scripts/train_stage3.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage3.yaml `
    root=C:/Users/utopi/YOAKE_pre-train
```

> Stage 3 スクリプトの `_find_checkpoint()` は以下の順で自動探索する:
> 1. `{root}/outputs/*/stage2/stage2_best.pth` (バリアント別優先)
> 2. `{root}/outputs/stage2/stage2_best.pth` (フラット構造)
> 3. 上記で見つからない場合は同じロジックで stage1 を探索
>
> パスを明示したい場合は `train.resume=<path>` を指定する。

##### (d) 期待される loss 推移

| epoch | train id_loss | train metric_loss | val id_loss | 状態 |
|---|---|---|---|---|
| 1 | ~4.5 | ~0.8 | ~4.3 | 初期 |
| 10 | ~2.8 | ~0.4 | ~3.0 | 改善中 |
| 20 | ~2.0 | ~0.25 | ~2.3 | 安定 |
| 30 | ~1.7 | ~0.20 | ~2.0 | 収束 |

##### (e) 収束確認・早期終了の判断基準

- **正常**: metric_loss が単調減少し、ID re-identification accuracy > 60%
- **要注意**: ID loss が 5 epoch 以上改善なし → `max_ids` を確認 (GT track_id の最大値 < `max_ids` であること)

---

#### 4-4. Stage 4: Unified Fine-tune

##### (a) 目的・学習対象・凍結モジュール

| 項目 | 内容 |
|---|---|
| **目的** | 全モジュールを統合した end-to-end fine-tune で各タスクの相互強化 |
| **学習モジュール** | **全モジュール** (`unfreeze_module(self)`) |
| **凍結モジュール** | なし |
| **入力形式** | シーケンス `(B, T, 3, H, W)` — `forward()` を使用 |
| **損失** | detection + action + ID + temporal smoothing の統合 loss |
| **データ** | CalMS21 (bbox + 擬似 track_id + action_id) |
| **評価指標** | val_loss (低いほど良い) → `stage4_best.pth` |

##### (b) RTX 8000 最適化パラメーター

```python
cfg.data.batch_size = 4         # 全モジュール gradient: VRAM ~38-44GB
cfg.data.window_size = 16
cfg.data.window_stride = 8
cfg.data.num_workers = 8
cfg.train.use_amp = True
cfg.train.max_epochs = 20
cfg.optimizer.lr = 1e-5         # 小さい lr で全体 fine-tune
cfg.optimizer.backbone_lr_factor = 0.1  # backbone: 1e-6
```

##### (c) 実行コマンド

```powershell
cd C:/Users/utopi/YOAKE

python scripts/train_stage4.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage4.yaml `
    root=C:/Users/utopi/YOAKE_pre-train
```

> Stage 4 スクリプトの `_find_checkpoint()` は stage3 → stage2 → stage1 の順でフォールバック探索する。各 stage で:
> 1. `{root}/outputs/*/stage{N}/stage{N}_best.pth` (バリアント別優先)
> 2. `{root}/outputs/stage{N}/stage{N}_best.pth` (フラット構造)
>
> パスを明示したい場合は `train.resume=<path>` を指定する。

##### (d) 期待される loss 推移

| epoch | train total_loss | val total_loss | 状態 |
|---|---|---|---|
| 1 | ~4.0 | ~3.8 | 初期 |
| 5 | ~2.5 | ~2.7 | 改善中 |
| 10 | ~1.8 | ~2.1 | 安定 |
| 20 | ~1.5 | ~1.8 | 収束 |

##### (e) 収束確認・早期終了の判断基準

- **正常**: 全 loss 成分が同時に減少 (いずれか1つが増加し続ける場合は loss weight 調整)
- **要注意**: detection loss が急増 → backbone の lr が高すぎる。`backbone_lr_factor=0.01` に下げる

---

### 5. Stage 間の重み引き継ぎコード

各ステージのスクリプトは `load_model_weights(path, model, strict=False)` で自動的に重みをロードする。以下は手動で重みを引き継ぐ場合のコード:

```powershell
# Stage N の checkpoint から Stage N+1 モデルに重みを転送する
cd C:/Users/utopi/YOAKE
python weight_transfer.py --src C:/Users/utopi/YOAKE_pre-train/outputs/large/stage3/stage3_best.pth --dst_stage 4 --dst C:/Users/utopi/YOAKE_pre-train/outputs/large/stage4/stage4_init.pth
```

**Stage 間転送の要点**:

| 転送方向 | 転送されるモジュール | 再初期化されるモジュール |
|---|---|---|
| Stage 1 → Stage 2 | `detector` 全体 | `temporal`, `action_head`, `interaction` |
| Stage 2 → Stage 3 | `detector`, `temporal`, `query_temporal_fusion` | `id_head` |
| Stage 3 → Stage 4 | 全モジュール | なし (全て引き継ぎ) |

---

### 6. 複数種混在シーン向け設定

`use_species_separated_pools=True` を有効にすると、推論時に種ごとに独立した ID プールを使ってマッチングを行い、異種間の ID 混同を防ぐ。

#### 6-1. Config 設定

```powershell
cd C:/Users/utopi/YOAKE
python multi_species_pretrain_config.py
```

#### 6-2. Pre-training 戦略

複数種混在シーンの Pre-training には、**種ごとにラベルを区別した Stage 1 データ**が必要:

```powershell
# Stage 1: COCO + AP-10K (多クラス) → num_classes を種数に合わせて学習
python scripts/train_stage1.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage1.yaml `
    train_anno=C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations.json `
    val_anno=C:/Users/utopi/YOAKE_pre-train/data/stage1/val/annotations.json `
    root=C:/Users/utopi/YOAKE_pre-train

# Stage 4: 種分離プール有効
python scripts/train_stage4.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage4_multispecies.yaml `
    root=C:/Users/utopi/YOAKE_pre-train
```

> **重要**: `use_species_separated_pools=True` は**推論時のみ**動作する (`forward_inference`)。学習時 (`forward_train`) には影響しない。学習時の種分離は `num_classes=N_SPECIES` の分類ヘッドが担う。

---

### 7. 学習スケジュール全体サマリー

| ステージ | データセット | epoch 数 | 推定時間 (RTX 8000) | チェックポイント |
|---|---|---|---|---|
| Stage 1 | COCO (330K) + AP-10K (10K) | 24 | 約 28-35 時間 | `outputs/large/stage1/stage1_best.pth` |
| Stage 2 | Animal Kingdom (~200K clips) + CalMS21 | 15 | 約 25-35 時間 | `outputs/large/stage2/stage2_best.pth` |
| Stage 3 | DanceTrack + BEE23 + AnimalTrack | 30 | 約 20-28 時間 | `outputs/large/stage3/stage3_best.pth` |
| Stage 4 | CalMS21 擬似ラベル (~50K windows) | 20 | 約 8-12 時間 | `outputs/large/stage4/stage4_best.pth` |
| **合計** | — | — | **約 80-110 時間 (3.5-4.5 日)** | — |

> Stage 2 と Stage 3 は独立して並列実行可能 (どちらも Stage 1 の checkpoint を入力とする)。
> 並列実行時の合計時間は **約 55-75 時間 (2.5-3 日)**。

---

### 8. Fine-tune 手順 (汎用モデル → 種固有モデル)

#### 8-1. 最小データ量の目安

| タスク | 最小データ量 | 推奨データ量 |
|---|---|---|
| 検出のみ (Stage 1 fine-tune) | 200 アノテーション済みフレーム | 1,000+ フレーム |
| 行動分類 (Stage 2 fine-tune) | 50 シーケンス (各16フレーム) | 500+ シーケンス |
| 個体追跡 (Stage 3 fine-tune) | 10 動画 (各100フレーム以上) | 50+ 動画 |
| 統合 (Stage 4 fine-tune) | 20 動画 (bbox + track_id + action 全付き) | 100+ 動画 |

#### 8-2. 種固有 Fine-tune コマンド

```powershell
$WORK = "C:/Users/utopi/YOAKE_pre-train"

# --- Step 1: 種固有データを YOAKE 形式に変換 ---
# (上記の変換スクリプトを使用)

# --- Step 2: 検出 fine-tune (Stage 1) ---
python C:/Users/utopi/YOAKE/scripts/train_stage1.py `
    train_anno="$WORK/data/species/train/annotations.json" `
    val_anno="$WORK/data/species/val/annotations.json" `
    root="$WORK" `
    train.max_epochs=50 `
    train.early_stopping_patience=20 `
    train.use_amp=true `
    train.resume="$WORK/outputs/large/stage1/stage1_best.pth" `
    data.batch_size=16 `
    data.num_workers=8 `
    optimizer.lr=5e-5 `
    optimizer.backbone_lr_factor=0.01

# --- Step 3: 行動 fine-tune (Stage 2) ---
python C:/Users/utopi/YOAKE/scripts/train_stage2.py `
    root="$WORK" `
    train.max_epochs=30 `
    train.use_amp=true `
    data.batch_size=8 `
    data.window_size=16

# --- Step 4: 追跡 fine-tune (Stage 3) ---
python C:/Users/utopi/YOAKE/scripts/train_stage3.py `
    root="$WORK" `
    train.max_epochs=30 `
    train.use_amp=true `
    data.batch_size=8 `
    data.window_size=16

# --- Step 5: 統合 fine-tune (Stage 4) ---
python C:/Users/utopi/YOAKE/scripts/train_stage4.py `
    root="$WORK" `
    train.max_epochs=20 `
    train.use_amp=true `
    data.batch_size=4 `
    data.window_size=16
```

---

### 9. トラブルシューティング

#### 9-1. VRAM OOM (Out of Memory)

```powershell
# VRAM 使用量の確認
nvidia-smi

# 対処: batch_size を半減
# Stage 1: 32 → 16 → 8
python scripts/train_stage1.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage1.yaml `
    train_anno=... val_anno=... `
    data.batch_size=16

# Stage 2/3: 8 → 4
python scripts/train_stage2.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage2.yaml `
    root=C:/Users/utopi/YOAKE_pre-train `
    data.batch_size=4

# Stage 4: 4 → 2 + window_size を小さく
python scripts/train_stage4.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage4.yaml `
    root=C:/Users/utopi/YOAKE_pre-train `
    data.batch_size=2 `
    data.window_size=8
```

> window_size を小さくする場合は `model.temporal.{short,mid,long}_branch.num_frames <= data.window_size` を満たすこと。large 設定では `long_branch.num_frames=16` なので `window_size` は 16 以上が必要。

> OOM で `window_size=8` にする場合は **`clamp_temporal_branches(cfg)`** を使うと branch の `num_frames` を自動的に `window_size` に合わせてクリップできる:
>
> ```python
> from htrtdetr.config import get_variant_config, clamp_temporal_branches, validate_config
> cfg = get_variant_config("large", stage=2)
> cfg = cfg.merge({"data": {"window_size": 8}})
> clamp_temporal_branches(cfg)   # long_branch.num_frames が 8 に自動修正される
> validate_config(cfg)           # エラーなし
> ```
>
> 手動で下げる場合は YAML の `model.temporal.long_branch.num_frames: 8` を指定する。

#### 9-2. Loss 発散 (NaN / Inf)

```powershell
# 対処1: grad_clip_norm を下げる (デフォルト 0.1)
# setup_pretrain_configs.py で optimizer.grad_clip_norm=0.05 に変更して再生成

# 対処2: learning rate を 1/10 に下げる
python scripts/train_stage1.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage1.yaml `
    train_anno=... val_anno=... `
    optimizer.lr=1e-5

# 対処3: warmup epochs を増やす (config 再生成で scheduler.warmup_epochs=10 に)
```

#### 9-3. Stage 1 の AP50 が上がらない (< 0.1 after epoch 10)

1. アノテーション変換を確認: bbox が `[x1, y1, x2, y2]` 形式か（COCO の `[x, y, w, h]` と混同しないこと）
2. 画像パスが正しいか確認: `SingleFrameDataset` の最初の数サンプルを確認

```python
import sys
sys.path.insert(0, "C:/Users/utopi/YOAKE/src")
from htrtdetr.data import SingleFrameDataset, load_annotations

videos, _, _ = load_annotations(
    "C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations.json"
)
ds = SingleFrameDataset(videos[:10], image_size=(640, 640))
sample = ds[0]
print("Image shape:", sample["image"].shape)
print("Boxes:", sample["boxes"][:3])
print("Labels:", sample["labels"][:3])
```

#### 9-4. Stage 2/3 で action/id loss が改善しない

- Stage 2: `SlidingWindowDataset` の `require_action=True` により、`action_id == -1` のフレームを含むウィンドウは除外される。変換後データの `action_id` フィールドが正しく設定されているか確認

```python
import json
with open("C:/Users/utopi/YOAKE_pre-train/data/stage2/train/annotations.json") as f:
    data = json.load(f)

# action_id が -1 以外のフレームがあるか確認
annotated = 0
for vid in data["videos"][:10]:
    for fr in vid["frames"]:
        for obj in fr["objects"]:
            if obj["action_id"] >= 0:
                annotated += 1
print(f"Annotated objects in first 10 videos: {annotated}")
# 0 の場合は変換スクリプトの action_id 設定を確認
```

- Stage 3: `require_track=True` により `track_id == -1` のデータが除外される。MOT 変換後データの `track_id` が正しいか確認

#### 9-5. ダウンロードエラー時の代替手段

| データセット | 代替手段 |
|---|---|
| Animal Kingdom | 著者に直接メール (公式サイト記載のコンタクト). 代替: Kinetics-400 の動物クラスサブセット |
| BEE23 | 著者リポジトリの Issues で入手方法を問い合わせ。代替: MOT17 のみで Stage 3 を進める |
| CalMS21 | Harvard Dataverse から再ダウンロード (登録制)。代替: MABE22 (Kaggle) |

#### 9-6. Stage 間のパス不一致エラー

Stage 2/3/4 スクリプトは `_find_checkpoint()` で以下の順に自動探索する:
1. `{root}/outputs/*/stage{N}/stage{N}_best.pth` (バリアント別: large / medium / small)
2. `{root}/outputs/stage{N}/stage{N}_best.pth` (フラット構造・旧形式)

`large` バリアント用に `outputs/large/stage{N}/` に出力した場合はバリアント別パス (1) が自動検出される。複数のバリアントが同時に存在する場合は最終更新日時が最新のものが使われる。

チェックポイントが見つからない場合やパスを明示したい場合は `train.resume` で指定する:

```powershell
python scripts/train_stage2.py `
    C:/Users/utopi/YOAKE_pre-train/configs/large_stage2.yaml `
    root=C:/Users/utopi/YOAKE_pre-train `
    train.resume=C:/Users/utopi/YOAKE_pre-train/outputs/large/stage1/stage1_best.pth
```

---

## 付録: Config フィールド早見表

| フィールド | デフォルト (large) | Stage 1 推奨 | Stage 2/3 推奨 | Stage 4 推奨 |
|---|---|---|---|---|
| `data.batch_size` | 4 | **32** | **8** | **4** |
| `data.window_size` | 16 | 1 | **16** | **16** |
| `data.num_workers` | 8 | 8 | 8 | 8 |
| `data.image_size` | (640,640) | (640,640) | (640,640) | (640,640) |
| `train.use_amp` | True | True | True | True |
| `train.max_epochs` | 100 | **24** | **15/30** | **20** |
| `optimizer.lr` | 1e-4 | **1e-4** | 1e-4 | **1e-5** |
| `optimizer.backbone_lr_factor` | 0.1 | 0.1 | 0.0 | **0.1** |
| `optimizer.grad_clip_norm` | 0.1 | 0.1 | 0.1 | 0.1 |
| `model.detector.head.num_queries` | 100 | **300** | 100 | 100 |
| `model.id_head.max_ids` | 50 | 50 | **100** | 50 |
| `model.id_head.use_metric_loss` | False | False | **True** | **True** |
| `loss.w_action` | 1.0 | **0.0** | **1.0/0.0** | 1.0 |
| `loss.w_id_cls` | 1.0 | **0.0** | **0.0/1.0** | 1.0 |

---

## 付録: API クイックリファレンス

```python
from htrtdetr.config import (
    get_variant_config,      # バリアント × ステージの設定を取得
    build_model_config,      # バリアントの ModelConfig のみ取得
    validate_config,         # 設定の整合性チェック (エラー時 ConfigValidationError)
    clamp_temporal_branches, # window_size に合わせて temporal branch を自動クリップ
    HTRTDETRConfig,          # トップレベル設定クラス (yaml 入出力対応)
)

# 典型的な使用フロー
cfg = get_variant_config("large", stage=2, overrides={"data": {"batch_size": 4}})
clamp_temporal_branches(cfg)  # OOM で window_size を小さくした場合に必要
validate_config(cfg)          # 問題なければ何もしない
cfg.save_yaml("configs/my_stage2.yaml")

# YAML から再ロード
cfg2 = HTRTDETRConfig.from_yaml("configs/my_stage2.yaml")
```

| 関数 | 用途 |
|---|---|
| `get_variant_config(variant, stage)` | `"small"/"medium"/"large"` × Stage 1-4 の完全な設定を返す |
| `build_model_config(variant)` | `ModelConfig` だけ取得したい場合 |
| `validate_config(cfg)` | 学習前の整合性検証。エラーがあれば `ConfigValidationError` |
| `clamp_temporal_branches(cfg)` | `window_size < long_branch.num_frames` のとき必須。`validate_config` 前に呼ぶ |
| `cfg.merge(overrides)` | dict で部分上書きした新しい config を返す (immutable) |
| `cfg.save_yaml(path)` / `from_yaml(path)` | YAML ファイルとの相互変換 |
