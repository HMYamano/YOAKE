# HT-RTDETR 汎用 Pre-trained モデル学習パイプライン

> **モデルバリアント**: large (ResNet-50 backbone, feature_dim=512, ~60M params)
> **GPU**: NVIDIA RTX 8000 (VRAM 48GB) × 1
> **リポジトリ**: `C:/Users/hayam/YOAKE`
> **データ・出力先**: `C:/Users/hayam/Desktop/YOAKE_tryal`

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
cd C:/Users/hayam/YOAKE
pip install pycocotools h5py scipy tqdm pyyaml
```

---

### 3. データセット変換スクリプト

> **YOAKE アノテーション形式 v1.1 仕様 (要約)**
>
> - `bbox`: pixel 座標 `[x1, y1, x2, y2]` (left-top, right-bottom)
> - `track_id`: 動画内ユニーク整数 (-1 = 未追跡)
> - `action_id`: 0-indexed (-1 = 未アノテーション)
> - `class_id`: 0-indexed (class_names のインデックス)

#### 3-1. COCO → YOAKE

```python
# convert_coco_to_yoake.py
# 実行例 (Anaconda PowerShell):
#   python convert/convert_coco_to_yoake.py `
#       --coco_ann C:/Users/hayam/Desktop/YOAKE_tryal/raw/coco/annotations/instances_train2017.json `
#       --image_root C:/Users/hayam/Desktop/YOAKE_tryal/raw/coco/train2017 `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations.json `
#       --split train

import argparse
import json
from pathlib import Path


def convert_coco(coco_ann_path: str, image_root: str, output_path: str) -> None:
    with open(coco_ann_path, "r") as f:
        coco = json.load(f)

    # id → info マッピング
    img_info = {img["id"]: img for img in coco["images"]}
    # category id → 0-indexed class_id
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_id_to_class = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in sorted(coco["categories"], key=lambda x: x["id"])]

    # image_id → annotations
    from collections import defaultdict
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_img[ann["image_id"]].append(ann)

    videos = []
    for img_id, img in img_info.items():
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue  # 物体なし画像はスキップ

        rel_path = str(Path(image_root) / img["file_name"])
        objects = []
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            objects.append({
                "object_id": i + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "class_id": cat_id_to_class[ann["category_id"]],
                "track_id": -1,
                "action_id": -1,
                "is_crowd": bool(ann.get("iscrowd", 0)),
            })

        videos.append({
            "video_id": f"coco_{img_id:012d}",
            "fps": 1.0,
            "width": img["width"],
            "height": img["height"],
            "num_frames": 1,
            "frames": [{
                "frame_index": 0,
                "image_path": rel_path,
                "width": img["width"],
                "height": img["height"],
                "objects": objects,
            }],
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": "COCO 2017 converted to YOAKE format",
            "created": "2026-03-18",
            "fps_default": 1.0,
            "image_root": image_root,
        },
        "class_names": class_names,
        "action_names": ["none"],
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} entries → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_coco(args.coco_ann, args.image_root, args.output)
```

#### 3-2. AP-10K → YOAKE

```python
# convert_ap10k_to_yoake.py
# AP-10K は COCO keypoint 形式 (bbox あり)
# 実行例 (Anaconda PowerShell):
#   python convert/convert_ap10k_to_yoake.py `
#       --ann C:/Users/hayam/Desktop/YOAKE_tryal/raw/ap10k/annotations/ap10k-train-split1.json `
#       --image_root C:/Users/hayam/Desktop/YOAKE_tryal/raw/ap10k/data `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/ap10k_annotations.json

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_ap10k(ann_path: str, image_root: str, output_path: str) -> None:
    with open(ann_path, "r") as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    cat_ids = sorted(c["id"] for c in data["categories"])
    cat_id_to_class = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]

    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    videos = []
    for img_id, img in img_info.items():
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        rel_path = str(Path(image_root) / img["file_name"])
        objects = []
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            objects.append({
                "object_id": i + 1,
                "bbox": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                "class_id": cat_id_to_class[ann["category_id"]],
                "track_id": -1,
                "action_id": -1,
            })

        videos.append({
            "video_id": f"ap10k_{img_id:08d}",
            "fps": 1.0,
            "width": img["width"],
            "height": img["height"],
            "num_frames": 1,
            "frames": [{
                "frame_index": 0,
                "image_path": rel_path,
                "width": img["width"],
                "height": img["height"],
                "objects": objects,
            }],
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": "AP-10K converted to YOAKE format",
            "created": "2026-03-18",
            "image_root": image_root,
        },
        "class_names": class_names,
        "action_names": ["none"],
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} entries → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_ap10k(args.ann, args.image_root, args.output)
```

#### 3-3. COCO + AP-10K のアノテーション統合

Stage 1 では COCO と AP-10K を1つの annotations.json にまとめる。

```python
# merge_stage1_annotations.py
# 実行例 (Anaconda PowerShell):
#   python convert/merge_stage1_annotations.py `
#       --inputs `
#           C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations.json `
#           C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/ap10k_annotations.json `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations_merged.json

import argparse
import json
from pathlib import Path


def merge_annotations(input_paths: list, output_path: str) -> None:
    all_videos = []
    all_classes = set()
    video_id_set = set()

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls in data.get("class_names", []):
            all_classes.add(cls)
        for v in data["videos"]:
            if v["video_id"] not in video_id_set:
                all_videos.append(v)
                video_id_set.add(v["video_id"])

    class_names = sorted(all_classes)
    result = {
        "meta": {
            "version": "1.1",
            "description": "COCO + AP-10K merged for Stage 1 pretraining",
            "created": "2026-03-18",
        },
        "class_names": class_names,
        "action_names": ["none"],
        "videos": all_videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(all_videos)} videos, {len(class_names)} classes → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_annotations(args.inputs, args.output)
```

#### 3-4. Animal Kingdom → YOAKE (Stage 2)

```python
# convert_animal_kingdom_to_yoake.py
# Animal Kingdom の action segment CSV/JSON をシーケンス形式に変換する
# 公式フォーマット: action_recognition/annotation/ 以下に CSV
# 実行例 (Anaconda PowerShell):
#   python convert/convert_animal_kingdom_to_yoake.py `
#       --ann_dir C:/Users/hayam/Desktop/YOAKE_tryal/raw/animal_kingdom/annotation/AR `
#       --video_dir C:/Users/hayam/Desktop/YOAKE_tryal/raw/animal_kingdom/dataset/AR `
#       --frame_dir C:/Users/hayam/Desktop/YOAKE_tryal/raw/animal_kingdom/frames `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage2/train/annotations.json `
#       --window_size 16 `
#       --split train

import argparse
import json
import os
from pathlib import Path


# Animal Kingdom の action label (一部。実際は 140 クラス)
AK_ACTIONS = [
    "eating", "running", "walking", "swimming", "flying",
    "jumping", "grooming", "fighting", "mating", "resting",
    # ... (実際の AK ラベルに合わせて拡張)
]


def convert_animal_kingdom(
    ann_dir: str,
    frame_dir: str,
    output_path: str,
    window_size: int = 16,
    split: str = "train",
) -> None:
    """
    Animal Kingdom の segment-level アノテーションを YOAKE 形式に変換する。

    Animal Kingdom は各クリップに行動ラベルがつくが bbox は提供されない場合がある。
    bbox なしの場合はフレーム全体を bbox として扱う (x1=0, y1=0, x2=W, y2=H)。
    学習では forward_geo_sequence を使う場合は bbox プロキシで十分。
    """
    ann_csv = Path(ann_dir) / f"{split}.csv"
    if not ann_csv.exists():
        # JSON 形式の場合
        ann_csv = Path(ann_dir) / f"{split}.json"

    if ann_csv.suffix == ".csv":
        import csv
        entries = []
        with open(ann_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    else:
        with open(ann_csv, "r") as f:
            entries = json.load(f)

    # action 名リスト構築 (実際のファイルから収集)
    action_set = set()
    for e in entries:
        action_set.add(e.get("action", e.get("label", "unknown")))
    action_names = sorted(action_set)
    action_to_id = {a: i for i, a in enumerate(action_names)}

    videos = []
    for e in entries:
        vid_id = e.get("video_id", e.get("filename", ""))
        action_name = e.get("action", e.get("label", "unknown"))
        action_id = action_to_id.get(action_name, -1)
        start_f = int(e.get("start_frame", 0))
        end_f = int(e.get("end_frame", start_f + window_size))
        W = int(e.get("width", 640))
        H = int(e.get("height", 480))

        # フレーム画像のパスを構築
        frames = []
        for fi in range(start_f, end_f):
            img_path = str(Path(frame_dir) / vid_id / f"{fi:06d}.jpg")
            frames.append({
                "frame_index": fi - start_f,
                "image_path": img_path,
                "width": W,
                "height": H,
                "objects": [{
                    "object_id": 1,
                    "bbox": [0.0, 0.0, float(W), float(H)],  # 全画面プロキシ
                    "class_id": 0,
                    "track_id": 1,  # 単一個体として仮定
                    "action_id": action_id,
                }],
            })

        videos.append({
            "video_id": f"ak_{vid_id}_{start_f}",
            "fps": float(e.get("fps", 30.0)),
            "width": W,
            "height": H,
            "num_frames": len(frames),
            "frames": frames,
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": "Animal Kingdom converted to YOAKE format",
            "created": "2026-03-18",
        },
        "class_names": ["animal"],
        "action_names": action_names,
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} clips → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    convert_animal_kingdom(args.ann_dir, args.frame_dir, args.output,
                           args.window_size, args.split)
```

#### 3-5. MOT17/DanceTrack/AnimalTrack → YOAKE (Stage 3)

```python
# convert_mot_to_yoake.py
# MOT17 / DanceTrack / AnimalTrack はすべて同一のテキスト形式:
# <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
# 実行例 (Anaconda PowerShell):
#   python convert/convert_mot_to_yoake.py `
#       --dataset_root C:/Users/hayam/Desktop/YOAKE_tryal/raw/dancetrack/train `
#       --image_root C:/Users/hayam/Desktop/YOAKE_tryal/raw/dancetrack `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage3/train/annotations.json `
#       --dataset_name dancetrack

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_mot_dataset(
    dataset_root: str,
    image_root: str,
    output_path: str,
    dataset_name: str = "mot",
    class_name: str = "object",
) -> None:
    """
    MOT 形式テキストアノテーションを YOAKE 形式に変換する。

    dataset_root 以下の各サブディレクトリが1動画に対応することを想定:
      dataset_root/
        seq01/
          gt/gt.txt
          img1/000001.jpg ...
        seq02/
          ...
    """
    sequences = sorted(p for p in Path(dataset_root).iterdir() if p.is_dir())
    videos = []

    for seq_dir in sequences:
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            # DanceTrack は annotations 以下の場合もある
            gt_path = seq_dir / "annotations.txt"
        if not gt_path.exists():
            continue

        # seqinfo.ini から映像情報を読む
        seqinfo_path = seq_dir / "seqinfo.ini"
        width, height, fps = 1920, 1080, 25.0
        img_dir = seq_dir / "img1"
        if seqinfo_path.exists():
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(str(seqinfo_path))
            si = cfg["Sequence"]
            width = int(si.get("imWidth", width))
            height = int(si.get("imHeight", height))
            fps = float(si.get("frameRate", fps))
            img_dir = seq_dir / si.get("imDir", "img1")

        # GT アノテーションを読む
        frame_objects = defaultdict(list)
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                if conf < 0.5:
                    continue  # visibility が低いものを除外
                frame_objects[frame_id].append({
                    "track_id": track_id,
                    "bbox": [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height],
                })

        if not frame_objects:
            continue

        frames = []
        for frame_id in sorted(frame_objects.keys()):
            img_name = f"{frame_id:06d}.jpg"
            img_path = str(img_dir / img_name)
            objects = []
            for oi, obj in enumerate(frame_objects[frame_id]):
                x1, y1, x2, y2 = obj["bbox"]
                objects.append({
                    "object_id": oi + 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "class_id": 0,
                    "track_id": obj["track_id"],
                    "action_id": -1,
                })
            frames.append({
                "frame_index": frame_id - 1,
                "image_path": img_path,
                "width": width,
                "height": height,
                "objects": objects,
            })

        videos.append({
            "video_id": f"{dataset_name}_{seq_dir.name}",
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": len(frames),
            "frames": frames,
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": f"{dataset_name} converted to YOAKE format",
            "created": "2026-03-18",
            "image_root": image_root,
        },
        "class_names": [class_name],
        "action_names": ["none"],
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} sequences → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_name", default="mot")
    parser.add_argument("--class_name", default="object")
    args = parser.parse_args()
    convert_mot_dataset(args.dataset_root, args.image_root, args.output,
                        args.dataset_name, args.class_name)
```

#### 3-6. CalMS21 + 擬似 track_id 付与 → YOAKE (Stage 4)

```python
# convert_calms21_to_yoake_stage4.py
# CalMS21 は HDF5 形式。行動ラベルと bbox (keypoints) を持つ。
# 連続フレームの bbox から track_id を Hungarian matching で自動生成する。
# 実行例 (Anaconda PowerShell):
#   python convert/convert_calms21_to_yoake_stage4.py `
#       --calms21_npy C:/Users/hayam/Desktop/YOAKE_tryal/raw/calms21/calms21_task1_train.npy `
#       --output C:/Users/hayam/Desktop/YOAKE_tryal/data/stage4/train/annotations.json `
#       --window_size 16

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


# CalMS21 task1 action labels
CALMS21_ACTIONS = ["attack", "investigation", "mount", "other"]


def bbox_iou(b1, b2):
    """[x1,y1,x2,y2] 形式の IoU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def assign_track_ids(bboxes_per_frame: list) -> list:
    """
    各フレームの bbox リストから IoU ベースで track_id を割り当てる。
    bboxes_per_frame: List[List[[x1,y1,x2,y2]]]
    returns: List[List[int]] (各フレームの各 bbox に対する track_id)
    """
    next_id = 0
    prev_boxes = []
    prev_ids = []
    all_track_ids = []

    for boxes in bboxes_per_frame:
        if not boxes:
            all_track_ids.append([])
            prev_boxes, prev_ids = [], []
            continue

        if not prev_boxes:
            # 最初のフレーム: 全て新規 ID
            ids = list(range(next_id, next_id + len(boxes)))
            next_id += len(boxes)
        else:
            # IoU コスト行列
            cost = np.zeros((len(boxes), len(prev_boxes)))
            for i, b in enumerate(boxes):
                for j, pb in enumerate(prev_boxes):
                    cost[i, j] = 1.0 - bbox_iou(b, pb)

            row_ind, col_ind = linear_sum_assignment(cost)
            ids = [-1] * len(boxes)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 0.7:  # IoU > 0.3 なら同一個体
                    ids[r] = prev_ids[c]
                else:
                    ids[r] = next_id
                    next_id += 1
            for i in range(len(boxes)):
                if ids[i] == -1:
                    ids[i] = next_id
                    next_id += 1

        all_track_ids.append(ids)
        prev_boxes = boxes
        prev_ids = ids

    return all_track_ids


def convert_calms21(npy_path: str, output_path: str, window_size: int = 16) -> None:
    """
    CalMS21 .npy ファイルを YOAKE Stage 4 形式に変換する。

    CalMS21 task1 形式:
      data[seq_name] = {
        "keypoints": np.ndarray (T, 2, 7, 2),  # 2 animals, 7 keypoints, xy
        "annotations": np.ndarray (T,),          # 0-3 behavior labels
        "metadata": {"fps": ..., ...}
      }
    """
    data = np.load(npy_path, allow_pickle=True).item()

    videos = []
    for seq_name, seq_data in data.items():
        keypoints = seq_data["keypoints"]  # (T, 2, 7, 2) [animal, kpt, xy]
        labels = seq_data.get("annotations", np.full(keypoints.shape[0], -1))
        T = keypoints.shape[0]
        n_animals = keypoints.shape[1]

        # bbox をキーポイントの bounding box として計算
        # keypoints: (T, n_animals, 7, 2) → bbox per animal per frame
        # 座標は pixel 単位と仮定 (実際の CalMS21 は 1024x570 等)
        W, H = 1024, 570  # CalMS21 の標準解像度

        bboxes_per_frame_per_animal = []
        for t in range(T):
            frame_boxes = []
            for a in range(n_animals):
                kpts = keypoints[t, a]  # (7, 2) xy
                x1 = float(np.min(kpts[:, 0])) - 10
                y1 = float(np.min(kpts[:, 1])) - 10
                x2 = float(np.max(kpts[:, 0])) + 10
                y2 = float(np.max(kpts[:, 1])) + 10
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                frame_boxes.append([x1, y1, x2, y2])
            bboxes_per_frame_per_animal.append(frame_boxes)

        # 各動物の track_id を割り当て (動物ごとに独立して処理)
        track_ids_per_animal = []
        for a in range(n_animals):
            boxes_a = [bboxes_per_frame_per_animal[t][a] for t in range(T)]
            tids = assign_track_ids([[b] for b in boxes_a])
            track_ids_per_animal.append([t[0] if t else -1 for t in tids])

        # window_size フレームずつ切り出して video を生成
        for start in range(0, T - window_size + 1, window_size // 2):
            end = start + window_size
            frames = []
            for fi, t in enumerate(range(start, end)):
                action_id = int(labels[t]) if labels[t] >= 0 else -1
                objects = []
                for a in range(n_animals):
                    x1, y1, x2, y2 = bboxes_per_frame_per_animal[t][a]
                    objects.append({
                        "object_id": a + 1,
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2, 2), round(y2, 2)],
                        "class_id": 0,
                        "track_id": track_ids_per_animal[a][t],
                        "action_id": action_id,
                    })
                frames.append({
                    "frame_index": fi,
                    "image_path": f"calms21/{seq_name}/{t:06d}.jpg",
                    "width": W,
                    "height": H,
                    "objects": objects,
                })

            videos.append({
                "video_id": f"calms21_{seq_name}_{start}",
                "fps": float(seq_data.get("metadata", {}).get("fps", 30.0)),
                "width": W,
                "height": H,
                "num_frames": window_size,
                "frames": frames,
            })

    result = {
        "meta": {
            "version": "1.1",
            "description": "CalMS21 with pseudo track_id for Stage 4",
            "created": "2026-03-18",
        },
        "class_names": ["mouse"],
        "action_names": CALMS21_ACTIONS,
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} windows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calms21_npy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    args = parser.parse_args()
    convert_calms21(args.calms21_npy, args.output, args.window_size)
```

#### 3-7. 変換実行スクリプト（全 Stage 一括）

```powershell
# run_all_conversions.ps1
# C:/Users/hayam/YOAKE から実行する

$REPO = "C:/Users/hayam/YOAKE"
$WORK = "C:/Users/hayam/Desktop/YOAKE_tryal"
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

```python
# setup_pretrain_configs.py
# 実行: python C:/Users/hayam/YOAKE/setup_pretrain_configs.py
# 生成先: C:/Users/hayam/Desktop/YOAKE_tryal/configs/

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from htrtdetr.config.config import get_variant_config

WORK = "C:/Users/hayam/Desktop/YOAKE_tryal"


def make_stage1_config():
    cfg = get_variant_config("large", stage=1, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage1/train",
            "val_root": f"{WORK}/data/stage1/val",
            "batch_size": 32,
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 1,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 1,
            "max_epochs": 24,
            "early_stopping_patience": 10,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/large/stage1",
            "log_interval": 100,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.1,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 24,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 1,
            "detector": {
                "head": {
                    "num_classes": 84,   # COCO(80) + AP-10K 追加クラス
                    "num_queries": 300,
                },
            },
            "action_head": {"num_actions": 5},
            "id_head": {"max_ids": 50},
        },
        "loss": {
            "w_class": 2.0,
            "w_bbox_l1": 5.0,
            "w_bbox_giou": 2.0,
            "w_action": 0.0,   # Stage 1: action loss 無効
            "w_id_cls": 0.0,   # Stage 1: ID loss 無効
        },
    })
    cfg.save_yaml(f"{WORK}/configs/large_stage1.yaml")
    print(f"Saved: {WORK}/configs/large_stage1.yaml")


def make_stage2_config():
    cfg = get_variant_config("large", stage=2, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage2/train",
            "val_root": f"{WORK}/data/stage2/val",
            "batch_size": 8,
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 2,
            "max_epochs": 15,
            "early_stopping_patience": 8,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/large/stage2",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.0,   # detector は freeze_module() で制御
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 2,
            "total_epochs": 15,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 2,
            "detector": {
                "head": {
                    "num_classes": 1,    # Stage 2: 行動学習なので汎用1クラスでよい
                    "num_queries": 100,
                },
            },
            "action_head": {
                "num_actions": 140,      # Animal Kingdom の全行動クラス数 (実際に合わせること)
            },
            "id_head": {"max_ids": 50},
        },
        "loss": {
            "w_class": 0.0,     # Stage 2: detection loss 無効
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
            "w_action": 1.0,
            "w_id_cls": 0.0,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/large_stage2.yaml")
    print(f"Saved: {WORK}/configs/large_stage2.yaml")


def make_stage3_config():
    cfg = get_variant_config("large", stage=3, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage3/train",
            "val_root": f"{WORK}/data/stage3/val",
            "batch_size": 8,
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 3,
            "max_epochs": 30,
            "early_stopping_patience": 15,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/large/stage3",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 30,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 3,
            "detector": {
                "head": {"num_classes": 1, "num_queries": 100},
            },
            "action_head": {"num_actions": 5},
            "id_head": {
                "max_ids": 100,
                "use_metric_loss": True,
                "metric_loss_margin": 0.3,
                "memory_ttl": 30,
            },
        },
        "loss": {
            "w_class": 0.0,
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
            "w_action": 0.0,
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/large_stage3.yaml")
    print(f"Saved: {WORK}/configs/large_stage3.yaml")


def make_stage4_config():
    cfg = get_variant_config("large", stage=4, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage4/train",
            "val_root": f"{WORK}/data/stage4/val",
            "batch_size": 4,
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 4,
            "max_epochs": 20,
            "early_stopping_patience": 10,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/large/stage4",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-5,           # 全モジュール fine-tune: 小さい lr
            "backbone_lr_factor": 0.1,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 2,
            "total_epochs": 20,
            "eta_min": 1e-7,
        },
        "model": {
            "training_stage": 4,
            "detector": {
                "head": {"num_classes": 1, "num_queries": 100},
            },
            "action_head": {"num_actions": 4},  # CalMS21 の4クラス
            "id_head": {
                "max_ids": 50,
                "use_metric_loss": True,
                "use_action_summary": True,
                "memory_ttl": 30,
            },
        },
        "loss": {
            "w_class": 1.0,
            "w_bbox_l1": 2.0,
            "w_bbox_giou": 1.0,
            "w_action": 1.0,
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,
            "w_temporal_smooth": 0.1,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/large_stage4.yaml")
    print(f"Saved: {WORK}/configs/large_stage4.yaml")


if __name__ == "__main__":
    import pathlib
    pathlib.Path(f"{WORK}/configs").mkdir(parents=True, exist_ok=True)
    make_stage1_config()
    make_stage2_config()
    make_stage3_config()
    make_stage4_config()
    print("All configs generated.")
```

```powershell
# Config ファイルを生成する
cd C:/Users/hayam/YOAKE
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
cd C:/Users/hayam/YOAKE

python scripts/train_stage1.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage1.yaml `
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations.json `
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/val/annotations.json `
    root=C:/Users/hayam/Desktop/YOAKE_tryal
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

> **注**: Stage 2 スクリプトは `optimizer_cfg` を Trainer に渡さないため、lr は `OptimizerConfig()` デフォルトの `1e-4` が使われる。

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
cd C:/Users/hayam/YOAKE

python scripts/train_stage2.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage2.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal
```

> Stage 2 スクリプトは `f"{root}/outputs/large/stage1/stage1_best.pth"` を自動探索する。
> **注**: デフォルト探索パスは `{root}/outputs/stage1/stage1_best.pth` のため、Stage 1 の出力先と合わせること。
> または `train.resume` で直接指定:

```powershell
python scripts/train_stage2.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage2.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal `
    train.resume=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/large/stage1/stage1_best.pth
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
cd C:/Users/hayam/YOAKE

python scripts/train_stage3.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage3.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal
```

> Stage 3 スクリプトは `stage2_best.pth` → `stage1_best.pth` の順でフォールバック探索する。

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
cd C:/Users/hayam/YOAKE

python scripts/train_stage4.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage4.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal
```

> Stage 4 スクリプトは `stage3_best.pth` → `stage2_best.pth` → `stage1_best.pth` の順で探索する。

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

```python
# weight_transfer.py
# Stage N の checkpoint から Stage N+1 モデルに重みを転送する
# 実行例: python weight_transfer.py --src stage3_best.pth --dst stage4_init.pth

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import torch
from pathlib import Path

from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

WORK = "C:/Users/hayam/Desktop/YOAKE_tryal"


def transfer_weights(src_path: str, dst_stage: int, dst_path: str) -> None:
    """
    src_path の checkpoint を dst_stage のモデルにロードし、dst_path に保存する。

    引き継ぐレイヤー / 初期化するレイヤー:
      Stage 1 → Stage 2: detector (引き継ぎ), temporal/action head (新規初期化)
      Stage 2 → Stage 3: detector + temporal (引き継ぎ), id_head (新規初期化)
      Stage 3 → Stage 4: 全モジュール (引き継ぎ)
    """
    # 転送先モデルを作成
    cfg = get_variant_config("large", stage=dst_stage)
    model = build_model(cfg.model)

    # 転送元 checkpoint を読み込む
    ckpt = torch.load(src_path, map_location="cpu")
    src_state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # strict=False: 形状が一致しないキーはスキップ
    missing, unexpected = model.load_state_dict(src_state, strict=False)
    print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # 保存
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": dst_stage,
        "source": src_path,
    }, dst_path)
    print(f"Saved: {dst_path}")


def show_transferable_keys(src_path: str, dst_stage: int) -> None:
    """転送可能なキーと初期化されるキーを表示する"""
    cfg = get_variant_config("large", stage=dst_stage)
    model = build_model(cfg.model)

    ckpt = torch.load(src_path, map_location="cpu")
    src_state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    dst_keys = set(model.state_dict().keys())
    src_keys = set(src_state.keys())

    transferable = dst_keys & src_keys
    new_keys = dst_keys - src_keys
    removed_keys = src_keys - dst_keys

    # モジュール別に集計
    modules = ["detector", "temporal", "id_head", "action_head", "interaction",
               "query_temporal_fusion", "det_adapter", "feature_router",
               "geo_projector", "geo_action_adapter", "geo_id_adapter"]

    print("\n=== Weight Transfer Summary ===")
    for mod in modules:
        t = [k for k in transferable if k.startswith(mod)]
        n = [k for k in new_keys if k.startswith(mod)]
        print(f"  {mod:30s}: {len(t):4d} transferred, {len(n):4d} re-initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source checkpoint path")
    parser.add_argument("--dst_stage", type=int, required=True, help="Destination stage (1-4)")
    parser.add_argument("--dst", required=True, help="Destination checkpoint path")
    parser.add_argument("--show_keys", action="store_true")
    args = parser.parse_args()

    if args.show_keys:
        show_transferable_keys(args.src, args.dst_stage)
    else:
        transfer_weights(args.src, args.dst_stage, args.dst)
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

```python
# multi_species_pretrain_config.py
import sys
sys.path.insert(0, "C:/Users/hayam/YOAKE/src")

from htrtdetr.config.config import get_variant_config

WORK = "C:/Users/hayam/Desktop/YOAKE_tryal"
N_SPECIES = 3  # 例: ショウジョウバエ / マウス / 魚 の3種混在

cfg = get_variant_config("large", stage=4, overrides={
    "model": {
        "training_stage": 4,
        "detector": {
            "head": {
                "num_classes": N_SPECIES,   # 種ごとにクラスを分ける
                "num_queries": 300,
            }
        },
        "id_head": {
            "max_ids": 100,
            "use_species_separated_pools": True,
            "num_species": N_SPECIES,       # num_classes と一致させること
            "use_metric_loss": True,
        },
        "action_head": {
            "num_actions": 5,
        },
    },
    "data": {
        "batch_size": 4,
        "window_size": 16,
        "num_workers": 8,
    },
    "train": {
        "stage": 4,
        "use_amp": True,
        "max_epochs": 20,
        "output_dir": f"{WORK}/outputs/large/stage4_multispecies",
    },
})

cfg.save_yaml(f"{WORK}/configs/large_stage4_multispecies.yaml")
print("Saved multi-species config.")
```

#### 6-2. Pre-training 戦略

複数種混在シーンの Pre-training には、**種ごとにラベルを区別した Stage 1 データ**が必要:

```powershell
# Stage 1: COCO + AP-10K (多クラス) → num_classes を種数に合わせて学習
python scripts/train_stage1.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage1.yaml `
    train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations.json `
    val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/val/annotations.json `
    root=C:/Users/hayam/Desktop/YOAKE_tryal

# Stage 4: 種分離プール有効
python scripts/train_stage4.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage4_multispecies.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal
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
$WORK = "C:/Users/hayam/Desktop/YOAKE_tryal"

# --- Step 1: 種固有データを YOAKE 形式に変換 ---
# (上記の変換スクリプトを使用)

# --- Step 2: 検出 fine-tune (Stage 1) ---
python C:/Users/hayam/YOAKE/scripts/train_stage1.py `
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
python C:/Users/hayam/YOAKE/scripts/train_stage2.py `
    root="$WORK" `
    train.max_epochs=30 `
    train.use_amp=true `
    data.batch_size=8 `
    data.window_size=16

# --- Step 4: 追跡 fine-tune (Stage 3) ---
python C:/Users/hayam/YOAKE/scripts/train_stage3.py `
    root="$WORK" `
    train.max_epochs=30 `
    train.use_amp=true `
    data.batch_size=8 `
    data.window_size=16

# --- Step 5: 統合 fine-tune (Stage 4) ---
python C:/Users/hayam/YOAKE/scripts/train_stage4.py `
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
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage1.yaml `
    train_anno=... val_anno=... `
    data.batch_size=16

# Stage 2/3: 8 → 4
python scripts/train_stage2.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage2.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal `
    data.batch_size=4

# Stage 4: 4 → 2 + window_size を小さく
python scripts/train_stage4.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage4.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal `
    data.batch_size=2 `
    data.window_size=8
```

> window_size を小さくする場合は config.py の `validate_config()` のチェックに注意:
> `model.temporal.{short,mid,long}_branch.num_frames <= data.window_size` を満たすこと。
> large 設定では `long_branch.num_frames=16` なので `window_size` は 16 以上が必要。
> OOM で window_size=8 にする場合は `model.temporal.long_branch.num_frames=8` も合わせて下げること。

#### 9-2. Loss 発散 (NaN / Inf)

```powershell
# 対処1: grad_clip_norm を下げる (デフォルト 0.1)
# setup_pretrain_configs.py で optimizer.grad_clip_norm=0.05 に変更して再生成

# 対処2: learning rate を 1/10 に下げる
python scripts/train_stage1.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage1.yaml `
    train_anno=... val_anno=... `
    optimizer.lr=1e-5

# 対処3: warmup epochs を増やす (config 再生成で scheduler.warmup_epochs=10 に)
```

#### 9-3. Stage 1 の AP50 が上がらない (< 0.1 after epoch 10)

1. アノテーション変換を確認: bbox が `[x1, y1, x2, y2]` 形式か（COCO の `[x, y, w, h]` と混同しないこと）
2. 画像パスが正しいか確認: `SingleFrameDataset` の最初の数サンプルを確認

```python
import sys
sys.path.insert(0, "C:/Users/hayam/YOAKE/src")
from htrtdetr.data import SingleFrameDataset, load_annotations

videos, _, _ = load_annotations(
    "C:/Users/hayam/Desktop/YOAKE_tryal/data/stage1/train/annotations.json"
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
with open("C:/Users/hayam/Desktop/YOAKE_tryal/data/stage2/train/annotations.json") as f:
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

Stage 2/3/4 スクリプトは **`root` 変数から探索パスを構築**する。本ドキュメントでは `large` バリアント用に `outputs/large/stage{N}/` に出力しているが、スクリプトデフォルトは `outputs/stage{N}/` を探索する:

```python
# train_stage2.py の探索パス (スクリプト内固定)
stage1_path = f"{root}/outputs/stage1/stage1_best.pth"
```

**対処**: `train.resume` で直接パスを指定する:

```powershell
python scripts/train_stage2.py `
    C:/Users/hayam/Desktop/YOAKE_tryal/configs/large_stage2.yaml `
    root=C:/Users/hayam/Desktop/YOAKE_tryal `
    train.resume=C:/Users/hayam/Desktop/YOAKE_tryal/outputs/large/stage1/stage1_best.pth
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
