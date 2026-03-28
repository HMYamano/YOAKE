# YOAKE-small 事前学習完全ガイド

> **対象モデル**: yoake-small (ResNet-18 backbone, feature_dim=128, ~15-20M パラメータ)
> **実行環境**: Windows 11 + Anaconda PowerShell Prompt
> **GPU**: NVIDIA RTX 8000 (48GB VRAM) ← 推奨。12GB VRAM 以上で動作可能
> **推定総学習時間**: 約 40〜55 時間 (RTX 8000 単体)

---

## 全体フロー

```
公開データセット: COCO (330K) + AP-10K (10K)
         │
    ┌────▼────┐
    │ Stage 1 │  単フレーム入力 (B, 3, 640, 640)
    │ 検出    │  学習: backbone(ResNet-18) / FPN / decoder
    │ 事前学習 │  損失: cls + bbox_l1 + giou
    └────┬────┘
         │ small_stage1_best.pth
    ┌────┴────────────────┐
    │                     │
┌───▼────┐          ┌────▼────┐
│Stage 2 │          │Stage 3  │
│行動認識│          │トラッキング│
│事前学習│          │事前学習  │
│(AK+   │          │(DanceTrack│
│CalMS21)│          │+AnimalTrack│
└───┬────┘          └────┬────┘
    │                     │ (優先)
    └──────────┬──────────┘
               │
          ┌────▼────┐
          │ Stage 4 │  シーケンス入力 (B, 16, 3, 640, 640)
          │ 統合    │  学習: 全モジュール (unfreeze all)
          │ Fine-tune│  損失: cls + bbox + action + id + metric
          └────┬────┘
               │
        small_stage4_best.pth
        (汎用 pre-trained モデル)
```

---

## 目次

1. [前提条件](#1-前提条件)
2. [環境セットアップ](#2-環境セットアップ)
3. [ディレクトリ作成](#3-ディレクトリ作成)
4. [データセットのダウンロード](#4-データセットのダウンロード)
5. [設定ファイルの生成](#5-設定ファイルの生成)
6. [Stage 1: 検出事前学習](#6-stage-1-検出事前学習)
7. [Stage 2: 行動認識事前学習](#7-stage-2-行動認識事前学習)
8. [Stage 3: トラッキング事前学習](#8-stage-3-トラッキング事前学習)
9. [Stage 4: 統合ファインチューン](#9-stage-4-統合ファインチューン)
10. [最終モデルの確認](#10-最終モデルの確認)
11. [トラブルシューティング](#11-トラブルシューティング)

---

## 1. 前提条件

| 項目 | 要件 |
|------|------|
| GPU | VRAM 12GB 以上 (RTX 8000 48GB 推奨) |
| CUDA | 11.8 以上 |
| Python | 3.10 以上 |
| Anaconda | 最新版 |
| ストレージ | 作業用 500GB 以上 (生データ + 変換済み) |
| ffmpeg | Animal Kingdom フレーム展開に使用 |

### 必要なデータセット

| ステージ | データセット | サイズ | 入手先 |
|---------|------------|--------|--------|
| Stage 1 | COCO 2017 (train/val) | ~20GB | https://cocodataset.org |
| Stage 1 | AP-10K | ~1.7GB | https://github.com/AlexTheBad/AP-10K |
| Stage 2 | Animal Kingdom (AR) | ~35GB | https://sutdcv.github.io/Animal-Kingdom |
| Stage 2 | CalMS21 (task1, .npy) | ~1GB | https://data.caltech.edu/records/s0vdx-0k302 |
| Stage 3 | DanceTrack | ~15GB | https://github.com/DanceTrack/DanceTrack |
| Stage 3 | AnimalTrack | ~3GB | https://github.com/AlexTheBad/AP-10K (別リンク) |
| Stage 4 | CalMS21 (動画フレーム) | ~5GB | 同上 (Stage 2 の .npy と別に動画が必要) |

---

## 2. 環境セットアップ

Anaconda PowerShell Prompt を**管理者として**起動する。

```powershell
# 仮想環境作成 (既存の env があればスキップ)
conda create -n yoake python=3.10 -y
conda activate yoake

# PyTorch のインストール (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# YOAKE パッケージのインストール
cd C:\Users\utopi\YOAKE
pip install -e .

# 追加依存 (CalMS21 変換に必要)
pip install scipy

# インストール確認
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

期待される出力例:
```
CUDA: True | GPU: Quadro RTX 8000
```

---

## 3. ディレクトリ作成

Anaconda PowerShell Prompt で実行する。

```powershell
# 作業ルートを変数に設定
$WORK = "C:\Users\utopi\YOAKE_small_pretrain"

# 生データ格納ディレクトリ
New-Item -ItemType Directory -Force -Path "$WORK\raw\coco\images\train2017"
New-Item -ItemType Directory -Force -Path "$WORK\raw\coco\images\val2017"
New-Item -ItemType Directory -Force -Path "$WORK\raw\coco\annotations"
New-Item -ItemType Directory -Force -Path "$WORK\raw\ap10k\data"
New-Item -ItemType Directory -Force -Path "$WORK\raw\ap10k\annotations"
New-Item -ItemType Directory -Force -Path "$WORK\raw\animal_kingdom\annotation\AR"
New-Item -ItemType Directory -Force -Path "$WORK\raw\animal_kingdom\frames"
New-Item -ItemType Directory -Force -Path "$WORK\raw\calms21"
New-Item -ItemType Directory -Force -Path "$WORK\raw\dancetrack\train"
New-Item -ItemType Directory -Force -Path "$WORK\raw\dancetrack\val"
New-Item -ItemType Directory -Force -Path "$WORK\raw\animaltrack\train"
New-Item -ItemType Directory -Force -Path "$WORK\raw\animaltrack\val"

# 変換済みデータ格納ディレクトリ
foreach ($stage in 1,2,3,4) {
    New-Item -ItemType Directory -Force -Path "$WORK\data\stage$stage\train"
    New-Item -ItemType Directory -Force -Path "$WORK\data\stage$stage\val"
}

# 出力ディレクトリ
foreach ($stage in 1,2,3,4) {
    New-Item -ItemType Directory -Force -Path "$WORK\outputs\small\stage$stage"
}

# 設定ファイルディレクトリ
New-Item -ItemType Directory -Force -Path "$WORK\configs"

Write-Host "ディレクトリ作成完了"
```

---

## 4. データセットのダウンロード

### 4.1 COCO 2017

[cocodataset.org](https://cocodataset.org/#download) からダウンロードして展開する。

```powershell
$WORK = "C:\Users\utopi\YOAKE_small_pretrain"

# 公式サイトからダウンロードした zip を展開する例:
# train2017.zip → $WORK\raw\coco\images\train2017\
# val2017.zip   → $WORK\raw\coco\images\val2017\
# annotations_trainval2017.zip → $WORK\raw\coco\annotations\

# 展開後の構造確認
ls "$WORK\raw\coco\annotations"
# 期待: instances_train2017.json, instances_val2017.json などが存在
```

### 4.2 AP-10K

[GitHub: AlexTheBad/AP-10K](https://github.com/AlexTheBad/AP-10K) から Releases → ap10k.zip をダウンロード。

```powershell
# 展開後の構造:
# $WORK\raw\ap10k\data\          ← 画像ファイル群
# $WORK\raw\ap10k\annotations\   ← ap10k-train-split1.json 等
ls "$WORK\raw\ap10k\annotations"
```

### 4.3 Animal Kingdom

[公式サイト](https://sutdcv.github.io/Animal-Kingdom) から Action Recognition (AR) データをダウンロード。
動画ファイルとアノテーション CSV が含まれる。

```powershell
# 展開後:
# $WORK\raw\animal_kingdom\annotation\AR\train.csv  (または train.json)
# $WORK\raw\animal_kingdom\dataset\AR\<video_id>.mp4
ls "$WORK\raw\animal_kingdom\annotation\AR"
```

### 4.4 CalMS21

[Caltech データポータル](https://data.caltech.edu/records/s0vdx-0k302) から:
- `calms21_task1_train.npy` と `calms21_task1_test.npy` をダウンロード
- 動画フレーム (Stage 4 用): 同ページの動画ファイルもダウンロードして後述の方法でフレーム展開

```powershell
# 配置:
# $WORK\raw\calms21\calms21_task1_train.npy
# $WORK\raw\calms21\calms21_task1_test.npy
ls "$WORK\raw\calms21"
```

### 4.5 DanceTrack

[GitHub: DanceTrack/DanceTrack](https://github.com/DanceTrack/DanceTrack) の README からダウンロードリンクを取得。

```powershell
# 展開後:
# $WORK\raw\dancetrack\train\<seq_name>\img1\000001.jpg ...
#                                        \gt\gt.txt
#                                        \seqinfo.ini
ls "$WORK\raw\dancetrack\train"
```

### 4.6 AnimalTrack

[GitHub](https://github.com/AlexTheBad/AP-10K) 関連リポジトリまたは論文著者ページを参照。
DanceTrack と同じ MOT 形式。

```powershell
# 展開後:
# $WORK\raw\animaltrack\train\<seq_name>\img1\...
#                                         \gt\gt.txt
```

---

## 5. 設定ファイルの生成

```powershell
cd C:\Users\utopi\YOAKE
conda activate yoake

python setup_pretrain_configs_small.py
```

期待される出力:
```
Saved: C:/Users/utopi/YOAKE_small_pretrain/configs/small_stage1.yaml
Saved: C:/Users/utopi/YOAKE_small_pretrain/configs/small_stage2.yaml
Saved: C:/Users/utopi/YOAKE_small_pretrain/configs/small_stage3.yaml
Saved: C:/Users/utopi/YOAKE_small_pretrain/configs/small_stage4.yaml

All yoake-small configs generated.
```

---

## 6. Stage 1: 検出事前学習

### 6.1 COCO → YOAKE 形式に変換

```powershell
$WORK = "C:\Users\utopi\YOAKE_small_pretrain"
cd C:\Users\utopi\YOAKE

# train セット
python convert/convert_coco_to_yoake.py `
    --coco_ann "$WORK\raw\coco\annotations\instances_train2017.json" `
    --image_root "$WORK\raw\coco\images\train2017" `
    --output "$WORK\data\stage1\train\coco_annotations.json"

# val セット
python convert/convert_coco_to_yoake.py `
    --coco_ann "$WORK\raw\coco\annotations\instances_val2017.json" `
    --image_root "$WORK\raw\coco\images\val2017" `
    --output "$WORK\data\stage1\val\coco_annotations.json"
```

### 6.2 AP-10K → YOAKE 形式に変換

```powershell
# train セット
python convert/convert_ap10k_to_yoake.py `
    --ann "$WORK\raw\ap10k\annotations\ap10k-train-split1.json" `
    --image_root "$WORK\raw\ap10k\data" `
    --output "$WORK\data\stage1\train\ap10k_annotations.json"

# val セット
python convert/convert_ap10k_to_yoake.py `
    --ann "$WORK\raw\ap10k\annotations\ap10k-val-split1.json" `
    --image_root "$WORK\raw\ap10k\data" `
    --output "$WORK\data\stage1\val\ap10k_annotations.json"
```

### 6.3 COCO + AP-10K をマージ

```powershell
# train マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage1\train\coco_annotations.json" `
        "$WORK\data\stage1\train\ap10k_annotations.json" `
    --output "$WORK\data\stage1\train\annotations.json"

# val マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage1\val\coco_annotations.json" `
        "$WORK\data\stage1\val\ap10k_annotations.json" `
    --output "$WORK\data\stage1\val\annotations.json"
```

### 6.4 クラス数を確認して設定を更新

マージ後の実際のクラス数を確認する。

```powershell
python -c "
import json
d = json.load(open(r'$WORK\data\stage1\train\annotations.json'))
print('train classes:', len(d['class_names']))
print('train videos:', len(d['videos']))
"

python -c "
import json
d = json.load(open(r'$WORK\data\stage1\val\annotations.json'))
print('val classes:', len(d['class_names']))
print('val videos:', len(d['videos']))
"
```

クラス数が 84 以外だった場合、`setup_pretrain_configs_small.py` の `make_stage1_config(num_classes=実際の数)` に変更して再実行する:

```powershell
# 例: クラス数が 90 だった場合
# setup_pretrain_configs_small.py の make_stage1_config() 呼び出しを変更:
# make_stage1_config(num_classes=90)
python setup_pretrain_configs_small.py
```

### 6.5 Stage 1 学習実行

```powershell
cd C:\Users\utopi\YOAKE
conda activate yoake

python scripts/train_stage1.py `
    "$WORK\configs\small_stage1.yaml" `
    train_anno="$WORK\data\stage1\train\annotations.json" `
    val_anno="$WORK\data\stage1\val\annotations.json"
```

**起動直後に確認すべきログ項目**:

学習が正常に開始されると、以下のようなログが出力される。
特に `train_data_root` / `val_data_root` の値と画像パス検証の結果を必ず確認すること。

```
INFO  - [train] annotation: .../stage1/train/annotations.json
INFO  - [val]   annotation: .../stage1/val/annotations.json
INFO  - train_data_root: 'C:/Users/utopi/YOAKE_small_pretrain/data/stage1/train'
INFO  - val_data_root  : 'C:/Users/utopi/YOAKE_small_pretrain/data/stage1/val'
INFO  - [train] Validating image paths ...
INFO  - [train] 30/30 images found
INFO  - [val]   Validating image paths ...
INFO  - [val]   30/30 images found
INFO  - Train samples: XXXXX | Val samples: YYYYY
```

> **重要**: `train_data_root` と `val_data_root` が**別のパス**を指していることを確認する。
> 同じパスになっている場合、val が train 画像で評価されているため AP50 が正確に計測されない。
>
> `[train] N/30 images found` の N が 0 や極端に少ない場合は学習開始前にエラーで停止する。
> その場合は後述の「[画像パスの検証エラー](#画像パスの検証エラー)」を参照。

**ログ確認**:

```powershell
# 別のターミナルで学習中のログを確認
Get-Content "$WORK\outputs\small\stage1\stage1.log" -Wait -Tail 20
```

**期待される進行** (RTX 8000, batch_size=64):

| エポック | 目安の cls_loss | 目安の AP50 |
|---------|---------------|------------|
| 1〜3    | ウォームアップ中 | - |
| 5〜8    | < 1.0 | ~30% |
| 15〜20  | < 0.5 | ~45-55% |
| 完了時   | < 0.3 | ~50-60% |

完了後:
```
C:\Users\utopi\YOAKE_small_pretrain\outputs\small\stage1\
  stage1_best.pth   ← このファイルが次のステージに引き継がれる
  stage1_last.pth
  stage1.log
  metrics.csv
```

---

## 7. Stage 2: 行動認識事前学習

### 7.1 Animal Kingdom の動画をフレームに展開

Animal Kingdom は動画ファイル (.mp4) で提供される。
ffmpeg でフレームに展開する。

```powershell
# ffmpeg がインストールされていない場合:
# https://ffmpeg.org/download.html からダウンロードして PATH を通す

$AK_VIDEO = "$WORK\raw\animal_kingdom\dataset\AR"
$AK_FRAMES = "$WORK\raw\animal_kingdom\frames"

# 動画一覧を取得してフレーム展開 (並列処理で高速化)
Get-ChildItem "$AK_VIDEO\*.mp4" | ForEach-Object -Parallel {
    $vid = $_.BaseName
    $outdir = "$using:AK_FRAMES\$vid"
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
    ffmpeg -i $_.FullName -q:v 2 "$outdir\%06d.jpg" -hide_banner -loglevel error
} -ThrottleLimit 4

Write-Host "フレーム展開完了"
```

> **注意**: 35GB の動画をフレーム展開すると ~200GB 以上になる場合があります。
> ストレージ容量を事前に確認してください。

### 7.2 Animal Kingdom → YOAKE 形式に変換

```powershell
# train セット
python convert/convert_animal_kingdom_to_yoake.py `
    --ann_dir "$WORK\raw\animal_kingdom\annotation\AR" `
    --frame_dir "$WORK\raw\animal_kingdom\frames" `
    --output "$WORK\data\stage2\train\ak_annotations.json" `
    --window_size 16 `
    --split train

# val セット
python convert/convert_animal_kingdom_to_yoake.py `
    --ann_dir "$WORK\raw\animal_kingdom\annotation\AR" `
    --frame_dir "$WORK\raw\animal_kingdom\frames" `
    --output "$WORK\data\stage2\val\ak_annotations.json" `
    --window_size 16 `
    --split val
```

### 7.3 CalMS21 → YOAKE 形式に変換 (Stage 2 補強用)

CalMS21 は行動認識データとしても使える (4 クラス: attack/investigation/mount/other)。
Stage 2 では Animal Kingdom の補強として任意で追加する。

```powershell
# train
python convert/convert_calms21_to_yoake_stage4.py `
    --calms21_npy "$WORK\raw\calms21\calms21_task1_train.npy" `
    --output "$WORK\data\stage2\train\calms21_annotations.json" `
    --window_size 16

# val
python convert/convert_calms21_to_yoake_stage4.py `
    --calms21_npy "$WORK\raw\calms21\calms21_task1_test.npy" `
    --output "$WORK\data\stage2\val\calms21_annotations.json" `
    --window_size 16
```

### 7.4 アノテーションをマージ

```powershell
# train マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage2\train\ak_annotations.json" `
        "$WORK\data\stage2\train\calms21_annotations.json" `
    --output "$WORK\data\stage2\train\annotations.json"

# val マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage2\val\ak_annotations.json" `
        "$WORK\data\stage2\val\calms21_annotations.json" `
    --output "$WORK\data\stage2\val\annotations.json"
```

### 7.5 アクション数を確認して設定を更新

```powershell
python -c "
import json
d = json.load(open(r'$WORK\data\stage2\train\annotations.json'))
print('action classes:', len(d['action_names']))
print('train windows:', len(d['videos']))
"
```

アクション数が 140 以外だった場合、`setup_pretrain_configs_small.py` の `make_stage2_config(num_actions=実際の数)` を変更して再実行:

```powershell
# setup_pretrain_configs_small.py を編集後
python setup_pretrain_configs_small.py
```

### 7.6 Stage 2 学習実行

Stage 1 の重みは自動的に検索・ロードされる (`outputs/small/stage1/stage1_best.pth`)。

```powershell
python scripts/train_stage2.py `
    "$WORK\configs\small_stage2.yaml" `
    root="$WORK"
```

**ログ確認**:

```powershell
Get-Content "$WORK\outputs\small\stage2\stage2.log" -Wait -Tail 20
```

**期待される進行** (RTX 8000, batch_size=16):

| エポック | 目安の action_loss |
|---------|------------------|
| 1〜2    | ウォームアップ中 |
| 5〜8    | < 2.0 |
| 完了時   | < 1.0 |

完了後:
```
C:\Users\utopi\YOAKE_small_pretrain\outputs\small\stage2\
  stage2_best.pth
  stage2_last.pth
  stage2.log
```

---

## 8. Stage 3: トラッキング事前学習

### 8.1 DanceTrack → YOAKE 形式に変換

```powershell
# train セット
python convert/convert_mot_to_yoake.py `
    --dataset_root "$WORK\raw\dancetrack\train" `
    --image_root "$WORK\raw\dancetrack" `
    --output "$WORK\data\stage3\train\dancetrack_annotations.json" `
    --dataset_name dancetrack `
    --class_name person

# val セット
python convert/convert_mot_to_yoake.py `
    --dataset_root "$WORK\raw\dancetrack\val" `
    --image_root "$WORK\raw\dancetrack" `
    --output "$WORK\data\stage3\val\dancetrack_annotations.json" `
    --dataset_name dancetrack `
    --class_name person
```

### 8.2 AnimalTrack → YOAKE 形式に変換

```powershell
# train セット
python convert/convert_mot_to_yoake.py `
    --dataset_root "$WORK\raw\animaltrack\train" `
    --image_root "$WORK\raw\animaltrack" `
    --output "$WORK\data\stage3\train\animaltrack_annotations.json" `
    --dataset_name animaltrack `
    --class_name animal

# val セット
python convert/convert_mot_to_yoake.py `
    --dataset_root "$WORK\raw\animaltrack\val" `
    --image_root "$WORK\raw\animaltrack" `
    --output "$WORK\data\stage3\val\animaltrack_annotations.json" `
    --dataset_name animaltrack `
    --class_name animal
```

### 8.3 アノテーションをマージ

```powershell
# train マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage3\train\dancetrack_annotations.json" `
        "$WORK\data\stage3\train\animaltrack_annotations.json" `
    --output "$WORK\data\stage3\train\annotations.json"

# val マージ
python convert/merge_stage1_annotations.py `
    --inputs `
        "$WORK\data\stage3\val\dancetrack_annotations.json" `
        "$WORK\data\stage3\val\animaltrack_annotations.json" `
    --output "$WORK\data\stage3\val\annotations.json"
```

### 8.4 データ確認

```powershell
python -c "
import json
d = json.load(open(r'$WORK\data\stage3\train\annotations.json'))
print('train sequences:', len(d['videos']))
"
```

### 8.5 Stage 3 学習実行

Stage 2 の重みが自動検索される。なければ Stage 1 にフォールバック。

```powershell
python scripts/train_stage3.py `
    "$WORK\configs\small_stage3.yaml" `
    root="$WORK"
```

**ログ確認**:

```powershell
Get-Content "$WORK\outputs\small\stage3\stage3.log" -Wait -Tail 20
```

**期待される進行** (RTX 8000, batch_size=16):

| エポック | 目安の id_loss | 目安の metric_loss |
|---------|--------------|------------------|
| 1〜3    | ウォームアップ中 | - |
| 10〜15  | < 1.5 | < 0.5 |
| 完了時   | < 0.8 | < 0.3 |

完了後:
```
C:\Users\utopi\YOAKE_small_pretrain\outputs\small\stage3\
  stage3_best.pth
  stage3_last.pth
  stage3.log
```

---

## 9. Stage 4: 統合ファインチューン

### 9.1 CalMS21 の動画フレームを展開

CalMS21 の動画ファイルから実際のフレーム画像を生成する。
変換スクリプトが生成するパスは `calms21/<seq_name>/<frame_index:06d>.jpg` 形式。

```powershell
# CalMS21 の動画ファイルが $WORK\raw\calms21\videos\ にあると仮定
$CALMS21_VIDEOS = "$WORK\raw\calms21\videos"
$CALMS21_FRAMES = "$WORK\data\stage4\train\calms21"

Get-ChildItem "$CALMS21_VIDEOS\*.mp4" | ForEach-Object -Parallel {
    $seq_name = $_.BaseName
    $outdir = "$using:CALMS21_FRAMES\$seq_name"
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
    ffmpeg -i $_.FullName -q:v 2 "$outdir\%06d.jpg" -hide_banner -loglevel error
} -ThrottleLimit 2

Write-Host "CalMS21 フレーム展開完了"
```

> **CalMS21 にビデオがない場合**: CalMS21 は元々キーポイントデータセットのため、
> キーポイントから合成画像を生成するか、Stage 4 を他のデータで代替することを検討してください。
> Stage 3 の出力 (stage3_best.pth) をそのまま最終モデルとして使用することも可能です。

### 9.2 CalMS21 → YOAKE Stage 4 形式に変換

```powershell
# train
python convert/convert_calms21_to_yoake_stage4.py `
    --calms21_npy "$WORK\raw\calms21\calms21_task1_train.npy" `
    --output "$WORK\data\stage4\train\annotations.json" `
    --window_size 16

# val (test セットを使用)
python convert/convert_calms21_to_yoake_stage4.py `
    --calms21_npy "$WORK\raw\calms21\calms21_task1_test.npy" `
    --output "$WORK\data\stage4\val\annotations.json" `
    --window_size 16
```

### 9.3 データ確認

```powershell
python -c "
import json
d = json.load(open(r'$WORK\data\stage4\train\annotations.json'))
print('Stage 4 train windows:', len(d['videos']))
print('action_names:', d['action_names'])
"
```

期待される出力:
```
Stage 4 train windows: 数千〜数万
action_names: ['attack', 'investigation', 'mount', 'other']
```

### 9.4 Stage 4 学習実行

Stage 3 の重みが自動検索される。なければ Stage 2 → Stage 1 の順にフォールバック。

```powershell
python scripts/train_stage4.py `
    "$WORK\configs\small_stage4.yaml" `
    root="$WORK"
```

**ログ確認**:

```powershell
Get-Content "$WORK\outputs\small\stage4\stage4.log" -Wait -Tail 20
```

**期待される進行** (RTX 8000, batch_size=8):

| エポック | 目安の total_loss |
|---------|-----------------|
| 1〜2    | ウォームアップ中 |
| 5〜10   | 全 loss が低下傾向 |
| 完了時   | 安定した低損失 |

完了後:
```
C:\Users\utopi\YOAKE_small_pretrain\outputs\small\stage4\
  stage4_best.pth   ← これが最終 pre-trained モデル
  stage4_last.pth
  stage4.log
```

---

## 10. 最終モデルの確認

### 10.1 チェックポイントの内容を確認

```powershell
cd C:\Users\utopi\YOAKE
conda activate yoake

python -c "
import torch
ckpt = torch.load(
    r'C:/Users/utopi/YOAKE_small_pretrain/outputs/small/stage4/stage4_best.pth',
    map_location='cpu'
)
print('Keys:', list(ckpt.keys()))
print('Stage:', ckpt.get('stage', 'N/A'))
sd = ckpt.get('model_state_dict', ckpt.get('state_dict', {}))
modules = set(k.split('.')[0] for k in sd.keys())
print('Modules:', sorted(modules))
total_params = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
print(f'Total parameters: {total_params:,}')
"
```

期待される出力例:
```
Keys: ['model_state_dict', 'epoch', 'best_loss', 'stage']
Stage: 4
Modules: ['action_head', 'backbone', 'det_adapter', 'detector', 'feature_router', 'id_head', 'temporal']
Total parameters: 15,000,000〜20,000,000  (yoake-small)
```

### 10.2 モデルのロードテスト

```powershell
python -c "
import sys; sys.path.insert(0, 'src')
import torch
from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model
from htrtdetr.utils import load_model_weights

cfg = get_variant_config('small', stage=4)
model = build_model(cfg.model)
load_model_weights(
    'C:/Users/utopi/YOAKE_small_pretrain/outputs/small/stage4/stage4_best.pth',
    model,
    strict=False
)
model.eval()
print('Model loaded successfully')
print(f'Parameters: {model.num_parameters():,}')

# ダミー入力でフォワードテスト
dummy = torch.randn(1, 16, 3, 640, 640)
with torch.no_grad():
    out = model({'frames': dummy})
print('Forward pass OK')
"
```

### 10.3 各ステージのチェックポイントを一覧確認

```powershell
$WORK = "C:\Users\utopi\YOAKE_small_pretrain"
Get-ChildItem "$WORK\outputs\small" -Recurse -Filter "*.pth" | `
    Select-Object FullName, Length, LastWriteTime | `
    Format-Table -AutoSize
```

---

## 11. トラブルシューティング

### OOM (Out of Memory) エラー

```powershell
# batch_size を半分に下げて再実行
python scripts/train_stage1.py `
    "$WORK\configs\small_stage1.yaml" `
    train_anno="$WORK\data\stage1\train\annotations.json" `
    val_anno="$WORK\data\stage1\val\annotations.json" `
    data.batch_size=32
```

### DummyDataset が使われてしまう

アノテーションファイルのパスを直接指定する:

```powershell
# Stage 1
python scripts/train_stage1.py `
    "$WORK\configs\small_stage1.yaml" `
    train_anno="$WORK\data\stage1\train\annotations.json" `
    val_anno="$WORK\data\stage1\val\annotations.json"

# Stage 2〜4 は configs の train_root が正しければ自動検出される
# configs/small_stage2.yaml の data.train_root を確認:
python -c "
import yaml
with open(r'$WORK\configs\small_stage2.yaml') as f:
    cfg = yaml.safe_load(f)
print('train_root:', cfg['data']['train_root'])
"
```

### 前ステージの重みが見つからない

重みのパスを明示する:

```powershell
# Stage 2 に Stage 1 の重みを明示的に渡す
python scripts/train_stage2.py `
    "$WORK\configs\small_stage2.yaml" `
    root="$WORK" `
    train.resume="$WORK\outputs\small\stage1\stage1_best.pth"
```

### loss が NaN になる

学習率を下げるか、grad_clip を有効化する:

```powershell
python scripts/train_stage1.py `
    "$WORK\configs\small_stage1.yaml" `
    train_anno="$WORK\data\stage1\train\annotations.json" `
    val_anno="$WORK\data\stage1\val\annotations.json" `
    optimizer.lr=1e-5 `
    optimizer.grad_clip_norm=0.05
```

### 途中から再開する (resume)

```powershell
python scripts/train_stage1.py `
    "$WORK\configs\small_stage1.yaml" `
    train_anno="$WORK\data\stage1\train\annotations.json" `
    val_anno="$WORK\data\stage1\val\annotations.json" `
    train.resume="$WORK\outputs\small\stage1\stage1_last.pth"
```

### weight_transfer で明示的に重みを転送する

トレーナーの自動検索を使わず、手動で重みを転送したい場合:

```powershell
# Stage 1 → Stage 2
python weight_transfer.py `
    --src "$WORK\outputs\small\stage1\stage1_best.pth" `
    --dst_stage 2 `
    --dst "$WORK\outputs\small\stage2\stage2_init.pth"

# 転送内容を確認する場合
python weight_transfer.py `
    --src "$WORK\outputs\small\stage1\stage1_best.pth" `
    --dst_stage 2 `
    --dst "$WORK\outputs\small\stage2\stage2_init.pth" `
    --show_keys

# Stage 3 → Stage 4
python weight_transfer.py `
    --src "$WORK\outputs\small\stage3\stage3_best.pth" `
    --dst_stage 4 `
    --dst "$WORK\outputs\small\stage4\stage4_init.pth"
```

---

## 学習時間の目安 (RTX 8000 単体)

| ステージ | epoch数 | 推定時間 |
|---------|--------|--------|
| Stage 1 (batch=64) | 24 | 10〜15 時間 |
| Stage 2 (batch=16) | 15 | 8〜12 時間 |
| Stage 3 (batch=16) | 30 | 12〜18 時間 |
| Stage 4 (batch=8)  | 20 | 8〜12 時間 |
| **合計** | | **38〜57 時間** |

> Stage 2 と Stage 3 は別々のデータセットを使うため、
> Stage 1 完了後に同時に開始することはできません（単一 GPU の場合）。
> 並列化したい場合はマルチ GPU 環境が必要です。

---

## 完了後の次のステップ

最終的に以下のファイルが得られる:

```
C:\Users\utopi\YOAKE_small_pretrain\outputs\small\
  stage1\stage1_best.pth   ← 検出器 pre-trained
  stage2\stage2_best.pth   ← + 行動認識
  stage3\stage3_best.pth   ← + トラッキング
  stage4\stage4_best.pth   ← 汎用 pre-trained モデル (最終成果物)
```

`stage4_best.pth` を種固有のデータで fine-tune することで、
少量データでも高精度な動物行動解析モデルが得られる。
