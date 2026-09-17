# Migration Guide — OBB / track-aligned / streaming update

## AnnotatorのProject形式

Annotatorは複数動画を1つのProjectとして保持します。従来の `annotations.json`
と単一動画の起動方法は引き続き使えます。初回保存時には全動画の学習用JSONに加え、
候補・レビュー・作業位置を復元する `project.yoake.json` と `.yoake/snapshots/` を生成します。
移動時は隠しフォルダ `.yoake` を含めてProjectフォルダ全体をコピーしてください。
動画が見つからなくてもアノテーションは保持され、Project Videosから再リンクできます。

新しい操作手順は [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md)、実装と検証のまとめは
[ANNOTATOR_WORKSPACE_CHANGELOG.md](ANNOTATOR_WORKSPACE_CHANGELOG.md) を参照してください。

このリリースは **既存資産を壊さない**ことを前提に設計されています。
何もしなくても、既存の config・アノテーション・checkpoint・GUI ワークフローは
そのまま動きます。新機能は明示的に有効化したときだけ効きます。

このドキュメントは「何が変わったか」「新機能を使うには何をすればよいか」をまとめます。

---

## TL;DR — 何もしなくてよいこと

| 資産 | 状態 |
|---|---|
| v1.1 アノテーション JSON | **そのまま読める**。変換不要 |
| 既存 YAML config | **そのまま動く**。新しいキーはすべて既定値あり |
| 既存 checkpoint | `load_legacy_checkpoint()` で読める。新規 head だけ random init |
| Stage 1–4 の学習スクリプト | 変更不要 |
| `yoake predict` / `yoake train` / GUI | 変更不要 |
| AABB detector | 既定のまま (`detector.type: rtdetr_aabb`) |

唯一の**挙動の変更**は、推論が既定で streaming になったことです
(`inference.mode: streaming`)。出力される検出結果は従来と同一で、速くなるだけです。
厳密に旧挙動を再現したい場合は `inference.mode=batch` を指定してください。

---

## 1. アノテーション: v1.1 → v2.0

### 変更点

v2.0 で追加された任意フィールド:

```json
{
  "bbox": [x1, y1, x2, y2],
  "obb": {"cx": 100, "cy": 200, "w": 40, "h": 15, "angle": 0.52},
  "keypoints": [[100, 195, 2], [100, 205, 2]],
  "track_id": 3,
  "action_id": 2
}
```

- `obb` は **dict / 5 要素配列 / 4 頂点 8 要素配列** のどれで書いても読めます。
- `bbox` は常に存在し、`obb` がある場合はその**外接矩形**になります。
  これにより bbox しか見ない既存コードが OBB の有無に関係なく動きます。
- `obb` を持たないデータを保存すると、出力は **v1.1 とバイト等価**です
  (旧ツールでもそのまま読めます)。

### 移行方法

移行は**必須ではありません**。明示的に v2.0 として保存したい場合のみ:

```python
from htrtdetr.data.annotation import migrate_annotation_file

report = migrate_annotation_file("old.json", "new_v2.json")
print(report)
```

```
Annotation schema 1.1 -> 2.0
  objects:       12480
  true OBB:      0 (0.0%)
  derived OBB:   0 (theta=0 fallback)
  [WARN] 真の OBB アノテーションが 1 つもありません。...
```

### AABB → OBB の fallback について

OBB を持たない物体には `derive_obb=True` で `theta=0` の代用 OBB を付けられますが、
**これは真の回転アノテーションではありません**。

```python
migrate_annotation_file("old.json", "new.json", derive_obb=True)
```

代用 OBB は `obb_source="derived"` と記録され、

- **角度損失から除外されます** (「すべて水平」を学習させないため)
- **JSON に書き出されません** (情報量がゼロのため)
- 学習前の検証で警告されます

回転検出の性能を出すには、実際に回転アノテーションを付ける必要があります
(GUI の OBB モードを使ってください)。

---

## 2. Config

### 新しいキー (すべて既定値あり)

```yaml
model:
  detector:
    type: rtdetr_aabb        # rtdetr_aabb | rtdetr_obb
    obb:
      enabled: false
      angle_encoding: sincos2
      use_rotated_matching: true
    pose:
      enabled: false         # pose が無いデータセットでも完全に動作
      num_keypoints: 0
    fpn:
      use_p2: false          # stride 4 レベル (小個体向け)
      p2_in_memory: true

  tracking:
    iou_weight: 0.5
    appearance_weight: 0.5
    motion_weight: 0.0       # > 0 で等速度予測を使う
    orientation_weight: 0.0  # > 0 で体軸の一致を使う (OBB 必須)
    use_rotated_iou: true    # OBB が無ければ軸平行に自動 fallback
    max_age: 30
    reid_max_age: 90

  behavior:
    context_scale: 2.0
    classifier_type: single  # single | hierarchical | multi_head
    use_actor: true
    use_context: true
    use_motion: true
    use_orientation: true
    use_interaction: true
    use_temporal: true
    use_pose: false
    temporal_window: 16

  explainability:
    enabled: true
    use_evidence_decomposition: true
    use_gating: true

  track_aligned_temporal: true   # false で旧 query-aligned 挙動 (非推奨)

inference:
  mode: streaming            # streaming | batch
  use_amp: true
  use_channels_last: true
```

### OBB を有効にする最短手順

```python
from htrtdetr.config.config import get_variant_config, enable_obb, validate_config

cfg = get_variant_config("small", stage=1)
enable_obb(cfg, use_p2=True)    # 依存フラグをまとめて整合させる
validate_config(cfg)
```

`enable_obb()` は `detector.type` / `obb.enabled` / `fpn.num_levels` を同時に
整合させます。YAML を手で書くと `num_levels` の更新を忘れやすいので推奨です。

---

## 3. Checkpoint

### 旧 checkpoint を新モデルへ

```python
from htrtdetr.models import build_model
from htrtdetr.utils.misc import load_legacy_checkpoint

model = build_model(cfg.model)
report = load_legacy_checkpoint("runs/old/best.pth", model)
print(report)
```

```
Legacy checkpoint load report
  loaded:            282 tensors
  randomly init'd:   96 (new heads/modules)
  ignored:           0 (removed modules)
  newly initialized (expected for new features):
    detector.pred_head.angle_head.0.weight
    ...
    behavior.actor_encoder.0.weight
    ...
```

- 一致するパラメータはすべて読み込まれます (backbone / FPN / decoder / HTM / ID head)
- 新規 head (角度・pose・behavior) は **明示的に一覧されたうえで** random init
- 形状が変わったパラメータは読み飛ばして報告
- `query_temporal_fusion.*` → `track_temporal_fusion.*` のリネームは自動対応

### 推奨する学習手順 (AABB → OBB への移行)

1. 既存の AABB checkpoint を `load_legacy_checkpoint()` で読む
2. Stage 1 を**低い学習率**で再開し、角度ヘッドだけを学習させる
   (角度ヘッドは `theta=0` で初期化されているので AABB 相当から始まる)
3. Stage 2 以降は通常どおり

---

## 4. 推論モード

### streaming (既定)

各フレームを detector に 1 回だけ通し、過去は track ごとの ring buffer から再利用します。

```bash
yoake predict source=video.mp4 weights=best.pth
```

### batch (旧挙動)

毎フレーム窓全体を detector に通します。数値比較・回帰検証用です。

```bash
yoake predict source=video.mp4 weights=best.pth inference.mode=batch
```

**両者の検出結果 (bbox / score) は厳密に一致します** — streaming が省くのは
「過去フレームの再計算」だけだからです。この一致は
`tests/test_streaming.py::TestStreamingVsBatch::test_detections_match_exactly`
が固定しています。

track ID の付き方は異なりうる点に注意してください: batch は窓ごとに tracker を
作り直すのに対し、streaming は動画全体で tracker を持ち続けます
(streaming の方が長距離の同一性を保ちます)。

### 速度比較

```bash
python tools/benchmark.py mode=streaming windows=4,8,16 num_frames=24
```

```
window=  4: batch   120.20 ms/f | streaming  26.85 ms/f | speedup 4.48x | detector calls 42 -> 12
window=  8: batch   148.38 ms/f | streaming  25.24 ms/f | speedup 5.88x | detector calls 68 -> 12
window= 16: batch   173.23 ms/f | streaming  27.98 ms/f | speedup 6.19x | detector calls 78 -> 12
```

---

## 5. API の変更

### `QueryTemporalFusion` は非推奨

```python
# 旧 (非推奨・DeprecationWarning が出る)
from htrtdetr.models.fusion import QueryTemporalFusion

# 新
from htrtdetr.models.fusion import TrackTemporalFusion
```

DETR の query index はフレーム間で同一個体を保証しないため、query 単位で
時系列を組むと異なる個体の特徴が混ざります。`ModelConfig.track_aligned_temporal=False`
で旧挙動を再現できますが、ablation 比較以外では推奨しません。

### `DetectionPredHead.forward` の戻り値

タプル `(logits, boxes)` から dict へ変わりました
(角度・pose を任意フィールドとして足せるようにするため)。

```python
# 旧
logits, boxes = head(x)

# 新
out = head(x)
logits, boxes = out["logits"], out["boxes"]
angles = out["angles"]        # OBB 無効なら None
```

### `DetectionOutput` / `HTRTDETROutput` の新フィールド

いずれも `Optional` で、既定 (AABB) では `None` です:
`pred_obb`, `pred_angles`, `pred_keypoints`, `pred_polarity`,
`behavior`, `track_ids`, `behavior_rows`

### `model.forward()` の新しい任意引数

```python
out = model(images, memory_list=None, track_ids_per_frame=None)
```

`track_ids_per_frame` は学習時の teacher forcing 用です。省略すると
内部 tracker が track_id を付けるので、既存の呼び出し (`model(images)`) は
そのまま動きます。

---

## 6. GUI

### アノテーションツール

新しい操作 (既存の AABB 操作はすべてそのまま):

| 操作 | 内容 |
|---|---|
| `R` | OBB モード切替 (新規ボックスを回転矩形として作る) |
| `[` / `]` | 選択中のボックスを −5° / +5° 回転 |
| 黄色ハンドルをドラッグ | 自由回転 |
| 右クリック → Convert to rotated box | 既存の箱を OBB へ昇格 |
| 右クリック → Convert to axis-aligned | OBB を外接矩形の AABB へ戻す |

OBB を 1 つも使わないプロジェクトは v1.1 形式のまま保存されます。

### Video Analysis GUI

表示トグル: Individual OBB / Behavior context / Trajectory / Track ID /
Pose / Action / Confidence / Explanation

個体をクリックすると右パネルに判断根拠が出ます:

```
Track ID: 3
Action: courtship
Confidence: 0.82
OBB: 41x12 @ +28.4deg

Evidence
  interaction   +1.10  ##############
  motion        +0.54  #######
  temporal      +0.41  #####
  appearance    +0.32  ####
  context       +0.28  ###
  total         +2.65
```

視覚的な区別: **実線の回転ボックス = 個体検出 (tight)**、
**破線の大きいボックス = 行動認識の context**。
