# Model Card: YOAKE (Hierarchical Temporal RT-DETR)

<!-- このテンプレートを各チェックポイント公開時に埋めてください -->

## Model Description

| 項目 | 内容 |
|------|------|
| **モデル名** | YOAKE-R18 (ResNet-18 backbone) |
| **バージョン** | v0.1.0 |
| **タスク** | Multi-object detection + tracking + action recognition |
| **対象** | ショウジョウバエ (*Drosophila melanogaster*) / 小動物 |
| **入力** | RGB video frames (640×640, T=16 frames) |
| **出力** | bbox, track ID, action label, confidence |
| **ライセンス** | MIT |

---

## Intended Use

### Primary Use Cases
- 実験室環境での小動物行動解析
- 神経科学・行動生物学研究での定量的行動評価
- 高速カメラ映像のリアルタイム処理

### Out-of-Scope Uses
- 屋外・自然環境でのトラッキング (未評価)
- 大型動物への適用 (未調整)
- 人物追跡・監視目的での使用

---

## Training Data

| 項目 | 詳細 |
|------|------|
| **データセット名** | [TODO: データセット名] |
| **規模** | [TODO: 動画数, フレーム数, 個体数] |
| **取得環境** | [TODO: 撮影装置, 照明, fps] |
| **アノテーション** | bbox, track ID (frame-wise), action label |
| **行動クラス** | idle, walk, groom, interact, other |

---

## Evaluation Results

### Detection (val set)

| Metric | Value |
|--------|-------|
| AP50 | TODO |
| AP75 | TODO |
| Recall@50 | TODO |
| Center Error (px) | TODO |

### Tracking (val set)

| Metric | Value |
|--------|-------|
| IDF1 | TODO |
| ID Switches | TODO |
| Fragments | TODO |

### Action Classification (val set)

| Metric | Value |
|--------|-------|
| Frame-wise Accuracy | TODO |
| Macro F1 | TODO |
| idle F1 | TODO |
| walk F1 | TODO |
| groom F1 | TODO |
| interact F1 | TODO |
| other F1 | TODO |

### Runtime (RTX 8000)

| Metric | Value |
|--------|-------|
| Mean FPS | TODO |
| Mean Latency (ms) | TODO |
| GPU Memory (MB) | TODO |

---

## Model Architecture

```
ResNet-18 + FPN + AIFI + DETR Decoder
  + Hierarchical Temporal Module (Short/Mid/Long branches)
  + Memory-based ID Head (GRU)
  + Action Head (MLP)
```

詳細は `docs/architecture.md` を参照。

---

## How to Use

```python
from htrtdetr.config import HTRTDETRConfig
from htrtdetr.models import build_model
from htrtdetr.inference import Inferencer
from htrtdetr.utils import load_model_weights

cfg = HTRTDETRConfig.from_yaml("configs/default.yaml")
model = build_model(cfg.model)
load_model_weights("weights/full_model_best.pth", model)

inferencer = Inferencer(model, cfg.inference, action_names=[...])
results = inferencer.run(input_path="video.mp4", output_dir="outputs/")
```

---

## Limitations

- 重複・接触個体での ID switch 増加
- 行動ラベルの曖昧さ (クラス境界付近での精度低下)
- memory TTL 期間を超えた遮蔽後の再同定は困難
- 学習データと異なる撮影条件での性能低下

---

## Bias and Fairness

- 特定の実験条件・照明環境でのみ評価
- 個体サイズの極端な変化への対応は未検証

---

## Citation

```bibtex
@software{htrtdetr2026,
  title  = {YOAKE: Hierarchical Temporal RT-DETR for Animal Behavior Analysis},
  author = {TODO},
  year   = {2026},
  url    = {TODO},
}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | 2026-03 | Initial release |
