# OBB / track-aligned / streaming パイプライン

このドキュメントは、今回の改修で導入した設計とその理由をまとめます。
移行手順は [MIGRATION.md](MIGRATION.md)、全体像は [architecture.md](architecture.md) を参照。

---

## 設計の中心にある 1 つの考え

**個体検出と行動認識では、必要な情報の粒度が違う。**

| | 個体検出 | 行動認識 |
|---|---|---|
| 空間表現 | tight な **回転矩形 (OBB)** | 広い **軸平行 context box** |
| 粒度 | 細かい (個体を分離する) | 粗い (状況を含める) |
| 向き | 敏感 (体軸が意味を持つ) | 不変 (回転させると社会的文脈が壊れる) |
| 時間 | 単フレーム | track 単位の履歴 |

この違いをアーキテクチャそのものに反映したのが本パイプラインです。

---

## 1. なぜ OBB なのか

細長い動物が接近して並ぶと、軸平行の箱は互いに大きく重なります。

```
   軸平行 (AABB)                 回転 (OBB)
 ┌───────────────┐            ╱╲        ╱╲
 │ ╱╲        ╱╲  │           ╱  ╲      ╱  ╲
 │╱  ╲      ╱  ╲ │          ╱ A  ╲    ╱ B  ╲
 │╲  ╱      ╲  ╱ │          ╲    ╱    ╲    ╱
 │ ╲╱        ╲╱  │           ╲  ╱      ╲  ╱
 └───────────────┘            ╲╱        ╲╱
   IoU ≈ 0.9                   IoU ≈ 0.05
   → NMS が片方を消す          → 2 個体として残る
   → tracker が ID を混同      → ID が安定
```

`tests/test_geometry_obb.py::test_rotation_separates_close_elongated_boxes` が
この差を数値で固定しています (軸平行 IoU > 0.9 に対し rotated IoU < 0.2)。

### 角度の表現 — なぜ 2 倍角なのか

矩形は `theta` と `theta + π` が**同一の箱**なので、周期は 2π ではなく **π** です。

- 生の `(sin θ, cos θ)` は周期 2π。同じ箱に 2 つの正解ができ、
  境界 (±π/2) で損失が不連続に飛びます。
- `(sin 2θ, cos 2θ)` は周期 π。矩形の周期と一致するので**一意かつ連続**です。

つまり 2 倍角は「周期境界問題の回避策」ではなく、**問題そのものの消去**です。

残る 180° の曖昧さ (頭と尾の区別) は箱の幾何では原理的に決まらないので、
別の責務として `pose head` の head-tail polarity が担当します。

### 表現の相互変換

`geometry/obb.py` が唯一の真実で、4 つの表現を相互変換します:

| 表現 | 形 | 用途 |
|---|---|---|
| `obb` | `(..., 5)` `[cx, cy, w, h, theta]` | 交換用の正準形 |
| `encoded` | `(..., 6)` `[cx, cy, w, h, sin2θ, cos2θ]` | ネットワークの回帰対象 |
| `corners` | `(..., 4, 2)` | 描画・厳密 IoU |
| `aabb` | `(..., 4)` `[x1, y1, x2, y2]` | 既存 AABB 資産との境界 |

torch 非依存の段プレーン用に numpy 版が `stages/_geometry.py` にあり、
`TestTorchNumpyParity` が両者の数値一致 (< 1e-4) を固定しています。

### IoU の 2 系統

| 用途 | 関数 | 理由 |
|---|---|---|
| 評価 / NMS / 追跡 | `rotated_iou` | 凸多角形クリッピングによる**厳密**値 |
| 学習の回帰損失 | `probiou` | 厳密 IoU は場合分けが不連続で勾配が壊れる |

`probiou` は OBB を 2D ガウスとみなし Bhattacharyya 距離から求める微分可能な近似で、
中心・サイズ・角度すべてに滑らかな勾配が流れます。

---

## 2. なぜ track 単位の時間モデルなのか

**DETR の query index はフレーム間で同一個体を保証しません。**

```
  query-aligned (旧・誤り)            track-aligned (新・正しい)

  frame t   : query #17 → 個体 A      frame t   : track 3 → 個体 A
  frame t+1 : query #17 → 個体 B      frame t+1 : track 3 → 個体 A
  frame t+2 : query #17 → 個体 A      frame t+2 : track 3 → 個体 A
              ↓                                   ↓
   A,B,A を 1 本の時系列として          A の履歴だけが時系列になる
   HTM に入力してしまう                 → 速度・加速度・行動遷移が
   → 速度も加速度も無意味                  すべて意味を持つ
```

データフローは必ずこの順です:

```
Detection → Tracking → track_id → track-aligned history → HTM
```

`tests/test_tracking.py::test_history_groups_by_track_not_position` が、
検出の並び順を毎フレーム入れ替えても履歴が track ごとに正しく揃うことを固定しています。

### 欠損フレームの扱い

その track が検出されなかったフレームは `mask=False` で**明示**します。
無言のゼロ埋めや前フレーム複製をすると「動きが止まった」ように見え、
行動認識を誤らせるためです。埋め方は設定可能です:

- `interpolate` (既定) — 前後の有効フレームから線形補間
- `hold` — 直前の有効フレームを保持
- `zero` — ゼロ埋め (mask で下流に伝える)

---

## 3. streaming 推論 — 何を直したか

旧実装は 16 フレーム窓を 1 フレームずつずらすたびに、窓内の**全フレームを再び
detector へ通していました**。

```
旧: frame 16 の推論 → detector(1..16)   16 回の空間推論
    frame 17 の推論 → detector(2..17)   16 回 (うち 15 回は再計算)
    frame 18 の推論 → detector(3..18)   ...

新: frame 16 → detector(16) → ring buffer へ push → HTM は buffer を読むだけ
    frame 17 → detector(17) → 〃
    frame 18 → detector(18) → 〃
```

同じフレームが最大 `window_size` 回 detector を通っていたので、
空間推論のコストが窓長ぶん丸ごと重複していました。

### 実測 (CPU, 160px, 12 frames)

| window | batch (旧) | streaming (新) | speedup | detector 呼び出し |
|---|---|---|---|---|
| 4 | 120.2 ms/f | 26.9 ms/f | **4.48×** | 42 → 12 |
| 8 | 148.4 ms/f | 25.2 ms/f | **5.88×** | 68 → 12 |
| 16 | 173.2 ms/f | 28.0 ms/f | **6.19×** | 78 → 12 |

streaming 側の時間が window に依存しないことが要点です
(旧方式は window に比例して増える)。

```bash
python tools/benchmark.py mode=streaming windows=4,8,16 num_frames=24
```

### 一致の保証

**検出結果 (bbox / score) は batch と厳密に一致します**。
streaming が省くのは「過去フレームの再計算」だけで、各フレームに対する
detector の計算そのものは同一だからです。

track ID の付き方だけは異なりえます: batch は窓ごとに tracker を作り直しますが、
streaming は動画全体で tracker を持ち続けます (streaming の方が長距離の同一性を保つ)。

---

## 4. 2-stream behavior recognition

```
        個体 OBB
       ╱        ╲
      ╱          ╲
 actor stream   context stream
      │              │
 OBB に沿って     外接 AABB を
 回転 crop        context_scale 倍
      │              │
 体軸が横向きに   周辺の他個体を
 揃った ROI       含む軸平行 ROI
      │              │
 細かい appearance  social context
 と姿勢            と相対位置
```

**actor は回転させ、context は回転させない。** これは意図的な非対称です:

- actor を回転正規化すると、同じ姿勢の個体が向きに依らず同じ特徴に写ります。
- context を回転させると「相手がどちら側にいるか」という情報が失われ、
  courtship / chase の判別に必要な社会的文脈が壊れます。

`tests/test_behavior.py::test_context_roi_is_axis_aligned` がこれを固定しています。

### Kinematics

行動は見た目だけでは決まりません。静止画では groom と idle が区別できず、
courtship と chase は「誰が誰をどう追っているか」で決まります。

| グループ | 特徴 |
|---|---|
| MOTION | 速度 / 加速度 / 変位 / 経路長 / 直進度 / 移動方向 |
| ORIENTATION | 体軸方向 / 角速度 / **体軸と移動方向のなす角** / 伸長度 |
| INTERACTION | 最近傍距離 / 相対速度 / 接近率 / **facing A→B** / **facing B→A** / 重なり |

`facing_to_nn` と `faced_by_nn` を**別次元**にしているのが要点です。
「A が B を追う」と「B が A を追う」は行動として非対称なので、
対称な特徴では原理的に区別できません。

---

## 5. Evidence-based explainability

説明は後付けの可視化ではなく、**モデル構造そのもの**です。

```
action_logits = Σ_g  gate_g · head_g(feature_g)

              = appearance 寄与
              + context 寄与
              + motion 寄与
              + pose 寄与
              + interaction 寄与
              + temporal 寄与
```

各 feature group が独立に logit へ寄与し、**足し合わせると最終 logit に一致**します
(`tests/test_behavior.py::test_evidence_sums_to_logits` が厳密に固定)。

したがって GUI に出る

```
Appearance    +0.32
Motion        +0.54
Interaction   +1.10
Temporal      +0.41
Context       +0.28
```

は近似ではなく、予測そのものの分解です。

`gate` (重要度) と `実際の寄与` の**両方**を取得できるので、
「重視しているが今回は寄与が小さい」といった区別もできます。

```python
out = model(images)
out.explain_detection(batch_index=0, det_index=2)   # {"motion": 0.54, ...}
out.behavior_gates(batch_index=0, det_index=2)      # {"motion": 0.81, ...}
```

---

## 6. 3 つの行動分類方式

| `classifier_type` | 構造 | 使いどころ |
|---|---|---|
| `single` | 単一 softmax | 既存互換。相互排他な行動セット |
| `hierarchical` | 木構造 (条件付き確率の積) | 粗い区別は学習しやすく、細かい区別は該当サンプルだけで学ぶ。不均衡に強い |
| `multi_head` | 属性ごとに独立 head | 「歩きながら求愛する」のような同時成立を表現できる |

```yaml
behavior:
  classifier_type: multi_head
  heads:
    locomotion:    [stationary, walking, running]
    self_directed: [none, grooming, feeding]
    social:        [none, courtship, chase, aggression]
```

multi-head では GT が欠けている head は自動的に学習対象から外れるので、
「locomotion だけラベルがある」ような部分アノテーションでも学習できます。

---

## 7. Ablation

研究用途なので、各構成要素を config から切れます。**切ると本当に
パラメータが消えます** (ゼロ埋めではない) — そうでないと公平な比較になりません。

```yaml
model:
  detector:
    fpn: {use_p2: false}          # P2
    obb: {enabled: false}         # OBB
  tracking:
    orientation_weight: 0.0       # 向き一致
    motion_weight: 0.0            # 運動予測
  behavior:
    use_actor: false              # actor stream
    use_context: false            # context stream
    use_motion: false             # kinematics
    use_orientation: false
    use_interaction: false
    use_temporal: false           # HTM
    use_pose: false
  explainability:
    use_evidence_decomposition: false   # evidence gating
  track_aligned_temporal: false   # query-aligned に戻す (非推奨)
```

`tests/test_behavior.py::test_ablation_removes_group_and_params` が、
各フラグを落とすとパラメータ数が実際に減ることを固定しています。

---

## 8. 評価指標

| 領域 | 指標 |
|---|---|
| Detection | `rAP50` / `rAP75` / `rmAP` (rotated IoU), `AP50_small` / `_medium` / `_large` |
| Tracking | `MOTA`, `IDF1`, `HOTA`, `DetA`, `AssA`, `ID_switches`, `ID_fragments` |
| Behavior | `accuracy`, `balanced_accuracy`, `macro_f1`, per-class P/R/F1, confusion matrix |
| Multi-head | head ごとの上記 + `mean_macro_f1` |
| Performance | 段別 latency (preprocess / detector / tracking / temporal / behavior), FPS, VRAM |

**HOTA は検出精度と関連付け精度を分離します**:

```
perfect    : IDSW  0  DetA 1.00  AssA 1.00  HOTA 1.00
one swap   : IDSW  2  DetA 1.00  AssA 0.33  HOTA 0.58
thrashing  : IDSW 10  DetA 1.00  AssA 0.33  HOTA 0.58
```

検出が完璧でも関連付けが壊れていれば HOTA が下がります。
(旧 `TrackingEvaluator` は ID switch を 0 固定にしていたため、この区別ができませんでした。)

```python
from htrtdetr.evaluation import MOTEvaluator, RotatedDetectionEvaluator
```
