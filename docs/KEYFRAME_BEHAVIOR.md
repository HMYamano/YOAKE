# キーフレーム行動ラベリング — YOWO 型の再設計計画

> ブランチ: `keyframe-behavior`。この文書は実装の**前に**書いた計画であり、実装はこれに従う。
> 末尾の §11 が進捗チェックリスト、§12 が利用者に確認したい未決事項。

---

## 0. 要約

人がラベルを付けるのは **「動画のどのフレームか」と「そのフレームに写っている各個体の箱と行動」だけ**にする。

* **track_id を付けない。** 個体の同一性は、解析時にトラッキングアルゴリズムが検出結果から作る。
* **区間を付けない。** 行動の始まりと終わりは、モデルがフレームごとに出す行動確率を track に沿って
  復号 (Viterbi) して得る。
* **時間情報はタダで手に入る。** キーフレームは動画上の位置なので、その直前のクリップを動画から
  切り出せば、ラベルの追加コストなしに時系列の文脈が得られる。これが YOWO
  ([Köpüklü+ 2019](https://arxiv.org/abs/1911.06644)) の中核で、本計画はそれを
  「多個体・細長い動物・社会行動」に合わせて組み替える。

一言でいえば **「ラベルは 1 枚の画像、学習と解析は動画」**。

---

## 1. 背景 — 今の設計の何が重いのか

現状の YOAKE は、行動を**区間** (`annotation/intervals.py` の `BehaviorInterval`) として編集し、
保存時に「track_id 付きの各フレームの `action_id`」へ展開している。この形には 4 つのコストがある。

| | 何が起きるか |
|---|---|
| ID の維持 | 交尾中の 2 匹は 1 つの塊に見え、離れた瞬間に ID が入れ替わる。区間ラベルは ID に紐づくので、ID の修正 (`swap_tracks`) がラベル作業の前提条件になる。 |
| 区間の境界 | 「求愛がいつ始まったか」は人によって ±数百 ms ずれる。境界の判断が最も時間のかかる作業になる。 |
| 学習データの前提 | `SlidingWindowDataset` は「注釈済みフレームのリスト上で連続する T 個」を窓にする。**動画上で連続しているかは見ていない**ので、疎なラベルを渡すと 10 秒離れたフレームを 1 本の時系列として学習する (静かに壊れる)。 |
| Stage 3 | ID head の学習には全フレームの identity ラベルが要る。 |

さらに、この計画のために学習経路を読んでいて**潜在バグ**を見つけた (§6.1)。
学習中の内部トラッカーは appearance embedding を渡されないのに、その重みが類似度の分母に
入ったままなので、類似度が `0.5 × IoU` で頭打ちになり、閾値 0.5 を超えない。
結果として **学習中はどの検出もフレーム間で連結されず、track 単位の履歴は実質 1 フレーム**だった:

```
既定設定・embedding なしで 2 匹を 6 フレーム追跡した結果 (実測):
  [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]   ← 毎フレーム新しい ID
```

推論 (streaming) では embedding を渡すので連結される。つまり**学習時と推論時で時間特徴の意味が
違っていた**。キーフレーム学習は「ラベルの無い文脈フレームをトラッカーで連結して時間特徴を作る」
ことに依存するので、これは必ず直す。

---

## 2. コンセプト

### 2.1 YOWO から借りるもの

YOWO は 1 段の時空間行動検出器で、

* **キーフレーム** (クリップの最終フレーム) を 2D CNN で、
* **キーフレームまでのクリップ**を 3D CNN で見て、
* 両者を CFAM で融合し、キーフレーム上の箱と行動クラスを出す。

学習に要るのはキーフレームのラベルだけで、文脈フレームにはラベルが要らない。
フレーム単位の検出は後処理で連結して行動チューブにする。

### 2.2 YOAKE が組み替えるもの

| | YOWO | YOAKE (本計画) |
|---|---|---|
| 対象 | 人 (UCF101-24, J-HMDB, AVA) | 外見がほぼ同一の小動物が複数、接触・重なりあり |
| 箱 | 軸平行 | 回転矩形 (OBB) — 細長い個体の分離 (既存設計) |
| 時間文脈 | フレーム全体の 3D 特徴 | **2 系統**: ① 個体に沿った文脈 (トラッカーで連結した track 履歴 → HTM・運動学)、② 個体に依らない文脈 (キーフレームの context box をクリップ全体に固定した "tube" 特徴 = YOWO の 3D 分岐に相当) |
| 融合 | CFAM (channel attention) | 加法的 evidence 分解 `Σ_g gate_g · head_g` — どちらの文脈が効いたかが予測そのものの分解として出る (既存不変条件) |
| 同一性 | 箱の連結 (チューブ化) | OBB 対応オンライントラッカー (回転 IoU + 運動 + 向き)。**ラベル不要** |
| 時系列化 | チューブ単位の平均 | track ごとの HMM/Viterbi 復号 → 行動 bout、時間配分、遷移、潜時 |
| ラベル付け | 既存データセット | キーフレームを選んで箱と行動だけ付ける GUI (文脈ループ再生つき)、能動学習ループ |

### 2.3 データの流れ

```
 [ラベル付け]  動画 + フレーム番号 + {個体の OBB, 行動}   ← 人がするのはここだけ
                       │
                       ▼
 [学習]  キーフレーム t の直前クリップ  t-(T-1)s, …, t-s, t   (動画から切り出す・ラベル不要)
         ├─ detector (各フレーム 1 回)            → 検出
         ├─ tracker (クリップ内で連結)              → track 履歴 (ラベル不要)
         ├─ actor / context / 運動学 / HTM          → 個体に沿った evidence
         ├─ clip tube (context box を T フレームに固定) → 個体に依らない evidence
         └─ 損失はキーフレームの検出と行動だけ
                       │
                       ▼
 [解析]  全フレームを streaming 推論 (各フレームを detector に 1 回だけ通す)
         → 検出 + track_id + 行動確率 (フレームごと)
         → track ごとに Viterbi 復号 → bout → 統計 / エソグラム
```

---

## 3. 先行研究と新規性

### 3.1 位置づけ

| 手法 | ラベルの単位 | ID ラベル | 時間文脈 | 個体別 | 備考 |
|---|---|---|---|---|---|
| YOWO / YOWOv2 / v3 | キーフレームの箱+行動 | 不要 | クリップ全体の 3D 特徴 | 箱の位置でのみ | 人。軸平行 |
| AVA (+SlowFast 系) | 1 Hz のキーフレームの箱+行動 (±1.5 s を見て付ける) | 不要 | キーフレーム箱で時間プーリング | 箱単位 | 人。**「クリップを見て 1 枚に付ける」注釈手順の先例** |
| ChimpACT | 行動はキーフレーム (10 フレームおき)、**追跡は全フレーム** | 必要 | — | ○ | チンパンジー |
| YOLO-Behaviour | 1 枚の画像に行動を箱として付ける | 不要 | **なし** (後処理でイベント化) | 箱単位 | 単フレーム検出器 |
| JAABA | 追跡済み軌跡のフレームに行動 | 追跡が前提 | 軌跡の窓特徴 | ○ | 能動学習。ハエ・マウス |
| DeepEthogram | フレームに行動 | — | フレーム全体 | ✕ (個体を分けない) | |
| SIPEC | 個体のセグメント + 行動 | 必要 (識別網) | ○ | ○ | 霊長類・マウス |
| SimBA / MARS | 姿勢推定 + 追跡済み個体に行動 | 追跡が前提 | 姿勢の窓特徴 | ○ | |

### 3.2 YOAKE の差分 (我々の調査範囲で)

1. **ラベルと同一性の分離。** 注釈の単位は「キーフレーム上の個体インスタンス (OBB + 行動)」だけで、
   track_id も区間も付けない。同一性は解析時のトラッキングでしか作らない。JAABA・SimBA・MARS は
   追跡を注釈の前提に、ChimpACT は全フレームの追跡ラベルを、SIPEC は識別ラベルを要求する。
2. **「自分で作った track」で時間文脈を個体に揃える。** YOWO の時間文脈はフレーム全体の特徴なので、
   似た個体が接触していると「誰の動きか」が混ざる。YOAKE は学習時にもラベルの無い文脈フレームへ
   detector と tracker を走らせ、キーフレームの各個体に揃った履歴を作る。トラッカーは**ラベルの源ではなく
   特徴の作り手**として学習に入る。
3. **2 系統の時間 evidence と、その加法的な説明。** 個体に沿った文脈 (track) と個体に依らない文脈
   (tube) を並べ、`logit = Σ_g gate_g · head_g` で足す。交尾中の 2 匹が 1 塊になって track が切れる場面では
   tube 側が効き、どちらが効いたかが予測の分解として GUI に出る。
4. **注釈コストが単フレーム検出と同じ。** 1 個体 1 判断 (箱を確認して数字キー 1 回)。それでいて YOLO-Behaviour
   のような単フレーム法と違い、モデルは動きを見る。人も文脈ループ再生で動きを見て判断する (AVA の手順)。
5. **能動学習ループの中で完結する。** 不確実なキーフレームを選ぶ → 付ける → 学習 → 解析、を
   アノテーター内で回す (既存の round 機構)。評価用には**ランダムに選んだキーフレーム**を別に持つ
   (不確実性で選んだフレームは偏っているので、精度の推定に使えない)。

> 論文化する場合は、多個体の動物行動で「キーフレームのみ・ID ラベルなし・track 揃えの時間文脈」を
> 組み合わせた先行例が本当に無いか、追加調査が必要。上表は 2026-11 時点の調査に基づく。

---

## 4. ラベル形式

`annotations.json` の構造は**変えない** (v1.1 / v2.0 の読み手はそのまま読める)。意味だけを
`meta.labeling` で宣言する。

```json
{
  "meta": {
    "version": "2.0",
    "labeling": {
      "mode": "keyframe",
      "clip": {"length": 8, "stride": 4, "anchor": "end"}
    }
  },
  "class_names": ["fly"],
  "action_names": ["stationary", "walking", "grooming", "courtship", "copulation"],
  "videos": [{
    "video_id": "fruit_fly_copulation_01",
    "video_path": "../videos/fruit_fly_copulation_01.mp4",
    "fps": 30.0, "width": 640, "height": 480, "num_frames": 107987,
    "frames": [
      {"frame_index": 12345, "image_path": "images/fruit_fly_copulation_01/012345.jpg",
       "objects": [
         {"object_id": 1, "bbox": [..], "obb": {"cx": .., "cy": .., "w": .., "h": .., "angle": ..},
          "class_id": 0, "track_id": -1, "action_id": 3},
         {"object_id": 2, "bbox": [..], "class_id": 0, "track_id": -1, "action_id": -1}
       ]}
    ]
  }]
}
```

規則:

| 項目 | 意味 |
|---|---|
| `frames` | **キーフレームだけ**。並びは疎でよい (連続している必要はない)。 |
| キーフレームの網羅性 | キーフレームに載っている箱が、そのフレームの**全個体**。載っていない場所は背景 (検出の負例)。個体ゼロのキーフレームは「何もいない」という正当なラベル。 |
| `action_id` | 「この瞬間、前後の動きを見たうえで、この個体がしている行動」。`-1` は**行動を判断しなかった** (箱だけのラベル)。検出の学習には使い、行動の損失からは外す。 |
| `track_id` | 書かない (`-1`)。書いてあっても学習・解析は使わない。 |
| `meta.labeling.clip` | 学習でキーフレームの直前から切り出すクリップ。`length` フレームを `stride` おきに取り、キーフレームが末尾 (`anchor: end`)。既定 8 × 4 = 29 フレーム ≒ 1.0 秒 @30 fps。 |

旧形式 (区間 / 全フレーム + track_id) は `annotation.keyframes.to_keyframes()` で変換できる:
`stride` フレームおきにキーフレームとして間引き、track_id を外す。区間由来の `action_id` はそのまま使う。

---

## 5. 学習

| Stage | 入力 | 学習するもの | キーフレームモードでの扱い |
|---|---|---|---|
| 1 | キーフレーム 1 枚 | detector (+ OBB 角度) | `KeyframeClipDataset(single_frame=True)`。**OBB の GT を損失へ渡す** (現行の `SingleFrameDataset` は `obb` を落とすので、角度ヘッドが一度も学習されていなかった — これも同時に直る) |
| 2 | キーフレームの直前クリップ `(T, 3, H, W)` | HTM + behavior (+ clip tube) | `KeyframeClipDataset`。detector は凍結。GT はキーフレームのみ。クリップ内の検出は内部トラッカーで連結 |
| 3 | — | ID head | **不要・拒否**。同一性はトラッカーが作る。実行すると理由を表示して終了 |
| 4 | 同 Stage 2 | 全体 | キーフレームの検出 + 行動損失で微調整 |

`KeyframeClipDataset` の要点:

* クリップの番号は `clip_indices(t, length, stride, anchor, num_frames)` (純粋関数)。動画の先頭より前は
  先頭フレームで埋め、`clip_valid` マスクで区別する。
* フレームの読み元は 2 通り。ラウンドが書き出した**画像ツリー** (`images/<video>/<index>.jpg`) があれば
  それを、無ければ `video_path` から**直接デコード**する (`datasets/release/` はラベル+動画パスのみなので)。
* データ拡張は**クリップ全体で同一** (同じ flip・同じ色変換)。フレームごとに変えると動きが偽造される。
* targets は既存の `BatchSequenceTargets` 形 (B × T) で、キーフレーム以外は空。
  `last_frame_targets` がそのまま使えるので Trainer と損失は変更しない。

---

## 6. モデルの変更

### 6.1 トラッカー類似度の正規化 (バグ修正)

`OBBAssociator.similarity` を「**実際に計算した手がかりの重みの和**」で割るように直す。
全手がかりが揃う streaming 推論の結果は変わらない。embedding なし (学習時の内部トラッカー、
パイプラインの一部の段) と OBB なし (AABB detector) の経路だけが正しくなる。

### 6.2 clip tube ストリーム (YOWO の 3D 分岐に相当)

```
各フレーム t の最高解像度 FPN 特徴 (B, C, H, W)
   └─ 1×1 conv で C → c に圧縮 (フレームごとに 1 回だけ。streaming ではこの結果をリングバッファに保持)
キーフレームの個体 OBB → context AABB (既存 ContextROIExtractor と同じ箱)
   └─ T フレームすべてで同じ箱を ROIAlign → (N, T, c, S, S)
   └─ [x_t, x_t − x_{t−1}] をチャネル結合 → Conv3d ×2 → GN/ReLU → 平均プーリング → (N, d)
   └─ evidence group "clip"
```

* track に依らないので、検出が途切れても・2 匹が 1 塊でも値が出る。
* `behavior.use_clip` (既定 False — 既存 checkpoint の strict 読み込みを壊さないため)。
  キーフレームモードの学習では True にする。切ればパラメータごと消える (ablation 不変条件)。

### 6.3 streaming を学習と同じ時間間隔にする

学習のクリップは `stride` フレームおき。streaming も**トラッキングは毎フレーム**行いつつ、
時間モデルへの入力は同じ間隔で間引いた履歴にする (`TrackFeatureCache.stacked_history(stride=s)`、
リングバッファ長は `(T−1)·s + 1`)。間隔が違うと速度・加速度の単位が学習時と変わる。
clip tube 用の圧縮特徴マップも同じ長さのリングバッファで持つ。

### 守る不変条件 (既存)

* evidence は加法的 (`Σ_g gate_g · head_g == logits`)。clip も 1 group として足すだけ。
* 時間モデルの整列軸は track。
* streaming と batch の**検出**は厳密に一致する。
* ablation フラグは group ごとパラメータを消す。

---

## 7. 推論と解析

### 7.1 predictions.json

各フレームの検出に `action_probs` (行動確率ベクトル) を追加する。トップに `meta`
(`fps`, `action_names`, `clip`) を足す。既存キー (`boxes`, `scores`, `track_ids`, `action_ids`, `obb` …)
と、フレーム番号をキーにした既存の構造はそのまま。

### 7.2 行動 bout 解析 (`analysis/bouts.py`, numpy のみ)

1. track ごとに `(frame, action_probs)` の列を作る。`max_gap` を超える欠損で列を切る。
2. **Viterbi 復号**: 遷移行列は「留まる」確率 `stay` (既定 0.98) とそれ以外の一様遷移。観測尤度は
   モデルの行動確率。フレーム単位の argmax のちらつきを、遷移コストで抑える。
3. 最小 bout 長 (`min_bout_s`) 未満の bout は前後に吸収する。
4. 出力:
   * `bouts.csv` — `track_id, action, start_frame, end_frame, start_s, duration_s, mean_prob`
   * `summary.json` — track ごとの時間配分 (割合)、bout 数、平均持続、最初の出現までの潜時、遷移行列
   * `population.csv` — 時間ビンごとの「各行動をしている個体数」。**ID に依存しない**ので、ID が
     入れ替わっても壊れない集団レベルの指標
5. CLI: `yoake analyze mode=bouts predictions=<file> output=<dir> [fps=30 stay=0.98 min_bout_s=0.3]`

ID が入れ替わると個体別の統計は混ざるが、bout の検出そのものと集団集計は影響を受けない。
個体別の結論を出す前に ID の連続性を確認する、という手順を文書で明示する。

---

## 8. ラベリングツール

### 8.1 プロジェクトのラベル方式

`project.settings["label_mode"]` = `"keyframe"` | `"tracks"`。

* **新規プロジェクトは `keyframe`**。
* 既存プロジェクト (キーが無い) は `tracks` のまま (`use_track_ids` と同じ「無ければ従来どおり」の規則)。
* 切り替えは Settings から。切り替えてもディスク上の track_id や区間は**書き換えない**。

### 8.2 keyframe モードでの画面

| 要素 | 振る舞い |
|---|---|
| 箱 | track ID 欄なし (`use_track_ids` は実効的に False)。**色は行動**、行動未判定は灰色の破線 |
| 行動 | 箱を選んで数字キー `1`–`9` (既存)。Inspector の Behavior 欄 (既存) |
| **文脈ループ** (`P`) | キーフレームの前 1.0 秒〜後 0.5 秒を繰り返し再生する。キーフレームの箱は輪郭だけ重ねて表示し、ループ中もラベルはキーフレームにしか付かない。もう一度 `P` か `Esc` で戻る。AVA の「±1.5 秒を見て 1 枚に付ける」手順をそのまま道具にしたもの |
| タイムライン | Behavior レーンは区間ではなく**キーフレームごとの行動の目盛り** (色 = 行動) |
| 区間・tracklet・ID 入れ替え | 出さない (キーフレームでは意味を持たない) |
| Inspector (何も選ばない時) | 「このキーフレーム: 3 匹、行動未判定 1」 |

### 8.3 能動学習ラウンド

* 書き出し: キーフレームに加え、各キーフレームの**クリップ文脈フレーム**も JPEG で書き出す (既存の
  増分書き出し — 一度書いたフレームは再デコードしない)。
* 学習: `stage=1` (detector) → 成功したら同じラウンドで `stage=2` (行動、クリップ) を続けて実行し、
  その checkpoint を採用する。行動ラベルが少なすぎる場合は Stage 2 を飛ばしてそう表示する。
* Pick frames to label: 既存の不確実性サンプリングに加え、評価用のランダムなキーフレームを混ぜる。

---

## 9. 互換性

* `annotations.json` の構造は不変。`meta.labeling` が無いファイルは従来どおり (dense) と解釈する。
* `data.label_mode=auto` (既定) はアノテーションの `meta.labeling.mode` を読む。明示的に
  `keyframe` / `dense` も指定できる。学習時に解決した値を config に書き戻すので、checkpoint が
  「どの方式で学習したか」を持ち、推論はそれを見て履歴の間隔を合わせる。
* 既存 checkpoint は `use_clip=False` のまま strict に読める。
* 旧 GUI の区間編集・track 機能は `tracks` モードでそのまま動く (削除しない)。

---

## 10. 実装フェーズ

| # | 内容 | 主なファイル |
|---|---|---|
| P0 | 計画 (この文書) | `docs/KEYFRAME_BEHAVIOR.md` |
| P1 | 純粋ロジック: ラベル方式、クリップ番号、検証、旧形式からの変換、文脈ループの状態 | `annotation/keyframes.py` |
| P2 | トラッカー正規化の修正 | `tracking/associator.py` |
| P3 | `KeyframeClipDataset` + collate + CLI 統合 (Stage 3 拒否、label_mode 解決) | `data/keyframe_clip.py`, `cli.py`, `config/config.py` |
| P4 | clip tube ストリーム + streaming の間引き履歴・特徴リングバッファ | `models/behavior/clip.py`, `branch.py`, `ht_rtdetr.py`, `inference/streaming.py`, `tracking/track_buffer.py` |
| P5 | predictions.json の `action_probs` / `meta`、bout 解析、CLI | `inference/streaming.py`, `analysis/bouts.py`, `cli.py` |
| P6 | アノテーター: ラベル方式、行動色、文脈ループ、タイムライン、区間 UI の非表示 | `annotator.py`, `gui/context_peek.py`, `gui/settings.py` |
| P7 | ラウンド: クリップ書き出し + Stage 2 連鎖 | `gui/round_export.py`, `gui/train_operations.py` |
| P8 | 文書・ガイド更新 | `docs/DATASET_FORMAT.md`, `docs/ANNOTATOR_GUIDE.md`, README |

後続 (このブランチの範囲外として明記):

* 解析結果から行動候補を出して人が Y/N する (行動の pre-fill)。今は箱候補だけが出て、行動は数字キーで付ける。
* final pass (全フレーム走査 → 抜き取り確認) をキーフレーム方式の「解析の検証」に組み替える。
* ハブの解析ページに bout ビュー。
* track を正例マイニングに使う ID embedding の自己教師あり学習 (外見の似た個体の長時間 ID 維持)。
* キーフレームを中央に置く非因果クリップ (オフライン解析専用)。

---

## 11. 進捗

- [x] P0 計画
- [ ] P1 `annotation/keyframes.py`
- [ ] P2 トラッカー正規化
- [ ] P3 データセット + CLI
- [ ] P4 clip tube + streaming
- [ ] P5 bout 解析
- [ ] P6 アノテーター
- [ ] P7 ラウンド
- [ ] P8 文書

---

## 12. 未決事項 (確認したい点)

既定値を置いて先に進め、確認後に変える。

1. **行動の語彙と排他性。** 既定は「1 個体 1 行動 (排他)」+ 未判定 `-1`。歩きながら求愛する等の同時成立を
   扱うなら multi-head (`locomotion` / `social` …、既存の `classifier_type: multi_head`) にする。
2. **クリップの長さと間隔。** 既定 8 フレーム × 4 おき ≒ 1.0 秒 @30 fps。グルーミングの速い脚の動きを
   捉えるなら間隔を詰める (16 × 2 など。学習コストは T に比例)。
3. **キーフレームの位置。** 既定は末尾 (因果的 = streaming と同じ)。オフライン解析専用なら中央に置く方が
   精度は上がるはずだが、streaming と学習の形が変わる。
4. **交尾中の重なり。** 2 匹が 1 塊に見える場面を「2 つの箱 (各々 copulation)」とするか、
   「1 つの箱 (ペア)」とするか。既定は 2 つの箱。
5. **既存の区間ラベル。** 変換してキーフレームとして使う / 破棄する。

---

## 参考文献

* Köpüklü, Wei, Rigoll. *You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization.* arXiv:1911.06644 (2019).
* Yang, Dai. *YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection.* arXiv:2302.06848 (2023).
* Dang et al. *YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition.* arXiv:2408.02623 (2024).
* Gu et al. *AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions.* CVPR 2018 (arXiv:1705.08421).
* Ma et al. *ChimpACT: A Longitudinal Dataset for Understanding Chimpanzee Behaviors.* NeurIPS 2023 Datasets and Benchmarks.
* Chan et al. *YOLO-Behaviour: A simple, flexible framework to automatically quantify animal behaviours from videos.* Methods in Ecology and Evolution 16(4):760–774 (2025).
* Kabra et al. *JAABA: interactive machine learning for automatic annotation of animal behavior.* Nature Methods 10:64–67 (2013).
* Bohnslav et al. *DeepEthogram, a machine learning pipeline for supervised behavior classification from raw pixels.* eLife 10:e63377 (2021).
* Marks et al. *Deep-learning based identification, tracking, pose estimation, and behavior classification of interacting primates and mice in complex environments.* (bioRxiv 2020.10.26.355115; Nature Machine Intelligence 2022).
