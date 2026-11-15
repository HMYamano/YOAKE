# キーフレーム行動ラベリング — YOWO 型の再設計計画

> ブランチ: `keyframe-behavior`。この文書は実装の**前に**書いた計画であり、実装はこれに従う。
> 末尾の §11 が進捗チェックリスト、§12 が利用者に確認した決定事項、§15 が multi-head 行動 (オプション) の設計。

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
| 箱 | track ID 欄なし (`use_track_ids` は実効的に False)。**色は行動** (タイムラインと同じ色)、行動未判定は灰色・塗りなし (破線は「補間された箱」の意味で既に使われているため) |
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
- [x] P1 `annotation/keyframes.py` — `tests/test_keyframes.py` (純粋ロジック、torch/dpg 非依存を AST で確認)
- [x] P2 トラッカー正規化 — `tests/test_tracking.py` に再現テスト (既定設定・embedding なしで 6 フレーム同一 ID)
- [x] P3 データセット + CLI — `tests/test_keyframe_clip.py` (フレーム番号を画素に埋めた合成動画で「正しいフレームが入っているか」まで確認)
- [x] P4 clip tube + streaming — `tests/test_clip_stream.py`
- [x] P5 bout 解析 — `tests/test_bouts.py`
- [x] P6 アノテーター — `tests/test_keyframe_mode_gui.py`
- [x] P7 ラウンド — `tests/test_keyframe_round.py` (trainer は既存テストと同じく偽装)
- [x] P8 文書 — `DATASET_FORMAT.md` §8、`ANNOTATOR_GUIDE.md`「キーフレームで付ける」
- [x] P9 multi-head 行動 (§15、2026-11-15 の決定事項 1) — `tests/test_behavior_groups.py` (語彙・保存・データセット・GT 対応付け・損失・モデルの出力順・検証・CLI・streaming・bout)、`tests/test_behavior_groups_gui.py` (キー、箱の塗り、Inspector のグループ別メニュー、語彙の編集、Create Project)

合成データでの end-to-end (CPU, 2 epoch): `train stage=1` → `train stage=2` (クリップ) →
`predict` (checkpoint から keyframe / window / stride を復元) → `analyze mode=bouts` が通ることを確認。
**実データ (20 本 × 1 時間のハエ動画) での学習・精度評価はまだ行っていない。**

multi-head でも同じ流れを合成データ (2 匹、locomotion 3 クラス + social 2 クラス、キーフレーム 25 枚、CPU)
で確認: `train stage=1` → `train stage=2` (3 epoch で `loss_action` 0.35 → 0.17、`macro_f1_locomotion`
1.00 / `macro_f1_social` 0.50、平均 0.75) → `predict` (`head_action_ids`、グループごとに和 1 の
`action_probs`、`_meta.action_groups`) → `analyze mode=bouts` (グループ別の bout、`group` 列)。

---

## 12. 決定事項 (2026-11-15 に確認)

§12 で既定値を置いて先に進めていた 5 点に回答をもらった。

| # | 問い | 決定 | 実装への影響 |
|---|---|---|---|
| 1 | 行動の語彙と排他性 | **1 個体 1 行動 (排他) が既定。multi-head をオプションで選べる** | §15 で設計・実装 |
| 2 | クリップの長さと間隔 | 回答なし → 既定 8 × 4 (≒ 1.0 秒 @30 fps) のまま | 変更なし (`settings["clip"]` / `data.clip_length` で変えられる) |
| 3 | キーフレームの位置 | **末尾** (因果的、streaming と同じ形) | 変更なし。`anchor: center` は学習で拒否したまま |
| 4 | 交尾中の重なり | **個体ごとに囲む** (2 匹なら箱 2 つ) | 変更なし |
| 5 | 既存の区間ラベル | **使わない** | 変換経路 (`to_keyframes()`) は残すが、ラウンドや文書の手順には入れない |

---

## 13. 実装メモ — 計画からの変更点

実装中に分かったことで、計画 (§5–§8) を変えた点。

1. **学習時のキーフレーム個体を GT 箱に固定する** (`HTRTDETR._anchor_keyframe`,
   `train.anchor_keyframe_instances`, 既定 True)。最初の end-to-end で、Stage 2 は
   「detector の検出のうち GT と IoU 0.5 以上で重なったもの」しか行動を学習しない、
   つまり detector が見落とした個体のラベルは捨てられることが分かった (初期ラウンドでは
   全部捨てられ、損失に勾配が無く学習が落ちた)。AVA 系の行動検出と同じく、キーフレームの
   個体集合をラベルされた箱そのものにし、特徴は予測箱が最も重なる query から取る。
   文脈フレームは従来どおり detector + tracker。検証時も同じ (箱が正しいときの行動分類
   精度を測る。検出の精度は Stage 1 が測る)。
2. **凍結した detector は eval のまま** (`HTRTDETR.train`)。`freeze_module` は
   `requires_grad` を落とすだけで、Stage 2/3 でも BatchNorm が各時刻 B 枚の統計で正規化し、
   running statistics も書き換わっていた。dense 方式にも効く修正。
3. **勾配の無いバッチは飛ばす** (Trainer)。上記 1 の前は Stage 2 全体が例外で終わっていた。
4. **`load_model_weights(strict=False)` は形の変わったテンソルを飛ばす**。clip group を
   足すと gate の入力次元が変わり、部分読み込みのはずが例外になっていた。
5. **行動ラウンドの長さは optimizer step で決める** (`train_operations._round_begin_behavior`)。
   Stage 2 は行動の無いキーフレームを飛ばし、クリップはバッチ半分なので、1 pass あたりの
   step 数が Stage 1 と違う。Trainer の 100 step warmup だけで終わらないようにした。
6. **ラウンドの採用判定は detector のスコアのまま**。Stage 2 の macro-F1 (高いほど良い) を
   過去ラウンドの detector 損失 (低いほど良い) と比べないため。採用されるのは Stage 2 の
   checkpoint (同じ detector + 行動ヘッド) で、行動の F1 は `rounds.json` に別に記録する。
   Stage 2 が失敗しても detector は採用する。
7. **`annotator.py` の行数上限 (5100) を守るため**、キーフレーム方式の画面ロジックは
   `gui/keyframe_mode.py`、文脈ループは `gui/context_loop.py` に置いた。
8. **実測 (2026-11-15)**: 実データ (`fruit_fly_copulation_01.mp4`, 640x480 / 30 fps / 1 時間) から
   ランダムなキーフレームの 8×4 クリップを動画から直接デコードすると、1 クリップ中央値 110 ms
   (単スレッド、最大 128 ms)。ラウンドは書き出した JPEG を読むのでこの値は本学習
   (`datasets/release/` のようなラベル+動画パスのみのデータ) のもの。キーフレーム 200 枚なら
   1 epoch あたりデコードだけで約 22 秒なので、本学習では `data.num_workers` を上げる。

9. **multi-head の実装で変えた点 (§15)。** `BehaviorPrediction` は multi_head でも平坦な
   `logits` (head の連結) と `probs` (head ごとの softmax の連結) を持つようにした。検出順への
   並べ替え・保存・説明 (evidence) が single と同じ経路で済み、旧 ActionHead に落ちる分岐
   (§15.6 の 3) も自然に消える。GUI の区切りは `·` ではなく ` + ` (DearPyGui の既定フォントに
   無い文字を避ける)。グループまわりの画面は `gui/behavior_groups.py` に置き、`KeyframeModeMixin`
   が継承する。`_update_action_buttons` をそこへ移したので `annotator.py` は 5091 → 5045 行。

## 14. 次にやること

* 実データでの検証: 20 本のうち数本でキーフレーム 100–200 枚 → ラウンド数回 → 残りの動画で
  bout を出し、手作業の区間ラベル (あれば) と比較する。clip tube の有無 (`behavior.use_clip`)
  と track 揃えの有無の ablation がそのまま論文の表になる。
* 行動の pre-fill: Analyze でクリップ推論を走らせ、行動候補を Y/N で確認できるようにする
  (今は箱候補だけが出て、行動は数字キーで付ける)。
* Pick frames to label に「行動の不確実性」と「評価用ランダムキーフレーム」を足す。
* final pass をキーフレーム方式の「解析結果の抜き取り検証」に組み替える。その際、release
  (`annotation/release.py`) に multi-head の宣言と `action_groups` も書く (今は箱の `action_ids`
  は残るが、グループの定義は書かれない)。
* ハブの解析ページに bout / エソグラム表示。

---

## 15. multi-head 行動 (オプション)

§12-1 の決定: **既定は 1 個体 1 行動 (排他)**。「歩きながら求愛する」のように同時に成り立つ
行動を扱いたいときだけ、プロジェクトで **multi-head** を選ぶ。multi-head では行動を
いくつかの**グループ**に分け、各グループの中では排他、グループ同士は同時に成り立つ。

```
locomotion: still, walk, run
social:     none, courtship, copulation
```

この例では、1 匹につき locomotion から 1 つ、social から 1 つを選ぶ。単一の softmax では
「walk かつ courtship」を表せないが、グループごとの softmax なら表せる。グループを 1 つしか
作らなければ、従来の排他な 1 行動と同じになる。

### 15.1 なぜグループ (head) で、multi-label (行動ごとの sigmoid) ではないのか

* **欠けた答えと「していない」を区別できる。** multi-label では、チェックの付いていない行動が
  「していない」なのか「見ていない」なのか分からない。グループなら、各グループに `none` を置けば
  「していない」は明示的な答えになり、未回答 (`-1`) と区別できる。
* **ラベル付けの手間が 1 行動の場合と同じ形になる。** 1 グループにつきキーを 1 回押すだけで、
  行動ごとに on/off を判断する必要がない。
* **bout 解析がそのまま使える。** グループの中は排他なので、§7.2 の sticky-HMM をグループごとに
  回せばよい (状態がグループ内の行動)。
* **モデル側に部品がある。** `EvidenceBehaviorHead(classifier_type="multi_head")` と
  `MultiHeadActionLoss` は既に実装されている (ただし学習経路は §15.6 のとおり壊れていた)。

### 15.2 ラベル形式

```json
{
  "meta": {"labeling": {"mode": "keyframe", "clip": {...}, "behaviors": "multi_head"}},
  "action_names": ["still", "walk", "run", "none", "courtship", "copulation"],
  "action_groups": {"locomotion": ["still", "walk", "run"],
                    "social": ["none", "courtship", "copulation"]},
  "videos": [{"frames": [{"objects": [
    {"bbox": [..], "track_id": -1, "action_id": 1, "action_ids": [1, 4]}
  ]}]}]
}
```

| 項目 | 意味 |
|---|---|
| `meta.labeling.behaviors` | `"single"` (既定、無ければこちら) か `"multi_head"`。**宣言が無い限り multi-head として扱わない。** 既存の `action_groups` は階層分類の定義にも使われてきたため、それがあるだけでは multi-head にしない。 |
| `action_groups` | グループ名 → 行動名 (順序つき)。multi-head の語彙の**唯一の正**。 |
| `action_names` | グループを順に連結した平坦な一覧 (`action_groups` から機械的に作る)。行動番号 (色、キー 0-9、CSV) は全グループ通しの番号。別グループに同じ名前があるときは `social:none` のようにグループ名を付けて区別する。 |
| `action_ids` | 1 個体の答え。**グループの順に 1 つずつ**、通し番号で書く。`-1` はそのグループを判断しなかった (そのグループの損失から外す)。 |
| `action_id` | `action_ids[0]` (最初のグループの答え) の写し。multi-head を知らない読み手が、最初のグループだけの単一ラベルとして読めるように残す。 |

### 15.3 ラベリングツール

* **Create Project** の Behaviors 欄で「One behavior per animal (recommended)」か
  「Several behavior groups (multi-head)」を選ぶ。multi-head では 1 行に 1 グループ
  (`group: a, b, c`) を書く。**Settings > Labels** でも同じ書式で編集できる。
* キー 0-9 は全グループ通しの番号。押した行動は**そのグループの答えだけ**を置き換え、
  ほかのグループの答えは残る。Inspector の「Set behavior」はグループごとに見出しを付けて並べる。
* 箱は最初のグループの行動の色で描き、全グループに答えが付くまでは塗らない (枠だけ)。
  「塗られている = 終わった」は単一行動のときと同じ。ラベルは `walk + courtship` のように
  並べ、未回答のグループは `?` で示す。
* Inspector の進捗文は「3 animals - 1 still needs a behavior (social).」のように、
  答えの足りないグループ名を出す。
* 語彙を変えたら、既存の答えは**グループ名と行動名で付け直す**。グループの途中に行動を足しても、
  後ろのグループの答えがずれない。消した行動の答えは未回答 (`-1`) に戻る。
* multi-head は keyframe 方式でだけ選べる。tracks 方式の区間ラベルは単一行動のまま。

### 15.4 学習

* CLI はデータの宣言を読み、`model.behavior.classifier_type = multi_head`、
  `model.behavior.heads = action_groups`、`num_actions = len(action_names)` を設定する
  (`model.behavior.classifier_type` を明示したときはそちらを優先し、データと食い違えば止める)。
  決定は checkpoint の config に入るので、`predict` は同じ形のモデルを組み立てる。
* `KeyframeClipDataset` は `head_action_ids` (N × グループ数、グループ内番号、`-1` は未回答) を
  targets に加える。Stage 2 の「行動ラベルの無いキーフレームは飛ばす」は、どれか 1 グループでも
  答えがあれば残す。
* モデルは各グループの logit を**検出順に並べ直してから**損失に渡す (`action_head_logits`)。
  平坦な `action_logits` はグループの logit の連結、`action_probs` はグループごとの softmax の連結
  (各グループの区間の和が 1)。
* 損失はグループごとの focal cross-entropy の平均 (`MultiHeadActionLoss`)。答えの無いグループは飛ばす。
* 検証はグループごとの macro-F1 を出し (`macro_f1_<group>`)、その平均を `macro_f1` とする。
  ラウンドの記録 (`rounds.json` の behavior_score) もこの平均。
* ラウンドの「行動ラベルが 8 個以上」は、どれか 1 グループに答えのある箱を 1 個と数える。

### 15.5 推論と解析

* `predictions.json`: 検出ごとに `action_ids` (最初のグループ、互換のため)、`head_action_ids`
  (グループごとの通し番号)、`action_probs` (平坦、グループごとに和 1)、`head_predictions`
  (グループ名 → 行動名) を書く。`_meta` に `action_groups` を書く。
* bout 解析 (`analyze mode=bouts`) は**グループごとに独立に**復号する。1 匹が同時に
  「walk の bout」と「courtship の bout」を持てる。`bouts.csv` に `group` 列を足し、
  `summary.json` は `groups.<名前>` にグループごとの時間配分・遷移・潜時を入れる。
  `population.csv` の列は平坦な行動名で、各個体はグループごとに 1 匹分ずつ寄与する。

### 15.6 既存の multi_head 経路で見つかった不具合 (修正する)

既存の `classifier_type: multi_head` は、forward の形を確かめるテストはあったが、
学習が一度も通っていなかった。

1. `assign_gt_to_detections` が決め打ちのキー (`action_ids` など) しか検出に写さず、
   `MultiHeadActionLoss` が読む `action_<head>` は常に `-1` だった。つまり**損失が常に 0** で、
   multi-head の行動は学習されなかった。
2. `behavior.head_logits` は track 行の順で、GT は検出順。並べ直さずに損失を計算していた。
3. multi-head では `BehaviorPrediction.logits` が None なので、モデルは `action_logits` に旧
   ActionHead の出力を入れていた。検証の macro-F1 は、学習していない別のヘッドを測っていた。

### 15.7 互換性

* 既定は single。`behaviors` の宣言が無いファイル、`settings["behaviors"]` の無いプロジェクトは
  すべて従来どおり。
* single のファイルは `action_ids` を書かないので、保存結果は従来とバイト単位で同じ。
* 単一行動で付けたプロジェクトを後から multi-head にしても、既存の答えは最初のグループの答えとして
  残る (最初のグループの行動を元の一覧と同じにしておけば番号も変わらない)。

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
