# シルエット幾何 — ラベル不要の体軸・頭尾・形状・残差運動

`yoake predict ... geometry=true` で、検出された各個体について、箱 (bbox / OBB) からは
分からない幾何を測って出力に足します。**学習もラベルも要らない後処理**で、モデルの
出力 (検出・ID・行動) は一切変わりません。実装は
[`src/htrtdetr/analysis/silhouette.py`](../src/htrtdetr/analysis/silhouette.py)。

リアルタイム解析 (`realtime=true`) では、この測定が専用の geometry プロセスで並走します
([REALTIME.md](REALTIME.md))。

---

## 1. なぜ要るのか

行動ブランチ (`models/behavior/kinematics.py`) の orientation / interaction 特徴は体軸を
前提にしていますが、現状その入力は実質的に欠けています。

| 状況 | 体軸 θ | 頭尾 (180°) |
|---|---|---|
| 既定の AABB detector (`detector.type=rtdetr_aabb`) | 常に 0 (`ht_rtdetr._det_obb`) | なし |
| OBB detector | 予測されるが、学習経路で角度損失が走らない (`Trainer` が `obb_cfg` を渡さない) | なし |
| polarity head | — | 教師を出す Dataset が無く、推論でも捨てられる |
| pipeline の L4 / L5 | 速度方向 `atan2(v)` で代用 (静止中は無意味) | — |

シルエットから体軸を取り、移動方向で頭尾を決めれば、これらの穴を**ラベル無しで**
埋められます。骨格 (細線化) や Medial Axis は作りません。640px 画像で体長 15–40px の
ハエでは脚は二値化で消え、骨格に残るのは体軸と翅だけで、それは下の量で取れるからです。
Optical flow も計算しません。全身の移動は kinematics が持っていて、局所運動の「量」は
体固定のフレーム差分で足ります。

## 2. 使い方

```bash
# 逐次 (既定の streaming 推論 + 幾何)
yoake predict source=video.mp4 weights=best.pth geometry=true

# 設定の上書き (例: 遅い動物で頭尾の投票を始める速さを下げる)
yoake predict source=video.mp4 weights=best.pth geometry=true geometry_params.min_speed=0.5

# リアルタイム (カメラ 0、幾何を別プロセスで)
yoake predict source=0 weights=best.pth realtime=true geometry=true
```

Python から:

```python
from htrtdetr.analysis.silhouette import SilhouetteAnalyzer, rows_as_dicts

analyzer = SilhouetteAnalyzer(fps=30.0)
for index, frame_rgb in frames:                     # 時間順に渡す
    res = streaming_inferencer.process_frame(frame_rgb, frame_index=index)
    rows = analyzer.update(frame_rgb, index, res.boxes_xyxy,
                           corners=res.corners, track_ids=res.track_ids)
    for det in rows_as_dicts(rows):
        print(det["heading"], det["lateral_asym"], det["motion_energy"])
```

## 3. 出力

- `predictions.json`: 各フレームに `geometry` (検出と同じ順の行のリスト)。列名は
  `_meta.geometry.fields`、測ったときの設定は `_meta.geometry.params`。
- `geometry.csv`: 1 検出 1 行 (`frame, detection, track_id, <fields>`)。pandas でそのまま読める。
- `output.mp4`: 向きの矢印 (確信度 0.5 以上は緑、未満は橙)。向きの未定な個体は体軸の灰線。

| 列 | 単位 | 意味 |
|---|---|---|
| `valid` | 0/1 | シルエットを測れたか。0 なら `diagnosis` に理由 |
| `cx`, `cy` | px | 体の中心 |
| `axis` | rad, [-π/2, π/2) | 体軸 (頭尾なし) |
| `heading` | rad, [-π, π) | 頭の向き。一度も移動を見ていない track は null |
| `heading_conf` | 0–1 | 向きを支持した移動の量 |
| `length`, `width` | px | 体軸方向・直交方向の長さ (細い付属肢を除く) |
| `elongation` | — | 体の芯の細長さ = sqrt(PCA 固有値比)。翅を開いても下がりにくい |
| `area`, `body_area` | px² | blob 全体 / 細い付属肢を落とした体 |
| `appendage_ratio` | — | (area − body_area) / body_area |
| `solidity` | 0–1 | 輪郭面積 / 凸包面積。翅や脚が張り出すと下がる |
| `lateral_extent` | 体長 | 体軸の線から横へ最も遠い blob の点 |
| `lateral_asym` | 体長 | 左右の到達の差。片翅伸展で大きい |
| `width_skew` | −1–1 | (前 1/3 の幅 − 後 1/3 の幅) / 和。前 = heading 側 |
| `twin_core` | 0/1 | 同程度の芯が 2 つ = 2 個体が 1 つの blob (このとき length 等はペアのもの) |
| `motion_energy` | — | 体固定のフレーム差分 / 前景背景の濃度差。前フレームが無ければ null |
| `nn_dist` | 体長 | 最近傍個体までの距離 |
| `facing_nn` | rad, [0, π] | 自分の heading と最近傍への方向のなす角 |
| `faced_by_nn` | rad, [0, π] | 最近傍の heading と自分への方向のなす角 |
| `diagnosis` | — | `""` か、`click_segment.DIAGNOSES` のキー |

`yoake analyze mode=bouts` は、ファイルに幾何があれば `proximity.csv` の体長をシルエットで
測り、`a_faces_b` / `b_faces_a` (相手を ±45° 以内で向いていたフレームの割合) を足します。
向きは有向なので、追う側と追われる側が区別できます。

## 4. 測り方

1. **ROI**: 箱の長辺 × `roi_scale` (既定 2.0) の正方形。体が ROI の縁に当たったら 1.6 倍に
   広げて 1 度だけ測り直す。`max_roi_side` (既定 160px) より大きければ縮小してから測る
   (高解像度や大型動物でも 1 個体のコストを一定に保つ)。
2. **前景**: ROI 内の Otsu。暗い個体か明るい個体かは中心の画素から決める
   (`click_segment.threshold_gray`。Click to Box と同じ)。
3. **他個体の除外**: 同じフレームの他の検出の箱に入る画素は消す (自分の箱と重なる所は残す)。
   検出が 2 つある限り、接触した 2 個体はここで切り離せる。
4. **縮約**: `click_segment.reduce_mask` — 点の下の blob → 距離変換で付属肢を落とした芯 →
   芯の PCA で体軸、opening した体で長さと幅。Click to Box と**同じ関数**なので、注釈の OBB と
   測定が体軸について食い違うことがない。芯が 2 つあれば `twin_core`。
5. **形**: 凸包 (solidity)、体軸の線からの左右の到達 (lateral_*)、前後 1/3 の幅 (width_skew)。
   横に開いた翅は幅があるので opening では消えない — 張り出しは「blob − 体」ではなく
   blob 全体の到達で測る。

### 頭尾 (2π の向き)

体軸 (mod π) の 2 つの読み方のうち、直前の向きに近い方を選ぶ (連続性)。体軸方向に
`min_speed` (既定 1 体長/秒) 以上で移動していれば、進行方向を前とみなして投票する。
確立した向きは、逆向きの移動が `flip_seconds` (既定 0.25 秒) 続いたときだけ反転する
(短い後退歩行では反転しない)。芯が丸い (`elongation < min_elongation`、グルーミング中など)
フレームでは体軸を信用せず、向きを持ち越す。`max_gap_frames` (既定 15) より長く見失った
track は向きを忘れる。

一度も移動を見ていない track は `heading` を null にします。根拠の無い向きを出すと、
facing などの下流の量が黙って間違うからです。

### 残差運動 (`motion_energy`)

前後のフレームを、それぞれのフレームでの体の位置と向きで切り出して重ねる (体固定の
crop)。並進と回転は打ち消され、脚・翅・腹部の動きだけが差として残る。差は今の blob の
中で平均し、前景と背景の濃度差で割って照明に依らない量にする。合成データでは、歩き
ながら向きを変える個体が 0.02–0.04、脚だけを動かす個体が 0.36–0.38 (約 10 倍の差)。

## 5. 計算コスト

| 条件 | 1 フレーム |
|---|---|
| 640×480、10 個体 (合成画像、このマシンの CPU) | 約 3.9 ms (1 個体あたり約 0.39 ms) |

- **実測は合成画像での値です。** 実際のハエ映像でのマスク品質・失敗率・時間は未測定です。
- 小さな ROI を多数処理するので、測定中だけ OpenCV を 1 スレッドにしています
  (多スレッドのままだと 6.8 ms。抜けるときに元のスレッド数へ戻します)。
- GPU は使わず、GPU への転送もありません。モデルのサイズと VRAM は変わりません。

## 6. 限界と注意

- **解像度**: 脚は写らない前提です。翅は写る大きさですが、半透明なので Otsu で体と同じ側に
  入るかは撮影条件次第です。`lateral_asym` が片翅伸展を捉えるかは実データで確かめてください。
- **接触**: 検出が 1 つしか無い 2 個体は分けられません (`twin_core` で分かるだけ)。並んで
  くっついた 2 個体は芯が分かれず、`twin_core` も立たないことがあります
  (`docs/CLICK_SEGMENT.md` §5 と同じ制約)。
- **背景**: 一様でない背景 (アリーナの縁、餌、影) が ROI に入ると Otsu が背景に引きずられます。
  失敗は `valid=0` と `diagnosis` で出ます (推測はしません)。
- **ID の入れ替わり**: 頭尾の状態は track_id ごとなので、ID が入れ替わると移動で正されるまで
  向きが誤ることがあります。
- **非正方形の動画**: OBB detector の `obb` (predictions.json) は非正方形の動画で θ と w/h が
  ずれます (`streaming.py` の画素座標への戻し方)。ここでは正しく変換される `corners` から
  ROI を作るので影響を受けません。

## 7. 検証計画 (まだ実データでは行っていません)

行動分類への効果は**仮説**です。モデルを変える前に、次の順で確かめることを推奨します。

1. **幾何の妥当性 (行動ラベル不要)**: 体軸と注釈 OBB の θ の角度誤差 (ただし注釈 OBB も
   `reduce_mask` 由来なので部分的に循環)、頭尾の正答率 (数百個体フレームを目視で Y/N)、
   `valid` 率を接触・非接触で分けて。
2. **情報量 (モデル変更なし)**: 既存のキーフレーム行動ラベルの箱で幾何を計算し、既存の
   kinematics 生特徴 (`KinematicsEncoder.raw_features`) だけの分類器と、幾何を足した分類器を
   動画単位の grouped CV で比べる (クラスごとの F1 / AUC)。
3. **モデルへの統合 (2 で効果が出た特徴だけ)**: `BehaviorBranch` に `shape` group を
   既定 OFF のフラグで足し、Stage 2 だけを再学習する (detector は凍結のまま、アノテーションの
   追加は不要)。比較対象に「OBB の角度損失を正しく配線した構成」と `use_p2` を入れ、向き入力を
   直しただけの効果と区別する。
