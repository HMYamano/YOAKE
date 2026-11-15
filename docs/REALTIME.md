# リアルタイム解析 — multiprocessing パイプライン

`yoake predict ... realtime=true` は、streaming 推論と同じ計算を、CPU の仕事を別プロセスに
出したパイプラインで回します。カメラやネットワークストリームもこのモードで読めます。
実装は [`src/htrtdetr/inference/realtime.py`](../src/htrtdetr/inference/realtime.py)。

---

## 1. 使い方

```bash
# 動画ファイル (逐次の yoake predict と同じ結果を、より速く)
yoake predict source=video.mp4 weights=best.pth realtime=true

# カメラ 0 を、シルエット幾何つきで、画面に出しながら 10 分 (q / Esc か Ctrl+C で止まる)
yoake predict source=0 weights=best.pth realtime=true geometry=true \
    inference.realtime_display=true inference.realtime_max_seconds=600

# ネットワークカメラ
yoake predict source=rtsp://192.168.0.10/stream weights=best.pth realtime=true
```

`realtime=true` を付けずにカメラ番号やストリームを渡すと、モデルを読む前にエラーで止まります
(逐次ループはカメラを待てないため)。

## 2. 構成

```
capture ──▶ main (GPU) ──▶ geometry ──▶ writer
decode       detector        silhouette    draw / output.mp4
resize       tracker         head-tail     predictions.json / .jsonl
normalize    HTM / behavior  motion        geometry.csv / display
```

- **capture**: 読み込み、BGR→RGB、detector 入力の前処理 (`prepare_input`: 640² への
  リサイズと正規化)。
- **main**: `StreamingInferencer.process_prepared`。GPU (CUDA) を持つのはこのプロセスだけ。
  tracker が逐次なので、ここは 1 プロセスで順番に進む。
- **geometry**: シルエット幾何 (`geometry=true` のときだけ起動)。頭尾の状態を持つので
  1 プロセスで順番に。
- **writer**: 描画、mp4、`predictions.jsonl` (1 フレーム 1 行)、`predictions.json`、
  `geometry.csv`、表示ウィンドウ。

フレームの画素は pickle しません。共有メモリ上の固定数スロット (`FrameRing`) に capture が
生フレームとモデル入力を書き、キューに流れるのはスロット番号と検出結果だけです。スロットは
writer が書き終えると capture に戻ります。

## 3. 落とすか待つか (`inference.realtime_drop_frames`)

| 値 | 動作 | 既定の対象 |
|---|---|---|
| `never` | 空きスロットを待つ。1 フレームも落とさない | ファイル・画像フォルダ |
| `when_full` | 空きが無ければそのフレームを捨てる | カメラ・ストリーム |
| `auto` (既定) | 上の 2 つを入力の種類で選ぶ | — |

カメラを待たせると、ドライバのバッファが溢れて**どのフレームが消えたか分からないまま**時間が
詰まります。`when_full` は capture 側で明示的に捨て、フレーム番号を欠番として残します。tracker は
実フレーム番号で速度を取るので (`TrackState.velocity`)、欠番は「その間は観測が無い」として
扱われます。ライブ入力では各フレームに `t` (最初のフレームからの秒) が付き、`_meta.fps` は
実測の取り込み速度に置き換わります (カメラの申告 fps は当てにならないため)。

**スロット数** (`inference.realtime_slots`、0 = 自動) はパイプラインの深さで、遅延の上限です。
自動ではライブ入力 4、ファイル 8。ライブ入力では main が律速なので、深くしても遅延が増える
だけで速くはなりません。

## 4. 結果の一致

ファイル入力では、リアルタイム経路の結果は逐次の `yoake predict` と**完全に一致**します
(検出・ID・行動確率・evidence・幾何まで)。前処理は同じ関数 (`prepare_input`) を別プロセスで
走らせるだけで、推論は同じ main プロセスで同じ順番に行うからです。
`tests/test_realtime.py::test_realtime_matches_sequential_predict_exactly` が固定しています
(下の計測でも GPU + AMP で一致を確認)。

## 5. 実測

**条件**: このリポジトリの開発機 (RTX 5060 Ti)。既定サイズのモデル (ResNet-18、640²、AMP、
channels_last) を**ランダム重み**で、`num_queries=10`・`score_threshold=0` にして毎フレーム
ちょうど 10 個体を検出させた。入力は 640×480 の合成動画 300 フレーム (MJPG)。学習済みモデル
や実際の映像では検出数・行動ヘッドの負荷が変わるので、手元のチェックポイントで測り直してください。

| | 幾何なし | 幾何あり |
|---|---|---|
| 逐次 `yoake predict` | 28.2 fps | 25.6 fps |
| `realtime=true` (定常) | 39.8 fps | 39.7 fps |

main プロセスの 1 フレームあたり (ms):

| | 前処理 | detector | tracking | temporal | behavior | 合計 |
|---|---|---|---|---|---|---|
| 逐次 | 9.2 | 8.8 | 3.1 | 2.0 | 9.3 | 33.2 |
| realtime | 0.3 | 9.1 | 3.0 | 2.0 | 9.4 | 24.8 |

- 速くなった主因は前処理 (PIL でのリサイズと正規化、9 ms) が capture へ移ったこと。デコード・
  描画・mp4 書き出しも main から外れている。
- シルエット幾何 (10 個体で約 5 ms/frame) は geometry プロセスに隠れ、スループットに影響しない。
- 残りの律速は main の推論 (detector と behavior)。これ以上は GPU 側の最適化
  (`detect_only.py` の計測ではデコーダ層数が支配的) か、検出数を減らすことが要る。
- ファイル入力・スロット 8 での遅延 (取り込み → 書き出し) は p50 約 170 ms。ファイルでは
  capture が先読みしてスロットを埋めるためで、ほぼ「スロット数 × 推論時間」になる。
- 子プロセスの起動に 2–3 秒かかる (`htrtdetr` パッケージの import が torch を読み込むため)。
  短い動画では起動時間の分、逐次の方が早く終わることがある。

## 6. 出力

| ファイル | 内容 |
|---|---|
| `predictions.json` | 逐次の `yoake predict` と同じ形 (`_meta.realtime` が増える)。終了時に書く |
| `predictions.jsonl` | 1 フレーム 1 行 (`{"frame": i, ...}`)。実行中に追いかけて読め、途中で落ちても残る |
| `output.mp4` | 注釈付き映像 (`inference.realtime_write_video=false` で書かない) |
| `geometry.csv` | `geometry=true` のとき ([SILHOUETTE_GEOMETRY.md](SILHOUETTE_GEOMETRY.md)) |
| `runtime_stats.json` | 読んだ / 捨てた / 処理した / 書いたフレーム数、定常 fps、遅延 (p50/p95/max)、main の段ごとの時間、geometry の時間 |

## 7. 止め方と失敗

- 入力が尽きる、`max_frames`、`inference.realtime_max_seconds`、表示ウィンドウで q / Esc、
  Ctrl+C のどれでも、capture が止まり、終端の印が全段を流れ、writer がそこまでの結果を
  書き切ってから終わります。Ctrl+C の後に届いたフレームは推論せずに捨てます。
- 子プロセスは SIGINT を無視します (Ctrl+C はコンソールの全プロセスに届くため)。
- どこかの段が例外で落ちると、main がその traceback を付けて `RuntimeError` を上げます。
  下流の段はそれまでの結果を書き切ります。main が強制終了された場合も、子プロセスは
  親の消滅を検知して終わります。

## 8. 設定

| キー | 既定 | 意味 |
|---|---|---|
| `realtime` | false | このモードを使う (`realtime=true` は短縮形) |
| `inference.realtime_slots` | 0 | スロット数。0 = 自動 (ライブ 4 / ファイル 8) |
| `inference.realtime_drop_frames` | auto | `auto` / `never` / `when_full` |
| `inference.realtime_display` | false | 注釈付き映像をウィンドウに出す |
| `inference.realtime_write_video` | true | output.mp4 を書く |
| `inference.realtime_max_seconds` | 0 | この秒数で止める (0 = 制限なし) |
| `max_frames` | — | このフレーム数で止める |
| `geometry` | false | シルエット幾何を geometry プロセスで測る |

## 9. 制限

- フレームの大きさは 1 本のストリームの中で一定である必要があります (共有メモリのスロットが
  固定長のため)。大きさの混ざった画像フォルダは逐次モードで処理してください。
- `inference.mode=batch` はリアルタイムでは使えません (streaming 推論で回します)。
- `predictions.json` は writer がメモリに溜めて最後に書きます (逐次と同じ)。長時間のライブ解析
  では `predictions.jsonl` を正としてください。
