# YOAKE GUI スイート

Annotatorの複数動画Project、動画切替、一括候補生成とレビューの操作は
[ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) を参照してください。

YORU (github.com/Kamikouchi-lab/YORU) を参考にした DearPyGui 製の統一 GUI。
**Labeling（動画アノテータ）/ Training（ライブモニタ）/ Video Analysis（重畳再生）**
を 1 つのハブから利用できます。学習・推論・解析は既存の `yoake` CLI を裏で呼び出すため、
GUI 本体は torch 非依存で起動します（torch は学習/推論を実行する GPU 環境にのみ必要）。

## 起動

```bash
pip install -e ".[gui]"     # dearpygui + opencv-python (PyTorch は入りません)
yoake gui                    # または:  python -m htrtdetr.gui  /  gui
```

3 つの起動方法はどれも同じハブを開きます。学習と自動検出だけは PyTorch を要る
ので、その 2 つを使う場合は `pip install -e ".[ml]"` も入れてください
(ハブ側は何がロックされているかと、その解除方法を表示します)。

ハブ左ナビ:

| ページ | 内容 |
|--------|------|
| **Labeling** | 動画アノテータを別ウィンドウで起動 |
| **Training** | Dataset 準備 / Stage 1–4 学習・評価 / analyze + ライブ学習モニタ |
| **Video Analysis** | 動画を推論 or 予測 JSON を読み込み、bbox/ID/action を重畳再生 |

> DearPyGui は 1 プロセス 1 コンテキストのため、アノテータはハブから別プロセスで
> 起動します（YORU の `app.py` が各モジュールを起動するのと同じ設計）。

## プロジェクトフォルダ

**1 研究 = 1 フォルダ。** アノテーション・学習データ・解析結果はすべてその中に
書き出され、記録されるパスはすべてフォルダ基準の相対パスです。フォルダごと移動・
コピー・他人への受け渡しをしても、そのまま開けます。

```
<project>/  project.yoake.json  annotations.json
            videos/  datasets/  runs/  analysis/
```

Label ページ上部の **Project folder** が「いまどの研究にいるか」で、Train / Analyze
ページの出力先（`runs/train/stage{N}`、`analysis/predict` …）はこれに追従します。
以前は出力先がソースツリーの `runs/` 固定だったため、研究のデータとコードが
混ざっていました。ユーザが自分で打ち込んだパスは、プロジェクトを切り替えても
上書きされません。

フォルダ外の動画は相対パスと絶対パスの両方で記録され、
「プロジェクトだけ動かした」「一式まとめて動かした」のどちらでも解決できます。
両方が別々に動いた場合は **Project Videos ▸ Locate file**（または
`yoake project relink path=... from=<folder>`）で 1 回答えれば、そのフォルダは
記憶され以後は自動で解決されます。詳細は
[PROJECT_FOLDER.md](PROJECT_FOLDER.md)。

## Labeling — 動画アノテータ

フレーム毎の bbox 描画・track ID・行動(action)付与に加え、前フレームからの
box 伝播（Space）と OpenCV 自動追跡が使えます。

1. 動画を開き、キャンバス上でドラッグして box を作る（C を押して個体をクリック
   すれば box を自動フィットさせることもできます。`C` は 1 回 1 箱で、作ったあとは
   Select に戻りその箱が選択されます）。
2. Track ID と Action を割り当てる（Track ID を使わない Project では Action だけ）。
3. 次フレームへ進み、**Space**（前フレームから伝播）または自動追跡で box を運び、
   ずれた分だけ手で直す。

- 自動生成された box（伝播・自動追跡・予測の取り込み）は**破線**、手で描いた／
  編集した box は**実線**で描画され、区別できます。
- 破線 box をドラッグ/リサイズすると実線（手動確定）に昇格します。

保存形式は **YOAKE JSON v1.1**（回転矩形を使った場合のみ v2.0）。`keyframe` フラグは
エディタ内部用で JSON には出力しないため、既存の学習パイプラインは無変更で読み込めます。

## Training — ライブ学習モニタ

Stage 1–4 の学習フォームから `yoake train` を起動し、`<project>/runs/train/stage{N}/results.csv`
と `metrics_latest.json` を定期的に読み取って、損失曲線・主要指標
（Stage1: AP50 / Stage2: macro F1 / Stage3: IDF1 / Stage4: composite）をライブ描画します。
「監視」ボタンで既存の学習出力ディレクトリを後から接続することもできます。

## Video Analysis — オフライン動画解析

- **「推論実行」**: 入力動画とチェックポイントを指定して `yoake predict` を実行し、
  完了後 `output_dir/predictions.json` を自動読込。
- **「予測JSONを読込」**: 既存の `predictions.json`（`yoake predict` / `yoake pipeline`
  の出力、`{"<frame>": {boxes, scores, track_ids, action_ids}}` 形式）を読込。
- 動画を重畳再生（bbox/ID/action）、フレームスライダ/ジャンプ、表示 score しきい値、
  **個体別 行動タイムライン**（縦線＝現在フレーム）を表示。

### Use track identity in this analysis

個体の同一性をこの解析で使うかどうかのチェックボックス（既定オン）。Annotator 側の
Project 設定 `use_track_ids` と対になる、解析側の同じ質問です。

オフにすると

- Track ID / Trajectory / Show only the selected track をロック（隠さず理由を出す）
- Inspector から Track 行と「前後の出現フレームへ飛ぶ」を外す
- タイムラインは個体ごとのレーンではなく、フレームごとの行動を示す **1 本**になる
- `yoake predict` に `inference.show_id=False` / `inference.show_trajectory=False` を渡し、
  オーバーレイ動画から ID ラベルと軌跡を落とす

**これは「読み方」の宣言であって、モデルの tracking を止めるものではありません。**
時間モデルは track 単位で整列している（`model.track_aligned_temporal`）ので、そこを
切ることは表示設定ではなく別の推論です。`predictions.json` の `track_ids` はそのまま
書かれます。

読み込んだ `predictions.json` に `track_id >= 0` の検出が 1 つも無い場合は、
チェックボックスが自動でオフになり、その旨を Run log に残します。存在しないものを
「表示できる」と見せないためです。

> リアルタイム/Webカメラ解析・クローズドループ（外部機器制御）は本スイートの対象外です。

## 構成（開発者向け）

`src/htrtdetr/gui/`

- 純モジュール（dearpygui 非依存・単体テスト対象）
  - `labeling_ops.py` … bbox 補間 (`interpolate_between`) / クランプ・丸め
  - `predictions.py` … `predictions.json` ローダ・行動タイムライン構築
  - `monitor.py` … `results.csv` / `metrics_latest.json` パーサ + ポーラ
  - `commands.py` … `yoake` コマンド生成（`visualize`/`track`→`inference.show_*` のマッピング）
- UI モジュール（dearpygui）
  - `hub.py` / `canvas.py` / `training_panel.py` / `analysis_panel.py` / `runner.py` / `dialogs.py`

- ダイアログの幾何
  - `widgets.place_dialog` / `refit_dialogs` … ウィンドウをビューポート内に収める
  - `widgets.dialog_body` / `dialog_footer` … 「伸びる本体はスクロール、ボタン列は固定」
    の分割。固定サイズのウィンドウに内容が収まらなくなったとき、**押したいボタンだけが
    画面外に出る**という壊れ方をしないための型で、ダイアログは全てこれに従う。

テスト: `tests/test_labeling_ops.py`, `tests/test_gui_commands.py`（いずれも torch 不要）、
`tests/test_gui_dialogs.py`（900x600 ＝ アノテータの最小ウィンドウで、各ダイアログと
その主ボタンが画面内にあることを実測で確認）、`tests/test_track_identity.py`。
