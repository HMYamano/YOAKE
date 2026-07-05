# YOAKE GUI スイート

YORU (github.com/Kamikouchi-lab/YORU) を参考にした DearPyGui 製の統一 GUI。
**Labeling（範囲ラベリング対応）/ Training（ライブモニタ）/ Video Analysis（重畳再生）**
を 1 つのハブから利用できます。学習・推論・解析は既存の `yoake` CLI を裏で呼び出すため、
GUI 本体は torch 非依存で起動します（torch は学習/推論を実行する GPU 環境にのみ必要）。

## 起動

```bash
pip install -e ".[gui]"     # dearpygui + opencv-python
yoake gui                    # または:  python -m htrtdetr.gui  /  gui
```

ハブ左ナビ:

| ページ | 内容 |
|--------|------|
| **Labeling** | 動画アノテータを別ウィンドウで起動（範囲ラベリング対応） |
| **Training** | Dataset 準備 / Stage 1–4 学習・評価 / analyze + ライブ学習モニタ |
| **Video Analysis** | 動画を推論 or 予測 JSON を読み込み、bbox/ID/action を重畳再生 |

> DearPyGui は 1 プロセス 1 コンテキストのため、アノテータはハブから別プロセスで
> 起動します（YORU の `app.py` が各モジュールを起動するのと同じ設計）。

## Labeling — 範囲ラベリング

従来のフレーム毎の bbox 描画・track ID・行動(action)・CSRT 自動追跡に加え、
サイドバー **「Range Labeling」** で時間区間の一括付与ができます。

1. 開始フレームで対象個体の box を描く（キーフレーム＝実線）。
2. 終了フレームへ移動し、同じ track ID で box を描く（キーフレーム）。
3. 「開始/終了」に区間を設定（「現在」ボタンで現フレームを設定）。
4. Track ID と Action を選び、**「範囲に適用」**。

- **bbox を補間する**（既定 ON）: キーフレーム間を線形補間し、中間フレームの box を
  自動生成します（**破線**表示＝補間 box）。全フレームを手描きする必要はありません。
  区間内に手動キーフレームが 3 個以上あれば、それぞれの間を区分線形で補間します。
- キーフレームが 1 個だけなら、その box を区間全体に保持します。
- box が無い区間では行動(action)だけを既存 box に付与します。
- 補間 box を手動でドラッグ/リサイズすると、その box はキーフレーム（実線）に昇格します。

保存形式は **YOAKE JSON v1.1 のまま**（補間 box も通常の per-frame object として保存）。
`keyframe` フラグはエディタ内部用で JSON には出力しません。そのため既存の学習
パイプラインは無変更で読み込めます。

## Training — ライブ学習モニタ

Stage 1–4 の学習フォームから `yoake train` を起動し、`runs/train/stage{N}/results.csv`
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

> リアルタイム/Webカメラ解析・クローズドループ（外部機器制御）は本スイートの対象外です。

## 構成（開発者向け）

`src/htrtdetr/gui/`

- 純モジュール（dearpygui 非依存・単体テスト対象）
  - `labeling_ops.py` … bbox 補間 (`interpolate_between` / `resolve_range_boxes`)
  - `predictions.py` … `predictions.json` ローダ・行動タイムライン構築
  - `monitor.py` … `results.csv` / `metrics_latest.json` パーサ + ポーラ
  - `commands.py` … `yoake` コマンド生成（`visualize`→`inference.show_*` 等のマッピング）
- UI モジュール（dearpygui）
  - `hub.py` / `canvas.py` / `training_panel.py` / `analysis_panel.py` / `runner.py` / `dialogs.py`

テスト: `tests/test_labeling_range.py`, `tests/test_gui_commands.py`（いずれも torch 不要）。
