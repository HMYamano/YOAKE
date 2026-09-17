# Annotator Project Workspace — 実装・検証記録

## Architecture changes

`ProjectSession` が共通ラベル、動画一覧、設定、保存先、サンプリング履歴を所有し、
各 `VideoSession` がアノテーション、候補、品質レビュー、行動区間、作業位置、Undo/Redoを所有します。
既存Canvasと編集操作は現在のVideoSessionを参照し、frame番号やtrack_idが別動画と衝突しない構造にしました。

GUIの `AnnObject` は編集用アダプターとして独立させ、正式な `ObjectAnno` / `DatasetAnno`
を経由して保存します。クラス、AABB/OBB、OBBの由来、キーポイント、行動、source、score、provenanceを保持します。

保存時には全動画の切り離したスナップショットを作成し、一時ファイルへの書込み後に置換します。
Project manifestは確定したデータ世代を参照します。Autosave中に新しい編集が入った場合は
その編集のdirty状態を残し、次の保存対象にします。最近のデータ世代を2つ保持します。

動画デコードは開く動画／バッチ対象に限定します。推論は既存CLIを子プロセスとして実行し、
特徴抽出・一括サンプリング・品質検査・フレーム書き出しはworkerを使います。
遅れて返った処理結果は元のProjectと動画を照合して適用します。

## Modified files

| ファイル | 変更 |
|---|---|
| `src/htrtdetr/annotator.py` | Project UIとの接続、動画別編集履歴、クラス編集、候補Accept/Reject、行動区間、既存起動方法と新CLI引数 |
| `src/htrtdetr/data/annotation.py` | 正式スキーマでのprovenance保持、atomic保存、derived OBBの編集用保存オプション |
| `src/htrtdetr/data/__init__.py` | データセットのPyTorch依存を遅延import |
| `src/htrtdetr/annotation/quality.py` | レビュー項目のvideo_id対応 |
| `src/htrtdetr/gui/auto_label.py` | 候補から正式アノテーションへのクラス・provenance保持 |
| `src/htrtdetr/gui/click_segment.py`, `video_segment.py` | クリック／伝播のクラス情報保持 |
| `src/htrtdetr/gui/inspector.py`, `settings.py`, `smart_annotate.py` | 目的別操作、Continue this animal、Project処理範囲と結果表示 |
| `tests/test_gui_auto_label.py`, `tests/test_workflows.py` | 環境に依存しないcapabilityテスト、非同期Autosave検証 |
| `docs/ANNOTATOR_GUIDE.md`, `docs/gui.md`, `docs/MIGRATION.md` | 現在のUI、保存形式、移行方法 |

この一覧はWorkspace改修に関係するファイルです。作業開始時点にあったモデル・学習等の別変更は維持しています。

## New files

| ファイル | 責務 |
|---|---|
| `annotation/project.py`, `annotation/session.py` | Project／動画の状態、旧JSON取込み、スナップショットの保存・復元 |
| `annotation/objects.py`, `annotation/storage.py` | GUI編集モデルとatomic JSON書込み |
| `annotation/batch.py` | 動画単位の推論、キャッシュ、キャンセル、失敗分離、動画をまたぐサンプリング配分 |
| `gui/workspace.py`, `gui/workspace_operations.py` | 既存アプリとの接続、非同期保存、レビュー、追跡結果の反映 |
| `gui/video_browser.py`, `gui/batch_progress.py` | 複数選択・フィルタ付き動画一覧、処理範囲と進捗 |
| `gui/onboarding.py`, `gui/workspace_help.py` | 初回ガイド、ワークフローと実登録キーに基づくHelp |
| `tests/test_annotation_project.py` | 保存、互換性、キャッシュ、バッチ、依存関係の回帰テスト |
| `tests/test_project_workspace_gui.py` | 実Viewportを使う複数動画の編集・保存・再開テスト |

コードの相対パスは `src/htrtdetr/` を基準とします。

## User workflow

Create Project → Add Folder → Project Videosで動画を選択 → Draw／Click animal →
ClassとBehaviorを設定 → Continue this animal → 必要に応じてAuto Annotate Project →
Review Nextで動画をまたいでAccept／Reject／修正 → Mark Done → Save and Export Frames。

Current video、Selected videos、All unfinished videos、All unannotated videos、Entire projectを
一括処理の対象にできます。サンプリング予算はProject全体で配分し、動画間の均等配分も選べます。
AI候補はAcceptするまで学習用GTへ入りません。

## Compatibility

- 従来の単一動画起動と `--load annotations.json` を維持しました。
- 旧JSON、複数動画で同じframe番号、class_id省略、AABBのみのデータを読み込めます。
- 正式データは引き続き全動画を含む `annotations.json` です。Project状態は別manifestに保存します。
- 別動画の同じtrack_idを自動統合しません。
- キーフレーム、derived／annotated OBB、候補の判断、作業位置をProject再開時に復元します。
- DearPyGui／OpenCVがなくてもimport時に終了せず、GUI起動時に依存関係を案内します。

## Test results

Windows上で実行。PyTorchありの検証には一時的なPython 3.12環境、なしの検証には既存のPython 3.14環境を使いました。

| 検証 | 成功 | 失敗 | スキップ |
|---|---:|---:|---:|
| PyTorchあり: リポジトリ全体の `pytest -q`（examples、実GUIを含む） | 800 | 0 | 0 |
| PyTorchなし: Annotation／GUI関連12ファイル | 479 | 0 | 1 |
| 最後のWindows描画領域・入力幅調整後: Workflow／Project GUI | 48 | 0 | 0 |

PyTorchなしの1件は、インストール済みPyTorchの検出を検証する既存テストです。
PyTorchなしを想定した手動操作、追跡、保存、一括サンプリングとGUIテストは成功しています。

再現コマンド:

```powershell
# CPU版PyTorchを含む検証環境
.tmpbuild/workspace-test-env/Scripts/python.exe -m pytest -q

# PyTorchなしの既存環境
python -m pytest -q -ra tests/test_annotation_core.py tests/test_annotation_v2.py tests/test_annotation_project.py tests/test_annotator_wiring.py tests/test_gui_auto_label.py tests/test_gui_click_segment.py tests/test_gui_shortcuts.py tests/test_gui_sidecar.py tests/test_gui_timeline.py tests/test_gui_video_segment.py tests/test_workflows.py tests/test_project_workspace_gui.py

# 最後のレイアウト修正に対する検証
python -m pytest -q tests/test_workflows.py tests/test_project_workspace_gui.py
```

実GUIではProject作成、フォルダ追加、2動画の編集、切替、Undo/Redo、保存・再読込、
動画をまたぐReview Next、選択動画のバッチ、全動画サンプリング、フレーム書き出しを検証しました。
Canvas／Inspector／Project Videosの表示も画像で目視確認しました。

静的チェック:

- 変更したPython 25ファイル: Black、isort、構文／未定義名／未使用import・変数のRuff検査。
- 新規13ファイル: Ruffの全有効ルール。
- 新規データモデル・保存・バッチ5ファイル: mypy成功。
- Python 3.10構文としてのAST検査、`git diff --check` 成功。

mypyは導入済みNumPyの型定義がPython 3.12構文を使うため、`--python-version 3.12 --follow-imports=silent`
で実施しました。アプリのPython 3.10実行検証は未実施です。
検証ログは `.tmpbuild/workspace-final-all.log`、`workspace-no-torch-final.log`、`workspace-layout.log` にあります。

## Known limitations

- 学習済みの実checkpoint、GPU推論、外部SAM helperを使う実データ検証は未実施です。
  バッチの候補生成・キャッシュ・部分失敗・キャンセルはテスト用predictor／子プロセスで検証しています。
- Undo/Redoは起動中に動画別で保持します。履歴自体はアプリ再起動をまたいで復元しません。
- 推論再開は完了した動画のキャッシュを単位とします。処理途中の動画は再実行します。
- 数百動画・長時間動画でのメモリ使用量と待ち時間は未計測です。
- 動画間の個体同定は対象外です。隣接フレームの簡易接続も候補として人間が確認します。
