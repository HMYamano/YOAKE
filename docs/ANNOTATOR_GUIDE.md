# YOAKE Annotator — Project Annotation Workspace

複数動画をひとつのProjectとして管理し、手作業・AI候補・人間によるレビューを組み合わせてデータセットを作成します。AI候補はAcceptするまで学習用の `annotations.json` に入りません。

## 起動

```bash
pip install -e ".[annotate]"
annotate
annotate video.mp4
annotate video.mp4 --load annotations.json
annotate --project experiment/project.yoake.json
annotate --folder videos --recursive
```

`python tools/annotate.py` でも同じ引数を使用できます。従来の単一動画は内部的に1動画のProjectになります。DearPyGui・OpenCVがなくてもモジュールのimportは可能で、GUIを起動する時点で必要な依存関係を案内します。

## 最初の作業

1. **Create Project** で名前・保存フォルダ・Classes・Behaviors、そして **Follow each individual across frames (track IDs)** を指定します。ラベルの並び順がIDになります。
2. **Add Files... / Add Folder...** で動画を登録します。フォルダの再帰走査は任意です。MP4、AVI、MOV、MKV、WebM、M4Vを対象にし、ドットで始まるファイルやフォルダを除外します。
3. **Project Videos** で動画を選びます。動画を開くまで全フレームの読み込みは行いません。
4. `W` でboxを描くか、`C` の **Click animal** で個体をクリックします。Inspectorでクラスと行動を設定します。
5. boxを選択して **Continue this animal** で続きを作成します。
6. **Auto Annotate Project / Smart Annotate** で候補・確認するフレームを生成します。
7. **Review Next** で動画をまたいで確認し、Accept / Reject / 編集を行います。
8. 完了した動画を **Mark Done** にし、`Ctrl+S` または **Save and Export Frames** で保存します。

初回には4段階のGuided Annotationを表示します。いつでも閉じられ、Projectごとに自動表示を無効化できます。Helpから再表示できます。

## 画面の役割

| 領域 | 操作 |
|---|---|
| 上部 | Project名、現在の動画、動画順、Previous/Next Video、Review Next、Save |
| Tool Rail | Select、Draw Box、Draw Rotated Box、Click animal、Pan、Delete |
| Canvas | 現在動画のフレーム、アノテーションと候補。ズーム・パン・boxの編集 |
| Inspector | 選択対象のClass、Track、Behavior、OBB、次にできる操作 |
| Review Inbox | 現在動画の品質問題。Project Review Inboxには全動画の問題と候補を集約 |
| Timeline | 現在動画のアノテーション、トラック、行動、レビュー、サンプリング位置 |
| Project Videos | 動画の切替、検索、状態フィルタ、複数選択、一括処理、欠落ファイルの再リンク |

Timelineは動画ごとに切り替わります。全動画をつないだ一本のタイムラインではありません。

## 動画の切替と作業再開

**Project Videos** または上部のPrevious/Next Videoを使用します。動画ごとにアノテーション、候補、Undo/Redo、最後のフレーム、選択対象、ズーム・パン、行動区間を保持します。Undo/Redoは現在の動画だけに作用します。複数フレームのAcceptやID交換もひとつのUndoで戻せます。

`track_id` は動画内のIDです。動画AのTrack 1と動画BのTrack 1は別物で、自動的に同一個体へ統合しません。

## Track IDを使うか使わないか

個体を追わない実験（フレームごとの検出数や行動だけを見る場合）では、Track IDは「埋めろと言われ続ける欄」であり「決して発火しないチェック」です。Project設定 **use_track_ids** でProject単位に切り替えます。

- 設定場所: Create Projectの **Follow each individual across frames (track IDs)**、または **Project > Tracking...**
- 既定は**オン**です。この設定より前に作られたProjectはオンとして読みます（挙動は一切変わりません）。

オフにすると、

| 変わるもの | 内容 |
|---|---|
| 新規box | `track_id = -1`（JSON v1.1/v2.0の「同一性を主張しない」表現そのもの。新しいフィールドは増えません） |
| Inspector | New boxes use track と Track欄を非表示 |
| box の色・ラベル | Track色ではなくClass色。ラベルは `T3\|walking` ではなく `fly\|walking` |
| キャンバス右クリック | Set Track ID... を非表示 |
| `N`（New track）・`T`（Set Track ID）・Follow boxes to the next frame | 拒否し、設定の場所を状態行に表示 |
| Continue this animal | Copy to next frame のみ（Fast tracking / Accurate propagationは同一性を運ぶ操作なので） |
| 品質チェック | Possible ID switch・Track stops and restarts・位置/サイズ/向きの急変を実行しない。低確信度・個体数・重なり・merged boxは従来どおり |
| Auto Annotate Project | Connect detections into tracklets をロック |

**すでに保存されているTrack IDは書き換えません。** チェックボックス1つで既存の作業を消し、戻しても復元できないという事態を避けるためです。オフは「これから」の挙動だけを変えます。

解析側（3. Analyze）にも対になる **Use track identity in this analysis** があります。こちらは読み方の宣言で、`predictions.json` の `track_ids` は書かれたままです。詳細は [gui.md](gui.md) を参照してください。

Settings > Project > **At video end** で、最終フレームに留まるか、次の未完了動画へ進むかを設定できます。移動先ではその動画の作業位置から再開します。

Project Videosの状態は次の意味です。

| 状態 | 意味 |
|---|---|
| New | まだ訪問・アノテーションしていない |
| In progress | 作業を開始、または手動でこの状態に変更した |
| Needs review | 未確認の候補または未解決の品質問題がある |
| Done | ユーザーが明示的に完了にした。未確認候補がある場合はNeeds reviewを優先 |

**Visited** は訪問したフレーム数です。アノテーション完成率を意味しません。boxがないフレームも正当なデータとなるため、boxのあるフレーム割合から完了判定しません。

## 個体を見つける・続きを作る

- **Draw Box / Draw Rotated Box**: 手作業でboxを作成。InspectorのNew classが新規boxのクラスです。
- **Click animal**: 起動済みの画像ヘルパー、読み込み済み予測、ローカル画像処理の順に利用可能な方法を選びます。予測から選んだboxのクラスも保持します。`C` は **1回1箱** です。boxを作ると自動的にSelectへ戻り、その新しいboxが選択された状態になるので、そのままハンドルを掴んで調整できます（連続でクリックしたい場合は Tool Railの `[=]` または Settings > Assistance > Keep drawing after each box）。
- **Fast tracking**: 選択個体を最大100フレーム先まで追跡し、レビュー用候補を作ります。
- **Accurate propagation**: 動画伝播モデルを使って候補を生成します。利用できない場合は理由を表示します。
- **Copy to next frame**: 選択boxをクラス・行動・OBBとともに次のフレームへコピーします。手動コピーはUndo可能です。

Continue this animalのAutomaticは、動画伝播が利用可能ならそれを、次に高速追跡、最後にコピーを選びます。Settings > Trackingから既定を指定できます。

Click animalのバックエンド指定は **Settings > Assistance > Advanced assistance** にあります。既存の `Space` による前フレームからのコピーと `A` による次フレーム追跡も使用できます。追跡結果は候補として確認します。

## Auto Annotate Project

Projectメニュー、またはProject Videosの **Auto Annotate...** を開きます。

- Scope: Current video / Selected videos / All unfinished videos / All unannotated videos / Entire project
- Checkpoint: 学習済みモデルのファイル。必要ならConfigも指定
- Generate object candidates: 既存のpredict CLIを別プロセスで実行
- Connect detections into tracklets: 既存IDを維持し、IDがない検出は同じクラスの隣接フレーム間で関連付け
- Select informative frames / clips: レビューするフレームや行動区間を提案

依存関係やcheckpointが不足している場合、検出オプションを無効化し理由を表示します。推論が不要なサンプリング・手作業はそのまま利用できます。自動Acceptは行いません。

現在動画・全体の進捗、完了・キャッシュ再利用・失敗を表示します。Cancelは進行中の推論プロセスを停止し、完了済み動画の結果を残します。1動画が失敗しても他の動画を処理し、失敗理由を結果一覧に残します。

キャッシュキーには動画のパス・サイズ・更新日時、モデル、設定を含めます。同じ条件なら完了済みの推論を再利用します。再実行しても既存のAccept/Rejectや人手アノテーションを上書きしません。

## Smart Annotate / Smart Sample

Smart Annotateでも対象動画のScopeを選択できます。サンプル数は対象Project全体の予算です。**Balance samples across videos** を有効にすると各動画に基本配分し、余った枠を全体のスコアで選びます。短い動画で取れなかった枠は再配分します。

既存のuniform、random、diversity、uncertainty、interactionなどの戦略を使用します。見た目の特徴は利用可能なembedding providerを使い、追加モデルがなければ組み込みのフレーム記述子にフォールバックします。

サンプルはそれぞれの動画のTimelineに表示します。動画の長さや重ならないclip数により、実際に選べる数が予算を下回る場合があります。選択内容・seed・Scope内の動画IDは `sampling_log.json` に保存します。

Smart AnnotateのGenerate annotation candidatesは、Auto Annotate Projectで設定したcheckpointを使用して実際に推論します。モデル未設定の場合は無効化されます。予測ファイルの読込はFile > Load Predictions as Candidatesから行えます。

## レビューと品質チェック

**Project Review Inbox** は全動画を集約し、**Review Next** は次の問題・候補のある動画とフレームへ移動します。欠落動画はスキップして残し、Locate fileで復旧できます。

**Check Project Quality** またはProject VideosのQuality Checkは、位置・サイズ・方向の急変、重なり、個体数の不一致、トラックの途切れ、ID交換の疑い、低確信度、行動区間の問題を既存検出器で調べます。モデルの疑いは人間が判断します。

Acceptした候補だけがアノテーションになります。クラス、トラック、行動、OBB、source・score・モデル等のprovenanceを保持します。Rejectした候補とその判断はProject側に保持します。

## 保存・復元・エクスポート

```text
project/
  project.yoake.json        # 動画順、状態、位置、設定、候補、レビュー等
  annotations.json         # 全動画の正式なDatasetAnno、学習・他ツール用
  sampling_log.json         # サンプリング履歴
  .yoake/snapshots/         # 保存世代。Project再開に使用する確定データ
  .yoake_cache/             # 再生成できる推論・特徴キャッシュ
```

**Projectを移動するときは `.yoake` を含むフォルダ全体をコピーしてください。** 動画のパスは可能なら相対パスで保存します。別ドライブの場合は絶対パスが必要です。動画が見つからない場合もデータは保持し、Project Videos > **Missing: Locate...** で再リンクできます。

保存は一時ファイルへの書込み・flush・置換を用い、Projectは確定済みのデータ世代を参照します。Autosaveは約4秒の待機後にバックグラウンドで書き込みます。新しい編集が途中で入った場合は次の保存へ回します。保存先を指定していないProjectを閉じる・別Projectに切り替える場合はOSの一時フォルダ `YOAKE/recovery` に回復用データを残します。

**Save and Export Frames** は全動画のアノテーション対象フレームを `images/<video_id>/000000.jpg` 形式で出力します。取り込んだデータに別の画像パスが指定されていても、保存JSONはこの出力先に揃えます。処理はバックグラウンドで行い、欠落動画や読めないフレームを報告します。

Undo/Redo履歴はアプリを開いている間、動画ごとに保持します。再起動時は保存されたアノテーション・候補・レビュー判断・補間用キーフレームから再開し、Undo/Redo履歴自体は復元しません。

旧 `annotations.json` はそのまま開けます。動画情報だけで対応が確定できない場合はLocate fileを使ってください。同じframe番号を持つ複数動画も独立した状態として取り込みます。single-classの省略されたclass_idは0として扱います。annotated / derived OBBを区別し、GUI編集用の情報もProject保存で維持します。

## ショートカットとHelp

既存ショートカットは維持しています。**Help > Quick Start and Shortcuts** は実際のキー登録表から一覧を生成します。

- `W`: Draw Box、`C`: Click animal、`Esc`: キャンセル／選択解除
- `Right / N`、`Left / P`: フレーム移動
- `Space`: 前フレームからコピー、`A`: 次フレーム追跡
- `Ctrl+S`: 全動画を保存、`Ctrl+Z` / `Ctrl+Y`: 現在動画のUndo / Redo
- `Delete`: 選択box削除、数字キー: Behavior設定

回転、ズーム、レビューなどの詳細キーはbuilt-in Helpを参照してください。
