# YOAKE Annotator — Project Annotation Workspace

複数動画をひとつのProjectとして管理し、手作業・AI候補・人間によるレビューを組み合わせてデータセットを作成します。AI候補はAcceptするまで学習用の `annotations.json` に入りません。

## 起動

```bash
pip install -e ".[annotate]"
annotate
annotate video.mp4
annotate video.mp4 --load annotations.json
annotate --project experiment/               # フォルダでもよい
annotate --project experiment/project.yoake.json
annotate --folder videos --recursive
```

`python tools/annotate.py` でも同じ引数を使用できます。従来の単一動画は内部的に1動画のProjectになります。DearPyGui・OpenCVがなくてもモジュールのimportは可能で、GUIを起動する時点で必要な依存関係を案内します。

## 最初の作業

1. **Create Project** で名前・プロジェクトフォルダ・Classes・Behaviors、そして **What do you label?**（既定は Keyframes。Tracksを選ぶと **Follow each individual across frames (track IDs)** も指定できます）を指定します。ラベルの並び順がIDになります。フォルダ欄は `~/YOAKE/<Project名>` が既定で、名前を変えると追従します（自分で編集すると追従を止めます）。**Copy the videos into the project folder** を選ぶと動画も取り込み、フォルダ単体で完結します。
2. **Add Files... / Add Folder...** で動画を登録します。フォルダの再帰走査は任意です。MP4、AVI、MOV、MKV、WebM、M4Vを対象にし、ドットで始まるファイルやフォルダを除外します。
3. **Project Videos** で動画を選びます。動画を開くまで全フレームの読み込みは行いません。
4. **Pick frames to label**（`Ctrl+R`）で最初にラベルを付けるフレームを選びます。学習済みモデルは不要です。
5. `C`で個体の中心をクリックしてboxを作り、ハンドルで調整します。`W`で手描きもできます。Inspectorでクラス・行動を設定し、必要ならboxを選んで **Continue this animal** で続きを作成します。
6. 選ばれたフレームへは `Ctrl+Right` で順に進めます（Timelineの丸を直接クリックしても飛べます）。必要に応じて **Pick frames to label** で次のサンプルを選びます。学習済みモデルがある場合のみ、同じ窓の **Generate annotation candidates** で候補も出せます。
7. **Review** で候補と品質問題を確認し、Accept / Reject / 編集を行います。動画をまたぐときは Review > **Go to the next video needing review**。
8. 完了した動画を **Mark Done** にし、`Ctrl+S` または **Save and Export Frames** で保存します。
9. ラベルが十分たまったら、上部バーの **Train** を押します。**このウィンドウの中で**学習が走り、できたcheckpointがこのProjectのモデルになります（Modelチップが変わります）。続いて **Analyze** を押すと、そのモデルで動画を推論し、**モデルが自信を持てなかった場所**を優先してReview Inboxに集めます。ここまでがアクティブラーニングの1ラウンドで、Workflowウィンドウを開く必要はありません。

初回には5段階のGuided Annotationを表示します。内容は上の**モデル不要のループ**そのもの（フレームを選ぶ → 個体を出す → 続きを作る → 確認 → Trainしてもう一周）で、`W` で frame 0 から描き始めよ、とは言いません——1本1時間×20本では、それは同じような200枚を作る道だからです。最後の **Start annotating** を押すと以後は自動表示しません。Helpから再表示できます。

## 画面の役割

| 領域 | 操作 |
|---|---|
| 上部1段目 | Videos / Previous Video / Next Video、保存状態チップ、Modelチップ、**Activityチップ**、Project名・現在の動画。**「いまどこにいるか」だけ** |
| 上部2段目 | ループを左から右へ: **Pick frames to label → Review → Train → Analyze**。4つとも**このウィンドウの中で完結**します。続けて Undo / Redo / Save、Settings |
| Tool Rail | **Tools**: Select、Box、Rotate、Click、Pan、Delete / **Assist**: Continue、Keep tool / **View**: Fit、Panels、ズーム率 |
| Canvas | 現在動画のフレーム、アノテーションと候補。ズーム・パン・boxの編集 |
| Inspector | 選択対象のClass、Track、Behavior、OBB、次にできる操作 |
| Review Inbox | 現在動画の品質問題。Review > **Review Inbox (all videos)** に全動画の問題と候補を集約 |
| Timeline | 現在動画のアノテーション、トラック、行動、レビュー、サンプリング位置 |
| Project Videos | 動画の切替、検索、状態フィルタ、複数選択、一括処理、欠落ファイルの再リンク |

Timelineは動画ごとに切り替わります。全動画をつないだ一本のタイムラインではありません。

## 動画の切替と作業再開

**Project Videos** または上部のPrevious/Next Videoを使用します。動画ごとにアノテーション、候補、Undo/Redo、最後のフレーム、選択対象、ズーム・パン、行動区間を保持します。Undo/Redoは現在の動画だけに作用します。複数フレームのAcceptやID交換もひとつのUndoで戻せます。

`track_id` は動画内のIDです。動画AのTrack 1と動画BのTrack 1は別物で、自動的に同一個体へ統合しません。

## キーフレームで付ける（新規Projectの既定）

新しいProjectは **Keyframes** 方式で作られます（Create Projectの **What do you label?**）。考え方と先行研究との違いは [KEYFRAME_BEHAVIOR.md](KEYFRAME_BEHAVIOR.md) にあります。

**人が付けるのは「選んだフレームの、全個体の箱と行動」だけです。** Track IDも行動の区間も付けません。個体の同一性は解析時のトラッキングが作り、行動のbout（始まりと終わり）は解析時にモデルのフレームごとの確率をtrackに沿って復号して得ます。学習では、各キーフレームの直前のクリップ（既定 8フレーム×4おき ≒ 1秒）を動画から切り出して時間文脈に使うので、そのためのラベルは要りません。

| 操作 | 内容 |
|---|---|
| 箱 | いつもどおり `C` / `W` / `E`。そのフレームの**全個体**を囲みます（囲まない場所は背景として学習されます） |
| 行動 | 箱を選んで数字キー `1`〜`9`（Inspectorの Behavior でも可）。判断できない個体は付けなくて構いません — 箱は検出の学習に使われ、行動の学習からは外れます |
| **`P` 文脈ループ** | そのフレームの **1.0秒前〜0.5秒後** を繰り返し再生します。キーフレームの箱は輪郭だけ重なり、ループ中に数字キーを押してもラベルはキーフレームに付きます。`P` / `Esc` / キャンバスのクリック / フレーム移動で戻ります |
| 箱の色 | 行動の色（Timelineと同じ）。行動未判定は灰色・塗りなし |
| Inspector | 「3 animals - 1 still needs a behavior.」のように、そのキーフレームの残りを表示 |
| Timeline | Behaviorレーンは区間ではなく、ラベルされた個体ごとの行動の目盛り。行動の例がまだ無い区間が一目で分かります |
| Train（ラウンド） | キーフレームに加えて**クリップの文脈フレーム**も書き出し、detector（stage 1）→ 行動（stage 2、クリップ）の順に学習して後者を採用します。行動ラベルが8個未満ならdetectorだけを学習し、そう表示します |

Settings > Tracking > **What do you label?** でいつでも切り替えられます。**切り替えても保存済みのTrack IDや区間は書き換えません。** 既存Project（この設定が無いもの）は **Tracks** として開きます。`meta.labeling.mode = "keyframe"` を持つ `annotations.json` を開くと Keyframes のProjectになります。

学習後の解析は `yoake predict` → `yoake analyze mode=bouts predictions=... output_dir=...` です（bouts.csv / summary.json / population.csv / proximity.csv）。

## Track IDを使うか使わないか

この節は **Tracks** 方式のProjectの話です（Keyframes方式では常にTrack IDを使いません）。


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
| Generate annotation candidates | tracklet の接続を行わない（設定から導出するので、チェックボックス自体がありません） |

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

- **Draw Box / Draw Rotated Box**: 手作業でboxを作成。InspectorのClassが新規boxのクラスです。
- **Click animal**: `C` を押して個体の中心をクリックします。既定のAutomaticでは、ヘルパーも予測もない場合、最初のクリックでインストール済みの補助エンジンを探索・起動します。初回は時間がかかりますが、クリックは記憶され、準備ができると実行されます。使えない環境ではローカル画像処理に切り替わり、以降のクリックで起動を繰り返しません。待っている間にフレーム・ツール・クラス等を変えた場合、そのクリックは取り消されます。`C` は **1回1箱** で、作成後はSelectへ戻り、ハンドルで調整できます。連続作成は Tool Rail の Assist にある **Keep tool** または Settings > Assistance > Keep drawing after each box で指定します。
- **Find all at once (optional) > Find all: &lt;class&gt;**: Inspector内の任意の一括補助です。自分で学習した検出器は不要ですが、名前から対象を探せるAIヘルパーが必要です。探す語はInspectorのClassに従います。結果はpending候補として入り、Acceptするまで学習用アノテーションには入りません。利用できない場合はボタンに理由を表示します。`C`による1匹ずつのクリックや`W`による手描きは引き続き使えます。
- **Fast tracking**: 選択個体を最大100フレーム先まで追跡し、レビュー用候補を作ります。
- **Accurate propagation**: 動画伝播モデルを使って候補を生成します。利用できない場合は理由を表示します。
- **Copy to next frame**: 選択boxをクラス・行動・OBBとともに次のフレームへコピーします。手動コピーはUndo可能です。

Continue this animalのAutomaticは、動画伝播が利用可能ならそれを、次に高速追跡、最後にコピーを選びます。Settings > Trackingから既定を指定できます。

Click animalのバックエンド指定は **Settings > Assistance > Advanced assistance** にあります。既存の `Space` による前フレームからのコピーと `A` による次フレーム追跡も使用できます。追跡結果は候補として確認します。

## モデルの置き場所（Project > Settings > Model...）

**checkpoint が要るのは事前ラベリングだけです。** 下のループは checkpoint 無しで回るので、ここは「学習を1周させた後、ラベリングを速くするために戻ってくる場所」です。未設定でも欠陥ではありません。

checkpoint は Project の設定です。**Project > Settings > Model...**（= Settings の Model タブ）で
Checkpoint と Config を指定します。いま何が設定されているかは上部バーの
**Model: none / Model: best.pth** チップで分かり、チップをクリックすると同じタブが開きます。
このタブは常に「この環境で実際に推論できるか」の1行を表示します（PyTorch の有無、
ファイルの有無、その両方が揃った場合の "Ready to run auto-annotation."）。

モデルが無いときは、フレーム選択と`C`でラベルを作れることを案内します。
**Choose a model...** は **Advanced** 内にあります。新規開始時に選択する必要はありません。

## Pick frames to label

**入口は1つです。** かつて Auto Annotate Project・Smart Sample Project・Smart Annotate
という3つの窓が1つの処理を呼んでいたものを1つの窓にまとめ、さらにその窓を指す
*Pre-label videos with the model...* という3つ目の名前（Projectメニュー・Inspector・
Project Videos）も削除しました。checkpoint が無いとき、その3つは隣の
*Pick frames to label* と**まったく同じ動作**でした——`sample=True` で開き、
prelabel の指定は捨てられ、窓のタイトルだけが、その窓が直後に隠すチェックボックスの
名前を名乗っていたためです。

事前ラベリングは窓の中の **Generate annotation candidates** です。checkpoint が
あるときだけ表示されます。

| 入口 | 開くもの |
|---|---|
| 上部バー **Pick frames to label** / `Ctrl+R` | Scope = 現在の動画（動画未オープンならProject全体） |
| Project > **Pick frames to label (whole project)...** | Scope = Project 全体 |
| Project Videos の **Pick frames to label...** | 選択中の動画、無ければProject全体 |

Projectメニューの行には `Ctrl+R` を表示しません。同じラベルで Scope だけが違い、
しかもそのキーは**現在の動画**を開くからです。Scope は窓の中の1つ目のコンボで
いつでも変えられます。

窓の中身は次のとおりです。

- **Scope**: Current video / Selected videos / All unfinished videos / All unannotated videos / Entire project。切り替えると見出しの「6 videos, 241,900 frames」が追従します
- **Annotate**: Individuals / Behavior / Individuals + behavior
- **Labeling budget**: サンプル数（50/100/200）または時間（5/15/30分）。予算はScope全体に対する数です
- **Generate annotation candidates**: 設定したcheckpointで候補を生成します。checkpointがない場合は非表示です。モデルの選択はAdvanced内にあります。
- **Choose which frames to label**: 候補生成を有効にしたときだけ表示します。オフにすると候補生成のみ実行します。候補生成が無効ならフレーム選択は常に有効で、何もしない実行にはなりません。
- **Use the GPU**: 検出器のデバイスだけを決めます。checkpoint が無いときは非表示です（`BatchOptions.device` は `if options.detect:` の中でしか読まれないので、検出しない実行では何も変わりません。フレーム走査は常に CPU、セグメンテーション補助は自分のプロセスで自分で決めます）
- **Balance samples across videos**: 複数動画のときだけ表示します。各動画に基本配分し、余った枠を全体のスコアで選びます。
- **Advanced**: strategy（既定は多様性に基づく選択。等間隔もここで選択）、random seed、clip length、scan stride。candidate score threshold（既定 0.3）はcheckpointがあるときだけ表示します。

実行中は進捗、完了後は結果に画面が切り替わります。**Start on N samples**でラベル付けへ進み、設定を変える場合は**Change options**で戻ります。サンプルがなくレビュー対象だけがある場合は、候補数付きのReviewボタンを表示します。

### 走り出す前に、かかる時間を出します

見積りカードは**2つ**の数を出します。**Scanning takes about N minutes**（機械の時間）と
**Your labelling afterwards**（あなたの時間）です。以前は後者しか出しておらず、
20本×1時間（1,475,270フレーム）に対して Analyze を押すと、何も予告されないまま
1時間超の走査が始まっていました。

走査は **seek をやめて順方向に grab** します。同じ動画で実測すると、1スキャンフレームあたり

| stride | seek | grab 送り |
|---|---|---|
| 15（既定） | 19.5 ms | **4.0 ms** |
| 30 | 15.7 ms | 8.2 ms |
| 60 | ~16 ms | 16.8 ms |

既定 stride では**約5倍**速く、60を超えると損になるのでそこから先は seek に戻します。
返るフレームは同一です（実測1000枚すべてバイト一致）。この `read` は
encode と scan で**同じ列を2回**読むので、効果はそのまま2倍効きます。

**stride は自動で決まります**（`sampling.scan_stride`）。scope の長さから導き、
下限15・上限2秒分。例えば1本1時間なら従来どおり15、20本なら60です。
予算（50/100/200）からは**導きません**：フレーム特徴量のキャッシュキーに stride が
入っているので、100⇒200 に変えただけで全動画を再走査することになるためです。
20本×1時間で走査位置は 144,000 → 36,000 になります。

**そして Review に入るのは選ばれた50〜200フレームだけです**（`batch.keep_chosen`）。
以前は走査した全フレームの候補を丸ごと Review Inbox に流し込んでおり、
100枚頑しいだけなのに20本×1時間だと約7万2千個の候補が入っていました。
サンプリングは全動画を走査し終わってからしか確定しないので、マージは最後に1回だけ行われます。
**途中で止めた場合は何もマージされません**（選択が確定していないため）。
ただし走査結果はキャッシュに残るので、やり直しは速いです。

Scope はダイアログを開いた時点で全動画の尺を読みます（ヘッダのみの読み取りで、
実測20本80ms）。以前は1本でも未計測の動画があると合計を出さず、
「Total length appears once every video in scope has been read.」とだけ表示していました
——プロジェクトを作った直後、つまりその数字が一番必要な日に、です。

**Connect detections into tracklets のチェックボックスはありません。** Project の
`use_track_ids` から導出します（オンなら繋ぎ、オフなら繋ぐ先が無い）。
どちらの答えも1つしか無い問いだったためです。

実行中は、いま処理している動画の5段階（Reading the video / Generating annotation candidates /
Checking tracks and behavior / Computing embeddings / Choosing informative frames）と
その動画の進捗バー、さらに **Video 3 of 6** の行・全体の進捗バー・動画ごとの結果一覧を表示します。
段階の並びは `BatchRunner` が実際に行う順序です。

Cancelは進行中の推論プロセスを停止し、完了済み動画の結果を残します。1動画が失敗しても
他の動画を処理し、失敗理由を結果一覧に残します。自動Acceptは行いません。

終了時は **やったことの次に来る行動** をボタンで出します。サンプルを選んだなら
**Start on 12 samples**、候補や品質問題ができたなら **Review 412 candidates and 37 locations**
（押すとレビューに入ります）。どちらも0件なら、押しても何も起きないボタンは出しません。

既存のuniform、random、diversity、uncertainty、interactionなどの戦略を使用します。見た目の特徴は利用可能なembedding providerを使い、追加モデルがなければ組み込みのフレーム記述子にフォールバックします。

選ばれたフレームには2通りで行けます。**`Ctrl+Right` / `Ctrl+Left`** が次 / 前へ移動し、
状態行に `Sample 12 of 65` と出ます。**Timelineのsamplesレーンの丸は直接クリックできます**
（±6pxの許容つき）。以前はどちらも無く、`Ctrl+Right` は候補が無いと
「No candidates are loaded.」と答え、丸は描かれるだけで当たり判定が無かったので、
選ばれた65フレームのうち**実際に行けるのは最初の1枚だけ**でした。
`Ctrl+Right` は候補があればそちらを優先し、無ければ選ばれたフレームを辿ります。

サンプルはそれぞれの動画のTimelineに表示します。動画の長さや重ならないclip数により、実際に選べる数が予算を下回る場合があります。選択内容・seed・Scope内の動画IDは `sampling_log.json` に保存します。

キャッシュキーには動画のパス・サイズ・更新日時、モデル、設定を含めます。同じ条件なら完了済みの推論を再利用します。再実行しても既存のAccept/Rejectや人手アノテーションを上書きしません。予測ファイルの読込は **File > Load Predictions > As candidates to review...** から行えます（同じサブメニューの **As boxes for Click animal...** は、`C` のクリックが選ぶ箱の供給元にします）。

## ラベリングのループ（モデル不要）

**このループは学習を含みません。** ラベルが十分になったと判断してから Train / Analyze に進みます。

1. **Pick frames to label** で次に塗るフレームを選ぶ
2. 候補を出す。3通りあり、クリップ単位が最も効率的です:
   - **Continue every animal across this clip** / **across all sampled clips**（Reviewメニュー）— クリップ内で最も個体が写っているフレームの箱を種に、**全個体をまとめて**クリップ全体へ伝播します
   - **Find every &lt;class&gt; in this frame** — そのフレームの全個体をまとめて検出
   - **Continue this animal** — 1個体に1回プロンプトして前後へ伝播
3. **Review Inbox** で Accept / Reject
4. **1 に戻る** — 次の回は「すでにラベルしたフレームと似ていないもの」を選びます

5. **Train** — 上部バーのボタン。ラベル済みフレームで検出器を学習し、できた
   checkpointをこのProjectのモデルにします。**フォームはありません**（下記）。
6. **Analyze** — そのモデルで動画を推論し、uncertaintyで並べ替えて候補を
   Review Inboxに入れます。Pick frames to labelと同じ窓を、事前ラベリングを
   オンにした状態で開きます。モデルが無いうちはロックされ、理由に
   「Trainで作れる」と出ます。

### ラウンドは10回、そのあと全フレーム

`Pick → 直す → Train` を1ラウンドとして**10回前後**回すのが想定された使い方です。
そのために3つが見えるようになっています。

- **何ラウンド目か** — Trainダイアログの文章が `Round 7.` で始まります。
- **このラウンドがあと何枚か** — Reviewドックの *This round* に
  `12 of 100 frames checked here, 31 of 200 across the project.`。ボタンで次の未確認
  フレームへ飛び、全部終わるとボタンが *Train on these labels* に変わります。
  **何も写っていないフレームは `K`**（*Nothing here - checked*）。空もラベルですし、
  これが無いとカウンタが100に到達しません。
- **もう伸びないか** — 「モデルの提案のうち直さずに採用した割合」が3ラウンド
  横ばいになると、TrainダイアログとReviewドックがそう言います。holdoutスコアでは
  ありません：ラベルが増えるとholdout自体が変わるので、ラウンド間で比べられない数字だからです。

そして終わり方は **Project > Check every frame (final pass)...**です。
全動画の全フレームにモデルをかけ（20本×1時間で約2時間、途中で止めても続きから）、
チェックが引っかかった場所と抜き取り検査40箇所だけを人が見て、動画ごとにsign offします。
最後に **File > Export the verified dataset...** で
`datasets/release/annotations.json`（ラベルと動画パスのみ、画像なし）を書き出します。
sign off していない動画は**名前を挙げて**除外されます。

4 が効くのが要点です。`diversity` の farthest-point 探索は既にラベル済みのフレームを**種**として開始するため、2 回目は 1 回目の続きになり、振り直しにはなりません。1 回 20〜50 フレーム程度の小分けで回してください。何をラベルしたかが次に何を訊かれるかを変えるので、その効果は回と回の間にしか現れません。

学習済み checkpoint があるとこのループは**速くなります**（同じ窓の *Generate annotation candidates* が動画単位で事前ラベルし、uncertainty がサンプリングに加わります）。ただしここまで来るのに必須ではありません。

**検出がまだ無い間、uncertainty と interaction は答えを持ちません。** 全フレームで値が同じになり、順位付けは添字順に落ちて「冒頭30秒」を返してしまうため、この2つは検出が出るまで自動的に diversity（見た目に基づく選択）へ切り替わります。

### クリップ単位の伝播

起動方法は3つあります。

1. **Timelineのsamplesレーンでクリップを直接クリック** — Inspectorが「Sampled clip」表示になり、範囲・サンプルされた理由・**何個体をどのフレームから追うか**とボタンが出ます。クリックは選択だけで、実行はしません（伝播は数分かかり、クリックでは取り消せないため。箱をクリックしても編集ではなく選択になるのと同じ形です）。マーカーと違いクリック位置へのシークも行います — クリップは範囲なので、狙ったフレームが情報になります
2. Reviewメニューの **Continue every animal across this clip** — 再生位置を含むサンプルクリップ
3. Reviewメニューの **Continue every animal across all sampled clips** — Pick frames to labelが提案した全クリップ

Inspectorの「Would follow」欄が最も重要です。伝播はプロンプトされた個体しか追えないので、**実行前に何個体が対象かが分かります**。0個体なら何をすべきか（W / C / Find every &lt;class&gt; in this frame）を表示し、ボタンはロックされます。

- 種フレームは**クリップ内で最も個体が写っているフレーム**を自動選択します。1個体しか描いていないフレームを種にすると残りの個体がクリップ全体で提案されず、しかも結果にはその理由が出ないためです。同数なら中央寄りを選び、あなたが見ているフレームは「同数以上なら」優先します
- 中央を既定にするのは、SAM2 がプロンプトから片方向にしか伝播しないためです。中央なら順方向・逆方向の長い側が最短になり、ドリフトは走の長さに沿って蓄積します
- **1クリップにつき export は1回**です。helper が `frames_dir` 単位で inference state をキャッシュするので、2個体目以降は encode を払いません。実測（40フレーム・2個体・RTX 5070 Ti）で **1個体目 5.98秒 / 2個体目 3.09秒（52%）**、被覆は両個体 40/40
- 実行中にもう一度押すと停止します（helper 内で走っているパスは中断できないので、その後が始まらない、という停止です）
- 箱がないクリップはスキップし、**何件スキップしたかを表示します**
- track ID を切っているプロジェクトでは使えません（伝播の出力は tracklet で、identity が無いと1個体として Accept できないため）。`Continue` の挙動と同じ理由です

## Train — ラベリングGUI内での学習

**アクティブラーニングの学習はこのウィンドウで完結します。** 上部バーの **Train**
（または Project > Train on these labels...）で開く窓には**フォームがありません**。
押す前に決めることはゼロで、画面上部に「何を・どれだけ・どのくらいの時間で」を
平文で出し、Startを押すだけです。

窓は4段階を上から順に埋めていきます。

| 段階 | 走る場所 |
|---|---|
| Saving your labels | メインスレッド（未保存なら保存。学習はディスク上のJSONを読むため） |
| Collecting the frames | ワーカースレッド。ラベル済みフレームだけをJPEGに書き出す（**差分**。前ラウンドで書いた分は再デコードしない） |
| Training the detector | 子プロセス `yoake train stage=1`。進捗は `results.csv` を毎秒読んで表示 |
| Checking what it learned | メインスレッド。checkpointを採用するか決める |

**Adjust this round...**（既定で畳んである）にあるのは2つだけです。**Effort**
（Quick / Normal / Thorough）と **Use the GPU**。
それ以外はすべてProjectから導出します。導出の中身と理由は
[`src/htrtdetr/annotation/round.py`](../src/htrtdetr/annotation/round.py) にあり、
どれも**黙って壊れる**罠への対処です。

- `model.detector.head.num_classes` はクラス数から渡します。既定は1で、損失は
  `num_classes` を**背景**インデックスに使うため、2クラス projectで渡し忘れると
  `class_id=1` の個体が全部「物体なし」として学習されます。エラーは出ません。
- `data.batch_size` は学習フレーム数でクランプします。train loaderは `drop_last=True`
  なので、バッチがフレーム数を超えると**optimizer stepが0回**になり、train_loss 0.0 で
  完走してcheckpointまで書かれます。
- epoch数は**ステップ予算**から決めます（冷スタート400、温スタート150ステップ）。
  時間予算ではありません —— 1分で何ステップ回るかはラベル数に依存するので、同じ
  「12分」がラベル100枚では400ステップ、2000枚では3600ステップを意味してしまい、
  プロジェクトが育つと同じボタンが黙って45分の処理になっていました。**ステップで
  区切ると、ラベルが何枚になっても1ラウンドのコストは一定です**（実測で冷4.3分 /
  温1.7分）。見積りの分数は実測レート **0.65秒/step** から逆算して表示します
  （旧実装は0.20秒/stepという当て推量で、実測の3倍楽観的でした）。
- `data.num_workers=0`。実測でこのマシンでは2でも4でも速くならず、Windowsでは
  中断時に孤児プロセスが残ります。
- `train.val_interval = epochs//4`。checkpointは実測で `last.pth` 348MB /
  `best.pth` 249MB、これを毎epoch書いていたので100 epochで約35GBになっていました。
  最終epochが必ず検証に当たるようepoch数を`val_interval`の倍数に丸めます
  （`best.pth`は検証して改善したepochでしか書かれないため）。
- `train.save_optimizer_state=False`。ラウンドは**resumeしません**（次のラウンドは
  `init_weights` で重みだけを引き継ぎ、optimizerもepochもスケジュールも作り直す）。
  Adamはパラメータ1つにつきモーメントを2つ持つので、checkpointの約2/3は一度も
  読まれない状態です。実測で **249MB → 約94MB**。
- `model.detector.head.num_decoder_layers=2`（本学習は4のまま）。**ラウンドが
  唯一アーキテクチャを変えている箇所です。** 下の「ラウンドのモデル」を参照。

### ラウンドのモデル

ラウンドが学習するのは **RT-DETR系 `rtdetr_aabb` / ResNet-18 backbone / FPN 256ch ×
3レベル / 640×640** で、本学習と同じ既定です。違うのは decoder の層数だけ。

detector単体で **18.81M param**（モデル全体は24.59M。stage 1でも temporal / id /
action / behavior は構築だけされて freeze されます）。

実測（RTX 5070 Ti, fp16, channels_last, 640px, detector単体）の内訳:

```
decoder    5.93 ms  (46%)   ← ここが主役
backbone   2.98 ms  (23%)
aifi       1.69 ms
fpn        1.18 ms
その他     1.17 ms
```

**backboneを小さくしても速くなりません。** ResNet-50でも batch 1 で15.8ms、
ResNet-18で15.0ms、入力を384pxに落としても変わりません。batch 1 では計算ではなく
カーネル起動が律速だからです。だから効くのは (a) バッチ化と (b) decoder層数の2つで、
ラウンドは両方を使います:

| | params | batch 1 | batch 16 |
|---|---|---|---|
| dec=4（本学習） | 18.81M | 73.7 fps | 350.6 fps |
| **dec=2（ラウンド）** | 16.43M | 99.9 fps | 473.0 fps |

1層まで落とすと DETR decoder の反復的な精緻化が無くなるので、2層で止めています。
**推論側は何も指定しなくても追従します** —— checkpointが自分のconfigを埋め込み、
`predict` がそれを復元するためです（`cli.load_runtime_config_from_checkpoint`）。

その代わり、decoder層数が違う前ラウンドのcheckpointからは**ウォームスタートしません**。
`init_weights` は `strict=False` で読むので、層数が違っても例外を出さずに
decoderを乱数初期化のまま残してしまうからです。この変更後の最初のラウンドは
コールドスタートになります。

### 2回目以降はウォームスタート

ラウンド2以降は前回のcheckpointから始めます（`yoake train` に追加した
`init_weights=` で**重みだけ**読み込み）。`train.resume` は使いません —— optimizerと
schedulerの状態まで復元するため学習率が減衰しきった値から再開し、さらに前回の
`best_metric` を引き継いでしまうからです。実測でウォーム3 epochの初期val_lossは
13.56、コールドは14.86でした。

クラス数が変わったラウンド（検出ヘッドの形が変わるので読み込めません）、
decoder層数が変わったラウンド（`strict=False` が黙って通してしまいます）、そして
5ラウンドに1回は強制的にコールドに戻します。最後のものは、微調整の連鎖が初期の誤りを
固定してしまわないようにするためです。

**checkpointは `best.pth` があればそれ、無ければ `last.pth`。** `best.pth` は
validationが改善したepochでしか書かれないため、「終了コード0なのにcheckpointが無い」
は実際に起こります。その場合は採用せず、Projectのモデルは変えません。

### holdoutは固定します

ラウンド間でスコアを比べられるように、検証に使うフレームは `rounds.json` に記録して
**次のラウンドでも同じものを使います**（ラベルが増えれば追加はしますが、外しません）。
毎回選び直すと「前より良くなった」が別の物差しでの比較になります。holdoutが変わった
ラウンドは比較せず、単に新しいモデル（より多くのラベルで学習したもの）を採用します。

動画が3本以上あれば**ラベル数が最も少ない1本をまるごと**、1〜2本なら**等間隔に散らした
フレーム**を holdout にします。12フレーム未満では holdout を取らず、
「学習したフレームで検証しているのでスコアは良く見える」と見積り文に書きます。

### 置き場所

ラウンドは `<project>/datasets/rounds/`（`images/` と各ラウンドのJSONを**同じ階層に**
置きます。学習器は画像パスを「アノテーションJSONのあるフォルダ」基準で解決するため、
splitsを別フォルダに置くと画像が見つからず、**黙って真っ黒な画像で学習します**）と
`<project>/runs/rounds/round-NNN/stage1/` に書きます。`runs/train/` を避けるのは、
CLIが `runs/train/*/stage{N}/weights/best.pth` を新しい順に拾って stage>1 の初期値に
するためで、ラウンドのcheckpointが本学習の出発点になってしまいます。

## Analyze — 学習したモデルで候補を出す

上部バーの **Analyze** は *Pick frames to label* と**同じ窓**を、
*Generate annotation candidates* をオンにし、strategyを
**diversity + uncertainty** にした状態で開きます（200枚しか見ていない検出器の
確信度は較正されていないので、純uncertaintyだけだと似た難所に偏ります）。

**フレーム選択を伴う実行では、検出器は `stride` おき（既定15フレームおき）にしか
走りません。** サンプラーが選べるのはスキャンしたフレームだけなので、その間の14枚
に対する推論は計算してキャッシュして二度と読まれない純粋な無駄でした。実測
121 fpsで、20本×1時間は**全フレームなら4.1時間、stride 15なら17分**です。
事前ラベリングのみ（フレーム選択オフ）は「動画を通して見直したい」という別の意図
なので、従来どおり全フレームに走ります。

疎な実行では併せて2つを切り替えます。検出を繋いでtrackletにする処理は切ります
（15フレーム離れた検出を「1フレーム隣の同一個体」として繋いでしまうため）。
品質チェックに渡すフレームは**実際に推論したフレームだけ**にします（動画全体を
渡すと、推論していない14/15のフレームが全部「検出ゼロ」の指摘になります）。モデルが未設定のあいだはロックされ、理由に「Trainで作れる」と出ます。
（キャプションが窓の隠すチェックボックスの名前を名乗らないのは、この
ロックがあるからです。）

出てくる候補はすべてpendingです。Acceptするまで `annotations.json` には入りません。

### 候補生成は detector だけを、まとめて走らせます

`predict` に `detect_only=true` を渡します（[`inference/detect_only.py`](../src/htrtdetr/inference/detect_only.py)）。
アノテーションループが欲しいのは**箱だけ**で、ラウンドが学習するのは stage 1 だけ
なので、streaming推論が毎フレーム回している tracker / temporal / behavior は
**乱数初期化のまま**動いていて、その出力は捨てられていました。

それらを外すと、detectorを1フレームずつ回す理由（trackerが逐次的であること）も
無くなるので、**複数フレームをまとめて**投げられます。同じ重み・同じGPUで実測:

| | model fps |
|---|---|
| streaming（従来、検出0の最良ケース） | 36.3 |
| detect_only batch 16, dec=4 | 250.4 |
| detect_only batch 16, dec=2 | 308.0 |

検出が100個/フレーム出る条件では 6.8 → 160.8 fps（24倍）でした。**箱は変わりません** ——
同じ重みでstreamingと比べて、120フレーム×100箱で最大差 0.35px / score 5e-5 です
（fp16のリダクション順がバッチサイズで変わるため完全一致ではありません）。
注釈動画 `output.mp4` も書きません（20本×1時間をstride 15で走らせると、誰も開かない
ファイルのために144,000フレームを描画・エンコードしていました）。

**引き換えはIDです。** 候補は `track_id = -1` で返ります。疎な走査（stride > 1）では
そもそも15フレーム離れた検出を繋ぐべきではなく、従来はモデル内蔵のtrackerがそれを
やってしまっていました。密な走査（stride 1）では `connect_detections` がIoUで繋ぎます。

## 本学習はWorkflowウィンドウ

Project > **Open the workflow window (full training)...** で開きます。stage 1〜4、
モデルモード、window size、imbalance、評価、フレーム単位のレビューといった
「最後に1回やる」ための全部がそこにあります。ラウンドはstage 1だけを短く回すもので、
別物です。書き出し先も `runs/train/stage{N}/` と `runs/rounds/` で分かれています。

## 候補のAccept / Reject

**AIが出した箱は、Acceptするまで `annotations.json` に入りません。** 画面上では
**破線のアンバー**で描かれ、実線の箱（＝確定済みアノテーション）とは別物です。
`Analyze`、`Click animal`、`Continue`、`Find all at once`、`Load Predictions` の
出力はすべてこの pending 候補として入ってきます。

Acceptの入口は5つあり、**どれも同じ1本の昇格処理**を通ります。1件ずつから
動画1本まで、粒度が違うだけです。

| やり方 | 粒度 | どこで |
|---|---|---|
| 候補を**クリック** | その1個 | Canvas。破線の箱を直接クリックすると、それだけがAcceptされます |
| `Y` / `N` | このフレーム全部 | Canvas。`Y`で全Accept、`N`で全破棄 |
| Inspectorの **Accept** / **Reject** | 選択中の1個 | 候補を選ぶとInspectorが候補用の表示になります |
| Inspectorの **Accept more than one** | This frame / This track / This clip / All visible / Similar ones | 同上。*This track* は tracklet 全体（数百フレーム分）を1クリック |
| **Review mode**（Review > Review issues one by one） | 指摘1件ずつ | 前後をループ再生しながら `Y` Accept / `N` Reject / `E` 編集 / `S` skip / `Esc` 終了 |

候補のあるフレームの渡り歩きは `Ctrl+Right` / `Ctrl+Left` です。**Review Inbox**
は品質チェックが引っかけた場所の一覧で、種別を選んで一括Acceptもできます。

Acceptすると箱は実線になり、`provenance.human_review = "accepted"` と
source / score / モデル名を持ったまま保存対象になります。Rejectした候補は
アノテーションには入りませんが、**判断そのものはProjectに残ります**。同じ動画に
もう一度Analyzeをかけても、既にAccept / Rejectしたものは内容キーで引き継がれる
ので、レビューがゼロに戻ることはありません。

複数フレームにまたがるAcceptも**Undoは1手**です。tracklet を1クリックで300枚
Acceptしたあと、`Ctrl+Z` 一回で全部戻ります。

## レビューと品質チェック

**Review Inbox (all videos)** は全動画を集約し、**Go to the next video needing review** は次の問題・候補のある動画とフレームへ移動します。欠落動画はスキップして残し、Locate fileで復旧できます。

**Check Project Quality** またはProject VideosのQuality Checkは、位置・サイズ・方向の急変、重なり、個体数の不一致、トラックの途切れ、ID交換の疑い、低確信度、行動区間の問題を既存検出器で調べます。モデルの疑いは人間が判断します。

Acceptした候補だけがアノテーションになります。クラス、トラック、行動、OBB、source・score・モデル等のprovenanceを保持します。Rejectした候補とその判断はProject側に保持します。

## Activity — 進捗・ログ・エラー

上部1段目の **Activity** チップ、`Ctrl+L`、または Help > **Activity and logs** で開きます。
**普段は出ません。** チップだけが常に出ていて、そこに1行で要約が出ます:

| チップの表示 | 意味 |
|---|---|
| `Activity` | 何も動いていない |
| `Activity: Round 3: Training the detector - pass 14 of 96` | 実行中。複数あるときは `+2` のように件数 |
| `Activity: 2 errors` | 赤。このセッションで2件失敗している |

窓の中は3段です。

1. **Now** — いま動いているもの。ラウンドの現在フェーズとpass数、Analyzeの実行、
   候補の再チェック待ちまで、ダイアログを閉じていても読めます。
2. **セッションのログ** — ステータス行が言ったことを全部、時刻付きで残します。
   ステータス行は1文しか持てず次の1文で消えるので、20分のラウンドが失敗した
   理由がマウスを動かすまでしか残らない、という状態を直すためのものです。
   `All` / `Problems` / `Jobs` で絞れます。**Copy** でそのまま貼り付けられます。
3. **Run log** — 子プロセスが書いた**本物のログ**です。学習は
   `runs/rounds/round-NNN/stage1/round.log`、推論は出力先の `inference.log` に
   パイプではなくファイルへ直接書かれているので、落ちたときのtracebackがそのまま
   残っています。新しい順に並び、**Open folder** でその実行フォルダを開けます。

## 保存・復元・エクスポート

```text
project/
  project.yoake.json        # 動画順、状態、位置、設定、候補、レビュー等
  annotations.json          # 全動画の正式なDatasetAnno、学習・他ツール用
  sampling_log.json         # サンプリング履歴
  README.txt                # この構造の説明（フォルダ自身に書き出される）
  videos/                   # プロジェクトに取り込んだ動画（任意）
  datasets/                 # 学習データ: 書き出しフレームと train/val/test 分割
  runs/                     # 学習・評価の実行結果（weights, results.csv, metrics）
  analysis/                 # 解析結果: predictions.json、重畳動画、図
  .yoake/snapshots/         # 保存世代。Project再開に使用する確定データ
  .yoake_cache/             # 再生成できる推論・特徴キャッシュ
```

フォルダは **Create Project**（または `yoake project new`）の時点で作られます。学習も解析もこの中に書き出されるため、研究の成果物が4か所に散らばりません。

**Projectはフォルダごと移動・コピーできます。** 内部のパスはすべてフォルダ基準の相対パスです。フォルダ外の動画は相対パスと絶対パスの両方で記録するので、「プロジェクトだけ動かした」「一式をまとめて動かした」のどちらでも解決できます。両方が別々に動いた場合は Project Videos > **Missing: Locate...**（または `yoake project relink path=... from=<folder>`）で1回答えれば、そのフォルダは記憶され以後の動画も自動で解決します。ファイル名での再発見が起きたときはステータス行で報告します——正しい別ファイルを黙って開くことだけは避けるためです。動画が見つからない場合もアノテーションは保持します。

**Project > Copy Videos into Project Folder** は動画を `videos/` に取り込み、フォルダ単体で完結させます（`yoake project collect path=...` と同じ）。詳細は [PROJECT_FOLDER.md](PROJECT_FOLDER.md)。

保存は一時ファイルへの書込み・flush・置換を用い、Projectは確定済みのデータ世代を参照します。Autosaveは約4秒の待機後にバックグラウンドで書き込みます。新しい編集が途中で入った場合は次の保存へ回します。保存先を指定していないProjectを閉じる・別Projectに切り替える場合はOSの一時フォルダ `YOAKE/recovery` に回復用データを残します。

**Save and Export Frames** は全動画のアノテーション対象フレームを `images/<video_id>/000000.jpg` 形式で、既定では `<project>/datasets/frames/` に出力します（`annotations.json` と `project.yoake.json` も同梱するので、書き出し先だけを単独で開けます）。書き出し先がどこであっても、作業中のProjectの居場所は変わりません。取り込んだデータに別の画像パスが指定されていても、保存JSONはこの出力先に揃えます。処理はバックグラウンドで行い、欠落動画や読めないフレームを報告します。

Undo/Redo履歴はアプリを開いている間、動画ごとに保持します。再起動時は保存されたアノテーション・候補・レビュー判断・補間用キーフレームから再開し、Undo/Redo履歴自体は復元しません。

旧 `annotations.json` はそのまま開けます。動画情報だけで対応が確定できない場合はLocate fileを使ってください。同じframe番号を持つ複数動画も独立した状態として取り込みます。single-classの省略されたclass_idは0として扱います。annotated / derived OBBを区別し、GUI編集用の情報もProject保存で維持します。

## ショートカットとHelp

既存ショートカットは維持しています。**Help > Quick Start and Shortcuts** は実際のキー登録表から一覧を生成します。

- `W`: Draw Box、`C`: Click animal、`Esc`: キャンセル／選択解除
- `Right / D`、`Left / A`: フレーム移動
- `Space`: 前フレームからコピー、`T`: 選択した個体を次フレームへ追跡
- `Ctrl+R`: Pick frames to label、`Ctrl+Right` / `Ctrl+Left`: 次 / 前の**見るべきフレーム**
- `Ctrl+S`: 全動画を保存、`Ctrl+Z` / `Ctrl+Y`: 現在動画のUndo / Redo
- `Y` / `N`: このフレームの候補を全Accept / 全破棄、`Ctrl+L`: Activity（進捗とログ）
- `Delete`: 選択box削除、数字キー: Behavior設定

回転、ズーム、レビューなどの詳細キーはbuilt-in Helpを参照してください。
