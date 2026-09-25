# Click to Box — クリック 1 回で OBB を作る

このドキュメントは、アノテータに **「個体をクリックすると適切な回転矩形 (OBB) が出る」**
ツールを追加するための設計です。規範は [GUI_DESIGN.md](GUI_DESIGN.md) で、本書はその
§0.3.2 (五つのツール) に 6 つ目を足し、§7 (candidate lifecycle) に 1 行足す差分にあたります。
矛盾したら GUI_DESIGN.md が勝ちます。

---

> **実装状況 (Phase 0-5 実装完了、ただし §6.7 の但し書きを読むこと)**
>
> | 部品 | ファイル | 状態 |
> |---|---|---|
> | エンジン (`local` / `model` / mask→OBB) | [click_segment.py](../src/htrtdetr/gui/click_segment.py) | 完了・実測で検証 |
> | プロトコル + クライアント | [sidecar.py](../src/htrtdetr/gui/sidecar.py) | 完了・別インタプリタ間で実走確認 |
> | ヘルパー本体 + stub | [tools/sam_sidecar.py](../tools/sam_sidecar.py) | 完了。SAM 2 (point/box/video/embedding) と OWLv2 (`concept`) を実測で検証 (§6.7) |
> | アノテータ結線 (ツール・バックエンド選択・候補レビュー) | [annotator.py](../src/htrtdetr/annotator.py) | 完了 |
> | テスト | [test_gui_click_segment.py](../tests/test_gui_click_segment.py) 62 + [test_gui_sidecar.py](../tests/test_gui_sidecar.py) 35 | 97 tests、全 green |
>
> 本文で実装と設計初稿が食い違っていた箇所は、**実装した側**に書き換えてあります。

## TL;DR

* ツールは **1 つ**、バックエンドは **3 つ**（差し替え可能）。
  * `local` — OpenCV だけ。GUI プロセス内、数 ms、依存ゼロ。**既定**。
  * `sam` — SAM 系モデルを**別プロセス・別インタプリタ**で動かすサイドカー。
  * `model` — 既存の Auto-Annotate 候補から最寄りを拾う（学習が回った後はこれが最良）。
* **ツール自体は絶対にロックされない。** `local` が常に使えるので、SAM が無い環境でも
  クリック→箱は成立する。選べないバックエンドを選ぶと必ず `local` に戻り、理由が出る。
* mask → OBB の変換は**リポジトリ内に 1 つだけ**置き、GUI もサイドカーも同じ関数を呼ぶ。
* サイドカーは**動画を自分で開かない**。GUI がデコード済みフレームを渡す (P15)。
* サイドカーは **GUI とは別のインタプリタ**で起動する。これが `sys.executable` 前提の
  既存設計と決定的に違う点で、この機械では前提そのもの（GUI は 3.14 / torch は 3.9）。
* **人が対象を選んだ箱は即確定、機械が対象を選んだ箱はレビュー必須** (§8.4)。

---

## §1 なぜこの形か — 測定された前提

GUI_DESIGN.md §0 と同じ流儀で、**この機械で測った事実**だけを前提に置きます。

| 事実 | 値 | 設計への影響 |
|---|---|---|
| GUI の python | 3.14、torch **無し** | SAM を GUI プロセスに入れる案は不可能 |
| torch のある環境 | `miniconda3/envs/` の `yoru` / `yoake` / `SUPPORT` の 3 つ (Python 3.9、torch 2.8.0+cu128) | サイドカーは**別インタプリタ**で起動する必要がある |
| **GPU** | **NVIDIA GeForce RTX 5070 Ti** (2026-09-18 再実測) | `torch.cuda.is_available()` が True。helper は `device=auto` を自分で解決して `cuda` を返す |
| CPU | 16 スレッド | CPU 経路は残してあるが、もはや唯一の選択肢ではない |
| **SAM ViT-B 相当の encode** | **1024x1024 で約 1.6 s/frame** (86.4M param、windowed attention 9 + global 3 の実測) | 先読み encode が必須。クリックそのものは数十 ms |
| インタプリタ探索 | 7 個を 1.8 s で判定 | 遅延起動なら体感に出ない |
| cv2 / numpy / scipy / skimage | GUI 側に有り | `local` バックエンドは追加依存ゼロで書ける |
| `segment_anything` / `sam2` / `transformers` / `onnxruntime` | **どの環境にも無し**（7 インタプリタを実測） | SAM は「今すぐ動くもの」ではない。§6.7 |

> **2026-09-18 の訂正**: この表はもともと「Intel Arc の iGPU だけで NVIDIA GPU は無い、
> CUDA は今後も使えない」と書いていました。**現在は誤りです。** RTX 5070 Ti があり、
> `yoake` env の torch 2.10.0+cu128 で `torch.cuda.is_available()` は True で、
> サイドカーは実際に `device=cuda` で起動します。設計そのものは変わりません —— CPU で
> 成立するように組んであるので GPU は速さの話であり、成否の話ではありません。
> ただし「CPU で回すしかない」を前提に判断しないでください。

なお `yoake` env は **dearpygui + opencv + torch を同時に**持ちます。「GUI の python に
torch は入らない」という §1 の前提はこの env には当てはまりません。torch-free の
純粋性契約 (`src/htrtdetr/annotation/`) は引き続き正しい設計ですが、それは
「両側からテストできる」ためであって、このマシンの制約ではありません。

**この表から出る最重要の結論**: サイドカーに渡す python は `sys.executable` では**ない**。
既存の `auto_label.predict_argv` が `[sys.executable, *argv]` を前提にしているのと、ここが
決定的に違います。インタプリタのパスは設定項目になります。

---

## §2 モジュール配置

```
src/htrtdetr/gui/click_segment.py   純粋エンジン。dpg 無し / torch 無し / spawn 無し / cv2 は遅延 import
src/htrtdetr/gui/sidecar.py         プロトコル + 子プロセス管理 + クライアント。dpg 無し / torch 無し
tools/sam_sidecar.py                子プロセス本体。torch を import してよい唯一の場所
tests/test_gui_click_segment.py     エンジンとアノテータ結線 (62 tests)
tests/test_gui_sidecar.py           プロトコルと実プロセス往復 (35 tests)
src/htrtdetr/annotator.py           差分。ツール・バックエンド選択・非同期適用・候補レビュー
src/htrtdetr/gui/shortcuts.py       差分。tool.click_box を C に割り当てる
```

**3 層に割れている理由**は不変条件が違うからです。

| モジュール | spawn | threads | torch | dpg |
|---|:-:|:-:|:-:|:-:|
| `click_segment.py` | ✗ | ✗ | ✗ | ✗ |
| `sidecar.py` | ○ | ○ | ✗ | ✗ |
| `tools/sam_sidecar.py` | — | — | ○ | ✗ |
| `annotator.py` | — | — | ✗ | ○ |

`click_segment.py` に spawn を持ち込まないのは、「クリック → 箱」の計算が
**プロセスも GUI も要らない純関数のまま**であることを保証するためです。
この一線があるおかげで、mask→OBB の規則は GUI・サイドカー・テストの
どこからでも同じものを呼べます。

`auto_label.py` / `runner.py` / `predictions.py` などは**触りません** (GUI_DESIGN §9.6)。

`click_segment.py` が守る不変条件は `auto_label.py` と同一です。

### 単発にした理由 (2026-09-17 変更)

当初 `C` は sticky にしていた。「連続して個体を拾う道具」だからというのが理由だが、
実際の操作列はそうならなかった。フィットした箱が一発で正しいことはまれで、次に手が
するのは**ハンドルを掴むこと**である。ツールが armed のままだと、その掴みが 2 回目の
クリックとして解釈され、1 個目の上にもう 1 個箱が落ちる。ドラッグを省くための道具が
削除操作を増やしていた。

いまは `_apply_click_result` — 同期 (local / model) と非同期 (sam) の合流点 — で
Select へ戻し、作った箱を選択する。バックエンドによらず同じ挙動になる。
連続配置が要るときの逃げ道は `W` / `E` と共通の
`Keep drawing after each box` (`_sticky_draw`) で、新しい設定は増やしていない。

* `dearpygui` を import しない（モジュールスコープでも関数内でも）
* **`torch` を絶対に import しない。** 能力判定は `importlib.util.find_spec` と子プロセス探査のみ
* `cv2` / `numpy` は関数内で遅延 import
* `subprocess` を**呼ばない**。argv と env は**データとして返す**だけ、spawn は呼び側

---

## §3 唯一の公開インタフェース

```python
@dataclass(frozen=True)
class ClickRequest:
    frame_index: int
    x: float                     # 画像座標 (px)。annotator._c2i の出力をそのまま
    y: float
    polarity: bool = True        # True=前景として取る / False=ここは背景 (Ctrl+click)
    radius_hint: Optional[float] = None               # 探索窓の初期半径
    roi: Optional[Tuple[int, int, int, int]] = None   # Shift+drag で明示 ROI

@dataclass(frozen=True)
class ClickResult:
    obb: Optional[Tuple[float, float, float, float, float]]  # cx, cy, w, h, theta[rad]
    bbox: Optional[Tuple[int, int, int, int]]                # obb の外接矩形
    score: Optional[float]
    backend: str                                    # "local" | "sam:<model>" | "model"
    detail: str                                     # status 行へ出す 1 文
    diagnosis: str = ""                             # 成功なら ""、失敗なら DIAGNOSES のキー

    @property
    def ok(self) -> bool: ...                       # obb is not None
```

バックエンドは `segment(frame_bgr, req) -> ClickResult` を満たすだけ。
annotator は**どのバックエンドか知らずに**結果を `_create_object` へ流します。

**失敗は例外でも `None` でもなく、`ok=False` の結果**です。マウスハンドラから直接
呼ばれるので例外は論外であり、`None` だと P5 が要求する「原因の区別」が消えるからです。

`radius_hint` は設計初稿では「現在の zoom」でしたが、**取りやめました**。zoom は
「個体が画像上で何 px か」と相関しません。実装は代わりに、窓に収まらなければ
2 倍に広げる (最大 3 回) 自動探索を行い、成功したサイズを次のクリックの初期値として
annotator が渡します。ユーザが調整する設定項目はありません。

---

## §4 mask → OBB — ここが品質の全部

ハエのような細長い被写体では、**mask の精度より「mask をどう矩形に落とすか」の方が効きます。**
`cv2.minAreaRect` を生 mask にかけると脚と翅を含めて箱が膨らみ、しかも体軸の向きが狂います。

```python
def mask_to_obb(mask, at) -> Tuple[Optional[Tuple[float, ...]], str]:
    """2値 mask から体軸に沿った OBB を作る。click_segment.py が唯一の実装。

    成功なら (obb, "")、失敗なら (None, diagnosis) を返す。
    """
```

手順は 4 段:

1. **クリック点を含む連結成分だけ残す** — `connectedComponentsWithStats`。
   クリック点がどの成分にも入らなければ、最も近い成分（窓の 1/4 以内）。
   それも無ければ失敗。閾値処理は輪郭でぴったり合わないので、縁を 1-2 px 外した
   クリックは救います。
2. **軸用に付属肢を落とす** — 距離変換 `cv2.distanceTransform` の最大値の 40% 未満を
   捨てて「胴体コア」を得る。脚と翅は細いので消え、胴体は残ります。
   閾値を**その個体自身の太さの比**で決めるので、px 数のチューニングが要りません。
3. **主軸はコアから PCA、寸法は opening から** —
   `minAreaRect` ではなく PCA なのは、コアが多少欠けても軸が動かないから。
   寸法は、半径 `0.15 * dmax` の楕円カーネルによる **morphological opening**
   （erode → dilate）をかけた「胴体」から測ります。opening は
   **直径より細いものを消し、生き残ったものの寸法は保つ**ので、脚を落としつつ
   胴体の実寸が得られる。生の成分だと脚で膨らみ、コアだと痩せる — その中間が要るのは
   このためです。
   `w` = 主軸方向の伸び（体長）、`h` = 直交方向の伸び（体幅）。
4. **正準化** — `theta` を `[-pi/2, pi/2)` へ畳む。
   矩形は π 周期なので、π 畳み込みは `w` / `h` の割り当てを壊しません。

> **設計初稿からの変更**: 初稿は「寸法は元の成分から取る」としていましたが、これだと
> 脚がそのまま箱に入ります。距離変換の閾値でそのまま測って `2t` を足し戻す案も試し、
> 合成楕円 (真値 61 x 17 px) で 61.7 x 18.6 px と幅が 9% 過大でした。opening では
> **60.9 x 17.8 px**（脚を 6 本生やしても 60.9 x 18.5 px、角度誤差 0.25°）。
> 数値は `tests/test_gui_click_segment.py` が固定しています。

**角度規約**: `geometry/obb.py::canonicalize_obb` は「長辺を `w` に固定することは**しない**」と
明記しています（境界でサイズ回帰が飛ぶため。向きの一意性は 2 倍角符号化が担保）。
本ツールはそれに従いつつ、**ラベリング規約として `w` = 体長 / `h` = 体幅 を常に満たします。**
PCA の主軸が常に第 1 固有ベクトルなので、これは自動的に決定的です。頭尾の区別は
2 倍角符号化の下では表現不能なので、**解決しようとしません**（mod π で十分）。

`geometry/obb.py` は torch 実装なので、`click_segment.py` は同じ規則を numpy で書き直します。
二重実装になるため、**両者の一致を assert するテスト**を置きます
（`pytest.importorskip("torch")` 付き。torch 無し環境では skip）。

---

## §5 バックエンド A — `local` (OpenCV)

追加依存ゼロ、GUI プロセス内、**既定でありフォールバック**。

1. クリック中心に一辺 `2R` の ROI。`R` は既定 48 px、直近の成功サイズで補正。
   成分が窓の縁に達していたら `R` を 2 倍にして最大 3 回まで再試行（上限 256 px）。
   **縁に接する場合だけ広げる**のは、他の失敗はどの大きさでも同じ結果になるからです。
2. **極性の自動判定** — クリック近傍 7×7 の中央値と ROI 全体の中央値を比べ、
   暗い個体／明るい個体のどちらでも同じ操作で取れるようにする。設定項目なし。
3. ROI をぼかして Otsu 二値化 → 3×3 open → §4 の `mask_to_obb`。
4. 失敗したときだけ GrabCut（ROI 矩形初期化、3 iter）で再試行。
   最初から GrabCut を使わないのは、ほとんどの場合 Otsu で足りて桁違いに速いから。

**失敗を名前で区別する** (P5)。それぞれ status 行に出す 1 文を固定します。

| 診断 | 判定条件 | 文言 |
|---|---|---|
| `nothing_found` | クリック点が背景側、成分面積 < `MIN_BODY_PX²` | `Nothing found at that point. Click on the animal itself, or draw the box by hand with W.` |
| `touches_edge` | 成分が探索窓の縁に接する（窓を広げても解けなかった） | `The shape kept running past the search area. Click nearer its centre, or draw the box by hand with W.` |
| `merged_background` | 面積 > 窓の 40% | `The animal merged with the background here. Zoom in and click again, or draw the box by hand with W.` |
| `two_animals` | コアが 2 つ以上、面積比 > 0.6 | `Two animals look joined at that point. Click nearer this one's centre, or draw the box by hand with W.` |

**失敗したら箱を作らない。** 適当な矩形を置くくらいなら何も置かない方が良い、という判断です。
それらしく見える誤った箱は、誰にも気づかれずに学習データへ入ります。

**既知の限界（実測）**: `two_animals` が効くのは**コアが分離したとき**だけです。
頭と頭が接した 2 個体は検出できますが、**横腹どうしがくっついた 2 個体**は
くびれの無い 1 つの太い塊になり、`local` は 2 個体を囲む 1 つの箱を作ります。
T 字・V 字の接触も同様です。これは閾値ベースの手法の原理的な限界で、
§6 の SAM バックエンドを用意する理由のひとつです。

---

## §6 バックエンド B — `sam` サイドカー

### §6.1 なぜサイドカーか

GUI の python に torch を入れられないから、というのが第一の理由です。副次的に:

* 重いモデルが GUI のメモリと起動時間に触らない
* SAM がクラッシュしても GUI は死なない（再起動して続行）
* CUDA のある別マシンへ差し替える余地が残る（将来 TCP 化しても protocol は同じ）

これは既存の「子プロセスで `predict` を回す」設計の延長で、新方式ではありません。

### §6.2 プロトコル — NDJSON over stdin/stdout

1 行 1 メッセージの JSON。応答は**順不同**で、必ず `id` を持ちます。

**GUI → サイドカー**

| op | フィールド | 意味 |
|---|---|---|
| `hello` | `protocol: 1` | 起動確認。能力を返させる |
| `embed` | `id`, `frame`, `path`, `shape` | このフレームを encode してキャッシュしろ |
| `point` | `id`, `frame`, `points`, `path?`, `box?` | プロンプトから 1 個体を取れ |
| `concept` | `id`, `frame`, `text`, `max`, `path?` | テキストから全個体を取れ |
| `drop` | `frame` | このフレームの embedding を捨てろ |
| `shutdown` | — | 終了しろ |

`points` は `[[x, y, label], ...]`、`label` は 1 が前景、0 が背景。
`point` / `concept` の `path` は「まだ encode していなければこれを読め」という保険で、
先読みが間に合わなかったときに往復を 1 回節約します。

`hello` は `protocol` を突き合わせ、**食い違ったら拒否します**。バージョンを黙って
読み替えるより、2 つのチェックアウトが混ざっていると言う方が安全だからです。

**サイドカー → GUI**

```json
{"id":7,"ok":true,"obb":[512.0,300.5,38.2,12.4,-0.41],"bbox":[493,294,531,307],"score":0.93,"ms":42}
{"id":8,"ok":false,"error":"no_mask","detail":"SAM returned an empty mask at that point."}
{"event":"log","level":"warn","text":"..."}
```

**mask は原則として送りません。** サイドカーが `htrtdetr.gui.click_segment.mask_to_obb` を
import して自分で OBB に落とし、5 個の float だけ返します。§4 の実装が 1 つで済み、
パイプも軽くなります（リポジトリは `auto_label.child_env` の `PYTHONPATH` で子から見えます）。
プレビュー用に mask が要る場合のみ、`{"rle": [...]}` を付ける拡張を認めます。

`hello` の応答が `capabilities` を返し、そこに `concept` が無ければ **concept 機能は
`locked()` されます** (P20: 無い契約を UI が勝手に仮定しない)。

### §6.3 フレームの渡し方

GUI は**既にデコード済みの BGR 配列を持っています**。これを
`<temp>/yoake-sam-<pid>/f<idx>.npy` に `np.save` して**パスだけ**送ります。

* base64 も JPEG 再エンコードも通さない（1024x1024 BGR で約 3 MB、書き込み約 2 ms）
* **子プロセスに動画を開かせない。** これは P15 の直接の帰結です。
  `cap.set(POS_FRAMES, n)` は非整数 fps で 1 フレームずれることがあり、しかも
  frame 0 では一致するので気づけない。子が自分で seek すると、**全部の箱が 1 フレームずれた
  画像に対して引かれる**という、発見が最も遅れる種類のバグになります。
* 温存するのは直近 8 フレーム。`shutdown` で temp ディレクトリごと削除。

### §6.4 レイテンシ設計

CPU では **encode が支配的で、click ごとの decode は無視できる**、という非対称性が全てです。

```
frame 移動 → (即座に) embed を投機実行 ──┐
                                        ├─ ユーザがクリックする頃には終わっている
ユーザが観察して個体を狙う (0.5〜3 s) ───┘
click → point (数十 ms) → 箱が出る
```

* `goto_frame` から **200 ms の debounce 後**に `embed` を投げる（スクラブ中に投げ続けない）。
* embedding は LRU で 8 フレーム保持。
* encode が終わる前にクリックが来たら、その `point` は**キューに積んで待たせる**。
  待っている間だけ status chip に `Segmenting… 1.4 s` を出す。
* **バーは出しません** — 進捗は取れないので、経過時間だけ出す (P20)。

**実測値 (Phase 0)**: SAM ViT-B の image encoder と同じ形
(1024x1024 → 64x64 token、dim 768、12 block、14x14 windowed attention 9 + global 3、
86.4M param) を torch 2.8 / CPU 16 スレッドで走らせて **1.6 s/frame**。
SAM ViT-B は 89M param なので、実機の値もこの近傍になります。

したがって:

* ユーザがフレームを見てから個体を狙うまで、ふつう 1-3 秒。**先読みが間に合う**。
* シークで飛ばしている間に投げない (200 ms debounce) ことが効きます。encode 1 回は
  1.6 秒ぶんの CPU なので、飛ばしたフレームぶん投げると全部詰まります。
* ViT-H (636M) は 7 倍以上、つまり **CPU では実用外**。使うなら ViT-B か、
  もっと軽い MobileSAM / SAM2-tiny です。

### §6.5 プロセス管理

* **遅延起動** — バックエンドを `sam` にした瞬間、または最初のクリックで起動。
* `hello` のタイムアウトは 30 s（torch の import とモデルロードがある）。
  その間 chip は `Starting the segmentation helper… 6 s`。
* **stderr は専用スレッドで吸い続ける。** 溜まったパイプで子がブロックするのを防ぎます。
  内容は runner のログ形式でログ欄へ。
* **異常終了したら 1 回だけ自動再起動。** 2 回目は `local` へ自動フォールバックし、
  `sam` 行を stderr の末尾 3 行を添えて `locked()`。無言で壊れたままにしない。
* GUI 終了時・バックエンド切替時に `shutdown` → 3 s 待って `terminate()`。
  **子は stdin が閉じたら自分で終了する**こと（Windows で取り残されないため）。
* Windows: `CREATE_NO_WINDOW`、`text=True, encoding="utf-8", errors="replace"`。

### §6.6 インタプリタの探索

`auto_label.probe()` と同じ形の `probe_sidecar() -> SidecarCapability` を置きます。
探索順:

1. 環境変数 `YOAKE_SAM_PYTHON`
2. `~/.yoake/click_segment.json` に保存された設定
3. `sys.executable`（torch があれば）
4. `~/miniconda3/envs/*/python.exe` と `~/anaconda3/envs/*/python.exe`

**判定は必ず子プロセスで行います** — `python -c "import importlib.util, json; ..."` を
3 s タイムアウトで回し、`torch` / `sam2` / `segment_anything` / `transformers` の
`find_spec` 結果を JSON で受け取る。自プロセスで import しないのは、`auto_label` と同じ理由
（秒単位のコストと CUDA DLL のロード）です。

見つからないときの理由は §3.2 の三部構成で、しかも**原因によって文面が変わります**。
「torch が無い」と「torch はあるが SAM パッケージが無い」を同じ文言で片付けると、
ユーザを間違った方向に走らせるからです。この機械では後者が該当します。

```
The segmentation helper needs a SAM package next to PyTorch, and none was found (present: none).
Install one with:  pip install sam2 huggingface_hub
Until then Click to Box keeps using the Local (OpenCV) fit, which needs nothing installed
and is already the default - press C and click an animal.
```

原因は 3 通りあり、3 通りとも文面が違います。3 つ目は **`sam2` は入っているのに
`huggingface_hub` が無い**ケースで、これが一番厄介です —— パッケージが見えているので
ユーザは導入成功だと思い込み、ロードのたびに `sam2` の奥で失敗します
(`from_pretrained` → `build_sam._hf_download` が内部 import する)。
`InterpreterInfo.sam2_needs_hub` がこの状態を名指しし、`locked_reason` は
**どのインタプリタに入れるか**までパス付きで出します。

```
sam2 is installed next to PyTorch but cannot reach any weights: huggingface_hub is
missing, and sam2 does not depend on it.
Finish the install with:  "C:\Users\you\miniconda3\envs\YOAKE\python.exe" -m pip install huggingface_hub
Until then Click to Box keeps using the Local (OpenCV) fit, ...
```

文字列はすべて ASCII に保っています。Windows の既定コンソール (cp932) に流れても
`UnicodeEncodeError` にならないためで、これはテストで固定してはいませんが
`locked_reason` と `DIAGNOSES` を触るときの約束です。

### §6.7 実モデルのアダプタ — `sam2` は実走済み、他の 2 つは未検証

`tools/sam_sidecar.py` には 3 つの実アダプタが入っています。

| `--backend` | 必要なもの | API |
|---|---|---|
| `sam2` | `sam2` パッケージ | `SAM2ImagePredictor.from_pretrained(...)` → `set_image` / `predict` |
| `segment_anything` | `segment_anything` + `.pth` | `sam_model_registry[key](checkpoint=...)` → `SamPredictor` |
| `transformers` | `transformers` | `SamModel` + `SamProcessor`、`get_image_embeddings` を 1 回 |

`sam2` は**実走検証済み**です (`facebook/sam2-hiera-tiny`, CUDA)。
`segment_anything` と `transformers` の 2 つは依然として未検証で、
P20 の精神からすれば検証していないものを「動く」と書くわけにはいきません。

導入は 2 パッケージです。`sam2` は `huggingface_hub` に依存していないのに
それ無しではチェックポイントに到達できないので、**片方だけ入れた状態が一番危険**です:

```
pip install sam2 huggingface_hub
```

入れたら最初にこれを実行してください:

```
<torch のある python> tools/sam_sidecar.py --selftest
<torch のある python> tools/sam_sidecar.py --backend sam2 --model facebook/sam2-hiera-tiny
```

`--device` の既定は `cpu` ではなく **`auto`** です。デバイスを決められるのは torch を
持つヘルパだけで、GUI 側は import できないため判定できません。解決結果は `hello` の
`device` で返り、それが UI に出ます。

実測は `docs/ANNOTATION_COPILOT.md` §11 に、回帰テストは
`tests/test_gui_video_segment.py` の `live_model` ブロックにあります。

### `concept` (テキストから全個体) — 実装済み (2026-09-18)

SAM 系アダプタ 3 つは今も `capabilities` に `("point", "box")` しか持ちません。
**それが正しい**からです: SAM / SAM 2 は promptable で、「この点にあるものは何か」
には答えられても「ハエはどこか」には答えられません。`concept` が長らくプロトコル
だけ存在して stub しか実装者がいなかったのは、この一点が埋まっていなかったためです。

埋め方は**置き換えではなく合成**です:

```
  OwlGroundingAdapter (OWLv2 / OWL-ViT, transformers)
      -> テキスト 1 本で軸並行 box を N 個            ← 「どこか」を決める
  Session.ground: 各 box + その中心点で image adapter を prompt
      -> SAM 2 が mask を返す
  handle(): mask -> cs.mask_to_obb
      -> 回転込みの OBB                                ← 「向き」を決める
```

どちらのモデルも単独では成立しません。SAM 2 は自力で何も見つけられず、OWLv2 は
向きの概念を持たない (細長い動物ではそれがラベルの大半です)。

* 能力申告は `Session.extra_capabilities()` が `video` / `embedding` と同じ流儀で
  行い、`importlib.util.find_spec` だけを見てモデルはロードしません。GUI は
  `hello` の `capabilities` を見るだけで、名前から推測しません (P20)。
* image adapter が box prompt を取れない場合 (stub 等) は軸並行 box を
  `theta = 0` の OBB として返します。**向きが分からないから何も返さない**のでは
  なく、人が回せる box を返します。精緻化が例外を投げた場合も同じ経路です。
* `transformers` が無ければ `unsupported` を返し、`detail` に
  `pip install transformers` と「それまで C と Continue は使える」を書きます。
* 閾値はリクエストの `threshold` で上書きできます (既定 0.05)。OWLv2 のスコアは
  確率ではなく低めに出るため、recall を残して人が捨てる側に倒してあります。

**実測** (RTX 5070 Ti, `google/owlv2-base-patch16-ensemble` + `sam2-hiera-tiny`,
512x512 の合成シーン、`tests/test_sidecar_grounding.py` の `live_model`):

| | |
|---|---|
| 中心誤差 | 0.3 px (真値 (300,300) に対し (300.2, 300.2)) |
| サイズ | 93.4 x 33.5 (描画した本体は 92 x 32) |
| 角度 | -0.70 rad = -40.1 deg (描画角度 -40 deg) |
| recall | planted 3 個のうち **1 個** |

recall が低いのは**合成画像だから**です。OWLv2 は写真のモデルで、平坦な背景に
描いたグレーの楕円には自信を持ちません。テストが固定しているのは合成の正しさ
(検出器が位置を決め、SAM 2 が向きを決めている) で、recall ではありません。
実素材での recall は閾値と語句で調整してください。

`stub` も引き続き `concept` を名乗るので、レビュー動線はモデル無しでテストできます。

**発見して直したバグ**: `concept` 分岐だけ `session.load()` を呼んでおらず、
helper にフレームが渡っていませんでした。stub は `session.shape` から箱を合成して
256x256 にフォールバックするので何も気づかず、実モデルでは**常に空リスト**が
返り、UI では「このフレームに動物はいない」と区別できませんでした。
capability 判定を先、フレーム読み込みを後にしてあります —— 逆だと検出器が未
インストールの環境で「フレームが届いていない」と出て、唯一の解決策である
インストール手順が表示されません。

---

## §7 バックエンド C — `model`

Auto-Annotate が既に `predictions.json` から `Candidate` を作れます。`model` バックエンドは
**推論を走らせません**。`File > Load Predictions > As boxes for Click animal...` で読んだ検出のうち、
クリック点に当たるものを 1 つ採用して返すだけです (`click_segment.pick_candidate`)。
学習済みモデルはハエ専用に訓練されているので、**学習が回ったあとはこれが最も正確**です。
SAM はあくまで「学習データが無い立ち上げ期」の道具です。

選び方の規則は 2 段:

1. **クリック点を含む箱が、近いだけの箱に勝つ。** OBB があれば回転を考慮して内外判定します。
2. 同格なら中心が近い方。

どれにも入っておらず、かつ箱 1 つぶんの距離より遠ければ **不採用**にして
`no_candidate` を返します。一番近いものへ引き寄せると、何も無い場所のクリックが
隣の個体の箱を複製してしまい、しかもそれが正しく見えるからです。

---

## §8 アノテータ側の差分

### §8.1 ツール 6 — `Click to Box`

GUI_DESIGN §0.3.2 の表に 1 行追加:

| # | Tool | Tag | Key | Behaviour | Sticky? |
|---|---|---|---|---|---|
| 6 | `Click to Box` | `tool_click_btn` | `C` | クリックで個体を推定し OBB を作る | **単発** — `W` / `E` と同じ。1 箱作ると Select へ戻り、その箱が選択された状態になる |

* `C` は現在の全バインディングで未使用（`shortcuts.BINDINGS` を走査して確認、
  テストでも固定）。左手側なので P14（左手偏重キーマップ）にも合致します。
* `_on_mouse_press` の tool 分岐の先頭に 1 分岐。座標変換 `_c2i` はその直前にあります。
  ドラッグを伴わないので `_mode` は `idle` のまま、release 側には何も残しません。
* 生成は `_create_object(bbox, source="wand")` → `obj.obb = [...]` → `sync_bbox()`。
* タグは仕様書 §0.3.2 の `tb_tool_*` ではなく、現行アノテータの命名
  (`tool_select_btn` / `tool_draw_btn`) に合わせた `tool_click_btn` です。
  現行実装にはまだツールバーが無く、ツールはサイドバーのボタンだからです。

**未実装のまま残したもの**: `Ctrl+click`（負のプロンプト）と `Shift+drag`（ROI 明示）。
`ClickRequest` は `polarity` / `roi` を持ち、`local` は `roi` を解釈し、プロトコルは
`points` の label と `box` を運べます。**繋いでいないのはアノテータ側のジェスチャだけ**です。
負のプロンプトは「1 回目の結果を見て直す」操作なので、実モデルで結果を見られるように
なってから形を決めるべきで、先に UI を作っても当て推量になります。

### §8.2 作られた箱の扱い — P1 からの意図的な逸脱

GUI_DESIGN P1 は「機械の出力は accepted になるまでディスクに触れない」と定めています。
**Click to Box はこれに従いません。箱は最初から accepted で、選択状態で作られます。**

理由: Auto-Annotate では機械が「**何を**ラベルするか」を選んでいるので、人の承認が要ります。
Click to Box では**人が対象を選んでおり**、機械がやったのは矩形のフィットだけです。
そしてその結果は即座に目に見え、ハンドルで直せます。1 クリックごとに承認キーを
押させるのは、主要操作としては壊れた設計です。

取り消しの担保は、**作った箱を選択状態にすること**です。現行アノテータに undo は
まだ無い（仕様書 §0.3.6 は未実装）ので、`Ctrl+Z` を逃げ道として当てにはできません。
クリック直後は必ずその箱が選択されているので、`Delete` が即座に効きます。
この「選択したまま返す」はテストで固定してあります。

代わりに正直さは provenance で担保します。§7.1 の表に 1 行追加:

| State | `accepted` | `source` | Produced by | On disk? |
|---|---|---|---|---|
| wand / accepted | `True` | `"wand"` | Click to Box | yes, `"source":"wand"` (+ SAM 時は `"score"`) |

`AnnObject` に `source: str = "human"` を足し、`to_dict()` は **`"human"` 以外のときだけ**
`source` キーを書きます（`auto_label.provenance_fields()` と同じ規則）。これで人手だけの
プロジェクトの `annotations.json` は 1 バイトも変わらず、`from_dict` は
`d.get("source", "human")` で読み戻します。人が後からハンドルで直しても `source` は変わりません
（§7.1 の既定のとおり、provenance は「どこから来たか」であって「最後に誰が触ったか」ではない）。

**concept モード（SAM3 でテキストから全個体）は逆です。** あれは機械が対象を選ぶので、
P1 のまま既存の `ReviewSession` に pending candidate として流します。分岐の基準は
「対象を選んだのは人か機械か」であって、モデルの種類ではありません。

### §8.3 UI

サイドバー `Tools` の中に、上から順に:

* ツールボタン 3 つ目 `Click to Box (C)`。アクティブなツールは `>` 付きラベルと
  `Active tool:` 行の色で示します（現行アノテータの既存パターン）。
* `Fit with:` — `Local (OpenCV)` / `SAM helper` / `Model predictions` の 3 ボタン。
  コンボではなくボタンなのは、dpg のコンボが**行単位で無効化できない**からで、
  「使えないものも見えていて、理由が読める」(§3 / P4) を満たすにはこの形が要ります。
  **ツールボタン自体は決してロックされません。**
* コストの事前提示 (P17) はツールチップに常駐。SAM の行には
  「ヘルパーの初回ロードは数十秒、フレームごとの初回クリックは encode 待ちで
  この機械なら約 1.6 秒」と、**実測値**で書いてあります。
* `Find all in frame` — テキスト入力 + ボタン + 候補カウンタ +
  `Accept all` / `Discard all`。候補が 0 のときボタンは無効です。
* `Keyboard Shortcuts` ウィンドウに `C` 行、Click to Box 節、`Fit with:` 表、
  `Find all in frame` 節。`Quick Start` の手順 2 にも C を併記。
* status 行が唯一のフィードバック経路（現行アノテータに chip はまだ無い）。

選べないバックエンドを選んだときは**必ず `local` に戻り**、理由を status に出します。
行き止まりを作らないのが、この UI で唯一ゆずれない性質です。

---

### §8.4 `Find all in frame` — ここだけ扱いが逆になる

`concept` は機械が「**何を**囲むか」まで決めます。だから §8.2 の逸脱は**適用しません**。
結果は `auto_label.ReviewSession` に pending として入り、`self.annotations` には
**一切触れません**。保存経路が物理的に存在しないので、未レビューの実行が
学習データを汚すことがありません (P1)。

| | Click to Box (C) | Find all in frame |
|---|---|---|
| 対象を選んだのは | **人** | 機械 |
| 機械がやったのは | 矩形のフィットだけ | 探索 + フィット |
| 箱の初期状態 | accepted・選択済み | **pending・未保存** |
| 承認の行為 | クリックそのもの | 候補をクリック / Accept all |
| 描画 | 実線 | **破線・琥珀色・`? 0.90`** |
| `source` | `wand` | `auto` (+ `score`) |

分岐の基準は**モデルの種類ではなく「対象を選んだのは誰か」**です。SAM で 1 匹を
クリックすれば `wand` で即確定、同じ SAM にテキストで探させれば `auto` で要レビュー。

候補は破線・琥珀色で、承認済みの実線とは一目で違います。**保存されないかもしれない
ものが、アノテーションと同じ見た目をしていてはいけない**という一点に尽きます。

---

## §9 テスト

`tests/test_gui_click_segment.py`（`test_gui_auto_label.py` と同じ流儀）。
**Phase 1 で実装済み: 42 tests。**

| テスト | 固定する性質 |
|---|---|
| `test_module_never_imports_dearpygui_or_torch` | エンジンの純粋性（AST 走査、`subprocess` も禁止） |
| `test_module_scope_imports_are_stdlib_only` | cv2 無しの機械でも import できる |
| `test_min_body_px_agrees_with_the_annotator` | 作った箱をアノテータが弾かない |
| `test_normalize_angle_matches_geometry_obb` | torch 版 `normalize_angle` と 1e-9 以内で一致 |
| `test_mask_to_obb_recovers_a_known_angle` | 6 通りの既知角で ±2° 以内 |
| `test_mask_to_obb_recovers_a_known_size` | 真値 61 x 17 px に対し ±8% |
| `test_mask_to_obb_ignores_appendages` | 脚 6 本を生やしても寸法 ±15% / 角度 ±2°。**併せて `minAreaRect` が実際に 1.4 倍以上に膨らむことも assert**（テストが空振りに変質しないため） |
| `test_mask_to_obb_puts_the_body_axis_in_w` | `w` = 体軸、`theta ∈ [-pi/2, pi/2)` |
| `test_mask_to_obb_reports_*` | 4 つの診断がそれぞれ正しく出る |
| `test_segment_local_fits_a_synthetic_animal[dark/light]` | 極性 2 通りが同じ操作で通る |
| `test_segment_local_widens_the_window_for_a_large_animal` | ヒント無しで大きい個体も取れる |
| `test_segment_local_reports_instead_of_guessing` | 空のアリーナでは**箱を作らない** |
| `test_segment_local_never_raises_on_hostile_input` | 6 通りの異常入力で例外を投げない |
| `test_local_is_always_available` | ツールがロックされない根拠 |
| `test_recoverable_diagnoses_name_the_escape_hatch` | 全診断が次の行動を書いている (§3.2) |
| `test_annotator_click_creates_a_rotated_box` | 結線。OBB・bbox・**選択状態**まで |
| `test_annotator_click_records_provenance` | `source="wand"` が JSON に出る |
| `test_a_hand_drawn_box_writes_no_provenance_key` | 人手だけの出力は 1 バイトも変わらない |
| `test_annotator_click_remembers_the_animal_size` | 次のクリックの初期半径 |
| `test_tool_switching_round_trips` | C / C / Esc |
| `test_the_click_tool_has_a_shortcut_and_it_is_free` | `C` の重複が無い |

`tests/test_gui_sidecar.py` (35 tests)。**この 12 個は本物の子プロセスを起動します** ——
stub が存在する理由がこれで、パイプ・フレーミング・スレッド・終了処理が
モデルもGPUもダウンロードも無しに実走します。

| テスト | 固定する性質 |
|---|---|
| `test_module_never_imports_dearpygui_or_torch` | spawn する層でも GUI と torch に触れない |
| `test_module_scope_imports_are_stdlib_only` | 子が codec のために import しても安い |
| `test_encode_never_emits_a_newline` | 1 メッセージ = 1 行のフレーミング |
| `test_decode_ignores_noise_instead_of_raising` | ライブラリの print でセッションを落とさない |
| `test_argv_uses_the_given_interpreter_not_this_one` | **設計の中心**: `sys.executable` ではない |
| `test_env_puts_src_on_pythonpath` | src-layout の子が `ModuleNotFoundError` で死なない |
| `test_probe_argv_only_uses_find_spec` | 探査で torch を import しない |
| `test_probe_reports_this_interpreter_honestly` | 実インタプリタを実際に探査 |
| `test_locked_reason_*` | 原因・コマンド・逃げ道、かつ原因で文面が変わる |
| `test_frame_store_round_trips_and_evicts` | .npy 受け渡しと LRU |
| `test_waiting_for_one_reply_keeps_the_others` | **実測バグの回帰**（下記） |
| `test_helper_starts_and_announces_what_it_can_do` | 実プロセス起動 + hello |
| `test_a_click_comes_back_as_a_click_result` | 実プロセス往復 → `ClickResult` |
| `test_replies_carry_the_id_that_asked` | 順不同応答が正しい待ち手に届く |
| `test_an_unknown_op_is_an_error_not_a_dead_helper` | 未知 op で落ちない |
| `test_a_failing_request_is_a_reply_not_a_crash` | 1 回の失敗でヘルパーを失わない |
| `test_a_protocol_mismatch_is_refused_loudly` | バージョン食い違いを黙認しない |
| `test_stop_is_idempotent_and_leaves_no_frames` | temp を残さない |
| `test_the_child_self_test_passes` | `--selftest` が緑 |

アノテータ側 (`test_gui_click_segment.py`) の非同期・レビュー分:
`test_a_sam_click_creates_a_box_when_the_reply_arrives` /
`test_a_reply_for_a_frame_the_user_left_is_discarded` (P9 世代ガード) /
`test_a_dead_helper_falls_back_to_local_instead_of_dead_ending` /
`test_find_all_produces_candidates_not_annotations` /
`test_candidates_never_reach_the_saved_file` /
`test_accept_all_and_discard_all` /
`test_a_click_on_the_model_backend_takes_the_detection` ほか。

> **実測バグ 1 件 (Phase 2)**: `wait_for` が「取り出した束の途中で return する」
> 実装になっており、**一致より後ろのメッセージを捨てていました**。返事は届いている
> のに次の待ちが 30 秒タイムアウトする、という最も診断しづらい壊れ方をします
> (テスト全体が 3 秒 → 120 秒超になって発覚)。束を最後まで走査してから返すよう修正し、
> `test_waiting_for_one_reply_keeps_the_others` で固定しました。

**`tools/sam_sidecar.py --backend stub`** を用意します。torch 無しで決定的な箱を返す
偽バックエンドで、**CI（torch の無い 3.14 環境）でもパイプ全体を実走できます。**
プロトコルのテストが「モックのテスト」に堕ちるのを防ぐための措置です。

---

## §10 非同期の規律 (P9)

1. ワーカースレッド (stdout / stderr の 2 本) は **`dpg` に一切触らない**。
   `queue.Queue` に積み、stderr は溜めずに吸い続ける（詰まると子が返事の途中で止まる）。
2. 自己再武装する `dpg.set_frame_callback` (`_pump_sidecar`) がメインスレッドで drain する。
   再武装は `finally` に置いてあり、途中で例外が出てもポンプは止まりません。
3. 依頼ごとに `{kind, frame}` を覚え、返事が来たら **依頼時のフレームと現在のフレームを照合**。
4. 不一致なら**結果を捨てる**。
5. ただしユーザは待たされているので、`Frame changed; the segmentation was discarded.` を
   1 行だけ出します（`concept` なら `...the search was discarded.`）。無言だと
   「クリックしたのに何も起きない」と区別がつきません。
6. 待っている間は `Segmenting... 1.4 s` と**経過時間だけ**。進捗は取れないので出しません (P20)。

ポンプは SAM を使うと宣言したときだけ起動し、`local` / `model` では 1 フレームも回りません。

---

## §11 段階

| Phase | 内容 | 状態 |
|---|---|---|
| **0** | 測定: GPU、encode 実時間、インタプリタ探索、SAM パッケージの有無 | **完了。** §1 に記録 |
| **1** | エンジン + `local` + ツール + テスト | **完了。** torch 無しで動く |
| **2** | プロトコル + `--backend stub` + 実プロセステスト | **完了。** torch 無しで配管が検証される |
| **3** | 実 SAM アダプタ + インタプリタ探索 + 非同期結線 | **コードは完了、モデル実走は未検証** (§6.7) |
| **4** | `concept` → ReviewSession による候補レビュー | **完了。** stub で全経路をテスト。実モデル側の `concept` 対応は未実装 |
| **5** | `model` バックエンド | **完了。** `predictions.json` から選ぶ |

各フェーズの境目で**止めても壊れません**。Phase 1 だけでも torch 無しで完結し、
Phase 2 までで配管が検証済み、Phase 3 以降は「SAM を入れるかどうか」の判断待ちです。

### 残っている作業（優先順）

1. **SAM パッケージを 1 つ入れて `--selftest` と実クリックを通す** (§6.7)。
   `sam2` + `facebook/sam2-hiera-tiny` が CPU では第一候補。
2. 実モデルで結果を見てから `Ctrl+click`（負のプロンプト）と `Shift+drag` を繋ぐ (§8.1)。
3. `concept` を名乗れるアダプタ (SAM3 系) を足す。プロトコルと UI は待っている状態。
4. `local` の「横腹どうしが接した 2 個体」問題 (§5) は SAM が入れば解消する見込み。

---

## §12 やらないこと

| やらないこと | 理由 |
|---|---|
| GUI プロセスに torch / onnxruntime を入れる | `test_gui_auto_label.py` が固定する不変条件を壊す。3.14 にそもそも入らない |
| サイドカーに動画パスを渡して自分で seek させる | P15。1 フレームずれが全アノテーションに伝播し、しかも気づけない |
| mask を GUI 側へ常時転送する | §4 の実装が 2 箇所に増える。OBB 5 float で足りる |
| CLI の stdout から進捗を正規表現で拾う | P20。既に明文で否定されている |
| 頭尾（前後）の判定 | 2 倍角符号化の下では表現できない。mod π で十分 |
| クリック 1 回ごとの承認キー | §8.2。人が対象を選んでいるので、承認はクリックそのもの |
| バックエンド名からの機能推測 | P20。`concept` が使えるかは `hello` の `capabilities` だけが決める |
| 実走させていないアダプタを「動く」と書くこと | §6.7。配管は検証済み / モデル呼び出しは未検証、と分けて書く |
| `wait_for` を GUI のクリック経路で使うこと | 起動時の `hello` とテスト専用。クリックの待ちは必ず非同期 (P9) |
| 候補 (`concept` の結果) を保存経路に載せること | P1。accept を通らない限り `annotations` に触れない |
