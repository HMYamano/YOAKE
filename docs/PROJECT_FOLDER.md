# プロジェクトフォルダ / The project folder

**A YOAKE study is a folder, not a file — and the folder can be moved.**

Until now a project was `annotations.json`, and everything else it produced landed
somewhere else: training runs under the source checkout's `runs/`, figures under
`outputs/`, caches wherever the caller pointed them. Archiving a study meant
remembering four locations; sending one to a colleague meant reconstructing them.

一つの研究 = 一つのフォルダ。アノテーション・学習データ・解析結果がすべてその中に入り、
フォルダごと移動・コピーしても動作します。

---

## 1. The layout

```
<project>/
  project.yoake.json    マニフェスト: 動画一覧・レビュー状態・設定
  annotations.json      データセット (学習フォーマット)
  sampling_log.json     どのフレームをなぜ選んだか
  README.txt            この構造の説明 (フォルダ自身に書き込まれる)
  videos/               プロジェクトに取り込んだ動画 (任意)
  datasets/             学習データ: 書き出したフレームと train/val/test 分割
  runs/                 学習・評価の実行結果 (weights, results.csv, metrics)
  analysis/             解析結果: predictions.json, オーバーレイ動画, 図
  .yoake/               ワークスペースのスナップショット (内部)
  .yoake_cache/         再計算可能なキャッシュ (消しても時間を失うだけ)
```

The shape is fixed rather than configurable. A path the user has to supply is a
path they can supply *differently* next session, and the result is a study whose
halves no longer know about each other. Everything is derived from one root, so
the only thing to remember is where the project is.

Conventional sub-locations (`htrtdetr.annotation.layout.ProjectLayout`):

| Property | Path | Written by |
|---|---|---|
| `annotations` | `annotations.json` | annotator (Ctrl+S, autosave) |
| `splits` | `datasets/splits/` | Train ▸ Create splits |
| `frames` | `datasets/frames/` | annotator ▸ Save and Export Frames |
| `summary` | `datasets/summary/` | Train ▸ Dataset statistics |
| `train_run(n)` | `runs/train/stage{n}/` | Train ▸ Run training |
| `eval_run(n)` | `runs/val/stage{n}/` | Train ▸ Run evaluation |
| `predictions` | `analysis/predict/` | Analyze ▸ Run inference |
| `reports` | `analysis/reports/` | Train ▸ Analyze |
| `cache` | `.yoake_cache/` | batch detection / feature scans |

The GUI pre-fills every path field from this table, so the four pages cannot
disagree about where a study's data is. A field the user types into is never
overwritten by a project switch; a field still holding a suggestion follows it.

## 2. Making one

**GUI** — the annotator's first screen. The folder field is pre-filled with
`~/YOAKE/<project name>` and follows the name until you edit it. *Browse* asks
for the folder to create the project *in*; picking a folder that already holds a
project opens that one instead of overwriting it.

**Headless**

```bash
yoake project new dir="D:/studies" name="courtship 2026" videos=D:/captures
yoake project new dir=. name=pilot videos=./raw recursive=true copy=true
```

`copy=true` copies the videos into `videos/`, which makes the folder
self-contained (see §3). The same thing later: `yoake project collect path=...`,
or **Project ▸ Copy Videos into Project Folder** in the annotator.

## 3. Why it survives a move

Every path written into the project is relative to its root, so moving, copying
or zipping the folder keeps it working — on this machine or another one.

That is only half the problem. A capture usually lives *outside* the project (a
NAS share, an acquisition drive), and a relative path to it breaks the moment
the project moves on its own. So each video is stored twice: relative, and
(only when it is outside the folder) absolute. Opening a project tries them in
a fixed order, and **the order is the design**:

| # | Tried | Right when |
|---|---|---|
| 1 | relative to the project root | the whole tree was copied somewhere else |
| 2 | the recorded absolute path | only the project folder moved |
| 3 | a search by file name, under the project and under folders located before | both ends moved — a study restored from a backup |
| — | otherwise: reported missing, **annotations kept** | the file is genuinely gone |

Relative wins first because it is the only answer that is right when the tree was
copied: a stale absolute path can still resolve on the old machine and quietly
open *the wrong file*. When step 3 answers, the annotator says so on the status
line — silently opening a different file with the right name is the one failure
the search could cause.

| what moved | result |
|---|---|
| nothing | opens |
| the project folder only | opens (absolute fallback) |
| project + captures, arrangement kept | opens (relative) |
| both, independently | missing → `relink` once, remembered afterwards |
| captures copied in (`collect`) | opens anywhere, needs nothing outside |

### Repairing a broken link

One answer fixes all of them: a restored backup breaks every link for the same
reason. The folder you point at is remembered in the project (up to eight), and
later opens search it automatically.

```bash
yoake project info   path="D:/studies/courtship 2026"     # what resolves, what does not
yoake project relink path="D:/studies/courtship 2026" from=E:/archive/captures
```

In the annotator: **Project Videos ▸ Locate file**.

## 4. Opening one

```bash
yoake gui                     # 1. Label ▸ Open project folder...
annotate --project "D:/studies/courtship 2026"
annotate --project "D:/studies/courtship 2026/project.yoake.json"
```

The Hub remembers the active project across sessions and points the Train and
Analyze pages at it, so a run started on the Train page lands in *that study's*
`runs/`, not next to the source code.

A folder is also accepted anywhere a manifest is: `ProjectSession.open()` takes
the folder, the manifest, or a bare `annotations.json` (the pre-folder format,
which still opens unchanged).

## 5. Compatibility

- `annotations.json` is **unchanged** — same schema, same place, still the file
  training reads. The folder is built around it, not instead of it.
- `project.yoake.json` stays at `schema_version: 1`. The new keys (`abs_path`,
  `identity`, `settings.search_roots`) are optional, and a manifest written
  before them opens exactly as it did. Bumping the version would make an older
  YOAKE refuse a project it can in fact read.
- A project with no folder still works and autosaves to a temporary recovery
  folder, as before. It is no longer the default, because that behaviour is
  precisely how output ends up scattered.

## 6. Where this lives in the code

| File | Role |
|---|---|
| `src/htrtdetr/annotation/layout.py` | the layout and the path policy — torch-free, dpg-free |
| `src/htrtdetr/annotation/project.py` | `ProjectSession.create/open/save/import_media/relink_missing` |
| `src/htrtdetr/project_cli.py` | `yoake project new/info/collect/relink` |
| `src/htrtdetr/gui/workspace.py` | annotator: open, autosave, export, collect |
| `src/htrtdetr/gui/hub.py` | the active project the Train and Analyze pages follow |
| `tests/test_project_layout.py` | all four move cases, driven against real files |
