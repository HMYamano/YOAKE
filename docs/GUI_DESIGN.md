# YOAKE GUI Design Contract

**Status: normative.** This is the contract that the four GUI owners (`annotator.py`, `hub.py`,
`training_panel.py`, `analysis_panel.py` + `canvas.py`) build against, and the single written
record of the interaction design mined from the **YOMU** reference project (`ref/YOMU-main/`)
before that reference was deleted. Nothing about YOMU is recoverable any more except what is
written here. Appendices A and B exist for that reason and are deliberately over-detailed.

**Scope.** Everything in Part II (§1–§9) is binding on implementers. Part I is orientation: why
the design is shaped this way, what the screens are, and how the two loops (label, and
auto-annotate) actually run. Part III is the YOMU archive.

**Reading order.**

| You are | Read |
|---|---|
| New to the project | §0 → Part I (§0.1–§0.4) → §6 → §7 |
| Implementing `theme.py` / `widgets.py` | §1, §2, §3 |
| Implementing `shortcuts.py` | §6 and §0.3.8 |
| Implementing `auto_label.py` | §7, §0.4, §9 |
| Implementing a consumer panel | §4, §5, §3, then your panel's part of §0.2 |
| Writing any user-visible string | §4, all of it, no exceptions |
| Wondering "why did they do it that way?" | Part I and Appendix A |

---

## §0 Status and verified environment facts

Every fact in this section was **measured on this machine during the synthesis run**, not
recalled. Where a claim in the circulated implementation spec failed to reproduce, the measured
value is recorded here and the spec's value is marked wrong. Bind these; do not re-derive them.

### §0.0.1 The machine

| Component | Version |
|---|---|
| Python | 3.14.3 |
| dearpygui | 2.3.1 |
| opencv-python | 4.13.0 |
| numpy | 2.4.4 |
| torch | **not installed** |
| ultralytics / onnx / onnxruntime | **not installed** |
| tkinter, pyyaml, matplotlib, scipy, tqdm, PIL | present |

### §0.0.2 Blocking facts

| # | Fact | Consequence |
|---|---|---|
| **V1** | `importlib.util.find_spec("htrtdetr") is None`. The repo is src-layout and the package is **not installed**. `python -m htrtdetr.cli …` dies with `ModuleNotFoundError: No module named 'htrtdetr'`. | **Every** GUI subprocess button in the suite is a silent no-op until `<repo>/src` is on `PYTHONPATH`. `hub._launch_annotator` currently Popen-succeeds and prints *"The annotator opened in a separate window."* — a lie. Fix with `auto_label.ensure_child_env()` before anything else. |
| **V2** | `cv2.TrackerCSRT_create` does **not** exist. `cv2.legacy` does **not** exist. `cv2.TrackerMIL_create` **does**. `cv2.TrackerGOTURN_create` exists but needs downloaded ONNX weights and fails at init. | `annotator.py:1793` calls `cv2.TrackerCSRT_create()` unguarded and raises `AttributeError` the moment auto-track steps a frame. The only torch-free automation in the repo is dead. The probe chain must be CSRT → `cv2.legacy` CSRT → MIL → pure Lucas-Kanade; on this machine it resolves to **MIL** (~47 ms/frame). |
| **V3** | `dpg.add_font_range_hint` / `add_font_chars` / `add_font_range` still **exist as attributes** but are deprecated no-ops ("character ranges are now automatic"). | `hasattr()` is not a guard here. Load the TTF, `bind_font`, stop. Do not write char-range code. |
| **V4** | `mvKey_Backspace` does **not** exist — the constant is `mvKey_Back`. `mvKey_Control` does **not** exist — use `mvKey_ModCtrl` with `is_key_down`. | Binding a non-existent constant raises `AttributeError` **at registration**, so the annotator would not start at all. |
| **V5** | Present and usable: `mvKey_{V,W,E,R,H,X,Q,Y,N,T,F,Z,D,A,S,O,P,0-9,F1,Escape,Delete,Home,End,Left,Right,Plus,Minus,Spacebar,Back}`, `mvKey_Mod{Ctrl,Shift,Alt,Super}`, `add_mouse_wheel_handler`, `add_mouse_move_handler`, `bind_item_theme`, `get_item_theme`, `output_frame_buffer`, `draw_layer`, `is_key_down`, `get_focused_item`, `get_item_type`, `is_item_active`, `add_font`. | The toolbar + zoom/pan + theme-readback + typing-guard plan is buildable exactly as specified. |
| **V6** | A `child_window` has **no** `rect_min` (`get_item_rect_min` raises `KeyError`); a `drawlist` has one. During a held drag `is_item_hovered(<child_window>)` is **False** while `is_item_hovered(<drawlist>)` stays **True**. | These two facts are the entire reason boxes could not be drawn in the annotator. Already repaired there; still latent at `analysis_panel.py:529`. See §5. |
| **V7** | `tools/convert_annotations.py:538` reads `mode=` / `input=` / `output=` (modes `csv \| json \| video_csv`). `tools/build_splits.py:270-275` reads `input=` / `output_dir=` / `val_ratio=` / `test_ratio=` / `strategy=` / `seed=` — **there is no `train_ratio`**; train is the remainder. | `training_panel.py` sends `format=`, `anno=`, `output=`, `train_ratio=`. **"Create Splits" currently splits the bundled sample dataset into the default folder and reports success.** Highest-priority data-loss bug in the suite. |
| **V8** | `pytest tests/test_labeling_ops.py tests/test_annotation_v2.py tests/test_gui_commands.py` → **52 passed**. `AnnObject.to_dict()` hardcodes `"class_id": 0` and omits `keyframe`; both are asserted by tests. | This is the green line. A human-only project's JSON must stay byte-identical. |

### §0.0.3 Corrections to the implementation spec found in this run

These four items in the circulated spec are **wrong as written**. Implement what this table
says, not what the spec said.

| Spec claim | Measured reality | What to do |
|---|---|---|
| `obb.rotate_ccw` key suffix is `"LBracket"`, `obb.rotate_cw` is `"RBracket"` | Neither `mvKey_LBracket` nor `mvKey_RBracket` exists in dpg 2.3.1. The `[` / `]` constants are **`mvKey_Open_Brace`** and **`mvKey_Close_Brace`**. | Store `key="Open_Brace"` / `key="Close_Brace"` in `shortcuts.BINDINGS` and assert those in `test_legacy_bindings_preserved`. The physical keys and the user-facing labels stay `[` and `]`. (`annotator.py:804-811` already carries this fallback chain; it was right and the spec dropped it.) |
| `edit.propagate` key suffix is `"Space"` | `mvKey_Space` does not exist; the constant is **`mvKey_Spacebar`**. | Store `key="Spacebar"`. The user-facing label stays `Space`. |
| White on `ACCENT` measures **1.91:1** | Measured **2.63:1** (WCAG relative luminance). | The prohibition is unchanged and the invariant `contrast(white, ACCENT) < 3.0` still holds. Quote **2.63:1**. Never quote 1.91. |
| `TEXT_DIM/SURFACE` 5.30 · `ACCENT/SURFACE` 5.52 · `ON_ACCENT/ACCENT` 6.99 · `INTERPOLATED/SURFACE` 4.05 | 5.34 · 5.55 · 7.02 · **4.45** | Use the measured column in §1. `theme.verify_tokens()` is the authority; if it and this document ever disagree, `verify_tokens()` wins and this table gets corrected. |

Because a wrong key suffix is an `AttributeError` at startup,
`tests/test_gui_shortcuts.py::test_every_key_constant_exists` — `hasattr(dpg, "mvKey_" + b.key)`
for every binding — is the single most valuable test in the suite. It must exist before
`shortcuts.py` ships.

---

# Part I — Orientation

## §0.1 Design principles distilled from YOMU

YOMU is a more mature project by the same lab: a Svelte/TypeScript front end over a
FastAPI/Python backend, covering five "modes" (YOLO detection, POSE keypoints, MOTEN motion
energy, COMBO, LINK). YOAKE is Dear PyGui and single-purpose. **Nothing was copied from YOMU's
code.** What was mined is interaction design and information architecture. Each principle below
names the concrete YOMU behaviour that produced it, so a future reader can judge whether the
principle still applies when the situation changes.

### P1 — Candidate vs accepted is the review primitive, and candidates never reach disk

*YOMU behaviour.* An `Annotation` carries `accepted: bool`. `save_labels()` skips every
`!accepted` row, and `save_scores()` records confidences only for accepted rows. The server
returns a hard 400 on `reviewed=true` while pending candidates exist, with the message
*"Accept or discard pending candidates before marking this frame reviewed."* Batch detection
drops the frame's existing candidates and keeps its accepted boxes, then pushes new detections
as `accepted: false`.

*Why it is load-bearing.* Because nothing pending is persisted, **re-running the detector is
always safe**, and an unreviewed run can never poison a training set. Accept becomes the only
act with consequences. Every other property of the auto-annotation design rests on this one.

*In YOAKE.* `AnnObject.accepted` is in-memory only; `_write_annotation` filters to
`[o for o in objs if o.accepted]`. See §7.

### P2 — Hit-test and draw share one predicate

*YOMU behaviour.* `showVertices(v, a, i)` is called by **both** `render.ts:drawScene` and
`geometry.ts:handleAt`, with a source comment saying so explicitly: what you see is exactly what
you can grab. Class-tab-hidden annotations are not drawn, not hit-testable, not selectable and
not reachable with `Q`; selecting a tab that hides the current selection **clears the
selection**, with a comment explaining that otherwise Delete/Space would act on something
invisible.

*In YOAKE.* `_visible_objects(frame)` returns the drawable list, and `_hit_object` iterates the
*same* list reversed. Nothing visible is un-grabbable; nothing invisible is selectable.

### P3 — Never overwrite human work, and re-check that immediately before writing

*YOMU behaviour.* The CoTracker auto-label endpoint states its provenance policy at the top:
**anchor** = a frame with labels whose origin contains `human`, or whose origin is empty (legacy
= trusted); **target** = a frame with no labels, or one whose origin is exactly `["track"]`
(a previous auto pass). Frames tagged `human` or `llm` are never touched; excluded frames are
neither anchor nor target, because "the reason for exclusion is usually that the label is wrong,
so don't propagate a wrong label into unlabelled frames."

Targets are chosen before a job that can run 15 minutes, and the review screen has no write
lock — so `commit_new()` **re-reads the provenance sidecar immediately before each write** and
skips any frame that has since gained `human`/`llm`. The count is reported to the user as
`skipped_edited` → *" (protected {e} frames edited during the run)"*.

*In YOAKE.* `merge_candidates(..., policy.skip_frames_with_accepted=True)` skips any frame
holding an accepted human box and counts it in `skipped_protected`. `Track Forward` writes only
into frames where that `track_id` has no object.

### P4 — A gated control is visible, dimmed, disabled and tooltipped — never hidden

*YOMU behaviour.* A feature whose capability is missing gets class `ui-locked`
(`opacity:.45; filter:saturate(.5)`) **plus** `disabled` **plus** the capability's explanation
as its `title`. Applied uniformly: the four training-engine buttons, the AI buttons, the prompt
optimiser. `setEngine()` silently falls back to the default if you try to select a locked one.
The workflow guide states the convention in prose: *"Buttons for features whose dependencies are
missing are dimmed and cannot be pressed (hover a button for the reason)."*

*Why.* Hiding teaches the user the feature does not exist. Dimming teaches them it exists, why
it is off, and what to type to turn it on.

*In YOAKE.* `widgets.locked(tag, reason)`. See §3.

### P5 — Diagnose an empty result into distinct causes, and name the next action and where it lives

*YOMU behaviour.* `screenChecked()` produces **three different** messages for "zero items",
never one generic shrug:

* `AI screening needs per-class criteria (questions) — set them in tab 1`
* `No pending candidates in the checked frames — run "Detect all frames" in tab 1 first (accepted boxes are excluded; press Space to turn one back into a candidate)`
* `No candidates for classes that have criteria (all candidates belong to classes without one)`

And every completion message hands off: `Batch done: candidates on {f}/{n} frames — uncheck
unwanted frames, then "Accept checked & save"`.

*In YOAKE.* `auto_label.describe()` emits three distinct empty-result diagnoses (§7.6), and
every terminal message names the next key or the next step number.

### P6 — A stop is a stop, never an error

*YOMU behaviour.* `state.autolabel_stopped` exists **solely** so that `returncode == -15` with
`done != true` is not reported as a failure. A cancelled CoTracker run shows no error dialog even
when a partial error exists, and the completion string is composed from optional clauses:
`Auto-labeling interrupted: {n} frames already applied (anchors: {a})` `+ " (skipped {k}
low-confidence points)"` `+ " (protected {e} frames edited during the run)"`.

*In YOAKE.* Set `self._stopped = True` **before** `terminate()`. Report `Auto-annotation
stopped. The {n} candidates already produced are kept.` Never `Error (code=1)`.

### P7 — Undo: snapshot at press, push at release only if moved, restore verbatim otherwise

*YOMU behaviour.* Per-frame stacks capped at **50**, each entry a `structuredClone` of both the
annotations and the edge overrides. `pushUndo` precedes every discrete mutation. For drags the
snapshot is captured at `pointerDown` and pushed at `pointerUp` **only if `moved`**; if the
pointer did not move it is restored verbatim, with a source comment saying that a missing restore
lets sub-pixel drift silently dirty the frame so later edits never save. `undo()` pops, then
**clears the selection** because indices no longer mean anything, and reports
`Undone ({n} steps left)` / `Nothing to undo`. **There is no redo anywhere in YOMU.**

*In YOAKE.* Identical, 50 deep, no redo. See §0.3.6.

### P8 — Three-state dirtiness: dirty / touched / clean

*YOMU behaviour.* `markDirty(i,'human')` = a real human edit (unsaved **and** stamp the 👤
badge). `touched()` = redraw but **do not** mark unsaved, used when only candidates changed —
because candidates are never persisted, so producing them must not make the frame look dirty. A
third guard (`withRepo()`, `internal++/--`) stops the editor's own repository writes from
tripping its external-change detector.

*In YOAKE.* Candidates never set a dirty flag and never count toward the "unsaved" warning; they
are reported separately: `{n} candidates are not saved. Press Y to accept them first, or N to
discard them.`

### P9 — Async ownership: op token plus generation counter, checked after every await

*YOMU behaviour.* Every long-running operation takes `++opToken`; `setBusy(true)` and the
button-enabled flags belong to the token holder only, so a stale operation's `finally` restores
nothing. A project `generation()` is captured **before** any confirm dialog and re-checked after
every `await`; on mismatch **both result and error are discarded silently** — no toast, no alert,
no mutation. The same discipline appears in five places (`openSeq`, `repo.generation`, per-frame
`versions`, folder identity, audit generation). Where the user was actually waiting, the drop is
announced: `Frame changed; detection results discarded`.

*In YOAKE.* Dear PyGui has no event loop to await on, so the equivalent is: a worker thread never
touches `dpg`; it pushes into a `queue.Queue`; a self-re-arming `dpg.set_frame_callback` drains it
on the main thread; and every drained message carries the job id it belongs to, which is compared
against the current job before it is applied. This is also the fix for the three existing
cross-thread `dpg` mutations catalogued in §9.

### P10 — A permanent on-screen cheat sheet beats a help dialog

*YOMU behaviour.* The entire discoverability strategy for the editor is a `.help` block
permanently visible in the right-hand column at 12 px in `--text-dim`, listing every key and
mouse gesture. There is no help modal for the editor at all.

*In YOAKE.* The sidebar `Quick Start` block names the real keys, the toolbar chip names the next
action for the active tool, and `F1` opens the generated full list. All three are generated from
`shortcuts.BINDINGS`, so none of them can drift.

### P11 — The step number is in the navigation label; there is no checklist

*YOMU behaviour.* Tabs are literally named `0. Workflow`, `1. Source / Detect`,
`2. Review / Edit`, `3. Export`, `4. Train`, `5. Scan / Evaluate`, `6. Video list`, and every
help string refers to them by number ("fix in tab 2, then retrain in tab 4"). Optional sections
are prefixed `Extra.` rather than moved to a menu. A tab renames itself per mode
(`1. Source / Detect` ↔ `1. Project`) instead of proliferating tabs. **Nothing is ever ticked
off.** "Have I done step 4?" is answered by run-history counters (`trained xN` / `untrained`), by
locked controls that explain the unmet precondition, and by empty states that name the next
action (`(no trained project — train in tab 4 first)`).

*Why no checklist.* A checklist is a fourth source of truth that goes stale the first time
someone does the steps out of order.

*In YOAKE.* Pages are `0. Workflow` / `1. Label` / `2. Train` / `3. Analyze`, `Ctrl+1..4`
switches them, and prose says "step 1", never "the Labeling tab".

### P12 — Refuse to let a number look more meaningful than it is, and record the measurement next to the threshold

*YOMU behaviour.* Shipped strings include *"Registration confidence is not a calibrated success
probability."*, *"Individual order is not a tracking ID."*, *"Matching array indices refer to the
same detection within one frame, not tracking IDs."* The Codex refine endpoint's docstring says
the score "is the model's self-report (almost always 1.0) and is not calibrated, so **no
threshold is applied**". Conversely, thresholds that *were* measured carry the measurement:
`_B2_NEAR_RATIO = 0.3` is annotated with the false-positive rate at 0.5 / 0.3 / 0.2 on a named
dataset; `context = 1` carries the median error and hit rate at ctx 1 / 2 / 8; the
active-learning panel's help says *"98.0% really were off by more than 10px - 6.2x better than
picking at random (Spearman -0.426)"*.

*In YOAKE.* `MergePolicy.iou_suppress = 0.60` is documented **as a judgement, not a
measurement**, in its own docstring, with a note to record the dataset and false-merge rate once
there is a real video to measure against. Tracking confidence is labelled "the tracker's own
estimate, not an accuracy".

### P13 — Colour spends on one axis per view

*YOMU behaviour.* In the **editor**, box colour encodes *acceptance* (green accepted / amber
candidate / red selected) and class appears only as text in the label chip. On the **tiles**,
colour encodes *class* (to match the legend) and acceptance is carried by solid-vs-dashed stroke.
For POSE points, colour is already spent on class, so candidacy is shown by an extra dashed ring
and a trailing `" ?"` on the label — with a source comment saying exactly that.

*In YOAKE.* Track identity owns hue (`TRACK_PALETTE`); candidacy owns **stroke** (dashed) plus
the single semantic colour `PENDING`; selection owns **handles** (`DANGER`) and +1 px. No view
encodes two things in hue at once.

### P14 — A left-hand-biased keymap so the right hand never leaves the mouse

*YOMU behaviour.* `A D Q R Z X Y N F G L T` plus `1`–`9`, all reachable by the left hand while
the right hand stays on the mouse. `Ctrl+S` is the only binding that survives while typing in a
field.

*In YOAKE.* `V W E R H X` (tools), `A D` (frames), `Q Y N T F Z` (review and view), `0`–`9`
(actions). `Ctrl+S` is the sole `ALWAYS_ALLOWED` action. See §6.

### P15 — Never seek in a video

*YOMU behaviour.* Stated twice in the source, with the measurement:
`cap.set(CAP_PROP_POS_FRAMES, n)` is off-by-one on non-integer fps — at fps = 30.0032, seeking to
6000 returns what sequential decoding calls 5999 — and it **agrees at frame 0**, so the error is
near-impossible to notice. Frame extraction and analysis both decode sequentially, so seeking
would anchor on the neighbouring frame and write every point one frame off. Everything uses
sequential `grab()` with selective `retrieve()`.

*In YOAKE.* The auto-annotate frame export **must** use `grab()`/`retrieve()` and write
`index_map.json`. Playback in `analysis_panel` uses sequential decode for `+1` steps and
`cap.set` only for jumps.

The annotator's `_read_frame_bgr` now follows the same rule. It used to seek on **every**
access, which was written off as "acceptable for random single-frame access" — but the access
it served most is not random: it is the Next-frame key, held down. On a long-GOP codec each
seek re-decodes from the previous keyframe, so stepping through a video walked the decoder
over the same group of pictures again and again, and the cost grew with the GOP length rather
than with the step. `_decoded_idx` records where the capture actually is, so `idx + 1` is a
bare `read()`, a hop of up to `_SEQ_SKIP_MAX` frames is `grab()` over the gap, and only a
backwards move or a real jump seeks. It removes the off-by-one from the common path as well,
for the reason stated above.

*Related Windows trap, also from YOMU:* `cv2.imread` returns `None` silently on non-ASCII
Windows paths. Read images with `np.fromfile` + `cv2.imdecode`. A silent `None` inside a
Japanese-named folder drops a whole segment with no warning.

### P16 — One status line, one persistent context line, and panels own their own progress

*YOMU behaviour.* A single flat snapshot object is broadcast to the whole chrome, and the status
bar has exactly **two** text slots: `status` (left, transient) and `statusRight` (right,
persistent context — zoom %, output path, counts). There are **no toasts**. Background work is
represented globally by one queue badge; every other progress readout lives inside the panel that
started the job. Panels never write into each other's status, and the shell never invents
progress of its own.

*In YOAKE.* `hub_status` (left, transient) + `hub_status_right` (right, persistent) with public
`Hub.status(msg)` / `Hub.context(msg)`. This is the cheapest fix for "a failed run on the
Training page is invisible once you switch tabs".

### P17 — Quote the cost before the user commits

*YOMU behaviour.* `Ask {a} to estimate keypoints for {n} frames (10-20 s per frame). Proceed?` ·
`Long anchor spans can take several to ~15 minutes. Run?` · `This takes minutes to tens of
minutes.` · `(⚠ over 100 — extraction and detection will take a while)` · and, best of all, a
VRAM estimator that prints **the batch size that would fit**: `Estimated VRAM {est} GB exceeds
this PC's {tot} GB. Lower the batch to {fit} or less, or use -1 (auto)`.

*In YOAKE.* Any dialog that starts a multi-minute job states the cost in the confirm text, and
the auto-annotate dialog's `Apply to` radio makes the scope — and therefore the cost — explicit.

### P18 — State the negative guarantee on every risky control

*YOMU behaviour.* Nearly every dangerous-looking control carries an explicit "what this does NOT
do" clause: `Does not modify a single byte of your labels` · `the source analysis runs are never
modified` · `Images and labels are kept; the frames are only left out of training, export and
auto-labeling. Reversible` · `Display only — saved labels and detection are unaffected` ·
`nothing is saved`.

*In YOAKE.* `Candidates are not saved as labels. Review them, then accept the ones you want.` ·
`Boxes you drew by hand are kept.` · `Ctrl+Z can undo this.`

### P19 — Write partial results as you go, and let the UI apply them mid-run

*YOMU behaviour.* Long jobs rewrite `out.json` after **every finished segment** via temp +
`os.replace`, and the browser polls a `/api/status` that returns a **cumulative** list of applied
frame indices plus a job counter — explicitly not a delta, because "missing a poll or
double-applying gives the same result". Tiles update live while the request is still open.
Combined with P3's protect-edited-frames rule, a 15-minute job feels interactive and is safe to
work alongside.

*In YOAKE (aspiration, not yet required).* The current auto-annotate design reads
`predictions.json` once at the end because the CLI writes it once at the end. If per-frame
streaming output is ever added to `cli.run_predict`, this is the pattern to adopt; until then the
honest UI is an indeterminate bar plus elapsed time plus the last log line (P20).

### P20 — Do not invent a protocol the backend does not have

*YOMU behaviour.* Where real progress exists it is shown to the frame
(`Scanning… {a}/{b} frames`, `[3/6] Analyzing frames 1204/8000 (15%) ETA 00:04:12 | elapsed
45s`). Where it does not, the UI says so instead of faking one.

*In YOAKE.* `yoake predict` emits no per-frame progress, so the auto-annotate dialog shows an
**indeterminate** bar, `Running auto-annotation… 0:42 elapsed`, the last log line, and a `Stop`
button. Regex-sniffing the CLI's stdout for a percentage is explicitly rejected: it invents a
contract the CLI does not offer and will show a confidently wrong bar the first time a format
string changes.

---

## §0.2 Information architecture

Two processes. The **hub** (`python -m htrtdetr.gui`, four pages) and the **annotator**
(`python -m htrtdetr.annotator`, its own window). They communicate through files and through the
hub's launcher; there is no IPC.

```
   hub (one viewport, four pages)                  annotator (separate process)
   +-------------------------------+               +-----------------------------+
   | 0. Workflow   capability,      |  launches --> | menubar                     |
   |               steps, keys      |   with        | toolbar (tools/view/review) |
   | 1. Label      launch, recents  |   child_env   | +------------+-----------+  |
   | 2. Train      dataset/train/   |               | |  canvas    |  sidebar  |  |
   |               evaluate         | annotations   | | (drawlist) | 11 panels |  |
   | 3. Analyze    predict, review  | <-- .json --  | +------------+-----------+  |
   |                                | predictions   | navbar + status             |
   | status bar (2 slots)           | --- .json --> |                             |
   +-------------------------------+               +-----------------------------+
```

### §0.2.1 The hub shell

| Region | Tag | Contents | Rules |
|---|---|---|---|
| Left rail | `nav` (260 px) | Brand, four page buttons, `Current Workspace`, `Project Root`, `Output Root`, `Tips` | Active page = `theme.accent_theme()` plus a 3 px `ACCENT` left bar. The `">  Label"` whitespace hack is deleted. |
| Content | `content` | Exactly one visible `panel_*` child window | Pages are built once at startup and toggled with `show`. |
| Status bar | `hub_status` / `hub_status_right` | Left: transient one-liner. Right: persistent context (`Project: … \| Output: …`) | Panels call `Hub.status()` / `Hub.context()`; they never write into another panel's log. |

Viewport is clamped: `WINDOW_W = min(1440, display_w - 80)`, so the window cannot exceed a
1366×768 laptop screen.

### §0.2.2 `0. Workflow` — the orientation page

Always enabled, always first, built at startup. It is the only page that works with no project,
no video and no checkpoint.

| Block | Contents | Rule |
|---|---|---|
| Capability | One row per probe from `auto_label.probe()`: `✓` or `⚠`, the detail, and the exact fix command, with a `Copy` button on the combined command | **Rendered only when something fails.** A clean machine sees nothing. |
| `Step \| What to do \| Tips` | A `dpg.add_table` with four rows, one per page | The Tips column carries the hard-won notes — e.g. `Each run writes to runs/train/stage{N}/ and is never overwritten.` and `Picking "detection" as the model mode forces stage 1 regardless of the stage you chose - the log says so when it happens.` |
| Keyboard and mouse | `widgets.kbd_table(shortcuts.help_rows("annotator"))` and the hub's own | Generated; cannot drift. |
| Recommended loop | Four numbered lines | Ends: `One round of human review beats more threshold tuning. Use step 3's timeline to find the moments the model got wrong, add those frames, and correct them in step 1.` |

**No checkbox checklist** (P11).

### §0.2.3 `1. Label`

| Block | Contents |
|---|---|
| Launchers | `Open Annotator`, `Open Annotator with Video…`, `Open Annotator with Annotation JSON…` |
| Recents | Per-field recent paths (`recents.get`), which also seed the file picker's `initialdir` |
| Pointer | A short block pointing at `0. Workflow` for the generated key table — **not** a second hand-maintained prose list |

Launch is honest: `The annotator is starting…` → `Annotator running (PID {n}).` or
`The annotator could not start: {reason}` with the first captured stderr line. It spawns with
`env=auto_label.child_env(root)`, `stdin=DEVNULL`, `CREATE_NO_WINDOW`, and the hub polls
`proc.poll()` for about 1.5 s from the render loop.

### §0.2.4 `2. Train`

Three tabs.

| Tab | Purpose | Key controls |
|---|---|---|
| `1. Dataset` | Convert → split → inspect | `Convert Annotations` (`mode=`/`input=`/`output=`, modes `csv \| json \| video_csv`), `Create Splits` (`input=`/`output_dir=`/`val_ratio=`/`test_ratio=` with a live `train 70% / val 15% / test 15%` readout clamped so val+test < 1.0), `Show Statistics` |
| `2. Training` | Stage select → data → hyperparameters → run → evaluate | Stage combo (1–4) with per-stage defaults, model mode, paths, epochs/batch/lr/window/imbalance, `Start Training`, `Watch Output Folder`, `Stop`, then the evaluation block |
| `3. Analyze Outputs` | Post-hoc plots | `timeline \| distribution \| id_switches \| confidence` |

Right-hand pane: the live run monitor — two plots (loss, metrics) plus a one-line summary.

Three rules this page must obey:

1. **Pre-flight summary into the log before `runner.run`**, aligned label column, including
   `Effective:` with the reason when the model-mode coercion fires
   (`1 (forced by model mode "detection")`). Cheapest possible fix for "I clicked Start and
   something else happened".
2. **Pre-validate `data.train_root`** and refuse to launch when it does not exist:
   `Training data not found: {path}. Create splits in step 1, or point "Training data" at an
   existing folder.` That is the only path on which `cli.run_train` reaches its `input()` prompt,
   and `gui/runner.py` cannot be given `stdin=DEVNULL` without editing it (out of scope), so the
   hang is prevented at source.
3. **Attach the monitor *after* `runner.run`**, clearing the four `mon_s_*` series first.
   Attaching before means the first 2 s poll paints the *previous* run's curve.

Stage defaults, for reference (`training_panel.STAGE_META`):

| Stage | Label | Epochs | Batch | LR | Model default | Window | Imbalance |
|---|---|---|---|---|---|---|---|
| 1 | Stage 1 - Detector | 50 | 4 | 1e-4 | `detection` | — | — |
| 2 | Stage 2 - Action Head | 30 | 8 | 5e-5 | `detection+temporal` | yes | yes |
| 3 | Stage 3 - ID Head | 30 | 8 | 5e-5 | `detection+temporal` | yes | — |
| 4 | Stage 4 - Unified Fine-Tuning | 20 | 2 | 1e-5 | `detection+temporal` | yes | — |

### §0.2.5 `3. Analyze`

| Region | Purpose |
|---|---|
| `1. Input Files` | Video, checkpoint, output folder |
| `2. Run or Load Predictions` | Score threshold, window size, max frames, `Run Inference`, `Load Predictions JSON`, `Load From Output Folder`, `Stop` |
| `3. Review Playback` | Transport, speed, display threshold, action names, frame slider, nine display toggles |
| Canvas + inspector | `ana_canvas` drawlist; the right pane shows ID / action / confidence / OBB / heads / evidence bars |
| Timeline | Per-track (or per-head) action lanes with a white current-frame cursor |
| Status + log | `ana_status`, `ana_log` |

Its job is **review**, not creation: a run writes to a timestamped folder that is never
overwritten, and this page reads it back. The loop closes with `Send Predictions to Annotator`,
which launches the annotator with `--import-predictions <path> --video <path>`.

Display toggle set (nine keys; `DEFAULT_TOGGLES` in `canvas.py`):

| Key | Label | Default | Draws |
|---|---|---|---|
| `obb` | `Rotated box (OBB)` | on | Filled rotated rect (alpha 40) plus a heading line to the front-edge midpoint |
| `bbox` | `Axis-aligned box` | on | AABB — used when `obb` is absent or its toggle is off. **Currently has no checkbox; that is the bug.** |
| `context_box` | `Behavior context box` | off | Dashed, thin, unfilled, alpha 120 |
| `trajectory` | `Trajectory` | off | Per-track centre polyline, alpha ramping 40→220 newest-last, causal only |
| `track_id` | `Track ID` | on | `ID{n}` chip |
| `pose` | `Pose` | off | Keypoint circles, skipping `v <= 0` |
| `action` | `Action` | on | Action name in the label |
| `confidence` | `Confidence` | on | `{score:.2f}` in the label |
| `explanation` | `Explanation` | off | Top-3 evidence groups by absolute value, `+0.00` formatted, gold if ≥ 0 else blue |

### §0.2.6 The annotator

| Region | Contents |
|---|---|
| Menubar | File / Edit / Project / Review / Help — `gui/menubar.py` |
| **Top bar** (68 px, two rows) | Row 1, *where am I*: Videos / Previous Video / Next Video · save chip · model chip · identity text. Row 2, *the loop*: `Pick frames to label` → `Review` → `Train` → `Analyze` · Undo / Redo / Save · Settings · tool chip. Both in `gui/chrome.py`. |
| **Tool rail** (108 px, left) | Three captioned groups — **Tools** (the six of §0.3.2), **Assist** (`Continue`, `Keep tool`), **View** (`Fit`, `Panels`, zoom readout). `gui/chrome.py`. |
| Canvas | `canvas_win` child window containing the `canvas` drawlist. Two draw layers: `layer_image`, `layer_overlay`. |
| Right column (280 px) | Inspector **or** Review Inbox, one at a time. The eleven-panel sidebar is gone; `annotator._LEGACY_TAGS` records where each panel went. |
| Navbar | First / Prev / frame input / Next / Last / Copy from prev, over the multi-lane timeline |
| Status | One status line |

`Train` and `Analyze` are the third and fourth arrow of §0.2.7 and they close
**inside this window**. Only `Project > Open the workflow window (full
training)...` spawns `python -m htrtdetr.gui --page workflow --project <dir>`
(`gui/commands.workflow_command`, parsed by `hub.parse_argv`), for the
production run. Those two buttons used to spawn it too, which made the loop a
thing you left the labelling window to continue — and therefore did not.

Canvas height is `max(vh - 152, 300)` in **both** `_build_canvas_panel` and
`_on_viewport_resize` (it was `vh - 110`; the toolbar takes 42 px). Changing only one of the two
clips the canvas and fails `--selftest` case 1 immediately.

### §0.2.7 The intended user path

```
 step 0  Workflow ......... check capabilities, read the four steps
 step 1  Label ............ open the annotator, draw boxes on a few frames
            +- T .......... track one box forward, review the amber candidates, Y / N
            +- Ctrl+R ..... (torch present) auto-annotate a range, review, Ctrl+Right Y
            +- Ctrl+S ..... save annotations.json into the project folder
            +- Train ...... one round: export the labelled frames, train
                            stage 1 on them, adopt the checkpoint (§0.4.9)
            +- Analyze .... run that checkpoint back over the videos, rank by
                            uncertainty, review the candidates -> round again
 step 2  Train ............ the PRODUCTION run, once: stages 1-4, full controls
 step 3  Analyze .......... predict over the whole video, scrub the timeline,
                            find the moments the model got wrong
            +- Send Predictions to Annotator --> back to step 1 on those frames
```

The loop is the product. A single pass through steps 1→2→3 is not the goal; the goal is that the
third arrow (Analyze → Label) is cheap enough that people actually take it.

**The loop is a series, and it ends.** Ten turns of `Pick → correct → Train` is the shape the
study is built around, so three things that were invisible now have to be said out loud
(§0.4.10):

* **which turn this is.** `Round 7.` opens the Train dialog's plan paragraph, and the Review
  dock's *This round* block carries the same number.
* **how much of this turn is left.** `12 of 100 frames checked here, 31 of 200 across the
  project.` Counted from the selection against `annotations`, never stored.
* **whether the turns have stopped teaching.** Agreement — the share of the model's proposals
  a human kept — across the last three rounds. Not the holdout score: the holdout grows with
  the label count, so no two rounds of this shape are scored on the same frames, which is the
  comparison `adoption()` already refuses to make.

And then it ends: `Project > Check every frame (final pass)...` runs the model over every frame
of every video, the human checks the places worth checking, signs off video by video, and
`File > Export the verified dataset...` writes what they signed off (§0.4.11).

**Every arrow has to exist inside the window the user is in — and the loop's arrows have to
close there too.** The Label→Train and Train→Analyze arrows were drawn in this document and
nowhere in the product. Making them spawn the other window was not enough: an arrow that opens a
second application is one the user takes once, and the loop is only worth anything taken twenty
times. Train and Analyze are therefore *operations of the annotator* (§0.4.9), and steps 2 and 3
keep the job that genuinely is done once — the production run over finished labels.

---

## §0.3 The annotation interaction model

This section is the behavioural specification for `annotator.py`. Build on the code that is
already there: `_canvas_origin`, `_get_canvas_mouse`, `_is_over_canvas`, `_recompute_scale`,
`set_tool`, `toggle_draw_tool`, `_on_sticky_draw_toggle`, `_refresh_tool_ui`, `_escape`,
`_sticky_draw`, the `W` binding and the single-shot return-to-Select were root-caused and
repaired in a previous session and verified by synthetic input. Extend them; do not re-derive
them.

### §0.3.1 The design thesis

1. **The active tool must be impossible to misread.** Three redundant channels at once, one of
   them drawn *inside the canvas*, because the user's eye is on the image, not on a sidebar
   600 px away.
2. **Candidate vs accepted is the auto-annotation primitive** (P1). Machine output arrives as
   dashed amber candidates that are never written to `annotations.json`.
3. **The review loop, not the detection, is the product.** `Ctrl+Right` jumps to the next frame
   that still holds candidates; `Q` cycles them; `Y`/`N` clear a frame. Without that jump, a
   400-candidate run degenerates into arrowing through empty frames hunting for amber.

### §0.3.2 The five tools

Mutually exclusive. The old hidden `Rotated box / OBB mode (R)` checkbox is **removed** — the
mode becomes a visible tool.

| # | Tool | Tag | Key | Behaviour | Sticky? |
|---|---|---|---|---|---|
| 1 | `Select` | `tb_tool_select` | `V`, `Esc` | Click selects the topmost box under the cursor. Drag inside moves it. Drag a corner handle resizes it. Drag the yellow handle rotates an OBB. Click empty space deselects and starts a rubber band that creates a box on release. | permanent |
| 2 | `Draw Box` | `tb_tool_draw` | `W` | Press-drag-release **always** starts a new axis-aligned box, even on top of or inside an existing one. Release with both sides ≥ `_MIN_BOX_PX` (5 image px) creates it. | single-shot unless `Keep drawing` |
| 3 | `Draw Rotated Box` | `tb_tool_obb` | `E`, `R` | Identical to `Draw Box`, then `make_obb()` (θ = 0), and the new object is selected with its rotation handle live so the next gesture can be the rotation. `R` is a legacy alias kept because it already meant "rotated". | single-shot unless `Keep drawing` |
| 4 | `Pan` | `tb_tool_pan` | `H` | Left-drag moves the view. (Middle-drag pans from **every** tool always; this button is the discoverable form of a gesture that never goes away.) | permanent |
| 5 | `Delete Box` | `tb_tool_delete` | `X` | Click removes the topmost box under the cursor, one undo step each. | **sticky** — it is an eraser; you rarely delete just one. `Esc` or `V` exits. |
| 6 | `Click animal` | `rail_click` | `C` | One press fits one box to the animal under the cursor (`docs/CLICK_SEGMENT.md`). | single-shot unless `Keep drawing` — **changed 2026-09-17**, see below |

**Why `Click animal` is single-shot.** It shipped sticky, on the reasoning that it is a tool
for picking up animals one after another. The actual gesture sequence is not that. A fitted
box is rarely right first time and the next thing the hand does is grab a handle; with the
tool still armed that grab lands as a second click and drops another box on top of the first.
A tool meant to save a drag was costing a delete. The return-to-Select lives in
`_apply_click_result`, the single confluence of the synchronous (local / model) and
asynchronous (sam) backends, so the behaviour cannot differ by backend. The opt-out is the
existing `Keep drawing after each box` (`_sticky_draw`) that `W` and `E` already use — no new
setting was added for it.

### §0.3.3 The tool rail, top to bottom, and the top bar

The horizontal `tb_*` toolbar this section used to specify no longer exists. The tools moved into
a 108 px vertical rail beside the canvas (`toolrail`, built by `gui/chrome.toolrail`) and the
commands moved into the top bar's second row (`gui/chrome.topbar`). Every `tb_*` tag below is
gone; the rail's tags are `rail_*` and the top bar's are `top_*`.

**The rail is three captioned groups, and the captions are load-bearing.** It was eleven
identical buttons in one column, two of which — `Continue` and `Keep tool` — are not tools at
all: pressing them does not change what the next click does. A first-time user reading top to
bottom had no way to tell those apart from `Delete Box`, which does.

| Caption | Rows | Tags |
|---|---|---|
| `Tools` | the six of §0.3.2, each labelled `word + key` | `rail_select` … `rail_delete` |
| `Assist` | `Continue` (never locked — §`_refresh_propagation_ui`), `Keep tool` (`_sticky_draw`) | `rail_propagate`, `rail_sticky` |
| `View` | `Fit F`, `Panels`, live zoom readout | `rail_view_fit`, `rail_panels`, `rail_zoom` |

**Zoom in / Zoom out are not buttons.** The wheel zooms over the canvas, `+` and `-` zoom from
the keyboard, and `rail_zoom` reads out the result: three channels for one thing did not need a
fourth that also cost two of the rail's eleven rows. The readout's tooltip names the wheel and
both keys, so the removal does not take the discovery with it.

**The top bar's second row is the loop** (§0.2.7), left to right, and nothing else:

| Label | Tag | Key | Behaviour |
|---|---|---|---|
| `Pick frames to label` | `top_smart` | `Ctrl+R` | The one sampling/pre-labelling window (§0.4.2). Current video, or the project when none is open. |
| `Review` | `top_review` | — | Toggle the Review Inbox for this video. |
| `Train` | `top_train` | — | Run one active-learning round **here** (§0.4.9) and adopt the checkpoint it produces. |
| `Analyze` | `top_analyze` | — | The `Pick frames to label` window on its analysis preset: detect on, strategy `uncertainty`. Locked, with the reason, until a checkpoint exists. |
| `Undo` / `Redo` / `Save` | `top_undo` / `top_redo` / `top_save` | `Ctrl+Z` / `Ctrl+Y` / `Ctrl+S` | 50-deep per-frame stack; silent write to `project_dir` when set. |
| `Settings` | `top_settings` | — | The six-tab dialog. It is the only door: the Project menu has one `Settings...` row, not one row per tab. |
| *(chip)* tool sentence | `tool_mode_text` | — | One live sentence (§0.3.4). |

`Accept all` / `Discard all` and the candidate count are **not** here. They belong to a frame that
has candidates, so they live in the Inspector's no-selection context and appear only when the
count is above zero — an Accept All that is always present and usually disabled teaches the user
to ignore it.

Every label that has a key gets it appended by `shortcuts.with_key()`, and every button gets a
tooltip generated as `f"{binding.help} ({binding.label})"`. A shortcut can therefore never drift
from its documentation.

### §0.3.4 Making the active tool impossible to misread — four channels

1. **Fill.** `_refresh_tool_ui()` binds `theme.accent_theme()` to the active button and
   `theme.ghost_theme()` to the other four via `dpg.bind_item_theme`. Filled accent with
   near-black text against bordered transparent ghosts is unmistakable, and `ACCENT` measures
   4.75:1 against the toolbar strip.
2. **Chip.** `tb_tool_chip` carries one sentence in `ACCENT` (`TEXT_DIM` for Select) that always
   names the next action:

   | Tool | Chip text |
   |---|---|
   | Select | `Tool: Select — click a box to edit it, or press W to draw.` |
   | Draw Box | `Tool: Draw Box — drag on the image to create a box. Esc cancels.` |
   | Draw Rotated Box | `Tool: Draw Rotated Box — drag, then turn the yellow handle or press [ / ].` |
   | Pan | `Tool: Pan — drag to move the image. F fits it to the window.` |
   | Delete Box | `Tool: Delete Box — click a box to remove it. Esc stops.` |

3. **Canvas frame plus caption.** When the tool is not Select, `_render_canvas` draws, as the
   last items in the **overlay layer**, a 2 px rectangle just inside the canvas edge in the tool
   colour (`ACCENT`, or `DANGER` for Delete) plus a 13 px caption at `(12, 10)`:
   `DRAW BOX — Esc to cancel` / `DRAW ROTATED BOX — Esc to cancel` / `PAN — F to fit` /
   `DELETE — Esc to stop`.
4. **Crosshair.** In `draw`, `obb` and `delete`, an `add_mouse_move_handler` stores
   `self._cursor_canvas` and the overlay layer draws a full-width plus full-height 1 px guide at
   the cursor (`ACCENT` at α 90; `DANGER` for Delete). Guarded by
   `self._tool in ("draw","obb","delete") and self._is_over_canvas()`, so Select mode pays
   nothing.

The `tool_mode_text` tag is **kept** and repointed at the chip, so `_refresh_tool_ui` needs one
edit.

### §0.3.5 The full mouse grammar

| Gesture | Tool | Result |
|---|---|---|
| Left press on a handle of the **selected** box | Select | `resize`, or `rotate` on the yellow handle. Snapshot taken. |
| Left press inside a box | Select | Select it and start `move`. Snapshot taken. |
| Left press on empty canvas | Select | Deselect, start a rubber band → new box on release. |
| Left press **anywhere** | Draw Box / Draw Rotated Box | **Always** start a new rubber band, even inside an existing box. |
| Left click on a box | Delete Box | Remove it (one undo step). |
| Left click on a **candidate** | Select | **Accept it** and select it. Status `Accepted: track {n}.` |
| Left drag | Pan | Move the view. |
| Left release, box < 5 px in either axis | Draw* | Discarded with `The box was too small. Drag an area of at least 5 x 5 pixels.` — never silent. |
| Left release, drag ended off-canvas | any | Still commits; coordinates clamped to the frame. Never cancelled. |
| Middle drag | any | Pan. |
| Wheel | any | Zoom at the cursor, clamped `[0.2, 20.0]`. |
| Right click on a box | any | Select it and open the context menu. In a draw tool, right-click **cancels** the in-progress rubber band instead. |
| Right click on a candidate | any | Delete it immediately. `Discarded 1 candidate.` |
| Mouse move | draw / obb / delete | Update crosshair; update `_hover_obj` (throttled to every 2nd frame) so the hovered box draws at `thickness + 1` with a `(255,255,255,60)` fill. |

Geometry constants, inherited and unchanged:

| Constant | Value | Unit | Meaning |
|---|---|---|---|
| `_MIN_BOX_PX` | 5 | image px | Minimum side on creation and resize |
| `_HANDLE_R` | 7 | canvas px | Handle radius; hit tolerance is `_HANDLE_R + 3` = 10 |
| `_ROT_HANDLE_OFFSET` | 26 | canvas px | Rotation handle distance beyond the box edge |
| zoom clamp | `[0.2, 20.0]` | × | Multiplied onto the fit scale |
| pan clamp | 64 | canvas px | Minimum amount of image that must stay on screen |
| undo depth | 50 | steps | Per frame |

**Zoom and pan touch exactly three methods.** `_recompute_scale` stores `_fit_scale` and composes
`_scale = _fit_scale * _zoom`, with `_offset = centering + _pan`. `_i2c` and `_c2i` are **not
touched** — every hit test, every handle test, the rotation-handle position and all drawing
already route through them, so zoom and pan are correct everywhere by construction. `_HANDLE_R`
stays in canvas px, so handles keep a constant screen size while zoomed; `_MIN_BOX_PX` stays in
image px. Wheel zoom anchors the image point under the cursor:

```
cx, cy = self._get_canvas_mouse()
ix, iy = self._c2i(cx, cy)          # anchor under the cursor
self._zoom = clamp(self._zoom * (1.25 if up else 0.8), 0.2, 20.0)
self._recompute_scale()
nx, ny = self._i2c(ix, iy)
self._pan_x += cx - nx;  self._pan_y += cy - ny
self._clamp_pan()
```

### §0.3.6 Undo

YOMU's rule, the version that does not produce phantom dirty state (P7):

* `_snapshot()` deep-copies the current frame's object list.
* `_on_mouse_press` stores it in `self._pending_snap`.
* `_on_mouse_release` pushes it **only if `self._moved`**, otherwise discards it.
* Every discrete mutation pushes first: `delete_selected`, `_create_object`, `set_action`,
  `propagate_from_prev`, accept, discard, `rotate_selected`.
* `undo()` restores, **clears `_selected`** (indices no longer mean anything), and reports
  `Undone. {n} steps left.` / `Nothing to undo.`
* **No redo.** It doubles the state machine for a case neither reference implements.

### §0.3.7 The sidebar, in order

| # | Header | Contents |
|---|---|---|
| — | *(brand)* | `YOAKE Annotator` / `Draw, track, and label behavior` |
| 1 | `Quick Start` | `1. Open a video (Ctrl+O).` / `2. Press W, then drag on the image to draw a box.` / `3. Press T to track it forward, then Y to accept the result.` / `4. A and D move between frames. Ctrl+S saves.` then `Tip: press F1 for the full shortcut list.` |
| 2 | `Active Track` | `Track ID:` `active_tid` · `New Track (Ctrl+N)` · `active_track_color_label` swatch (renders now that a font is loaded) |
| 3 | **`Auto-Annotate`** *(new)* | §0.4.1 |
| 4 | `Objects in current frame` | `obj_listbox`; rows prefixed with a state marker — `●` accepted, `?` candidate, `~` interpolated — and suffixed with the score when present. Selection stores **parallel object ids**; it no longer matches the rendered label string. |
| 5 | `Selected Object` | `Track ID:` · `Action:` · `Occluded` · `Is Crowd` · **new** read-only `Source: human / auto (0.82) / tracked` |
| 6 | `Actions (0-9)` | Renamed from `Quick Actions (0-4)`. Buttons for every defined action; keys `0`–`9` now cover ten, fixing the hardcoded `range(5)` under a header that already claimed `0-4`. |
| 7 | `Tools` | Hint text plus the mirrored `Keep drawing after each box`. The two tool buttons moved to the toolbar. |
| 8 | `Auto-Track` | `Enable auto-track mode (Ctrl+T)` · tracker backend line · `Reset all trackers` · `Reset selected tracker`. Help text gets `wrap=_SIDEBAR_W-20` (today it is unwrapped and clipped at 280 px). |
| 9 | `Labels` *(closed)* | Classes / Actions lists plus add buttons |
| 10 | `Frame Info` | Unchanged plus a `Zoom` line |

### §0.3.8 The typing guard — the worst live bug in the annotator

Today every key handler is global with no focus guard. Typing a track ID into `sel_tid` fires
`A` (auto-track), `N`/`P` (frame jumps), `R` (OBB mode) and **`Delete` (deletes the selected
box)**.

The contract:

```python
EDITING_TYPES = ("mvInputText", "mvInputInt", "mvInputFloat",
                 "mvInputIntMulti", "mvInputFloatMulti",
                 "mvCombo", "mvSliderInt", "mvSliderFloat")
ALWAYS_ALLOWED = ("file.save",)      # Ctrl+S works even while typing (YOMU's rule)

def _typing(self) -> bool:
    f = dpg.get_focused_item()
    if not f:
        return False
    try:
        t = dpg.get_item_type(f)
    except Exception:
        return False
    return any(t.endswith(k) for k in shortcuts.EDITING_TYPES) and dpg.is_item_active(f)
```

**`is_item_active`, not `is_item_focused`.** The first input in a window auto-focuses, so focus
alone is sticky and useless — an `is_item_focused` guard would suppress every shortcut in the
application from the moment the window opens.

Every callback is wrapped as *typing-guard → modifier check → action*:

```python
def _wrap(self, b):
    def cb(sender, app_data):
        if b.action not in shortcuts.ALWAYS_ALLOWED and self._typing():
            return
        if b.mods and not shortcuts.mods_held(dpg, b):
            return
        if not b.mods and dpg.is_key_down(dpg.mvKey_ModCtrl):
            return              # do not fire bare W while Ctrl is held
        self._ACTIONS[b.action]()
    return cb
```

`self._ACTIONS` is a `dict[str, Callable]`, and a build-time assertion requires every binding in
scope to have an entry — so an unimplemented action is a startup error, not a dead key.

---

## §0.3.9 Track identity is a project-level question

Not every experiment follows individuals. For one that does not, a track id is a field the
Inspector keeps asking a human to fill in, a set of checks that can never fire, and a column
in `annotations.json` asserting an identity nobody established.

`ProjectSession.settings["use_track_ids"]` answers it once, for the project. It is read
through `AnnotationApp.use_track_ids` and written only through `set_use_track_ids`.

| Rule | Why |
|---|---|
| Defaults to `True`, and an **absent** key reads as `True` | A project written before the setting existed must behave exactly as it did. Reading absent as `False` would silently disable checks the user relies on. |
| Off means new boxes get `track_id = -1` | That is the file format's own "no identity asserted". No new field, no new reader, no schema change. |
| Off never rewrites ids already on disk | A checkbox that destroys work which switching back on cannot restore is not a setting, it is a trap. The switch changes what happens *next*. |
| Off drops the track-dependent detectors via `quality.detector_keys()` | They are wired to `analyze`'s existing `enabled` argument — not to a new branch inside each detector. `TRACK_DEPENDENT` is the list, and a test asserts every name in it is a real detector. |
| Off refuses `N`, `T` and `Follow boxes to the next frame` out loud, naming `Project > Tracking...` | §35: a gated action states cause, fix and escape hatch. Every tracker is keyed by track id, so with one id for everything they would overwrite each other — the feature is refused, not silently broken. |
| Off colours boxes by **class**, not by track | Every box would otherwise share the untracked colour and a crowded frame would be unreadable. The class is the identity such a project does assert. |
| `Continue this animal` falls back to `Copy to next frame` | Fast tracking and accurate propagation both carry one animal's identity forward. Copying the box is the part that still means something. |
| The Settings dialog is the only writer of `use_track_ids_cb` and the controls it gates | The annotator calls `SettingsDialog.set_use_track_ids`; it does not reach into the widget. Two writers on one widget is how a checkbox and the thing it controls end up disagreeing. |

The analysis page (step 3) carries the matching switch, `Use track identity in this analysis`.
It is a statement about **how the run is read**, not about what the model did: the temporal
heads are aligned by track (`model.track_aligned_temporal`), so turning tracking off inside
inference would be a different run, not a display setting. It hides the id-dependent overlays,
collapses the timeline to one lane, and passes `inference.show_id=False` /
`inference.show_trajectory=False` to `yoake predict`. `predictions.json` still carries
`track_ids`. Loading a file with no `track_id >= 0` in it switches the box off by itself and
says so in the run log — offering a view of data that is not there is worse than not offering it.

## §0.3.10 A project is a folder, and the folder can be moved

A project used to be `annotations.json` plus an *optional* autosave directory, and everything
else a study produced went somewhere unrelated: runs under the source checkout's `runs/`,
figures under `outputs/`, caches wherever the caller pointed them. Three of those four are not
paths a user chose; they are paths the code defaulted to. Archiving a study meant remembering
all four, and one of them was a git checkout.

`htrtdetr.annotation.layout.ProjectLayout` owns one shape, derived from one root:

```
<project>/  project.yoake.json  annotations.json  sampling_log.json  README.txt
            videos/  datasets/  runs/  analysis/  .yoake/  .yoake_cache/
```

| Rule | Why |
|---|---|
| The shape is fixed, not configurable | A path the user supplies is a path they can supply *differently* next session. The result is a study whose halves no longer know about each other. |
| The folder is created when the project is **named**, not at the first save | Anything run in between — a batch detection, a training run — has to have somewhere to write, and "somewhere" is the bug. |
| The folder field on the setup screen is pre-filled (`~/YOAKE/<name>`) and follows the name until edited | Optional was the default, and optional is how output scatters. Still overridable: clearing it restores the old temporary-recovery behaviour. |
| Creating a project on top of an existing one **opens it instead** | The alternative is overwriting a study with an empty one, from a screen whose button says "Create". |
| Every page's path fields come from one table (`TrainingPanel._defaults`) | Two panels that both want "the folder for stage 2" must not be able to disagree about its name. |
| Switching projects rewrites only fields still holding a suggestion | A field the user typed into is an instruction, not a default. |
| Export writes a manifest but **never** relocates the live session | The default destination is inside the project (`datasets/frames`), so adopting the export as the new home would move the study one level into itself. |
| `annotations.json` is unchanged, and `schema_version` stays 1 | The new manifest keys are optional; a bump would make an older YOAKE refuse a project it can in fact read. |

**Portability.** Every stored path is relative to the root, and a capture *outside* the folder
additionally records its absolute path. `layout.resolve_media` tries relative, then absolute,
then a name search under the project and under folders the user has located files in before —
and the order is the design. Relative wins first because it is the only answer that is right
when the whole tree was copied: a stale absolute path can still resolve on the old machine and
quietly open **the wrong file**. When the name search is what answered, the status line says
so, because that is the one failure this fallback can cause. A relink remembers the folder it
was given (`settings.search_roots`), so a restored backup is answered once rather than forty
times. `Project > Copy Videos into Project Folder` copies captures into `videos/`, which is
what makes a folder survive being handed to someone who never had the capture drive.

## §0.3.11 Every dialog: scrolling body, pinned footer

A dialog is created with a fixed `width` and `height`. Its content grows — one more threshold,
one more phase, one more option — and the button the user opened it for ends up below the
bottom edge. The code still reads fine and it still works at the author's window size. This
failure mode has hit Settings (Done, under the Checks tab), Smart Annotate (both `Analyze
video` and `Start annotating`), the project batch form (Run, and the results box entirely) and
the setup screen (Create Project).

The contract, in `widgets`:

| Helper | Contract |
|---|---|
| `viewport_size()` | Client size, or `1280x820` when there is no viewport. `get_viewport_client_*` raises a **bare** `Exception` before `create_viewport`, not the `SystemError` the rest of dearpygui raises. |
| `fit_dialog(w, h)` | `(w, h, x, y)` clamped inside the viewport with a margin. The requested size is a preference, never a promise. |
| `place_dialog(tag, w, h)` | Applies it and registers the dialog. Called from the dialog's own `show()`. |
| `refit_dialogs()` | Re-clamps every registered dialog. Called from both viewport-resize callbacks. Size is re-applied; position is only pulled back inside, so a dialog the user dragged stays where they put it. |
| `dialog_body(tag, footer=…)` / `dialog_footer()` | The split. Everything that can grow goes in the body, which scrolls; the buttons go after it and do not. |

`tests/test_gui_dialogs.py` runs at **900x600** — the annotator's own declared minimum — opens
each dialog, and measures from dearpygui that the window is inside the client area and its
primary button is inside the window. Measured, not derived from the constants, so a layout
that only works on one screen does not pass.

## §0.4 The auto-annotation loop

**configure → run → review → accept → export → retrain.** The review stage is the product; the
detection is a means to it.

### §0.4.1 The `Auto-Annotate` sidebar panel

```
Auto-Annotate
  ● Model detection: unavailable
    PyTorch is not installed in this Python environment, so the detector
    cannot run here. Frame-to-frame tracking below still works.

  [ Auto-Annotate... (Ctrl+R) ]            <- locked when torch is absent
  [ Import Predictions as Candidates... ]  <- ALWAYS available
  ------------------------------------------------------------------
  Track forward from this frame
  Frames to track: [ 30 ]    Tracker: MIL
  [ Track forward (T) ]                    <- ALWAYS available
  ------------------------------------------------------------------
  3 candidates in this frame - click one to accept it, right-click to delete.
  [ Accept all (Y) ]   [ Discard all (N) ]
```

### §0.4.2 Configure — the Auto-Annotate dialog

Modal, 560 × 420, tag `autolabel_dlg`.

| Label | Control | Default |
|---|---|---|
| `Detector checkpoint (.pth or .pt)` | `path_field`, required | last used (`recents` key `checkpoint`) |
| `Config (.yaml, optional)` | `path_field`, collapsed | empty |
| `Score threshold` | slider 0.05–0.95 step 0.05 | `0.30` |
| `Apply to` | radio | `This frame` / `This frame and the next N` / `Whole video` |
| `Frames (N)` | input_int, shown for the middle option | `60` |
| `Skip frames that already have boxes I drew` | checkbox | **on** |
| *(note, `TEXT_DIM`)* | text | `Candidates are not saved as labels. Review them, then accept the ones you want.` |

Config tooltip: `Only needed when the checkpoint was saved without its training configuration. If
a run fails with a message about checkpoint metadata, point this at the config used for
training.`

Buttons: `Run` (accent) / `Cancel`.

### §0.4.3 Run — the five steps

1. **Gate.** `auto_label.probe(repo_root)` at build time and again on click. See §0.4.6.
2. **Frame range.** `yoake predict` has **no start-frame option**: `max_frames` counts from 0,
   and an image directory is re-indexed from 0. So for `This frame` and `This frame and the next
   N`, `auto_label.export_plan()` gives the frame list, and the annotator writes `%06d.jpg`
   (q95) into `<project>/.cache/autolabel/<job_id>/` using **sequential `grab()` plus selective
   `retrieve()`** (P15), never `cap.set(POS_FRAMES)`. It writes `index_map.json` alongside.
   **Without that map every candidate lands on the wrong frame.**
3. **Spawn.**
   `Popen([sys.executable, *auto_label.predict_argv(...)], cwd=repo_root,
   env=auto_label.child_env(repo_root), stdin=subprocess.DEVNULL, stdout=PIPE, stderr=STDOUT,
   text=True, encoding="utf-8", errors="replace",
   creationflags=CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP)`.
   A reader thread pushes lines into a `queue.Queue`; a self-re-arming `dpg.set_frame_callback`
   drains it. **No `dpg` call ever happens off the main thread** (P9).
4. **Progress.** Indeterminate bar, `Running auto-annotation… 0:42 elapsed`, the last log line,
   and a `Stop` button (P20). `Stop` sets `self._stopped = True` **before** `terminate()`, then
   escalates to `taskkill /T /F /PID` after 3 s so DataLoader/CUDA children die too. A stop is
   reported as a stop (P6).
5. **Ingest.** `load_candidates()` → `remap_frame_indices()` → `merge_candidates()` →
   `AnnObject(accepted=False, source="auto", score=…, keyframe=False)`.

The exact argv (`auto_label.predict_argv`, after `sys.executable`):

```
-u -m htrtdetr.cli predict
  source=<video or exported-frame directory>
  weights=<checkpoint>
  output_dir=<folder>
  score_threshold=0.30
  window_size=16
  [max_frames=<n>]          # only when not None
  [config=<path>]           # only when non-empty
  inference.show_bbox=false
  inference.show_id=false
  inference.show_action=false
  inference.show_confidence=false
```

`-u` plus `PYTHONUNBUFFERED=1` gives live output; without them CPython block-buffers a piped
stdout and the run looks frozen for 8 KB at a time. The four `show_*=false` overrides blank the
overlay; the CLI writes `output.mp4` regardless and the caller may delete it. The artefact to
read back is always `<output_dir>/predictions.json`.

The child environment (`auto_label.child_env`):

| Variable | Value | Why |
|---|---|---|
| `PYTHONPATH` | `<repo>/src` prepended, preserving any existing value | V1 — without it every child dies with `ModuleNotFoundError` |
| `PYTHONIOENCODING` | `utf-8` | The child writes cp932 here; the pipe is read as utf-8 |
| `PYTHONUNBUFFERED` | `1` | Live output |

### §0.4.4 The merge rule

`MergePolicy(skip_frames_with_accepted=True, iou_suppress=0.60, replace_pending=True,
min_score=0.0)`. Applied in this order, per frame:

1. If the frame holds any object where `is_accepted(o) and is_human(o)` **and**
   `skip_frames_with_accepted` → skip the whole frame; `skipped_protected += 1` (P3).
2. If `replace_pending` → drop this frame's existing `not is_accepted(o)` objects. **A re-run
   supersedes, never stacks.**
3. For each incoming candidate: drop if `score < min_score` (`below_threshold += 1`); drop if the
   maximum IoU against any **accepted** box in that frame is `>= iou_suppress`
   (`suppressed_iou += 1`).
4. Otherwise `make_obj(cand)` and append.

> `iou_suppress = 0.60` **is a judgement, not a measurement** (P12). It is a policy field and its
> docstring says so. Record the dataset and the false-merge rate here once there is a real video
> to measure against. For reference, YOMU has **no IoU suppression anywhere** — a grep for
> `iou|nms` across its whole backend returns zero hits; it avoids duplicates with cheaper,
> more legible mechanisms instead (candidate-vs-accepted counting, one-point-per-class slots,
> frame-identity checks, and temporal spacing with progressive gap halving).

### §0.4.5 Review — states, rendering, and the loop

The four states of a box:

```
                   accept (click / Y)
   auto/pending  ───────────────────────►  auto/accepted ──► saved with "source":"auto"
        │                                        ▲
        │ discard (right-click / N)               │ manual edit promotes to keyframe
        ▼                                        │
     deleted                        track/pending ──accept──► track/accepted
                                                              saved "source":"track"

   human/accepted ──────────────────────────────────────────► saved, no extra keys
```

Rendering table:

| State | Stroke | Colour | Label | Chip |
|---|---|---|---|---|
| Human, accepted | solid 2 px | track colour | `T3 \| walking` | — |
| Machine, **pending** | **dashed `[7,5]`** 2 px, fill α 28 | `PENDING` | `T? \| ? 0.82` | — |
| Machine, accepted | solid 2 px | track colour | `T3 \| walking` | 3 px `ACCEPTED` left edge on the chip |
| Interpolated / tracked, accepted | dashed `[4,4]` 1.5 px | `INTERPOLATED` | dim | — |
| Selected (any) | +1 px | — | — | 7×7 px handles filled `DANGER` with a 1 px white rim |

The loop:

| Action | Key | Effect |
|---|---|---|
| Next frame with candidates | `Ctrl+Right` | **This is what makes 400 candidates reviewable.** Without it the user arrows through hundreds of empty frames hunting for amber. |
| Previous frame with candidates | `Ctrl+Left` | Same, backwards |
| Cycle boxes in this frame | `Q` | Candidates first |
| Accept one | left-click it | Accept and select. `Accepted: track {n}.` |
| Discard one | right-click it | `Discarded 1 candidate.` |
| Accept all in frame | `Y` | |
| Discard all in frame | `N` | Accepted boxes untouched |
| Undo | `Ctrl+Z` | 50 deep |

### §0.4.6 What the UI does with no torch installed

This is the default state on this machine, so it is not an edge case — it is the primary path.

**Probe with `importlib.util.find_spec` only. Never `import torch` to test**: it costs seconds and
pulls CUDA DLLs. `tests/test_gui_auto_label.py` asserts `"torch" not in sys.modules` after
`probe()` returns.

| Capability | Present | Absent |
|---|---|---|
| `torch` | `Auto-Annotate…` enabled | `widgets.locked("tb_autolabel", cap.torch_reason)`; `Start Training` / `Run Evaluation` / `Run Analysis` / `Run Inference` locked the same way |
| `htrtdetr` importable | nothing shown | If `<root>/src/htrtdetr` exists → **fixed silently** by `child_env()`; `package_reason` is `""` and the user must never see `ModuleNotFoundError`. Only if neither exists: `YOAKE is not installed in this Python environment. Run: pip install -e .` |
| OpenCV tracker | `Tracker: MIL. Install opencv-contrib-python for CSRT, which is more accurate.` | `No OpenCV tracker is available. Falling back to optical flow, which is less accurate on fast motion.` |

`cap.torch_reason`, the exact string shown in the locked tooltip:

```
Automatic detection needs PyTorch, which is not installed in this Python environment.
Install it with:  pip install torch torchvision
You can also run the detector on another machine and bring the result back with
"Import Predictions as Candidates...":
python -m htrtdetr.cli predict source=<video> weights=<checkpoint> output_dir=<folder> score_threshold=0.30
```

**What still works with no torch — say this explicitly, do not leave the user guessing:**

* `Track forward (T)` — propagate a box you already drew across later frames with OpenCV MIL (or
  Lucas–Kanade). It **cannot find animals on its own**; it extends your work. Precondition: at
  least one accepted box in the current frame, otherwise
  `empty_state("Draw at least one box first. Tracking extends boxes you have already drawn - it
  cannot find animals on its own.")`
* `Import Predictions as Candidates…` — load a `predictions.json` produced on a GPU machine.
* Everything in `3. Analyze` except `Run Inference`: load, scrub, inspect, and the four
  `yoake analyze` modes (`timeline`, `distribution`, `id_switches`, `confidence`) all run
  torch-free — verified end to end this run.

**Hard invariant:** importing `htrtdetr.gui.*` must never import torch at module scope. The
green-line check is:

```
python -c "import htrtdetr.gui, htrtdetr.gui.theme, htrtdetr.gui.shortcuts, htrtdetr.gui.auto_label, sys; assert 'torch' not in sys.modules"
```

### §0.4.7 Torch-free tracking — `Track forward (T)`

Backend probe chain, in order, with the measured outcome on this machine:

| Probe | Exists here? | Result |
|---|---|---|
| `cv2.TrackerCSRT_create` | no | — |
| `cv2.legacy.TrackerCSRT_create` | no (`cv2.legacy` absent) | — |
| `cv2.TrackerMIL_create` | **yes** | `"mil"` ← selected |
| none of the above | — | `"lk"`, pure Lucas–Kanade |

GOTURN / Nano / Vit / DaSiamRPN are **deliberately excluded**: they require downloaded ONNX
weights and fail at init with an opaque error.

Runs on a worker thread touching only plain Python lists (MIL measures ~47 ms/frame, so 30 frames
× 4 tracks would otherwise freeze the UI for ~6 s), applying results on the main thread through
the same frame-callback pump, with `Tracking… {i}/{n}` and a working `Stop`.

Produces `accepted=False, source="track", keyframe=False` objects **only in frames where that
`track_id` has no object** — never overwriting human work (P3). They enter the **same amber
review queue**, so the torch-free fallback is one branch of one loop, not a second feature.

Loss message: `Tracking lost for track {n} at frame {f} - {k} of {m} tracks are still following.`

### §0.4.8 Export and retrain

| Step | What happens |
|---|---|
| Export | `Ctrl+S` writes `<project_dir>/annotations.json` with **only accepted objects**; frames that become empty are pruned. `Save and Export Frames…` additionally writes `images/<video_id>/<idx:06d}.jpg` (q95) matching the `image_path` the JSON records. |
| Warn | Saving with candidates present warns **once**: `{n} candidates are not saved. Press Y to accept them first, or N to discard them.` |
| Hand off | `Saved {a} boxes across {f} frames. {p} candidates are still pending and were not saved as labels.` |
| Retrain | **Inside the annotator**, `Train` (§0.4.9). Step 2's page is the production run over finished labels, not the loop. |
| Close the loop | **Inside the annotator**, `Analyze`. Step 3's page is the workstation for reviewing a finished model, reached through `Send Predictions to Annotator`. |
| End the loop | `Project > Check every frame (final pass)...`, then `File > Export the verified dataset...` (§0.4.11). A different artefact from the `Export` row above: that one writes the frames the human labelled, this one writes every frame of every video they signed off. |

### §0.4.9 The round — training inside the labelling window

A *round* is the loop's second half: train a detector on the labels that exist
right now, run it back over the videos, and let what it was unsure about decide
what to label next. It runs in the annotator because a loop whose third step
opens a second application is a loop taken once.

**The dialog has no form.** `gui/train_round.py` shows a plan paragraph, a
four-phase list and `Start`; `Adjust this round...` is collapsed and holds two
answers a biologist actually has — how long they will wait, and whether to use
the GPU. Everything else is derived in `annotation/round.py`, which is pure and
tested, because each derivation is a trap whose failure mode is **silent**:

| Derived | Why it cannot be left to the default |
|---|---|
| `model.detector.head.num_classes` | Defaults to 1. The loss uses `num_classes` as the **background** index, so class 1 trains as "no object" — no error, no warning, a checkpoint at the end. Nothing else in the GUI sets it. |
| `data.batch_size`, clamped to `n_train` | The train loader is `drop_last=True`: a batch bigger than the training set is **zero** optimizer steps, `train_loss` 0.0, and a checkpoint anyway. |
| `epochs`, from a **step** budget | 400 steps cold, 150 warm, times the effort multiplier. Not a minute budget: the steps a minute buys depend on the label count, so one "12 minutes" meant 400 steps on a small project and 3,600 on a grown one. Steps make a round cost the same as the study grows — measured 4.3 min cold, 1.7 min warm. |
| `data.num_workers = 0` | Measured: 2 and 4 workers were no faster at this dataset size, and on Windows `terminate()` does not reap them. |
| `train.val_interval = epochs // 4` | Measured: `last.pth` is 348 MB and `best.pth` 249 MB, written every epoch — ~35 GB per 100-epoch round. The epoch count is rounded to a multiple of the interval so the **last epoch always validates**, because `best.pth` is only written on a validating epoch that improved. |
| `init_weights` on a warm round | Rounds 2+ carry on from the last checkpoint (measured: warm starts at val_loss 13.56 against 14.86 cold). **Not** `train.resume`, which also restores the optimizer and scheduler — so the learning rate resumes at its decayed floor — and the previous run's `best_metric`. Forced cold when the class count changed (the head changes shape) and every 5th round. |

**`SEC_PER_STEP` is measured, not guessed.** It was 0.20 s/step against a measured
0.6–0.8, so a round advertised as 12 minutes took closer to 45. The quote is derived
from the step budget at the measured rate, so the number on screen and the work that
runs are one decision read two ways.

**Phases and where they run:** *Saving your labels* (main thread) → *Collecting
the frames* (worker; `gui/round_export.py`) → *Training the detector* (child
process, log to a **file** not a pipe, `stdin=DEVNULL`) → *Checking what it
learned* (main thread). Progress is read from `results.csv` by
`tick_train_round()` on the render loop, never by `monitor.TrainingMonitor`,
which calls back from a daemon thread.

**Three rules the round cannot break.**

1. *It never writes the user's label file.* `export_project` repoints every
   `image_path` on the live session and saves that into the project's own
   `annotations.json`; the round builds its dataset dict separately and leaves
   the project as it found it.
2. *It verifies its own images before spawning.* The training dataset opens
   `<root>/<image_path>` with `strict=False` and substitutes a **black frame**
   on failure, so a wrong root produces a complete run with a falling loss and
   a checkpoint trained on nothing. `round_export.missing_images` is the only
   thing that will ever report it. The split files therefore live **beside**
   `images/`, because the root is the annotation file's own directory.
3. *It refuses to adopt a checkpoint that does not exist or scored worse.*
   `weights/best.pth` is only written when validation improves, so exit 0 with
   no checkpoint is a real state. Adopting a worse detector makes the next
   pass worse than having none, because the frames it picks are the ones it
   was unsure about. The holdout is frozen in `rounds.json` and grown, never
   re-rolled, so "better than last round" compares one ruler.

Rounds write to `datasets/rounds/` and `runs/rounds/`. Never `runs/train/`:
`cli.find_checkpoint` globs `runs/train/*/stage{N}/weights/best.pth` newest
first and uses it to initialise any stage>1 run, so a round would silently
become what a production stage-2 warm-starts from.

### §0.4.10a The round is a work list, not an analysis

Three changes, and the loop is only usable ten times over with all three.

**A round brings back the frames it chose, and nothing else.** The sampler
spends its budget project-wide, so no video's picks are known until every video
has been scanned — which is why `BatchRunner.run` emits a full scan's worth of
candidates on the `video` event and the trimmed set on `finished`, and why the
annotator merges on `finished`. Without `batch.keep_chosen` a 100-frame round
over twenty hour-long videos merges roughly 72,000 candidates into a Review
Inbox that was asked for a hundred frames: the timeline fills end to end, the
round's own progress cannot be read off anything, and nothing ever removes a
candidate once merged. **On cancel, nothing is merged** — a stopped run never
reached `sample_project`, so nothing was trimmed. What it keeps is the cached
scan, and the stopped message says so.

**The scan stride follows the size of the scope.** `sampling.scan_stride` is
derived from the scope's frame count and deliberately *not* from the budget:
frame features are cached under the stride that produced them, so a stride that
moved when the user changed 100 samples to 200 would re-walk every video for an
identical selection. A pool of `SCAN_POOL = 20,000` places is already a hundred
candidates per pick; at the old fixed 15 a 20x1h project scanned 144,000
positions to choose 100, and paid for it again on every round. Floored at 15 so
a single video behaves exactly as before, and capped at two seconds of video —
past about stride 60 the grab-forward reader loses to a seek, so a coarser scan
costs the same wall clock and returns a worse pool.

**A chosen frame can be checked and empty.** `K` writes `annotations[frame] =
[]`, and the round's counter asks whether a chosen frame is *present* in
`annotations`, not whether it has boxes. Without it the counter stalls below its
target on exactly the frames the sampler works hardest to find, and the user
cannot tell a finished round from a stalled one. It is also a training sample:
a detector never shown an empty arena learns there is always a fly in one.

### §0.4.11 The final pass — every frame, and the dataset that comes out

The loop's end. The rounds produced a detector; this produces the deliverable.

**The scan.** `Project > Check every frame (final pass)...` runs `yoake predict`
over every frame of every video — dense, `detect_only`, batch 16, ~2 hours for
twenty hour-long videos at the measured 300 fps end to end. Dense is right here
and nowhere else in the annotator: a round scans one frame in sixty because the
sampler can only pick a scanned frame, and here every frame is the product. At
stride 1 no JPEGs are written at all — `predict_video` hands the video path
straight to the child — so the 86 GB image tree the strided path needs does not
exist.

**Resumable at video granularity, and not finer.** A video is scanned or it is
not; re-running skips the ones that are, by checkpoint fingerprint. Not per
chunk: `connect_detections` assigns track ids by linking a video's detections to
each other, so a video scanned in twelve pieces would restart its ids twelve
times and an animal would change identity every five minutes of the exported
dataset.

**The ledger, and what it is not.** Boxes go to `datasets/final/<video>.jsonl`,
one JSON line per frame that has any, with a `.idx` every 1,000 frames so a
window is a seek rather than a scan. They do **not** go to `annotations.json`
and they do **not** go to `video.review`. An hour of video with two animals is
216,000 boxes; the manifest is rewritten whole on every autosave and the review
session is deep-copied on every undo, so either route turns a keystroke into a
second of work and a save into a minute of it.

**What the human actually checks.** Not 2.16M frames — that is 600 hours at a
second each, and the absence of that number is what made the first version of
this design wrong. They are taken to every span the checks flagged **plus forty
evenly-spaced spot checks inside the spans that looked clean**. The spot checks
are not optional: a model that is stably and confidently wrong — an animal it
never detects — produces no flagged span at all, so without them the pass
verifies only the places the model already doubted itself. ~134 stops per video,
~25 minutes a video, ~8 hours for twenty.

**Sign-off is per video, and it ships the video whole.** `finalpass.can_sign_off`
refuses until every stop has been visited. Signing off is a statement about the
recording, so the export contains all of it — and the difference between "a
human looked at this stretch" and "a human vouched for the video this is in"
survives as `provenance.human_review`: `drawn`, `accepted`, `signed_off` (inside
a checked span) and `sampled` (inside a signed-off video, outside the spans).
Flattening those four into "ground truth" would make the file unauditable.

**The export.** `File > Export the verified dataset...` writes
`datasets/release/annotations.json` in this repo's own format — labels and a
`video_path` per video, no images. A video that is not signed off is excluded
**and named**, in the dialog and in `summary_sentence`, because an export
covering 17 of 20 videos is indistinguishable on disk from one covering 20, and
the difference is three hours of recordings missing from the paper.

### §0.4.10 Sparse detection — the other half of the cost

`BatchOptions.detect_stride` runs the detector on every Nth frame. It is set to
the scan stride **exactly when the detections are for choosing frames**, and
left at 1 for a pre-label-only pass, where the user means to review the whole
video. The sampler can only ever pick a scanned frame, so detections on the
fourteen frames between two scan positions were computed, cached and never
read: measured at 121 frames/s, twenty one-hour videos cost **4.1 hours dense
and 17 minutes at stride 15**.

Three things move together, and the feature is wrong without all three:

1. **The frames are exported and the indices are mapped back.** `predict`
   re-indexes an image directory from 0 and has no start-frame option, so
   `write_scan_frames` writes `%06d.jpg` locally and
   `auto_label.remap_frame_indices` puts the candidates back on real frame
   numbers. An empty map is the identity, so the dense path pays nothing.
2. **Track linking is off.** `connect_detections` means "these two are the same
   animal a frame apart"; fifteen frames apart it invents identities.
3. **The checks are told which frames were examined.** `quality.analyze` has an
   opinion about every frame it is given, so the whole video's range after a
   strided pass declares fourteen frames in fifteen to be places the detector
   found nothing.

The export walks the video with sequential `grab()` and reads the target with
`read()`. Never `grab()` through the target and then `retrieve()`: retrieve
returns the frame the *last* grab consumed, so every exported frame is one
early — and it agrees at frame 0, where anyone would look first. The first
version of `write_scan_frames` had that bug and the index test did not catch
it, because the frame numbers came from the index map and were right; only the
pixels were wrong. `tests/test_sparse_detection.py` encodes each frame's index
in its own pixels for that reason.

---

# Part II — The contract

## §1 Design tokens

One module, `gui/theme.py`, owns every colour in the suite. **After it lands, no other file may
contain a colour literal.** Contrast ratios below were computed with the WCAG relative-luminance
formula and re-measured during this run; `theme.verify_tokens()` recomputes all of them and
returns a list of violation strings (empty list = pass).

### §1.1 Surfaces

| Token | RGB | Use |
|---|---|---|
| `BG` | `(24, 26, 31)` | Viewport and window ground |
| `SURFACE` | `(38, 41, 47)` | Panels, cards, child windows |
| `SURFACE_RAISE` | `(48, 52, 60)` | Toolbar strip, headers, hovered rows, button fill |
| `FIELD` | `(23, 24, 28)` | Input backgrounds, log panes |
| `ROW_HOVER` | `(60, 66, 78)` | List row hover |
| `ROW_ACTIVE` | `(50, 72, 104)` | Selected list row, active tab |

### §1.2 Lines

| Token | RGB | vs `SURFACE` | Use |
|---|---|---|---|
| `BORDER_SUBTLE` | `(74, 80, 92)` | 1.80:1 | Decorative separators **inside a card only** |
| `BORDER_STRONG` | `(112, 119, 133)` | **3.24:1** ✔ WCAG 1.4.11 | **Every interactive control.** Paired with `FrameBorderSize 1`, this is what replaces today's invisible 1.49:1 border and makes buttons findable by their edge rather than their label. |

### §1.3 Text

| Token | RGB | vs `SURFACE` | vs `BG` | Use |
|---|---|---|---|---|
| `TEXT` | `(216, 219, 226)` | 10.52:1 ✔ AAA | 12.56:1 | Body |
| `TEXT_DIM` | `(150, 157, 170)` | 5.34:1 ✔ AA | 6.38:1 | Secondary, help text, captions |
| `TEXT_FAINT` | `(120, 127, 140)` | 3.62:1 ✔ AA-large | — | Paths, disabled labels — **16 px and above only** |
| `ON_ACCENT` | `(16, 20, 28)` | — | — | **Text on any accent-filled control.** 7.02:1 against `ACCENT`. |

### §1.4 Semantic

| Token | RGB | vs `SURFACE` | Meaning |
|---|---|---|---|
| `ACCENT` | `(77, 163, 255)` | 5.55:1 | Selection, active tool, links, checkmarks, sliders |
| `ACCENT_HOVER` | `(104, 180, 255)` | — | Accent hover |
| `ACCENT_ACTIVE` | `(54, 140, 232)` | — | Accent pressed |
| `ACCEPTED` | `(46, 204, 113)` | 6.94:1 | Accepted, confirmed, capability present |
| `PENDING` | `(245, 166, 35)` | 7.19:1 | **Candidate**, unsaved, warning |
| `DANGER` | `(255, 82, 82)` | 4.57:1 | Destructive, missing file, error, selection handles |
| `INTERPOLATED` | `(120, 144, 176)` | 4.45:1 | Machine-filled boxes that are not candidates |
| `TRACK_GREY` | `(150, 150, 150)` | — | `track_id < 0` |

### §1.5 Control fills

| Token | RGB |
|---|---|
| `BTN` | `SURFACE_RAISE` `(48, 52, 60)` |
| `BTN_HOVER` | `(60, 66, 78)` |
| `BTN_ACTIVE` | `(70, 78, 92)` |
| `MODAL_DIM` | `(10, 11, 14, 180)` |

### §1.6 Track palette

`TRACK_PALETTE` — the existing 20-entry set, moved verbatim from `canvas.py:36-42` and
`annotator.py:58-64` (they were already byte-identical copies). Indexed `tid % 20`. It is the one
palette the annotator and the analysis canvas must share.

| # | RGB | # | RGB | # | RGB | # | RGB |
|---|---|---|---|---|---|---|---|
| 0 | `(230, 25, 75)` | 5 | `(145, 30, 180)` | 10 | `(70, 153, 144)` | 15 | `(170, 255, 195)` |
| 1 | `(60, 180, 75)` | 6 | `(66, 212, 244)` | 11 | `(220, 190, 255)` | 16 | `(128, 128, 0)` |
| 2 | `(255, 225, 25)` | 7 | `(240, 50, 230)` | 12 | `(154, 99, 36)` | 17 | `(255, 216, 177)` |
| 3 | `(67, 99, 216)` | 8 | `(191, 239, 69)` | 13 | `(255, 250, 200)` | 18 | `(0, 0, 117)` |
| 4 | `(245, 130, 49)` | 9 | `(250, 190, 212)` | 14 | `(128, 0, 0)` | 19 | `(169, 169, 169)` |

Known limitation, recorded honestly: tracks 3 and 23 get the same hue with no disambiguation.
Acceptable for the expected track counts (2–10 animals); revisit if a project needs more than 20
concurrent tracks.

### §1.7 Spacing scale

Replaces the 21 ad-hoc `add_spacer` calls with six magic values in `hub.py`. **No other vertical
spacing is permitted.**

| Token | px | Use |
|---|---|---|
| `SP_XS` | 2 | Between a label and the control it names |
| `SP_S` | 4 | Within a control group |
| `SP_M` | 8 | Between control groups |
| `SP_L` | 14 | Between sections |
| `SP_XL` | 22 | Between major regions |

`widgets.gap(n)` maps `1..4` onto `SP_S`/`SP_M`/`SP_L`/`SP_XL`.

### §1.8 Style variables

Thirteen style vars, replacing the current five. This is why scrollbars are currently square next
to 7 px-rounded frames.

| Var | Value | Var | Value |
|---|---|---|---|
| `FrameRounding` | 6 | `FramePadding` | (10, 6) |
| `ChildRounding` | 8 | `ItemSpacing` | (8, 7) |
| `WindowRounding` | 8 | `ItemInnerSpacing` | (6, 5) |
| `PopupRounding` | 6 | `WindowPadding` | (12, 10) |
| `TabRounding` | 6 | `ScrollbarSize` | 13 |
| `GrabRounding` | 5 | `ScrollbarRounding` | 6 |
| `GrabMinSize` | 12 | `IndentSpacing` | 18 |
| `FrameBorderSize` | 1 | `CellPadding` | (7, 5) |
| `ChildBorderSize` | 1 | | |

### §1.9 Colour role rules — when to use which token

| Situation | Token | Never |
|---|---|---|
| Text on any accent-filled control | `ON_ACCENT` | **White.** White on `ACCENT` measures **2.63:1** and fails WCAG for both normal and large text. |
| The active tool button | `accent_theme()` fill | A tint of `SURFACE_RAISE` — it is indistinguishable from hover |
| The other four tool buttons | `ghost_theme()` (bordered, transparent) | Plain `add_button` — that is what makes the active state unreadable today |
| Any interactive control's border | `BORDER_STRONG` | `BORDER_SUBTLE` — 1.80:1 is decorative, not perceptible as an edge |
| A separator inside a card | `BORDER_SUBTLE` | `BORDER_STRONG` — too loud for decoration |
| A machine proposal not yet confirmed | `PENDING` **plus a dashed stroke** | Colour alone — dash carries it when hue is spent on track identity (P13) |
| A confirmed / present / capability-OK state | `ACCEPTED` | `ACCENT` — accent means "active/selected", not "good" |
| Destructive action, missing file, error | `DANGER` | `PENDING` — amber means "needs review", not "broken" |
| Machine-filled box that is **not** a candidate (interpolated, tracked-and-accepted) | `INTERPOLATED` | `PENDING` — it is not awaiting a decision |
| A path, a disabled label, a hint | `TEXT_FAINT`, ≥ 16 px | Below 16 px — 3.62:1 only passes as large text |
| A track's identity | `TRACK_PALETTE[tid % 20]` | Any semantic token — hue belongs to identity here |
| `track_id < 0` | `TRACK_GREY` | Palette index 0 — an untracked box must not look like track 0 |

### §1.10 The invariants `verify_tokens()` asserts

```
contrast(TEXT,          SURFACE) >= 7.0     measured 10.52
contrast(TEXT_DIM,      SURFACE) >= 4.5     measured  5.34
contrast(TEXT_FAINT,    SURFACE) >= 3.0     measured  3.62
contrast(BORDER_STRONG, SURFACE) >= 3.0     measured  3.24
contrast(ACCENT,        SURFACE) >= 3.0     measured  5.55
contrast(ON_ACCENT,     ACCENT)  >= 4.5     measured  7.02
contrast(ACCEPTED,      SURFACE) >= 3.0     measured  6.94
contrast(PENDING,       SURFACE) >= 3.0     measured  7.19
contrast(DANGER,        SURFACE) >= 3.0     measured  4.57
contrast((255,255,255), ACCENT)  <  3.0     measured  2.63   <-- white-on-accent MUST fail
```

The last line is the important one: it encodes the prohibition as a test, so the current mistake
cannot come back. Run `python -m htrtdetr.gui.theme --check` before handing off; it prints the
report and exits non-zero on violations.

### §1.11 Theme-object caching

All theme objects are created once and cached at module level — a second `dpg.theme()` per call
leaks. `apply_global_theme()` and `load_fonts()` are idempotent no-ops on a second call. Tokens
live at module scope; `dpg` is touched only inside functions, so `import htrtdetr.gui.theme`
works before `create_context()`.

---

## §2 Typography

**There is no font loaded anywhere in the suite today.** Everything renders in Dear PyGui's
default 13 px bitmap ProggyClean, so all four type levels are the same size and hierarchy is
carried 100 % by hue — which is why the current palette has 37 near-identical greys. Three real
sizes let the palette shrink to the ~20 tokens in §1.

### §2.1 The scale

| Token | px | Face | Registered tag | Use |
|---|---|---|---|---|
| `FONT_H1` | 21 | Segoe UI | `font_h1` | Page title — **one per screen** |
| `FONT_H2` | 18 | Segoe UI | `font_h2` | Section heading |
| `FONT_BODY` | 16 | Segoe UI | `font_ui` | Everything else; bound globally with `dpg.bind_font` |
| `FONT_MONO` | 14 | Consolas | `font_mono` | Log panes, key legends, aligned report blocks |

### §2.2 Loading

`theme.load_fonts()` tries, in order, `C:/Windows/Fonts/segoeui.ttf`, `SegoeUI.ttf`, `arial.ttf`;
mono from `consola.ttf`. It returns `{}` and logs **one** line if no face is found, and the
caller then simply skips `bind_font` — the app still runs.

**Do not call `add_font_range_hint` / `add_font_chars` / `add_font_range` (V3).** All three still
exist as attributes but are deprecated no-ops; character ranges are automatic in dpg 2.x. Loading
the TTF is what makes `●` `■` `°` `—` `×` render, and it is the single largest visual payoff per
line in this document.

### §2.3 The rule

**Hierarchy is carried by size and weight, not by inventing another grey.** If a heading does not
stand out, the fix is `bind_h2`, not a new colour. The suite is allowed exactly three text
colours (`TEXT`, `TEXT_DIM`, `TEXT_FAINT`) plus `ON_ACCENT` and the semantic set; adding a
fourth grey is a design bug.

Non-ASCII glyphs that are already user-visible and depend on a real font being loaded:

| Glyph | Where |
|---|---|
| `●` | `widgets.path_field` status dot; the `Auto-Annotate` capability rows |
| `■` | `active_track_color_label` swatch in the annotator |
| `°` | OBB angle in the context menu and status line |
| `—` | Tooltips, chip sentences, completion messages |
| `×` | Frame size readouts (`{w}×{h}`) |
| `✓` `⚠` | Workflow capability rows |

---

## §3 The locked pattern

The single most transferable habit from YOMU (P4).

### §3.1 Definition

```python
widgets.locked(tag: str, reason: str) -> None
widgets.unlock(tag: str, tooltip_text: str = "") -> None
```

`locked()` does exactly three things:

1. `dpg.configure_item(tag, enabled=False)`
2. Binds the disabled theme — `theme_component(dpg.mvButton, enabled_state=False)` gives fill
   `SURFACE`, text `TEXT_FAINT`, border `BORDER_SUBTLE`
3. Attaches a tooltip whose text is `reason`

**A gated control is never hidden.** The user must learn that the feature exists and why it is
off.

### §3.2 The three-part reason

`reason` must state, in this order:

| Part | Example |
|---|---|
| **The cause** | `Automatic detection needs PyTorch, which is not installed in this Python environment.` |
| **The fix command**, literal and copyable | `Install it with:  pip install torch torchvision` |
| **The escape hatch** — what to do instead, right now | `You can also run the detector on another machine and bring the result back with "Import Predictions as Candidates...": python -m htrtdetr.cli predict source=<video> weights=<checkpoint> output_dir=<folder> score_threshold=0.30` |

A reason with only the cause is a dead end. A reason with only the fix assumes the user can
install things on this machine. The escape hatch is what makes the locked state survivable.

### §3.3 Where it is applied

| Control | Condition | Reason source |
|---|---|---|
| `tb_autolabel` (annotator) | `not cap.torch` | `cap.torch_reason` |
| `Start Training` / `Run Evaluation` / `Run Analysis` (training panel) | `not cap.torch` | `cap.torch_reason` |
| `Run Inference` (analysis panel) | `not cap.torch` | `cap.torch_reason` |
| Anything requiring the package | `not cap.package` **and** `<root>/src/htrtdetr` missing | `cap.package_reason` |

Note the deliberate asymmetry: when `package_via == "src"`, `package_reason` is **empty** and
nothing is locked — `child_env()` fixes it silently. Surfacing a `ModuleNotFoundError` that the
GUI can repair itself is a bug, not honesty.

---

## §4 Copy rules

**Binding on every implementer.** The GUI is already 100 % English — all Japanese in
`annotator.py` and `gui/*.py` is in comments and docstrings, which is permitted. The work here is
therefore **accuracy and usefulness, not translation.** These rules are distilled from YOMU's
shipped `I18N_EN` dictionary (2133 entries, read in full before deletion).

### §4.1 Capitalisation

| Element | Rule | Examples |
|---|---|---|
| Buttons, menu commands | **Sentence case** — first word only | `Accept all candidates`, `Track forward`, `Run auto-annotation`, `Reset all trackers`, `Create splits` |
| Section / panel headings | Sentence case; `&` joins two nouns, `/` joins alternatives | `Auto-Annotate`, `Objects in current frame`, `Range labeling`, `Source / Detect` |
| Page labels | `N. Words` — digit, period, space, sentence case | `0. Workflow`, `1. Label`, `2. Train`, `3. Analyze` |
| Field labels | Sentence case, no trailing colon inside the string | `Score threshold`, `Detector checkpoint`, `Frames to track` |
| Enum options and inline values | **lowercase** — they are values, not commands | `on`, `off`, `none`, `auto`, `all classes`, `human`, `tracked`, `csv`, `json` |
| Job states | Sentence case, one word where possible | `Queued`, `Running`, `Done`, `Failed`, `Stopped` |
| Acronyms and product names | Always uppercase | `YOLO`, `OBB`, `CSRT`, `MIL`, `CSV`, `JSON`, `VLM`, `YOAKE` |
| Emphasis on a dangerous scope | Full caps on the scope word only | `Discard ALL candidates` |

**Never Title Case.** There is not one `Start Training` or `Add To Queue` in 2754 lines of YOMU.

*The one exception:* the five tool names are proper nouns of the toolbar — `Select`, `Draw Box`,
`Draw Rotated Box`, `Pan`, `Delete Box` — and are capitalised consistently wherever they appear,
including inside sentences.

### §4.2 Sentence style

* A short tooltip is a bare verb or noun phrase with **no final period**: `Fit the frame to the
  window`, `Show only frames containing this class`.
* Multi-sentence help ends **every** sentence with a period, including the last.
* Imperative for instructions; declarative for behaviour.
* **State the negative guarantee** on every risky control (P18): `Candidates are never written to
  disk.` · `Nothing is saved until you accept it.` · `Images and labels are kept.` · `Display
  only - saved labels are unaffected.` · `The source run is not modified.`
* **Refuse to let a number look more meaningful than it is** (P12): `The detector's confidence is
  not a calibrated probability.` · `Tracking confidence is the tracker's own estimate, not an
  accuracy.`
* **Name the next step and where it lives**, after an em dash or a comma: `- press Ctrl+Right to
  jump to the first one` · `Create splits in step 1` · `review it in 3. Analyze`.
* **Quote the cost before the user commits** (P17): `(about 10-20 s per frame)`, `This takes a few
  minutes.`
* Punctuation: `—` for a consequence or a next step; `·` for compact separators; `→` for a
  pipeline; `…` (the single character, not three dots) only on in-progress strings.

### §4.3 Errors — five registers, each with a fixed shape

| # | Register | Shape | Examples |
|---|---|---|---|
| 1 | Operation + colon + raw error | verb-first, no apology | `Save error: {e}` · `Load error: {e}` · `Tracking failed: {e}` |
| 2 | Could not / Cannot + what, then the check to run | the system takes the blame | `Cannot open the video. Check its path and format.` |
| 3 | Missing precondition as the imperative that fixes it | never "you must" | `Open a video first.` · `Draw at least one box first.` · `Select a box first.` |
| 4 | Missing dependency + the exact install command | command in backticks | ``Automatic detection needs PyTorch. Install it with: `pip install torch torchvision` `` |
| 5 | Validation as the rule | declarative, present tense, no "error" | `Score threshold must be between 0.05 and 0.95.` · `Track IDs must be whole numbers.` |

Stopping something that is not running is a **statement of fact, not an error**:
`No auto-annotation is running.`

### §4.4 Progress

| Situation | Shape | Example |
|---|---|---|
| In flight | gerund + `…` | `Loading…`, `Saving…`, `Tracking…`, `Running auto-annotation…`, `Stop requested…` |
| With a denominator | `… {a}/{b} <noun>` | `Tracking… 12/30 frames` |
| Terminal states | fixed triple on the subject noun | `X finished` / `X failed` / `X stopped` |
| Completion | counts first, next action after an em dash | `Auto-annotation done: {n} candidates on {f} of {t} frames — press Ctrl+Right to jump to the first one.` |
| Interruption | report what survived | `Auto-annotation stopped. The {n} candidates already produced are kept.` |

**A stop is never an error** (P6): `Stopped by user`, never `Error (code=1)`.

### §4.5 Destructive actions

* The button names the verb **and the exact scope**: `Discard all candidates`, `Delete selected`,
  `Discard ALL labels on this frame`.
* The confirmation is a question ending in `?`, built as *what will happen → what else is
  affected → reversibility →* `Continue?`
* A parenthetical always says what is preserved: `Boxes you drew by hand are kept.` · `Ctrl+Z can
  undo this.`
* Unsaved-work dialogs offer **three named outcomes**, never OK/Cancel: `Save and close` /
  `Discard and close` / `Cancel`, with **Cancel focused on open** so Enter can never destroy
  work, and Esc cancelling.

### §4.6 Units and numbers

| Rule | Form | Example |
|---|---|---|
| Unit | in parentheses, lowercase, at the end of the label | `(px)`, `(s)`, `(°)`, `(%)`, `(frames)` |
| Sentinel values | documented inline with `=` | `0 = all frames`, `empty = latest checkpoint`, `-1 = auto` |
| Defaults | in parentheses at the end of the help | `(default 0.30)` |
| Numeric ranges | en dash | `0.05–0.95` |
| Keyboard ranges | hyphen | `0-9` |
| Counts | `{n} <noun>`, plural-agnostic | `{n} frames`, `{n} candidates`, `{n} tracks` |
| Ratios | `{a} / {b}` with spaces | `Showing {a} / {b} frames` |
| Shortcut on a button | appended in parentheses at the end | `Save (Ctrl+S)`, `Accept all (Y)`, `Fit (F)`, `Track forward (T)` — generated by `shortcuts.with_key()` so it cannot drift |
| Key legends | `Key: action / Key: action` | `Wheel: zoom / middle-drag: pan` |

### §4.7 Empty states

Every blank region gets a sentence that names the next action. The three required ones:

| Region | Text |
|---|---|
| Analysis timeline | `No predictions loaded. Run inference above, or load a predictions.json.` |
| Analysis inspector | `Click a box on the video to see its ID, action, confidence and the evidence behind the decision.` |
| Analysis canvas | `Load a video and a predictions.json to review detections frame by frame.` |

### §4.8 Copy that is currently wrong and must be fixed

| Where | Current | Replacement |
|---|---|---|
| Annotator setup screen | `...choose an optional folder for autosave.` | `...choose a folder. Save (Ctrl+S) then writes annotations.json there without a dialog.` — **there is no autosave** |
| Annotator sidebar header | `Quick Actions (0-4)` | `Actions (0-9)` — more than five actions can already be defined |
| Annotator module docstring | advertises `tools/annotate.py` and a `T` binding that was never registered; omits `R` / `[` / `]` | Regenerate from `shortcuts.help_groups()` |
| Auto-track help | unwrapped, clipped at 280 px | `When on, stepping to the next frame moves every initialized tracker forward. Tracked boxes appear dashed - correct one and it becomes a keyframe again.` with `wrap=` |
| Hub launch status | `The annotator opened in a separate window.` | `The annotator is starting…` → `Annotator running (PID {n}).` or `The annotator could not start: {reason}` |
| Analysis toggle | `Individual OBB` | `Rotated box (OBB)`, plus a real `Axis-aligned box` checkbox |
| Training tip | `Tip: the video labeling workflow is available from the Labeling page in the left navigation.` | `Labeling lives in step 1. Open it with Ctrl+1.` |
| Too-small drag | *(silently discarded)* | `The box was too small. Drag an area of at least 5 x 5 pixels.` |
| Tracker lost | one footer line | `Tracking lost for track {n} at frame {f}. Draw the box again to re-anchor it.` |

### §4.9 Domain glossary

The vocabulary is the product's shared language. Pick one word per concept and never alternate.

| Term | Means | Never call it |
|---|---|---|
| **Box** | One annotated rectangle, axis-aligned or rotated | "annotation", "bbox", "ROI" in user-facing text |
| **Rotated box (OBB)** | An oriented bounding box, `[cx, cy, w, h, theta]`, theta in radians | "angled box"; "OBB" alone in a label |
| **Candidate** | A machine proposal not yet confirmed. The central review primitive. | "suggestion" in one place and "candidate" in another — pick **candidate** |
| **Accept / discard** | The two verbs for resolving a candidate | "approve" / "reject" |
| **Track** | One subject's identity across frames, keyed by `track_id` | "ID" alone, "object" |
| **Action** | The behaviour label on a box (`walking`, `grooming`) | "class" — class is the object category |
| **Class** | The object category (`fly`, `worm`) | "label" |
| **Keyframe** | A box a human placed or corrected (drawn solid) | "manual box" |
| **Frame** | One video frame, indexed from 0 | "image" when it came from a video |
| **Project folder** | The folder Save writes `annotations.json` into | "workspace", "autosave folder" |
| **Auto-annotate** | Run the detector and bring the results back as candidates | "auto-label", "AI labeling" |
| **Track forward** | Propagate an existing box across later frames with OpenCV | "auto-track" — that is the *persistent mode*, this is the one-shot action |
| **Auto-track** | The persistent mode where stepping a frame advances every initialized tracker | "track forward" |
| **Source** | Where a box came from: `human`, `auto`, `track` | "provenance" in user-facing text |
| **Checkpoint** | A trained `.pth` / `.pt` weights file | "model file"; "weights" alone in a label |
| **Run** | One execution with its own timestamped output folder | "job" in the annotator |
| **Step** | A numbered hub page (`1. Label`) | "tab", "page" in help text — say "step 1" |
| **Stage** | One of the four training stages (detector / action head / ID head / unified) | "phase"; never confuse with **step** |
| **Evidence** | The per-group logit contributions behind an action decision | "explanation" in a label (that is the toggle's name, not the concept) |

---

## §5 The three measured Dear PyGui pitfalls

These cost a full debugging session each. They are measured, not inferred (V6).

### §5.1 Pitfall 1 — a `child_window` has no `rect_min`

| Item type | `get_item_rect_min` | `get_item_rect_size` | `is_item_hovered` while a button is held |
|---|---|---|---|
| `drawlist` | **works** | works | **True** |
| `child_window` | **raises `KeyError`** | works | **False** |

**The rule: `rect_min` always comes from the drawlist.** In the annotator that is `"canvas"`, not
`"canvas_win"`; in the analysis panel `"ana_canvas"`, not `"ana_canvas_win"`.

### §5.2 Pitfall 2 — hover goes False on a child window mid-drag

During a held drag, `is_item_hovered(<child_window>)` returns **False** while
`is_item_hovered(<drawlist>)` stays **True**. A drag handler guarded on the child window
therefore stops firing the instant the user presses the button — which is exactly the symptom
"I drag and no box appears".

**The rule: hover is always tested on the drawlist.** `analysis_panel.py:529` still tests
`is_item_hovered("ana_canvas_win")`. It works today only because the handler is click-only; the
moment anyone adds a drag it breaks exactly as the annotator did.

### §5.3 Pitfall 3 — the bare `except` that hides both of the above

```python
# WRONG — this is how the bug survived for months
try:
    ox, oy = dpg.get_item_rect_min("canvas_win")
except Exception:
    ox, oy = 0, 0        # silently wrong by exactly the canvas offset
```

A silent `(0, 0)` origin costs exactly the 8 px / 27 px canvas offset that made every click land
in the wrong place, and it is indistinguishable from a legitimate origin at the top-left.

**The rule: never wrap a `rect_min` or hover lookup in a bare `except`.** Catch
`(KeyError, SystemError, TypeError)` explicitly and log **once**. In the analysis panel
(`analysis_panel.py:534`) the current bare `except: return` makes a mis-click indistinguishable
from a miss.

### §5.4 Secondary dpg notes worth keeping

| Note | Detail |
|---|---|
| Key-press handlers carry no modifier state | A modified binding is a plain key-press handler whose callback first checks `dpg.is_key_down(dpg.mvKey_ModCtrl)` / `mvKey_ModShift`. |
| `is_item_active` ≠ `is_item_focused` | The first input in a window auto-focuses, so focus alone is sticky and useless as a typing guard. See §0.3.8. |
| `delete_item(children_only=True)` wipes the whole scene | Use two `dpg.draw_layer`s (`layer_image`, `layer_overlay`) so a mouse-move rebuilds ~3 items instead of everything. |
| A raw texture in `mvFormat_Float_rgba` costs three full-frame allocations per frame | `np.dstack` + `astype(np.float32)/255` + `.flatten()` on a 1920×1080 frame is ~33 MB of churn per displayed frame. A `uint8` dynamic texture with a preallocated buffer avoids it. |
| Blocking tkinter dialogs freeze the render loop | `dialogs._dialog` creates a fresh `Tk()` per call on the render thread; the viewport stops painting and Windows may show "Not Responding". Hoist one long-lived hidden root, pass `initialdir`, and own the dialog to the viewport HWND. |
| A dpg `handler_registry` tag cannot be re-created | `_build_main_window` re-creates `key_handlers` / `mouse_handlers` on "New Project…", which is an alias collision in dpg 2.x. |

### §5.5 Measured — what got slow after a detector run, and the three fixes

Symptom: the annotator is fine until *Analyze* has run over a video, and then it is treacle.
Measured on this study's own scale — one hour at 30 fps (108,000 frames), detections every
15th frame, four animals, so **7,200 candidate frames and 28,800 boxes** — with
`YOAKE_PYTHON` and real dearpygui:

| Path | What it did | Cost | After |
|---|---|---|---|
| **Autosave, every ~4 s** | `ProjectSession.snapshot()` — `asdict` over every candidate of every video, on the main thread | **0.85 s per video**; ~**17 s** for a twenty-video study, with the window frozen for all of it | **0.07 s**, whatever the project's size |
| **Every accept / reject** | `rebuild_copilot()` — 9 quality detectors over every candidate, plus the tracklet index and the interval merge | **0.79 s** (`quality.analyze` 0.65 + `build_tracklets` 0.13) **per candidate** | scheduled; one rebuild per burst |
| **Every frame change** | `_render_timeline()` — reassemble the model and rebuild all four lanes | **6 ms** build + **15 ms** draw = 21 ms, on a 16 ms budget | **0.012 ms** (playhead layer only) |
| **Behavior lane** | one `draw_rectangle` per interval, unbucketed | 28,800 nodes, **0.15 s** per redraw, re-submitted every rendered frame | bucketed by pixel column, <4,000 draws |

The autosave one is the largest and the least obvious, because nothing about
it is drawing. `_autosave_workspace` re-arms on every frame step, so the
symptom is a multi-second freeze every four seconds while the user is simply
looking around. Two changes, and the order matters:

* **The detach got cheap.** `dataclasses.asdict` is a recursive walk with a
  per-field `deepcopy`; `Candidate.to_dict` writes the eight fields out and is
  3x faster, and `ReviewSession.saved_rows` reads the three parallel lists
  directly instead of paying a `sorted()` and a copied list per frame. Together
  0.41 s -> 0.028 s for one video's candidates, byte-identical output.
* **It stopped re-detaching what nobody touched.** `snapshot` caches each
  video's row against `VideoSession.revision`. The autosave names the open
  video in `always=`, because `visited`, `last_frame` and `ui` change as the
  user merely looks around and deliberately do **not** mark a video dirty — a
  frame step must not make the save chip say "Unsaved". Every other write goes
  through `touch()`. `save()` and the frame export pass every id, so a
  deliberate Ctrl+S is never anything but a full, current detach.

The cache is the one risky thing in this release — a stale row is lost work —
so it is pinned by seven tests in `tests/test_annotation_project.py`, including
`to_dict` against `asdict` and `saved_rows` against the public accessors.

The three drawing/analysis fixes, in the order they matter:

1. **Accepting schedules the re-analysis; it does not run it.** `schedule_copilot_rebuild()` sets
   a flag and `_tick_copilot()` runs it from the render loop once 0.35 s have passed with no
   further change. Holding `Y` through fifty frames was fifty full re-analyses of the same
   video — forty seconds of frozen window for a result only the last one produced. What the
   user is looking at (the boxes, the canvas, the undo entry) stays synchronous; only the
   analysis behind the Review Inbox waits. The wait is visible: `running_jobs()` reports
   *Re-checking the candidates*, so the deferral cannot read as an ignored click.

2. **The playhead is its own `draw_layer`.** This is §5.4's rule applied to the timeline:
   stepping a frame changes one integer on the model and not one thing the four lanes draw, so
   `render_playhead()` clears a layer holding three items instead of rebuilding fifteen
   hundred. `_render_timeline(playhead_only=True)` is the frame-change path; every mutation
   still takes the full rebuild, which is what keeps the reused model honest.

3. **The behavior lane buckets like the other three.** It was the one lane that broke this
   module's own stated "O(width), not O(frames)" rule. `merge_to_intervals` can only join
   frames within `max_gap` (2) of each other, so a stride-15 run merges nothing and the
   "intervals" are one per box. Sub-pixel intervals now collapse to one rectangle per
   (column, action), confirmed winning inside a column; intervals wide enough to read are
   still drawn whole.

Guarded by `tests/test_gui_timeline.py` (draw-call counts through a stubbed dpg, no viewport)
and `tests/test_workflows.py` (one rebuild per burst, and the accept itself still immediate).

---

## §6 Keyboard

**Source of truth: `gui/shortcuts.BINDINGS`.** The handler registration, every tooltip suffix, the
sidebar cheat sheet and the Shortcuts window are all generated from this one table, so they can
never drift again. Today the annotator's module docstring advertises a `T` binding that was never
registered and the File menu advertises `Ctrl+O` that was never bound.

**Global rule:** every binding except `Ctrl+S` is suppressed while an input widget is active
(§0.3.8).

**Key-constant rule:** keys are stored as dpg constant **suffixes**, so a non-existent constant is
caught by a unit test instead of crashing at startup. The measured suffixes that differ from the
obvious guess (see §0.0.3):

| Physical key | User-facing label | dpg suffix to store |
|---|---|---|
| Backspace | `Backspace` | `Back` (not `Backspace`) |
| Space | `Space` | `Spacebar` (not `Space`) |
| `[` | `[` | `Open_Brace` (not `LBracket`) |
| `]` | `]` | `Close_Brace` (not `RBracket`) |
| Ctrl (modifier) | `Ctrl` | `ModCtrl` (not `Control`) |

### §6.1 Annotator — Tools

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `V` | `V` | `tool.select` | Select tool: click a box to edit it, drag inside to move, drag a handle to resize |
| `W` | `W` | `tool.draw_box` | Draw Box tool: drag to create a new axis-aligned box, even on top of an existing one *(LabelImg "Create RectBox")* |
| `E` | `E` | `tool.draw_rotated` | Draw Rotated Box tool: drag to create a rotated box, then turn the yellow handle |
| `R` | `R` | `tool.draw_rotated` *(alias)* | Same as E; kept because R already meant "rotated" |
| `H` | `H` | `tool.pan` | Pan tool: drag to move the image |
| `X` | `X` | `tool.delete` | Delete Box tool: click a box to remove it |
| `Esc` | `Escape` | `tool.cancel` | Cancel the box being drawn, return to Select, and clear the selection |

### §6.2 Annotator — View

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `F` | `F` | `view.fit` | Fit the frame to the window |
| `+` | `Plus` | `view.zoom_in` | Zoom in |
| `-` | `Minus` | `view.zoom_out` | Zoom out |
| *wheel* | — | — | Zoom at the cursor (not a key binding) |
| *middle-drag* | — | — | Pan, from any tool (not a key binding) |

### §6.3 Annotator — Navigation

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `A` | `A` | `nav.prev` | Previous frame |
| `Left` | `Left` | `nav.prev` | Previous frame |
| `D` | `D` | `nav.next` | Next frame |
| `Right` | `Right` | `nav.next` | Next frame |
| `Home` | `Home` | `nav.first` | First frame |
| `End` | `End` | `nav.last` | Last frame |
| `Ctrl+Right` | `Right` + `Ctrl` | `review.next_pending_frame` | Jump to the next frame that still has candidates |
| `Ctrl+Left` | `Left` + `Ctrl` | `review.prev_pending_frame` | Jump to the previous frame that still has candidates |

### §6.4 Annotator — Review

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `Q` | `Q` | `review.cycle` | Select the next box in this frame, candidates first |
| `Y` | `Y` | `review.accept_all` | Accept every candidate in this frame |
| `N` | `N` | `review.discard_all` | Discard every candidate in this frame |
| `T` | `T` | `auto.track_forward` | Track the selected box forward from this frame |
| `Ctrl+R` | `R` + `Ctrl` | `auto.annotate` | Open Auto-Annotate |
| `Ctrl+T` | `T` + `Ctrl` | `auto.toggle_autotrack` | Turn auto-track mode on or off |

### §6.5 Annotator — Edit

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `Delete` | `Delete` | `edit.delete` | Delete the selected box |
| `Backspace` | `Back` | `edit.delete` | Delete the selected box |
| `Space` | `Spacebar` | `edit.propagate` | Copy boxes from the previous frame |
| `Ctrl+Z` | `Z` + `Ctrl` | `edit.undo` | Undo the last change (50 steps) |
| `Ctrl+N` | `N` + `Ctrl` | `edit.new_track` | Start a new track ID |
| `[` | `Open_Brace` | `obb.rotate_ccw` | Rotate the selected box 5 degrees counter-clockwise |
| `]` | `Close_Brace` | `obb.rotate_cw` | Rotate the selected box 5 degrees clockwise |
| `0`–`9` | `0`…`9` | `action.set_0` … `action.set_9` | Set the selected box's action to 0-9 |

### §6.6 Annotator — File

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `Ctrl+S` | `S` + `Ctrl` | `file.save` | Save — **works even while typing** |
| `Ctrl+Shift+S` | `S` + `Ctrl`,`Shift` | `file.save_as` | Save As |
| `Ctrl+O` | `O` + `Ctrl` | `file.open_video` | Open a video *(advertised in the File menu since day one; bound for the first time)* |
| `F1` | `F1` | `help.shortcuts` | Show the shortcut list |

### §6.7 Hub

| Key | dpg suffix | Action id | Description |
|---|---|---|---|
| `Ctrl+1` | `1` + `Ctrl` | `page.workflow` | Go to 0. Workflow |
| `Ctrl+2` | `2` + `Ctrl` | `page.label` | Go to 1. Label |
| `Ctrl+3` | `3` + `Ctrl` | `page.train` | Go to 2. Train |
| `Ctrl+4` | `4` + `Ctrl` | `page.analyze` | Go to 3. Analyze |
| `F1` | `F1` | `help.shortcuts` | Show the shortcut list |

The hub currently registers **zero** shortcuts while advertising some in prose.

### §6.8 Changed in this version

Rendered at the top of the Shortcuts window, because rebinds need announcing.

| Was | Now | Why |
|---|---|---|
| `A` toggled auto-track | `A` is **previous frame**; auto-track is `Ctrl+T` | LabelImg and YOMU both use A/D for prev/next, and LabelImg familiarity is an explicit goal |
| `N` / `P` moved between frames | `N` **discards candidates**; `P` is free | `N` completes the `Y`/`N` accept/discard pair; A/D and the arrows cover navigation |
| `R` toggled a hidden OBB mode | `R` selects the **Draw Rotated Box** tool | The mode becomes a visible tool with a visible active state |
| `0`–`4` set actions 0-4 | `0`–`9` set actions 0-9 | The panel already allowed more than five actions with no key |

**Unchanged on purpose:** `Space` still propagates from the previous frame, `[` / `]` still rotate
by 5 degrees, `W` still starts a new box anywhere, `Esc` still cancels, `Del` still deletes,
`Ctrl+S` still saves.

### §6.9 The LabelImg familiarity contract

Asserted literally by `tests/test_gui_shortcuts.py::test_labelimg_contract`, because these four
are what make an experienced annotator productive in the first minute:

| Action | Binding |
|---|---|
| `tool.draw_box` | bare `W` |
| `tool.cancel` | `Escape` |
| `edit.delete` | `Delete` |
| `file.save` | `Ctrl+S` |

---

## §7 The candidate lifecycle

The hard rule, restated because everything depends on it: **candidates never reach disk** (P1).

### §7.1 The four states

| State | `accepted` | `source` | Produced by | On disk? |
|---|---|---|---|---|
| human / accepted | `True` | `"human"` | Drawing, editing, propagating | yes, with no extra keys |
| auto / pending | `False` | `"auto"` | Auto-annotate ingest | **no** |
| auto / accepted | `True` | `"auto"` | Accepting an auto candidate | yes, `"source": "auto"` + `"score"` |
| track / pending | `False` | `"track"` | `Track forward`, auto-track mode | **no** |
| track / accepted | `True` | `"track"` | Accepting a tracked candidate | yes, `"source": "track"` + `"score"` |

A manual edit of any machine-produced box promotes it to `keyframe=True` (it renders solid,
as a human-confirmed box) but does **not** change its `source` — provenance records where a box came
from, not who last touched it.

### §7.2 What reaches `annotations.json`

| Field | In memory | Written |
|---|---|---|
| `accepted: bool` | `True` for human boxes, `False` for candidates | **Never.** Objects with `accepted == False` are excluded from the file entirely. |
| `source: str` | `"human" \| "auto" \| "track"` | `"source": "<value>"` **only when it is not `"human"`** |
| `score: Optional[float]` | detector or tracker confidence | `"score": <float>` only when `source != "human"` and a score exists |
| `keyframe: bool` | `True` = human-confirmed box | **Never** — see §9.3 |

`AnnObject.to_dict()` does `d.update(auto_label.provenance_fields(self.source, self.score))`, and
`provenance_fields` returns `{}` for a plain human box.

Why this shape:

* **A human-only project's JSON stays byte-identical to today**, so `tests/test_annotation_v2.py`
  (v1.1 byte-identical round trip) and `tests/test_labeling_ops.py` (`keyframe` absent,
  `class_id == 0`) stay green with no schema change.
* `data/annotation.py` is **not modified**. Its `_parse_object` reads named keys and ignores
  unknown ones, so `source` / `score` load harmlessly through the library path;
  `_object_to_dict` will not round-trip them, which is correct — the library's job is the
  training contract, not the GUI's provenance.
* Because candidates never reach disk, re-running the detector is always safe and an unreviewed
  run can never poison a training set.

### §7.3 Rendering

Repeated here because it is the part implementers get wrong. See §0.4.5 for the full table. The
three-line summary:

* **Dashed stroke = not yet accepted.** `[7,5]` for candidates, `[4,4]` for accepted-but-machine-filled.
* **Amber (`PENDING`) = awaiting your decision.** Never used for anything else.
* **Hue is track identity.** Candidacy never steals hue (P13).

### §7.4 Warnings and hand-offs

| Moment | Message |
|---|---|
| Saving with candidates present (once) | `{n} candidates are not saved. Press Y to accept them first, or N to discard them.` |
| Save status | `Saved {a} boxes across {f} frames. {p} candidates are still pending and were not saved as labels.` |
| Accepting one | `Accepted: track {n}.` |
| Discarding one | `Discarded 1 candidate.` |

### §7.5 The three `predictions.json` writers

`auto_label.load_candidates()` must tolerate all three. They do **not** agree, and this is a real
trap.

| Writer | How it is produced | Keys |
|---|---|---|
| **streaming** (default) | `yoake predict` | `boxes`, `scores`, `track_ids`, `action_ids`, optional `obb`, `context_boxes`, `evidence`, `gates`, `head_predictions` — **no `class_ids`** |
| **batch** | `yoake predict inference.mode=batch` | `boxes`, `scores`, `track_ids`, `action_ids` — four keys only |
| **pipeline** | `yoake pipeline` | `boxes`, `scores`, `class_ids`, `track_ids`, `interpolated`, `relations` — **no `action_ids`, no `obb`** |

Padding defaults, matching `gui/predictions.py::_as_list` exactly: `scores → 1.0`,
`track_ids → -1`, `action_ids → -1`, `class_ids → 0`. Malformed input raises `ValueError` naming
the file and the reason; a `KeyError` or `TypeError` must never escape.

### §7.6 The three empty-result diagnoses

`auto_label.describe()` must produce a **distinct** message for each cause (P5), never one
generic shrug:

| Cause | Message |
|---|---|
| Detections existed but all below threshold | `No detections above the score threshold. Lower "Score threshold" and run again.` |
| Every frame in range was protected | `Every frame in the range already has boxes you drew. Untick "Skip frames that already have boxes I drew" to re-run them.` |
| The run produced no artefact | `The run produced no predictions.json. Check the log for the reason.` |

Success: `Auto-annotation done: {added} candidates on {frames_touched} of {frames_total} frames
({skipped_protected} frames protected, {suppressed_iou} overlapped an existing box) — press
Ctrl+Right to jump to the first one, Y to accept a frame, N to discard it.`

Stopped: `Auto-annotation stopped. The {added} candidates already produced are kept.`

---

## §8 Manual acceptance procedure

No `tools/` file is in scope for this work, so the synthetic-input acceptance suite ships **inside
`annotator.py`** and the theme check ships inside `theme.py`.

### §8.1 Theme check

```
python -m htrtdetr.gui.theme --check
```

Prints every contrast pair with its measured ratio and its required threshold, and exits
**non-zero** on any violation, including the assertion that white-on-accent *fails*. The theme
owner runs this before handing off.

### §8.2 Annotator self-test

```
python -m htrtdetr.annotator --selftest
```

Builds a real viewport over a 16-frame synthetic clip, drives `SetCursorPos` + `mouse_event` +
`keybd_event` with `render_dearpygui_frame()` pumped 2–3× per step, restores the cursor at exit,
prints `PASS` / `FAIL` per case, and `sys.exit(1)` on any failure.

**Screen mapping (verified):** `viewport_pos + get_item_rect_min("canvas") + _i2c(ix, iy)` — the
**drawlist**, never `canvas_win` (§5.1).

| # | Case | Assertion |
|---|---|---|
| 1 | Coordinate accuracy | `_c2i(_get_canvas_mouse())` error is **exactly (0.0, 0.0)** at fit, and < 0.5 px at zoom 2.5 with a non-zero pan |
| 2 | Pitfall 1 lock | `get_item_rect_min("canvas")` succeeds; `get_item_rect_min("canvas_win")` raises `KeyError` |
| 3 | Pitfall 2 lock | During a held drag, `is_item_hovered("canvas")` is True and `is_item_hovered("canvas_win")` is False for **every** drag callback; `_is_over_canvas()` stays True |
| 4 | Five tools by key | `V W E H X` → `_tool` in `select draw obb pan delete` |
| 5 | Five tools by button | Invoking each callback reaches the same five states |
| 6 | Active tool unmistakable | For each tool: exactly one toolbar button's `get_item_theme` equals `accent_theme()`, the other four equal `ghost_theme()`; `tb_tool_chip` starts with `Tool: <name>` |
| 7 | Screenshot | `output_frame_buffer(png)` → `cv2.imread` → ≥ 200 `ACCENT` pixels in the toolbar row for a non-Select tool, 0 for Select. **Skips with a printed warning** if the PNG is < 100 bytes (driver quirk), because case 6 already covers the claim |
| 8 | Draw in Select mode | Drag on empty canvas → 1 box within ±1 px |
| 9 | **W draws inside an existing box** | With box A present, `W` + drag entirely inside A → 2 boxes, A unmoved |
| 10 | Single-shot | After case 9, `_tool == "select"` |
| 11 | Sticky | Tick `Keep drawing`, draw → `_tool == "draw"` |
| 12 | `E` creates an OBB | `has_obb` True, θ == 0, `bbox` equals the circumscribing rect |
| 13 | Delete tool + undo | `X`, click → count −1; `Ctrl+Z` → restored |
| 14 | Zoom / pan | Wheel +1 → `_zoom == 1.25`, the cursor's image point invariant within 1 px; `F` → `_zoom == 1.0`, `_pan == (0,0)` |
| 15 | Pan clamp | 50 wheel-downs plus a 5000 px pan → ≥ 64 px of the image still on canvas |
| 16 | **Typing guard** | `focus_item("sel_tid")`, send `W` → `_tool` unchanged **and** `is_item_active("sel_tid")` True; then `Ctrl+S` still saves |
| 17 | `Esc` mid-drag | No box created, `_tool == "select"`, `_mode == "idle"`, nothing selected |
| 18 | Drag leaving the canvas | Box commits, clamped to `frame_w` |
| 19 | Candidate lifecycle | Inject 3 candidates → `Y` accepts all; re-inject → `N` discards all; `to_dict()` output contains **zero** candidates |
| 20 | Review navigation | Candidates on frames 3 and 11 only; from frame 0, `Ctrl+Right` lands on 3, again on 11, again reports `No candidates after this frame.` |
| 21 | Undo depth | 60 deletes then 60 undos → exactly 50 restored, then `Nothing to undo.` |
| 22 | Track forward | Draw a box, `T` → ≥ 20 of 30 frames gain a candidate; none overwrites an existing object of that track |

### §8.3 The green line at every step

```
python -m pytest tests/ -q
python -c "import htrtdetr.gui, htrtdetr.gui.theme, htrtdetr.gui.shortcuts, htrtdetr.gui.auto_label, sys; assert 'torch' not in sys.modules"
python -m htrtdetr.gui.theme --check
python -m htrtdetr.annotator --selftest
```

The baseline to protect is **52 passed** on
`tests/test_labeling_ops.py tests/test_annotation_v2.py tests/test_gui_commands.py` (V8).

---

## §9 Known remaining gaps, stated honestly

These are real, they are not fixed by this work, and pretending otherwise would waste the next
person's afternoon.

### §9.1 The CLI emits Japanese that a GUI log pane will show

Both panels pipe subprocess stdout/stderr verbatim into their log widgets. Roughly 25 Japanese
strings are reachable this way. Known locations:

| File | Lines | Reachable from |
|---|---|---|
| `runners/val_runner.py` | 108-113, 143 | `Run Evaluation` — the multi-line "checkpoint not found" error and the DummyDataset warning |
| `cli.py` | 689, 701 | The banner, printed on `--help` or an unknown command |
| `inference/inferencer.py` | 52, 103 | `Run Inference` — the OpenCV `ImportError` |
| `inference/streaming.py` | 224, 226 | `Run Inference` — `torch.compile` info/warning |
| `config/config.py` | 655, 663 | Any command with `config=` — the pyyaml `ImportError` |
| `pipeline/*` (`io.py`, `builder.py`, `registry.py`, `resolver.py`, `stage.py`) | many | A future pipeline button |
| `stages/**` (`_common.py`, `rtdetr_stage.py`, `yolo_stage.py`, `fused_stage.py`, `memory_id_stage.py`, `obb_track_stage.py`, `gnn_stage.py`, `learned_pair_stage.py`, `keypoint_stage.py`) | many | Pipeline stage construction |

`PYTHONIOENCODING=utf-8` in `child_env()` makes these **legible** instead of mojibake, which is
all this work does. The real fix is English CLI messages, which is out of scope here. **A
hand-maintained translation table over another module's stdout is explicitly rejected**: it rots
on the first new message and gives false confidence.

### §9.2 `gui/runner.py` cannot be given `stdin=DEVNULL`

`runner.py` is out of scope, so the `cli.run_train` `input()` prompt cannot be closed at the
spawn site. `training_panel` prevents the hang by pre-validating `data.train_root` instead
(§0.2.4). That covers the only path on which `run_train` reaches `input()` today, but it is a
workaround, not a fix — a future `input()` anywhere else in the CLI will hang the GUI again.

`runner.py` carries further known defects, all out of scope and all recorded here so they are not
rediscovered:

| Defect | Detail |
|---|---|
| Cross-thread `dpg` mutation | `log()` and `_status()` call `dpg.set_value` from the worker thread; `on_done` runs there too and its consumers create and delete items |
| O(n²) log rendering | Every line re-joins up to 800 lines and pushes the whole string through `set_value` |
| Double-launch race | `is_running()` tests `self._proc`, which is only assigned *after* `Popen` returns |
| `stop()` races and mis-reports | A stop before `_proc` is assigned is silently swallowed; a successful stop is then reported as `Error (code=N)` |
| `terminate()` does not kill the tree on Windows | DataLoader / CUDA children survive |
| No `PYTHONUNBUFFERED`, no `-u` | Output arrives in 8 KB bursts, so a live run looks frozen |
| `on_done` is not called on failure paths | `FileNotFoundError` and the generic handler return without invoking it, leaking button state |

### §9.3 `keyframe` is still lost on reload

`AnnObject.to_dict()` does not write `keyframe`, and `from_dict` forces it to `True` — both
behaviours are **asserted by `tests/test_labeling_ops.py`**. Consequence after a save/reload
cycle: the solid/dashed distinction between boxes a human confirmed and boxes a machine filled
in is lost — every loaded box renders solid.

The correct long-term answer is a `scores.json`-style on-disk sidecar (YOMU's design: a
positionally-aligned file carrying `scores[]`, `origin[]`, `individual_ids[]`, `reviewed`,
`exclude`, `edges`, written temp + `os.replace` through one locked mutator, never adding columns
to the training-format label file). **It was deliberately not built here**, because a second file
that must be written by the same save call will silently desynchronise if anything writes one
without the other. Provenance for accepted objects goes inline instead (§7.2); candidates stay in
memory. **This bug is left open on purpose and recorded here.**

### §9.4 `tools/gui.py` is a live 877-line duplicate GUI

It still imports `htrtdetr.gui_cli` and `htrtdetr.gui.commands`, which freezes both APIs.
**Confirm with the user that it is dead before ever changing those signatures.**

### §9.5 Other known-broken things outside this work's scope

| Where | Problem |
|---|---|
| `annotator.py` | Opening a video silently discards all annotations (`self.annotations = {}`) with no dirty check and no confirmation. There is no unsaved-changes prompt anywhere and no autosave. |
| `annotator.py` | "New Project…" while a project is open re-creates existing handler-registry tags (alias collision) and never clears `annotations` / `cap` / `_trackers`. |
| `annotator.py` | `class_id` is un-assignable: class names are collected but `AnnObject` has no class field and `to_dict` hardcodes `0`. A multi-class project silently produces single-class data. |
| `annotator.py` | Round-tripping a v2.0 file destroys data: `keypoints`, `obb_source`, `keypoint_names`, `action_groups`, `video_path` and per-frame sizes are all dropped on load or never written. |
| `canvas.py` | The drawlist is a fixed 1100 × 560 inside a `-330 × -290` child window and never reflows, so the video neither grows nor shrinks with the window. |
| `dialogs.py` | `CKPT_TYPES` is defined and never used; checkpoint pickers fall back to `*.*`. |
| `gui_cli.py` | Values are never quoted or escaped; a path containing `=` or spaces produces a malformed override. Booleans stringify as Python `False`, and a test asserts that capital-F form. |
| suite-wide | The entire UI layer is untested. `tests/test_gui_commands.py` covers only `commands`, `monitor`, `predictions`. |

### §9.6 Deliberately not built

Stated so nobody mistakes an omission for an oversight.

| Not built | Why |
|---|---|
| VLM screening / confirmation-card flow | YOMU's best idea, but it needs a model, an API key, a crop pipeline and a modal grid |
| Gallery / tile view with provenance badges | Large new surface; the annotator is single-frame |
| `scores.json` sidecar | §9.3 |
| Redo | Doubles the state machine for a case neither reference implements |
| Multi-select, marquee, copy/paste | YOMU has none of these in its editor either; selection is a single index |
| `Shift+1..9` class assignment | A new per-object axis with no consumer, while a test asserts `class_id == 0` |
| Japanese CLI translation table | §9.1 |
| Regex progress-sniffing of CLI stdout | P20 |
| Changes to `data/annotation.py`, `commands.py`, `gui_cli.py`, `monitor.py`, `predictions.py`, `labeling_ops.py`, `runner.py`, `dialogs.py`, `recents.py`, `tools/` | Four test files pin the first six; the rest are outside the assigned buckets |

---

# Part III — The YOMU archive

## Appendix A — Borrowed from YOMU: reference behaviours

The reference tree (`ref/YOMU-main/`) was deleted after this run. This appendix records what was
in it in enough detail that nobody needs the original. Numbers are exact; they were read from
source, not estimated.

### A.1 What YOMU is

A web app: Svelte + TypeScript front end (`frontend/src/`), FastAPI + Python backend (`yomu/`).
Five "modes" sharing one project tree: **YOLO** (detection/segmentation), **POSE** (keypoint
tracking with CoTracker3), **MOTEN** (motion energy), **COMBO** (YOLO × POSE), **LINK** (a node
graph for live input → condition → output). Tabs within a mode: `0. Workflow`,
`1. Source / Detect`, `2. Review / Edit`, `3. Export`, `4. Train`, `5. Scan / Evaluate`,
`6. Video list`, `7. Live`, plus `Extra.` sections.

### A.2 The editor's architecture (worth copying, and we did)

A DOM-free controller plus pure functions plus one imperative canvas painter:

| File | Lines | Role |
|---|---|---|
| `controller.ts` | 881 | All state and behaviour. Never touches the DOM. Emits an immutable snapshot to subscribers. |
| `geometry.ts` | 178 | Pure: transforms, hit tests, handle detection, drag math, cursor shape |
| `edges.ts` | 148 | Pure: skeleton-edge derivation and index bookkeeping |
| `lut.ts` | 63 | Pure: LUT clamping, drag, auto-clip, filter-string generation, histogram |
| `render.ts` | 250 | `drawScene(ctx2d, w, h, view, context, t)` — one imperative call |
| `types.ts` | — | Snapshot shapes plus `controls(v, ctx)`, which computes every button's enabled/locked state **in one place** |

The annotation arrays live in a shared `FrameRepository`; the editor mutates the *same object
references* rather than keeping a second cache, and if some other actor **replaces** the
annotations array it drops `selected`, `edgeDraft` and `polyDraft` because indices no longer mean
anything.

### A.3 YOMU's tool model — and why YOAKE deliberately differs

**YOMU has almost no tool palette.** There is no select/draw/pan/erase toolbar. Behaviour is
determined by *where you click*, *which app mode is active*, and *modifier keys*. Pan is always
middle-drag; zoom is always wheel; delete is always right-click; create is always
drag-or-click-on-empty. The only tool switch is a two-radio `Editing tool:` `bbox` /
`segmentation`, and only in YOLO mode.

**YOAKE adds an explicit five-tool toolbar anyway**, because (a) YOAKE has rotated boxes, which
YOMU does not, so "draw" has two variants that must be distinguishable; (b) the previous YOAKE
build hid OBB mode behind a sidebar checkbox 600 px from the canvas and users could not tell
which mode they were in; and (c) LabelImg familiarity (bare `W` = new box) is an explicit goal.
The mouse grammar underneath is still YOMU's.

### A.4 YOMU's exact interaction constants

These port directly and several are already in YOAKE.

| Constant | Value | Meaning |
|---|---|---|
| `CLICK_TOL` | 5 px | Movement under 5 px in **both** axes is still a click, not a drag |
| handle hit tolerance | 7 px | Applied to the *edge lines*, not just the drawn squares — far more forgiving than it looks |
| polygon vertex grab | 7 px | Checked **before** box edges so vertices win |
| pose point hit | 12 px | On screen, so you can grab points without zooming |
| edge hit | 8 px | Perpendicular distance to the segment |
| polygon close | 10 px | Click within this of the first vertex closes the draft |
| handle draw size | 7 × 7 px | `hs = 3.5`, fill `#ff5252`, 1 px `#fff` stroke |
| `MIN` box | 4 px | Minimum box dimension |
| fit margin | 16 px | Total, not per side |
| wheel zoom | `factor = exp(-deltaY * 0.0015)` | Clamped `[fitScale * 0.2, fitScale * 40]`, cursor position held fixed |
| undo depth | 50 | Per frame |
| peek tap threshold | 300 ms | Space held longer = peek; shorter = toggle |
| auto-LUT clip | 0.35 % | ImageJ's value |
| deblock threshold | `scale > 1.5` | Below this, no blur |

### A.5 YOMU's left-button decision order

1. POSE + Alt + no polyDraft → edge-creation click
2. POSE + `Drag to fix` on + no polyDraft → `poseHitNear` (≤ 12 px) → immediately start a move
   **and** select that point (the "no pre-selection required" path)
3. Something selected → `handleAt(selected)` → vertex / resize / move, with
   `snapshot = structuredClone(annotations)` held for undo
4. Else → record `pressPos`, `dragging = false` (candidate for a new box)

On `pointerUp`: if an edit state exists and `moved`, push undo and mark dirty (and if the
annotation was a candidate, **promote it**: *"Edited candidate accepted"*); if an edit state
exists and not moved, **restore the pre-drag snapshot verbatim** and fall through to `clickAt()`.

`clickAt` on a YOLO box: hit-test in image space with `2/scale` px slop; among all boxes
containing the point the **smallest-area one wins**, so a small box nested inside a large one
stays grabbable. **If the hit annotation is a candidate, the click accepts it.**

Right-click: hit test → **delete immediately**. No context menu, no confirmation, anywhere in
YOMU.

### A.6 YOMU's asymmetric move clamping (deliberate, and commented)

* **Boxes** clamp so the whole box stays inside the image.
* **Points** clamp the **centre** to the image bounds — box-based clamping would teleport a
  border point `r` px inward the instant you grabbed it, making the outer `r`-px band
  unlabelable.

### A.7 YOMU's provenance model — four orthogonal channels

This is the single most valuable idea mined, and YOAKE implements a reduced version of it.

**Axis 1 — per annotation: `accepted`.** Solid vs dashed stroke; `✓` vs `○` in the list; the
suffix `" (cand)"` on the label.

**Axis 2 — per frame: `origin[]` tags** stored in a sidecar `labels/scores.json`:

| Tag | Badge | Filter label | Written by |
|---|---|---|---|
| `human` | 👤 | `👤 Human-corrected` | Any editor edit, undo, bulk class discard, confirmation apply |
| `llm` | 🤖 | `🤖 LLM-screened` | AI screening (both apply paths) |
| `added` | ➕ | `➕ Manually added` | A hand-drawn box |
| `track` | 📍 | `📍 Auto-labeled (CoTracker)` | CoTracker auto-labeling |
| `model` | 🎯 | `🎯 From the scan model` | Frames added from a full-video scan |
| `convert` | ♻ | `♻ Converted` | Imported from another mode |
| `burst` | 🎬 | `🎬 Burst start` | First frame of an extracted burst |
| `exclude` flag | 🚫 | `🚫 Excluded from training` | Exclude-from-training |

**Axis 3 — per frame: `reviewed`.** `✓` = reviewed and clean; `◇` = has labels but not signed
off; blank = unlabelled. Note that **dirty demotes** `✓` back to `◇`.

**Axis 4 — audit severity.** `⛔` blocker / `⚠` warn / `ℹ` info, from a read-only label-database
audit.

So a frame simultaneously carries: which machines touched it, whether a human signed off, whether
each box is confirmed, whether it is excluded, and whether the audit flagged it — **five
orthogonal states, no single "unconfirmed" flag.**

The sidecar shape, worth copying if the §9.3 gap is ever closed:

```jsonc
{"frame_000120": {
   "scores": [0.87, 0.64, null],            // per-row, positionally aligned with the .txt
   "origin": ["human", "llm"],
   "individual_ids": [0, 0, 1],             // omitted entirely if all null
   "reviewed": true,                        // omitted if false
   "edges": {"add": [[0,2]], "del": [[1,3]]},
   "exclude": true                          // omitted if false
 }}
```
Both the label file and the sidecar are written temp + `os.replace` — a direct `write_text`
truncates first, and a concurrent reader would silently see fewer rows. All sidecar mutation
funnels through one function holding a module-level lock, and the contract is that the mutator
**only touches its own keys** — a whole-dict snapshot assignment rolls back concurrent updates
(this actually happened in YOMU's `delete_class`).

### A.8 YOMU's four "machine labels the data" flows

A naive reading collapses these into one. They differ in what they write and when.

| # | Flow | Model | Writes to disk? | origin tag |
|---|---|---|---|---|
| A | Batch detect over all frames | SAM3 text-prompt | **No** — in-memory `accepted:false` candidates only | none until a human saves |
| B | VLM screening of candidates | Claude / OpenAI-compatible / `claude -p` CLI | No — deletes rejected candidates in browser state | `llm` |
| C | Pose auto-point | CoTracker3 | **Yes**, per segment, mid-run | `track` |
| D | AI point suggestion | Claude/Codex CLI VLM, 2-pass | **Yes**, per frame | `llm` (or `["human","llm"]`) |
| E | Carry-over on frame ingest | A trained YOLO from a scan run | **Yes**, at frame creation | `model` / `track` + `added` |

The honesty principle running through all of it: **A and B never touch disk.** C, D and E do, and
are ring-fenced by the anchor/target policy (P3).

### A.9 YOMU's confirmation modal — the diff review

Full-screen overlay (`rgba(0,0,0,.55)`, z-index 50) containing a `min(1060px, 100%)` modal, max
height 88vh, as fixed head / scrolling grid / fixed foot.

* Head: **`Confirm AI verdicts`** · `{n} candidates` · spacer · `Reset to AI suggestions`
* Grid: `repeat(auto-fill, minmax(200px, 1fr))`, one card per candidate. Each card is a
  220 × 165 canvas crop plus: class name (bold), `AI: {Keep|Discard}({confidence to 2dp})
  「{reason}」`, the **frame file name** in dimmed monospace, and a two-button segmented
  `Keep` / `Discard` control filling the card width (active keep = green, active discard = red,
  white text).
* Foot: `Keep {k} / discard {r}` · spacer · `Cancel` · `Apply (accept kept, delete discarded)`

**Granularity is per candidate box, not per frame and not per batch.** Cards from multiple frames
are interleaved; the frame name is the only grouping cue.

**Pre-seeded from the model**: `keep = !!verdict.match` on open, so it is a *diff review* — you
only touch the ones the model got wrong. `Reset to AI suggestions` is the only bulk operation;
there is no "keep all" / "discard all".

**Crop preview:** margin `m = max(w,h) * 0.6`, crop clamped to the image, letterboxed into the
canvas over black, then the original box stroked in `#ff2828` at lineWidth 2. The **server does
exactly the same thing** before sending the crop to the VLM, so the human sees exactly what the
model saw.

**Weakest point, do not copy:** the modal handles only `Escape` (cancel). No arrow navigation, no
per-card key, no Enter-to-apply. If YOAKE ever builds this, add `←`/`→` between cards, `K`/`D` to
keep/discard, and `Enter` to apply.

### A.10 YOMU's gallery bulk vocabulary

The "checked frames" set is the universal *what am I operating on*, and the generator auto-checks
the frames it produced candidates on. Every bulk action reads it, and every one guards with
`No frames are checked` first.

Row 1 (view): `{n} checked` · `Check all` (visible/filtered only) · `Uncheck` · `Size`
(140–480, default 240) · `Filter` · `Sort` (`Frame order`, `Confidence low→high`,
`Confidence high→low`, `Classes high→low`, `Instances (duplicate labels) high→low`) ·
`Showing {a} / {b} frames` · `Audit`

Row 2 (actions on the checked set): `Discard this class's candidates` · the five AI buttons ·
`Accept checked & save` · `Discard checked candidates` · `Clear reviewed` ·
`Exclude from training` · `Un-exclude` · and, right-aligned and styled danger,
`Discard ALL labels on checked`.

### A.11 YOMU's read-only audit

Design rules worth importing wholesale even though the feature is not built in YOAKE:

* **Read-only; one byte written is a bug.** It imports neither the web framework nor app state.
* **Do not re-implement the export's conditions — call the export and catch**, then classify the
  exception by *re-deriving the cause*, never by string-matching the message. This way the audit
  can never say "clean" while training dies.
* **Severity is engine-dependent.** `severity` answers "does anything stop?"; `blocks` answers
  "which engines stop?"; `summary_by_engine` answers "can I train with *my* engine?"
* **Only frames that will actually be used are audited** (excluded frames skipped), so
  `blocker: 0` literally means "you can start training now".
* **Thresholds are ratios of animal size, never fixed pixels** — a fixed px threshold changes
  meaning with resolution and subject size.
* **A `warnings[]` channel for "this check could not run"**, so "no findings" never silently
  means "nothing was checked".
* **`AUDIT_NO_EXCLUDE`** — a set of finding codes that get a filter/jump affordance but **no**
  bulk-exclude button, because bulk-excluding them would be wrong (legitimate empty labels,
  keep-one duplicates, things fixed by moving a point).

### A.12 YOMU's queue

A drawer mounted outside the tab router, reachable from any tab, with a header badge (`▶{n}`
running, `{n}` queued). Status vocabulary and colours:

| Server status | Label | Colour |
|---|---|---|
| `queued` | `Queued` | `#9aa4b2` |
| `running` | `Running` | `#4da3ff` |
| `done` | `Done` | `#2ecc71` |
| `failed` | `Failed` | `#ff5252` |
| `interrupted` | `Interrupted (restart)` | `#f5a623` |
| `canceled` | `Stopped` | `#f5a623` |

Note the deliberate split: the **button** is `Stop`, the **state** is `Stopped`, kept as separate
strings so they can diverge.

Mechanics worth copying into any DPG polling code: **never `setInterval`** — schedule the next
tick only after a response lands; every request takes a monotonic sequence number and a response
with `seq <= applied` is discarded; concurrent GETs collapse into one in-flight promise; a failed
poll is silently skipped and the previous state stays on screen; the poller stops itself when
there is nothing active and restarts on the next add.

And the single-entry-point rule, stated as a hard contract in YOMU's source: **every training and
analysis start goes through the queue; there is no code path that calls `/api/*/start`
directly.**

### A.13 YOMU's `job_record.json`

Written next to every run's artifacts. The `provenance` vocabulary is the interesting part:

| Value | Meaning |
|---|---|
| `snapshot` | Frozen into the run; reproducible |
| `shared` | Points at a shared export dir that the *next* export will overwrite |
| `external` | Outside the project, unmanaged |
| `derived` | Another run's output |

It exists so a reader can answer "would re-running these params reproduce this?" — and the answer
is **no** whenever an input is `shared`. `params` is canonical (not argv), the environment is
re-snapshotted **per run** (not per process, because each runner subprocess re-reads whatever is
on disk at launch), prior records fold into `history`, and `save()` **never raises** — a failed
record must not fail a successful training run.

### A.14 YOMU's visual language, for reference

```
--bg        #1e2024     --bg-panel  #26292f     --bg-input  #17181c
--border    #3a3e46     --text      #d8dbe2     --text-dim  #8b909c
--accent    #4da3ff     --accepted  #2ecc71     --pending   #f5a623
--selected  #ff5252
CLASS_PALETTE = ['#f5a623','#ff5abf','#38bdf8','#a78bfa','#facc15','#fb7185']
font: "Segoe UI","Yu Gothic UI","Meiryo",system-ui  at 13px
```

YOAKE's `ACCENT` `(77,163,255)` **is** `#4da3ff`; `ACCEPTED` `(46,204,113)` is `#2ecc71`;
`PENDING` `(245,166,35)` is `#f5a623`; `DANGER` `(255,82,82)` is `#ff5252`. The four semantic
colours are taken from YOMU directly. The greys were re-derived for contrast (§1) rather than
copied.

Box rendering: label chip is a filled rect **in the outline colour** with text drawn in
**`#111`** — *dark text on a coloured chip*, never coloured text on dark. This is where §1.9's
`ON_ACCENT` rule comes from.

**Hover: nothing is drawn on hover over the image in YOMU.** The only hover feedback is the
cursor shape — deliberately, to keep visual noise off the subject. YOAKE adds a light hover
(thickness + 1 and a 60-alpha white fill) because DPG cannot change the cursor per-region as
cheaply as CSS can.

### A.15 YOMU's "hold Space to peek"

Holding Space hides **all** overlay (annotations and the in-progress draft) so you can judge raw
pixels; releasing restores it. A tap under 300 ms instead toggles the selected annotation between
accepted and candidate. Key repeat is ignored so it does not flicker, and `blur()` also clears
the peek state because releasing Space outside the window never delivers a keyup.

Not adopted in YOAKE — `Space` is already `edit.propagate` and that muscle memory is older — but
recorded because "hide the overlay to see the pixels" is a real need in a fly-sized-subject
project and a free key (`P` is now unbound) could carry it later.

### A.16 YOMU's LUT bar — the image adjustment that is not brightness/contrast sliders

Recorded because it is the single best idea YOAKE has not adopted and the maths is exact.

It is an **ImageJ-style min/max intensity window drawn directly on a luminance histogram**:
drag either handle on the histogram, double-click to reset, `Auto` applies ImageJ's 0.35 %
clip. The equivalent CSS filter pair is derived as:

```
L = lo/255 ;  W = max(hi-lo, 1)/255
c = 1 + 2L/W ;  b = 1/(W + 2L)      ->  brightness(b) contrast(c)
```

The exact numpy/OpenCV equivalent, for a DPG port, is
`out = clip((img - lo) * 255/(hi - lo), 0, 255)`. The histogram is computed by downscaling to
480 px wide and binning `(R*77 + G*151 + B*28) >> 8` into 256 bins.

**The critical design point:** the LUT is sent to the detector — the detect request carries
`contrast: {lo, hi}` — so **the model sees exactly the stretched image the human is looking at**.
It never modifies the stored image or the labels, and the tooltip says so:
`Applies to the display and individual SAM3 detection. Original images and training data remain
unchanged`.

### A.17 YOMU's active-learning triage panel

`Frames to label next (lowest confidence first)`. Reads `likelihood` from an existing analysis
run; no extra training, no ensemble. Ranking: primary key = number of bad points
(`n_low + n_miss`, NaN counted as worst), secondary = mean likelihood ascending. **Missing and
low-confidence are deliberately separated** — mixing them lets all-missing frames monopolise the
top and the list becomes useless.

Minimum-gap logic worth copying: start at `gap = max(user_gap, len(series) // (count*2))`,
greedily pick, and if short of `count`, **halve the gap and retry** until `gap <= 1`; then report
the gap actually achieved.

Clicking a row adds that second's frame to the training data through the same code path as
"add frames from video", so carry-over and origin tagging behave identically.

**This is the highest-leverage screen YOMU has that YOAKE does not.** It is cheap (reads existing
confidences from `predictions.json`), it needs no new model, and it directly answers the only
question that matters after a training run: *what should I label next?* If there is budget for
one more feature after this work, build this one.

### A.18 YOMU's CoTracker propagation, for reference

Not adopted (YOAKE has no keypoints yet), but the identity rule is transferable: the tracking
query key is `f"i{individual_id}:c{class_id}"`, and the source comments **twice** that you must
not key by class name alone — two individuals sharing a bodypart would collapse to one point and
one animal would vanish entirely.

Also: inferred individual IDs are **never written back to disk**, because "persisting an
ambiguous guess means a human's later correction gets reverted".

And the visibility-gate asymmetry, which is a genuinely good rule: in batch mode, points with
visibility ≤ 0.5 are **skipped** and counted as `low_vis`; in interactive mode the point **is**
returned with `low_vis: true`, "because a human is looking at it and will fix it". Same
threshold, opposite policy depending on whether a human is present.

---

## Appendix B — Borrowed from YOMU: verbatim English strings

Every line below is the **exact** value from YOMU's shipped `I18N_EN` dictionary or its component
source. `{x}` placeholders are as-is. Reuse them directly where the situation matches; they are
already battle-tested phrasing.

### B.1 Candidate review

```
Candidates
{n} candidates
Accept all candidates (Y)
Discard all candidates (N)
Accept every pending candidate in this frame (Y). Not saved yet
Discard every pending candidate in this frame (N). Accepted points are kept
Accepted {n} candidates
Discarded {n} candidates
Accepted: {c}
Reverted to candidate
(cand)
Accept or discard pending candidates before marking this frame reviewed.
{n} candidates — click to accept, right-click to delete, or "Accept all"
```

### B.2 Auto-annotation lifecycle

```
Starting batch detection (first run loads the model)
Batch detecting {i}/{n}: {name} — "{p}" (thr {t})
Batch done: candidates on {f}/{n} frames — uncheck unwanted frames, then "Accept checked & save"
Batch detection stopped ({n} frames processed)
Detection complete: processed {n} frames, candidates in {f}, skipped {s} reviewed frames.
Batch detection failed
Detection error:\n{e}
AI screening batch {i}/{n} ({m} candidates, {model})
AI screening done — review the cards and apply
AI screening done: discarded {r}, kept {k}
Auto-labeling done: {n} frames (anchors: {a})
Auto-labeling interrupted: {n} frames already applied (anchors: {a})
 (skipped {k} low-confidence points)
 (protected {e} frames edited during the run)
 ({b} frames skipped: before the first anchor)
Stop requested…
Stop error: {e}
Stop AI point labeling (frames already written are kept)
Stop CoTracker auto-labeling (frames already applied are kept)
Applied: accepted {k}, deleted {r} (unsaved)
Cancelled — candidates left unchanged
Frame changed; detection results discarded
Discarded the result because the image or project changed
```

### B.3 Preconditions and empty-result diagnoses

```
No frames are checked
Load a video or image folder first
Set at least one class & prompt
Open a project first
Open a video first
Add a class first
Select a class first
Select a model, then start LINK.
Press "Inspect" first
AI screening needs per-class criteria (questions) — set them in tab 1
No pending candidates in the checked frames — run "Detect all frames" in tab 1 first (accepted boxes are excluded; press Space to turn one back into a candidate)
No candidates for classes that have criteria (all candidates belong to classes without one)
Nothing to place — every class is already labeled here
```

### B.4 Errors

```
Error:\n{e}
Save error: {e}
Load error: {e}
Detection error:\n{e}
Tracking failed: {e}
Failed to fetch: {e}
Cannot open the video. Check its path and format
Could not list cameras. Check dependencies and camera permissions.
Image unavailable. Check whether the source video moved or registration failed.
The selected camera is no longer available. Refresh the list and select it again.
Could not determine the source video — please select the video file
ultralytics is not installed (`uv sync --extra train`)
YOLO training and scanning require `uv sync --extra full`
CoTracker requires torch (re-install the dependencies with `uv sync`)
The codex command was not found. Install it with `npm install -g @openai/codex`, run `codex login`, then restart YOMU
No scan in progress
No training in progress
No tracking in progress
No analysis is running
Not recording
```

### B.5 Destructive actions and confirmations

```
This will permanently discard all labels on the {n} checked frames.\nHuman-made and accepted labels are deleted too. This cannot be undone. Continue?
Delete class "{c}"?\nIts saved annotations are removed from ALL frames and later class IDs are re-packed.\n(Unsaved candidates will be cleared)
Exclude {n} frames with "{title}" from training? Images and labels are kept.
Unsaved edits
There are unsaved edits on {n} frames before {action}.
There are unsaved edits on {n} frames before {action}.\nThe {p} pending candidates are not persisted by a normal save; choosing save will accept all of them.
Save and continue
Discard and continue
Accept all candidates, save, and continue
Cancel
Ask {a} to estimate keypoints for {n} frames (10-20 s per frame). Proceed?
Using human-labeled frames as anchors, CoTracker will auto-place points on the remaining unlabeled frames.\nLong anchor spans can take several to ~15 minutes. Run?
Start anyway? (Training may crash from insufficient VRAM)
```

Action-name gerunds for `{action}` interpolation:
`deleting a class` · `opening a project` · `extracting frames` · `exporting` ·
`switching mode` · `mining frames` · `importing from another mode` ·
`adding a frame to the training data` · `adding low-confidence frames`

### B.6 Progress

```
Loading…   Saving…   Starting…   Stopping…   Detecting…   Exporting…   Importing…
Computing…   Writing…   Extracting frames…   Loading model…   Stop requested…
Scanning… {a}/{b} frames
Tracking… {a}/{b} frames
Training {name}: epoch {e}/{n}
Preparing   Building clip   Analyzing frames   Filtering   Creating labeled video
Re-encoding to H.264   Creating dataset   Training   Evaluating (validation)   Done
ETA   elapsed   Elapsed {d}   ~{d} left   ~{d} left (ETA {t2})   Updated {t2}
{h}h {m}m   {m}m {s}s   {s}s
Training finished ({m} min)
Scan finished ({m} min). Review the result in "6. Video list"
Export finished
```

### B.7 Empty states

```
Nothing yet
No frames match
(none)
(no match)
(no analysis runs)
(no runs yet — created when you start training)
(no trained project — train in tab 4 first)
(no annotated videos yet — create them in "5. Evaluate / Video")
(not judged)
No history yet (recorded every 5 s since the server started)
Waiting for a processing result
No coordinate data (no plot)
The queue is empty. Configure a run above and press "Add to queue".
Load a video or image folder in the "1. Source / Detect" tab
Start the input to show the preview here
Select a node on the canvas to edit its settings here.
```

### B.8 The permanent editor cheat sheet (YOLO)

Reproduced verbatim; this is YOMU's entire discoverability strategy for the editor (P10).

```
Tiles: click to open editor / check for bulk actions
Left click: accept candidate / select
Right click: delete box
Drag: add a box manually
Selected: resize with handles, drag inside to move
Space: accept⇔candidate (hold to hide all labels) / G: tiles⇔editor
Shift+W / Shift+A: next / previous class tab
1-9: change class of selected box
Q: select next box / R: remove selection / Z: undo
Y: accept all candidates / N: discard all / X: exclude from training
L: lock view (keep zoom)
Polygons: vertices are always draggable / to draw a new one set the editing tool to segmentation, empty-click, then Enter
Wheel: zoom / middle-drag: pan
```

### B.9 Negative guarantees (the phrasings worth stealing wholesale)

```
Does not modify a single byte of your labels
the source analysis runs are never modified
Images and labels are kept; the frames are only left out of training, export and auto-labeling. Reversible
Display only — saved labels and detection are unaffected
Changes annotations only; downstream values are unchanged.
Labels and existing analysis runs are left untouched
Output files are kept.
(Saved annotation data on disk is kept)
Nothing is saved — review them, press "Accept all candidates", then save
```

### B.10 Calibration disclaimers

```
Registration confidence is not a calibrated success probability.
Axis values do not identify behaviors, and islands are not classified automatically.
Array order does not track individuals across frames.
Individual order is not a tracking ID.
Matching array indices refer to the same detection within one frame, not tracking IDs.
Request counts and processing results are separate. They do not verify physical output.
```

### B.11 Provenance tooltip (the whole badge vocabulary in one string)

```
👤=human-edited, 🤖=LLM-screened, ➕=added by hand, 📍=CoTracker auto-labeled, 🎯=scan model detection, ♻=converted from another mode, 🎬=burst start, 🚫=excluded from training
```

### B.12 Workflow-page copy worth adapting

```
Core idea: instruct SAM3 in language, let a VLM screen, humans only audit. Human work shifts from "draw from scratch" to "fix machine output".
Rule of thumb: one human-audit round beats more prompt tuning or mining (measured: weak-class AP 0.067 - 0.817). Filters + shortcuts make audits fast.
Each scan gets its own yolo/preview/<timestamp>/, so earlier scans are never overwritten.
Keep the default fixed time-block split — frames always fall on the same side as data grows, so generations stay comparable.
Tab 5 is where you create, tab 6 where you revisit. Added frames carry the scan model's detections as labels, so you only refine.
"Continue" speeds up retraining after adding data; after big label fixes, "from pretrained base" is safer.
Buttons for features whose dependencies are missing are dimmed and cannot be pressed (hover a button for the reason).
```

### B.13 Notes on what YOMU deliberately does **not** have

Recorded so nobody assumes an omission is an oversight in the reference either.

| Absent from YOMU | Note |
|---|---|
| Rotation / oriented boxes | Boxes are strictly axis-aligned `[x1,y1,x2,y2]`. YOAKE's OBB work has no reference to lean on. |
| Multi-select / marquee in the editor | `selected` is a single `number \| null`. Multi-select exists only in the gallery, as per-tile checkboxes with Shift-range. |
| Redo | Nowhere in the product. |
| Arrow-key nudging of boxes | Arrows are frame navigation. |
| A context menu | Right-click is an unconfirmed immediate delete. |
| Per-track colouring | Colour is class or acceptance, never track; the individual appears only as an `i{n} · ` text prefix. |
| Hover highlight on annotations | Only the cursor shape changes. |
| IoU / NMS de-duplication | Zero hits for `iou\|nms` across the whole backend. |
| Interpolation in exported time series | Gaps are left blank so "not detected" stays distinguishable from "at the origin". |

---

*End of document. If you change behaviour that this file describes, change this file in the same
commit — it is the only record of why the behaviour is what it is.*
