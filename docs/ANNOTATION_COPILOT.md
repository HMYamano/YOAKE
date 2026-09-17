# YOAKE Annotation Copilot

**Status: describes what is built.** This is the companion to `docs/GUI_DESIGN.md`,
which stays the normative contract for the GUI's tokens, copy, keyboard and candidate
lifecycle. Nothing here overrides it; everything here builds on it.

The one-line change in thesis:

> **Before.** A human opens a 30-minute video, decides which of 54,000 frames to
> annotate, and draws every box.
>
> **After.** YOAKE analyses the video, says *"annotate these 72 frames and these 12
> clips; here are the candidate boxes; 94 places need your eyes"*, and the human
> reviews exceptions.

---

## 1. Where the code lives

```
src/htrtdetr/
    annotation/              NEW - pure logic. No dearpygui, no torch.
        status.py            VERIFIED / PREDICTED / PENDING, derived not stored
        tracklet.py          Tracklet, TrackletIndex, build_tracklets
        quality.py           9 issue detectors -> ReviewItem
        review_queue.py      ReviewQueue: filter, navigate, decide, carry over
        intervals.py         BehaviorInterval, merge_to_intervals, IntervalTrack
        features.py          frame descriptors: thumbnail, motion, focus, scene
        sampling.py          6 strategies, Budget, WorkloadModel, plan()
        propagation.py       Correction -> Transform -> PropagationPlan
        cache.py             CacheKey / DiskCache, keyed on video+model+settings

    gui/
        theme.py             (existing) design tokens + the Copilot aliases
        widgets.py           (existing) shared widget vocabulary
        shortcuts.py         (existing) + a new "review" scope
        auto_label.py        (existing) Candidate, ReviewSession, track_forward
        sidecar.py           (existing) the subprocess channel for heavy AI
        timeline.py          NEW - four-lane timeline over one frame axis
        inspector.py         NEW - context-sensitive right pane, 6 contexts
        review.py            NEW - ReviewInbox (docked) + ReviewMode (overlay)
        smart_annotate.py    NEW - the Smart Annotate dialog
        settings.py          NEW - the five-tab project settings window
        video_segment.py     NEW - SAM 2 video propagation driver
        embedding.py         NEW - EmbeddingProvider: sidecar / thumbnail / null

    annotator.py             REFACTORED in place - the four-region shell that
                             wires all of the above
```

### The purity contract, and why it is load-bearing

`htrtdetr.annotation` imports **neither dearpygui nor torch**, at module scope or
inside a function; `cv2` and `numpy` are imported lazily inside the two functions in
`features.py` that cannot avoid them. `tests/test_annotation_core.py` asserts this
with an AST walk, not a grep.

This is not tidiness. Measured on this machine:

| Interpreter | dearpygui | OpenCV | torch |
|---|---|---|---|
| Python 3.14.3 (runs the GUI) | 2.3.1 | 4.13.0 | **absent** |
| miniconda 3.13.5 (runs training) | **absent** | **absent** | 2.10.0+cu128 |

Neither interpreter can import both. A rule that lived in the annotator would be
untestable from the training side; a rule that imported torch would be unusable from
the GUI. Keeping the logic pure is what lets the whole review loop be unit-tested on
either one — and both suites run it today.

---

## 2. The loop

```
  Smart Annotate                       AI proposes
      | scan at 1/stride fps
      | thumbnail descriptor + motion + focus + scene stats
      | strategy(features, budget, seed) -> frames + clips
      v
  Auto pre-label                       AI proposes
      | predictions.json (from the analysis page, or --import-predictions)
      | -> ReviewSession, all PENDING, none on disk
      v
  Tracklets + checks                   AI triages its own output
      | build_tracklets: one animal, one continuous run
      | quality.analyze: 9 detectors -> intervals, not per-frame items
      v
  Review                               human verifies exceptions
      | Review Inbox: filter by reason
      | Review Mode: Y / N / E / S, context loop
      | Accept tracklet: 300 boxes, one click
      v
  Correct + re-propagate               AI repeats the correction
      | Correction -> Transform -> the frames until the next keyframe
      v
  Train -> infer -> back to the top
```

### The invariant everything rests on

**A machine proposal never reaches `self.annotations`, and therefore never reaches
disk, until a human accepts it** (`GUI_DESIGN.md` P1). Tracklets, the review queue,
behavior intervals and the timeline's prediction lane are all *views over the pending
set*. There is exactly one promotion path — `AnnotationApp._accept_targets` — and
every batch action funnels through it.

`status.status_of()` derives VERIFIED / PREDICTED / PENDING rather than storing it,
for the same reason: a stored status is a second source of truth, and `GUI_DESIGN.md`
§9.3 records what that already cost for the `keyframe` flag.

---

## 3. Smart Sampling

Two regimes, one `plan()` entry point. The caller passes whatever signals it has.

**Cold start** (no model): the only signal is appearance. `diversity` does greedy
farthest-point traversal over 256-dimensional thumbnail descriptors. Two properties
earn it its place over k-means: it is deterministic given a seed, and the k-pick
traversal is a *prefix* of the 2k-pick traversal, so changing the budget recomputes
nothing.

**Active learning** (a model has run): uncertainty, novelty, temporal inconsistency,
interaction and rarity are all available, each rank-normalised to `0..1` and summed
with `ScoreWeights`. `diversity+uncertainty` then takes the best `k * 4` by score and
spreads `k` across *them* — because difficulty is temporally autocorrelated, and a
pure ranking of hard frames returns k frames from the same two seconds.

Detection samples **frames**; behavior samples **clips**, because "is this courtship?"
is unanswerable from a still.

| Strategy | Use |
|---|---|
| `random` | The honest baseline. Everything else should beat it. |
| `uniform` | Predictable, blind to content. |
| `diversity` | Before a model exists. |
| `uncertainty` | Where the model was least sure. |
| `diversity+uncertainty` | The default once a model exists. |
| `interaction` | Social behavior: where animals are close or touching. |

`SamplingRequest.to_metadata()` records target, strategy, seed, budget and every
weight, so a selection is reproducible.

### The budget is a guess, and says so

`WorkloadModel`'s four constants were **not measured** — there is no instrumented
annotation session to measure. They turn "15 minutes" into a plausible sample count,
and the dialog labels the result *estimated* and never quotes two significant figures
(`GUI_DESIGN.md` P12). The ratio that matters is real, though: an assisted frame is
budgeted at 6 s against 25 s for a blank one, which is the entire argument for
generating candidates first.

---

## 4. Quality control: 18,000 predictions, ~90 questions

Nine detectors, all unsupervised — there is no ground truth at review time, so every
one works from internal consistency.

| Code | Fires when |
|---|---|
| `low_confidence` | A run of frames where a track scored below `low_score` |
| `missing_animal` | Fewer animals than `expected_animals` |
| `extra_animal` | More animals than `expected_animals` |
| `merged_detection` | A box above `merged_area_factor` x the median area |
| `id_switch_risk` | Two tracks within `contact_distance_frac` of a box diagonal |
| `position_jump` | A step above `position_jump_factor` x *that track's own* median |
| `size_jump` | Area changing by more than `size_jump_factor` in one frame |
| `orientation_jump` | A body axis turning more than `orientation_jump_deg` |
| `track_fragment` | One track id stopping and restarting |
| `overlap` | Two boxes over `overlap_iou`, grouped per pair of tracks |

Three rules they obey:

1. **Intervals, never per-frame items.** A fly occluded for 40 frames is *one* thing
   to look at. Every detector merges consecutive hits through `_runs`. This is the
   difference between a 94-row queue and an 18,000-row list.
2. **Never guess which side is wrong.** `id_switch_risk` says identities are
   ambiguous; it does not say they swapped. Deciding is one click in the UI
   (`IDs are correct` / `Swap IDs from here`), and swapping is one undo step.
3. **Every threshold is in `QualityConfig`, scale-free where it can be.** A position
   jump is measured in multiples of that track's own median step, so the same config
   works on a 1024x1024 arena and a 4K one.

**`expected_animals` is the highest-value setting in the file.** "This preparation has
exactly two flies" is knowledge the detector does not have and the experimenter always
does. Set it from *Review > Expected number of animals*; without it the two count
detectors stay silent, because with a variable count every frame is consistent with
the design.

Verified on a 240-frame two-fly fixture with four planted defects: **470 predictions →
5 review items**, one per defect plus the contact they create.

---

## 5. Behavior as intervals, with no format change

`merge_to_intervals()` smooths per-frame predictions with a majority filter, run-length
encodes them, and absorbs sub-threshold runs into their neighbours. The human then
edits *edges*: accept, drag start, drag end, relabel, split.

**`annotations.json` does not change.** `IntervalTrack.to_frame_actions()` projects the
intervals back onto the per-object `action_id` the format has always stored. Intervals
are a view built on load and flattened on save, so every existing training path and
analysis script keeps working and no second file exists to desynchronise.

---

## 6. The four-region annotator

```
┌──────────────────────────────────────────────────────────────┐
│ menubar:  File   Edit   Project   Review   Help              │
│ topbar:   YOAKE  project [Saved]  Smart Annotate  Review      │
│           Undo Redo Save  Panels  Settings   tool sentence    │
├──────┬────────────────────────────────────┬──────────────────┤
│ Tool │                                    │ Inspector        │
│ Rail │             CANVAS                 │  one of six      │
│      │                                    │  contexts        │
│ 6 ›› │   zoom / pan / crosshair /         │   — or —         │
│ tools│   tool border + caption            │ Review Inbox     │
│ +/-  │                                    │                  │
│ fit  │                                    │                  │
├──────┴────────────────────────────────────┴──────────────────┤
│ TIMELINE   status / review / behavior / samples lanes        │
│ First Prev [frame] Next Last  Propagate    240 predicted …   │
│ status line                                                  │
└──────────────────────────────────────────────────────────────┘
```

**The legacy sidebar is gone.** All eleven panels moved, with their tags:

| Was | Is now |
|---|---|
| Frame Info, Active Track, Objects in Frame, per-frame candidates, Find all in frame | Inspector — *nothing selected* |
| Selected Object, Quick Actions, Box Shape conversions | Inspector — *box selected* |
| Select / Draw / Click buttons, Keep drawing | Tool Rail |
| Quick Start | the canvas empty state |
| Labels, Auto-Track, Click to Box backend, Project folder | **Settings**, five tabs, built hidden at startup |

`AnnotationApp._LEGACY_TAGS` records every move as data, and
`tests/test_workflows.py` asserts each one still resolves after the window is
built. The widget moved; the contract did not — eight callbacks and four
refresh helpers were not touched.

**Inspector contexts.** Nothing selected shows frame, animal count against the
expected count, the boxes on this frame, the track new boxes get, the per-frame
candidate actions, and the next action. A box shows class, track, behavior,
rotation, occlusion, the one-press behavior buttons, provenance, delete. A
candidate shows confidence, why it is queued, and the six accept scopes. A
tracklet shows range, **minimum before mean** confidence, issues, and
`Accept all N`. An interval shows label, timecode and edge edits. A review item
shows the reason, what to check, and the one-click answers.

**Timeline lanes.** Shape carries meaning, not only colour: an ID-switch risk is
a cross, a missing or merged animal is a triangle, everything else is a diamond;
a sample is a hollow circle and a keyframe is a filled square. Cost is
O(width), not O(frames).

**Layout is derived, not guessed.** `_bottom_height()` is
`Timeline.height + _UNDER_TIMELINE_H`, so adding a lane reflows the canvas
instead of pushing the controls out of the window — which is exactly what used
to happen: at the default 1280x820 the transport row sat at y=824 and the status
line at y=859, both below the bottom edge and unreachable.
`test_every_control_fits_inside_the_default_viewport` asserts the result.
**Panels** hides the right column and takes the canvas from 51% to 67% of the
viewport.

**Keyboard.** Every handler is generated from `shortcuts.BINDINGS`, with the
typing guard. Review Mode is a **separate scope**, so `E` means *Draw Rotated
Box* on the canvas and *Edit this item* in review without either being a
conflict. `tests/test_annotator_wiring.py` asserts both directions.

**Undo and redo** are multi-frame. An entry is a *set* of frame snapshots plus
where the user was, because Swap IDs rewrites a track across hundreds of frames
and a one-frame snapshot would restore the frame the user happened to be on and
leave the rest swapped. A half-undone swap is worse than no undo, because the
user believes it was taken back.

---

## 7. Threading, and why the GUI cannot freeze

The Smart Annotate scan runs on a worker thread that opens **its own**
`cv2.VideoCapture` — seeking one capture from two threads is not safe, and sharing the
annotator's would make the canvas show frames from the scan.

The worker never touches dearpygui. It puts closures on a queue; the render loop
drains them between frames (`_pump_ui`). That loop is explicit rather than
`dpg.start_dearpygui()` precisely so it can also carry the debounced autosave and
toast expiry.

Heavy models stay out of this process entirely, through `gui/sidecar.py`, which already
exists and already runs a helper under a *different interpreter* — the only design that
works when the GUI's Python cannot have torch.

---

## 8. Compatibility

Verified end to end on a synthetic project:

| Promise | How it holds |
|---|---|
| Existing annotation JSON loads | `AnnObject.from_dict` untouched |
| A human-only project's JSON is byte-identical | `to_dict` untouched; no new keys |
| Candidates never reach disk | They live in `ReviewSession`, not `annotations` |
| The library loader still accepts our output | `validate_annotation` → `Valid: True` |
| Existing manual annotation works | Every drawing path untouched |
| Click-to-Box works | `click_segment` untouched; backend still selectable |
| Predictions still import | `auto_label.load_candidates`, all three writers |
| The candidate workflow is unchanged | `ReviewSession` is still the state machine |
| Training and analysis pipelines | Not touched at all |

**One bug fixed in passing:** `analysis_panel.py` has been launching the annotator with
`--import-predictions`, a flag `annotator.main()` never accepted — so *Send Predictions
to Annotator* died with an argparse error before reaching the annotator. The flag now
exists and routes to `import_predictions()`, which is the auto-pre-label entry point.

---

## 9. What is not built

Stated so an omission is not mistaken for an oversight.

| Not built | Why, and what exists instead |
|---|---|
| A measured `WorkloadModel` | Its four constants are guesses and the UI labels them *estimated*. Replace them from an instrumented session and the feature gets better with no code change. |
| In-process inference | The annotator's Python has no torch. Pre-labelling goes through `predictions.json`, which is how the suite already moves model output. |
| `class_id` on a box | `to_dict` hardcodes `0` and `tests/test_labeling_ops.py` asserts it. The Inspector says so rather than appearing to work. |
| Timeline drag-editing of behavior intervals | Edges move from the Inspector against the current frame. Dragging on the lane is the next increment. |
| Per-sample reason shown *on the timeline* | Reasons are in the Smart Annotate summary and in `Selection.reasons`. The timeline draws samples as one marker kind. |

---

## 10. Running the checks

```bash
# logic, on either interpreter
python -m pytest tests/test_annotation_core.py -q          # 110 tests

# GUI wiring, needs dearpygui + OpenCV
python -m pytest tests/test_gui_timeline.py tests/test_annotator_wiring.py -q

# design tokens and their contrast invariants
PYTHONPATH=src python -m htrtdetr.gui.theme --check
```

```bash
# the four user journeys, end to end (opens a window)
python -m pytest tests/test_workflows.py -q
python -m pytest tests/ -q -m "not gui"          # skip the windowed ones

# the optional backends, incl. a live helper subprocess
python -m pytest tests/test_gui_video_segment.py -q

# ...without loading a real checkpoint (skip the ~150 MB first-run download)
python -m pytest tests/ -q -m "not live_model"

# the sidecar's own protocol check, on either interpreter
python tools/sam_sidecar.py --selftest
```

**Do not export `PYTHONIOENCODING=utf-8` when running the suite.**
`tests/test_cli_smoke.py` spawns `python -m htrtdetr.cli --help` with
`capture_output=True, text=True`. The CLI prints a Japanese banner
(`GUI_DESIGN.md` §9.1); with that variable set the child writes UTF-8 while the parent
still decodes with the console codepage, the reader thread dies, and `r.stdout`
arrives as `None` — three tests then fail with a `TypeError` that has nothing to do
with the code under test.

---

## 11. The optional backends

Both are **capabilities the helper advertises**, never guesses from a package
name, and both leave YOAKE fully usable when absent.

### SAM 2 video propagation

**Installed and verified.** Two packages, into the Python that runs the helper —
not necessarily the one running the GUI:

```bash
pip install sam2 huggingface_hub          # or: pip install -e ".[sam2]"
```

`huggingface_hub` is listed because **sam2 does not depend on it and cannot
reach a checkpoint without it.** `SAM2VideoPredictor.from_pretrained` imports it
inside `build_sam._hf_download`, so `pip install sam2` alone installs a package
that raises `ModuleNotFoundError` on every load — with the package sitting
right there, which is the most confusing failure state available. Both
`Sam2VideoAdapter.available()` and `InterpreterInfo.backends` therefore require
the pair, and `locked_reason()` names the half-installed case specifically.

```
  select a box on frame N
      -> gui/video_segment.py exports the window as 00000.jpg upward
      -> sidecar op "propagate", FORWARD from the prompt
      -> sidecar op "propagate", BACKWARD from the prompt   (see below)
      -> tools/sam_sidecar.py: Sam2VideoAdapter, SAM2VideoPredictor
      -> a mask per frame -> click_segment.mask_to_obb, in the child
      -> five floats per frame come back over the pipe
      -> video_segment.merge_results: two passes -> one result
      -> Candidates, source="track", one track id
      -> one Tracklet, the quality checks, the Review Inbox
```

**Why two passes.** SAM 2 propagates in one direction from the prompt. A
forward pass covers the prompt frame and everything after it and produces
nothing before it, so a window built with `before` frames — the default is 100
each way — came back half empty while having been decoded, JPEG-exported and
paid for in full. Measured on a 30-frame window prompted at offset 15: forward
alone returns 15 boxes, backward alone 16, the merge 30. `merge_results` takes
the maximum `window` rather than the sum, because both passes run over the same
exported directory and summing would make `describe()` report a loss that never
happened. The prompt frame comes back from both; the forward result is passed
first and wins.

The second pass costs a second `init_state`, which re-encodes the window. That
is the price of the half that was previously discarded, and it is worth it.

**Which device.** `SidecarSpec.device` defaults to `"auto"` and the *helper*
resolves it, because only the helper has torch — the GUI cannot import it to
ask, which is the entire reason the helper exists. The answer comes back in
`hello` and is what `VideoCapability.describe()` shows. The old default was a
hardcoded `"cpu"`: measured over one 30-frame pass, **269 ms/frame on cuda
against 1153 ms/frame on cpu** — a factor of 4.3, which over the 200-frame
window the UI offers is 54 s against 231 s. It said nothing about why.

**Measured** on an RTX 5060 Ti with `facebook/sam2-hiera-tiny`, 256x256 frames,
against a synthetic scene with exact ground truth
(`tests/test_gui_video_segment.py`, `live_model`):

| | |
|---|---|
| Throughput | ~180 ms/frame, both prompt kinds |
| Rotated-box IoU | mean 0.88, worst 0.86, 100% above 0.7 |
| Centre error | 0.6 px mean, 1.1 px worst |
| Box size | 1.04-1.09x truth — consistently slightly loose |
| Review items raised on a clean track | 0 |

`sam2` builds an optional CUDA extension that is absent on this machine, so the
helper logs `cannot import name '_C' from 'sam2'` and skips one mask
post-processing step. It is a warning, not a failure, and the numbers above
were measured with it.

The window goes through the filesystem because `init_state` wants a directory
of JPEGs, that path works on every SAM 2 version, and the GUI has OpenCV for
certain while the helper's interpreter may not. Frames are **zero-padded**: the
helper sorts by name, and a plain sort puts `10.jpg` before `9.jpg`, which
would land every propagated box on the wrong frame, silently.

A propagation is bounded to `MAX_WINDOW = 600` frames. SAM 2 holds per-frame
state for the whole window, and — the reason that actually decides it — a human
cannot review 2,000 boxes as one decision anyway. **The unit of propagation is
the unit of review.**

Without `sam2`: `video_segment.capability()` returns a value whose `reason` is
the install sentence, the rail button is *locked* rather than hidden (P4), and
pressing it opens a modal that names the package and says what still works.

### Visual embeddings

`EmbeddingProvider` is a vector and a dimension, and nothing else — because
Smart Sampling needs a metric that separates different-looking frames, not a
classifier:

| Provider | When |
|---|---|
| `SidecarProvider` | the helper advertises `embedding`. DINOv2 via `transformers`, then `torch.hub`, then a torchvision ResNet-18 — whichever the torch interpreter actually has |
| `ThumbnailProvider` | always. The built-in 16x16 descriptor, 256-d, ~1 ms/frame |
| `NullProvider` | last resort; sampling runs on motion and scene statistics |

`provider_for()` picks the best that can run *right now*. `features.scan()`
takes `embeddings=` and falls through per frame, so a 384-d DINOv2 vector and a
256-d thumbnail are interchangeable and **swapping the encoder changes no
caller**. A batch that errors or times out gives up on the helper and returns
what it has; the run continues on the descriptor rather than being lost.

