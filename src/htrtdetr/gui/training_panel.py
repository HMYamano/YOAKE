"""
training_panel.py - Training workspace (Dear PyGui)
===================================================

Dataset preparation, staged training, checkpoint evaluation and offline output
analysis in one page, with a single shared run lifecycle (launch / cancel /
status pill / elapsed / exact command line / log) at the bottom.

設計方針
--------
* フォームは YOMU に倣い「必須フィールドの短いブロック + 折りたたみ Advanced」。
* 実行前バリデーションを必ず通す (欠損データセット・欠損 checkpoint・不正な
  epoch 数を、subprocess の stack trace ではなくインラインの英文で返す)。
* 起動前に pre-flight サマリをログへ出す。何が実際に走るのかを隠さない。
* torch が無い環境では実行系ボタンを ``widgets.locked`` で理由付きに封じ、
  データセット準備だけは動かせるままにする。

Every user-visible string in this file is English (see docs/GUI_DESIGN.md §4).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import dearpygui.dearpygui as dpg

from . import auto_label, commands, dialogs, shortcuts, theme, widgets
from .monitor import TrainingMonitor, metric_columns_for_stage, series
from .runner import CommandRunner

# ── layout constants ───────────────────────────────────────────────────────
_FORM_W = 620        # width of the left-hand form column
_WRAP = _FORM_W - 44  # text wrap inside the form column
_BOTTOM_H = 200      # height reserved for the shared run bar + log

_LOSS_SERIES: List[Tuple[str, str]] = [
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
]
_METRIC_SERIES: List[str] = ["ap50", "macro_f1", "idf1", "composite_score"]

#: Human labels for the metric series, so the legend is not raw column names.
_METRIC_LABELS: Dict[str, str] = {
    "ap50": "AP50",
    "macro_f1": "macro F1",
    "idf1": "IDF1",
    "composite_score": "composite",
}

STAGE_META: Dict[int, dict] = {
    1: {
        "label": "Stage 1 - Detector",
        "short": "1. Detector",
        "desc": (
            "Train the detector that finds animals in single frames. "
            "This is the starting point for the later temporal stages."
        ),
        "trains": "Single-frame detector",
        "needs": "Annotated frames",
        "produces": "weights/best.pth used by stages 2-4",
        "epochs": 50,
        "batch": 4,
        "lr": 1e-4,
        "model_default": "detection",
        "needs_window": False,
        "needs_imbalance": False,
    },
    2: {
        "label": "Stage 2 - Action Head",
        "short": "2. Action head",
        "desc": (
            "Train the temporal action head on top of the detector. "
            "This stage uses a frame window to classify behavior over time."
        ),
        "trains": "Temporal action head",
        "needs": "Stage 1 detector + action labels",
        "produces": "Behavior classification",
        "epochs": 30,
        "batch": 8,
        "lr": 5e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": True,
    },
    3: {
        "label": "Stage 3 - ID Head",
        "short": "3. ID head",
        "desc": (
            "Train the temporal ID head so the same animal keeps a consistent "
            "track ID across frames."
        ),
        "trains": "Temporal ID head",
        "needs": "Stage 1 detector + track IDs",
        "produces": "Stable track IDs",
        "epochs": 30,
        "batch": 8,
        "lr": 5e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": False,
    },
    4: {
        "label": "Stage 4 - Unified Fine-Tuning",
        "short": "4. Fine-tune",
        "desc": (
            "Fine-tune the full model with all major modules enabled at a lower "
            "learning rate."
        ),
        "trains": "Everything, jointly",
        "needs": "Stages 1-3 finished",
        "produces": "The checkpoint you ship",
        "epochs": 20,
        "batch": 2,
        "lr": 1e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": False,
    },
}

_STAGES: Tuple[int, ...] = (1, 2, 3, 4)
_STAGE_LABELS: List[str] = [STAGE_META[s]["label"] for s in _STAGES]
_LABEL_TO_STAGE: Dict[str, int] = {STAGE_META[s]["label"]: s for s in _STAGES}

# widgets.set_segmented_value() matches a segment by its LABEL, so the stage
# picker's option value has to equal its button label or a programmatic move
# would silently un-highlight every button.
_STAGE_BUTTON: Dict[int, str] = {s: STAGE_META[s]["short"] for s in _STAGES}
_BUTTON_STAGE: Dict[str, int] = {v: k for k, v in _STAGE_BUTTON.items()}

_MODEL_OPTIONS = ["detection", "detection+temporal"]
_FORMAT_OPTIONS = ["json", "yolo"]

#: tools/convert_annotations.py reads ``mode=``; these are the modes it accepts.
_CONVERT_MODES = ["csv", "json", "video_csv"]

#: tools/build_splits.py reads ``strategy=``.
_SPLIT_STRATEGIES = ["video", "track", "frame"]

_BBOX_FORMATS = ["xyxy", "xywh", "cxcywh"]

_ANALYZE_MODES = ["timeline", "distribution", "id_switches", "confidence"]

_ANALYZE_DESCS: Dict[str, str] = {
    "timeline": (
        "Plot the frame-by-frame prediction timeline. Needs a predictions JSON."
    ),
    "distribution": (
        "Summarize class and action distributions in labelled data. Needs an "
        "annotation JSON."
    ),
    "id_switches": (
        "Count identity switches across tracks. Needs a predictions JSON."
    ),
    "confidence": (
        "Inspect the score distribution of a run. Needs a predictions JSON. "
        "The detector's confidence is not a calibrated probability."
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════
def _quote(token: str) -> str:
    """Quote a command token so the copied line can be pasted into a shell."""
    text = str(token)
    if not text:
        return '""'
    return f'"{text}"' if any(c in text for c in ' \t"') else text


def _command_line(cmd: Sequence) -> str:
    """The exact line that :class:`CommandRunner` will execute."""
    return " ".join(_quote(c) for c in [sys.executable, *cmd])


def _fmt_elapsed(seconds: float) -> str:
    """``0:42`` / ``12:05`` / ``1:02:33``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _exists(path: str) -> bool:
    if not path:
        return False
    try:
        return Path(path).expanduser().exists()
    except (OSError, ValueError):
        return False


def _short_path(path: str, keep: int = 3) -> str:
    """Last ``keep`` components of a path, for a one-line status label.

    The full path always stays available in the field above and in the
    tooltip; a run folder nested ten levels deep must not push the plots off
    the bottom of the panel.
    """
    text = str(path or "")
    try:
        parts = Path(text).parts
    except (OSError, ValueError):
        return text
    if len(parts) <= keep:
        return text
    return "..." + str(Path(*parts[-keep:]))


def _mark(path: str) -> str:
    """``[exists]`` / ``[missing]`` / ``[not set]`` for the pre-flight block."""
    if not path:
        return "[not set]"
    return "[exists]" if _exists(path) else "[missing]"


def _value(tag: str, default="") -> str:
    """``dpg.get_value`` that tolerates a tag that has not been built yet."""
    if not dpg.does_item_exist(tag):
        return default
    value = dpg.get_value(tag)
    return default if value is None else value


def _text(tag: str, default: str = "") -> str:
    return str(_value(tag, default)).strip()


def _int(tag: str, default: int) -> int:
    try:
        return int(_value(tag, default))
    except (TypeError, ValueError):
        return default


def _float(tag: str, default: float) -> float:
    try:
        return _clean_float(float(_value(tag, default)))
    except (TypeError, ValueError):
        return default


def _clean_float(value: float, digits: int = 7) -> float:
    """Undo the float32 rounding a dearpygui float input applies.

    ``add_input_float`` stores single precision, so a learning rate typed as
    0.0001 reads back as 9.999999747378752e-05 and would be printed verbatim
    into the command line and the log. float32 carries about seven decimal
    digits, so rounding to seven significant figures is lossless and readable.
    """
    try:
        return float(f"{float(value):.{digits}g}")
    except (TypeError, ValueError, OverflowError):
        return float(value)


class _PanelRunner(CommandRunner):
    """:class:`CommandRunner` whose log can be cleared and copied.

    ``CommandRunner`` keeps the rendered log in ``_lines``; a subclass is the
    honest way to reach it rather than poking at another module's privates.
    """

    def clear(self) -> None:
        with self._lock:
            self._lines = []
        if dpg.does_item_exist(self.log_tag):
            dpg.set_value(self.log_tag, "")

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


# ══════════════════════════════════════════════════════════════════════════
# the panel
# ══════════════════════════════════════════════════════════════════════════
class TrainingPanel:
    """Training page embedded inside the GUI hub.

    Public API (unchanged): ``TrainingPanel(root)``, ``build(parent)``,
    ``teardown()``. ``tick()`` is new and optional - the host may call it once
    per render-loop iteration to refresh the elapsed clock from the main
    thread; when it is never called an internal clock thread does the same job.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tools = self.root / "tools"
        self.runs = self.root / "runs"
        # status_tag is deliberately empty: this panel owns "train_status" so a
        # user-requested stop is reported as a stop, never as "Error (code=1)".
        self.runner = _PanelRunner("train_log", "", cwd=self.root)
        self.monitor: Optional[TrainingMonitor] = None
        self.monitor_stage = 1
        self.monitor_dir = ""
        self.current_stage = 1

        #: Optional hook the hub may set to mirror status into its status bar:
        #: ``panel.on_status = hub.status``.
        self.on_status: Optional[Callable[[str], None]] = None

        # run lifecycle
        self._job_label = ""
        self._job_started = 0.0
        self._stop_requested = False
        self._clock_stop = threading.Event()
        self._clock: Optional[threading.Thread] = None

        # Every subprocess this panel spawns inherits os.environ (gui/runner.py
        # passes no env=), so the src layout has to be on PYTHONPATH before the
        # first launch or every button fails with ModuleNotFoundError. This is
        # idempotent and the hub calls it too.
        try:
            auto_label.ensure_child_env(self.root)
        except OSError:
            pass
        self._cap = auto_label.probe(self.root)

    # ─────────────────────────────────────────────────────────────────────
    # build
    # ─────────────────────────────────────────────────────────────────────
    def build(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_training",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=False,
        ):
            widgets.page_title(
                "Train",
                "Prepare a dataset, train each stage in order, evaluate a "
                "checkpoint, and watch the metrics update while it runs.",
                wrap=980,
            )
            self._build_capability_banner()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=_FORM_W, height=-_BOTTOM_H,
                                      border=False):
                    with dpg.tab_bar(tag="train_tabs"):
                        self._tab_dataset()
                        self._tab_training()
                        self._tab_evaluate()
                        self._tab_analyze()
                with dpg.child_window(width=-1, height=-_BOTTOM_H, border=True):
                    self._build_monitor()

            self._build_run_bar()

        self._apply_capability_locks()
        self._set_idle()

    # ─── capability banner ───────────────────────────────────────────────
    def _capability_problems(self) -> List[Tuple[str, str, str]]:
        """``[(headline, detail, fix command), ...]`` - empty when all is well."""
        problems: List[Tuple[str, str, str]] = []
        if not self._cap.torch:
            problems.append((
                "PyTorch is not installed in this Python environment.",
                "Training, evaluation and analysis cannot run here. Dataset "
                "preparation and the live monitor still work.",
                "pip install torch torchvision",
            ))
        if self._cap.package_reason:
            problems.append((
                "YOAKE is not importable in this Python environment.",
                "Every command on this page runs as a subprocess, so they "
                "would all fail to start.",
                "pip install -e .",
            ))
        return problems

    def _build_capability_banner(self) -> None:
        problems = self._capability_problems()
        if not problems:
            return
        dpg.add_separator()
        for headline, detail, fix in problems:
            with dpg.group(horizontal=True):
                dpg.add_text("●", color=theme.PENDING)
                dpg.add_text(headline, color=theme.PENDING)
                dpg.add_spacer(width=theme.SP_M)
                dpg.add_input_text(default_value=fix, readonly=True, width=280)
                theme.bind(dpg.last_item(), theme.mono_field_theme())
                dpg.add_button(
                    label="Copy", width=62, height=22,
                    callback=(lambda s, a, u, text=fix:
                              self._copy(text, "Install command copied.")),
                )
                theme.bind(dpg.last_item(), theme.ghost_theme())
            dpg.add_text(detail, color=theme.TEXT_DIM, wrap=940, indent=18)
        dpg.add_separator()
        widgets.gap(0)

    def _apply_capability_locks(self) -> None:
        """Gate the three torch-dependent launchers with a reason, not a crash."""
        reason = ""
        if self._cap.package_reason:
            reason = self._cap.package_reason
        elif not self._cap.torch:
            reason = self._cap.torch_reason
        for tag in ("tr_start", "tr_eval_run", "an_run"):
            if not dpg.does_item_exist(tag):
                continue
            if reason:
                widgets.locked(tag, reason)
            else:
                widgets.unlock(tag)
                theme.bind(tag, theme.accent_theme())

    # ─────────────────────────────────────────────────────────────────────
    # shared helpers used by the tabs
    # ─────────────────────────────────────────────────────────────────────
    def _field(self, label, tag, default="", browse=None, width=390, *,
               field_key=None, required=False, tooltip_text=None, hint=""):
        """Backwards-compatible wrapper around :func:`widgets.path_field`."""
        widgets.path_field(
            label, tag,
            field_key=field_key,
            default=default,
            hint=hint,
            browse=browse,
            required=required,
            tooltip_text=tooltip_text,
            width=width,
        )

    def _pick_json(self, _tag=None):
        return lambda: dialogs.pick_file("Select JSON or data file", dialogs.JSON_TYPES)

    def _pick_csv(self, _tag=None):
        return lambda: dialogs.pick_file(
            "Select CSV file", [("CSV", "*.csv"), ("All", "*.*")])

    def _pick_video(self, _tag=None):
        return lambda: dialogs.pick_file("Select video", dialogs.VIDEO_TYPES)

    def _pick_ckpt(self, _tag=None):
        return lambda: dialogs.pick_file("Select checkpoint", dialogs.CKPT_TYPES)

    def _pick_dir(self, _tag=None):
        return lambda: dialogs.pick_dir("Select folder")

    def _note(self, text: str, wrap: int = _WRAP) -> None:
        dpg.add_text(text, color=theme.TEXT_DIM, wrap=wrap)

    # ═════════════════════════════════════════════════════════════════════
    # Tab 1 - Dataset
    # ═════════════════════════════════════════════════════════════════════
    def _tab_dataset(self):
        with dpg.tab(label="Dataset", tag="train_tab_dataset"):
            widgets.gap(1)
            self._note(
                "Three tools, in order. Open the one you need - they are plain "
                "Python and run without PyTorch."
            )

            widgets.gap(1)

            # Three collapsed steps rather than one long scroll. All three
            # headers fit on screen together, so the order is visible at a
            # glance and a finished step folds back out of the way.
            with dpg.collapsing_header(label="1. Convert annotations",
                                       tag="ds_sec_conv", default_open=False):
                self._note("Turn an exported label file into the annotation "
                           "JSON the trainer reads.")
                widgets.labeled_input(
                    "Source format", "ds_conv_fmt", kind="combo",
                    items=_CONVERT_MODES, default="csv", width=150,
                    callback=lambda s, a, u: self._on_convert_mode_change(a),
                )
                widgets.attach_tooltip(
                    "ds_conv_fmt",
                    "csv = a flat table from the YOAKE annotator.\n"
                    "json = another tool's JSON.\n"
                    "video_csv = a video plus a CSV, frames extracted on the "
                    "way.",
                    wrap=320,
                )
                self._field(
                    "Input file", "ds_conv_in",
                    field_key="annotation_source",
                    required=True,
                    browse=self._pick_json("ds_conv_in"),
                    tooltip_text=(
                        "The file named by 'input=' on the tool's command line. "
                        "For video_csv this is the video, and the labels go in the "
                        "CSV field below."
                    ),
                )
                dpg.add_text("", tag="ds_conv_in_note", color=theme.TEXT_FAINT,
                             wrap=_WRAP, indent=18)
                with dpg.group(tag="ds_conv_csv_row", show=False):
                    self._field(
                        "CSV file (video_csv only)", "ds_conv_csv",
                        field_key="annotation_source",
                        browse=self._pick_csv("ds_conv_csv"),
                        tooltip_text="Required when the source format is video_csv.",
                    )
                self._field(
                    "Output JSON", "ds_conv_out",
                    field_key="annotation_json",
                    required=True,
                    browse=lambda: dialogs.save_file(),
                    tooltip_text=(
                        "Where the converted annotation JSON is written. A "
                        "label_map.json is written next to it."
                    ),
                )
                with widgets.advanced_section(
                    "Advanced: frame geometry", tag="ds_conv_adv",
                    description=(
                        "Only needed when the source file stores normalized or "
                        "differently shaped boxes. The defaults match the YOAKE "
                        "annotator's export."
                    ),
                ):
                    widgets.labeled_input(
                        "Frame width (px)", "ds_conv_w", default=640, width=110,
                        help_text="Used to denormalize boxes. (default 640)",
                    )
                    widgets.labeled_input(
                        "Frame height (px)", "ds_conv_h", default=640, width=110,
                        help_text="(default 640)",
                    )
                    widgets.labeled_input(
                        "Frame rate (fps)", "ds_conv_fps", kind="float",
                        default=30.0, width=110,
                        help_text="Recorded in the output so temporal stages can "
                                  "reason about time. (default 30.0)",
                    )
                    # labeled_input has no format argument and dearpygui defaults
                    # float inputs to "%.3f"; fps reads better with one decimal.
                    dpg.configure_item("ds_conv_fps", format="%.1f")
                    widgets.labeled_input(
                        "Box format", "ds_conv_bbox", kind="combo",
                        items=_BBOX_FORMATS, default="xyxy", width=130,
                        help_text="How the source file stores boxes. (default xyxy)",
                    )
                widgets.gap(1)
                widgets.primary_button(
                    "Convert annotations", self._start_convert,
                    height=32, width=260, tag="ds_conv_run",
                )
                dpg.add_text("", tag="ds_conv_problems", color=theme.TEXT_DIM,
                             wrap=_WRAP)

            with dpg.collapsing_header(
                    label="2. Create train / validation / test splits",
                    tag="ds_sec_split", default_open=False):
                self._note("Split one annotation JSON into three, by video so "
                           "the same animal cannot appear on both sides.")
                self._field(
                    "Annotation JSON", "ds_split_anno",
                    field_key="annotation_json",
                    required=True,
                    browse=self._pick_json("ds_split_anno"),
                    tooltip_text="The converted annotation JSON to split.",
                )
                self._field(
                    "Output folder", "ds_split_out",
                    field_key="split_dir",
                    required=True,
                    browse=self._pick_dir("ds_split_out"),
                    tooltip_text=(
                        "Receives annotations_train.json, annotations_val.json, "
                        "annotations_test.json and split_summary.json. It is "
                        "created if it does not exist yet."
                    ),
                )
                widgets.labeled_slider(
                    "Validation ratio", "ds_val_ratio", default=0.15,
                    min_value=0.0, max_value=0.45, width=200, fmt="%.2f",
                    callback=lambda s, a, u: self._refresh_split_readout(),
                    help_text="Fraction held out for validation. (default 0.15)",
                )
                widgets.labeled_slider(
                    "Test ratio", "ds_test_ratio", default=0.15,
                    min_value=0.0, max_value=0.45, width=200, fmt="%.2f",
                    callback=lambda s, a, u: self._refresh_split_readout(),
                    help_text=(
                        "Fraction held out for the final test set. Training gets "
                        "whatever is left. (default 0.15)"
                    ),
                )
                dpg.add_text("", tag="ds_ratio_readout", color=theme.ACCENT,
                             indent=4)
                with widgets.advanced_section(
                    "Advanced: split strategy", tag="ds_split_adv",
                    description=(
                        "Change these only when you know the recording layout. "
                        "A frame-level split leaks the same animal into both sides."
                    ),
                ):
                    widgets.labeled_input(
                        "Split by", "ds_split_strategy", kind="combo",
                        items=_SPLIT_STRATEGIES, default="video", width=130,
                        help_text=(
                            "video = whole recordings stay together (recommended). "
                            "track = one animal stays together. "
                            "frame = per frame, which leaks labels."
                        ),
                    )
                    widgets.labeled_input(
                        "Random seed", "ds_split_seed", default=42, width=110,
                        help_text="Same seed = same split. (default 42)",
                    )
                widgets.gap(1)
                widgets.primary_button(
                    "Create splits", self._start_split,
                    height=32, width=260, tag="ds_split_run",
                )
                dpg.add_text("", tag="ds_split_problems", color=theme.TEXT_DIM,
                             wrap=_WRAP)

            with dpg.collapsing_header(label="3. Review dataset statistics",
                                       tag="ds_sec_stats", default_open=False):
                self._note("Counts, action distribution and track lengths. "
                           "Read this before choosing an imbalance strategy "
                           "in stage 2.")
                self._field(
                    "Annotation JSON", "ds_stat_anno",
                    field_key="annotation_json",
                    required=True,
                    browse=self._pick_json("ds_stat_anno"),
                    tooltip_text="Any annotation JSON, split or not.",
                )
                with widgets.advanced_section(
                    "Advanced: output", tag="ds_stat_adv",
                ):
                    self._field(
                        "Figure folder", "ds_stat_out",
                        field_key="dataset_summary_dir",
                        default=str(self.root / "outputs" / "dataset_summary"),
                        browse=self._pick_dir("ds_stat_out"),
                        tooltip_text=("Receives the summary JSON and the plots. "
                                      "It is created if it does not exist yet."),
                    )
                    widgets.labeled_input(
                        "Draw plots", "ds_stat_plots", kind="checkbox", default=True,
                        help_text="Off prints the numbers only, which is faster.",
                    )
                widgets.gap(1)
                widgets.secondary_button(
                    "Show statistics", self._start_stats,
                    height=30, width=220, tag="ds_stat_run",
                )
                dpg.add_text("", tag="ds_stat_problems", color=theme.TEXT_DIM,
                             wrap=_WRAP)

            widgets.gap(2)
            self._note(
                "Labeling lives in step 1. Open it with "
                f"{shortcuts.label_for('page.label', scope='hub') or 'the left navigation'}."
            )
            widgets.gap(2)

        self._on_convert_mode_change("csv")
        self._refresh_split_readout()

    def _on_convert_mode_change(self, mode: str) -> None:
        notes = {
            "csv": "input= is the CSV exported by the YOAKE annotator.",
            "json": "input= is another tool's JSON. The generic schema is assumed.",
            "video_csv": (
                "input= is the video file. Frames are extracted next to the "
                "output, and the labels come from the CSV field below."
            ),
        }
        if dpg.does_item_exist("ds_conv_in_note"):
            dpg.set_value("ds_conv_in_note", notes.get(mode, ""))
        if dpg.does_item_exist("ds_conv_csv_row"):
            dpg.configure_item("ds_conv_csv_row", show=(mode == "video_csv"))

    def _refresh_split_readout(self) -> None:
        val = max(0.0, min(0.45, _float("ds_val_ratio", 0.15)))
        test = max(0.0, min(0.45, _float("ds_test_ratio", 0.15)))
        train = max(0.0, 1.0 - val - test)
        if dpg.does_item_exist("ds_ratio_readout"):
            dpg.set_value(
                "ds_ratio_readout",
                f"train {train * 100:.0f}% / val {val * 100:.0f}% / "
                f"test {test * 100:.0f}%",
            )

    # ═════════════════════════════════════════════════════════════════════
    # Tab 2 - Train
    # ═════════════════════════════════════════════════════════════════════
    def _tab_training(self):
        with dpg.tab(label="Train", tag="train_tab_train"):
            widgets.gap(1)
            self._note(
                "The four stages run in order. Each one starts from the "
                "checkpoint the previous one wrote, so train 1 before 2, and "
                "2 and 3 before 4."
            )
            widgets.gap(1)
            dpg.add_text("Stage to train:", color=theme.TEXT, indent=2)
            widgets.segmented_control(
                "tr_stage_seg",
                [_STAGE_BUTTON[s] for s in _STAGES],
                lambda value: self._on_stage_change(
                    _BUTTON_STAGE.get(value, 1)),
                default=_STAGE_BUTTON[1], width=136, height=28,
            )
            dpg.add_text("", tag="tr_stage_desc", color=theme.TEXT_DIM,
                         wrap=_WRAP)
            dpg.add_button(
                label="Restore stage defaults", height=26, width=200,
                tag="tr_restore",
                callback=lambda: self._on_stage_change(self.current_stage,
                                                       reset=True),
            )
            theme.bind("tr_restore", theme.ghost_theme())
            widgets.tooltip(
                "Put epochs, batch size, learning rate, window size and the "
                "output folder back to this stage's defaults.",
                wrap=280,
            )
            with widgets.advanced_section(
                "How the four stages fit together", tag="tr_stage_help",
                description="What each stage trains and what it needs first.",
            ):
                self._build_stage_table()
                widgets.gap(1)

            # ── required ─────────────────────────────────────────────────
            widgets.section_header(
                "Required",
                "Two paths and a stage. Everything else has a stage default.",
                wrap=_WRAP,
            )
            self._field(
                "Training data", "tr_train",
                field_key="train_data",
                required=True,
                browse=self._pick_json("tr_train"),
                tooltip_text=(
                    "annotations_train.json from 'Create splits', or a YOLO "
                    "dataset root when the annotation format is yolo."
                ),
            )
            self._field(
                "Validation data", "tr_val",
                field_key="val_data",
                browse=self._pick_json("tr_val"),
                tooltip_text=(
                    "Optional but strongly recommended: without it there is no "
                    "validation loss curve and no early stopping signal."
                ),
            )
            self._field(
                "Run output folder", "tr_out",
                field_key="train_output_dir",
                required=True,
                default=str(self.runs / "train" / "stage1"),
                browse=self._pick_dir("tr_out"),
                tooltip_text=(
                    "Receives checkpoints, results.csv and metrics_latest.json. "
                    "Each stage writes to its own folder and is never "
                    "overwritten by another stage. It is created if it does "
                    "not exist yet."
                ),
            )
            widgets.gap(1)
            dpg.add_text("", tag="tr_hp_summary", color=theme.ACCENT,
                         wrap=_WRAP, indent=4)
            self._note(
                "Change any of those under Advanced below.", wrap=_WRAP,
            )

            # ── advanced ─────────────────────────────────────────────────
            with widgets.advanced_section(
                "Advanced: data format", tag="tr_adv_data",
                description=(
                    "The model mode decides which stage actually runs. Picking "
                    "'detection' forces stage 1 whatever the stage buttons say, "
                    "and the pre-flight block in the log says so when it "
                    "happens."
                ),
            ):
                widgets.labeled_input(
                    "Annotation format", "tr_fmt", kind="combo",
                    items=_FORMAT_OPTIONS, default="json", width=130,
                    help_text=(
                        "json = YOAKE annotation JSON. "
                        "yolo = a YOLO dataset root with images/ and labels/."
                    ),
                )
                widgets.labeled_input(
                    "Model mode", "tr_model", kind="combo",
                    items=_MODEL_OPTIONS,
                    default=STAGE_META[1]["model_default"], width=200,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text=(
                        "detection = single-frame detector only. "
                        "detection+temporal = the full temporal model."
                    ),
                )
                self._field(
                    "Class names file (yolo only)", "tr_classes",
                    field_key="classes_file",
                    browse=self._pick_json("tr_classes"),
                    tooltip_text=(
                        "Text file listing class names, one per line. Ignored "
                        "when the annotation format is json."
                    ),
                )

            with widgets.advanced_section(
                "Advanced: hyperparameters", tag="tr_adv_hp",
                description=(
                    "The stage defaults are a reasonable starting point. Change "
                    "the batch size first if the run stops for GPU memory."
                ),
            ):
                widgets.labeled_input(
                    "Epochs", "tr_epochs", default=STAGE_META[1]["epochs"],
                    width=100,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text="Full passes over the training data. Must be 1 or "
                              "more.",
                )
                widgets.labeled_input(
                    "Batch size", "tr_batch", default=STAGE_META[1]["batch"],
                    width=100,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text="Samples per gradient step. Lower it if the run "
                              "runs out of GPU memory.",
                )
                widgets.labeled_input(
                    "Learning rate", "tr_lr", kind="float",
                    default=STAGE_META[1]["lr"], width=130,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text="Must be greater than 0 and 1.0 or less. Stage "
                              "defaults are between 0.00001 and 0.0001.",
                )
                # Dearpygui defaults float inputs to "%.3f", which renders every
                # stage default as "0.000". Eight decimals shows all of them.
                dpg.configure_item("tr_lr", format="%.8f")
                widgets.labeled_input(
                    "Window size (frames)", "tr_window", default=16, width=100,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text="Consecutive frames fed to the temporal head. "
                              "Stage 1 ignores it. (default 16)",
                )
                widgets.labeled_input(
                    "Imbalance strategy", "tr_imb", kind="combo",
                    items=["none", "class_weight", "focal"],
                    default="class_weight", width=160,
                    callback=lambda s, a, u: self._refresh_hp_summary(),
                    help_text=(
                        "How the action loss handles rare behaviors. "
                        "class_weight = inverse-frequency weights (safe "
                        "default). focal = for very skewed data. none = no "
                        "reweighting. Used by stage 2 only."
                    ),
                )

            # ── launch ───────────────────────────────────────────────────
            widgets.section_header("Start the run", wrap=_WRAP)
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Start training", self._start_train,
                    height=38, width=210, tag="tr_start",
                )
                dpg.add_spacer(width=theme.SP_S)
                dpg.add_button(
                    label="Check settings", height=34, width=150,
                    tag="tr_check",
                    callback=lambda: self._report_train_validation(quiet=False),
                )
                theme.bind(dpg.last_item(), theme.ghost_theme())
                widgets.tooltip(
                    "Validate the paths and numbers without launching "
                    "anything.",
                    wrap=260,
                )
                dpg.add_spacer(width=theme.SP_S)
                dpg.add_button(
                    label="Watch this run folder", height=34, width=190,
                    tag="tr_watch",
                    callback=lambda: self.attach_monitor(
                        _text("tr_out"), self.current_stage),
                )
                theme.bind(dpg.last_item(), theme.ghost_theme())
                widgets.tooltip(
                    "Point the live monitor at the output folder above so you "
                    "can watch an existing run without starting a new one.",
                    wrap=280,
                )
            widgets.gap(1)
            dpg.add_text("", tag="tr_problems", color=theme.TEXT_DIM,
                         wrap=_WRAP)
            widgets.gap(2)

            self._on_stage_change(1, reset=True)

    def _build_stage_table(self) -> None:
        with dpg.table(tag="tr_stage_table", header_row=True,
                       borders_innerH=False, borders_outerH=True,
                       borders_innerV=False, borders_outerV=True,
                       policy=dpg.mvTable_SizingStretchProp, width=-1):
            dpg.add_table_column(label="Stage", init_width_or_weight=0.9)
            dpg.add_table_column(label="Trains", init_width_or_weight=1.1)
            dpg.add_table_column(label="Needs", init_width_or_weight=1.4)
            dpg.add_table_column(label="Epochs", init_width_or_weight=0.5)
            for stage in _STAGES:
                meta = STAGE_META[stage]
                with dpg.table_row():
                    dpg.add_text(meta["short"], tag=f"tr_stage_row_{stage}",
                                 color=theme.TEXT)
                    dpg.add_text(meta["trains"], color=theme.TEXT_DIM)
                    dpg.add_text(meta["needs"], color=theme.TEXT_DIM)
                    dpg.add_text(str(meta["epochs"]), color=theme.TEXT_DIM)

    def _refresh_stage_table(self) -> None:
        for stage in _STAGES:
            tag = f"tr_stage_row_{stage}"
            if dpg.does_item_exist(tag):
                dpg.configure_item(
                    tag,
                    color=theme.ACCENT if stage == self.current_stage
                    else theme.TEXT,
                )
        if not dpg.does_item_exist("tr_stage_table"):
            return
        for idx in range(len(_STAGES)):
            try:
                if idx == self.current_stage - 1:
                    dpg.highlight_table_row("tr_stage_table", idx,
                                            theme.rgba(theme.ROW_ACTIVE, 110))
                else:
                    dpg.unhighlight_table_row("tr_stage_table", idx)
            except (SystemError, TypeError, ValueError):
                return

    def _on_stage_change(self, stage: int, *, reset: bool = True) -> None:
        stage = stage if stage in STAGE_META else 1
        meta = STAGE_META[stage]
        self.current_stage = stage
        widgets.set_segmented_value("tr_stage_seg", _STAGE_BUTTON[stage])

        if dpg.does_item_exist("tr_stage_desc"):
            dpg.set_value(
                "tr_stage_desc",
                f"{meta['desc']}  Needs: {meta['needs']}. "
                f"Produces: {meta['produces']}.",
            )
        if reset:
            for tag, value in (
                ("tr_epochs", meta["epochs"]),
                ("tr_batch", meta["batch"]),
                ("tr_lr", meta["lr"]),
                ("tr_model", meta["model_default"]),
                ("tr_window", 16),
            ):
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, value)
            if dpg.does_item_exist("tr_out"):
                dpg.set_value("tr_out", str(self.runs / "train" / f"stage{stage}"))
            if dpg.does_item_exist("tr_eval_out"):
                dpg.set_value("tr_eval_out", str(self.runs / "val" / f"stage{stage}"))
        if dpg.does_item_exist("tr_window"):
            dpg.configure_item("tr_window", enabled=meta["needs_window"])
        if dpg.does_item_exist("tr_imb"):
            dpg.configure_item("tr_imb", enabled=meta["needs_imbalance"])
        widgets.refresh_all_indicators(["tr_out", "tr_eval_out"])
        self._refresh_stage_table()
        self._refresh_hp_summary()
        self._report_train_validation(quiet=True)

    def _refresh_hp_summary(self) -> None:
        if not dpg.does_item_exist("tr_hp_summary"):
            return
        meta = STAGE_META[self.current_stage]
        window = _int("tr_window", 16) if meta["needs_window"] else None
        imb = _text("tr_imb", "class_weight") if meta["needs_imbalance"] else None
        parts = [
            meta["label"],
            f"{_int('tr_epochs', meta['epochs'])} epochs",
            f"batch {_int('tr_batch', meta['batch'])}",
            f"lr {_float('tr_lr', meta['lr']):.6f}",
        ]
        if window is not None:
            parts.append(f"window {window}")
        if imb:
            parts.append(f"imbalance {imb}")
        dpg.set_value("tr_hp_summary", "  |  ".join(parts))

    # ═════════════════════════════════════════════════════════════════════
    # Tab 3 - Evaluate
    # ═════════════════════════════════════════════════════════════════════
    def _tab_evaluate(self):
        with dpg.tab(label="Evaluate", tag="train_tab_eval"):
            widgets.gap(1)
            self._note(
                "Score a finished checkpoint on held-out data. This does not "
                "modify the run it came from."
            )
            widgets.section_header("Required", wrap=_WRAP)
            self._field(
                "Checkpoint", "tr_ckpt",
                field_key="checkpoint",
                required=True,
                browse=self._pick_ckpt("tr_ckpt"),
                tooltip_text=(
                    "The .pth or .pt file a training run wrote, usually "
                    "weights/best.pth inside the run output folder."
                ),
            )
            self._field(
                "Validation data", "tr_eval_val",
                field_key="val_data",
                required=True,
                browse=self._pick_json("tr_eval_val"),
                tooltip_text=(
                    "annotations_val.json or annotations_test.json from "
                    "'Create splits'."
                ),
            )
            self._field(
                "Evaluation output folder", "tr_eval_out",
                field_key="eval_output_dir",
                default=str(self.runs / "val" / "stage1"),
                browse=self._pick_dir("tr_eval_out"),
                tooltip_text=(
                    "Receives the metrics JSON and the plots. The source run "
                    "is not modified. It is created if it does not exist yet."
                ),
            )
            widgets.gap(1)
            self._note(
                "The stage selected on the Train tab decides which metrics are "
                "computed.", wrap=_WRAP,
            )
            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Run evaluation", self._start_eval,
                    height=34, width=210, tag="tr_eval_run",
                )
                dpg.add_spacer(width=theme.SP_S)
                dpg.add_button(
                    label="Check settings", height=32, width=150,
                    tag="tr_eval_check",
                    callback=lambda: self._report_eval_validation(quiet=False),
                )
                theme.bind(dpg.last_item(), theme.ghost_theme())
            widgets.gap(1)
            dpg.add_text("", tag="tr_eval_problems", color=theme.TEXT_DIM,
                         wrap=_WRAP)
            widgets.gap(2)

    # ═════════════════════════════════════════════════════════════════════
    # Tab 4 - Analyze
    # ═════════════════════════════════════════════════════════════════════
    def _tab_analyze(self):
        with dpg.tab(label="Analyze", tag="train_tab_analyze"):
            widgets.gap(1)
            self._note(
                "Quick offline analysis of a predictions file or of labelled "
                "data, without leaving this page. Frame-by-frame review lives "
                "in step 3."
            )
            widgets.section_header("Required", wrap=_WRAP)
            widgets.labeled_input(
                "Mode", "an_mode", kind="combo", items=_ANALYZE_MODES,
                default="timeline", width=180,
                callback=lambda s, a, u: self._on_analyze_mode_change(a),
                help_text="The required input changes with the mode.",
            )
            dpg.add_text("", tag="an_mode_desc", color=theme.TEXT_DIM,
                         wrap=_WRAP, indent=4)
            widgets.gap(1)
            self._field(
                "Predictions JSON", "an_pred",
                field_key="predictions_json",
                browse=self._pick_json("an_pred"),
                tooltip_text=(
                    "predictions.json written by 'yoake predict' or by the "
                    "Analyze page."
                ),
            )
            self._field(
                "Annotation JSON", "an_anno",
                field_key="annotation_json",
                browse=self._pick_json("an_anno"),
                tooltip_text="Labelled data. Required by the distribution mode.",
            )
            self._field(
                "Output folder", "an_out",
                field_key="analyze_output_dir",
                default=str(self.runs / "analyze"),
                browse=self._pick_dir("an_out"),
                tooltip_text=("Receives the figures and the analysis JSON. "
                              "It is created if it does not exist yet."),
            )
            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Run analysis", self._start_analyze,
                    height=34, width=200, tag="an_run",
                )
                dpg.add_spacer(width=theme.SP_S)
                dpg.add_button(
                    label="Check settings", height=32, width=150,
                    tag="an_check",
                    callback=lambda: self._report_analyze_validation(quiet=False),
                )
                theme.bind(dpg.last_item(), theme.ghost_theme())
            widgets.gap(1)
            dpg.add_text("", tag="an_problems", color=theme.TEXT_DIM,
                         wrap=_WRAP)
            widgets.gap(2)

            self._on_analyze_mode_change("timeline")

    def _on_analyze_mode_change(self, mode: str) -> None:
        if dpg.does_item_exist("an_mode_desc"):
            dpg.set_value("an_mode_desc", _ANALYZE_DESCS.get(mode, ""))
        self._report_analyze_validation(quiet=True)

    # ═════════════════════════════════════════════════════════════════════
    # live monitor
    # ═════════════════════════════════════════════════════════════════════
    def _build_monitor(self):
        dpg.add_text("Live run monitor", color=theme.ACCENT)
        theme.bind_h2(dpg.last_item())
        dpg.add_separator()
        dpg.add_text(
            "Polls a run folder every two seconds. Reading only - nothing is "
            "started or modified.",
            color=theme.TEXT_DIM, wrap=460,
        )
        widgets.gap(0)

        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="mon_dir", width=-190,
                default_value=str(self.runs / "train" / "stage1"),
                hint="Run folder, for example runs/train/stage1",
            )
            dpg.add_button(label="Browse", width=70, height=24,
                           callback=self._browse_monitor_dir)
            theme.bind(dpg.last_item(), theme.ghost_theme())
            dpg.add_button(label="Watch", width=70, height=24,
                           tag="mon_watch", callback=self._attach_monitor_manual)
            theme.bind(dpg.last_item(), theme.accent_theme())
            widgets.tooltip(
                "Attach the monitor to this folder. Nothing is started or "
                "modified - the folder is only read.", wrap=280,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("●", tag="mon_dot", color=theme.TEXT_FAINT)
            dpg.add_text("No folder is being watched yet.", tag="mon_watch_label",
                         color=theme.TEXT_DIM, wrap=430)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Stop watching", width=130, height=24,
                           tag="mon_stop", callback=self.detach_monitor)
            theme.bind(dpg.last_item(), theme.ghost_theme())
            dpg.add_button(label="Open in Train tab", width=150, height=24,
                           callback=self._adopt_monitor_dir)
            theme.bind(dpg.last_item(), theme.ghost_theme())
            widgets.tooltip(
                "Copy this folder into the Train tab's run output folder.",
                wrap=260,
            )
        widgets.gap(1)
        dpg.add_text("", tag="mon_summary", color=theme.ACCEPTED, wrap=460)

        with dpg.group(tag="mon_empty"):
            widgets.empty_state(
                "No metrics yet. Pick a run folder above and press Watch. "
                "Curves appear once the first epoch has finished and "
                "results.csv exists.",
                wrap=440,
            )
        with dpg.group(tag="mon_plots", show=False):
            with dpg.plot(label="Loss", height=178, width=-1, tag="mon_loss_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_loss_x")
                with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="mon_loss_y"):
                    for tag, label in _LOSS_SERIES:
                        dpg.add_line_series([], [], label=label,
                                            tag=f"mon_s_{tag}")
            with dpg.plot(label="Metrics", height=178, width=-1,
                          tag="mon_metric_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_metric_x")
                with dpg.plot_axis(dpg.mvYAxis, label="value",
                                   tag="mon_metric_y"):
                    for col in _METRIC_SERIES:
                        dpg.add_line_series([], [],
                                            label=_METRIC_LABELS.get(col, col),
                                            tag=f"mon_s_{col}")
        theme.bind("mon_loss_plot", theme.plot_theme())
        theme.bind("mon_metric_plot", theme.plot_theme())

    def _browse_monitor_dir(self) -> None:
        path = dialogs.pick_dir("Select run folder")
        if path:
            dpg.set_value("mon_dir", path)

    def _adopt_monitor_dir(self) -> None:
        run_dir = _text("mon_dir")
        if not run_dir:
            self._status("Pick a run folder first.", "warn")
            return
        if dpg.does_item_exist("tr_out"):
            dpg.set_value("tr_out", run_dir)
            widgets.refresh_all_indicators(["tr_out"])
        self._status(f"Train tab output folder set to {run_dir}.", "ok")

    def _stage_from_dir(self, run_dir: str) -> int:
        stage = self.monitor_stage
        for candidate in _STAGES:
            if f"stage{candidate}" in str(run_dir):
                stage = candidate
        return stage

    def attach_monitor(self, run_dir: str, stage: int) -> None:
        """Watch ``run_dir`` and plot its curves. Reads only; starts nothing."""
        run_dir = str(run_dir or "").strip()
        if not run_dir:
            self._set_monitor_label(
                "No folder is being watched yet. Type or browse to a run "
                "folder, then press Watch.", "")
            self._status("Pick a run folder before pressing Watch.", "warn")
            return
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None
        self.monitor_stage = stage if stage in STAGE_META else 1
        self.monitor_dir = run_dir
        if dpg.does_item_exist("mon_dir"):
            dpg.set_value("mon_dir", run_dir)
        self._reset_monitor_series()
        short = _short_path(run_dir)
        if _exists(run_dir):
            self._set_monitor_label(
                f"Watching {short} (stage {self.monitor_stage}). Waiting for "
                "results.csv - it appears when the first epoch finishes.",
                "pending", run_dir)
        else:
            self._set_monitor_label(
                f"Watching {short}. The folder does not exist yet; it is "
                "created when the run starts.", "pending", run_dir)
        self.monitor = TrainingMonitor(run_dir, self._on_monitor_update,
                                       interval=2.0)
        self.monitor.start()

    def detach_monitor(self) -> None:
        """Stop watching. Nothing on disk is touched."""
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None
        self.monitor_dir = ""
        self._reset_monitor_series()
        self._set_monitor_label("No folder is being watched yet.", "")
        if dpg.does_item_exist("mon_summary"):
            dpg.set_value("mon_summary", "")

    def _attach_monitor_manual(self) -> None:
        run_dir = _text("mon_dir")
        self.attach_monitor(run_dir, self._stage_from_dir(run_dir))

    def _set_monitor_label(self, text: str, state: str,
                           full_path: str = "") -> None:
        if dpg.does_item_exist("mon_watch_label"):
            dpg.set_value("mon_watch_label", text)
            if full_path:
                widgets.attach_tooltip("mon_watch_label", full_path, wrap=420)
        if dpg.does_item_exist("mon_dot"):
            dpg.configure_item("mon_dot", color=theme.status_dot(state))

    def _show_monitor_plots(self, has_rows: bool) -> None:
        if dpg.does_item_exist("mon_empty"):
            dpg.configure_item("mon_empty", show=not has_rows)
        if dpg.does_item_exist("mon_plots"):
            dpg.configure_item("mon_plots", show=has_rows)

    def _reset_monitor_series(self) -> None:
        """Blank every series and hide the ones this stage does not produce.

        Called before a monitor is attached so the first two-second poll cannot
        paint the previous run's curve, and so stage 1 does not show three
        empty legend entries it will never fill.
        """
        active = {col for col, _ in metric_columns_for_stage(self.monitor_stage)}
        for tag, _label in _LOSS_SERIES:
            if dpg.does_item_exist(f"mon_s_{tag}"):
                dpg.set_value(f"mon_s_{tag}", [[], []])
                dpg.configure_item(f"mon_s_{tag}", show=True)
        for col in _METRIC_SERIES:
            if dpg.does_item_exist(f"mon_s_{col}"):
                dpg.set_value(f"mon_s_{col}", [[], []])
                dpg.configure_item(f"mon_s_{col}", show=(col in active))
        if dpg.does_item_exist("mon_summary"):
            dpg.set_value("mon_summary", "")
        self._show_monitor_plots(False)

    def _on_monitor_update(self, rows: List[Dict[str, str]],
                           latest: Optional[dict]) -> None:
        """Monitor-thread callback. Mirrors gui/monitor.py's existing contract."""
        try:
            has_rows = bool(rows)
            self._show_monitor_plots(has_rows)
            # A series is shown when it has data. Before the first row arrives
            # the stage's own set stays visible, so the legend already says what
            # this run is going to report.
            for tag, _label in _LOSS_SERIES:
                xs, ys = series(rows, "epoch", tag)
                if dpg.does_item_exist(f"mon_s_{tag}"):
                    dpg.set_value(f"mon_s_{tag}", [xs, ys])
                    dpg.configure_item(f"mon_s_{tag}",
                                       show=bool(xs) or not has_rows)
            active = {c for c, _ in metric_columns_for_stage(self.monitor_stage)}
            for col in _METRIC_SERIES:
                tag = f"mon_s_{col}"
                if not dpg.does_item_exist(tag):
                    continue
                if col not in active:
                    dpg.configure_item(tag, show=False)
                    continue
                xs, ys = series(rows, "epoch", col)
                dpg.set_value(tag, [xs, ys])
                dpg.configure_item(tag, show=bool(xs) or not has_rows)
            for axis in ("mon_loss_x", "mon_loss_y", "mon_metric_x",
                         "mon_metric_y"):
                if dpg.does_item_exist(axis):
                    dpg.fit_axis_data(axis)

            if has_rows:
                self._set_monitor_label(
                    f"Watching {_short_path(self.monitor_dir)} (stage "
                    f"{self.monitor_stage}). {len(rows)} epochs recorded.",
                    "ok", self.monitor_dir)
            if latest and dpg.does_item_exist("mon_summary"):
                dpg.set_value(
                    "mon_summary",
                    f"Epoch {latest.get('epoch', '?')}  |  "
                    f"{latest.get('primary_metric', 'primary metric')} = "
                    f"{latest.get('primary_metric_value', '?')}  |  "
                    f"best so far {latest.get('best_so_far', '?')}",
                )
        except (SystemError, TypeError, ValueError):
            # dearpygui was torn down under us, or a row held something odd.
            return

    # ═════════════════════════════════════════════════════════════════════
    # the shared run bar
    # ═════════════════════════════════════════════════════════════════════
    def _build_run_bar(self) -> None:
        dpg.add_separator()
        widgets.gap(0)
        with dpg.group(horizontal=True):
            widgets.status_pill("job_pill", "Idle", "idle")
            dpg.add_spacer(width=theme.SP_S)
            dpg.add_text("", tag="job_elapsed", color=theme.TEXT_FAINT)
            dpg.add_spacer(width=theme.SP_M)
            dpg.add_button(label="Cancel run", width=120, height=24,
                           tag="job_cancel", callback=self._cancel_run)
            theme.bind("job_cancel", theme.ghost_theme())
            widgets.tooltip(
                "Terminate the running command. A cancelled run is reported as "
                "stopped, not as an error. Files already written are kept.",
                wrap=300,
            )
            dpg.add_button(label="Clear log", width=100, height=24,
                           tag="job_clear", callback=self._clear_log)
            theme.bind("job_clear", theme.ghost_theme())
            dpg.add_button(label="Copy log", width=100, height=24,
                           tag="job_copy_log", callback=self._copy_log)
            theme.bind("job_copy_log", theme.ghost_theme())
        dpg.add_text("Nothing has been started yet.", tag="train_status",
                     color=theme.TEXT_DIM, wrap=1120)
        with dpg.group(horizontal=True):
            dpg.add_text("Command:", color=theme.TEXT_DIM)
            dpg.add_input_text(
                tag="job_cmdline", default_value="", readonly=True, width=-130,
                hint="The exact command line appears here when a run starts.",
            )
            theme.bind("job_cmdline", theme.mono_field_theme())
            dpg.add_button(label="Copy", width=110, height=24,
                           tag="job_copy_cmd", callback=self._copy_command)
            theme.bind("job_copy_cmd", theme.ghost_theme())
            widgets.tooltip(
                "Copy the command so it can be pasted into a terminal, or sent "
                "to whoever owns the GPU machine.", wrap=280,
            )
        with dpg.child_window(tag="train_log_win", width=-1, height=-1,
                              border=True):
            dpg.add_text("", tag="train_log", wrap=1110)
            theme.bind_mono("train_log")
        theme.bind("train_log_win", theme.panel_theme())

    # ─── status helpers ──────────────────────────────────────────────────
    def _status(self, message: str, kind: str = "idle") -> None:
        widgets.status_text("train_status", message, kind)
        if self.on_status is not None:
            try:
                self.on_status(message)
            except Exception:  # noqa: BLE001 - a host hook must never break us
                pass

    def _set_pill(self, text: str, kind: str) -> None:
        widgets.status_pill("job_pill", text, kind)

    def _set_idle(self) -> None:
        self._set_pill("Idle", "idle")
        self._set_enabled("job_cancel", False)
        self._set_enabled("tr_start", True, accent=True)
        if dpg.does_item_exist("job_elapsed"):
            dpg.set_value("job_elapsed", "")

    def _set_enabled(self, tag: str, on: bool, *, accent: bool = False) -> None:
        """Enable/disable a control without stepping on a capability lock."""
        if not dpg.does_item_exist(tag) or widgets.is_locked(tag):
            return
        try:
            dpg.configure_item(tag, enabled=bool(on))
        except SystemError:
            return
        if on:
            theme.bind(tag, theme.accent_theme() if accent
                       else theme.ghost_theme())
        else:
            theme.bind(tag, theme.disabled_theme())

    def _copy(self, text: str, confirmation: str) -> None:
        if not text:
            self._status("There is nothing to copy yet.", "idle")
            return
        try:
            dpg.set_clipboard_text(text)
        except (SystemError, AttributeError):
            self._status("The clipboard is not available in this window.",
                         "warn")
            return
        self._status(confirmation, "ok")

    def _copy_command(self) -> None:
        self._copy(_text("job_cmdline"), "Command line copied to the clipboard.")

    def _copy_log(self) -> None:
        self._copy(self.runner.text(), "Log copied to the clipboard.")

    def _clear_log(self) -> None:
        self.runner.clear()
        self._status("Log cleared. The running command is not affected.",
                     "idle")

    # ─── the run lifecycle ───────────────────────────────────────────────
    def _run_job(self, label: str, cmd: Sequence, *,
                 preflight: Sequence[str] = ()) -> bool:
        """Launch ``cmd``, owning the pill, the elapsed clock and the log.

        Returns False when the launch was refused (something else is running).
        """
        if self.runner.is_running():
            self._status(
                f"{self._job_label or 'A command'} is still running. Cancel it "
                "first, or wait for it to finish.", "warn")
            return False

        self._job_label = label
        self._stop_requested = False
        self._job_started = time.monotonic()

        line = _command_line(cmd)
        if dpg.does_item_exist("job_cmdline"):
            dpg.set_value("job_cmdline", line)
        if preflight:
            self.runner.log("Pre-flight\n" + "\n".join(preflight))

        self._set_pill("Running", "running")
        self._status(f"{label} started. Watch the log below.", "running")
        self._set_enabled("job_cancel", True)
        self._set_enabled("tr_start", False)
        if dpg.does_item_exist("job_elapsed"):
            dpg.set_value("job_elapsed", "0:00 elapsed")

        self._start_clock()
        self.runner.run(list(cmd), on_done=self._on_job_done)
        return True

    def _on_job_done(self, code: int) -> None:
        """Runner-thread callback. A user cancel is a stop, never an error."""
        self._stop_clock()
        label = self._job_label or "The command"
        elapsed = _fmt_elapsed(time.monotonic() - self._job_started)
        try:
            if self._stop_requested:
                self._set_pill("Stopped", "warn")
                self._status(
                    f"{label} stopped after {elapsed}. Anything already written "
                    "to the output folder is kept.", "warn")
            elif code == 0:
                self._set_pill("Finished", "ok")
                self._status(f"{label} finished in {elapsed}.", "ok")
            else:
                self._set_pill("Failed", "error")
                self._status(
                    f"{label} failed after {elapsed} (exit code {code}). The "
                    "last lines of the log say why.", "error")
            self._set_enabled("job_cancel", False)
            self._set_enabled("tr_start", True, accent=True)
            if dpg.does_item_exist("job_elapsed"):
                dpg.set_value("job_elapsed", f"{elapsed} total")
            self._refresh_log_scroll()
        except (SystemError, TypeError):
            return

    def _cancel_run(self) -> None:
        if not self.runner.is_running():
            self._status("Nothing is running.", "idle")
            return
        self._stop_requested = True
        self._set_pill("Stopping", "warn")
        self._status(f"Stop requested for {self._job_label or 'the command'}...",
                     "warn")
        self.runner.stop()

    # ─── elapsed clock ───────────────────────────────────────────────────
    def _start_clock(self) -> None:
        self._stop_clock()
        self._clock_stop = threading.Event()
        self._clock = threading.Thread(target=self._clock_loop, daemon=True)
        self._clock.start()

    def _stop_clock(self) -> None:
        self._clock_stop.set()
        self._clock = None

    def _clock_loop(self) -> None:
        # gui/monitor.py and gui/runner.py already write into dearpygui from a
        # daemon thread; this follows that established pattern rather than
        # competing with the host for set_frame_callback.
        while not self._clock_stop.is_set():
            self.tick()
            self._clock_stop.wait(0.5)

    def _refresh_log_scroll(self) -> None:
        if dpg.does_item_exist("train_log_win"):
            try:
                dpg.set_y_scroll("train_log_win", -1.0)
            except (SystemError, TypeError):
                pass

    def tick(self) -> None:
        """Refresh the elapsed clock and keep the log scrolled to the bottom.

        Safe to call from the host render loop (main thread) or not at all -
        an internal clock thread calls it while a run is in flight.
        """
        if not self.runner.is_running():
            return
        try:
            if dpg.does_item_exist("job_elapsed"):
                dpg.set_value(
                    "job_elapsed",
                    f"{_fmt_elapsed(time.monotonic() - self._job_started)} elapsed",
                )
            self._refresh_log_scroll()
        except (SystemError, TypeError):
            return

    # ═════════════════════════════════════════════════════════════════════
    # validation
    # ═════════════════════════════════════════════════════════════════════
    def _capability_problem(self) -> str:
        if self._cap.package_reason:
            return ("YOAKE is not importable in this Python environment. Run: "
                    "pip install -e .")
        if not self._cap.torch:
            return ("PyTorch is not installed in this Python environment, so "
                    "this cannot run here. Install it with: pip install torch "
                    "torchvision")
        return ""

    def _validate_train(self) -> List[str]:
        problems: List[str] = []
        cap = self._capability_problem()
        if cap:
            problems.append(cap)

        meta = STAGE_META[self.current_stage]
        train_root = _text("tr_train")
        val_root = _text("tr_val")
        out = _text("tr_out")

        if not train_root:
            problems.append(
                'Training data is required. Point it at the '
                'annotations_train.json that "Create splits" wrote on the '
                'Dataset tab.')
        elif not _exists(train_root):
            problems.append(
                f'Training data not found: {train_root}. Create splits in '
                'step 1, or point "Training data" at an existing folder.')
        if val_root and not _exists(val_root):
            problems.append(
                f"Validation data not found: {val_root}. Leave it empty to "
                "train without validation metrics.")
        if not out:
            problems.append(
                "Run output folder is required. It receives the checkpoints "
                "and results.csv.")

        if _text("tr_fmt", "json") == "yolo":
            classes = _text("tr_classes")
            if classes and not _exists(classes):
                problems.append(
                    f"Class names file not found: {classes}. Leave it empty to "
                    "use the class ids as they are.")

        epochs = _int("tr_epochs", meta["epochs"])
        if epochs < 1:
            problems.append("Epochs must be 1 or more.")
        batch = _int("tr_batch", meta["batch"])
        if batch < 1:
            problems.append("Batch size must be 1 or more.")
        lr = _float("tr_lr", meta["lr"])
        if lr <= 0:
            problems.append("Learning rate must be greater than 0.")
        elif lr > 1.0:
            problems.append(
                "Learning rate must be 1.0 or less. Stage defaults are between "
                "0.00001 and 0.0001.")
        if meta["needs_window"] and _int("tr_window", 16) < 2:
            problems.append(
                "Window size must be 2 or more for a temporal stage. "
                "(default 16)")
        return problems

    def _validate_eval(self) -> List[str]:
        problems: List[str] = []
        cap = self._capability_problem()
        if cap:
            problems.append(cap)
        ckpt = _text("tr_ckpt")
        val_root = _text("tr_eval_val")
        if not ckpt:
            problems.append(
                "A checkpoint is required. It is the .pth file a training run "
                "wrote, usually weights/best.pth inside the run folder.")
        elif not _exists(ckpt):
            problems.append(
                f"Checkpoint not found: {ckpt}. Check the path, or point it at "
                "weights/best.pth inside a finished run folder.")
        if not val_root:
            problems.append(
                "Validation data is required. Use annotations_val.json or "
                "annotations_test.json from Create splits.")
        elif not _exists(val_root):
            problems.append(f"Validation data not found: {val_root}.")
        if not _text("tr_eval_out"):
            problems.append("An evaluation output folder is required.")
        return problems

    def _validate_analyze(self) -> List[str]:
        problems: List[str] = []
        cap = self._capability_problem()
        if cap:
            problems.append(cap)
        mode = _text("an_mode", "timeline")
        pred = _text("an_pred")
        anno = _text("an_anno")
        missing = commands.analyze_missing_field(mode, pred, anno)
        if missing == "predictions":
            problems.append(
                f"Mode {mode} needs a predictions JSON. Run inference in step "
                "3, or point it at a predictions.json you already have.")
        elif missing == "annotation":
            problems.append(
                f"Mode {mode} needs an annotation JSON. Use the labelled data "
                "you trained on.")
        if pred and not _exists(pred):
            problems.append(f"Predictions JSON not found: {pred}.")
        if anno and not _exists(anno):
            problems.append(f"Annotation JSON not found: {anno}.")
        if not _text("an_out"):
            problems.append("An output folder is required.")
        return problems

    def _report(self, tag: str, problems: Sequence[str], ok_message: str, *,
                quiet: bool) -> bool:
        """Render a validation result inline. Returns True when it is runnable."""
        if problems:
            body = "\n".join(f"- {p}" for p in problems)
            widgets.status_text(tag, body, "error")
            if not quiet:
                self._status(
                    f"{len(problems)} problem(s) to fix before this can run.",
                    "error")
            return False
        widgets.status_text(tag, ok_message, "ok")
        if not quiet:
            self._status(ok_message, "ok")
        return True

    def _report_train_validation(self, *, quiet: bool) -> bool:
        return self._report(
            "tr_problems", self._validate_train(),
            "Settings look complete. Start training when you are ready.",
            quiet=quiet)

    def _report_eval_validation(self, *, quiet: bool) -> bool:
        return self._report(
            "tr_eval_problems", self._validate_eval(),
            "Settings look complete. The source run is not modified.",
            quiet=quiet)

    def _report_analyze_validation(self, *, quiet: bool) -> bool:
        return self._report(
            "an_problems", self._validate_analyze(),
            "Settings look complete.", quiet=quiet)

    # ═════════════════════════════════════════════════════════════════════
    # launchers
    # ═════════════════════════════════════════════════════════════════════
    def _start_train(self) -> None:
        if not self._report_train_validation(quiet=False):
            return

        stage = self.current_stage
        meta = STAGE_META[stage]
        train_root = _text("tr_train")
        val_root = _text("tr_val")
        out = _text("tr_out")
        window = _int("tr_window", 16) if meta["needs_window"] else None
        imb = _text("tr_imb", "class_weight") if meta["needs_imbalance"] else None
        fmt = _text("tr_fmt", "json")
        model_mode = _text("tr_model", meta["model_default"])
        classes_file = _text("tr_classes")

        effective_stage = stage
        coercion = ""
        if model_mode == "detection" and stage != 1:
            effective_stage = 1
            window = None
            imb = None
            coercion = 'forced by model mode "detection"'
        elif model_mode == "detection+temporal" and stage == 1:
            effective_stage = 2
            if window is None:
                window = _int("tr_window", 16)
            coercion = 'forced by model mode "detection+temporal"'

        cmd = commands.train_command(
            effective_stage, train_root, val_root, out,
            epochs=_int("tr_epochs", meta["epochs"]),
            batch_size=_int("tr_batch", meta["batch"]),
            lr=_float("tr_lr", meta["lr"]),
            window_size=window,
            imbalance_strategy=imb,
            annotation_format=fmt,
            classes_file=classes_file,
        )

        effective_text = str(effective_stage)
        if coercion:
            effective_text = f"{effective_stage} ({coercion})"
        preflight = [
            f"  Stage:        {stage} ({meta['label']})",
            f"  Effective:    {effective_text}",
            f"  Format:       {fmt}   model mode: {model_mode}",
            f"  Train root:   {train_root}   {_mark(train_root)}",
            f"  Val root:     {val_root or '(none)'}   {_mark(val_root)}",
            f"  Output:       {out}",
            "  Epochs {e} / batch {b} / lr {l:.6f} / window {w} / imbalance {i}"
            .format(
                e=_int("tr_epochs", meta["epochs"]),
                b=_int("tr_batch", meta["batch"]),
                l=_float("tr_lr", meta["lr"]),
                w=window if window is not None else "-",
                i=imb or "-",
            ),
        ]
        if coercion:
            preflight.append(
                f"  Note: the model mode overrode the stage buttons. Stage "
                f"{effective_stage} is what actually runs.")
        if not val_root:
            preflight.append(
                "  Note: no validation data, so there will be no validation "
                "loss curve.")

        if not self._run_job("Training", cmd, preflight=preflight):
            return
        # Attach AFTER the launch and with the series cleared, so the first
        # two-second poll cannot paint the previous run's curve.
        self.attach_monitor(out, effective_stage)

    def _start_eval(self) -> None:
        if not self._report_eval_validation(quiet=False):
            return
        stage = self.current_stage
        ckpt = _text("tr_ckpt")
        val_root = _text("tr_eval_val")
        out = _text("tr_eval_out")
        cmd = commands.val_command(stage, ckpt, val_root, out)
        preflight = [
            f"  Stage:        {stage} ({STAGE_META[stage]['label']})",
            f"  Checkpoint:   {ckpt}   {_mark(ckpt)}",
            f"  Val root:     {val_root}   {_mark(val_root)}",
            f"  Output:       {out}",
        ]
        self._run_job("Evaluation", cmd, preflight=preflight)

    def _start_analyze(self) -> None:
        if not self._report_analyze_validation(quiet=False):
            return
        mode = _text("an_mode", "timeline")
        pred = _text("an_pred")
        anno = _text("an_anno")
        out = _text("an_out")
        cmd = commands.analyze_command(mode, out, predictions=pred,
                                       annotation=anno)
        preflight = [
            f"  Mode:         {mode}",
            f"  Predictions:  {pred or '(none)'}   {_mark(pred)}",
            f"  Annotations:  {anno or '(none)'}   {_mark(anno)}",
            f"  Output:       {out}",
        ]
        self._run_job("Analysis", cmd, preflight=preflight)

    # ─── dataset tools ───────────────────────────────────────────────────
    def _start_convert(self) -> None:
        mode = _text("ds_conv_fmt", "csv")
        src = _text("ds_conv_in")
        out = _text("ds_conv_out")
        csv_path = _text("ds_conv_csv")

        problems: List[str] = []
        if not src:
            problems.append(
                "An input file is required." if mode != "video_csv"
                else "A video file is required for the video_csv format.")
        elif not _exists(src):
            problems.append(f"Input file not found: {src}.")
        if not out:
            problems.append(
                "An output JSON path is required. The converted annotations "
                "are written there.")
        if mode == "video_csv":
            if not csv_path:
                problems.append(
                    "The video_csv format also needs the CSV file that holds "
                    "the labels.")
            elif not _exists(csv_path):
                problems.append(f"CSV file not found: {csv_path}.")
        if not self._report("ds_conv_problems", problems,
                            "Ready to convert.", quiet=False):
            return

        # tools/convert_annotations.py reads mode= / input= / output=.
        cmd: List = [
            self.tools / "convert_annotations.py",
            f"mode={mode}",
            f"input={src}",
            f"output={out}",
            f"video_width={_int('ds_conv_w', 640)}",
            f"video_height={_int('ds_conv_h', 640)}",
            f"fps={_float('ds_conv_fps', 30.0)}",
            f"bbox_format={_text('ds_conv_bbox', 'xyxy')}",
        ]
        if mode == "video_csv":
            cmd.append(f"csv={csv_path}")
        preflight = [
            f"  Mode:         {mode}",
            f"  Input:        {src}   {_mark(src)}",
            f"  Output:       {out}",
        ]
        if mode == "video_csv":
            preflight.append(f"  CSV:          {csv_path}   {_mark(csv_path)}")
        self._run_job("Annotation conversion", cmd, preflight=preflight)

    def _start_split(self) -> None:
        src = _text("ds_split_anno")
        out = _text("ds_split_out")
        val = max(0.0, min(0.45, _float("ds_val_ratio", 0.15)))
        test = max(0.0, min(0.45, _float("ds_test_ratio", 0.15)))

        problems: List[str] = []
        if not src:
            problems.append("An annotation JSON is required.")
        elif not _exists(src):
            problems.append(f"Annotation JSON not found: {src}.")
        if not out:
            problems.append(
                "An output folder is required. It receives the three split "
                "files.")
        if val + test >= 1.0:
            problems.append(
                "The validation and test ratios must add up to less than 1.0, "
                "otherwise nothing is left to train on.")
        if not self._report("ds_split_problems", problems,
                            "Ready to create splits.", quiet=False):
            return

        # tools/build_splits.py reads input= / output_dir= / val_ratio= /
        # test_ratio=. There is no train_ratio: training gets the remainder.
        cmd: List = [
            self.tools / "build_splits.py",
            f"input={src}",
            f"output_dir={out}",
            f"val_ratio={val:.4f}",
            f"test_ratio={test:.4f}",
            f"strategy={_text('ds_split_strategy', 'video')}",
            f"seed={_int('ds_split_seed', 42)}",
        ]
        preflight = [
            f"  Input:        {src}   {_mark(src)}",
            f"  Output:       {out}",
            f"  Ratios:       train {(1.0 - val - test) * 100:.0f}% / "
            f"val {val * 100:.0f}% / test {test * 100:.0f}%",
            f"  Strategy:     {_text('ds_split_strategy', 'video')}   "
            f"seed {_int('ds_split_seed', 42)}",
        ]
        self._run_job("Split creation", cmd, preflight=preflight)

    def _start_stats(self) -> None:
        src = _text("ds_stat_anno")
        problems: List[str] = []
        if not src:
            problems.append("An annotation JSON is required.")
        elif not _exists(src):
            problems.append(f"Annotation JSON not found: {src}.")
        if not self._report("ds_stat_problems", problems,
                            "Ready to summarize.", quiet=False):
            return

        out = _text("ds_stat_out",
                    str(self.root / "outputs" / "dataset_summary"))
        plots = "1" if bool(_value("ds_stat_plots", True)) else "0"
        cmd: List = [
            self.tools / "summarize_dataset.py",
            f"anno={src}",
            f"output_dir={out}",
            f"plots={plots}",
        ]
        preflight = [
            f"  Annotations:  {src}   {_mark(src)}",
            f"  Output:       {out}",
        ]
        self._run_job("Dataset statistics", cmd, preflight=preflight)

    # ═════════════════════════════════════════════════════════════════════
    # teardown
    # ═════════════════════════════════════════════════════════════════════
    def teardown(self) -> None:
        """Stop everything this panel started, so nothing survives GUI close."""
        self._stop_clock()
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None
        try:
            self.runner.stop()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
