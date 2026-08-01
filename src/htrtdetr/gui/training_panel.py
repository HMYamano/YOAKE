"""
training_panel.py - Training GUI (Dear PyGui)
=============================================

Dataset preparation, staged training, evaluation, and output analysis in one
panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import dearpygui.dearpygui as dpg

from . import commands, dialogs, widgets
from .monitor import TrainingMonitor, metric_columns_for_stage, series
from .runner import CommandRunner

_LOSS_SERIES = [("train_loss", "Train Loss"), ("val_loss", "Validation Loss")]
_METRIC_SERIES = ["ap50", "macro_f1", "idf1", "composite_score"]

STAGE_META = {
    1: {
        "label": "Stage 1 - Detector",
        "desc": (
            "Train the detector that finds animals in single frames. "
            "This is the starting point for the later temporal stages."
        ),
        "epochs": 50,
        "batch": 4,
        "lr": 1e-4,
        "model_default": "detection",
        "needs_window": False,
        "needs_imbalance": False,
    },
    2: {
        "label": "Stage 2 - Action Head",
        "desc": (
            "Train the temporal action head on top of the detector. "
            "This stage uses a frame window to classify behavior over time."
        ),
        "epochs": 30,
        "batch": 8,
        "lr": 5e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": True,
    },
    3: {
        "label": "Stage 3 - ID Head",
        "desc": (
            "Train the temporal ID head so the same animal keeps a consistent "
            "track ID across frames."
        ),
        "epochs": 30,
        "batch": 8,
        "lr": 5e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": False,
    },
    4: {
        "label": "Stage 4 - Unified Fine-Tuning",
        "desc": (
            "Fine-tune the full model with all major modules enabled at a lower "
            "learning rate."
        ),
        "epochs": 20,
        "batch": 2,
        "lr": 1e-5,
        "model_default": "detection+temporal",
        "needs_window": True,
        "needs_imbalance": False,
    },
}

_STAGE_LABELS = [STAGE_META[s]["label"] for s in (1, 2, 3, 4)]
_LABEL_TO_STAGE = {STAGE_META[s]["label"]: s for s in (1, 2, 3, 4)}

_MODEL_OPTIONS = ["detection", "detection+temporal"]


class TrainingPanel:
    """Training page embedded inside the GUI hub."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tools = self.root / "tools"
        self.runs = self.root / "runs"
        self.runner = CommandRunner("train_log", "train_status", cwd=self.root)
        self.monitor: Optional[TrainingMonitor] = None
        self.monitor_stage = 1
        self.current_stage = 1

    def build(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_training",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=False,
        ):
            dpg.add_text("Training Workspace", color=(132, 178, 236))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text(
                "Prepare the dataset, train each stage, evaluate checkpoints, and watch "
                "metrics update live in the monitor.",
                wrap=980,
                color=(196, 202, 210),
            )
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=640, height=-150, border=False):
                    with dpg.tab_bar():
                        self._tab_dataset()
                        self._tab_training()
                        self._tab_analyze()
                with dpg.child_window(width=-1, height=-150, border=True):
                    self._build_monitor()

            dpg.add_text("", tag="train_status", color=(184, 224, 186), wrap=1160)
            with dpg.child_window(tag="train_log_win", width=-1, height=130, border=True):
                dpg.add_text("", tag="train_log", wrap=1140)

    def _field(self, label, tag, default="", browse=None, width=390, *,
               field_key=None, required=False, tooltip_text=None, hint=""):
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

    def _pick_ckpt(self, _tag=None):
        return lambda: dialogs.pick_file("Select checkpoint", dialogs.CKPT_TYPES)

    def _pick_dir(self, _tag=None):
        return lambda: dialogs.pick_dir("Select folder")

    def _set(self, tag, value):
        if value:
            dpg.set_value(tag, value)

    def _section(self, label: str) -> None:
        dpg.add_spacer(height=4)
        dpg.add_text(label, color=(166, 205, 241))
        dpg.add_separator()

    def _note(self, text: str, wrap: int = 620) -> None:
        dpg.add_text(text, color=(170, 176, 184), wrap=wrap)

    def _tab_dataset(self):
        with dpg.tab(label="1. Dataset"):
            self._section("Dataset Preparation")
            self._note(
                "Start here when you need to convert annotations, create train/validation/test "
                "splits, or inspect dataset statistics before training."
            )

            self._section("1. Convert Annotation Format")
            self._field(
                "Input annotations (CSV or MOT17)", "ds_conv_in",
                field_key="annotation_source",
                required=True,
                browse=self._pick_json("ds_conv_in"),
                tooltip_text=(
                    "Original annotation file exported from your labeling "
                    "tool. Supports CSV (as produced by the YOAKE annotator) "
                    "or MOT17 gt.txt files."
                ),
            )
            self._field(
                "Output JSON", "ds_conv_out",
                field_key="annotation_json",
                required=True,
                browse=lambda: dialogs.save_file(),
                tooltip_text="Path where the converted annotation JSON will be written.",
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Format:", indent=4)
                dpg.add_combo(["csv", "mot17"], tag="ds_conv_fmt",
                              default_value="csv", width=140)
                widgets.tooltip(
                    "csv: YOAKE annotator export\nmot17: MOT17-style gt.txt",
                    wrap=220,
                )
            widgets.primary_button(
                "Convert Annotations",
                lambda: self.runner.run([
                    self.tools / "convert_annotations.py",
                    f"input={dpg.get_value('ds_conv_in')}",
                    f"output={dpg.get_value('ds_conv_out')}",
                    f"format={dpg.get_value('ds_conv_fmt')}",
                ]),
                height=32, width=260,
            )

            self._section("2. Build Train / Validation / Test Splits")
            self._field(
                "Annotation JSON", "ds_split_anno",
                field_key="annotation_json",
                required=True,
                browse=self._pick_json("ds_split_anno"),
                tooltip_text="Converted annotation JSON to split into train/val/test.",
            )
            self._field(
                "Output folder", "ds_split_out",
                field_key="split_dir",
                required=True,
                browse=self._pick_dir("ds_split_out"),
                tooltip_text="Directory that will receive train.json / val.json / test.json.",
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Train ratio:", indent=4)
                dpg.add_input_float(tag="ds_train_ratio",
                                    default_value=0.7, width=80,
                                    format="%.2f", step=0)
                widgets.tooltip("Fraction assigned to training (e.g. 0.7 = 70%).", wrap=220)
                dpg.add_text("Validation ratio:")
                dpg.add_input_float(tag="ds_val_ratio",
                                    default_value=0.15, width=80,
                                    format="%.2f", step=0)
                widgets.tooltip("Fraction for validation. The rest goes to test.", wrap=220)
            widgets.primary_button(
                "Create Splits",
                lambda: self.runner.run([
                    self.tools / "build_splits.py",
                    f"anno={dpg.get_value('ds_split_anno')}",
                    f"output={dpg.get_value('ds_split_out')}",
                    f"train_ratio={dpg.get_value('ds_train_ratio')}",
                    f"val_ratio={dpg.get_value('ds_val_ratio')}",
                ]),
                height=32, width=260,
            )

            self._section("3. Review Dataset Statistics")
            self._field(
                "Target annotation JSON", "ds_stat_anno",
                field_key="annotation_json",
                browse=self._pick_json("ds_stat_anno"),
                tooltip_text="Any annotation JSON. Prints counts, action distribution, etc.",
            )
            widgets.secondary_button(
                "Show Statistics",
                lambda: self.runner.run([
                    self.tools / "summarize_dataset.py",
                    f"anno={dpg.get_value('ds_stat_anno')}",
                ]),
                height=30, width=200,
            )
            dpg.add_spacer(height=6)
            self._note("Tip: the video labeling workflow is available from the Labeling page in the left navigation.")

    def _tab_training(self):
        with dpg.tab(label="2. Training"):
            self._section("Stage Selection")
            self._note(
                "Choose the stage you want to train. The description, defaults, and enabled "
                "fields update automatically."
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Training stage:", indent=4)
                dpg.add_combo(
                    _STAGE_LABELS,
                    tag="tr_stage",
                    default_value=_STAGE_LABELS[0],
                    width=330,
                    callback=lambda s, a, u: self._on_stage_change(a),
                )
                dpg.add_button(
                    label="Restore Stage Defaults",
                    callback=lambda: self._on_stage_change(dpg.get_value("tr_stage")),
                )
            dpg.add_text("", tag="tr_stage_desc", color=(188, 202, 224), wrap=620)

            self._section("Input Data")
            self._note("Point the stage to the training and validation data that matches the selected format.")
            with dpg.group(horizontal=True):
                dpg.add_text("Annotation format:", indent=4)
                dpg.add_combo(["json", "yolo"], tag="tr_fmt",
                              default_value="json", width=110)
                widgets.tooltip(
                    "json: YOAKE annotation JSON.\n"
                    "yolo: YOLO-style dataset (images/ + labels/).",
                    wrap=260,
                )
                dpg.add_text("Model mode:")
                dpg.add_combo(
                    _MODEL_OPTIONS,
                    tag="tr_model",
                    default_value=STAGE_META[1]["model_default"],
                    width=200,
                )
                widgets.tooltip(
                    "detection: single-frame detector (Stage 1).\n"
                    "detection+temporal: full temporal model (Stages 2-4).",
                    wrap=280,
                )
            self._field(
                "Training data", "tr_train",
                field_key="train_data",
                required=True,
                browse=self._pick_json("tr_train"),
                tooltip_text="Training annotation JSON (or YOLO dataset root).",
            )
            self._field(
                "Validation data", "tr_val",
                field_key="val_data",
                browse=self._pick_json("tr_val"),
                tooltip_text="Optional validation JSON. Metrics are logged if provided.",
            )
            self._field(
                "Class names file (YOLO only, optional)", "tr_classes",
                field_key="classes_file",
                browse=self._pick_json("tr_classes"),
                tooltip_text="Text file listing class names, one per line. YOLO only.",
            )
            self._field(
                "Run output folder", "tr_out",
                field_key="train_output_dir",
                required=True,
                default=str(self.runs / "train" / "stage1"),
                browse=self._pick_dir("tr_out"),
                tooltip_text=(
                    "Directory that receives checkpoints, results.csv, "
                    "metrics_latest.json, and the live monitor data."
                ),
            )

            self._section("Hyperparameters")
            self._note(
                "Window size is only used by temporal stages. Imbalance strategy is mainly "
                "useful for action classification."
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Epochs:", indent=4)
                dpg.add_input_int(tag="tr_epochs",
                                  default_value=STAGE_META[1]["epochs"],
                                  width=90)
                widgets.tooltip("Number of full passes over the training data.", wrap=240)
                dpg.add_text("Batch size:")
                dpg.add_input_int(tag="tr_batch",
                                  default_value=STAGE_META[1]["batch"],
                                  width=80)
                widgets.tooltip("Samples per gradient step. Lower if you run out of GPU memory.", wrap=260)
                dpg.add_text("Learning rate:")
                dpg.add_input_float(
                    tag="tr_lr",
                    default_value=STAGE_META[1]["lr"],
                    width=110,
                    format="%.6f",
                    step=0,
                )
                widgets.tooltip("Optimizer learning rate. Stage defaults are usually a good start.", wrap=260)
            with dpg.group(horizontal=True):
                dpg.add_text("Window size:", indent=4)
                dpg.add_input_int(tag="tr_window",
                                  default_value=16, width=90, enabled=False)
                widgets.tooltip(
                    "Number of consecutive frames fed to the temporal head. "
                    "Ignored by Stage 1 (single-frame detector).",
                    wrap=280,
                )
                dpg.add_text("Imbalance strategy:")
                dpg.add_combo(
                    ["none", "class_weight", "focal"],
                    tag="tr_imb",
                    default_value="class_weight",
                    width=150,
                    enabled=False,
                )
                widgets.tooltip(
                    "How the action loss handles class imbalance.\n"
                    "class_weight: inverse-frequency weights (safe default).\n"
                    "focal: focal loss for very skewed distributions.\n"
                    "none: no reweighting.",
                    wrap=300,
                )

            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Start Training", self._start_train,
                    height=38, width=220,
                )
                dpg.add_spacer(width=6)
                dpg.add_button(
                    label="Watch Output Folder",
                    height=34, width=200,
                    callback=lambda: self.attach_monitor(
                        dpg.get_value("tr_out"), self.current_stage),
                )
                widgets.tooltip(
                    "Attach the live monitor to an existing run folder so you "
                    "can watch loss/metric plots update without re-training.",
                    wrap=280,
                )
                dpg.add_button(
                    label="Stop", height=34, width=90,
                    callback=lambda: self.runner.stop(),
                )
                widgets.tooltip("Terminate the running training/evaluation process.", wrap=260)

            self._section("Evaluation")
            self._note("Use this section to evaluate a trained checkpoint on validation data.")
            self._field(
                "Checkpoint (.pt)", "tr_ckpt",
                field_key="checkpoint",
                required=True,
                browse=self._pick_ckpt("tr_ckpt"),
                tooltip_text="Trained model weights produced by a training run.",
            )
            self._field(
                "Validation data", "tr_eval_val",
                field_key="val_data",
                browse=self._pick_json("tr_eval_val"),
                tooltip_text="Annotation JSON used to compute evaluation metrics.",
            )
            self._field(
                "Evaluation output folder", "tr_eval_out",
                field_key="eval_output_dir",
                default=str(self.runs / "val" / "stage1"),
                browse=self._pick_dir("tr_eval_out"),
                tooltip_text="Directory that receives metrics, plots, and evaluation JSON.",
            )
            widgets.primary_button(
                "Run Evaluation", self._start_eval, height=34, width=220,
            )

            self._on_stage_change(_STAGE_LABELS[0])

    def _tab_analyze(self):
        with dpg.tab(label="3. Analyze Outputs"):
            self._section("Analysis Mode")
            self._note(
                "Run quick offline analysis from predictions or ground-truth annotations without "
                "leaving the training workspace."
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Mode:", indent=4)
                dpg.add_combo(
                    ["timeline", "distribution", "id_switches", "confidence"],
                    tag="an_mode",
                    default_value="timeline",
                    width=170,
                    callback=lambda s, a, u: self._on_analyze_mode_change(a),
                )
                widgets.tooltip(
                    "Choose what to analyze. The description below shows the "
                    "required input for the selected mode.",
                    wrap=260,
                )
            dpg.add_text("", tag="an_mode_desc", color=(188, 202, 224), wrap=620)

            self._section("Inputs")
            self._field(
                "Predictions JSON", "an_pred",
                field_key="predictions_json",
                browse=self._pick_json("an_pred"),
                tooltip_text="Predictions produced by 'yoake predict' or the Video Analysis page.",
            )
            self._field(
                "Annotation JSON", "an_anno",
                field_key="annotation_json",
                browse=self._pick_json("an_anno"),
                tooltip_text="Ground-truth annotation JSON (required for 'distribution' mode).",
            )
            self._field(
                "Output folder", "an_out",
                field_key="analyze_output_dir",
                default=str(self.runs / "analyze"),
                browse=self._pick_dir("an_out"),
                tooltip_text="Directory that receives the analysis figures / JSON.",
            )
            dpg.add_spacer(height=8)
            widgets.primary_button(
                "Run Analysis", self._start_analyze, height=34, width=200,
            )

            self._on_analyze_mode_change("timeline")

    def _on_stage_change(self, label: str) -> None:
        stage = _LABEL_TO_STAGE.get(label, 1)
        meta = STAGE_META[stage]
        self.current_stage = stage

        if dpg.does_item_exist("tr_stage_desc"):
            dpg.set_value("tr_stage_desc", meta["desc"])
        if dpg.does_item_exist("tr_epochs"):
            dpg.set_value("tr_epochs", meta["epochs"])
        if dpg.does_item_exist("tr_batch"):
            dpg.set_value("tr_batch", meta["batch"])
        if dpg.does_item_exist("tr_lr"):
            dpg.set_value("tr_lr", meta["lr"])
        if dpg.does_item_exist("tr_model"):
            dpg.set_value("tr_model", meta["model_default"])
        if dpg.does_item_exist("tr_out"):
            dpg.set_value("tr_out", str(self.runs / "train" / f"stage{stage}"))
        if dpg.does_item_exist("tr_eval_out"):
            dpg.set_value("tr_eval_out", str(self.runs / "val" / f"stage{stage}"))
        if dpg.does_item_exist("tr_window"):
            dpg.configure_item("tr_window", enabled=meta["needs_window"])
        if dpg.does_item_exist("tr_imb"):
            dpg.configure_item("tr_imb", enabled=meta["needs_imbalance"])
        widgets.refresh_all_indicators(["tr_out", "tr_eval_out"])

    _ANALYZE_DESCS = {
        "timeline": "Visualize the frame-by-frame prediction timeline. Requires a predictions JSON file.",
        "distribution": "Summarize class and action distributions from annotation data. Requires an annotation JSON file.",
        "id_switches": "Detect identity switches across tracks. Requires a predictions JSON file.",
        "confidence": "Inspect score distributions from predictions. Requires a predictions JSON file.",
    }

    def _on_analyze_mode_change(self, mode: str) -> None:
        if dpg.does_item_exist("an_mode_desc"):
            dpg.set_value("an_mode_desc", self._ANALYZE_DESCS.get(mode, ""))

    def _start_train(self):
        stage = self.current_stage
        train_root = dpg.get_value("tr_train")
        val_root = dpg.get_value("tr_val")
        out = dpg.get_value("tr_out")
        if not train_root:
            self.runner.log(f"[ERROR] Stage {stage}: training data is required.")
            return
        if not out:
            self.runner.log(f"[ERROR] Stage {stage}: output folder is required.")
            return

        meta = STAGE_META[stage]
        window = int(dpg.get_value("tr_window")) if meta["needs_window"] else None
        imb = dpg.get_value("tr_imb") if meta["needs_imbalance"] else None
        fmt = dpg.get_value("tr_fmt")
        model_mode = dpg.get_value("tr_model")
        classes_file = dpg.get_value("tr_classes")

        effective_stage = stage
        if model_mode == "detection" and stage != 1:
            self.runner.log(f"[INFO] model=detection selected. Training will use stage=1 instead of stage={stage}.")
            effective_stage = 1
            window = None
            imb = None
        elif model_mode == "detection+temporal" and stage == 1:
            self.runner.log("[INFO] model=detection+temporal selected. Training will use stage=2 instead of stage=1.")
            effective_stage = 2
            if window is None:
                window = 16

        cmd = commands.train_command(
            effective_stage,
            train_root,
            val_root,
            out,
            epochs=int(dpg.get_value("tr_epochs")),
            batch_size=int(dpg.get_value("tr_batch")),
            lr=float(dpg.get_value("tr_lr")),
            window_size=window,
            imbalance_strategy=imb,
            annotation_format=fmt,
            classes_file=classes_file,
        )
        self.attach_monitor(out, effective_stage)
        self.runner.run(cmd)

    def _start_eval(self):
        stage = self.current_stage
        ckpt = dpg.get_value("tr_ckpt")
        if not ckpt:
            self.runner.log(f"[ERROR] Stage {stage}: evaluation checkpoint is required.")
            return
        cmd = commands.val_command(stage, ckpt, dpg.get_value("tr_eval_val"), dpg.get_value("tr_eval_out"))
        self.runner.run(cmd)

    def _start_analyze(self):
        mode = dpg.get_value("an_mode")
        pred = dpg.get_value("an_pred")
        anno = dpg.get_value("an_anno")
        missing = commands.analyze_missing_field(mode, pred, anno)
        if missing:
            self.runner.log(f"[ERROR] mode={mode} requires {missing}=...")
            return
        cmd = commands.analyze_command(mode, dpg.get_value("an_out"), predictions=pred, annotation=anno)
        self.runner.run(cmd)

    def _build_monitor(self):
        dpg.add_text("Live Run Monitor", color=(220, 204, 128))
        dpg.add_text(
            "Attach the monitor to a run folder to watch loss and metrics update while training is running.",
            color=(172, 178, 186),
            wrap=520,
        )
        dpg.add_spacer(height=6)
        dpg.add_text("", tag="mon_summary", color=(184, 224, 186), wrap=520)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="mon_dir", width=370, default_value=str(self.runs / "train" / "stage1"))
            dpg.add_button(label="Watch Folder", callback=self._attach_monitor_manual)
        with dpg.plot(label="Loss", height=180, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_loss_x")
            with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="mon_loss_y"):
                for tag, label in _LOSS_SERIES:
                    dpg.add_line_series([], [], label=label, tag=f"mon_s_{tag}")
        with dpg.plot(label="Metrics", height=180, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_metric_x")
            with dpg.plot_axis(dpg.mvYAxis, label="value", tag="mon_metric_y"):
                for col in _METRIC_SERIES:
                    dpg.add_line_series([], [], label=col, tag=f"mon_s_{col}")

    def attach_monitor(self, run_dir: str, stage: int):
        if not run_dir:
            return
        if self.monitor is not None:
            self.monitor.stop()
        self.monitor_stage = stage
        dpg.set_value("mon_dir", run_dir)
        self.monitor = TrainingMonitor(run_dir, self._on_monitor_update, interval=2.0)
        self.monitor.start()

    def _attach_monitor_manual(self):
        run_dir = dpg.get_value("mon_dir")
        stage = self.monitor_stage
        for s in (1, 2, 3, 4):
            if f"stage{s}" in str(run_dir):
                stage = s
        self.attach_monitor(run_dir, stage)

    def _on_monitor_update(self, rows: List[Dict[str, str]], latest: Optional[dict]):
        for tag, _label in _LOSS_SERIES:
            xs, ys = series(rows, "epoch", tag)
            if dpg.does_item_exist(f"mon_s_{tag}"):
                dpg.set_value(f"mon_s_{tag}", [xs, ys])
        active_cols = {c for c, _ in metric_columns_for_stage(self.monitor_stage)}
        for col in _METRIC_SERIES:
            xs, ys = series(rows, "epoch", col) if col in active_cols else ([], [])
            if dpg.does_item_exist(f"mon_s_{col}"):
                dpg.set_value(f"mon_s_{col}", [xs, ys])
        for ax in ("mon_loss_x", "mon_loss_y", "mon_metric_x", "mon_metric_y"):
            if dpg.does_item_exist(ax):
                dpg.fit_axis_data(ax)
        if latest and dpg.does_item_exist("mon_summary"):
            dpg.set_value(
                "mon_summary",
                f"Epoch {latest.get('epoch', '?')} | "
                f"Primary metric: {latest.get('primary_metric', '?')} = {latest.get('primary_metric_value', '?')} | "
                f"Best so far: {latest.get('best_so_far', '?')}",
            )

    def teardown(self):
        if self.monitor is not None:
            self.monitor.stop()
