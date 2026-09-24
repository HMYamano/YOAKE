"""Project batch form and progress; all callbacks execute on the UI thread."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from ..annotation import sampling
from ..annotation.batch import BatchOptions
from ..annotation.project import SCOPES
from . import auto_label, theme, widgets


def review_label(candidates: int, issues: int) -> str:
    """The finished-run button's label, or ``""`` when there is nothing to do.

    The count is in the label because the button is the only place the run's
    outcome and the next action meet: "Review 412 candidates" says both what
    happened and what to press. An empty string is the caller's signal to show
    no button at all.
    """
    parts = []
    if candidates > 0:
        parts.append(f"{candidates:,} candidate{'s' if candidates != 1 else ''}")
    if issues > 0:
        parts.append(f"{issues:,} location{'s' if issues != 1 else ''}")
    return "Review " + " and ".join(parts) if parts else ""


class BatchDialog:
    """The project-wide Auto Annotate / Smart Sample form.

    The form is long (three combos, six toggles, an Advanced block) and the
    progress it reports arrives at the bottom, so it is built as a scrolling
    body with Run / Cancel and the two progress bars pinned under it. With
    everything in one flow the Run button sat at roughly y=640 of a 740 px
    window and the results box below the bottom edge entirely.
    """

    #: Preferred size. :func:`widgets.place_dialog` shrinks it to whatever the
    #: viewport actually has.
    WIDTH, HEIGHT = 660, 720
    #: Run / Cancel, two progress bars, the current-operation line, the
    #: results box and the review button a finished run offers, none of which
    #: may scroll away while a batch is running.
    FOOTER_H = 336

    def __init__(self, app):
        self.app = app
        self.tag = "project_batch"
        self.running = False

    def show(self, mode="auto"):
        if dpg.does_item_exist(self.tag):
            if self.running:
                widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
                dpg.show_item(self.tag)
                return
            dpg.delete_item(self.tag)
        settings = self.app.project.settings
        with dpg.window(
            tag=self.tag,
            label="Auto Annotate / Smart Sample Project",
            width=self.WIDTH,
            height=self.HEIGHT,
        ):
            with widgets.dialog_body("batch_body", footer=self.FOOTER_H):
                dpg.add_combo(
                    list(SCOPES),
                    tag="batch_scope",
                    width=-1,
                    default_value=(
                        "Selected videos" if self.app.project.selected_videos else "Entire project"
                    ),
                )
                dpg.add_checkbox(
                    label="Generate object candidates",
                    tag="batch_detect",
                    default_value=mode == "auto",
                )
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        label="Checkpoint",
                        tag="batch_weights",
                        width=390,
                        default_value=settings.get("weights", ""),
                        callback=self._gate,
                    )
                    dpg.add_button(label="Browse...", callback=self._weights)
                dpg.add_input_text(
                    label="Config (optional)",
                    tag="batch_config",
                    width=390,
                    default_value=settings.get("config", ""),
                )
                dpg.add_text("", tag="batch_reason", color=theme.TEXT_DIM, wrap=610)
                # A project that does not follow individuals has nothing to
                # connect detections into, so the option is locked rather than
                # offered and then quietly ignored.
                tracks = bool(settings.get("use_track_ids", True))
                dpg.add_checkbox(
                    label="Connect detections into tracklets",
                    tag="batch_track",
                    default_value=tracks,
                )
                if not tracks:
                    widgets.locked(
                        "batch_track",
                        "This project does not follow individuals across "
                        "frames. Project > Tracking... turns track IDs on.",
                    )
                dpg.add_checkbox(
                    label="Select informative frames / clips",
                    tag="batch_sample",
                    default_value=True,
                )
                dpg.add_input_int(
                    label="Total samples",
                    tag="batch_count",
                    default_value=100,
                    min_value=1,
                    min_clamped=True,
                    width=140,
                )
                dpg.add_combo(
                    ["Individuals", "Behavior", "Individuals + behavior"],
                    tag="batch_target",
                    default_value="Individuals + behavior",
                    width=240,
                )
                dpg.add_combo(
                    list(sampling.STRATEGY_LABELS.values()),
                    tag="batch_strategy",
                    default_value=sampling.STRATEGY_LABELS[sampling.DIVERSITY],
                    width=300,
                )
                dpg.add_checkbox(
                    label="Balance samples across videos", tag="batch_balance", default_value=True
                )
                with dpg.collapsing_header(label="Advanced"):
                    dpg.add_input_int(
                        label="Scan stride",
                        tag="batch_stride",
                        default_value=15,
                        min_value=1,
                        min_clamped=True,
                        width=140,
                    )
                    dpg.add_input_int(
                        label="Random seed", tag="batch_seed", default_value=0, width=140
                    )
                    dpg.add_input_float(
                        label="Candidate score threshold",
                        tag="batch_threshold",
                        default_value=0.3,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                        width=140,
                    )
                    dpg.add_combo(
                        ["cpu", "cuda"],
                        label="Device",
                        tag="batch_device",
                        default_value="cpu",
                        width=140,
                    )
            widgets.dialog_footer()
            dpg.add_text("Results stay as suggestions until you Accept them.", color=theme.TEXT_DIM)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Run", tag="batch_run", callback=self._run, width=140, height=32
                )
                dpg.add_button(
                    label="Cancel",
                    tag="batch_cancel",
                    callback=self.app.stop_project_batch,
                    enabled=False,
                    width=140,
                    height=32,
                )
            dpg.add_text("Ready", tag="batch_current", wrap=610)
            dpg.add_progress_bar(tag="batch_video_progress", width=-1)
            dpg.add_progress_bar(tag="batch_overall_progress", width=-1)
            dpg.add_input_text(
                tag="batch_results", multiline=True, readonly=True, height=84, width=-1
            )
            # A run that ends in a count is a dead end: the user is told 412
            # candidates exist and left to find Review Next on their own. The
            # finished state offers the next action instead, and only when
            # there is one -- a button that does nothing when pressed teaches
            # that the feature is broken (GUI_DESIGN.md P6).
            widgets.primary_button("Review", self._review, tag="batch_review", height=34, width=-1)
            dpg.configure_item("batch_review", show=False)
            dpg.add_text("", tag="batch_followup", color=theme.TEXT_DIM, wrap=610, show=False)
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        self._gate()

    def _review(self, *_a):
        """Close, and take the user to the first thing this run produced."""
        dpg.hide_item(self.tag)
        self.app.review_next_project()

    def _weights(self):
        from .file_dialogs import open_file as _tk_open_file

        path = _tk_open_file(
            "Detector checkpoint", [("Checkpoint", "*.pth *.pt"), ("All files", "*.*")]
        )
        if path is not None:
            dpg.set_value("batch_weights", str(path))
        self._gate()

    def _gate(self):
        from .sam_session import repo_root as _repo_root

        status = auto_label.check_backend(_repo_root(), dpg.get_value("batch_weights") or "")
        dpg.configure_item("batch_detect", enabled=status.ok)
        if not status.ok:
            dpg.set_value("batch_detect", False)
        dpg.set_value("batch_reason", status.message)

    def _run(self):
        target = {
            "Individuals": sampling.TARGET_DETECTION,
            "Behavior": sampling.TARGET_BEHAVIOR,
            "Individuals + behavior": sampling.TARGET_BOTH,
        }[dpg.get_value("batch_target")]
        strategy = next(
            k for k, v in sampling.STRATEGY_LABELS.items() if v == dpg.get_value("batch_strategy")
        )
        options = BatchOptions(
            detect=dpg.get_value("batch_detect"),
            weights=dpg.get_value("batch_weights"),
            config=dpg.get_value("batch_config"),
            device=dpg.get_value("batch_device"),
            score_threshold=dpg.get_value("batch_threshold"),
            connect_tracks=(
                dpg.get_value("batch_track")
                and self.app.project.settings.get("use_track_ids", True)
            ),
            sample=dpg.get_value("batch_sample"),
            stride=max(1, dpg.get_value("batch_stride")),
            balanced=dpg.get_value("batch_balance"),
            request=sampling.SamplingRequest(
                target=target,
                budget=sampling.Budget(samples=max(1, dpg.get_value("batch_count"))),
                strategy=strategy,
                seed=dpg.get_value("batch_seed"),
                prelabelled=dpg.get_value("batch_detect"),
            ),
            quality=self.app.quality_config,
        )
        self.app.project.settings.update(weights=options.weights, config=options.config)
        if self.app.start_project_batch(dpg.get_value("batch_scope"), options):
            self.running = True
            dpg.configure_item("batch_run", enabled=False)
            dpg.configure_item("batch_cancel", enabled=True)
            dpg.set_value("batch_results", "")
            dpg.configure_item("batch_review", show=False)
            dpg.configure_item("batch_followup", show=False)

    def event(self, event):
        if not dpg.does_item_exist(self.tag):
            return
        if event["kind"] == "progress":
            dpg.set_value(
                "batch_current",
                f"Processing {event['index']} / {event['total']}: {event['video_id']}\n{event['operation']}",
            )
            dpg.set_value("batch_video_progress", event["fraction"])
            dpg.set_value("batch_overall_progress", event["overall"])
        elif event["kind"] == "video":
            r = event["result"]
            line = f"{r.video_id}: {r.status}" + (f" — {r.error}" if r.error else "")
            dpg.set_value("batch_results", dpg.get_value("batch_results") + line + "\n")
        elif event["kind"] == "finished":
            self.running = False
            dpg.configure_item("batch_run", enabled=True)
            dpg.configure_item("batch_cancel", enabled=False)
            results = event["results"]
            summary = ", ".join(
                f"{sum(r.status == status for r in results)} {status}"
                for status in ("completed", "cached", "failed")
            )
            dpg.set_value(
                "batch_current", ("Cancelled. " if event["cancelled"] else "Finished. ") + summary
            )
            if not event["cancelled"]:
                dpg.set_value("batch_overall_progress", 1.0)
            # Cancel keeps whatever the finished videos produced, so the offer
            # is made on the counts, not on how the run ended.
            self._offer_review()

    def _offer_review(self):
        """Turn the finished state into the one action that follows from it."""
        label = review_label(*self.app.project_review_counts())
        dpg.configure_item("batch_review", label=label or "Review", show=bool(label))
        dpg.configure_item("batch_followup", show=not label)
        if not label:
            dpg.set_value(
                "batch_followup",
                "Nothing to review. No candidates and no quality issues were found.",
            )
