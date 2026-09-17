"""A four-step guide stored per project, with a persistent opt-out."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from . import theme, widgets

STEPS = (
    (
        "Find an animal",
        "Press W to draw a box, or C and click the animal. Select its class in the Inspector.",
    ),
    (
        "Continue the same animal",
        "Select a box, then Continue this animal. Tracking suggestions need your review.",
    ),
    (
        "Label behavior",
        "Choose a behavior in the Inspector. The timeline shows behavior intervals for this video.",
    ),
    (
        "Check and save",
        "Review Next moves across videos. Accept or reject suggestions, mark videos Done, then Save or Export.",
    ),
)


def show_guide(app, step=0):
    if dpg.does_item_exist("guided_annotation"):
        dpg.delete_item("guided_annotation")
    title, body = STEPS[step]
    width, height = 470, 260
    with dpg.window(
        tag="guided_annotation",
        label=f"Quick Start — {step+1} / 4",
        width=width,
        height=height,
    ):
        # The four buttons are the whole point of a wizard, so they are pinned
        # and only the step's text scrolls.
        with widgets.dialog_body("guide_body", footer=46):
            dpg.add_text(title, color=theme.ACCENT)
            dpg.add_text(body, wrap=width - 40)
            widgets.gap(1)
            dpg.add_checkbox(
                label="Do not show automatically for this project",
                tag="guide_skip",
                default_value=app.project.settings.get("onboarding_done", False),
                callback=lambda s, a: _remember(app, a),
            )
        dpg.add_separator()
        with dpg.group(horizontal=True):
            if step:
                dpg.add_button(
                    label="Previous", width=100, height=28,
                    callback=lambda: show_guide(app, step - 1),
                )
            if step < len(STEPS) - 1:
                dpg.add_button(
                    label="Next", width=100, height=28,
                    callback=lambda: show_guide(app, step + 1),
                )
            else:
                dpg.add_button(
                    label="Start annotating", width=160, height=28,
                    callback=lambda: _finish(app),
                )
            dpg.add_button(
                label="Close", width=100, height=28,
                callback=lambda: dpg.hide_item("guided_annotation"),
            )
    widgets.place_dialog("guided_annotation", width, height)


def _remember(app, value):
    app.project.settings["onboarding_done"] = bool(value)
    app._mark_dirty()


def _finish(app):
    _remember(app, True)
    dpg.hide_item("guided_annotation")
