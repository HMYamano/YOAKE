"""The top bar and the tool rail: the two always-visible rows of controls.

Lifted out of :mod:`htrtdetr.annotator` for the same reason the menu bar was
(:mod:`htrtdetr.gui.menubar`): neither of these has state of its own. They are
a list of buttons bound to methods of the app, and they are the part of the
window a first-time user reads first, so they are worth being able to read on
their own page.

``app`` is the :class:`~htrtdetr.annotator.AnnotationApp`. The tool table and
the rail's geometry live here too, because they describe the rail and nothing
outside it sets them; ``annotator`` re-exports them under their old private
names so existing call sites and tests are unchanged.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["TOOLS", "RAIL_W", "RAIL_BTN_W", "topbar", "toolrail"]

# dearpygui, theme and widgets are imported inside the two builders rather
# than here. ``annotator`` reads :data:`TOOLS` and :data:`RAIL_W` at module
# level and is required to import on a Python with no dearpygui at all (its
# own dpg import is in a try/except for exactly that), so this module must
# carry the data without dragging the toolkit in behind it.

#: ``(tool key, full name, rail word, shortcut action)``. The rail word is
#: short because the rail is 108 px wide; the full name is what the tooltip
#: and the canvas chip say.
TOOLS: Tuple[Tuple[str, str, str, str], ...] = (
    ("select", "Select", "Select", "tool.select"),
    ("draw", "Draw Box", "Box", "tool.draw_box"),
    ("obb", "Draw Rotated Box", "Rotate", "tool.draw_rotated"),
    ("click", "Click animal", "Click", "tool.click_box"),
    ("pan", "Pan", "Pan", "tool.pan"),
    ("delete", "Delete Box", "Delete", "tool.delete"),
)

#: Tool rail width.
#:
#: Wide enough for the label the rail actually draws. It was 60, giving 44 px
#: buttons; the theme sets ``FramePadding.x = 10``, so ImGui clipped every
#: label to 24 px and the 49 px ``"[k] (V)"`` rendered as ``"[k]"`` with the
#: key silently cut off. The rail advertised its shortcuts and showed none of
#: them, and the ASCII icon left behind said nothing on its own. This leaves
#: room for the widest label (``"Select V"``) plus the vertical scrollbar the
#: rail grows at the minimum window size, which is what turns the rail into
#: something a first-time user can read.
RAIL_W = 108
#: Rail button width: the rail minus its window padding and minus the width of
#: a scrollbar, so a button is never half-covered by one when it appears.
RAIL_BTN_W = RAIL_W - 26


# ─── Top bar ─────────────────────────────────────────────────────────────
def topbar(app) -> None:
    """Row one is *where am I*. Row two is *the loop*, left to right.

    The loop is the product (``docs/GUI_DESIGN.md`` §0.2.7): pick frames,
    review what came back, train on what you kept, analyze with what you
    trained, come back here. Until this row existed the annotator showed two
    of those arrows and the other two lived in a different window that nothing
    here mentioned - so a first project ended at "I have a hundred labels" with
    no visible next step.

    Save state is a chip rather than a dialog because it must be true at a
    glance and never require an action to read (§38). It sits in row one with
    the other two facts about the session, not in the row of verbs: a row that
    mixes *what is true* with *what to press* makes the user read every item to
    find out which kind it is.
    """
    import dearpygui.dearpygui as dpg

    from . import shortcuts, theme, widgets

    with dpg.child_window(tag="topbar", height=68, border=False, no_scrollbar=True):
        # ── row one: which video, and the two facts about the session ──
        # Fixed-width items first, elastic text last. With the identity text
        # in the middle the two chips were pushed off the right edge by a long
        # project name - and a save-state chip you cannot see is worse than no
        # save-state chip.
        with dpg.group(horizontal=True):
            widgets.toolbar_button(
                "Videos",
                lambda *_: app.show_video_browser(),
                tag="top_videos",
                height=26,
                help_text="Every video in this project: switch, search, "
                "filter by state, and run a batch over a selection.",
            )
            widgets.toolbar_button(
                "Previous Video",
                lambda *_: app.adjacent_video(-1),
                tag="top_prev_video",
                height=26,
                help_text="Open the previous video in project order, at the "
                "frame you left it on.",
            )
            widgets.toolbar_button(
                "Next Video",
                lambda *_: app.adjacent_video(1),
                tag="top_next_video",
                height=26,
                help_text="Open the next video in project order, at the frame "
                "you left it on.",
            )
            widgets.status_pill("top_saved", "No changes", "idle")
            # Whether a model exists decides what half the project menu can
            # do, so it is read off the top bar rather than discovered as a
            # grey checkbox inside a dialog. The chip is the way there too.
            widgets.status_pill(
                "top_model",
                "Model: none",
                "idle",
                callback=lambda *_: app.open_settings("model"),
            )
            widgets.attach_tooltip(
                "top_model",
                "The detector checkpoint this project pre-labels with. "
                "Nothing in labelling waits for it. Click to choose one.",
            )
            # The third fact: what the machine is doing, and whether anything
            # has gone wrong. It is a chip for the same reason the other two
            # are - it must be readable without acting - and it is the door to
            # the Activity window, which is where the sentence behind the
            # count lives. Everything long in this app runs somewhere the user
            # cannot see; this is the one place that admits it.
            widgets.status_pill(
                "top_activity",
                "Activity",
                "idle",
                callback=lambda *_: app.toggle_activity(),
            )
            widgets.attach_tooltip(
                "top_activity",
                "What is running, and anything that went wrong. Click for the "
                "full log of this session and the run logs the trainer and "
                "the detector wrote.",
            )
            dpg.add_text("Project: untitled | No video open", tag="workspace_identity")

        # ── row two: the loop, then the edit keys, then settings ─────────
        # No hand-picked widths in this row. dearpygui sizes a button with
        # width=0 to its own label, and the row has to survive the 900 px
        # minimum viewport: with fixed widths the last buttons were pushed
        # past the right edge and could not be reached at all until the
        # window was made wider.
        with dpg.group(horizontal=True):
            widgets.toolbar_button(
                "Pick frames to label",
                lambda *_: app.open_smart_annotate(),
                tag="top_smart",
                height=26,
                shortcut=shortcuts.label_for("auto.annotate"),
                help_text="Step 1 of the loop. Analyze the videos and choose "
                "the frames and clips worth your time, instead of scrolling "
                "for somewhere to start. No trained model needed.",
            )
            widgets.toolbar_button(
                "Review",
                app.toggle_inbox,
                tag="top_review",
                height=26,
                help_text="Step 2. Show the places the checks say are worth a "
                "second look, and the boxes waiting to be accepted.",
            )
            dpg.add_spacer(width=10)
            # Steps 3 and 4, and they happen HERE. They used to spawn the
            # workflow window, which made the loop a thing you left the
            # labelling window to continue - so in practice nobody continued
            # it. Train builds a detector from the labels that exist now;
            # Analyze runs it back over the videos and fills the Review Inbox.
            # The workflow window keeps the full training run, which is a
            # different job done once.
            widgets.toolbar_button(
                "Train",
                lambda *_: app.open_train_round(),
                tag="top_train",
                height=26,
                help_text="Step 3. Train a detector on the labels you have so "
                "far, here, and make it this project's model. Every number is "
                "worked out from the project; you choose how long to wait.",
            )
            widgets.toolbar_button(
                "Analyze",
                lambda *_: app.open_smart_annotate(preset="analyze"),
                tag="top_analyze",
                height=26,
                help_text="Step 4. Run this project's model over the videos, "
                "rank what it was least sure about, and bring the boxes back "
                "here as candidates to accept or reject.",
            )
            dpg.add_spacer(width=10)
            widgets.toolbar_button(
                "Undo",
                app.undo,
                tag="top_undo",
                height=26,
                shortcut=shortcuts.label_for("edit.undo"),
                help_text=shortcuts.help_for("edit.undo"),
            )
            widgets.toolbar_button(
                "Redo",
                app.redo,
                tag="top_redo",
                height=26,
                shortcut=shortcuts.label_for("edit.redo"),
                help_text=shortcuts.help_for("edit.redo"),
            )
            widgets.toolbar_button(
                "Save",
                app.save_annotation,
                tag="top_save",
                height=26,
                shortcut=shortcuts.label_for("file.save"),
                help_text=shortcuts.help_for("file.save"),
            )
            dpg.add_spacer(width=10)
            widgets.toolbar_button(
                "Settings",
                lambda *_: app.open_settings(),
                tag="top_settings",
                height=26,
                help_text="Project folder, labels, tracking, the model, click "
                "assistance, and what the checks flag for review.",
            )
            dpg.add_spacer(width=12)
            # The active-tool sentence, which used to sit in the sidebar 600 px
            # from where the user is looking. The canvas draws it too; this is
            # the version that survives a full-screen image.
            dpg.add_text("", tag="tool_mode_text", color=theme.TEXT_DIM)


# ─── Tool rail ───────────────────────────────────────────────────────────
def toolrail(app) -> None:
    """Three captioned groups: the mouse, the assistance, and the view.

    A rail rather than a toolbar row because the tool is the thing the user
    switches most and the canvas is the thing they look at: a vertical rail
    beside the image is one saccade away, a horizontal strip above it is two.

    The captions are the change that matters for a first session. The rail was
    eleven identical buttons in one column, and two of them - *Continue* and
    *Keep on* - are not tools at all: pressing them does not change what the
    next click does. A new user reading top to bottom had no way to know that,
    and the two that cost nothing to try sat directly under *Delete Box*,
    which does not.

    Zoom in and Zoom out are gone from the rail. The wheel zooms over the
    canvas, ``+`` and ``-`` zoom from the keyboard, and the readout at the
    bottom says which of those has happened - three channels for one thing did
    not need a fourth that also cost two of the rail's eleven rows.
    """
    import dearpygui.dearpygui as dpg

    from . import shortcuts, theme, widgets

    _cw, ch = app._canvas_size()
    button_w = RAIL_BTN_W

    def rail_label(word: str, action: str) -> str:
        """``"Select V"``. The key on the button, never only in a tooltip."""
        key = shortcuts.label_for(action)
        return f"{word} {key}" if key else word

    def caption(text: str) -> None:
        dpg.add_text(text, color=theme.TEXT_FAINT)

    # Scrollable, because the rail is taller than the canvas at the minimum
    # window size (900x600) and it used to be clipped there with no scrollbar:
    # Fit and the zoom readout simply did not exist until the user made the
    # window bigger, and nothing said so.
    with dpg.child_window(
        tag="toolrail", width=RAIL_W, height=ch, border=False, no_scrollbar=False
    ):
        caption("Tools")
        for tool, label, word, action in TOOLS:
            widgets.toolbar_button(
                rail_label(word, action),
                (lambda t=tool: (lambda *_: app.set_tool(t)))(),
                tag=f"rail_{tool}",
                width=button_w,
                height=32,
                active=(app._tool == tool),
                # hint_for, not help_for: it lists *every* key bound to the
                # action, so the E / R pair on Draw Rotated Box is
                # discoverable from the button that owns it.
                help_text=f"{label}. "
                f"{shortcuts.hint_for(action) or shortcuts.help_for(action)}",
            )

        dpg.add_separator()
        widgets.gap(1)
        caption("Assist")
        widgets.toolbar_button(
            "Continue",
            lambda *_: app.continue_animal(),
            tag="rail_propagate",
            width=button_w,
            height=32,
            help_text="Continue this animal into the following frames: "
            "tracking, accurate propagation, or a copy to the next frame, "
            "whichever this machine can do. Select a box first.",
        )
        widgets.toolbar_button(
            "Keep tool",
            lambda *_: app.toggle_sticky_draw(),
            tag="rail_sticky",
            width=button_w,
            height=26,
            active=app._sticky_draw,
            help_text="Keep the tool active after each box. Off means Draw, "
            "Draw Rotated and Click animal each return to Select once a box "
            "is made, which is what LabelImg does and what leaves the new box "
            "selected and ready to adjust.",
        )

        dpg.add_separator()
        widgets.gap(1)
        caption("View")
        widgets.toolbar_button(
            rail_label("Fit", "view.fit"),
            lambda *_: app.fit_view(),
            tag="rail_view_fit",
            width=button_w,
            height=26,
            help_text="Fit the frame to the window.",
        )
        # Moved off the top bar, which is now the loop and nothing else. It
        # belongs beside Fit: both answer "how much of the image do I want to
        # see", and neither changes a single annotation.
        widgets.toolbar_button(
            "Panels",
            lambda *_: app.toggle_panels(),
            tag="rail_panels",
            width=button_w,
            height=26,
            active=True,
            help_text="Hide the Inspector and the Review Inbox and give the "
            "whole width to the image.",
        )
        dpg.add_text("100%", tag="rail_zoom", color=theme.TEXT_FAINT)
        widgets.attach_tooltip(
            "rail_zoom",
            "Scroll the wheel over the image to zoom, or press "
            f"{shortcuts.label_for('view.zoom_in')} and "
            f"{shortcuts.label_for('view.zoom_out')}. "
            "Middle-drag pans from any tool.",
        )
