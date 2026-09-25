"""A five-step guide stored per project, with a persistent opt-out."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from . import theme, widgets

#: The five cards, in the order the work actually happens.
#:
#: They used to open with "Press W to draw a box", which is the right advice
#: for one frame and the wrong advice for a project: this window appears
#: immediately after Create Project, with video 1 parked at frame 0, and a
#: study of twenty one-hour videos holds more than a million frames. Starting
#: at frame 0 and drawing forwards is how a first session produces two hundred
#: near-identical labels. So card 1 is the sampler, and the whole sequence is
#: the labelling loop - which closes without a trained model, and says so,
#: because "no checkpoint yet" is every new project's starting state. Card 5
#: is the arrow out: the loop closes without a model, and the two steps that
#: make one are a window away, which nothing in here used to say.
STEPS = (
    (
        "Pick the frames worth labeling",
        "Press Ctrl+R. YOAKE reads the videos and chooses frames that differ "
        "from each other, so your first hundred labels are a hundred "
        "different situations instead of a hundred neighbouring frames. "
        "You do not need a trained model for this.",
    ),
    (
        "Find the animals on a frame",
        "Press C and click the centre of an animal to fit a box. Assistance "
        "starts automatically on the first click; if unavailable, basic fitting "
        "still works. Adjust the handles, or press W to draw by hand. Set the "
        "class and behavior in the Inspector. Find all at once is optional.",
    ),
    (
        "Carry an animal forward",
        "Select a box and press Continue to follow that animal into the next "
        "frames, or use Review > Continue every animal across this clip to do "
        "a whole clip in one job. What comes back is dashed and unsaved until "
        "you accept it.",
    ),
    (
        "Check it, then go round again",
        "Check dashed candidates in Review: accept or discard. Save with Ctrl+S. "
        "Use Ctrl+Right to go to the next picked sample. Then press Ctrl+R "
        "again - the next round avoids frames like the ones you just did. "
        "Label in rounds of 20-50 frames: what you label changes what you are "
        "asked next, and that only happens between rounds.",
    ),
    (
        "Train, here, then go round again",
        "Train in the top bar builds a detector from the labels you already "
        "have - in this window, no form to fill in - and makes it this "
        "project's model. Analyze then runs it over the videos and brings back "
        "what it was least sure about. That is the round: pick, label, train, "
        "analyze, pick again. Neither is needed to get this far; a model makes "
        "the loop faster rather than possible.",
    ),
)


def show_guide(app, step=0):
    if dpg.does_item_exist("guided_annotation"):
        dpg.delete_item("guided_annotation")
    title, body = STEPS[step]
    # Taller than it was: the cards now carry the reason a step exists, not
    # just the key that performs it, and a card that scrolls is a card whose
    # second half is never read.
    width, height = 470, 320
    with dpg.window(
        tag="guided_annotation",
        label=f"Quick Start — {step + 1} / {len(STEPS)}",
        width=width,
        height=height,
    ):
        # The four buttons are the whole point of a wizard, so they are pinned
        # and only the step's text scrolls.
        with widgets.dialog_body("guide_body", footer=46):
            dpg.add_text(title, color=theme.ACCENT)
            dpg.add_text(body, wrap=width - 40)
            # No "do not show again" checkbox: reaching the last card and
            # pressing *Start annotating* already means it, and Close already
            # means "not now". A checkbox that duplicates both buttons is a
            # third answer to a two-answer question. Help > Guided Annotation
            # brings it back.
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
