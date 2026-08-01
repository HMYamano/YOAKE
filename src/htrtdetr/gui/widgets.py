"""
widgets.py — Reusable Dear PyGui UI helpers for the YOAKE GUI.

Provides a consistent look-and-feel across panels:

- :func:`path_field` — a file/folder input row with a colored existence
  indicator, Browse button, and Recent-paths popup.
- :func:`tooltip` — hover-tooltip helper attached to the previously added
  widget.
- :func:`section_header` — a colored section title with a divider.
- :func:`primary_button` / :func:`secondary_button` — thin wrappers that keep
  button sizing and coloring uniform.

Colors and label conventions are shared so every panel behaves the same way
(green dot = file exists, red = missing, grey = empty; asterisk marks a
required field). Small footprint, no side effects on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import dearpygui.dearpygui as dpg

from . import recents

# ─── shared colors ────────────────────────────────────────────────────────
STATUS_OK = (120, 200, 130, 255)
STATUS_MISS = (208, 110, 110, 255)
STATUS_EMPTY = (110, 118, 130, 255)

LABEL_REQUIRED = (232, 214, 152)
LABEL_NORMAL = (204, 210, 220)
NOTE_COLOR = (170, 176, 184)
SECTION_COLOR = (166, 205, 241)
HINT_COLOR = (150, 210, 240)
PRIMARY_BG = (60, 128, 190)
PRIMARY_BG_HOVER = (78, 152, 214)


# ─── tooltips ─────────────────────────────────────────────────────────────
def tooltip(text: str, wrap: int = 320) -> None:
    """Attach a tooltip to the most recently added widget.

    Call this immediately after the widget you want to annotate. Safe if the
    parent has already been claimed by another tooltip (no-op in that case).
    """
    parent = dpg.last_item()
    if parent is None or not dpg.does_item_exist(parent):
        return
    try:
        with dpg.tooltip(parent):
            dpg.add_text(text, wrap=wrap, color=(212, 220, 232))
    except SystemError:
        # dearpygui refuses duplicate tooltips on the same parent
        pass


# ─── section header ───────────────────────────────────────────────────────
def section_header(label: str, description: Optional[str] = None,
                   wrap: int = 620) -> None:
    dpg.add_spacer(height=4)
    dpg.add_text(label, color=SECTION_COLOR)
    dpg.add_separator()
    if description:
        dpg.add_text(description, color=NOTE_COLOR, wrap=wrap)
        dpg.add_spacer(height=2)


# ─── file-existence indicator ─────────────────────────────────────────────
def _status_color(value: str):
    if not value:
        return STATUS_EMPTY
    try:
        p = Path(value).expanduser()
    except (OSError, ValueError):
        return STATUS_EMPTY
    return STATUS_OK if p.exists() else STATUS_MISS


def _refresh_indicator(status_tag: str, value: str) -> None:
    if dpg.does_item_exist(status_tag):
        dpg.configure_item(status_tag, color=_status_color(value))


def refresh_all_indicators(tags):
    """Utility for panels: refresh a batch of status dots after programmatic set."""
    for tag in tags:
        status_tag = f"{tag}__status"
        if dpg.does_item_exist(tag) and dpg.does_item_exist(status_tag):
            _refresh_indicator(status_tag, dpg.get_value(tag))


# ─── path field ───────────────────────────────────────────────────────────
def path_field(
    label: str,
    tag: str,
    *,
    field_key: Optional[str] = None,
    default: str = "",
    hint: str = "",
    browse: Optional[Callable[[], str]] = None,
    required: bool = False,
    tooltip_text: Optional[str] = None,
    width: int = 440,
) -> None:
    """Standard file/folder input row.

    Layout::

        Label*:  [tooltip on hover]
        ● [ text input ..................... ] [Browse] [Recent]

    Parameters
    ----------
    label : str
        Shown above the row. If ``required`` is true an asterisk is appended
        and the label is colored to draw attention.
    tag : str
        Dearpygui tag of the ``input_text`` widget. Reused for the associated
        status indicator (``<tag>__status``).
    field_key : str, optional
        Identifier used by the persistent recent-files store. When provided,
        a "Recent" popup button is added and picking a Browse target
        automatically records it.
    default : str, optional
        Initial value for the text input.
    hint : str, optional
        Placeholder shown when the input is empty.
    browse : callable, optional
        Callback that opens a native file dialog and returns the chosen path
        (or empty string).
    required : bool, optional
        Marks the field visually as required; does not enforce anything.
    tooltip_text : str, optional
        Hover text for the label.
    width : int, optional
        Width of the text input.
    """
    marker = " *" if required else ""
    dpg.add_text(f"{label}{marker}:", indent=4,
                 color=LABEL_REQUIRED if required else LABEL_NORMAL)
    if tooltip_text:
        tooltip(tooltip_text)

    status_tag = f"{tag}__status"
    with dpg.group(horizontal=True):
        dpg.add_text("●", tag=status_tag, color=_status_color(default))
        tooltip("Green = file exists  |  Red = not found  |  Grey = empty",
                wrap=260)

        dpg.add_input_text(
            tag=tag,
            default_value=default,
            hint=hint,
            width=width,
            callback=lambda s, a, u: _refresh_indicator(status_tag, a),
        )

        if browse is not None:
            def _do_browse(_s=None, _a=None, _u=None):
                path = browse()
                if not path:
                    return
                dpg.set_value(tag, path)
                _refresh_indicator(status_tag, path)
                if field_key:
                    recents.add(field_key, path)

            dpg.add_button(label="Browse", width=70, callback=_do_browse)

        if field_key:
            def _open_recent(_s=None, _a=None, _u=None):
                items = recents.get(field_key)
                popup_tag = f"{tag}__recent_popup"
                if dpg.does_item_exist(popup_tag):
                    dpg.delete_item(popup_tag)
                with dpg.window(
                    label="Recent Files", tag=popup_tag,
                    modal=True, width=640, height=340, pos=[380, 220],
                ):
                    if not items:
                        dpg.add_text(
                            "No recent paths yet. Use the Browse button once "
                            "and this list will remember it.",
                            color=NOTE_COLOR, wrap=600,
                        )
                    else:
                        dpg.add_text("Pick a recent path:", color=HINT_COLOR)
                        dpg.add_separator()
                        for p in items:
                            with dpg.group(horizontal=True):
                                def _pick(_s, _a, path=p):
                                    dpg.set_value(tag, path)
                                    _refresh_indicator(status_tag, path)
                                    recents.add(field_key, path)
                                    dpg.delete_item(popup_tag)
                                dpg.add_button(label="Use", width=60,
                                               callback=_pick)
                                dpg.add_text(p, color=(210, 218, 230),
                                             wrap=540)
                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Clear History",
                            callback=lambda: (
                                recents.clear(field_key),
                                dpg.delete_item(popup_tag),
                            ),
                        )
                        dpg.add_button(
                            label="Close",
                            callback=lambda: dpg.delete_item(popup_tag),
                        )

            dpg.add_button(label="Recent", width=70, callback=_open_recent)
            tooltip("Show recently used paths for this field.", wrap=200)


# ─── button helpers ───────────────────────────────────────────────────────
def _primary_theme_tag() -> str:
    tag = "yoake_primary_btn_theme"
    if dpg.does_item_exist(tag):
        return tag
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, PRIMARY_BG)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, PRIMARY_BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (48, 108, 168))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
    return tag


def primary_button(label: str, callback, *, height: int = 36, width: int = -1,
                   tag: Optional[str] = None) -> None:
    """Colored, prominent button for the panel's primary action."""
    kwargs = {"label": label, "height": height, "width": width,
              "callback": callback}
    if tag:
        kwargs["tag"] = tag
    dpg.add_button(**kwargs)
    dpg.bind_item_theme(dpg.last_item(), _primary_theme_tag())


def secondary_button(label: str, callback, *, height: int = 32,
                     width: int = -1, tag: Optional[str] = None) -> None:
    kwargs = {"label": label, "height": height, "width": width,
              "callback": callback}
    if tag:
        kwargs["tag"] = tag
    dpg.add_button(**kwargs)
