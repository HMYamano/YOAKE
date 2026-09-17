"""
widgets.py — The reusable Dear PyGui vocabulary for the YOAKE GUI.

Every colour, radius and spacing value comes from :mod:`htrtdetr.gui.theme`;
this module contains no colour literals. It provides the widgets the four
panels share so they behave identically everywhere:

Fields and text
    :func:`path_field`, :func:`labeled_input`, :func:`labeled_slider`,
    :func:`section_header`, :func:`tooltip`, :func:`help_icon`,
    :func:`empty_state`, :func:`gap`.

Buttons and toggles
    :func:`primary_button`, :func:`secondary_button`, :func:`danger_button`,
    :func:`toolbar_button`, :func:`set_toolbar_active`,
    :func:`segmented_control`.

State and gating
    :func:`locked` / :func:`unlock` (the *locked* pattern: a gated control is
    visible, dimmed, disabled and tooltipped with cause + fix + escape hatch —
    **never hidden**), :func:`status_pill`, :func:`badge`,
    :func:`status_text`, :func:`toast`, :func:`tick_toasts`.

Structure
    :func:`advanced_section`, :func:`kbd_table`, :func:`shortcut_table`,
    :func:`confirm_modal`.

Dialog geometry
    :func:`place_dialog` / :func:`refit_dialogs` keep a window inside the
    viewport, and :func:`dialog_body` / :func:`dialog_footer` split it into a
    part that may scroll and a button row that may not. Every dialog in the
    suite uses them, because a fixed-size window whose content outgrew it hides
    exactly the control the user came for.

Two invariants hold for every function here:

1. **Safe to call twice.** Builders reuse or replace their own tag; mutators
   are idempotent.
2. **Degrades gracefully on a missing tag.** Nothing raises because a tag has
   not been created yet — it returns quietly. A panel that is still building
   can therefore call a mutator without ordering its code around dpg.

Measured on dearpygui 2.3.1: a button with ``enabled=False`` still reports
``is_item_hovered() == True``, so the tooltip attached by :func:`locked`
really does appear. That is the whole reason a locked control can explain
itself.
"""

from __future__ import annotations

import inspect
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import dearpygui.dearpygui as dpg

from . import recents, theme

# ─── shared colors (retargeted onto the design tokens) ────────────────────
# Kept as module constants because three panels import them by name.
STATUS_OK = theme.ACCEPTED
STATUS_MISS = theme.DANGER
STATUS_EMPTY = theme.TEXT_FAINT

LABEL_REQUIRED = theme.TEXT
LABEL_NORMAL = theme.TEXT_DIM
NOTE_COLOR = theme.TEXT_DIM
SECTION_COLOR = theme.ACCENT
HINT_COLOR = theme.TEXT_FAINT
PRIMARY_BG = theme.ACCENT
PRIMARY_BG_HOVER = theme.ACCENT_HOVER

#: Reasons registered by :func:`locked`, keyed by tag. Panels may read this to
#: render the same explanation inline as body text (the Auto-Annotate panel
#: does exactly that, so the reason is visible without hovering).
LOCK_REASONS: Dict[str, str] = {}


# ══════════════════════════════════════════════════════════════════════════
# small internals
# ══════════════════════════════════════════════════════════════════════════
def _exists(tag) -> bool:
    if tag is None or tag == 0:
        return False
    try:
        return bool(dpg.does_item_exist(tag))
    except SystemError:
        return False


def _drop(tag) -> None:
    """Delete an item if it exists, so a builder can be called twice."""
    if _exists(tag):
        try:
            dpg.delete_item(tag)
        except SystemError:
            pass


def _configure(tag, **kwargs) -> None:
    if not _exists(tag):
        return
    try:
        dpg.configure_item(tag, **kwargs)
    except (SystemError, TypeError):
        pass


def _tagged(tag: Optional[str], kwargs: dict) -> dict:
    if tag:
        kwargs["tag"] = tag
    return kwargs


# ─── spacing ──────────────────────────────────────────────────────────────
_GAPS = (theme.SP_XS, theme.SP_S, theme.SP_M, theme.SP_L, theme.SP_XL)


def gap(n: int = 2) -> None:
    """Vertical spacer from the design scale. ``n`` is 0..4 (XS..XL).

    The only sanctioned way to add vertical space. ``add_spacer(height=11)``
    anywhere in the suite is a bug.
    """
    try:
        idx = max(0, min(len(_GAPS) - 1, int(n)))
    except (TypeError, ValueError):
        idx = 2
    dpg.add_spacer(height=_GAPS[idx])


# ══════════════════════════════════════════════════════════════════════════
# tooltips and help
# ══════════════════════════════════════════════════════════════════════════
def tooltip(text: str, wrap: int = 320) -> None:
    """Attach a tooltip to the most recently added widget.

    Call this immediately after the widget you want to annotate. Safe if the
    parent has already been claimed by another tooltip (no-op in that case).
    """
    parent = dpg.last_item()
    if not _exists(parent):
        return
    try:
        with dpg.tooltip(parent):
            dpg.add_text(text, wrap=wrap, color=theme.TEXT)
    except SystemError:
        # dearpygui refuses duplicate tooltips on the same parent
        pass


def attach_tooltip(tag, text: str, wrap: int = 320, tooltip_tag: Optional[str] = None) -> None:
    """Attach (or replace) a tooltip on an arbitrary existing item.

    Unlike :func:`tooltip` this targets a tag rather than the last item, so it
    can be called long after the widget was built — which is what makes
    :func:`locked` able to change a control's explanation at runtime.
    """
    if not _exists(tag) or not text:
        return
    ttag = tooltip_tag or f"{tag}__tip"
    _drop(ttag)
    try:
        with dpg.tooltip(tag, tag=ttag):
            dpg.add_text(text, wrap=wrap, color=theme.TEXT)
    except SystemError:
        pass


def help_icon(text: str, wrap: int = 320) -> None:
    """A dim ``(?)`` marker that reveals ``text`` on hover.

    Use for the explanation a user needs once; use :func:`tooltip` on the
    control itself for the explanation they need every time.
    """
    dpg.add_text("(?)", color=theme.TEXT_FAINT)
    tooltip(text, wrap=wrap)


# ══════════════════════════════════════════════════════════════════════════
# headings and empty states
# ══════════════════════════════════════════════════════════════════════════
def section_header(label: str, description: Optional[str] = None, wrap: int = 620) -> None:
    """A section title with a divider and an optional one-line description."""
    gap(1)
    dpg.add_text(label, color=SECTION_COLOR)
    theme.bind_h2(dpg.last_item())
    dpg.add_separator()
    if description:
        dpg.add_text(description, color=NOTE_COLOR, wrap=wrap)
        gap(0)


def page_title(label: str, subtitle: str = "", wrap: int = 900) -> None:
    """The single 21 px title of a screen, with an optional subtitle."""
    dpg.add_text(label, color=theme.TEXT)
    theme.bind_h1(dpg.last_item())
    if subtitle:
        dpg.add_text(subtitle, color=theme.TEXT_DIM, wrap=wrap)
    gap(2)


def empty_state(
    text: str,
    action_label: str = "",
    action: Optional[Callable] = None,
    *,
    wrap: int = 520,
    tag: Optional[str] = None,
) -> None:
    """The placeholder that replaces a blank rectangle.

    Says what is missing and, when ``action`` is given, offers the one button
    that fixes it. A blank panel is indistinguishable from a broken one; this
    is the cheapest fix for that.
    """
    kwargs = _tagged(tag, {})
    with dpg.group(**kwargs):
        dpg.add_text(text, color=theme.TEXT_DIM, wrap=wrap)
        if action_label and action is not None:
            gap(1)
            dpg.add_button(label=action_label, callback=action, height=30)
            theme.bind(dpg.last_item(), theme.ghost_theme())


# ══════════════════════════════════════════════════════════════════════════
# file-existence indicator
# ══════════════════════════════════════════════════════════════════════════
def _status_state(value: str) -> str:
    """'' (empty) | 'ok' (exists) | 'missing' (typed but not on disk)."""
    if not value:
        return ""
    try:
        return "ok" if Path(value).expanduser().exists() else "missing"
    except (OSError, ValueError):
        return "missing"


def _status_color(value: str):
    return theme.status_dot(_status_state(value))


def _refresh_indicator(status_tag: str, value: str) -> None:
    if _exists(status_tag):
        _configure(status_tag, color=_status_color(value))


def refresh_all_indicators(tags: Iterable[str]) -> None:
    """Utility for panels: refresh a batch of status dots after programmatic set."""
    for tag in tags:
        status_tag = f"{tag}__status"
        if _exists(tag) and _exists(status_tag):
            _refresh_indicator(status_tag, dpg.get_value(tag))


# ══════════════════════════════════════════════════════════════════════════
# path field
# ══════════════════════════════════════════════════════════════════════════
def _accepts_initialdir(fn: Callable) -> bool:
    """True when ``fn`` can be told where to open the picker."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "initialdir" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _start_dir(current: str, field_key: Optional[str]) -> str:
    """Best guess at where the picker should open.

    Order: the folder already typed in the field, then the most recent path
    remembered for this field, then nothing (the OS default). Without this the
    picker always opened at the process CWD, right next to a Recent list that
    already knew the answer.
    """
    for candidate in (current, (recents.get(field_key)[:1] or [""])[0] if field_key else ""):
        if not candidate:
            continue
        try:
            p = Path(candidate).expanduser()
        except (OSError, ValueError):
            continue
        if p.is_dir():
            return str(p)
        if p.parent.is_dir():
            return str(p.parent)
    return ""


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

    The status dot has three states: grey when empty, green when the path
    exists, red when text was typed but nothing is there — so a typo no longer
    looks identical to a valid path until a subprocess fails.

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
        a "Recent" popup button is added, picking a Browse target records it,
        and the picker opens at the most recent folder for this field.
    default : str, optional
        Initial value for the text input.
    hint : str, optional
        Placeholder shown when the input is empty.
    browse : callable, optional
        Callback that opens a native file dialog and returns the chosen path
        (or empty string). If it accepts an ``initialdir`` keyword it is given
        a starting folder; zero-argument callables keep working unchanged.
    required : bool, optional
        Marks the field visually as required; does not enforce anything.
    tooltip_text : str, optional
        Hover text for the label.
    width : int, optional
        Width of the text input.
    """
    marker = " *" if required else ""
    dpg.add_text(f"{label}{marker}:", indent=4, color=LABEL_REQUIRED if required else LABEL_NORMAL)
    if tooltip_text:
        tooltip(tooltip_text)

    status_tag = f"{tag}__status"
    with dpg.group(horizontal=True):
        dpg.add_text("●", tag=status_tag, color=_status_color(default))
        tooltip("Green = the path exists  |  Red = not found  |  Grey = empty", wrap=260)

        dpg.add_input_text(
            tag=tag,
            default_value=default,
            hint=hint,
            width=width,
            callback=lambda s, a, u: _refresh_indicator(status_tag, a),
        )

        if browse is not None:

            def _do_browse(_s=None, _a=None, _u=None):
                current = dpg.get_value(tag) if _exists(tag) else ""
                start = _start_dir(current, field_key)
                if start and _accepts_initialdir(browse):
                    path = browse(initialdir=start)
                else:
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
                _drop(popup_tag)
                pop_w, pop_h, pop_x, pop_y = fit_dialog(640, 340)
                with dpg.window(
                    label="Recent Files",
                    tag=popup_tag,
                    modal=True,
                    width=pop_w,
                    height=pop_h,
                    pos=[pop_x, pop_y],
                ):
                    with dialog_body(f"{popup_tag}__body"):
                        if not items:
                            dpg.add_text(
                                "No recent paths yet. Use the Browse button once "
                                "and this list will remember it.",
                                color=NOTE_COLOR,
                                wrap=600,
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
                                        _drop(popup_tag)

                                    dpg.add_button(label="Use", width=60, callback=_pick)
                                    dpg.add_text(p, color=theme.TEXT, wrap=540)
                    dialog_footer()
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Clear History",
                            width=160,
                            height=30,
                            callback=lambda: (
                                recents.clear(field_key),
                                _drop(popup_tag),
                            ),
                        )
                        dpg.add_button(
                            label="Close",
                            width=140,
                            height=30,
                            callback=lambda: _drop(popup_tag),
                        )

            dpg.add_button(label="Recent", width=70, callback=_open_recent)
            tooltip("Show recently used paths for this field.", wrap=200)


# ══════════════════════════════════════════════════════════════════════════
# labeled inputs
# ══════════════════════════════════════════════════════════════════════════
def labeled_input(
    label: str,
    tag: str,
    *,
    default=0,
    kind: str = "int",
    width: int = 120,
    help_text: str = "",
    step: int = 1,
    items: Optional[Sequence[str]] = None,
    callback: Optional[Callable] = None,
    hint: str = "",
) -> None:
    """A label + input pair with inline help, laid out the same way everywhere.

    ``kind`` is one of ``int`` | ``float`` | ``text`` | ``combo`` |
    ``checkbox``. ``help_text`` is rendered under the control in
    ``TEXT_DIM`` — the place to put the sentinel meaning of a value
    (``0 = all frames``) and the default.
    """
    with dpg.group(horizontal=True):
        dpg.add_text(f"{label}:", color=LABEL_NORMAL)
        gap(0)
        if kind == "float":
            dpg.add_input_float(
                tag=tag, default_value=float(default), width=width, step=0.0, callback=callback
            )
        elif kind == "text":
            dpg.add_input_text(
                tag=tag, default_value=str(default), width=width, hint=hint, callback=callback
            )
        elif kind == "combo":
            dpg.add_combo(
                list(items or []),
                tag=tag,
                default_value=str(default),
                width=width,
                callback=callback,
            )
        elif kind == "checkbox":
            dpg.add_checkbox(tag=tag, default_value=bool(default), callback=callback)
        else:
            dpg.add_input_int(
                tag=tag, default_value=int(default), width=width, step=step, callback=callback
            )
    if help_text:
        dpg.add_text(help_text, color=theme.TEXT_DIM, wrap=max(width + 180, 260), indent=4)


def labeled_slider(
    label: str,
    tag: str,
    *,
    default: float = 0.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
    kind: str = "float",
    width: int = 220,
    fmt: Optional[str] = None,
    help_text: str = "",
    callback: Optional[Callable] = None,
) -> None:
    """A label + slider pair with inline help and the range shown in the format.

    ``kind`` is ``float`` or ``int``. ``help_text`` should state the default
    and, when the number is not what it looks like, say so — e.g. "The
    detector's confidence is not a calibrated probability."
    """
    with dpg.group(horizontal=True):
        dpg.add_text(f"{label}:", color=LABEL_NORMAL)
        gap(0)
        if kind == "int":
            dpg.add_slider_int(
                tag=tag,
                default_value=int(default),
                min_value=int(min_value),
                max_value=int(max_value),
                width=width,
                format=fmt or "%d",
                callback=callback,
                clamped=True,
            )
        else:
            dpg.add_slider_float(
                tag=tag,
                default_value=float(default),
                min_value=float(min_value),
                max_value=float(max_value),
                width=width,
                format=fmt or "%.2f",
                callback=callback,
                clamped=True,
            )
    if help_text:
        dpg.add_text(help_text, color=theme.TEXT_DIM, wrap=max(width + 180, 260), indent=4)


# ══════════════════════════════════════════════════════════════════════════
# buttons
# ══════════════════════════════════════════════════════════════════════════
def _primary_theme_tag() -> int:
    """The accent button theme. Name kept for backwards compatibility."""
    return theme.accent_theme()


#: Advance of one character, in pixels, used to size a button before a frame
#: has been rendered (``dpg.get_text_size`` returns ``None`` until then, so it
#: cannot be asked while the window is being built).
#:
#: Measured both ways: dearpygui's built-in fixed-pitch font is exactly 7 px
#: per character, and once :func:`theme.setup` has loaded the suite's own font
#: it is about 5.6. 7 is therefore an upper bound for both, which is the side
#: to be wrong on - a button a few pixels wider than it needs to be is
#: invisible, a button a few pixels too narrow eats the end of its label.
_CHAR_W = 7
#: ``mvStyleVar_FramePadding`` x, as :func:`theme.setup` sets it. ImGui reserves
#: it on **both** sides of a button label and clips whatever does not fit.
_FRAME_PAD_X = 10


def fit_width(label: str, requested: int) -> int:
    """``requested``, widened to whatever ``label`` actually needs.

    A button narrower than its own label does not shrink the text, wrap it or
    ellipsise it: ImGui clips it at the frame edge and the tail simply is not
    drawn. That failure is silent and it hit the labels that matter most -
    ``Undo (Ctrl+Z)`` needs 111 px and was given 90, so every user saw
    ``Undo (Ct`` and the keyboard shortcut the button existed to teach was the
    part cut off. ``First`` was clipped to ``Firs``.

    ``width <= 0`` is dearpygui's "size yourself" and is passed through
    untouched, so this only ever rescues an explicit, too-small number.
    """
    if requested <= 0:
        return requested
    return max(int(requested), len(label) * _CHAR_W + 2 * _FRAME_PAD_X + 2)


def primary_button(
    label: str, callback, *, height: int = 36, width: int = -1, tag: Optional[str] = None
) -> None:
    """Filled, prominent button for the panel's primary action.

    Text is ``ON_ACCENT`` (near black), never white — white on the accent
    measures 1.91:1 and is prohibited by the design tokens.
    """
    kwargs = _tagged(
        tag,
        {
            "label": label,
            "height": height,
            "width": fit_width(label, width),
            "callback": callback,
        },
    )
    _drop(tag)
    dpg.add_button(**kwargs)
    theme.bind(dpg.last_item(), theme.accent_theme())


def secondary_button(
    label: str, callback, *, height: int = 32, width: int = -1, tag: Optional[str] = None
) -> None:
    """Bordered ghost button — visibly subordinate to :func:`primary_button`."""
    kwargs = _tagged(
        tag,
        {
            "label": label,
            "height": height,
            "width": fit_width(label, width),
            "callback": callback,
        },
    )
    _drop(tag)
    dpg.add_button(**kwargs)
    theme.bind(dpg.last_item(), theme.ghost_theme())


def danger_button(
    label: str, callback, *, height: int = 32, width: int = -1, tag: Optional[str] = None
) -> None:
    """Destructive action. The label must name the verb and the exact scope."""
    kwargs = _tagged(
        tag,
        {
            "label": label,
            "height": height,
            "width": fit_width(label, width),
            "callback": callback,
        },
    )
    _drop(tag)
    dpg.add_button(**kwargs)
    theme.bind(dpg.last_item(), theme.danger_theme())


def toolbar_button(
    label: str,
    callback,
    *,
    tag: str,
    shortcut: str = "",
    help_text: str = "",
    active: bool = False,
    width: int = 0,
    height: int = 28,
    icon: str = "",
    enabled: bool = True,
) -> None:
    """A toolbar tool/command button with a visible active state.

    The active button is filled with the accent colour; the inactive ones are
    bordered ghosts. That difference is the primary channel by which the user
    knows which tool is armed, so it must never be softened.

    ``shortcut`` is appended to the label in parentheses (``Save (Ctrl+S)``)
    and to the tooltip, so the key and its documentation cannot drift.
    """
    text = f"{icon} {label}".strip() if icon else label
    if shortcut:
        text = f"{text} ({shortcut})"
    _drop(tag)
    dpg.add_button(
        label=text,
        tag=tag,
        callback=callback,
        width=fit_width(text, width),
        height=height,
        enabled=enabled,
    )
    theme.bind(tag, theme.tool_theme(active))
    if help_text:
        tip = f"{help_text} ({shortcut})" if shortcut else help_text
        attach_tooltip(tag, tip, wrap=300)


def set_toolbar_active(tag: str, active: bool) -> None:
    """Flip a toolbar button between the filled and ghost states."""
    if not _exists(tag):
        return
    theme.bind(tag, theme.tool_theme(active))


def segmented_control(
    tag: str,
    options: Sequence,
    callback: Optional[Callable],
    *,
    default=None,
    width: int = 0,
    height: int = 26,
) -> None:
    """A row of mutually exclusive buttons — a radio group that reads as a switch.

    ``options`` is a sequence of ``value`` strings or ``(value, label)`` pairs.
    Exactly one button is accent-filled at a time. ``callback`` is invoked as
    ``callback(value)``.

    The selected value is readable later with :func:`segmented_value`.
    """
    pairs: List[Tuple[str, str]] = []
    for opt in options:
        if isinstance(opt, (tuple, list)) and len(opt) >= 2:
            pairs.append((str(opt[0]), str(opt[1])))
        else:
            pairs.append((str(opt), str(opt)))
    if not pairs:
        return
    current = str(default) if default is not None else pairs[0][0]

    _drop(tag)
    btn_tags = {value: f"{tag}__opt_{i}" for i, (value, _) in enumerate(pairs)}

    def _select(value: str, notify: bool = True) -> None:
        _SEGMENT_VALUES[tag] = value
        for val, btag in btn_tags.items():
            set_toolbar_active(btag, val == value)
        if notify and callback is not None:
            callback(value)

    with dpg.group(horizontal=True, tag=tag):
        for value, label in pairs:
            dpg.add_button(
                label=label,
                tag=btn_tags[value],
                width=width,
                height=height,
                callback=(lambda s, a, u, v=value: _select(v)),
            )
    _select(current, notify=False)


_SEGMENT_VALUES: Dict[str, str] = {}


def segmented_value(tag: str, default: str = "") -> str:
    """Current value of a :func:`segmented_control`."""
    return _SEGMENT_VALUES.get(tag, default)


def set_segmented_value(tag: str, value: str) -> None:
    """Programmatically move a :func:`segmented_control` without firing it."""
    if tag not in _SEGMENT_VALUES:
        return
    _SEGMENT_VALUES[tag] = str(value)
    idx = 0
    while _exists(f"{tag}__opt_{idx}"):
        idx += 1
    # Re-derive the button tags from the stored order.
    for i in range(idx):
        btag = f"{tag}__opt_{i}"
        label = dpg.get_item_label(btag) if _exists(btag) else ""
        set_toolbar_active(btag, label == str(value))


# ══════════════════════════════════════════════════════════════════════════
# the locked pattern
# ══════════════════════════════════════════════════════════════════════════
def locked(tag: str, reason: str) -> None:
    """Gate a control: visible, dimmed, disabled, and tooltipped with ``reason``.

    A gated control is **never hidden** — the user must be able to learn that
    the feature exists and why it is off. ``reason`` must state three things:

    1. the cause ("PyTorch is not installed in this Python environment"),
    2. the fix ("Install it with: pip install torch torchvision"),
    3. the escape hatch ("...or run the detector elsewhere and import it").

    The reason is also recorded in :data:`LOCK_REASONS` so a panel can print
    the same sentence inline instead of relying on the hover.

    No-op when ``tag`` does not exist.
    """
    LOCK_REASONS[str(tag)] = reason
    if not _exists(tag):
        return
    _configure(tag, enabled=False)
    theme.bind(tag, theme.disabled_theme())
    attach_tooltip(tag, reason, wrap=420, tooltip_tag=f"{tag}__lock_tip")


def unlock(tag: str, tooltip_text: str = "") -> None:
    """Re-enable a control locked by :func:`locked` and restore its tooltip."""
    LOCK_REASONS.pop(str(tag), None)
    if not _exists(tag):
        return
    _configure(tag, enabled=True)
    _drop(f"{tag}__lock_tip")
    theme.bind(tag, theme.ghost_theme())
    if tooltip_text:
        attach_tooltip(tag, tooltip_text, wrap=320, tooltip_tag=f"{tag}__lock_tip")


def is_locked(tag: str) -> bool:
    """True when :func:`locked` currently gates ``tag``."""
    return str(tag) in LOCK_REASONS


def lock_reason(tag: str) -> str:
    """The explanation registered for a locked control (``""`` when unlocked)."""
    return LOCK_REASONS.get(str(tag), "")


def set_locked(tag: str, reason: str, *, unlocked_tooltip: str = "") -> None:
    """Convenience: lock when ``reason`` is truthy, otherwise unlock."""
    if reason:
        locked(tag, reason)
    else:
        unlock(tag, unlocked_tooltip)


# ══════════════════════════════════════════════════════════════════════════
# status pills, badges and status lines
# ══════════════════════════════════════════════════════════════════════════
#: Pill kinds and their English default wording, for reference by panels.
PILL_KINDS: Tuple[str, ...] = ("idle", "running", "ok", "warn", "error", "pending")


def status_pill(tag: str, text: str, kind: str = "idle") -> None:
    """Create or update a small filled status pill.

    ``kind`` is ``idle`` | ``running`` | ``ok`` | ``warn`` | ``error`` |
    ``pending``. Calling it again with the same tag updates the existing pill
    in place, so it is safe from a render loop.

    Terminal wording is a fixed triple on the subject noun — "X finished" /
    "X failed" / "X stopped". A stop is never an error.
    """
    if kind not in PILL_KINDS:
        kind = "idle"
    if _exists(tag):
        _configure(tag, label=text)
        theme.bind(tag, theme.pill_theme(kind))
        return
    try:
        dpg.add_button(label=text, tag=tag, enabled=True, height=22)
    except SystemError:
        return
    theme.bind(tag, theme.pill_theme(kind))


def badge(text: str, color: Optional[Sequence[int]] = None, *, tag: Optional[str] = None) -> None:
    """A non-interactive colour chip — a class name, a track id, a source.

    ``color`` defaults to the accent. Text colour is chosen by measured
    contrast, so a pale palette entry gets dark text automatically.
    """
    _drop(tag)
    kwargs = _tagged(tag, {"label": text, "height": 20, "enabled": True})
    dpg.add_button(**kwargs)
    theme.bind(dpg.last_item(), theme.chip_theme(color or theme.ACCENT))


def track_badge(track_id: int, *, tag: Optional[str] = None) -> None:
    """A badge coloured with the shared track palette, labelled ``T{id}``."""
    label = f"T{track_id}" if track_id is not None and track_id >= 0 else "T?"
    badge(label, theme.track_color(track_id)[:3], tag=tag)


_STATUS_COLORS = {
    "idle": theme.TEXT_DIM,
    "running": theme.ACCENT,
    "ok": theme.ACCEPTED,
    "warn": theme.PENDING,
    "pending": theme.PENDING,
    "error": theme.DANGER,
}


def status_text(tag: str, message: str, kind: str = "idle") -> None:
    """Write a one-line status into an existing text item, colour-coded by kind.

    No-op when ``tag`` does not exist, so a panel can report progress before
    its status line has been built.
    """
    if not _exists(tag):
        return
    try:
        dpg.set_value(tag, message)
    except SystemError:
        return
    _configure(tag, color=_STATUS_COLORS.get(kind, theme.TEXT_DIM))


# ─── toasts ───────────────────────────────────────────────────────────────
_TOAST_TAG = "yoake_toast_window"
_TOAST_EXPIRY: List[float] = [0.0]


def toast(message: str, kind: str = "ok", *, duration: float = 4.0) -> None:
    """Show a transient message in the top-right corner of the viewport.

    Deliberately does **not** use ``set_frame_callback`` — the annotator owns
    a self-re-arming frame callback for its subprocess pump and a second
    scheduler would fight it. Instead the host calls :func:`tick_toasts` from
    its render loop; a toast also carries a Close button, so it can never get
    stuck even if nothing ticks.
    """
    color = _STATUS_COLORS.get(kind, theme.TEXT_DIM)
    _drop(_TOAST_TAG)
    vw, _vh = viewport_size()
    width = 380
    try:
        with dpg.window(
            tag=_TOAST_TAG,
            no_title_bar=True,
            no_resize=True,
            no_move=False,
            autosize=True,
            pos=[max(10, vw - width - 24), 16],
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("●", color=color)
                dpg.add_text(message, color=theme.TEXT, wrap=width - 60)
            dpg.add_button(label="Close", width=70, height=22, callback=lambda: _drop(_TOAST_TAG))
            theme.bind(dpg.last_item(), theme.ghost_theme())
    except SystemError:
        return
    _TOAST_EXPIRY[0] = time.monotonic() + max(0.5, float(duration))


def tick_toasts() -> None:
    """Hide an expired toast. Call once per frame from the host render loop."""
    if _TOAST_EXPIRY[0] and time.monotonic() >= _TOAST_EXPIRY[0]:
        _TOAST_EXPIRY[0] = 0.0
        _drop(_TOAST_TAG)


# ══════════════════════════════════════════════════════════════════════════
# structure: collapsible sections, keyboard tables, modals
# ══════════════════════════════════════════════════════════════════════════
@contextmanager
def advanced_section(
    label: str = "Advanced",
    *,
    default_open: bool = False,
    tag: Optional[str] = None,
    description: str = "",
):
    """Collapsible container for settings most users never touch.

    Usage::

        with widgets.advanced_section("Advanced"):
            widgets.labeled_input("Window size", "tr_window", default=16)

    Hiding a control behind this is the alternative to hiding it entirely —
    it stays discoverable and its defaults stay visible in the log.
    """
    _drop(tag)
    kwargs = _tagged(tag, {"label": label, "default_open": default_open})
    with dpg.collapsing_header(**kwargs):
        if description:
            dpg.add_text(description, color=theme.TEXT_DIM, wrap=560)
            gap(1)
        yield


def kbd_table(
    rows: Sequence[Tuple[str, str]], *, tag: Optional[str] = None, key_width: int = 150
) -> None:
    """A two-column key legend: the key in mono on a raised ground, then what it does.

    ``rows`` is ``[(key_label, description), ...]`` — feed it straight from
    ``shortcuts.help_rows(scope)`` so the legend can never drift from the
    handlers that are actually registered.
    """
    if not rows:
        dpg.add_text("No shortcuts are defined for this screen.", color=theme.TEXT_DIM)
        return
    _drop(tag)
    kwargs = _tagged(
        tag,
        {
            "header_row": False,
            "borders_innerH": False,
            "borders_outerH": False,
            "borders_innerV": False,
            "borders_outerV": False,
            "policy": dpg.mvTable_SizingFixedFit,
            "width": -1,
        },
    )
    with dpg.table(**kwargs):
        dpg.add_table_column(init_width_or_weight=key_width, width_fixed=True)
        dpg.add_table_column(width_stretch=True)
        for key_label, description in rows:
            with dpg.table_row():
                dpg.add_button(label=str(key_label), width=key_width - 8, height=22, enabled=True)
                btn = dpg.last_item()
                theme.bind(btn, theme.chip_theme(theme.SURFACE_RAISE))
                theme.bind_mono(btn)
                dpg.add_text(str(description), color=theme.TEXT_DIM, wrap=460)


def shortcut_table(
    groups: Sequence[Tuple[str, Sequence[Tuple[str, str]]]], *, note: str = "", key_width: int = 150
) -> None:
    """A grouped keyboard reference: one :func:`kbd_table` per section heading.

    ``groups`` is ``[(group_name, rows), ...]`` — feed it from
    ``shortcuts.help_groups(scope)``. ``note`` is rendered above everything,
    which is where the "Changed in this version" block belongs so a reviewer
    sees the rebinds in one place.
    """
    if note:
        dpg.add_text(note, color=theme.PENDING, wrap=620)
        gap(2)
    for name, rows in groups:
        section_header(name)
        kbd_table(rows, key_width=key_width)
        gap(2)


def confirm_modal(
    title: str,
    message: str,
    *,
    confirm_label: str = "Continue",
    on_confirm: Optional[Callable] = None,
    cancel_label: str = "Cancel",
    on_cancel: Optional[Callable] = None,
    extra: Optional[Sequence[Tuple[str, Callable]]] = None,
    danger: bool = True,
    preserved: str = "",
    tag: str = "yoake_confirm_modal",
    width: int = 520,
) -> str:
    """A confirmation modal with named outcomes — never a bare OK/Cancel.

    ``message`` should read as *what will happen → what else is affected →
    reversibility → Continue?* and ``preserved`` states what survives ("Boxes
    you drew by hand are kept."). ``extra`` adds named third outcomes, which
    is how an unsaved-work prompt offers "Save and close" / "Discard and
    close" / "Cancel".

    Cancel is focused when the modal opens, so Enter can never destroy work.
    Returns the modal's tag.
    """
    _drop(tag)
    vw, vh = viewport_size()
    with dpg.window(
        label=title,
        tag=tag,
        modal=True,
        no_resize=True,
        width=width,
        autosize=True,
        pos=[max(12, (vw - width) // 2), max(12, (vh - 260) // 2)],
    ):
        dpg.add_text(message, color=theme.TEXT, wrap=width - 40)
        if preserved:
            gap(1)
            dpg.add_text(preserved, color=theme.TEXT_DIM, wrap=width - 40)
        gap(2)
        dpg.add_separator()
        gap(1)

        def _close_then(fn: Optional[Callable]):
            def _cb(_s=None, _a=None, _u=None):
                _drop(tag)
                if fn is not None:
                    fn()

            return _cb

        with dpg.group(horizontal=True):
            cancel_tag = f"{tag}__cancel"
            dpg.add_button(
                label=cancel_label,
                tag=cancel_tag,
                width=140,
                height=30,
                callback=_close_then(on_cancel),
            )
            theme.bind(cancel_tag, theme.ghost_theme())
            for label, fn in extra or ():
                dpg.add_button(label=label, width=160, height=30, callback=_close_then(fn))
                theme.bind(dpg.last_item(), theme.ghost_theme())
            dpg.add_button(
                label=confirm_label, width=180, height=30, callback=_close_then(on_confirm)
            )
            theme.bind(dpg.last_item(), theme.danger_theme() if danger else theme.accent_theme())
    try:
        dpg.focus_item(f"{tag}__cancel")
    except SystemError:
        pass
    return tag


def close_modal(tag: str = "yoake_confirm_modal") -> None:
    """Dismiss a modal created by :func:`confirm_modal` (no-op if gone)."""
    _drop(tag)


# ══════════════════════════════════════════════════════════════════════════
# dialog geometry: a window the user can always reach every button of
# ══════════════════════════════════════════════════════════════════════════
#: Pixels kept clear between a dialog and the edge of the viewport.
DIALOG_MARGIN = 28

#: Height a pinned footer row needs: one 34 px button plus the separator and
#: the padding above and below it. A dialog reserves this from its body so the
#: buttons stay on screen however long the body grows.
FOOTER_H = 58

#: ``tag -> (preferred_width, preferred_height)`` for every dialog that asked
#: to be kept inside the viewport. Re-applied by :func:`refit_dialogs`.
_DIALOGS: Dict[str, Tuple[int, int]] = {}


def viewport_size() -> Tuple[int, int]:
    """Client size of the viewport, with a usable answer before one exists.

    ``get_viewport_client_*`` raises a **bare** ``Exception`` ("No viewport
    created") before ``create_viewport``, not the ``SystemError`` the rest of
    dearpygui raises, and a dialog sized from the 0 that a headless test sees
    is a dialog with no buttons on it. Both answers become 1280x820.
    """
    try:
        width = int(dpg.get_viewport_client_width())
        height = int(dpg.get_viewport_client_height())
    except Exception:  # noqa: BLE001 - "No viewport created" is a bare Exception
        width = height = 0
    if width <= 0 or height <= 0:
        return 1280, 820
    return width, height


def fit_dialog(
    width: int, height: int, *, margin: int = DIALOG_MARGIN
) -> Tuple[int, int, int, int]:
    """``(width, height, x, y)`` for a dialog that must stay fully on screen.

    The preferred size is a preference, never a promise: a 740 px form in a
    600 px window has to become a 540 px form with a scrollbar, because the
    alternative is a Run button below the bottom edge that no amount of
    dragging brings back.
    """
    vw, vh = viewport_size()
    out_w = max(260, min(int(width), vw - 2 * margin))
    out_h = max(180, min(int(height), vh - 2 * margin))
    return out_w, out_h, max(8, (vw - out_w) // 2), max(8, (vh - out_h) // 2)


def place_dialog(
    tag, width: int = 0, height: int = 0, *, margin: int = DIALOG_MARGIN, center: bool = True
) -> Tuple[int, int]:
    """Size ``tag`` to fit the viewport, and remember it for later resizes.

    Call it from the dialog's ``show()``. The preferred size defaults to the
    one it was registered or created with, so re-showing a window the user
    resized does not fight them for it.
    """
    if not _exists(tag):
        return (0, 0)
    want = _DIALOGS.get(str(tag))
    if width <= 0 or height <= 0:
        if want is None:
            try:
                want = (int(dpg.get_item_width(tag)), int(dpg.get_item_height(tag)))
            except SystemError:
                return (0, 0)
        width, height = want
    _DIALOGS[str(tag)] = (int(width), int(height))
    out_w, out_h, x, y = fit_dialog(width, height, margin=margin)
    kwargs = {"width": out_w, "height": out_h}
    if center:
        kwargs["pos"] = [x, y]
    _configure(tag, **kwargs)
    return (out_w, out_h)


def refit_dialogs(*, margin: int = DIALOG_MARGIN) -> None:
    """Re-clamp every registered dialog after the viewport changed size.

    Only the size is re-applied and the position is only pulled back inside —
    a dialog the user dragged somewhere stays where they put it.
    """
    vw, vh = viewport_size()
    for tag, (want_w, want_h) in list(_DIALOGS.items()):
        if not _exists(tag):
            _DIALOGS.pop(tag, None)
            continue
        out_w, out_h, _cx, _cy = fit_dialog(want_w, want_h, margin=margin)
        try:
            x, y = (int(v) for v in dpg.get_item_pos(tag))
        except (SystemError, TypeError, ValueError):
            x = y = 8
        _configure(
            tag,
            width=out_w,
            height=out_h,
            pos=[max(0, min(x, vw - out_w)), max(0, min(y, vh - out_h))],
        )


def forget_dialog(tag) -> None:
    """Stop tracking ``tag`` (its window was deleted, not merely hidden)."""
    _DIALOGS.pop(str(tag), None)


@contextmanager
def dialog_body(tag: Optional[str] = None, *, footer: int = FOOTER_H):
    """The scrolling part of a dialog, with room reserved for its footer.

    Everything a dialog can grow — a tab, a form, a result — goes in here, and
    the buttons that act on it go after it. That ordering is the whole fix for
    "the window is 620 px tall, the form is 780 px tall, and Done is at 700".
    """
    _drop(tag)
    kwargs = _tagged(tag, {"width": -1, "height": -int(footer), "border": False})
    with dpg.child_window(**kwargs):
        yield


def dialog_footer() -> None:
    """The separator above a dialog's pinned button row."""
    gap(1)
    dpg.add_separator()
    gap(1)


__all__ = [
    # constants
    "STATUS_OK",
    "STATUS_MISS",
    "STATUS_EMPTY",
    "LABEL_REQUIRED",
    "LABEL_NORMAL",
    "NOTE_COLOR",
    "SECTION_COLOR",
    "HINT_COLOR",
    "PRIMARY_BG",
    "PRIMARY_BG_HOVER",
    "LOCK_REASONS",
    "PILL_KINDS",
    # text / help
    "tooltip",
    "attach_tooltip",
    "help_icon",
    "section_header",
    "page_title",
    "empty_state",
    "gap",
    # fields
    "path_field",
    "refresh_all_indicators",
    "labeled_input",
    "labeled_slider",
    # buttons
    "primary_button",
    "secondary_button",
    "danger_button",
    "toolbar_button",
    "set_toolbar_active",
    "segmented_control",
    "segmented_value",
    "set_segmented_value",
    # gating and status
    "locked",
    "unlock",
    "is_locked",
    "lock_reason",
    "set_locked",
    "status_pill",
    "badge",
    "track_badge",
    "status_text",
    "toast",
    "tick_toasts",
    # structure
    "advanced_section",
    "kbd_table",
    "shortcut_table",
    "confirm_modal",
    "close_modal",
    # dialog geometry
    "DIALOG_MARGIN",
    "FOOTER_H",
    "viewport_size",
    "fit_dialog",
    "place_dialog",
    "refit_dialogs",
    "forget_dialog",
    "dialog_body",
    "dialog_footer",
]
