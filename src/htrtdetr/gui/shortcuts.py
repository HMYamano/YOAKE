"""
shortcuts.py — YOAKE GUI 全体のキーボードバインディング単一ソース.

The one place every keyboard binding in the suite is declared. The handler
registration, every tooltip suffix, the sidebar cheat-sheet and the in-app
"Keyboard Shortcuts" window are all generated from :data:`BINDINGS`, so the
documentation can never drift away from the handlers again. (Before this
module the annotator's docstring advertised a ``T`` binding that was never
registered, and the File menu advertised ``Ctrl+O`` that was never bound.)

Purity contract
---------------
**This module never imports dearpygui at module scope.** Keys are stored as the
*suffix* of the dearpygui constant (``"W"``, ``"Back"``, ``"Right"``,
``"Open_Brace"``), so the whole table is plain data that imports and unit-tests
on a machine with no display and no dearpygui installed. :func:`resolve` is the
only function that touches dearpygui, and the module is *passed in*::

    import dearpygui.dearpygui as dpg
    for b in shortcuts.for_scope("annotator"):
        dpg.add_key_press_handler(shortcuts.resolve(dpg, b), callback=...)

Key-constant rules measured on dearpygui 2.3.1
----------------------------------------------
These are facts, not style preferences. Writing the "obvious" name raises
``AttributeError`` at handler-registration time, i.e. the app does not start:

===================  ==========================  ===========================
Obvious name         Actually exists             Stored here as
===================  ==========================  ===========================
``mvKey_Backspace``  ``mvKey_Back``              ``"Back"``
``mvKey_Space``      ``mvKey_Spacebar``          ``"Spacebar"``
``mvKey_LBracket``   ``mvKey_Open_Brace``        ``"Open_Brace"``
``mvKey_RBracket``   ``mvKey_Close_Brace``       ``"Close_Brace"``
``mvKey_Control``    ``mvKey_ModCtrl``           (modifier, see below)
===================  ==========================  ===========================

:data:`KEY_ALTERNATES` keeps the discarded spellings as a fallback chain so
:func:`resolve` still works on a dearpygui that renames them back, and
:func:`canonical_key` accepts the friendly spellings for lookups.

Modifier contract
-----------------
dearpygui 2.x key-press handlers carry **no** modifier state. A modified
binding is therefore registered as a plain key-press handler whose callback
first calls :func:`mods_held`.

Typing guard contract
---------------------
Every binding except those in :data:`ALWAYS_ALLOWED` must be suppressed while
the user is typing into a widget. The consumer implements the guard (it needs
its own dearpygui handle); this module owns the rule::

    def _typing(self) -> bool:
        f = dpg.get_focused_item()
        if not f:
            return False
        try:
            t = dpg.get_item_type(f)
        except Exception:
            return False
        return any(t.endswith(k) for k in shortcuts.EDITING_TYPES) and dpg.is_item_active(f)

Use ``is_item_active``, **never** ``is_item_focused``: the first input widget in
a window auto-focuses, so focus alone is sticky and useless. Without this guard,
typing a track ID into the annotator's ``sel_tid`` field fires ``A``, ``N``,
``R`` and ``Delete`` — the last of which deletes the selected box.

Callback wrapper the consumers are expected to use::

    def _wrap(self, b):
        def cb(sender, app_data):
            if b.action not in shortcuts.ALWAYS_ALLOWED and self._typing():
                return
            if b.mods and not shortcuts.mods_held(dpg, b):
                return
            if not b.mods and dpg.is_key_down(dpg.mvKey_ModCtrl):
                return                      # do not fire bare W while Ctrl is held
            self._ACTIONS[b.action]()
        return cb
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = [
    "Binding",
    "Shortcut",
    "BINDINGS",
    "SCOPES",
    "GROUP_ORDER",
    "CONTEXTS",
    "GROUP_CONTEXT",
    "EDITING_TYPES",
    "ALWAYS_ALLOWED",
    "KEY_DISPLAY",
    "KEY_ALTERNATES",
    "KEY_ALIASES",
    "MOD_ORDER",
    "CHANGED_IN_THIS_VERSION",
    "UNCHANGED_ON_PURPOSE",
    "canonical_key",
    "modifier_names",
    "format_key",
    "format_binding",
    "for_scope",
    "for_group",
    "for_context",
    "for_action",
    "binding_for",
    "by_key",
    "label_for",
    "help_for",
    "with_key",
    "hint_for",
    "help_rows",
    "help_groups",
    "context_groups",
    "changed_rows",
    "actions",
    "collisions",
    "conflicts",
    "resolve",
    "mods_held",
]


# ─── typing guard contract ────────────────────────────────────────────────
#: ``dpg.get_item_type()`` suffixes that mean "the user is typing into this".
EDITING_TYPES: tuple[str, ...] = (
    "mvInputText",
    "mvInputInt",
    "mvInputFloat",
    "mvInputIntMulti",
    "mvInputFloatMulti",
    "mvCombo",
    "mvSliderInt",
    "mvSliderFloat",
)

#: Actions that fire even while an input widget is active. Saving must never be
#: swallowed by a focused text field — that is how work gets lost.
ALWAYS_ALLOWED: tuple[str, ...] = ("file.save",)


# ─── key spelling tables ──────────────────────────────────────────────────
#: Canonical dearpygui suffix -> what the user is shown.
KEY_DISPLAY: dict[str, str] = {
    "Escape": "Esc",
    "Back": "Backspace",
    "Spacebar": "Space",
    "Open_Brace": "[",
    "Close_Brace": "]",
    "Plus": "+",
    "Minus": "-",
    "Prior": "PageUp",
    "Next": "PageDown",
    "Return": "Enter",
}

#: Canonical suffix -> other suffixes :func:`resolve` will try, in order. Keeps
#: the module working across dearpygui versions that rename these constants.
KEY_ALTERNATES: dict[str, tuple[str, ...]] = {
    "Back": ("Backspace",),
    "Spacebar": ("Space",),
    "Open_Brace": ("LBracket", "LeftBracket"),
    "Close_Brace": ("RBracket", "RightBracket"),
    "Escape": ("Esc",),
    "Plus": ("Equal", "Add"),
    "Minus": ("Subtract",),
}

#: Friendly spellings accepted by :func:`canonical_key` and :func:`by_key`.
KEY_ALIASES: dict[str, str] = {
    "esc": "Escape",
    "backspace": "Back",
    "bksp": "Back",
    "space": "Spacebar",
    "spacebar": "Spacebar",
    "[": "Open_Brace",
    "lbracket": "Open_Brace",
    "leftbracket": "Open_Brace",
    "open_brace": "Open_Brace",
    "]": "Close_Brace",
    "rbracket": "Close_Brace",
    "rightbracket": "Close_Brace",
    "close_brace": "Close_Brace",
    "+": "Plus",
    "plus": "Plus",
    "-": "Minus",
    "minus": "Minus",
    "del": "Delete",
    "enter": "Return",
    "return": "Return",
    "pageup": "Prior",
    "pagedown": "Next",
}

#: Modifier order used when a label is assembled. Fixed so ``Ctrl+Shift+S``
#: never renders as ``Shift+Ctrl+S`` somewhere else in the app.
MOD_ORDER: tuple[str, ...] = ("Ctrl", "Shift", "Alt")

_MOD_NAMES_DEFAULT: dict[str, str] = {"Ctrl": "Ctrl", "Shift": "Shift", "Alt": "Alt"}
_MOD_NAMES_MAC: dict[str, str] = {"Ctrl": "Cmd", "Shift": "Shift", "Alt": "Option"}


def canonical_key(name: str) -> str:
    """Normalise a friendly key spelling onto its dearpygui suffix.

    ``canonical_key("Space") == "Spacebar"``, ``canonical_key("[") ==
    "Open_Brace"``. An unrecognised name comes back unchanged, so a caller can
    pass a canonical suffix straight through.
    """
    if not name:
        return ""
    return KEY_ALIASES.get(name.strip().lower(), name.strip())


def modifier_names(platform: Optional[str] = None) -> dict[str, str]:
    """Platform-correct modifier names.

    macOS says *Cmd* and *Option*; everything else says *Ctrl* and *Alt*.
    ``platform`` defaults to :data:`sys.platform` and exists so tests can pin a
    value without monkeypatching the interpreter.
    """
    plat = sys.platform if platform is None else platform
    return dict(_MOD_NAMES_MAC if plat == "darwin" else _MOD_NAMES_DEFAULT)


def format_key(key: str, platform: Optional[str] = None) -> str:
    """Render one key suffix the way a user should read it.

    ``format_key("Open_Brace") == "["``, ``format_key("Spacebar") == "Space"``.
    """
    k = canonical_key(key)
    if k in KEY_DISPLAY:
        return KEY_DISPLAY[k]
    # Function keys, digits and single letters already read correctly; anything
    # else carrying an underscore reads better with a space.
    return k.replace("_", " ") if "_" in k else k


def format_binding(key: str, mods: Iterable[str] = (),
                   platform: Optional[str] = None) -> str:
    """Assemble a human-readable accelerator: ``("S", ("Ctrl",)) -> "Ctrl+S"``."""
    names = modifier_names(platform)
    held = {m.strip().title() for m in mods}
    parts = [names[m] for m in MOD_ORDER if m in held]
    parts.append(format_key(key, platform))
    return "+".join(parts)


# ─── the binding record ───────────────────────────────────────────────────
@dataclass(frozen=True)
class Binding:
    """One keyboard binding.

    Attributes
    ----------
    action:
        Stable identifier, e.g. ``"tool.draw_box"``. Consumers map these onto
        callables. An action may carry more than one binding: ``nav.next`` is
        both ``D`` and ``Right``.
    key:
        The dearpygui constant **suffix** — never the constant itself, so this
        module stays importable without dearpygui. See :data:`KEY_ALTERNATES`.
    mods:
        Subset of ``("Ctrl", "Shift", "Alt")``, checked at callback time by
        :func:`mods_held` because dpg 2.x handlers carry no modifier state.
    scope:
        Which window registers it: ``"annotator"`` or ``"hub"``.
    group:
        Section heading in the generated help table — one of
        :data:`GROUP_ORDER`.
    context:
        Coarser bucket for grouping across scopes, one of :data:`CONTEXTS`.
        Defaults to the group's context; set explicitly where a binding lives
        in one section but belongs to another loop (the ``Ctrl+Arrow``
        candidate jumps sit under *Navigation* but belong to *Auto-label
        review*).
    label:
        Human-readable accelerator. Filled in automatically and
        platform-correctly; pass a value only to override it.
    help:
        English description, sentence case, **no** final period, ASCII only so
        it renders before any font is loaded.
    """

    action: str
    key: str
    mods: tuple[str, ...] = ()
    scope: str = "annotator"
    group: str = "Tools"
    context: str = ""
    label: str = ""
    help: str = ""

    def __post_init__(self) -> None:  # noqa: D105 - frozen dataclass fix-ups
        object.__setattr__(self, "key", canonical_key(self.key))
        object.__setattr__(self, "mods", tuple(m.strip().title() for m in self.mods))
        if not self.label:
            object.__setattr__(self, "label", format_binding(self.key, self.mods))
        if not self.context:
            object.__setattr__(self, "context", GROUP_CONTEXT.get(self.group, "Global"))

    # ── aliases, so consumers can read this record either way ──
    @property
    def description(self) -> str:
        """Alias of :attr:`help`."""
        return self.help

    @property
    def category(self) -> str:
        """Alias of :attr:`group`."""
        return self.group

    @property
    def keys(self) -> tuple[str, ...]:
        """The accelerator as display tokens, e.g. ``("Ctrl", "S")``."""
        names = modifier_names()
        return tuple([names[m] for m in MOD_ORDER if m in self.mods]
                     + [format_key(self.key)])

    @property
    def combo(self) -> tuple[str, tuple[str, ...]]:
        """The identity used for conflict detection: ``(key, sorted mods)``."""
        return (self.key, tuple(sorted(self.mods)))

    def row(self) -> tuple[str, str]:
        """``(label, help)`` — one row of the help table."""
        return (self.label, self.help)

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return f"{self.label} -> {self.action}"


#: The design docs say "shortcut", the registration code says "binding".
#: Same object.
Shortcut = Binding


#: Every scope, in the order a combined help window should show them.
#:
#: ``review`` is the Review Mode overlay (``gui/review.py``). It is a separate
#: scope, not extra annotator bindings, because it deliberately **reuses** keys
#: the annotator spends on something else: ``E`` is *Draw Rotated Box* on the
#: canvas and *Edit this item* in review, and both are right. Two scopes make
#: that expressible; one scope would make it a conflict.
SCOPES: tuple[str, ...] = ("annotator", "hub", "review")

#: Section headings, in render order.
GROUP_ORDER: tuple[str, ...] = (
    "Tools",
    "View",
    "Navigation",
    "Review",
    "Edit",
    "File",
    "Hub",
    "Review mode",
)

#: Coarse buckets, in render order.
CONTEXTS: tuple[str, ...] = (
    "Annotator canvas",
    "Playback",
    "Auto-label review",
    "Global",
)

#: Default group -> context mapping. An individual binding may override it.
GROUP_CONTEXT: dict[str, str] = {
    "Tools": "Annotator canvas",
    "View": "Annotator canvas",
    "Navigation": "Playback",
    "Review": "Auto-label review",
    "Edit": "Annotator canvas",
    "File": "Global",
    "Hub": "Global",
    "Review mode": "Auto-label review",
}


def _b(action: str, key: str, mods: tuple[str, ...], group: str, help_: str,
       scope: str = "annotator", context: str = "") -> Binding:
    """Terse constructor so the keymap below reads as a table."""
    return Binding(action=action, key=key, mods=mods, scope=scope, group=group,
                   context=context, help=help_)


# ─── the keymap ───────────────────────────────────────────────────────────
# LabelImg familiarity is an explicit design goal: W creates a box, Esc
# cancels, Delete deletes, Ctrl+S saves. Those four must not be re-churned.
BINDINGS: tuple[Binding, ...] = (
    # ── Annotator / Tools ──
    _b("tool.select", "V", (), "Tools",
       "Select tool: click a box to edit it, drag inside to move, drag a handle to resize"),
    _b("tool.draw_box", "W", (), "Tools",
       "Draw Box tool: drag to create a new axis-aligned box, even on top of an existing one"),
    _b("tool.draw_rotated", "E", (), "Tools",
       "Draw Rotated Box tool: drag to create a rotated box, then turn the yellow handle"),
    _b("tool.draw_rotated", "R", (), "Tools",
       "Draw Rotated Box tool: same as E, kept because R already meant rotated"),
    _b("tool.click_box", "C", (), "Tools",
       "Click to Box tool: click an animal and a rotated box is fitted to it. "
       "One box per press, then back to Select so the handles are ready"),
    _b("tool.pan", "H", (), "Tools",
       "Pan tool: drag to move the image"),
    _b("tool.delete", "X", (), "Tools",
       "Delete Box tool: click a box to remove it"),
    _b("tool.cancel", "Escape", (), "Tools",
       "Cancel the box being drawn, return to Select, and clear the selection"),

    # ── Annotator / View ──
    _b("view.fit", "F", (), "View",
       "Fit the frame to the window"),
    _b("view.zoom_in", "Plus", (), "View",
       "Zoom in"),
    _b("view.zoom_out", "Minus", (), "View",
       "Zoom out"),

    # ── Annotator / Navigation ──
    _b("nav.prev", "A", (), "Navigation",
       "Previous frame"),
    _b("nav.prev", "Left", (), "Navigation",
       "Previous frame"),
    _b("nav.next", "D", (), "Navigation",
       "Next frame"),
    _b("nav.next", "Right", (), "Navigation",
       "Next frame"),
    _b("nav.first", "Home", (), "Navigation",
       "First frame"),
    _b("nav.last", "End", (), "Navigation",
       "Last frame"),
    _b("review.next_pending_frame", "Right", ("Ctrl",), "Navigation",
       "Jump to the next frame that still has candidates",
       context="Auto-label review"),
    _b("review.prev_pending_frame", "Left", ("Ctrl",), "Navigation",
       "Jump to the previous frame that still has candidates",
       context="Auto-label review"),

    # ── Annotator / Review ──
    _b("review.cycle", "Q", (), "Review",
       "Select the next box in this frame, candidates first"),
    _b("review.accept_all", "Y", (), "Review",
       "Accept every candidate in this frame"),
    _b("review.discard_all", "N", (), "Review",
       "Discard every candidate in this frame"),
    _b("auto.track_forward", "T", (), "Review",
       "Track the selected box forward from this frame"),
    _b("auto.annotate", "R", ("Ctrl",), "Review",
       "Open Pick frames to label"),
    _b("auto.toggle_autotrack", "T", ("Ctrl",), "Review",
       "Turn auto-track mode on or off"),

    # ── Annotator / Edit ──
    _b("edit.delete", "Delete", (), "Edit",
       "Delete the selected box"),
    _b("edit.delete", "Back", (), "Edit",
       "Delete the selected box"),
    _b("edit.propagate", "Spacebar", (), "Edit",
       "Copy boxes from the previous frame"),
    _b("edit.undo", "Z", ("Ctrl",), "Edit",
       "Undo the last change (50 steps)"),
    _b("edit.redo", "Y", ("Ctrl",), "Edit",
       "Redo the change you just undid"),
    _b("edit.redo", "Z", ("Ctrl", "Shift"), "Edit",
       "Redo, for the hand that already knows Ctrl+Shift+Z"),
    _b("edit.new_track", "N", ("Ctrl",), "Edit",
       "Start a new track ID"),
    _b("obb.rotate_ccw", "Open_Brace", (), "Edit",
       "Rotate the selected box 5 degrees counter-clockwise"),
    _b("obb.rotate_cw", "Close_Brace", (), "Edit",
       "Rotate the selected box 5 degrees clockwise"),
    *tuple(
        _b(f"action.set_{i}", str(i), (), "Edit",
           f"Set the selected box's action to {i}")
        for i in range(10)
    ),

    # ── Annotator / File ──
    _b("file.save", "S", ("Ctrl",), "File",
       "Save - works even while typing in a field"),
    _b("file.save_as", "S", ("Ctrl", "Shift"), "File",
       "Save to a different file"),
    _b("file.open_video", "O", ("Ctrl",), "File",
       "Open a video"),
    _b("help.shortcuts", "F1", (), "File",
       "Show the shortcut list"),

    # ── Hub ──
    _b("page.workflow", "1", ("Ctrl",), "Hub",
       "Go to 0. Workflow", scope="hub"),
    _b("page.label", "2", ("Ctrl",), "Hub",
       "Go to 1. Label", scope="hub"),
    _b("page.train", "3", ("Ctrl",), "Hub",
       "Go to 2. Train", scope="hub"),
    _b("page.analyze", "4", ("Ctrl",), "Hub",
       "Go to 3. Analyze", scope="hub"),
    _b("help.shortcuts", "F1", (), "Hub",
       "Show the shortcut list", scope="hub"),

    # ── Review Mode ──
    # One decision per key, right hand on the mouse, left hand here (P14). The
    # letters are the same ones the annotator uses for accept and discard, so
    # the muscle memory carries over; E and the arrows are re-spent, which is
    # what the separate scope is for.
    _b("review.accept", "Y", (), "Review mode",
       "Keep this - the model was right here", scope="review"),
    _b("review.reject", "N", (), "Review mode",
       "Discard this - the model was wrong here", scope="review"),
    _b("review.edit", "E", (), "Review mode",
       "Leave review and fix this one by hand", scope="review"),
    _b("review.skip", "S", (), "Review mode",
       "Decide later, and do not count it as remaining", scope="review"),
    _b("review.next", "Right", (), "Review mode",
       "Next item", scope="review"),
    _b("review.next", "D", (), "Review mode",
       "Next item, for the left hand", scope="review"),
    _b("review.prev", "Left", (), "Review mode",
       "Previous item", scope="review"),
    _b("review.prev", "A", (), "Review mode",
       "Previous item, for the left hand", scope="review"),
    _b("review.exit", "Escape", (), "Review mode",
       "Leave review and go back to the annotator", scope="review"),
    _b("playback.toggle", "Spacebar", (), "Review mode",
       "Pause or resume the loop", scope="review"),
)


#: Rendered at the top of the Shortcuts window so the rebinds are announced
#: rather than discovered. ``(was, now, why)``.
CHANGED_IN_THIS_VERSION: tuple[tuple[str, str, str], ...] = (
    ("A toggled auto-track",
     "A is Previous frame; auto-track moved to Ctrl+T",
     "LabelImg and YOMU both use A and D for previous and next, "
     "and LabelImg familiarity is an explicit goal"),
    ("N and P moved between frames",
     "N discards candidates; P is free",
     "N completes the Y / N accept-discard pair, and A / D plus the arrows "
     "already cover navigation"),
    ("R toggled a hidden OBB mode",
     "R selects the Draw Rotated Box tool",
     "The hidden mode becomes a visible tool with a visible active state"),
    ("0-4 set actions 0-4",
     "0-9 set actions 0-9",
     "The panel already allowed more than five actions with no key for them"),
)

#: Bindings deliberately left alone, for the same help window.
UNCHANGED_ON_PURPOSE: tuple[str, ...] = (
    "Space still copies boxes from the previous frame",
    "[ and ] still rotate by 5 degrees",
    "W still starts a new box anywhere, even on top of an existing one",
    "Esc still cancels, Delete still deletes, Ctrl+S still saves",
)


# ─── lookup ───────────────────────────────────────────────────────────────
def for_scope(scope: str) -> tuple[Binding, ...]:
    """Every binding registered by one window (``"annotator"`` / ``"hub"``)."""
    return tuple(b for b in BINDINGS if b.scope == scope)


def for_group(group: str, scope: Optional[str] = None) -> tuple[Binding, ...]:
    """Every binding in one help-table section."""
    return tuple(b for b in BINDINGS
                 if b.group == group and (scope is None or b.scope == scope))


def for_context(context: str, scope: Optional[str] = None) -> tuple[Binding, ...]:
    """Every binding in one coarse context (see :data:`CONTEXTS`)."""
    return tuple(b for b in BINDINGS
                 if b.context == context and (scope is None or b.scope == scope))


def for_action(action: str, scope: Optional[str] = None) -> tuple[Binding, ...]:
    """Every binding of one action. ``nav.next`` has two: ``D`` and ``Right``."""
    return tuple(b for b in BINDINGS
                 if b.action == action and (scope is None or b.scope == scope))


def binding_for(action: str, scope: Optional[str] = None) -> Optional[Binding]:
    """The primary (first-declared) binding of an action, or ``None``."""
    found = for_action(action, scope)
    return found[0] if found else None


def by_key(key: str, mods: Iterable[str] = (),
           scope: Optional[str] = None) -> tuple[Binding, ...]:
    """Reverse lookup: which bindings fire for this key combination.

    Accepts friendly spellings (``by_key("Space")``, ``by_key("[")``), an
    accelerator string (``by_key("Ctrl+S")``) and the canonical suffix with
    separate ``mods``. Returns every match, so an over-bound key is visible
    here as well as through :func:`conflicts`.
    """
    raw = key.strip()
    if not mods and "+" in raw and len(raw) > 1:
        parts = [p for p in raw.split("+") if p != ""]
        if not parts:                       # the literal "+" key, spelled "+"
            parts = ["+"]
        raw, mods = parts[-1], tuple(parts[:-1])
    want_key = canonical_key(raw)
    want_mods = tuple(sorted(m.strip().title() for m in mods))
    return tuple(b for b in BINDINGS
                 if b.key == want_key and tuple(sorted(b.mods)) == want_mods
                 and (scope is None or b.scope == scope))


def label_for(action: str, scope: Optional[str] = None) -> str:
    """``"Ctrl+S"`` for ``file.save``; ``""`` when the action is unbound."""
    b = binding_for(action, scope)
    return b.label if b else ""


def help_for(action: str, scope: Optional[str] = None) -> str:
    """The English description of an action; ``""`` when unbound."""
    b = binding_for(action, scope)
    return b.help if b else ""


def with_key(text: str, action: str, scope: Optional[str] = None) -> str:
    """Append the accelerator to a control label: ``"Save"`` -> ``"Save (Ctrl+S)"``.

    An unbound action returns ``text`` unchanged, so a control never grows an
    empty ``()``.
    """
    lbl = label_for(action, scope)
    return f"{text} ({lbl})" if lbl else text


def hint_for(action: str, scope: Optional[str] = None) -> str:
    """One-line tooltip hint listing *every* key bound to an action.

    ``hint_for("nav.next") == "Next frame  (D / Right)"``. Unlike
    :func:`with_key`, which names only the primary accelerator for a button
    label, this is the discoverability string and shows the alternates too.
    Returns ``""`` for an unknown action.
    """
    found = for_action(action, scope)
    if not found:
        return ""
    return f"{found[0].help}  ({' / '.join(b.label for b in found)})"


def actions(scope: Optional[str] = None) -> tuple[str, ...]:
    """Every distinct action id, in declaration order.

    Consumers assert at build time that their ``_ACTIONS`` dict covers this, so
    an unimplemented action is a startup error rather than a dead key.
    """
    seen: list[str] = []
    for b in BINDINGS:
        if scope is not None and b.scope != scope:
            continue
        if b.action not in seen:
            seen.append(b.action)
    return tuple(seen)


# ─── help rendering ───────────────────────────────────────────────────────
def help_rows(scope: str) -> list[tuple[str, str]]:
    """Flat ``(key, description)`` rows for ``widgets.kbd_table``.

    Exactly one row per binding, so ``len(help_rows(s)) == len(for_scope(s))``
    and the documentation cannot go stale.
    """
    return [b.row() for b in for_scope(scope)]


def help_groups(scope: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(section heading, rows), ...]`` in :data:`GROUP_ORDER`, empties dropped."""
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for group in GROUP_ORDER:
        rows = [b.row() for b in for_group(group, scope)]
        if rows:
            out.append((group, rows))
    return out


def context_groups(scope: Optional[str] = None
                   ) -> list[tuple[str, list[tuple[str, str]]]]:
    """Same shape as :func:`help_groups`, bucketed by :data:`CONTEXTS` instead.

    Use this for a combined help window that spans the annotator and the hub.
    """
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for ctx in CONTEXTS:
        rows = [b.row() for b in for_context(ctx, scope)]
        if rows:
            out.append((ctx, rows))
    return out


def changed_rows() -> list[tuple[str, str, str]]:
    """The *Changed in this version* block, as table rows."""
    return [(was, now, why) for was, now, why in CHANGED_IN_THIS_VERSION]


# ─── validation ───────────────────────────────────────────────────────────
def conflicts() -> list[tuple[str, str, tuple[str, ...], list[str]]]:
    """Every over-bound key combination.

    Returns ``[(scope, key, mods, [action, ...]), ...]``. Two bindings collide
    only when scope **and** key **and** modifiers all match, so ``Right`` and
    ``Ctrl+Right`` are distinct and ``F1`` in the hub does not collide with
    ``F1`` in the annotator. Two bindings of the *same* action (``nav.next`` on
    both ``D`` and ``Right``) are not a conflict either, because their keys
    differ.
    """
    seen: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
    for b in BINDINGS:
        seen.setdefault((b.scope,) + b.combo, []).append(b.action)
    return [(scope, key, mods, acts)
            for (scope, key, mods), acts in seen.items() if len(acts) > 1]


def collisions() -> list[str]:
    """Human-readable conflict report. ``[] == the keymap is valid``."""
    return [
        f"{scope}: {format_binding(key, mods)} is bound to " + ", ".join(acts)
        for scope, key, mods, acts in conflicts()
    ]


# ─── the only dearpygui touch points ──────────────────────────────────────
def resolve(dpg_module: Any, binding: Binding) -> int:
    """``getattr(dpg, "mvKey_" + binding.key)``, with the version fallback chain.

    ``dpg_module`` is passed in so this module never imports dearpygui itself.
    Raises :class:`AttributeError` naming every spelling that was tried, which
    is far more useful than the bare ``mvKey_Backspace`` failure it replaces.
    """
    candidates = (binding.key,) + KEY_ALTERNATES.get(binding.key, ())
    for name in candidates:
        const = getattr(dpg_module, "mvKey_" + name, None)
        if const is not None:
            return int(const)
    tried = ", ".join("mvKey_" + n for n in candidates)
    raise AttributeError(
        f"No dearpygui key constant for binding {binding.action!r} (tried: {tried})"
    )


def mods_held(dpg_module: Any, binding: Binding) -> bool:
    """Are this binding's modifiers currently down?

    dearpygui 2.x key-press handlers carry no modifier state, so a modified
    binding is registered as a plain key handler and its callback calls this
    first. ``mvKey_Control`` does not exist in 2.x — the constants are
    ``mvKey_ModCtrl`` / ``mvKey_ModShift`` / ``mvKey_ModAlt``.

    An unmodified binding returns ``True``; the "is Ctrl wrongly held?" test is
    the consumer's, because only it knows whether a bare key should be
    suppressed.
    """
    if not binding.mods:
        return True
    for mod in binding.mods:
        const = getattr(dpg_module, "mvKey_Mod" + mod, None)
        if const is None:
            return False
        if not dpg_module.is_key_down(const):
            return False
    return True
