"""
theme.py — The single design-token module for the whole YOAKE GUI suite.

After this module lands, **no other file in the suite may contain a colour
literal**. The hub, the annotator, the analysis canvas and every panel import
their colours, spacing, radii and fonts from here so that a box drawn in the
annotator and the same box reviewed in the analysis panel read identically.

Two layers:

``tokens``
    Plain module-level data (tuples / ints). Importing this module never
    requires a Dear PyGui context, so the tokens are usable from tests,
    headless code and pure-geometry helpers.

``builders``
    Every function that touches ``dearpygui`` — :func:`load_fonts`,
    :func:`apply_global_theme` and the per-component theme factories. All of
    them are **idempotent and cached at module level**: calling
    ``accent_theme()`` in a render loop does not leak a new theme object per
    frame.

Verified against dearpygui 2.3.1 on this machine:

* ``add_font_range_hint`` / ``add_font_chars`` / ``add_font_range`` are
  deprecated no-ops ("character ranges are now automatic"). Loading the TTF
  and binding it is the whole job — do not write char-range code.
* ``theme_component(..., enabled_state=False)`` is the only way to give a
  disabled control a real look; that is what makes the *locked* pattern in
  :func:`htrtdetr.gui.widgets.locked` expressible.

Self-check::

    python -m htrtdetr.gui.theme --check

prints the contrast report and exits non-zero on any violation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import dearpygui.dearpygui as dpg

_log = logging.getLogger(__name__)

RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

# ══════════════════════════════════════════════════════════════════════════
# TOKENS — pure data, no dpg context required
# ══════════════════════════════════════════════════════════════════════════

# ─── surfaces ─────────────────────────────────────────────────────────────
BG: RGB = (24, 26, 31)              # viewport / window ground
SURFACE: RGB = (38, 41, 47)         # panels, cards, child windows
SURFACE_RAISE: RGB = (48, 52, 60)   # toolbar strip, headers, button fill
FIELD: RGB = (23, 24, 28)           # input backgrounds, log panes
ROW_HOVER: RGB = (60, 66, 78)       # list row hover
ROW_ACTIVE: RGB = (50, 72, 104)     # selected list row / active tab

# ─── lines ────────────────────────────────────────────────────────────────
BORDER_SUBTLE: RGB = (74, 80, 92)   # decorative separators inside a card only
BORDER_STRONG: RGB = (112, 119, 133)  # every interactive control (3.24:1)

# ─── text ─────────────────────────────────────────────────────────────────
TEXT: RGB = (216, 219, 226)         # body               10.53:1 vs SURFACE
TEXT_DIM: RGB = (150, 157, 170)     # secondary / help    5.30:1
TEXT_FAINT: RGB = (120, 127, 140)   # paths, disabled     3.62:1  (>=16 px)
ON_ACCENT: RGB = (16, 20, 28)       # text on accent fill 6.99:1 vs ACCENT

# ─── semantic ─────────────────────────────────────────────────────────────
ACCENT: RGB = (77, 163, 255)        # selection, active tool, links
ACCENT_HOVER: RGB = (104, 180, 255)
ACCENT_ACTIVE: RGB = (54, 140, 232)
ACCEPTED: RGB = (46, 204, 113)      # accepted / confirmed / capability present
PENDING: RGB = (245, 166, 35)       # candidate, unsaved, warning
DANGER: RGB = (255, 82, 82)         # destructive, missing file, error
INTERPOLATED: RGB = (120, 144, 176)  # machine-filled boxes that are not candidates
TRACK_GREY: RGB = (150, 150, 150)   # track_id < 0

# ─── control fills ────────────────────────────────────────────────────────
BTN: RGB = SURFACE_RAISE
BTN_HOVER: RGB = (60, 66, 78)
BTN_ACTIVE: RGB = (70, 78, 92)
MODAL_DIM: RGBA = (10, 11, 14, 180)

# ─── semantic role aliases ────────────────────────────────────────────────
# Some consumers speak in roles rather than token names. These are aliases,
# never independent values — changing a role changes the token it points at.
PANEL: RGB = SURFACE
BORDER: RGB = BORDER_STRONG
MUTED: RGB = TEXT_DIM
SUCCESS: RGB = ACCEPTED
WARNING: RGB = PENDING
INFO: RGB = ACCENT
AI_SUGGESTION: RGB = PENDING        # a machine proposal awaiting review

# The Annotation Copilot vocabulary. Same three colours under the names the
# review loop uses, so a panel written against the copilot's concepts never has
# to translate: a candidate is CANDIDATE, a failed check is ERROR.
BACKGROUND: RGB = BG
CANDIDATE: RGB = PENDING            # a candidate awaiting accept or reject
ERROR: RGB = DANGER
VERIFIED: RGB = ACCEPTED            # a human kept this
PREDICTED: RGB = INTERPOLATED       # machine output nothing objects to
NEEDS_REVIEW: RGB = PENDING         # machine output a detector flagged

ROLES: Dict[str, RGB] = {
    "background": BG,
    "surface": SURFACE,
    "panel": PANEL,
    "raised": SURFACE_RAISE,
    "field": FIELD,
    "border": BORDER,
    "border_subtle": BORDER_SUBTLE,
    "text": TEXT,
    "muted": MUTED,
    "faint": TEXT_FAINT,
    "accent": ACCENT,
    "on_accent": ON_ACCENT,
    "success": SUCCESS,
    "warning": WARNING,
    "danger": DANGER,
    "ai_suggestion": AI_SUGGESTION,
    "interpolated": INTERPOLATED,
    "candidate": CANDIDATE,
    "verified": VERIFIED,
    "predicted": PREDICTED,
    "needs_review": NEEDS_REVIEW,
    "error": ERROR,
}

# ─── spacing scale ────────────────────────────────────────────────────────
# No other vertical spacing value is permitted anywhere in the suite.
SP_XS = 2    # between a label and the control it names
SP_S = 4     # within a control group
SP_M = 8     # between control groups
SP_L = 14    # between sections
SP_XL = 22   # between major regions

SPACING: Tuple[int, int, int, int, int] = (SP_XS, SP_S, SP_M, SP_L, SP_XL)

# ─── radii ────────────────────────────────────────────────────────────────
R_FRAME = 6
R_CHILD = 8
R_WINDOW = 8
R_POPUP = 6
R_TAB = 6
R_GRAB = 5

# ─── type scale ───────────────────────────────────────────────────────────
FONT_H1 = 21    # page title (one per screen)
FONT_H2 = 18    # section heading
FONT_BODY = 16  # everything else (bound globally)
FONT_MONO = 14  # log panes, key legends, aligned report blocks

FONT_UI_TAG = "font_ui"
FONT_H1_TAG = "font_h1"
FONT_H2_TAG = "font_h2"
FONT_MONO_TAG = "font_mono"

_UI_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:/Windows/Fonts/segoeui.ttf",
    r"C:/Windows/Fonts/SegoeUI.ttf",
    r"C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_MONO_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:/Windows/Fonts/consola.ttf",
    r"C:/Windows/Fonts/cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

# ─── track / class palette ────────────────────────────────────────────────
# Moved verbatim from canvas.py / annotator.py. The one palette the annotator
# and the analysis canvas must share, indexed ``id % 20``.
TRACK_PALETTE: Tuple[RGB, ...] = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
    (191, 239, 69), (250, 190, 212), (70, 153, 144), (220, 190, 255),
    (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 216, 177), (0, 0, 117), (169, 169, 169),
)


# ══════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS — pure functions
# ══════════════════════════════════════════════════════════════════════════
def rgba(color: Sequence[int], alpha: int = 255) -> RGBA:
    """Expand a 3-tuple token to RGBA, or re-alpha an existing 4-tuple."""
    c = tuple(int(v) for v in color)
    if len(c) >= 4:
        return (c[0], c[1], c[2], int(alpha) if alpha != 255 else c[3])
    return (c[0], c[1], c[2], int(alpha))


def track_color(track_id: Optional[int], alpha: int = 255) -> RGBA:
    """Stable colour for a track id. ``track_id < 0``/``None`` -> grey.

    Every surface that draws a box must call this, so one subject keeps one
    colour from the annotator through the analysis canvas.
    """
    if track_id is None or track_id < 0:
        return rgba(TRACK_GREY, alpha)
    return rgba(TRACK_PALETTE[int(track_id) % len(TRACK_PALETTE)], alpha)


def class_color(class_id: Optional[int], alpha: int = 255) -> RGBA:
    """Stable colour for an object class, offset so class 0 != track 0."""
    if class_id is None or class_id < 0:
        return rgba(TRACK_GREY, alpha)
    idx = (int(class_id) * 7 + 3) % len(TRACK_PALETTE)
    return rgba(TRACK_PALETTE[idx], alpha)


def source_color(source: str, alpha: int = 255) -> RGBA:
    """Colour for a box's provenance: human / auto / track / interp."""
    return rgba({
        "human": ACCEPTED,
        "auto": PENDING,
        "track": INTERPOLATED,
        "interp": INTERPOLATED,
    }.get(source, TEXT_DIM), alpha)


def status_dot(state: str) -> RGB:
    """Colour for the ``●`` indicator: ok | pending | missing | '' (empty)."""
    return {
        "ok": ACCEPTED,
        "pending": PENDING,
        "missing": DANGER,
        "error": DANGER,
        "warn": PENDING,
    }.get(state, TEXT_FAINT)


def _srgb_channel(v: int) -> float:
    c = v / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(color: Sequence[int]) -> float:
    """WCAG 2.x relative luminance of an sRGB colour."""
    r, g, b = (int(color[0]), int(color[1]), int(color[2]))
    return (0.2126 * _srgb_channel(r)
            + 0.7152 * _srgb_channel(g)
            + 0.0722 * _srgb_channel(b))


def contrast_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    """WCAG contrast ratio between two colours, 1.0 .. 21.0."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = (la, lb) if la >= lb else (lb, la)
    return (hi + 0.05) / (lo + 0.05)


# The ten invariants the palette must satisfy. ``op`` is ">=" or "<".
_INVARIANTS: Tuple[Tuple[str, RGB, str, RGB, str, float], ...] = (
    ("TEXT", TEXT, "SURFACE", SURFACE, ">=", 7.0),
    ("TEXT_DIM", TEXT_DIM, "SURFACE", SURFACE, ">=", 4.5),
    ("TEXT_FAINT", TEXT_FAINT, "SURFACE", SURFACE, ">=", 3.0),
    ("BORDER_STRONG", BORDER_STRONG, "SURFACE", SURFACE, ">=", 3.0),
    ("ACCENT", ACCENT, "SURFACE", SURFACE, ">=", 3.0),
    ("ON_ACCENT", ON_ACCENT, "ACCENT", ACCENT, ">=", 4.5),
    ("ACCEPTED", ACCEPTED, "SURFACE", SURFACE, ">=", 3.0),
    ("PENDING", PENDING, "SURFACE", SURFACE, ">=", 3.0),
    ("DANGER", DANGER, "SURFACE", SURFACE, ">=", 3.0),
    # White on accent MUST fail. This assertion is what stops the single most
    # common regression: putting white text on an accent-filled button.
    ("WHITE", (255, 255, 255), "ACCENT", ACCENT, "<", 3.0),
)


def verify_tokens() -> List[str]:
    """Recompute every contrast invariant. Returns violations; ``[]`` == pass."""
    out: List[str] = []
    for name_a, ca, name_b, cb, op, want in _INVARIANTS:
        got = contrast_ratio(ca, cb)
        ok = got >= want if op == ">=" else got < want
        if not ok:
            out.append(
                f"{name_a} on {name_b}: {got:.2f}:1 (must be {op} {want:.2f}:1)"
            )
    return out


def token_report() -> str:
    """Human-readable contrast table, used by ``--check`` and docs."""
    lines = ["Foreground        Background     Ratio     Rule      Result"]
    lines.append("-" * 60)
    for name_a, ca, name_b, cb, op, want in _INVARIANTS:
        got = contrast_ratio(ca, cb)
        ok = got >= want if op == ">=" else got < want
        lines.append(
            f"{name_a:<17} {name_b:<14} {got:>6.2f}:1  {op:>2} {want:<5.2f}  "
            f"{'pass' if ok else 'FAIL'}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# BUILDERS — every dpg call lives below this line
# ══════════════════════════════════════════════════════════════════════════
_CACHE: Dict[str, int] = {}
_FONTS: Dict[str, int] = {}
_FONTS_TRIED = False
_GLOBAL_THEME: Optional[int] = None


def _alive(item) -> bool:
    """True when ``item`` still exists in the current dpg context."""
    if item is None:
        return False
    try:
        return bool(dpg.does_item_exist(item))
    except SystemError:
        return False


def _cached(key: str, builder: Callable[[], int]) -> int:
    """Build once per context; return the same theme uuid afterwards."""
    tid = _CACHE.get(key)
    if _alive(tid):
        return tid  # type: ignore[return-value]
    tid = builder()
    _CACHE[key] = tid
    return tid


def reset_cache() -> None:
    """Forget every cached theme and font id. Called by :func:`setup`.

    ``_alive`` is not enough on its own. dearpygui hands out small integer ids
    and starts again from the bottom after ``destroy_context()``, so an id
    cached in the previous context usually *does* exist in the next one — as a
    button, or a window, or whatever was created in its place. Binding it then
    fails with ``Item not a theme: 120``, which is a confusing way to learn
    that a cache outlived its context.

    In an application this happens once and costs nothing. It matters because a
    test session builds several contexts in one process, and because
    "restart the UI without restarting Python" should not be a trap.
    """
    global _GLOBAL_THEME, _FONTS_TRIED
    _CACHE.clear()
    _FONTS.clear()
    _FONTS_TRIED = False
    _GLOBAL_THEME = None


# ─── fonts ────────────────────────────────────────────────────────────────
def _first_existing(candidates: Sequence[str]) -> Optional[str]:
    for path in candidates:
        try:
            if Path(path).is_file():
                return path
        except OSError:
            continue
    return None


def load_fonts() -> Dict[str, int]:
    """Register the four type levels. Idempotent; ``{}`` when no face is found.

    Returns a mapping of ``font_ui`` / ``font_h1`` / ``font_h2`` /
    ``font_mono`` to their dpg uuids. The caller binds the body face with
    ``dpg.bind_font(theme.FONT_UI_TAG)``; when this returns ``{}`` it must
    simply skip that call and the app still runs on the 13 px bitmap face.

    Do **not** add ``add_font_range_hint`` / ``add_font_chars`` /
    ``add_font_range`` here: all three are deprecated no-ops in dearpygui
    2.3.1 ("character ranges are now automatic") and only emit warnings.
    """
    global _FONTS_TRIED
    if _FONTS and all(_alive(v) for v in _FONTS.values()):
        return dict(_FONTS)
    _FONTS.clear()

    ui_path = _first_existing(_UI_FONT_CANDIDATES)
    mono_path = _first_existing(_MONO_FONT_CANDIDATES) or ui_path

    if ui_path is None:
        if not _FONTS_TRIED:
            _FONTS_TRIED = True
            _log.warning(
                "No UI font file was found; falling back to the built-in "
                "bitmap face. Text will render smaller than designed."
            )
        return {}
    _FONTS_TRIED = True

    try:
        with dpg.font_registry():
            _FONTS[FONT_UI_TAG] = dpg.add_font(ui_path, FONT_BODY,
                                               tag=FONT_UI_TAG)
            _FONTS[FONT_H1_TAG] = dpg.add_font(ui_path, FONT_H1,
                                               tag=FONT_H1_TAG)
            _FONTS[FONT_H2_TAG] = dpg.add_font(ui_path, FONT_H2,
                                               tag=FONT_H2_TAG)
            if mono_path:
                _FONTS[FONT_MONO_TAG] = dpg.add_font(mono_path, FONT_MONO,
                                                     tag=FONT_MONO_TAG)
    except (SystemError, ValueError) as exc:
        _log.warning("Font registration failed (%s); using the built-in face.",
                     exc)
        _FONTS.clear()
        return {}
    return dict(_FONTS)


def font(name: str) -> Optional[int]:
    """Look up a registered font uuid by tag, or ``None`` if unavailable."""
    tid = _FONTS.get(name)
    return tid if _alive(tid) else None


def _bind_font(item, tag: str) -> None:
    tid = font(tag)
    if tid is None or not _alive(item):
        return
    try:
        dpg.bind_item_font(item, tid)
    except SystemError:
        pass


def bind_h1(item) -> None:
    """Bind the 21 px page-title face to ``item`` (no-op without fonts)."""
    _bind_font(item, FONT_H1_TAG)


def bind_h2(item) -> None:
    """Bind the 18 px section-heading face to ``item``."""
    _bind_font(item, FONT_H2_TAG)


def bind_mono(item) -> None:
    """Bind the 14 px monospace face to ``item`` (logs, key legends)."""
    _bind_font(item, FONT_MONO_TAG)


def bind_body(item) -> None:
    """Bind the 16 px body face to ``item``."""
    _bind_font(item, FONT_UI_TAG)


# ─── global theme ─────────────────────────────────────────────────────────
def build_theme() -> int:
    """Create (once) and return the global theme uuid without binding it."""
    return _cached("global", _build_global_theme)


def _build_global_theme() -> int:
    tid = dpg.add_theme()
    with dpg.theme_component(dpg.mvAll, parent=tid):
        c = dpg.add_theme_color
        s = dpg.add_theme_style
        # surfaces
        c(dpg.mvThemeCol_WindowBg, rgba(BG))
        c(dpg.mvThemeCol_ChildBg, rgba(SURFACE))
        c(dpg.mvThemeCol_PopupBg, rgba(SURFACE))
        c(dpg.mvThemeCol_MenuBarBg, rgba(BG))
        c(dpg.mvThemeCol_FrameBg, rgba(FIELD))
        c(dpg.mvThemeCol_FrameBgHovered, rgba(SURFACE_RAISE))
        c(dpg.mvThemeCol_FrameBgActive, rgba(SURFACE_RAISE))
        # lines
        c(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
        c(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))
        c(dpg.mvThemeCol_Separator, rgba(BORDER_SUBTLE))
        c(dpg.mvThemeCol_SeparatorHovered, rgba(BORDER_STRONG))
        c(dpg.mvThemeCol_SeparatorActive, rgba(ACCENT))
        # buttons
        c(dpg.mvThemeCol_Button, rgba(BTN))
        c(dpg.mvThemeCol_ButtonHovered, rgba(BTN_HOVER))
        c(dpg.mvThemeCol_ButtonActive, rgba(BTN_ACTIVE))
        # headers / selectables / tree nodes
        c(dpg.mvThemeCol_Header, rgba(SURFACE_RAISE))
        c(dpg.mvThemeCol_HeaderHovered, (58, 64, 76, 255))
        c(dpg.mvThemeCol_HeaderActive, (66, 74, 88, 255))
        # text
        c(dpg.mvThemeCol_Text, rgba(TEXT))
        c(dpg.mvThemeCol_TextDisabled, rgba(TEXT_FAINT))
        c(dpg.mvThemeCol_TextSelectedBg, rgba(ROW_ACTIVE))
        # titles
        c(dpg.mvThemeCol_TitleBg, rgba(BG))
        c(dpg.mvThemeCol_TitleBgActive, rgba(SURFACE))
        c(dpg.mvThemeCol_TitleBgCollapsed, rgba(BG))
        # scrollbars
        c(dpg.mvThemeCol_ScrollbarBg, rgba(BG))
        c(dpg.mvThemeCol_ScrollbarGrab, rgba(BORDER_SUBTLE))
        c(dpg.mvThemeCol_ScrollbarGrabHovered, rgba(BORDER_STRONG))
        c(dpg.mvThemeCol_ScrollbarGrabActive, rgba(ACCENT_ACTIVE))
        # controls
        c(dpg.mvThemeCol_CheckMark, rgba(ACCENT))
        c(dpg.mvThemeCol_SliderGrab, rgba(ACCENT))
        c(dpg.mvThemeCol_SliderGrabActive, rgba(ACCENT_ACTIVE))
        # tabs
        c(dpg.mvThemeCol_Tab, rgba(SURFACE))
        c(dpg.mvThemeCol_TabHovered, rgba(SURFACE_RAISE))
        c(dpg.mvThemeCol_TabActive, rgba(ROW_ACTIVE))
        # tables
        c(dpg.mvThemeCol_TableHeaderBg, rgba(SURFACE_RAISE))
        c(dpg.mvThemeCol_TableBorderStrong, rgba(BORDER_SUBTLE))
        c(dpg.mvThemeCol_TableBorderLight, (54, 58, 67, 255))
        c(dpg.mvThemeCol_TableRowBg, (0, 0, 0, 0))
        c(dpg.mvThemeCol_TableRowBgAlt, (255, 255, 255, 8))
        # modals / plots
        c(dpg.mvThemeCol_ModalWindowDimBg, MODAL_DIM)
        c(dpg.mvThemeCol_PlotLines, rgba(ACCENT))
        c(dpg.mvThemeCol_PlotLinesHovered, rgba(ACCENT_HOVER))

        # styles — 17 vars; a square scrollbar next to a 7 px-rounded frame is
        # the single most visible inconsistency in the current build.
        s(dpg.mvStyleVar_FrameRounding, R_FRAME)
        s(dpg.mvStyleVar_ChildRounding, R_CHILD)
        s(dpg.mvStyleVar_WindowRounding, R_WINDOW)
        s(dpg.mvStyleVar_PopupRounding, R_POPUP)
        s(dpg.mvStyleVar_TabRounding, R_TAB)
        s(dpg.mvStyleVar_GrabRounding, R_GRAB)
        s(dpg.mvStyleVar_GrabMinSize, 12)
        s(dpg.mvStyleVar_FrameBorderSize, 1)
        s(dpg.mvStyleVar_ChildBorderSize, 1)
        s(dpg.mvStyleVar_FramePadding, 10, 6)
        s(dpg.mvStyleVar_ItemSpacing, 8, 7)
        s(dpg.mvStyleVar_ItemInnerSpacing, 6, 5)
        s(dpg.mvStyleVar_WindowPadding, 12, 10)
        s(dpg.mvStyleVar_ScrollbarSize, 13)
        s(dpg.mvStyleVar_ScrollbarRounding, 6)
        s(dpg.mvStyleVar_IndentSpacing, 18)
        s(dpg.mvStyleVar_CellPadding, 7, 5)

    # Disabled state, defined for real. This is what makes widgets.locked()
    # read as "off, on purpose" instead of "broken".
    with dpg.theme_component(dpg.mvButton, enabled_state=False, parent=tid):
        dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(SURFACE))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(SURFACE))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(SURFACE))
        dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(TEXT_FAINT))
        dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)

    _style_plots(tid)
    return tid


def _style_plots(parent: int) -> None:
    """Dark plot chrome so the training curves stop clashing with the panels."""
    cat = getattr(dpg, "mvThemeCat_Plots", 0)
    try:
        with dpg.theme_component(dpg.mvPlot, parent=parent):
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, rgba(FIELD),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, rgba(FIELD),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, rgba(BORDER_SUBTLE),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, rgba(SURFACE),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder,
                                rgba(BORDER_SUBTLE), category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, rgba(TEXT),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_TitleText, rgba(TEXT),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_AxisText, rgba(TEXT_DIM),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (255, 255, 255, 26),
                                category=cat)
            dpg.add_theme_color(dpg.mvPlotCol_Crosshairs, rgba(ACCENT),
                                category=cat)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                category=cat)
    except (SystemError, AttributeError, ValueError):
        # A dearpygui build without the plot theme category: the plots simply
        # keep their default chrome. Never fatal.
        pass


def apply_global_theme() -> int:
    """Build and bind the global theme. Idempotent; returns the theme uuid."""
    global _GLOBAL_THEME
    tid = build_theme()
    if _GLOBAL_THEME != tid or not _alive(_GLOBAL_THEME):
        dpg.bind_theme(tid)
        _GLOBAL_THEME = tid
    return tid


def apply_theme() -> int:
    """Alias for :func:`apply_global_theme` (kept for readability at call sites)."""
    return apply_global_theme()


def setup(bind_font: bool = True) -> Dict[str, int]:
    """Load fonts, apply the global theme and bind the body face.

    The one call a top-level app makes after ``create_context()`` and before
    ``create_viewport()``. Returns the font map (``{}`` when no face loaded).

    It resets the theme cache first, because that is what "a new context has
    just been created" means and this is the one place that knows it.
    """
    reset_cache()
    fonts = load_fonts()
    apply_global_theme()
    if bind_font and FONT_UI_TAG in fonts:
        try:
            dpg.bind_font(fonts[FONT_UI_TAG])
        except SystemError:
            pass
    return fonts


# ─── per-component themes ─────────────────────────────────────────────────
def _button_theme(fill: RGB, hover: RGB, active: RGB, text: RGB,
                  border: Optional[RGB], rounding: int = R_FRAME) -> int:
    tid = dpg.add_theme()
    with dpg.theme_component(dpg.mvAll, parent=tid):
        dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(fill))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(hover))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(active))
        dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(text))
        dpg.add_theme_color(dpg.mvThemeCol_Border,
                            rgba(border) if border else (0, 0, 0, 0))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, rounding)
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize,
                            1 if border else 0)
    return tid


def accent_theme() -> int:
    """Filled accent control. Text is ``ON_ACCENT`` — **never white** (1.91:1)."""
    return _cached("accent", lambda: _button_theme(
        ACCENT, ACCENT_HOVER, ACCENT_ACTIVE, ON_ACCENT, None))


def ghost_theme() -> int:
    """Transparent, bordered control — the resting state of a toolbar button."""
    return _cached("ghost", lambda: _button_theme(
        (0, 0, 0, 0), SURFACE_RAISE, BTN_ACTIVE, TEXT, BORDER_STRONG))


def danger_theme() -> int:
    """Destructive action. Filled red with near-black text."""
    return _cached("danger", lambda: _button_theme(
        DANGER, (255, 112, 112), (226, 66, 66), ON_ACCENT, None))


def neutral_theme() -> int:
    """Ordinary raised button — the suite's default secondary action."""
    return _cached("neutral", lambda: _button_theme(
        BTN, BTN_HOVER, BTN_ACTIVE, TEXT, BORDER_STRONG))


def success_theme() -> int:
    """Confirming action (Accept)."""
    return _cached("success", lambda: _button_theme(
        ACCEPTED, (78, 219, 138), (34, 172, 94), ON_ACCENT, None))


def warning_theme() -> int:
    """Pending / candidate-related action."""
    return _cached("warning", lambda: _button_theme(
        PENDING, (255, 186, 74), (214, 142, 22), ON_ACCENT, None))


def tool_theme(active: bool) -> int:
    """Convenience: ``accent_theme()`` when active, ``ghost_theme()`` otherwise.

    The annotator's ``_refresh_tool_ui`` binds this to each of the five tool
    buttons, so exactly one button is filled at any moment.
    """
    return accent_theme() if active else ghost_theme()


def toggle_theme(active: bool) -> int:
    """Alias of :func:`tool_theme` for non-tool toggles (nav rows, filters)."""
    return tool_theme(active)


def chip_theme(color: Sequence[int]) -> int:
    """A small filled label chip in ``color`` with automatically legible text.

    Text is ``ON_ACCENT`` on a light chip and ``TEXT`` on a dark one, chosen
    by measured contrast rather than by guessing.
    """
    c = rgba(color)
    key = f"chip_{c[0]}_{c[1]}_{c[2]}_{c[3]}"

    def _build() -> int:
        fg = (ON_ACCENT if contrast_ratio(ON_ACCENT, c) >= contrast_ratio(TEXT, c)
              else TEXT)
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_Button, c)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, c)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, c)
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(fg))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
        return tid

    return _cached(key, _build)


#: Status-pill kinds -> (fill, text). ``idle`` is the only unfilled variant.
PILL_KINDS: Dict[str, Tuple[RGB, RGB]] = {
    "idle": (SURFACE_RAISE, TEXT_DIM),
    "running": (ACCENT, ON_ACCENT),
    "ok": (ACCEPTED, ON_ACCENT),
    "warn": (PENDING, ON_ACCENT),
    "error": (DANGER, ON_ACCENT),
    "pending": (PENDING, ON_ACCENT),
}


def pill_theme(kind: str = "idle") -> int:
    """Theme for a status pill. ``kind``: idle|running|ok|warn|error|pending."""
    fill, text = PILL_KINDS.get(kind, PILL_KINDS["idle"])
    key = f"pill_{kind}"

    def _build() -> int:
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(fill))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(fill))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(fill))
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(text))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, rgba(fill))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 3)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
        return tid

    return _cached(key, _build)


def disabled_theme() -> int:
    """Explicit dimmed look for a locked control (see ``widgets.locked``)."""
    def _build() -> int:
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(TEXT_FAINT))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, R_FRAME)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        with dpg.theme_component(dpg.mvAll, enabled_state=False, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(TEXT_FAINT))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        return tid

    return _cached("disabled", _build)


def panel_theme() -> int:
    """Card look for a child_window: SURFACE fill, subtle border, 8 px radius."""
    def _build() -> int:
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgba(SURFACE))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, R_CHILD)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
        return tid

    return _cached("panel", _build)


def toolbar_theme() -> int:
    """Raised strip used by the annotator toolbar and the hub navigation."""
    def _build() -> int:
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgba(SURFACE_RAISE))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(BORDER_SUBTLE))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, SP_S, SP_S)
        return tid

    return _cached("toolbar", _build)


def mono_field_theme() -> int:
    """Log pane / code block: FIELD ground, no rounding, tight padding."""
    def _build() -> int:
        tid = dpg.add_theme()
        with dpg.theme_component(dpg.mvAll, parent=tid):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, rgba(FIELD))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgba(FIELD))
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(TEXT))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
        return tid

    return _cached("mono_field", _build)


def plot_theme() -> int:
    """Standalone plot theme, for plots created outside the global binding."""
    def _build() -> int:
        tid = dpg.add_theme()
        _style_plots(tid)
        return tid

    return _cached("plot", _build)


def series_theme(index: int) -> int:
    """Line/marker colour for plot series ``index``, from the shared palette."""
    color = TRACK_PALETTE[int(index) % len(TRACK_PALETTE)]
    cat = getattr(dpg, "mvThemeCat_Plots", 0)

    def _build() -> int:
        tid = dpg.add_theme()
        try:
            with dpg.theme_component(dpg.mvLineSeries, parent=tid):
                dpg.add_theme_color(dpg.mvPlotCol_Line, rgba(color),
                                    category=cat)
        except (SystemError, AttributeError, ValueError):
            pass
        return tid

    return _cached(f"series_{int(index) % len(TRACK_PALETTE)}", _build)


def bind(item, theme_id: int) -> None:
    """``bind_item_theme`` that tolerates a missing tag instead of raising."""
    if not _alive(item) or not _alive(theme_id):
        return
    try:
        dpg.bind_item_theme(item, theme_id)
    except SystemError:
        pass


__all__ = [
    # tokens
    "BG", "SURFACE", "SURFACE_RAISE", "FIELD", "ROW_HOVER", "ROW_ACTIVE",
    "BORDER_SUBTLE", "BORDER_STRONG", "TEXT", "TEXT_DIM", "TEXT_FAINT",
    "ON_ACCENT", "ACCENT", "ACCENT_HOVER", "ACCENT_ACTIVE", "ACCEPTED",
    "PENDING", "DANGER", "INTERPOLATED", "TRACK_GREY",
    "BTN", "BTN_HOVER", "BTN_ACTIVE", "MODAL_DIM",
    "PANEL", "BORDER", "MUTED", "SUCCESS", "WARNING", "INFO",
    "AI_SUGGESTION", "BACKGROUND", "CANDIDATE", "ERROR", "VERIFIED",
    "PREDICTED", "NEEDS_REVIEW", "ROLES",
    "SP_XS", "SP_S", "SP_M", "SP_L", "SP_XL", "SPACING",
    "R_FRAME", "R_CHILD", "R_WINDOW", "R_POPUP", "R_TAB", "R_GRAB",
    "FONT_H1", "FONT_H2", "FONT_BODY", "FONT_MONO",
    "FONT_UI_TAG", "FONT_H1_TAG", "FONT_H2_TAG", "FONT_MONO_TAG",
    "TRACK_PALETTE", "PILL_KINDS",
    # pure helpers
    "rgba", "track_color", "class_color", "source_color", "status_dot",
    "relative_luminance", "contrast_ratio", "verify_tokens", "token_report",
    # builders
    "load_fonts", "font", "bind_h1", "bind_h2", "bind_mono", "bind_body",
    "build_theme", "apply_global_theme", "apply_theme", "setup",
    "accent_theme", "ghost_theme", "danger_theme", "neutral_theme",
    "success_theme", "warning_theme", "tool_theme", "toggle_theme",
    "chip_theme", "pill_theme", "disabled_theme", "panel_theme",
    "toolbar_theme", "mono_field_theme", "plot_theme", "series_theme",
    "bind", "reset_cache",
]


def _main(argv: Sequence[str]) -> int:
    if "--check" not in argv:
        print("usage: python -m htrtdetr.gui.theme --check")
        return 2
    print(token_report())
    print()
    violations = verify_tokens()
    if violations:
        print(f"{len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("All contrast invariants pass.")
    print(f"Tokens: {len(ROLES)} roles, {len(TRACK_PALETTE)} track colours, "
          f"spacing {SPACING}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main(sys.argv[1:]))
