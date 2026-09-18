"""
timeline.py — the whole video as one strip, so nothing has to be hunted for.

The annotator's old timeline drew a green tick per annotated frame and a
playhead. That answers one question ("have I been here?") and no others. After
a detector has run over 54,000 frames, the questions that matter are *where are
the 94 problems*, *which stretches has the model already done well*, *where did
Smart Annotate say I should work*, and *when does this behavior happen* — and
none of them can be answered by scrubbing.

So this is five stacked lanes over one shared frame axis::

    ┌──────────────────────────────────────────────────────── playhead ─┐
    │ status    ████░░░░████████░░░░░░░░████████████████░░░░░░░░       │
    │ issues        ◆        ✕     ◆ ◆            ▲                    │
    │ behavior  ▬▬▬▬▬▬  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬     ▬▬▬▬▬▬▬▬▬▬▬               │
    │ samples    ○    ○   ○  ■      ○    ○       ■   ○                 │
    └───────────────────────────────────────────────────────────────────┘

Three rules
-----------

**Shape carries meaning, not only colour** (§22). An ID-switch risk is a cross,
a missing animal is a triangle, everything else is a diamond; a Smart Annotate
pick is a hollow circle and a keyframe is a filled square. Colour is a second
channel, never the only one — 8% of men cannot use it as the first.

**Cost is O(width), not O(frames).** A 54,000-frame video is 1,100 pixels wide.
Every lane buckets into pixel columns before drawing, so the draw call count
depends on the window, not on the video. The old timeline already did this for
its one lane (``columns = {int(...) for f in annotations}``); this generalises
it, because five lanes over 54,000 frames drawn naively is 270,000 draw items
per render and dearpygui will not survive it.

**Clicking a marker goes to the issue, not to the pixel.** The nearest marker
inside a tolerance wins; only an empty column falls through to a plain seek.
Otherwise a 3-pixel-wide marker is unclickable and the lane is decoration.

Purity: this module imports dearpygui (it draws) but **never** torch or cv2, and
it holds no annotator state — everything it draws arrives in a
:class:`TimelineModel`, so it can be rendered in a test with no video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional,
                    Sequence, Set, Tuple)

import dearpygui.dearpygui as dpg

from . import theme

__all__ = [
    "Lane", "LANES", "TimelineModel", "Timeline",
    "MARKER_SHAPES", "SHAPE_HELP",
]


# ══════════════════════════════════════════════════════════════════════════
# Lanes
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Lane:
    """One horizontal strip. ``height`` is in pixels and fixed per lane."""

    key: str
    label: str
    height: int
    help: str = ""


#: Top to bottom. The scrubber is not a lane — it spans all of them.
LANES: Tuple[Lane, ...] = (
    Lane("status", "Annotations", 10,
         "Green: you verified these. Blue-grey: the model predicted them. "
         "Amber: they need review."),
    Lane("issues", "Review", 12,
         "One marker per thing to look at. Click one to jump to it."),
    Lane("behavior", "Behavior", 9,
         "Behavior intervals, coloured by label. Hollow means not yet confirmed."),
    Lane("samples", "Samples", 9,
         "Circles: frames Smart Annotate chose. Squares: frames you placed a box on."),
)

#: Issue kind -> marker shape. Kinds not named here use the default diamond.
#: Only the three kinds a human treats differently get their own shape; giving
#: all fourteen a distinct glyph would produce a legend nobody can hold.
MARKER_SHAPES: Dict[str, str] = {
    "id_switch_risk": "cross",
    "missing_animal": "triangle",
    "merged_detection": "triangle",
}

SHAPE_HELP: Tuple[Tuple[str, str], ...] = (
    ("X", "Possible ID switch"),
    ("^", "Missing or merged animal"),
    ("<>", "Everything else to review"),
    ("o", "Smart Annotate sample"),
    ("[]", "Frame you placed a box on"),
)


# ══════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TimelineModel:
    """Everything the timeline draws. Plain data, rebuilt when something changes.

    Kept separate from the renderer so the annotator can assemble it once per
    mutation rather than once per frame, and so a test can assert what *would*
    be drawn without a viewport.
    """

    num_frames: int = 0
    current: int = 0
    fps: float = 25.0

    #: Frames holding at least one box a human kept.
    verified: Set[int] = field(default_factory=set)
    #: Frames holding machine output nothing objected to.
    predicted: Set[int] = field(default_factory=set)
    #: Frames holding machine output that a detector flagged.
    needs_review: Set[int] = field(default_factory=set)

    #: ``(start, end, kind, severity)`` per review item, end inclusive.
    issues: List[Tuple[int, int, str, float]] = field(default_factory=list)
    #: ``(start, end, action_id, confirmed)`` per behavior interval.
    behaviors: List[Tuple[int, int, int, bool]] = field(default_factory=list)

    #: Frames Smart Annotate selected.
    samples: Set[int] = field(default_factory=set)
    #: ``(start, end)`` clips Smart Annotate selected.
    clips: List[Tuple[int, int]] = field(default_factory=list)
    #: Frames a human placed a box on — the propagation anchors.
    keyframes: Set[int] = field(default_factory=set)

    def is_empty(self) -> bool:
        return self.num_frames <= 0

    def has_lane(self, key: str) -> bool:
        """Is there anything in this lane? Empty lanes are drawn as a thin rule
        rather than omitted, so the layout does not jump when a run finishes."""
        return {
            "status": bool(self.verified or self.predicted or self.needs_review),
            "issues": bool(self.issues),
            "behavior": bool(self.behaviors),
            "samples": bool(self.samples or self.clips or self.keyframes),
        }.get(key, False)

    def summary(self) -> str:
        """The one line under the timeline."""
        if self.is_empty():
            return "No video loaded"
        parts: List[str] = []
        if self.verified:
            parts.append(f"{len(self.verified):,} verified")
        if self.predicted or self.needs_review:
            parts.append(f"{len(self.predicted) + len(self.needs_review):,} predicted")
        if self.issues:
            parts.append(f"{len(self.issues):,} to review")
        if self.samples or self.clips:
            parts.append(f"{len(self.samples) + len(self.clips):,} sampled")
        return "   ".join(parts) if parts else "No annotations yet"


# ══════════════════════════════════════════════════════════════════════════
# Renderer
# ══════════════════════════════════════════════════════════════════════════

class Timeline:
    """A drawlist that renders a :class:`TimelineModel` and reports clicks.

    Usage::

        self.timeline = Timeline("ann_timeline", on_seek=self.goto_frame,
                                 on_issue=self.jump_to_issue)
        self.timeline.build(parent="bottom_bar", width=-1)
        self.timeline.set_model(model)     # on any annotation change
        self.timeline.render()             # on any frame change
    """

    #: Gutter each side, so the first and last frame are clickable.
    PAD = 10
    #: Scrubber strip height, above the lanes.
    SCRUB_H = 18
    #: Click tolerance for a marker, in pixels.
    HIT_PX = 6

    def __init__(
        self,
        tag: str,
        *,
        on_seek: Optional[Callable[[int], None]] = None,
        on_issue: Optional[Callable[[Any], None]] = None,
        on_clip: Optional[Callable[[Tuple[int, int]], None]] = None,
        lanes: Sequence[Lane] = LANES,
    ) -> None:
        self.tag = tag
        self.on_seek = on_seek
        self.on_issue = on_issue
        self.on_clip = on_clip
        self.lanes = tuple(lanes)
        self.model = TimelineModel()
        self._width = 800
        self._markers: List[Tuple[float, float, Any]] = []
        self._scrubbing = False

    # ── geometry ─────────────────────────────────────────────────────────
    @property
    def height(self) -> int:
        """Total drawlist height, including the scrubber and inter-lane gaps."""
        return self.SCRUB_H + sum(lane.height + 3 for lane in self.lanes) + 4

    def _span(self) -> Tuple[float, float]:
        return float(self.PAD), float(max(self._width - self.PAD, self.PAD + 1))

    def frame_to_x(self, frame: int) -> float:
        x0, x1 = self._span()
        total = max(self.model.num_frames - 1, 1)
        ratio = min(max(float(frame) / total, 0.0), 1.0)
        return x0 + (x1 - x0) * ratio

    def x_to_frame(self, x: float) -> int:
        x0, x1 = self._span()
        if x1 <= x0:
            return 0
        ratio = min(max((float(x) - x0) / (x1 - x0), 0.0), 1.0)
        return int(round(ratio * max(self.model.num_frames - 1, 0)))

    def _lane_top(self, key: str) -> float:
        y = float(self.SCRUB_H)
        for lane in self.lanes:
            if lane.key == key:
                return y
            y += lane.height + 3
        return y

    # ── lifecycle ────────────────────────────────────────────────────────
    def build(self, parent: str, width: int = -1) -> None:
        """Create the drawlist. Safe to call twice — it replaces itself."""
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        dpg.add_drawlist(tag=self.tag, parent=parent,
                         width=self._width if width == -1 else width,
                         height=self.height)

    def resize(self, width: int) -> None:
        self._width = max(120, int(width))
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, width=self._width, height=self.height)

    def set_model(self, model: TimelineModel) -> None:
        self.model = model

    def alive(self) -> bool:
        return dpg.does_item_exist(self.tag)

    # ── rendering ────────────────────────────────────────────────────────
    def render(self) -> None:
        """Redraw everything. Cheap enough to call on every frame change."""
        if not self.alive():
            return
        dpg.delete_item(self.tag, children_only=True)
        self._markers = []

        x0, x1 = self._span()
        total_h = self.height

        dpg.draw_rectangle((0, 0), (self._width, total_h),
                           color=theme.rgba(theme.BG, 0), fill=theme.rgba(theme.FIELD),
                           parent=self.tag)

        if self.model.is_empty():
            dpg.draw_text((x0 + 6, self.SCRUB_H * 0.25), "No video loaded",
                          color=theme.rgba(theme.TEXT_FAINT), size=13,
                          parent=self.tag)
            return

        self._draw_scrub_track(x0, x1)
        self._draw_status_lane(x0, x1)
        self._draw_issue_lane(x0, x1)
        self._draw_behavior_lane(x0, x1)
        self._draw_sample_lane(x0, x1)
        self._draw_playhead(total_h)

    # -- scrubber --------------------------------------------------------
    def _draw_scrub_track(self, x0: float, x1: float) -> None:
        top, bot = 5.0, float(self.SCRUB_H - 5)
        dpg.draw_rectangle((x0, top), (x1, bot),
                           color=theme.rgba(theme.BORDER_SUBTLE),
                           fill=theme.rgba(theme.SURFACE), parent=self.tag)

    # -- lane 1: annotation status --------------------------------------
    def _draw_status_lane(self, x0: float, x1: float) -> None:
        top = self._lane_top("status")
        height = self._lane_of("status").height
        self._lane_ground(x0, x1, top, height)

        # Bucket by pixel column so a 54,000-frame video costs ~1,100 draws.
        # Precedence within a column: needs_review wins over predicted wins over
        # verified, because the column has to advertise the *worst* thing in it
        # — a column that is 99% verified and 1% broken is a column to visit.
        columns: Dict[int, int] = {}
        for frames, rank in ((self.model.verified, 1),
                             (self.model.predicted, 2),
                             (self.model.needs_review, 3)):
            for frame in frames:
                column = int(self.frame_to_x(frame))
                if columns.get(column, 0) < rank:
                    columns[column] = rank

        colors = {1: theme.VERIFIED, 2: theme.PREDICTED, 3: theme.NEEDS_REVIEW}
        for column, rank in columns.items():
            dpg.draw_line((column, top + 1), (column, top + height - 1),
                          color=theme.rgba(colors[rank], 235), thickness=1.0,
                          parent=self.tag)

    # -- lane 2: review markers ------------------------------------------
    def _draw_issue_lane(self, x0: float, x1: float) -> None:
        lane = self._lane_of("issues")
        top = self._lane_top("issues")
        self._lane_ground(x0, x1, top, lane.height)
        mid = top + lane.height / 2.0
        size = lane.height / 2.0 - 1

        # One marker per pixel column, keeping the most severe. Two issues three
        # frames apart are one pixel apart; drawing both draws them on top of
        # each other and makes the lane look sparser than the queue is.
        best: Dict[int, Tuple[float, int, int, str, float]] = {}
        for start, end, kind, severity in self.model.issues:
            column = int(self.frame_to_x((start + end) // 2))
            if column not in best or severity > best[column][0]:
                best[column] = (severity, start, end, kind, severity)

        for column, (_, start, end, kind, severity) in best.items():
            color = theme.rgba(
                theme.ERROR if severity >= 0.66 else theme.NEEDS_REVIEW,
                200 + int(55 * min(1.0, severity)))
            shape = MARKER_SHAPES.get(kind, "diamond")
            self._draw_marker(column, mid, size, shape, color)
            self._markers.append((float(column), mid, (start, end, kind)))

            # An interval wider than a few pixels also gets an underline, so a
            # long problem does not look like a point problem.
            span_x0, span_x1 = self.frame_to_x(start), self.frame_to_x(end)
            if span_x1 - span_x0 >= 3:
                dpg.draw_line((span_x0, top + lane.height - 1),
                              (span_x1, top + lane.height - 1),
                              color=color, thickness=1.5, parent=self.tag)

    # -- lane 3: behavior intervals --------------------------------------
    def _draw_behavior_lane(self, x0: float, x1: float) -> None:
        lane = self._lane_of("behavior")
        top = self._lane_top("behavior")
        self._lane_ground(x0, x1, top, lane.height)

        for start, end, action_id, confirmed in self.model.behaviors:
            left = self.frame_to_x(start)
            right = max(self.frame_to_x(end), left + 1.5)
            color = theme.class_color(action_id)
            if confirmed:
                dpg.draw_rectangle((left, top + 1), (right, top + lane.height - 1),
                                   color=theme.rgba(color, 255),
                                   fill=theme.rgba(color, 190), parent=self.tag)
            else:
                # Hollow = not yet confirmed. The same distinction the canvas
                # makes with a dashed stroke, in the one channel a 9-pixel-tall
                # bar has left.
                dpg.draw_rectangle((left, top + 1), (right, top + lane.height - 1),
                                   color=theme.rgba(color, 235),
                                   fill=theme.rgba(color, 55), parent=self.tag)

    # -- lane 4: samples and keyframes ------------------------------------
    def _draw_sample_lane(self, x0: float, x1: float) -> None:
        lane = self._lane_of("samples")
        top = self._lane_top("samples")
        self._lane_ground(x0, x1, top, lane.height)
        mid = top + lane.height / 2.0
        radius = lane.height / 2.0 - 1

        for start, end in self.model.clips:
            left = self.frame_to_x(start)
            right = max(self.frame_to_x(end), left + 2)
            dpg.draw_rectangle((left, top + 2), (right, top + lane.height - 2),
                               color=theme.rgba(theme.ACCENT, 200),
                               fill=theme.rgba(theme.ACCENT, 60), parent=self.tag)

        for column in {int(self.frame_to_x(f)) for f in self.model.samples}:
            dpg.draw_circle((column, mid), radius,
                            color=theme.rgba(theme.ACCENT, 230), thickness=1.2,
                            parent=self.tag)

        for column in {int(self.frame_to_x(f)) for f in self.model.keyframes}:
            dpg.draw_rectangle((column - radius + 1, mid - radius + 1),
                               (column + radius - 1, mid + radius - 1),
                               color=theme.rgba(theme.VERIFIED, 235),
                               fill=theme.rgba(theme.VERIFIED, 235), parent=self.tag)

    # -- playhead ---------------------------------------------------------
    def _draw_playhead(self, total_h: float) -> None:
        x = self.frame_to_x(self.model.current)
        color = theme.rgba(theme.PENDING if self._scrubbing else theme.TEXT, 255)
        dpg.draw_line((x, 2), (x, total_h - 2), color=color, thickness=1.5,
                      parent=self.tag)
        dpg.draw_triangle((x - 5, 0), (x + 5, 0), (x, 7),
                          color=color, fill=color, parent=self.tag)
        if self._scrubbing:
            x0, x1 = self._span()
            label_x = min(max(x + 8, x0), max(x1 - 56, x0))
            dpg.draw_text((label_x, 2), str(self.model.current),
                          color=color, size=13, parent=self.tag)

    # -- primitives -------------------------------------------------------
    def _lane_of(self, key: str) -> Lane:
        for lane in self.lanes:
            if lane.key == key:
                return lane
        return Lane(key, key, 8)

    def _lane_ground(self, x0: float, x1: float, top: float, height: int) -> None:
        """An empty lane is a thin rule, not nothing: the layout must not jump
        when a run finishes and a lane suddenly has content."""
        dpg.draw_rectangle((x0, top), (x1, top + height),
                           color=theme.rgba(theme.BORDER_SUBTLE, 90),
                           fill=theme.rgba(theme.SURFACE, 150), parent=self.tag)

    def _draw_marker(self, x: float, y: float, size: float,
                     shape: str, color: Sequence[int]) -> None:
        if shape == "cross":
            dpg.draw_line((x - size, y - size), (x + size, y + size),
                          color=color, thickness=1.6, parent=self.tag)
            dpg.draw_line((x - size, y + size), (x + size, y - size),
                          color=color, thickness=1.6, parent=self.tag)
        elif shape == "triangle":
            dpg.draw_triangle((x, y - size), (x + size, y + size),
                              (x - size, y + size),
                              color=color, fill=color, parent=self.tag)
        else:
            dpg.draw_quad((x, y - size), (x + size, y), (x, y + size), (x - size, y),
                          color=color, fill=color, parent=self.tag)

    # ── interaction ──────────────────────────────────────────────────────
    def mouse_pos(self) -> Optional[Tuple[float, float]]:
        """Cursor position in drawlist coordinates, or ``None`` when outside.

        Uses ``get_item_rect_min`` on the **drawlist** — a ``child_window`` has
        no ``rect_min`` and raises ``KeyError``, which is the measured pitfall
        that cost this project months (``docs/GUI_DESIGN.md`` §5.1).
        """
        if not self.alive():
            return None
        try:
            origin = dpg.get_item_rect_min(self.tag)
        except Exception:
            return None
        mx, my = dpg.get_mouse_pos(local=False)
        return mx - origin[0], my - origin[1]

    def hovered(self) -> bool:
        return self.alive() and dpg.is_item_hovered(self.tag)

    def marker_at(self, x: float, y: float) -> Optional[Any]:
        """The nearest review marker within :attr:`HIT_PX`, or ``None``.

        Tolerance in x only, plus a generous band in y: a marker is 5 pixels
        tall and demanding a hit inside it makes the lane decorative.
        """
        lane_top = self._lane_top("issues")
        lane = self._lane_of("issues")
        if not (lane_top - 4 <= y <= lane_top + lane.height + 4):
            return None
        best, best_dx = None, float(self.HIT_PX)
        for mx, _my, payload in self._markers:
            dx = abs(mx - x)
            if dx <= best_dx:
                best, best_dx = payload, dx
        return best

    def clip_at(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """The sampled clip under ``(x, y)``, or ``None``.

        No pixel tolerance, unlike :meth:`marker_at`: a clip is drawn as a
        rectangle spanning its own frames, so it is already as wide as it is
        and a tolerance would let a click next to a clip select it. The band in
        y is generous for the same reason the marker one is - the lane is 10
        pixels tall.

        The narrowest clip wins when two overlap. A short clip inside a long
        one is the harder of the two to hit, and it is also the more specific
        answer to "which clip did you mean".
        """
        lane_top = self._lane_top("samples")
        lane = self._lane_of("samples")
        if not (lane_top - 3 <= y <= lane_top + lane.height + 3):
            return None
        best = None
        for start, end in self.model.clips:
            left = self.frame_to_x(start)
            right = max(self.frame_to_x(end), left + 2)
            if left - 1 <= x <= right + 1:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (int(start), int(end))
        return best

    def on_press(self) -> bool:
        """Handle a left press. ``True`` when the timeline consumed it.

        Marker first, seek second. Clicking a marker is the feature; falling
        through to a seek would put the playhead near the issue and select
        nothing, which looks like the click missed.
        """
        if not self.hovered():
            return False
        pos = self.mouse_pos()
        if pos is None:
            return False
        x, y = pos
        marker = self.marker_at(x, y)
        if marker is not None and self.on_issue is not None:
            self.on_issue(marker)
            return True
        clip = self.clip_at(x, y)
        if clip is not None and self.on_clip is not None:
            # Seek as well, unlike a marker hit. A clip is a span rather than a
            # point, so the frame the user aimed at is information - and the
            # propagation the clip pane offers reads the playhead to decide
            # which frame the human was looking at.
            self._seek_to(x)
            self.on_clip(clip)
            return True
        self._scrubbing = True
        self._seek_to(x)
        return True

    def on_drag(self) -> bool:
        if not self._scrubbing:
            return False
        pos = self.mouse_pos()
        if pos is not None:
            self._seek_to(pos[0])
        return True

    def on_release(self) -> bool:
        if not self._scrubbing:
            return False
        self._scrubbing = False
        self.render()
        return True

    @property
    def scrubbing(self) -> bool:
        return self._scrubbing

    def _seek_to(self, x: float) -> None:
        frame = self.x_to_frame(x)
        if self.on_seek is not None and frame != self.model.current:
            self.on_seek(frame)
        else:
            self.render()


# ══════════════════════════════════════════════════════════════════════════
# Model assembly
# ══════════════════════════════════════════════════════════════════════════

def build_model(
    *,
    num_frames: int,
    current: int,
    fps: float = 25.0,
    annotations: Optional[Mapping[int, Sequence[Any]]] = None,
    candidates: Optional[Mapping[int, Sequence[Any]]] = None,
    review_items: Sequence[Any] = (),
    behavior_intervals: Sequence[Any] = (),
    selection: Any = None,
    confident_score: float = 0.70,
) -> TimelineModel:
    """Assemble a :class:`TimelineModel` from the annotator's live state.

    Kept a free function rather than a method so the annotator can call it from
    one place after any mutation, and so a test can build a model with no
    viewport. Every argument is optional: a model with only ``num_frames`` draws
    four empty lanes and a playhead, which is the correct empty state.
    """
    model = TimelineModel(num_frames=int(num_frames), current=int(current),
                          fps=float(fps))

    flagged: Set[int] = set()
    for item in review_items:
        start = int(getattr(item, "start", 0))
        end = int(getattr(item, "end", start))
        model.issues.append((start, end, str(getattr(item, "kind", "")),
                             float(getattr(item, "severity", 0.5))))
        flagged.update(range(start, min(end, model.num_frames - 1) + 1))

    for frame, objects in (annotations or {}).items():
        if objects:
            model.verified.add(int(frame))
            if any(getattr(o, "keyframe", True) for o in objects):
                model.keyframes.add(int(frame))

    for frame, row in (candidates or {}).items():
        if not row:
            continue
        frame = int(frame)
        weak = any(float(getattr(c, "score", 1.0) or 0.0) < confident_score
                   for c in row)
        if frame in flagged or weak:
            model.needs_review.add(frame)
        else:
            model.predicted.add(frame)

    for interval in behavior_intervals:
        model.behaviors.append((
            int(getattr(interval, "start", 0)),
            int(getattr(interval, "end", 0)),
            int(getattr(interval, "action_id", 0)),
            getattr(interval, "source", "auto") == "human",
        ))

    if selection is not None:
        model.samples.update(int(f) for f in getattr(selection, "frames", ()))
        for clip in getattr(selection, "clips", ()):
            model.clips.append((int(clip.start), int(clip.end)))

    return model
