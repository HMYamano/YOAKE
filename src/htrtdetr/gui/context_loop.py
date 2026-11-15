"""
context_loop.py — watch the motion around a keyframe without leaving it.

Keyframe labelling (``docs/KEYFRAME_BEHAVIOR.md``) attaches a behavior to one
frame, but behavior is motion: grooming and resting can be the same still
image. AVA's annotators labelled a keyframe while watching the 3 seconds around
it; this is that procedure made into a key.

``P`` loops the frames from 1.0 s before the keyframe to 0.5 s after it, on the
canvas, with the keyframe's boxes drawn over them as outlines. Labels are only
ever attached to the keyframe - the loop never moves ``current_frame_idx`` - so
selecting a box, pressing ``P`` and then a behavior number is the whole gesture.
``P``, ``Esc``, a click on the canvas or moving to another frame ends it.

The clock and the range are :class:`htrtdetr.annotation.keyframes.ContextLoop`
(pure, tested without a window). This mixin decodes and paints:

* The loop's frames are decoded **once**, when it starts, through the same
  sequential reader the canvas uses (one seek, then reads - P15), and kept as
  BGR ``uint8``. A 45-frame loop of 640x480 is 41 MB; the float RGBA texture
  dearpygui wants would be 220 MB, so the conversion happens per shown frame,
  at the same cost a normal frame step already pays.
* The loop is capped at :data:`MAX_FRAMES` so a 240 fps recording cannot ask
  for a gigabyte.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ..annotation import keyframes

__all__ = ["ContextLoopMixin", "BEFORE_S", "AFTER_S", "MAX_FRAMES"]

BEFORE_S = 1.0
AFTER_S = 0.5
MAX_FRAMES = 120


class ContextLoopMixin:
    """Mixed into ``AnnotationApp``. Uses its capture, texture and canvas."""

    # ── state ────────────────────────────────────────────────────────────
    def _context_loop_state(self) -> keyframes.ContextLoop:
        loop = self.__dict__.get("_ctx_loop")
        if loop is None:
            loop = keyframes.ContextLoop()
            self.__dict__["_ctx_loop"] = loop
            self.__dict__["_ctx_frames"] = {}
        return loop

    @property
    def context_loop_active(self) -> bool:
        return self._context_loop_state().active

    # ── the key ──────────────────────────────────────────────────────────
    def toggle_context_loop(self) -> None:
        if self.context_loop_active:
            self.stop_context_loop()
        else:
            self.start_context_loop()

    def start_context_loop(self) -> bool:
        """Start looping the frames around the current one. False if it cannot."""
        if getattr(self, "cap", None) is None:
            self._status("Open a video first - the loop plays the frames around this one.")
            return False
        anchor = int(self.current_frame_idx)
        lo, hi = keyframes.context_window(
            anchor, float(self.fps or 0.0),
            before_s=BEFORE_S, after_s=AFTER_S, num_frames=int(self.num_frames or 0),
        )
        if hi - lo + 1 > MAX_FRAMES:
            # Keep the after-part whole; trim the far past.
            lo = max(lo, hi - MAX_FRAMES + 1)
        frames = self._decode_context(lo, hi)
        if anchor not in frames and self._current_bgr is not None:
            frames[anchor] = self._current_bgr
        if len(frames) < 2:
            self._status("The frames around this one could not be read, so there is nothing to play.")
            return False
        loop = self._context_loop_state()
        self.__dict__["_ctx_frames"] = frames
        loop.start(anchor, lo, hi, now=time.time())
        fps = float(self.fps or 0.0)
        span = f"{(lo - anchor) / fps:+.1f} s to {(hi - anchor) / fps:+.1f} s" if fps > 0 \
            else f"{lo - anchor:+d} to {hi - anchor:+d} frames"
        self._status(
            f"Playing {span} around this frame. Labels still go on this frame - "
            "press a behavior number any time. P or Esc returns."
        )
        self._show_context_frame(lo)
        return True

    def stop_context_loop(self, restore: bool = True) -> None:
        """End the loop and put the keyframe back on the canvas."""
        loop = self._context_loop_state()
        if not loop.active:
            return
        loop.stop()
        self.__dict__["_ctx_frames"] = {}
        if restore and self._current_bgr is not None:
            self._upload_frame_texture(self._current_bgr)
            self._render_canvas()

    # ── the render loop ──────────────────────────────────────────────────
    def _tick_context_loop(self) -> None:
        loop = self._context_loop_state()
        if not loop.active:
            return
        nxt = loop.tick(time.time(), float(self.fps or 0.0))
        if nxt is not None:
            self._show_context_frame(nxt)

    # ── decoding and painting ────────────────────────────────────────────
    def _decode_context(self, lo: int, hi: int) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for index in range(int(lo), int(hi) + 1):
            frame = self._read_frame_bgr(index)
            if frame is not None:
                out[index] = frame
        return out

    def _upload_frame_texture(self, frame_bgr: np.ndarray) -> None:
        if dpg is None or cv2 is None or not dpg.does_item_exist(self._frame_tex):
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
        dpg.set_value(self._frame_tex, (rgba.astype(np.float32) / 255.0).flatten())

    def _show_context_frame(self, index: int) -> None:
        frames: Dict[int, np.ndarray] = self.__dict__.get("_ctx_frames") or {}
        if not frames:
            return
        frame = frames.get(int(index))
        if frame is None:
            nearest = min(frames, key=lambda i: abs(i - int(index)))
            frame = frames[nearest]
        self._upload_frame_texture(frame)
        self._render_canvas()

    def _draw_context_overlay(self, cw: int, ch: int) -> None:
        """Where in the loop we are: a bar, the keyframe's tick, and the offset."""
        loop = self._context_loop_state()
        if dpg is None or not loop.active or not dpg.does_item_exist("canvas"):
            return
        fps = float(self.fps or 0.0)
        bar_y = 4
        dpg.draw_rectangle((8, bar_y), (cw - 8, bar_y + 6), color=(0, 0, 0, 140),
                           fill=(0, 0, 0, 120), parent="canvas")
        x = 8 + (cw - 16) * loop.progress()
        dpg.draw_rectangle((8, bar_y), (x, bar_y + 6), color=(240, 200, 80, 220),
                           fill=(240, 200, 80, 160), parent="canvas")
        if loop.anchor is not None and loop.hi > loop.lo:
            kx = 8 + (cw - 16) * (loop.anchor - loop.lo) / float(loop.hi - loop.lo)
            dpg.draw_line((kx, bar_y - 2), (kx, bar_y + 8), color=(255, 255, 255, 255),
                          thickness=2.0, parent="canvas")
        offset = loop.offset_seconds(fps) if fps > 0 else float(loop.position - (loop.anchor or 0))
        here = "keyframe" if loop.position == loop.anchor else (
            f"{offset:+.2f} s" if fps > 0 else f"{int(offset):+d} frames")
        text = f"Context  {here}   (P: back to the keyframe)"
        dpg.draw_text((11, bar_y + 11), text, color=(0, 0, 0, 200), size=15, parent="canvas")
        dpg.draw_text((10, bar_y + 10), text, color=(255, 236, 170, 255), size=15, parent="canvas")

    def _context_outline_only(self) -> bool:
        """While looping, boxes are drawn unfilled so the motion under them shows."""
        return self.context_loop_active
