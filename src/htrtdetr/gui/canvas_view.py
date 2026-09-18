"""
canvas_view.py — キャンバスのズーム・パン・座標変換

annotator.py から切り出した表示まわりの責務。画像座標 ↔ キャンバス座標の
変換と、その基準になる scale / offset の管理だけを持つ。アノテーションの中身
(オブジェクト、track、action) には一切触れない。

切り出した理由は 2 つ:

* 5,900 行の annotator.py の中で、この一群だけは純粋な幾何計算に近く、
  他の責務と共有する状態が self._zoom / self._pan_x / self._pan_y /
  self._scale / self._offset_* / self._canvas_w / self._canvas_h
  に閉じている。
* 座標変換のズレ (キャンバス原点の取り違え、パンのクランプ漏れ) は
  「クリック位置が微妙にずれる」という形でしか現れず、見た目で気付きにくい。
  独立したモジュールにしておけば回帰テストを当てやすい。
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

import math
from typing import List, Optional, Tuple

#: Zoom clamp, multiplied onto the fit scale.
ZOOM_MIN, ZOOM_MAX = 0.2, 20.0
#: Minimum image that must stay on screen when panning, in canvas px.
PAN_KEEP = 64


def corners_of(obj) -> List[Tuple[float, float]]:
    """Four corners of anything with ``obb`` or ``bbox``.

    ``AnnObject`` has ``corners()``; ``auto_label.Candidate`` does not, and the
    review canvas draws both. One helper rather than two draw paths, so a box
    and the candidate that replaced it cannot be drawn differently.
    """
    obb = getattr(obj, "obb", None)
    if obb is not None:
        cx, cy, w, h, theta = (float(v) for v in obb[:5])
        hw, hh = w / 2.0, h / 2.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return [
            (cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]
    x1, y1, x2, y2 = (float(v) for v in getattr(obj, "bbox", (0, 0, 0, 0))[:4])
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class CanvasViewMixin:
    """CanvasViewMixin — annotator.py から切り出したミックスイン。

    ``AnnotationApp`` の属性 (``self._zoom`` など) をそのまま使う。
    メソッドは移動しただけで中身は変えていない。
    """

    def _canvas_origin(self) -> Tuple[float, float]:
        """
        描画キャンバス (drawlist) 左上の viewport 座標。

        dearpygui 2.x の child_window は item state に ``rect_min`` を持たない
        (``get_item_rect_min`` が KeyError を投げる)。drawlist は持っているので
        そちらを基準にする。ここを取り違えるとクリック座標がキャンバス原点
        (= メニューバー高さ + 余白) の分だけずれる。
        """
        for tag in ("canvas", "canvas_win"):
            if not dpg.does_item_exist(tag):
                continue
            try:
                ox, oy = dpg.get_item_rect_min(tag)
                return float(ox), float(oy)
            except (KeyError, SystemError, TypeError):
                continue
        return 0.0, 0.0

    def _get_canvas_mouse(self) -> Tuple[float, float]:
        """Mouse position in canvas-local pixel coords."""
        mx, my = dpg.get_mouse_pos(local=False)
        ox, oy = self._canvas_origin()
        return float(mx - ox), float(my - oy)

    def _recompute_scale(self) -> None:
        cw, ch = self._canvas_w, self._canvas_h
        for tag in ("canvas", "canvas_win"):
            if not dpg.does_item_exist(tag):
                continue
            try:
                cw, ch = dpg.get_item_rect_size(tag)
                break
            except (KeyError, SystemError, TypeError):
                continue
        self._canvas_w = int(cw) if cw > 1 else self._canvas_w
        self._canvas_h = int(ch) if ch > 1 else self._canvas_h
        if self.frame_w > 0 and self.frame_h > 0:
            # _scale composes the fit scale with the user's zoom, and _offset
            # composes the centering with the user's pan. Because _i2c / _c2i
            # are the only way anything converts coordinates, every hit test,
            # every handle and every draw call is correct under zoom without a
            # single further edit (docs/GUI_DESIGN.md §0.3.5).
            self._fit_scale = min(self._canvas_w / self.frame_w, self._canvas_h / self.frame_h)
            self._scale = self._fit_scale * self._zoom
            self._offset_x = (self._canvas_w - self.frame_w * self._scale) / 2 + self._pan_x
            self._offset_y = (self._canvas_h - self.frame_h * self._scale) / 2 + self._pan_y

    # ─── Zoom and pan ────────────────────────────────────────────────

    def _clamp_pan(self) -> None:
        """Keep at least ``PAN_KEEP`` px of image on screen.

        Without this the image can be panned entirely out of the canvas and the
        only way back is to reopen the video.
        """
        if self.frame_w <= 0 or self.frame_h <= 0:
            return
        width = self.frame_w * self._scale
        height = self.frame_h * self._scale
        base_x = (self._canvas_w - width) / 2
        base_y = (self._canvas_h - height) / 2
        self._pan_x = max(
            min(self._pan_x, self._canvas_w - PAN_KEEP - base_x), PAN_KEEP - width - base_x
        )
        self._pan_x = max(-self._canvas_w + PAN_KEEP, min(self._canvas_w - PAN_KEEP, self._pan_x))
        self._pan_y = max(
            min(self._pan_y, self._canvas_h - PAN_KEEP - base_y), PAN_KEEP - height - base_y
        )

    def fit_view(self) -> None:
        """Reset zoom and pan. The escape hatch from any lost view."""
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._recompute_scale()
        self._refresh_zoom_label()
        self._render_canvas()

    def zoom_by(self, factor: float, anchor: Optional[Tuple[float, float]] = None) -> None:
        """Multiply the zoom, keeping ``anchor`` (canvas px) over the same pixel.

        Anchoring is what makes wheel zoom usable: without it, zooming in on a
        fly in the corner walks it off the screen and the user has to pan back
        after every step.
        """
        if self.cap is None:
            return
        if anchor is None:
            anchor = (self._canvas_w / 2.0, self._canvas_h / 2.0)
        before = self._c2i(*anchor)
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * float(factor)))
        self._recompute_scale()
        after = self._i2c(*before)
        self._pan_x += anchor[0] - after[0]
        self._pan_y += anchor[1] - after[1]
        self._clamp_pan()
        self._recompute_scale()
        self._refresh_zoom_label()
        self._render_canvas()

    def _drag_pan(self, cx: float, cy: float) -> None:
        """Move the view by the distance dragged since the press.

        From the *press* origin rather than frame to frame: accumulating deltas
        drifts away from the cursor over a long drag, and the image ends up not
        under the hand that is moving it.
        """
        if self._pan_origin is None:
            return
        ox, oy, start_x, start_y = self._pan_origin
        self._pan_x = start_x + (cx - ox)
        self._pan_y = start_y + (cy - oy)
        self._clamp_pan()
        self._recompute_scale()
        self._render_canvas()

    def _on_middle_press(self, sender=None, app_data=None) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        self._middle_pan = (cx, cy, self._pan_x, self._pan_y)

    def _on_middle_drag(self, sender=None, app_data=None) -> None:
        if self._middle_pan is None or not self._is_over_canvas():
            return
        saved, self._pan_origin = self._pan_origin, self._middle_pan
        try:
            self._drag_pan(*self._get_canvas_mouse())
        finally:
            self._pan_origin = saved

    def _on_middle_release(self, sender=None, app_data=None) -> None:
        self._middle_pan = None

    def _refresh_zoom_label(self) -> None:
        if dpg.does_item_exist("rail_zoom"):
            dpg.set_value("rail_zoom", f"{self._zoom * 100:.0f}%")

    def _on_mouse_wheel(self, _sender=None, app_data=0) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        self.zoom_by(1.25 if float(app_data) > 0 else 0.8, anchor=self._get_canvas_mouse())

    def _on_mouse_move(self, _sender=None, _app_data=None) -> None:
        """Track the cursor for the crosshair the drawing tools show.

        Only the tools that need it pay: in Select the branch returns before
        touching anything, so the common case costs one comparison per move
        event rather than a re-render.
        """
        if self._tool not in ("draw", "obb", "delete", "click"):
            if self._cursor_canvas is not None:
                self._cursor_canvas = None
                self._render_canvas()
            return
        if not self._is_over_canvas():
            self._cursor_canvas = None
            return
        self._cursor_canvas = self._get_canvas_mouse()
        if self._mode == "idle":
            self._render_canvas()

    def _i2c(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale + self._offset_x, y * self._scale + self._offset_y

    def _c2i(self, cx: float, cy: float) -> Tuple[float, float]:
        s = max(self._scale, 1e-9)
        return (cx - self._offset_x) / s, (cy - self._offset_y) / s
