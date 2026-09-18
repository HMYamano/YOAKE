"""
review_mode.py — 差分レビュー画面 (前後比較 + 採否)

annotator.py から切り出した、レビューキューを 1 件ずつ見て「採用 / 却下 /
入れ替え」を決める画面の責務。通常の編集キャンバスとは別の drawlist へ描画し、
別のキー割り当てで動くので、編集側と混ぜずに持てる。

swap_tracks もここに置く: ID の取り違えを見つけて直す操作は、レビュー画面が
唯一の入口であり、**あるフレーム以降の 2 つの track を丸ごと入れ替える**という
アノテーションに対する最も破壊的な編集だからである (履歴には 1 手として積む)。
"""

from __future__ import annotations

import time
from typing import Tuple

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

try:
    from .canvas_view import corners_of
except ImportError:  # pragma: no cover - スクリプト直接実行時
    from htrtdetr.gui.canvas_view import corners_of

# ``theme`` は import しただけで dearpygui を要求する。``htrtdetr.annotator`` は
# dearpygui の無い環境でも import できなければならない (アノテーション I/O だけを
# 使うテストと、torch-free / GUI-free なパイプラインがそれに依存している) ので、
# dpg があるときだけ読み込む。無い環境ではレビュー画面自体が起動しない。
if dpg is not None:
    try:
        from . import theme
    except ImportError:  # pragma: no cover
        from htrtdetr.gui import theme
else:  # pragma: no cover - GUI なしの環境
    theme = None

class ReviewModeMixin:
    """ReviewModeMixin — annotator.py から切り出したミックスイン。

    ``AnnotationApp`` の属性 (``self._zoom`` など) をそのまま使う。
    メソッドは移動しただけで中身は変えていない。
    """

    def open_review_mode(self) -> None:
        if not self.review_mode.open(self._queue, fps=self.fps, num_frames=self.num_frames):
            self._status(f"Nothing left to review. {self._queue.summary()}.")
            return
        self._render_review_canvas()
        self._status("Review mode. Y accepts, N rejects, E edits, Esc leaves.")

    def _on_review_decision(self, item, verdict: str) -> None:
        self._render_review_canvas()
        if verdict == "accept":
            self._accept_item_candidates(item)
        elif verdict == "reject":
            self._reject_item_candidates(item)
        self.inbox.refresh()
        self._render_timeline()
        self._mark_dirty()

    def _on_review_edit(self, item) -> None:
        self.jump_to_issue(item)
        self._status("Edit this one, then press Review to carry on.")

    def _on_review_swap(self, item, swapped: bool) -> None:
        if swapped and len(item.track_ids) == 2:
            a, b = item.track_ids
            n = self.swap_tracks(a, b, from_frame=item.end)
            self._status(
                f"Swapped tracks {a} and {b} from frame {item.end} "
                f"({n} boxes). Ctrl+Z takes it back."
            )
        else:
            self._status("Left the identities as they were.")
        self.inbox.refresh()
        self._render_timeline()
        self._mark_dirty()

    def _on_review_exit(self) -> None:
        self._refresh_sidebar()
        self._refresh_inspector()
        self._render_canvas()
        self._render_timeline()

    def swap_tracks(self, a: int, b: int, *, from_frame: int) -> int:
        """Exchange two track ids from ``from_frame`` onward (§15).

        One undo step for the whole sweep, because it is one decision: a user
        who swaps and regrets it must not have to press Ctrl+Z three hundred
        times.
        """
        # Snapshot every frame the sweep will touch, *before* touching any of
        # them, so one Ctrl+Z takes back the whole swap rather than the one
        # frame the user happens to be on.
        touched = [
            frame
            for frame, objects in self.annotations.items()
            if frame >= from_frame and any(o.track_id in (a, b) for o in objects)
        ]
        if not touched:
            return 0
        snapshot = self._snapshot(touched)

        changed = 0
        for frame in touched:
            for obj in self.annotations.get(frame, []):
                if obj.track_id in (a, b):
                    obj.track_id = b if obj.track_id == a else a
                    changed += 1
        if changed:
            self._push_undo(snapshot)
            self._mark_dirty()
            self._refresh_sidebar()
            self._render_canvas()
        return changed

    # ─── Review Mode playback ────────────────────────────────────────

    def _tick_review(self) -> None:
        """Advance the review loop. Called once per rendered frame.

        Seeks rather than caching the loop's frames: a 50-frame loop costs one
        ``VideoCapture`` seek plus a decode per tick, which is single-digit
        milliseconds here, while caching would hold 150 MB for a 1024x1024
        video and still have to be invalidated on every item.
        """
        if not self.review_mode.active or self.cap is None:
            return
        if not self.review_mode.due(time.time(), self.fps):
            return
        nxt = self.review_mode.advance_loop(self.current_frame_idx)
        if nxt != self.current_frame_idx:
            self.goto_frame(nxt, light=True)
        self._render_review_canvas()

    def _render_review_canvas(self) -> None:
        """Paint the current frame and its boxes into Review Mode's drawlist.

        Review Mode owns the layout and the decision; the annotator owns the
        pixels, because the texture, the scale and the box-drawing rules all
        already live here and a second copy would drift from the canvas the
        user corrects things on.
        """
        tag = self.review_mode.canvas_tag
        if not dpg.does_item_exist(tag) or self._current_bgr is None:
            return
        dpg.delete_item(tag, children_only=True)

        try:
            width, height = dpg.get_item_rect_size(tag)
        except (KeyError, SystemError, TypeError):
            width, height = 900, 460
        width = max(int(width), 200)
        height = max(int(height), 160)

        if self.frame_w <= 0 or self.frame_h <= 0:
            return
        scale = min(width / self.frame_w, height / self.frame_h)
        off_x = (width - self.frame_w * scale) / 2
        off_y = (height - self.frame_h * scale) / 2

        if dpg.does_item_exist(self._frame_tex):
            dpg.draw_image(
                self._frame_tex,
                pmin=(off_x, off_y),
                pmax=(off_x + self.frame_w * scale, off_y + self.frame_h * scale),
                parent=tag,
            )

        item = self.review_mode.item
        focus = set(getattr(item, "track_ids", ()) or ())

        def to_canvas(x: float, y: float) -> Tuple[float, float]:
            return x * scale + off_x, y * scale + off_y

        def draw(box_source, dashed: bool) -> None:
            for obj in box_source:
                track_id = int(getattr(obj, "track_id", -1))
                # The tracks the item is about are drawn at full strength;
                # everything else is dimmed, so the eye goes to the question
                # rather than to whichever box happens to be brightest.
                dim = bool(focus) and track_id not in focus
                colour = theme.rgba(theme.track_color(track_id)[:3], 70 if dim else 255)
                corners = obj.corners() if hasattr(obj, "corners") else corners_of(obj)
                points = [to_canvas(px, py) for px, py in corners]
                dpg.draw_polygon(points, color=colour, thickness=1.0 if dim else 2.0, parent=tag)
                if not dim and track_id >= 0:
                    dpg.draw_text(
                        (points[0][0], points[0][1] - 16),
                        f"ID{track_id}",
                        color=colour,
                        size=15,
                        parent=tag,
                    )

        draw(self.annotations.get(self.current_frame_idx, []), dashed=False)
        draw([cand for _index, cand in self.candidates_here()], dashed=True)

        # The playhead against the loop, drawn as a bar rather than a number:
        # during a loop the useful information is "where in the loop am I",
        # and a frame index does not answer that at a glance.
        lo, hi = self.review_mode.loop
        if hi > lo:
            fraction = (self.current_frame_idx - lo) / max(1, hi - lo)
            fraction = max(0.0, min(1.0, fraction))
            dpg.draw_line(
                (8, height - 6),
                (width - 8, height - 6),
                color=theme.rgba(theme.BORDER_SUBTLE),
                thickness=3.0,
                parent=tag,
            )
            dpg.draw_line(
                (8, height - 6),
                (8 + (width - 16) * fraction, height - 6),
                color=theme.rgba(theme.ACCENT),
                thickness=3.0,
                parent=tag,
            )

    # ─── Accepting candidates in bulk (§18) ──────────────────────────
