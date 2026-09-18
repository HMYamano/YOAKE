"""
history.py — 取り消し / やり直し / 保存待ち状態

annotator.py から切り出した編集履歴の責務。1 つの編集を「変更前のフレーム
群のスナップショット」として積み、undo / redo でそれを入れ替える。

=== なぜ 1 か所にまとめるか ===

undo スタック・dirty フラグ・autosave のタイマーは、どれも「編集が起きた」と
いう同じ事実に反応する 3 つの表示で、バラバラに更新すると簡単に食い違う
(保存済みなのに Unsaved のまま、undo したのに dirty が落ちない等)。
入口を _push_undo と _mark_dirty に絞っておくと、その食い違いが
構造的に起きにくくなる。

autosave は debounce する。ボックスを 30 フレームにまたいでドラッグするのは
30 回の変更だが、書き込みは 1 回であるべき (AUTOSAVE_DEBOUNCE 秒の静止後)。
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, Optional

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

try:
    from . import widgets
except ImportError:  # pragma: no cover
    widgets = None

#: Seconds of quiet before an autosave fires. Long enough that dragging a box
#: across thirty frames writes once rather than thirty times, short enough that
#: a crash costs a few seconds of work (GUI_DESIGN P38).
AUTOSAVE_DEBOUNCE = 4.0

#: Undo depth, per frame (docs/GUI_DESIGN.md §0.3.5).
UNDO_DEPTH = 50


class EditHistoryMixin:
    """EditHistoryMixin — annotator.py から切り出したミックスイン。

    ``AnnotationApp`` の属性 (``self._zoom`` など) をそのまま使う。
    メソッドは移動しただけで中身は変えていない。
    """

    def _snapshot(self, frames: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        """Deep-copy the named frames (default: the current one) as one entry."""
        if frames is None:
            frames = [self.current_frame_idx]
        return {
            "video_id": self._video_session.video_id,
            "frame": self.current_frame_idx,
            "review": copy.deepcopy(self._review),
            "intervals": copy.deepcopy(self._intervals),
            "frames": {int(f): copy.deepcopy(self.annotations.get(int(f), [])) for f in frames},
        }

    def _push_undo(
        self, snapshot: Optional[Dict[str, Any]] = None, frames: Optional[Iterable[int]] = None
    ) -> None:
        """Record one undoable step.

        Taking a **new** action clears the redo stack: the future the user could
        have redone is no longer reachable from here, and offering it would
        replay edits onto a different history.
        """
        entry = snapshot if snapshot is not None else self._snapshot(frames)
        self._undo.append(entry)
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]
        self._redo.clear()
        self._refresh_history_ui()

    def _apply_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Restore one entry, returning the entry that undoes *this* restore."""
        if entry.get("video_id", self._video_session.video_id) != self._video_session.video_id:
            raise ValueError("Undo entry belongs to another video")
        inverse = {
            "video_id": self._video_session.video_id,
            "frame": self.current_frame_idx,
            "review": copy.deepcopy(self._review),
            "intervals": copy.deepcopy(self._intervals),
            "frames": {
                int(f): copy.deepcopy(self.annotations.get(int(f), []))
                for f in entry.get("frames", {})
            },
        }
        if "review" in entry:
            self._review = copy.deepcopy(entry["review"])
        if "intervals" in entry:
            self._intervals = copy.deepcopy(entry["intervals"])
        for frame, objects in entry.get("frames", {}).items():
            if objects:
                self.annotations[int(frame)] = copy.deepcopy(objects)
            else:
                self.annotations.pop(int(frame), None)
        return inverse

    def _after_history_move(self, entry: Dict[str, Any], verb: str, remaining: int) -> None:
        # The selection is an object identity and the restored list holds
        # different objects, so keeping it would leave Delete pointed at a box
        # that is no longer in the frame.
        self._selected = None
        target = int(entry.get("frame", self.current_frame_idx))
        if target != self.current_frame_idx and 0 <= target < max(self.num_frames, 1):
            self.goto_frame(target)
        self._mark_dirty()
        self.rebuild_copilot()
        self._refresh_sidebar()
        self._render_canvas()
        self._refresh_history_ui()
        touched = len(entry.get("frames", {}))
        where = f" across {touched} frames" if touched > 1 else ""
        self._status(f"{verb}{where}. {remaining} step" f"{'s' if remaining != 1 else ''} left.")

    def undo(self) -> None:
        if not self._undo:
            self._status("Nothing to undo.")
            return
        entry = self._undo.pop()
        self._redo.append(self._apply_entry(entry))
        self._after_history_move(entry, "Undone", len(self._undo))

    def redo(self) -> None:
        if not self._redo:
            self._status("Nothing to redo.")
            return
        entry = self._redo.pop()
        self._undo.append(self._apply_entry(entry))
        self._after_history_move(entry, "Redone", len(self._redo))

    def _refresh_history_ui(self) -> None:
        """Dim Undo and Redo when there is nothing behind them.

        A permanently-enabled Undo that sometimes does nothing teaches the user
        to distrust it; a dimmed one says so before they press it.
        """
        for tag, stack in (("top_undo", self._undo), ("top_redo", self._redo)):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=bool(stack))

    # ─── Dirty state and autosave ────────────────────────────────────

    def _mark_dirty(self) -> None:
        """Something changed. Schedule an autosave and update the chip."""
        self._video_session.touch()
        self._dirty = True
        self._save_due = time.time() + AUTOSAVE_DEBOUNCE
        self._refresh_save_chip()

    def _refresh_save_chip(self) -> None:
        if not dpg.does_item_exist("top_saved"):
            return
        if self._dirty:
            widgets.status_pill("top_saved", "Unsaved", "pending")
        elif self._last_saved is not None:
            widgets.status_pill("top_saved", "Saved", "ok")
        else:
            widgets.status_pill("top_saved", "No changes", "idle")

    def _tick_autosave(self) -> None:
        """Write at most once per quiet period (§38).

        Debounced rather than per-edit: dragging a box across thirty frames is
        thirty mutations and must be one write. Only fires when there is
        somewhere to write to, because an autosave that opens a file dialog is
        not an autosave.
        """
        self._autosave_workspace()
