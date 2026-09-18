"""
video_controller.py — 動画を開く / フレームへ移動する

annotator.py から切り出した「今どのフレームを見ているか」の責務。
capture の差し替え、フレームテクスチャの確保、指定フレームへの移動を持つ。

デコード位置そのものの管理は gui.video_source (annotator と解析ビューアが
共有する層) にある。ここはその上で、**dearpygui のテクスチャと UI をフレームに
追従させる**ところだけを担当する。

=== テクスチャを作り直す順番 ===

2 本目の動画を開くとき、古いテクスチャを先に消してはいけない。drawlist がまだ
draw_image でそれを掴んでいると、bind されたテクスチャの削除で dearpygui が
SystemError を投げ、アノテータごと落ちる。先に描画アイテムを消して参照を
手放してから、テクスチャを作り直す。
"""

from __future__ import annotations

from typing import Optional

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import numpy as np

try:
    from . import file_dialogs, video_source
except ImportError:  # pragma: no cover
    from htrtdetr.gui import file_dialogs, video_source

#: dearpygui のテクスチャタグ。動画を開き直すたびに作り直すので、
#: 連番を足した別タグへ逃がせるよう接頭辞として持つ。
FRAME_TEX = "yoake_frame_tex"

#: これ以下の前方ジャンプなら seek せず grab で読み飛ばす。
#: 25 fps で約 0.5 秒 — Right キーの押しっぱなしとレビューの小さな跳躍を
#: カバーし、本当のジャンプには seek を使わせる長さ。
SEQ_SKIP_MAX = 12

class VideoControllerMixin:
    """VideoControllerMixin — annotator.py から切り出したミックスイン。

    ``AnnotationApp`` の属性 (``self._zoom`` など) をそのまま使う。
    メソッドは移動しただけで中身は変えていない。
    """

    def _open_video_dialog(self) -> None:
        path = file_dialogs.open_file(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path is not None:
            self._load_video(path)

    def _install_video_capture(self, cap) -> None:
        self.cap = cap
        self._decoded_idx = None
        path = self.video_path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create frame texture.
        #
        # The canvases are cleared **first**. A drawlist that still holds a
        # draw_image of this texture keeps it bound, and deleting a bound
        # texture raises SystemError out of dearpygui - so opening a second
        # video used to take the annotator down. Clearing the draw items
        # releases the reference; the next render puts them back.
        for canvas in ("canvas", self.review_mode.canvas_tag):
            if dpg.does_item_exist(canvas):
                dpg.delete_item(canvas, children_only=True)
        self._current_bgr = None
        if dpg.does_item_exist(self._frame_tex):
            try:
                dpg.delete_item(self._frame_tex)
            except SystemError:
                # Still bound by something we do not own. The comment here used
                # to promise a fresh tag and the code did nothing, so the old
                # texture survived at the **old video's** size and the next
                # set_value pushed a differently-shaped buffer into it. Take
                # the fresh tag for real; a leaked texture costs memory, a
                # mismatched one costs the process.
                pass
        if dpg.does_item_exist(self._frame_tex):
            self._frame_tex = f"{FRAME_TEX}_{self._tex_serial}"
            self._tex_serial += 1
        blank = np.zeros(self.frame_w * self.frame_h * 4, dtype=np.float32)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                dpg.add_raw_texture(
                    self.frame_w,
                    self.frame_h,
                    blank,
                    tag=self._frame_tex,
                    format=dpg.mvFormat_Float_rgba,
                )
        elif not dpg.does_item_exist(self._frame_tex):
            dpg.push_container_stack("tex_reg")
            dpg.add_raw_texture(
                self.frame_w,
                self.frame_h,
                blank,
                tag=self._frame_tex,
                format=dpg.mvFormat_Float_rgba,
            )
            dpg.pop_container_stack()

        if dpg.does_item_exist("frame_input"):
            dpg.configure_item("frame_input", max_value=self.num_frames - 1)
        if dpg.does_item_exist("total_frames_text"):
            dpg.set_value("total_frames_text", f"/ {self.num_frames - 1}")

        dpg.set_viewport_title(f"YOAKE Annotation Tool — {path.name}")
        self.goto_frame(self.current_frame_idx, light=True)
        self._status(
            f"Loaded: {path.name}  ({self.num_frames} frames, "
            f"{self.fps:.1f} fps, {self.frame_w}×{self.frame_h})"
        )

    def _read_frame_bgr(self, idx: int) -> Optional[np.ndarray]:
        """Decode frame ``idx``, sequentially whenever that is possible (P15).

        The decode-position bookkeeping itself lives in
        :mod:`htrtdetr.gui.video_source`, because the analysis viewer needs the
        same thing and two copies of it drift apart. This method only owns the
        state: ``self.cap`` is the capture and ``self._decoded_idx`` is where it
        actually sits, so a failed read resets the position to "unknown" and the
        next access seeks rather than trusting a stale count.
        """
        frame, self._decoded_idx = video_source.read_frame_sequential(
            self.cap, idx, self._decoded_idx, seq_skip_max=SEQ_SKIP_MAX
        )
        return frame

    # ═══════════════════════════════════════════════════════════════════
    # Frame Navigation
    # ═══════════════════════════════════════════════════════════════════

    def goto_frame(self, idx: int, *, light: bool = False) -> None:
        """フレーム idx へ移動する。

        light=True はシークバーのドラッグ中に使う軽量パス。自動トラッキングと
        サイドバー更新を省き、フレーム描画とシークバーだけを更新する。
        """
        if self.cap is None:
            return
        if self._advance_at_end(idx):
            return
        idx = max(0, min(idx, self.num_frames - 1))

        # Auto-track: run trackers when advancing one frame. The frame it
        # decodes is kept rather than decoded a second time below - with
        # auto-track on, every step used to pay for the same frame twice.
        frame_bgr = None
        if self._auto_track and not light and idx == self.current_frame_idx + 1 and self._trackers:
            frame_bgr = self._read_frame_bgr(idx)
            if frame_bgr is not None:
                self._run_trackers_for_frame(idx, frame_bgr)

        self.current_frame_idx = idx
        self._visit_frame(idx)
        if frame_bgr is None:
            frame_bgr = self._read_frame_bgr(idx)
        self._current_bgr = frame_bgr

        if frame_bgr is not None and dpg.does_item_exist(self._frame_tex):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
            data = (rgba.astype(np.float32) / 255.0).flatten()
            dpg.set_value(self._frame_tex, data)

        if light:
            self._selected = None
        else:
            self.deselect(render=False)
        if dpg.does_item_exist("frame_input"):
            dpg.set_value("frame_input", idx)

        self._render_canvas()
        if not light:
            self._refresh_sidebar()
        self._render_timeline()

        # SAM を使っているなら、この新しいフレームを先に encode させておく。
        # ユーザが個体を探している間に終わらせるのが狙い。
        self._on_frame_changed_for_sam()

        annotated = len(self.annotations)
        pct = annotated / max(self.num_frames, 1)
        self._status(
            f"Frame {idx}/{self.num_frames - 1}  |  "
            f"Annotated: {annotated}/{self.num_frames} ({pct * 100:.0f}%)"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Selection
    # ═══════════════════════════════════════════════════════════════════
