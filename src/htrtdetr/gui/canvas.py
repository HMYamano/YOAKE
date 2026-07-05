"""
canvas.py — 共有 FrameCanvas (dearpygui)
========================================

動画フレームを dpg の drawlist にレターボックス表示し、bbox/ID/action オーバーレイを
描く再利用可能ウィジェット。Video Analysis GUI で使用する。座標変換ロジックは
アノテータ (``annotator.py`` の ``_recompute_scale``/``_i2c``/``_c2i``) と同じ考え方で、
既存アノテータの描画は壊さないよう独立実装している。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import dearpygui.dearpygui as dpg

_TRACK_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
    (191, 239, 69), (250, 190, 212), (70, 153, 144), (220, 190, 255),
    (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 216, 177), (0, 0, 117), (169, 169, 169),
]


def track_color(track_id: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    if track_id is None or track_id < 0:
        return (150, 150, 150, alpha)
    r, g, b = _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]
    return (r, g, b, alpha)


class FrameCanvas:
    """RGB フレーム + オーバーレイを描く drawlist ラッパ。"""

    def __init__(self, drawlist_tag: str, texture_tag: str, tex_registry_tag: str) -> None:
        self.drawlist = drawlist_tag
        self.texture = texture_tag
        self.tex_registry = tex_registry_tag
        self.frame_w = 0
        self.frame_h = 0
        self._scale = 1.0
        self._ox = 0.0
        self._oy = 0.0

    # ── テクスチャ管理 ────────────────────────────────────────────────
    def ensure_texture(self, w: int, h: int) -> None:
        """フレームサイズが変わったら raw テクスチャを作り直す。"""
        if w == self.frame_w and h == self.frame_h and dpg.does_item_exist(self.texture):
            return
        self.frame_w, self.frame_h = w, h
        if dpg.does_item_exist(self.texture):
            dpg.delete_item(self.texture)
        blank = np.zeros(w * h * 4, dtype=np.float32)
        if not dpg.does_item_exist(self.tex_registry):
            with dpg.texture_registry(tag=self.tex_registry, show=False):
                pass
        dpg.push_container_stack(self.tex_registry)
        dpg.add_raw_texture(w, h, blank, tag=self.texture,
                            format=dpg.mvFormat_Float_rgba)
        dpg.pop_container_stack()

    def update_frame_data(self, frame_rgb: np.ndarray) -> None:
        """テクスチャに RGB フレームを流し込む。"""
        h, w = frame_rgb.shape[:2]
        self.ensure_texture(w, h)
        rgba = np.dstack([frame_rgb, np.full((h, w), 255, dtype=np.uint8)])
        dpg.set_value(self.texture, (rgba.astype(np.float32) / 255.0).flatten())

    # ── 座標変換 ─────────────────────────────────────────────────────
    def recompute_scale(self) -> None:
        try:
            cw, ch = dpg.get_item_rect_size(self.drawlist)
        except Exception:
            cw, ch = self.frame_w, self.frame_h
        cw = int(cw) if cw and cw > 1 else max(self.frame_w, 1)
        ch = int(ch) if ch and ch > 1 else max(self.frame_h, 1)
        if self.frame_w > 0 and self.frame_h > 0:
            self._scale = min(cw / self.frame_w, ch / self.frame_h)
            self._ox = (cw - self.frame_w * self._scale) / 2
            self._oy = (ch - self.frame_h * self._scale) / 2

    def i2c(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale + self._ox, y * self._scale + self._oy

    # ── 描画 ─────────────────────────────────────────────────────────
    def render(
        self,
        frame_rgb: Optional[np.ndarray],
        detections: Optional[Sequence[dict]] = None,
        placeholder: str = "動画を読み込んでください",
    ) -> None:
        """フレーム画像 + 検出 bbox/ラベルを描画する。

        ``detections``: 各要素 ``{"bbox":[x1,y1,x2,y2], "track_id":int, "action":str,
        "score":float}`` のリスト。
        """
        if not dpg.does_item_exist(self.drawlist):
            return
        dpg.delete_item(self.drawlist, children_only=True)
        self.recompute_scale()

        if frame_rgb is None:
            dpg.draw_text((20, 20), placeholder, size=16,
                          color=(120, 120, 140, 255), parent=self.drawlist)
            return

        if dpg.does_item_exist(self.texture):
            dw = self.frame_w * self._scale
            dh = self.frame_h * self._scale
            dpg.draw_image(self.texture, pmin=(self._ox, self._oy),
                           pmax=(self._ox + dw, self._oy + dh), parent=self.drawlist)

        for det in detections or []:
            self._draw_det(det)

    def _draw_det(self, det: dict) -> None:
        x1, y1, x2, y2 = det["bbox"]
        c1 = self.i2c(x1, y1)
        c2 = self.i2c(x2, y2)
        tid = det.get("track_id", -1)
        col = track_color(tid)
        dpg.draw_rectangle(c1, c2, color=col, fill=track_color(tid, 40),
                           thickness=2.0, parent=self.drawlist)
        parts: List[str] = []
        if tid is not None and tid >= 0:
            parts.append(f"ID{tid}")
        if det.get("action"):
            parts.append(str(det["action"]))
        if det.get("score") is not None:
            parts.append(f"{float(det['score']):.2f}")
        label = " ".join(parts)
        if label:
            dpg.draw_text((c1[0] + 3, c1[1] + 2), label, size=13,
                          color=(0, 0, 0, 200), parent=self.drawlist)
            dpg.draw_text((c1[0] + 2, c1[1] + 1), label, size=13,
                          color=(255, 255, 255, 230), parent=self.drawlist)
