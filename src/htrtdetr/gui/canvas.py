"""
canvas.py — 共有 FrameCanvas (dearpygui)
========================================

動画フレームを dpg の drawlist にレターボックス表示し、box / ID / action オーバーレイを
描く再利用可能ウィジェット。Video Analysis GUI で使用する。座標変換ロジックは
アノテータ (``annotator.py`` の ``_recompute_scale``/``_i2c``/``_c2i``) と同じ考え方。

v2 で追加した点:

* **zoom / pan** — ``fit()`` / ``zoom_at()`` / ``pan_by()``。``recompute_scale`` が
  ``fit_scale * zoom + pan`` を合成するだけなので、``i2c`` / ``c2i`` を通る当たり判定と
  描画は自動的に正しいままになる。
* **逆変換 ``c2i``** — 以前は ``hit_test`` にインライン展開されていて、pan/zoom を足すたびに
  導出し直す必要があった。変換は 1 組だけ、が原則。
* **2 レイヤ構成** — ``layer_image`` / ``layer_overlay``。hover や選択の更新で画像まで
  作り直さない (以前は ``delete_item(children_only=True)`` で毎回全消去していた)。
* **パレット共有** — 色は ``gui/theme.py`` から取る。アノテータと解析ビューで同じ track が
  同じ色になる。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import dearpygui.dearpygui as dpg

from . import theme
from .predictions import obb_corners

# 表示 toggle の既定値。Video Analysis GUI のチェックボックスと 1 対 1 に対応する。
#   個体検出 = tight rotated box (実線・塗りあり)
#   行動認識 = larger axis-aligned context box (破線・塗りなし)
# という視覚的な区別を既定で成立させる。
DEFAULT_TOGGLES: Dict[str, bool] = {
    "bbox": True,          # 軸平行の個体ボックス (obb が無いとき)
    "obb": True,           # 回転した個体ボックス
    "context_box": False,  # 行動認識の context ボックス
    "trajectory": False,   # track 軌跡
    "track_id": True,
    "pose": False,
    "action": True,
    "confidence": True,
    "explanation": False,  # evidence 寄与
}

#: 20 色の track パレット。``theme`` が唯一の定義元 (アノテータと共有)。
TRACK_PALETTE: Tuple[Tuple[int, int, int], ...] = theme.TRACK_PALETTE
#: 旧名の後方互換エイリアス。
_TRACK_PALETTE = list(TRACK_PALETTE)

# ── zoom / pan の定数 (annotator.py と同じ値にそろえる) ──
MIN_ZOOM = 0.2
MAX_ZOOM = 20.0
ZOOM_STEP = 1.25
#: どれだけ pan しても、これだけの canvas px は必ず画面に残す。
MIN_VISIBLE_PX = 64.0

_TEXT_ON_DARK = (240, 243, 248, 255)


def track_color(track_id: Optional[int], alpha: int = 255) -> Tuple[int, int, int, int]:
    """track_id → RGBA。``track_id < 0`` / ``None`` は grey。

    ``theme.track_color`` への薄いラッパ。他モジュールが ``canvas.track_color`` を
    import しているので名前は残す。
    """
    return theme.track_color(track_id, alpha)


def class_color(class_id: Optional[int], alpha: int = 255) -> Tuple[int, int, int, int]:
    """class_id → RGBA (track とは別のパレット位相)。"""
    return theme.class_color(class_id, alpha)


def source_color(source: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """``human`` / ``auto`` / ``track`` / ``interp`` → RGBA。"""
    return theme.source_color(source, alpha)


def _obb_corners(obb: Sequence[float]) -> List[Tuple[float, float]]:
    """[cx, cy, w, h, theta] → 4 頂点 (反時計回り)。

    ``predictions.obb_corners`` と同一実装だったので、そちらへ委譲する薄い別名。
    """
    return obb_corners(list(obb))


def _text_width(text: str, size: float) -> float:
    """描画前にラベル chip の幅を見積もる (dpg は draw_text の実寸を返さない)。"""
    return len(text) * size * 0.54 + 12.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


class FrameCanvas:
    """RGB フレーム + オーバーレイを描く drawlist ラッパ。"""

    def __init__(self, drawlist_tag: str, texture_tag: str, tex_registry_tag: str) -> None:
        self.drawlist = drawlist_tag
        self.texture = texture_tag
        self.tex_registry = tex_registry_tag
        self.layer_image = f"{drawlist_tag}__layer_image"
        self.layer_overlay = f"{drawlist_tag}__layer_overlay"

        self.frame_w = 0
        self.frame_h = 0

        self._scale = 1.0
        self._fit_scale = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._canvas_w = 0.0
        self._canvas_h = 0.0

        # 直近の render 引数。overlay だけ描き直すときに使う。
        self._has_frame = False
        self._placeholder = ""
        self._detections: List[dict] = []
        self._toggles: Dict[str, bool] = dict(DEFAULT_TOGGLES)
        self._trajectories: Optional[Dict[int, Sequence[Tuple[int, float, float]]]] = None
        self._selected_track: Optional[int] = None
        self._selected_index: Optional[int] = None
        self._hovered_index: Optional[int] = None
        self._image_signature: Optional[tuple] = None

    # ══════════════════════════════════════════════════════════════════
    # テクスチャ管理
    # ══════════════════════════════════════════════════════════════════
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
        # テクスチャが差し替わったら画像レイヤは必ず描き直す
        self._image_signature = None

    def update_frame_data(self, frame_rgb: np.ndarray) -> None:
        """テクスチャに RGB フレームを流し込む。"""
        h, w = frame_rgb.shape[:2]
        self.ensure_texture(w, h)
        rgba = np.dstack([frame_rgb, np.full((h, w), 255, dtype=np.uint8)])
        dpg.set_value(self.texture, (rgba.astype(np.float32) / 255.0).flatten())

    # ══════════════════════════════════════════════════════════════════
    # 座標変換 (これだけが唯一の変換組)
    # ══════════════════════════════════════════════════════════════════
    def canvas_size(self) -> Tuple[float, float]:
        """drawlist の実サイズ。取得できないときはフレームサイズで代用する。"""
        try:
            cw, ch = dpg.get_item_rect_size(self.drawlist)
        except (KeyError, SystemError, TypeError, ValueError):
            cw, ch = self.frame_w, self.frame_h
        cw = float(cw) if cw and cw > 1 else float(max(self.frame_w, 1))
        ch = float(ch) if ch and ch > 1 else float(max(self.frame_h, 1))
        return cw, ch

    def recompute_scale(self) -> None:
        """``fit_scale * zoom`` + pan を合成する。

        ``i2c`` / ``c2i`` はこの結果だけを読むので、当たり判定・ハンドル位置・描画は
        zoom/pan を足しても自動的に一致する。
        """
        cw, ch = self.canvas_size()
        self._canvas_w, self._canvas_h = cw, ch
        if self.frame_w <= 0 or self.frame_h <= 0:
            self._fit_scale = 1.0
            self._scale = 1.0
            self._ox = self._oy = 0.0
            return
        self._fit_scale = min(cw / self.frame_w, ch / self.frame_h)
        self._zoom = _clamp(self._zoom, MIN_ZOOM, MAX_ZOOM)
        self._scale = self._fit_scale * self._zoom
        self._clamp_pan(cw, ch)
        self._ox = (cw - self.frame_w * self._scale) / 2 + self._pan_x
        self._oy = (ch - self.frame_h * self._scale) / 2 + self._pan_y

    def _clamp_pan(self, cw: float, ch: float) -> None:
        """画像を画面外へ完全に逃がせないようにする (最低 64 px は残す)。"""
        dw = self.frame_w * self._scale
        dh = self.frame_h * self._scale
        for axis, span, canvas_span in (("x", dw, cw), ("y", dh, ch)):
            keep = max(1.0, min(MIN_VISIBLE_PX, span, canvas_span))
            base = (canvas_span - span) / 2
            pan = self._pan_x if axis == "x" else self._pan_y
            origin = _clamp(base + pan, keep - span, canvas_span - keep)
            if axis == "x":
                self._pan_x = origin - base
            else:
                self._pan_y = origin - base

    def i2c(self, x: float, y: float) -> Tuple[float, float]:
        """画像 px → canvas px。"""
        return x * self._scale + self._ox, y * self._scale + self._oy

    def c2i(self, cx: float, cy: float) -> Tuple[float, float]:
        """canvas px → 画像 px。``i2c`` の厳密な逆写像。"""
        if self._scale <= 0:
            return (0.0, 0.0)
        return (cx - self._ox) / self._scale, (cy - self._oy) / self._scale

    # ── zoom / pan ────────────────────────────────────────────────────
    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def scale(self) -> float:
        return self._scale

    def zoom_percent(self) -> int:
        """UI に出す表示倍率 (fit = 100%)。"""
        return int(round(self._zoom * 100))

    def fit(self) -> None:
        """倍率 100% (= ウィンドウにフィット) に戻し、pan も解除する。"""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.recompute_scale()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = _clamp(float(zoom), MIN_ZOOM, MAX_ZOOM)
        self.recompute_scale()

    def zoom_at(self, cx: float, cy: float, notches: float = 1.0) -> None:
        """カーソル位置 ``(cx, cy)`` の画像点を固定したまま拡大縮小する。"""
        if self.frame_w <= 0 or self.frame_h <= 0:
            return
        ix, iy = self.c2i(cx, cy)
        self._zoom = _clamp(self._zoom * (ZOOM_STEP ** notches), MIN_ZOOM, MAX_ZOOM)
        self.recompute_scale()
        nx, ny = self.i2c(ix, iy)
        self._pan_x += cx - nx
        self._pan_y += cy - ny
        self.recompute_scale()

    def zoom_step(self, notches: float = 1.0) -> None:
        """canvas 中央を固定して拡大縮小する (キーボードの + / -)。"""
        self.zoom_at(self._canvas_w / 2, self._canvas_h / 2, notches)

    def pan_by(self, dx: float, dy: float) -> None:
        self._pan_x += float(dx)
        self._pan_y += float(dy)
        self.recompute_scale()

    def set_pan(self, px: float, py: float) -> None:
        self._pan_x = float(px)
        self._pan_y = float(py)
        self.recompute_scale()

    @property
    def pan(self) -> Tuple[float, float]:
        return (self._pan_x, self._pan_y)

    # ── マウス (PITFALL: hover も rect_min も drawlist から取る) ───────
    def is_hovered(self) -> bool:
        """canvas 上にカーソルがあるか。

        **child_window ではなく drawlist で判定する。** child_window の
        ``is_item_hovered`` はマウスボタンを押している間 False を返すので、
        ドラッグ中のガードに使うと必ず壊れる (測定済み)。
        """
        if not dpg.does_item_exist(self.drawlist):
            return False
        try:
            return bool(dpg.is_item_hovered(self.drawlist))
        except (KeyError, SystemError, TypeError):
            return False

    def origin(self) -> Optional[Tuple[float, float]]:
        """viewport 座標系での canvas 原点。取得できないときは ``None``。

        **フォールバックで ``(0, 0)`` を返してはいけない。** それは canvas 原点ぶん
        (メニューバーありで実測 8 px / 27 px) のズレを黙って生み、全クリックが
        別の場所に当たる。
        """
        if not dpg.does_item_exist(self.drawlist):
            return None
        try:
            ox, oy = dpg.get_item_rect_min(self.drawlist)
        except (KeyError, SystemError, TypeError, ValueError):
            return None
        return float(ox), float(oy)

    def mouse_canvas(self) -> Optional[Tuple[float, float]]:
        """現在のマウス位置を canvas ローカル座標で返す。"""
        origin = self.origin()
        if origin is None:
            return None
        mx, my = dpg.get_mouse_pos(local=False)
        return mx - origin[0], my - origin[1]

    def mouse_image(self) -> Optional[Tuple[float, float]]:
        """現在のマウス位置を画像 px で返す。"""
        pos = self.mouse_canvas()
        if pos is None:
            return None
        return self.c2i(*pos)

    # ══════════════════════════════════════════════════════════════════
    # レイヤ
    # ══════════════════════════════════════════════════════════════════
    def _ensure_layers(self) -> bool:
        if not dpg.does_item_exist(self.drawlist):
            return False
        if not dpg.does_item_exist(self.layer_image):
            dpg.add_draw_layer(tag=self.layer_image, parent=self.drawlist)
            self._image_signature = None
        if not dpg.does_item_exist(self.layer_overlay):
            dpg.add_draw_layer(tag=self.layer_overlay, parent=self.drawlist)
        return True

    # ══════════════════════════════════════════════════════════════════
    # 描画
    # ══════════════════════════════════════════════════════════════════
    def render(
        self,
        frame_rgb: Optional[np.ndarray],
        detections: Optional[Sequence[dict]] = None,
        placeholder: str = "Please load a video",
        toggles: Optional[Dict[str, bool]] = None,
        trajectories: Optional[Dict[int, Sequence[Tuple[int, float, float]]]] = None,
        selected_track: Optional[int] = None,
        *,
        selected_index: Optional[int] = None,
        hovered_index: Optional[int] = None,
    ) -> None:
        """フレーム画像 + 検出オーバーレイを描画する。

        Args:
            detections: 各要素 ``{"bbox":[x1,y1,x2,y2], "track_id":int, "action":str,
                "score":float, "obb":[cx,cy,w,h,theta], "context_box":[x1,y1,x2,y2],
                "pose":[[x,y,v],...], "source":"human|auto|track|interp"}``
            toggles: 表示する要素 (``DEFAULT_TOGGLES`` のキー)。部分指定でよい。
            trajectories: ``{track_id: [(frame, cx, cy), ...]}``
            selected_track: 強調表示する track_id
            selected_index: 強調表示する detections の index (track より優先)
            hovered_index: hover 中の detections の index
        """
        if not self._ensure_layers():
            return

        self._has_frame = frame_rgb is not None
        self._placeholder = placeholder
        self._detections = list(detections or [])
        tg = dict(DEFAULT_TOGGLES)
        if toggles:
            tg.update(toggles)
        self._toggles = tg
        self._trajectories = trajectories
        self._selected_track = selected_track
        self._selected_index = selected_index
        self._hovered_index = hovered_index

        self.recompute_scale()
        self._sync_image_layer()
        self._draw_overlay_layer()

    def refresh_overlay(self, *, hovered_index: Optional[int] = ...,
                        selected_index: Optional[int] = ...) -> None:
        """直近の render 引数のまま overlay だけ描き直す。

        hover / 選択の更新は 1 フレームに何度も走るので、画像レイヤを作り直さない
        この経路を通す。``...`` を渡すと値を据え置く。
        """
        if not self._ensure_layers():
            return
        if hovered_index is not ...:
            self._hovered_index = hovered_index
        if selected_index is not ...:
            self._selected_index = selected_index
        self.recompute_scale()
        self._sync_image_layer()
        self._draw_overlay_layer()

    def clear(self) -> None:
        """両レイヤを空にする (動画を閉じたとき)。"""
        if not self._ensure_layers():
            return
        dpg.delete_item(self.layer_image, children_only=True)
        dpg.delete_item(self.layer_overlay, children_only=True)
        self._has_frame = False
        self._detections = []
        self._image_signature = None

    # ── 画像レイヤ ────────────────────────────────────────────────────
    def _sync_image_layer(self) -> None:
        signature = (
            self._has_frame, self.frame_w, self.frame_h,
            round(self._ox, 2), round(self._oy, 2), round(self._scale, 5),
            self._placeholder,
        )
        if signature == self._image_signature:
            return
        self._image_signature = signature
        dpg.delete_item(self.layer_image, children_only=True)

        if not self._has_frame:
            if self._placeholder:
                dpg.draw_text((18, 18), self._placeholder, size=15,
                              color=theme.rgba(theme.TEXT_DIM),
                              parent=self.layer_image)
            return

        if not dpg.does_item_exist(self.texture):
            return
        dw = self.frame_w * self._scale
        dh = self.frame_h * self._scale
        pmin = (self._ox, self._oy)
        pmax = (self._ox + dw, self._oy + dh)
        dpg.draw_image(self.texture, pmin=pmin, pmax=pmax, parent=self.layer_image)
        # letterbox の境界が判るように 1 px の枠を置く
        dpg.draw_rectangle(pmin, pmax, color=theme.rgba(theme.BORDER_SUBTLE, 150),
                           thickness=1.0, parent=self.layer_image)

    # ── オーバーレイレイヤ ────────────────────────────────────────────
    def _draw_overlay_layer(self) -> None:
        dpg.delete_item(self.layer_overlay, children_only=True)
        if not self._has_frame:
            return

        tg = self._toggles
        # 軌跡は箱の下に描く (箱を隠さないため)
        if tg.get("trajectory") and self._trajectories:
            for tid, pts in self._trajectories.items():
                self._draw_trajectory(int(tid), pts)

        sel = self._selected_indices()
        # 非選択 → hover → 選択 の順に描くので、強調したものが必ず最前面に来る
        order = [i for i in range(len(self._detections))
                 if i not in sel and i != self._hovered_index]
        if self._hovered_index is not None and self._hovered_index not in sel:
            order.append(self._hovered_index)
        order.extend(sorted(sel))
        for i in order:
            if 0 <= i < len(self._detections):
                self._draw_det(self._detections[i], tg,
                               selected=i in sel,
                               hovered=i == self._hovered_index)

    def _selected_indices(self) -> set:
        """選択中の detection index 集合 (index 指定が track 指定より優先)。"""
        if self._selected_index is not None:
            return {int(self._selected_index)}
        if self._selected_track is None:
            return set()
        return {
            i for i, d in enumerate(self._detections)
            if d.get("track_id") == self._selected_track
        }

    def _draw_trajectory(self, tid: int, pts: Sequence[Tuple[int, float, float]]) -> None:
        """track の中心軌跡を折れ線で描く (新しいほど濃く)。"""
        if len(pts) < 2:
            return
        n = len(pts)
        for i in range(1, n):
            # 末尾ほど不透明にして進行方向が判るようにする
            alpha = int(40 + 180 * (i / n))
            a = self.i2c(pts[i - 1][1], pts[i - 1][2])
            b = self.i2c(pts[i][1], pts[i][2])
            dpg.draw_line(a, b, color=track_color(tid, alpha),
                          thickness=1.5, parent=self.layer_overlay)
        head = self.i2c(pts[-1][1], pts[-1][2])
        dpg.draw_circle(head, 2.5, color=track_color(tid, 230),
                        fill=track_color(tid, 230), parent=self.layer_overlay)

    def _det_style(self, det: dict, selected: bool, hovered: bool):
        """provenance と選択状態から (線色, 塗り色, 太さ, 破線か) を決める。

        source は annotator と同じ語彙: ``human`` (実線・track 色) /
        ``auto`` (破線・PENDING アンバー = 未確認の候補) / ``track`` /
        ``interp`` (細い破線・INTERPOLATED)。
        """
        tid = det.get("track_id", -1)
        source = str(det.get("source") or "human")
        accepted = bool(det.get("accepted", True))

        if source in ("auto", "track") and not accepted:
            line = theme.rgba(theme.PENDING)
            fill_rgb = theme.PENDING
            dashed = True
        elif source == "interp":
            line = theme.rgba(theme.INTERPOLATED)
            fill_rgb = theme.INTERPOLATED
            dashed = True
        else:
            line = track_color(tid)
            fill_rgb = line[:3]
            dashed = False

        thick = 2.0
        fill_a = 40
        if hovered:
            thick += 1.0
            fill_a = 70
        if selected:
            thick += 1.5
            fill_a = 90
        fill = (fill_rgb[0], fill_rgb[1], fill_rgb[2], fill_a)
        return line, fill, thick, dashed

    def _draw_det(
        self,
        det: dict,
        toggles: Dict[str, bool],
        *,
        selected: bool = False,
        hovered: bool = False,
    ) -> None:
        tid = det.get("track_id", -1)
        line, fill, thick, dashed = self._det_style(det, selected, hovered)
        obb = det.get("obb")
        bbox = det.get("bbox")

        # --- 行動認識の context box (広い軸平行) ---
        # 個体検出の箱とは見た目を明確に変える: 破線・細線・塗りつぶし無し。
        ctx = det.get("context_box")
        if toggles.get("context_box") and ctx:
            cx1, cy1, cx2, cy2 = (float(v) for v in ctx[:4])
            self._draw_dashed_rect(
                self.i2c(cx1, cy1), self.i2c(cx2, cy2),
                track_color(tid, 120), thickness=1.0,
            )

        # --- 個体検出の箱 ---
        anchor: Optional[Tuple[float, float]] = None
        corners: List[Tuple[float, float]] = []
        if toggles.get("obb") and obb is not None:
            corners = [self.i2c(px, py) for px, py in obb_corners(list(obb))]
            if dashed:
                self._draw_dashed_polygon(corners, line, thickness=thick)
            else:
                dpg.draw_polygon(corners, color=line, fill=fill,
                                 thickness=thick, parent=self.layer_overlay)
            # 体軸 (向き) を示す線
            ctr = self.i2c(float(obb[0]), float(obb[1]))
            front = ((corners[1][0] + corners[2][0]) / 2,
                     (corners[1][1] + corners[2][1]) / 2)
            dpg.draw_line(ctr, front, color=line, thickness=thick,
                          parent=self.layer_overlay)
            anchor = min(corners, key=lambda p: p[1])
        elif toggles.get("bbox", True) and bbox is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            c1 = self.i2c(x1, y1)
            c2 = self.i2c(x2, y2)
            corners = [c1, (c2[0], c1[1]), c2, (c1[0], c2[1])]
            if dashed:
                self._draw_dashed_rect(c1, c2, line, thickness=thick)
            else:
                dpg.draw_rectangle(c1, c2, color=line, fill=fill,
                                   thickness=thick, parent=self.layer_overlay)
            anchor = c1
        elif bbox is not None:
            x1, y1 = float(bbox[0]), float(bbox[1])
            anchor = self.i2c(x1, y1)
        elif obb is not None:
            corners = [self.i2c(px, py) for px, py in obb_corners(list(obb))]
            anchor = min(corners, key=lambda p: p[1])

        if anchor is None:
            return

        # --- 選択ハンドル (選択中だけ、角に小さな四角) ---
        if selected and corners:
            for px, py in corners:
                dpg.draw_rectangle((px - 3.5, py - 3.5), (px + 3.5, py + 3.5),
                                   color=(255, 255, 255, 235),
                                   fill=theme.rgba(theme.DANGER),
                                   thickness=1.0, parent=self.layer_overlay)

        # --- pose キーポイント ---
        if toggles.get("pose") and det.get("pose"):
            for kp in det["pose"]:
                if len(kp) >= 3 and float(kp[2]) <= 0:
                    continue
                dpg.draw_circle(self.i2c(float(kp[0]), float(kp[1])), 3,
                                color=(255, 255, 255, 220),
                                fill=line, parent=self.layer_overlay)

        # --- ラベル chip ---
        parts: List[str] = []
        if toggles.get("track_id", True) and tid is not None and int(tid) >= 0:
            parts.append(f"T{int(tid)}")
        if toggles.get("action", True) and det.get("action"):
            parts.append(str(det["action"]))
        if toggles.get("confidence", True) and det.get("score") is not None:
            parts.append(f"{float(det['score']):.2f}")
        if det.get("source") and str(det["source"]) != "human":
            parts.append(str(det["source"]))
        label = "  ".join(parts)
        if label:
            self._draw_label_chip(anchor[0], anchor[1] - 19, label, line,
                                  emphasise=selected)

        # --- 判断根拠 (上位 3 group を箱の下に) ---
        # 箱の中に重ねると動物そのものが読めなくなるので、必ず下辺の外に出す。
        if toggles.get("explanation") and det.get("evidence"):
            base_y = (max(p[1] for p in corners) + 4) if corners else (anchor[1] + 4)
            base_x = (min(p[0] for p in corners)) if corners else anchor[0]
            ev = sorted(det["evidence"].items(), key=lambda kv: -abs(kv[1]))[:3]
            for i, (group, val) in enumerate(ev):
                text = f"{group[:9]} {val:+.2f}"
                col = theme.rgba(theme.PENDING) if val >= 0 else theme.rgba(theme.ACCENT)
                self._draw_label_chip(base_x, base_y + i * 17, text, col, size=11)

    def _draw_label_chip(self, x: float, y: float, text: str,
                         color: Sequence[int], *, size: float = 13,
                         emphasise: bool = False) -> None:
        """明るいフレームの上でも読めるラベル。

        半透明の暗い板 + 左端 3 px の色帯 + 明るい文字。文字に薄い影を付けるだけの
        従来方式は、白い背景 (明視野顕微鏡など) の上で読めなくなっていた。
        """
        w = _text_width(text, size)
        h = size + 7
        x = max(0.0, x)
        y = max(0.0, y)
        dpg.draw_rectangle((x, y), (x + w, y + h),
                           color=(0, 0, 0, 0),
                           fill=(12, 14, 18, 225 if emphasise else 200),
                           rounding=3.0, parent=self.layer_overlay)
        dpg.draw_rectangle((x, y), (x + 3, y + h),
                           color=tuple(color), fill=tuple(color),
                           rounding=1.0, parent=self.layer_overlay)
        dpg.draw_text((x + 7, y + 3), text, size=size,
                      color=_TEXT_ON_DARK, parent=self.layer_overlay)

    def _draw_dashed_rect(self, c1, c2, color, thickness: float = 1.0,
                          dash: float = 6.0, gap: float = 4.0) -> None:
        """破線の矩形 (dpg に破線 API が無いため線分で近似)。"""
        x1, y1 = c1
        x2, y2 = c2
        self._draw_dashed_polygon(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            color, thickness=thickness, dash=dash, gap=gap,
        )

    def _draw_dashed_polygon(self, pts: Sequence[Tuple[float, float]], color,
                             thickness: float = 1.0, dash: float = 6.0,
                             gap: float = 4.0) -> None:
        """任意の閉多角形を破線で描く (OBB の候補表示に使う)。"""
        n = len(pts)
        if n < 2:
            return
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            length = math.hypot(bx - ax, by - ay)
            if length < 1.0:
                continue
            ux, uy = (bx - ax) / length, (by - ay) / length
            d = 0.0
            while d < length:
                e = min(d + dash, length)
                dpg.draw_line((ax + ux * d, ay + uy * d),
                              (ax + ux * e, ay + uy * e),
                              color=color, thickness=thickness,
                              parent=self.layer_overlay)
                d += dash + gap

    # ══════════════════════════════════════════════════════════════════
    # 当たり判定
    # ══════════════════════════════════════════════════════════════════
    def hit_test(self, canvas_x: float, canvas_y: float,
                 detections: Optional[Sequence[dict]] = None,
                 *, pad_px: float = 3.0) -> Optional[int]:
        """
        canvas 座標にある検出の index を返す (個体クリック → 詳細表示用)。

        OBB があれば回転矩形の内部判定、無ければ軸平行判定。最前面 (後に描いた方) を優先。
        ``pad_px`` は canvas px の当たり判定の余白 (小さい箱でも掴めるように)。
        ``detections`` を省略すると直近の render に渡されたリストを使う。
        """
        if self._scale <= 0:
            return None
        dets = list(detections) if detections is not None else self._detections
        ix, iy = self.c2i(canvas_x, canvas_y)
        pad = pad_px / self._scale if self._scale > 0 else 0.0
        for i in range(len(dets) - 1, -1, -1):
            det = dets[i]
            obb = det.get("obb")
            if obb is not None and self._toggles.get("obb", True):
                cx, cy, w, h, th = (float(v) for v in obb[:5])
                c, s = math.cos(-th), math.sin(-th)
                dx, dy = ix - cx, iy - cy
                if (abs(dx * c - dy * s) <= w / 2 + pad
                        and abs(dx * s + dy * c) <= h / 2 + pad):
                    return i
                continue
            bbox = det.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            if x1 - pad <= ix <= x2 + pad and y1 - pad <= iy <= y2 + pad:
                return i
        return None

    def hit_test_mouse(self, detections: Optional[Sequence[dict]] = None) -> Optional[int]:
        """現在のマウス位置で :meth:`hit_test` する。canvas 外なら ``None``。"""
        pos = self.mouse_canvas()
        if pos is None:
            return None
        return self.hit_test(pos[0], pos[1], detections)


# ══════════════════════════════════════════════════════════════════════
# 画像への焼き込み (PNG 書き出し用、dpg 非依存)
# ══════════════════════════════════════════════════════════════════════
def draw_overlays_bgr(
    frame_bgr: "np.ndarray",
    detections: Sequence[dict],
    toggles: Optional[Dict[str, bool]] = None,
    trajectories: Optional[Dict[int, Sequence[Tuple[int, float, float]]]] = None,
    selected_track: Optional[int] = None,
) -> "np.ndarray":
    """BGR フレームにオーバーレイを焼き込んだコピーを返す (OpenCV, フル解像度)。

    画面キャプチャではなく元解像度で書き出したいので、canvas の描画とは別に
    cv2 で描き直す。色と語彙は canvas 側とそろえてあるので、保存した PNG は
    画面で見たものと同じ意味になる。
    """
    import cv2

    out = frame_bgr.copy()
    tg = dict(DEFAULT_TOGGLES)
    if toggles:
        tg.update(toggles)

    def _bgr(rgba: Sequence[int]) -> Tuple[int, int, int]:
        return (int(rgba[2]), int(rgba[1]), int(rgba[0]))

    if tg.get("trajectory") and trajectories:
        for tid, pts in trajectories.items():
            col = _bgr(track_color(int(tid)))
            for i in range(1, len(pts)):
                cv2.line(out,
                         (int(pts[i - 1][1]), int(pts[i - 1][2])),
                         (int(pts[i][1]), int(pts[i][2])), col, 1, cv2.LINE_AA)

    for det in detections:
        tid = det.get("track_id", -1)
        source = str(det.get("source") or "human")
        if source in ("auto", "track") and not det.get("accepted", True):
            col = _bgr(theme.rgba(theme.PENDING))
        elif source == "interp":
            col = _bgr(theme.rgba(theme.INTERPOLATED))
        else:
            col = _bgr(track_color(tid))
        thick = 3 if (selected_track is not None and tid == selected_track) else 2

        anchor = None
        obb = det.get("obb")
        bbox = det.get("bbox")
        if tg.get("context_box") and det.get("context_box"):
            cx1, cy1, cx2, cy2 = (int(float(v)) for v in det["context_box"][:4])
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), col, 1, cv2.LINE_AA)
        if tg.get("obb") and obb is not None:
            pts = np.array([[int(px), int(py)] for px, py in obb_corners(list(obb))],
                           dtype=np.int32)
            cv2.polylines(out, [pts], True, col, thick, cv2.LINE_AA)
            anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))
        elif tg.get("bbox", True) and bbox is not None:
            x1, y1, x2, y2 = (int(float(v)) for v in bbox[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
            anchor = (x1, y1)
        elif bbox is not None:
            anchor = (int(float(bbox[0])), int(float(bbox[1])))
        if anchor is None:
            continue

        if tg.get("pose") and det.get("pose"):
            for kp in det["pose"]:
                if len(kp) >= 3 and float(kp[2]) <= 0:
                    continue
                cv2.circle(out, (int(float(kp[0])), int(float(kp[1]))), 3, col, -1,
                           cv2.LINE_AA)

        parts: List[str] = []
        if tg.get("track_id", True) and tid is not None and int(tid) >= 0:
            parts.append(f"T{int(tid)}")
        if tg.get("action", True) and det.get("action"):
            parts.append(str(det["action"]))
        if tg.get("confidence", True) and det.get("score") is not None:
            parts.append(f"{float(det['score']):.2f}")
        label = "  ".join(parts)
        if not label:
            continue
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        ax, ay = anchor[0], max(th + 6, anchor[1])
        cv2.rectangle(out, (ax, ay - th - 7), (ax + tw + 10, ay), (18, 14, 12), -1)
        cv2.rectangle(out, (ax, ay - th - 7), (ax + 3, ay), col, -1)
        cv2.putText(out, label, (ax + 6, ay - 4), font, fs, (248, 243, 240), ft,
                    cv2.LINE_AA)
    return out
