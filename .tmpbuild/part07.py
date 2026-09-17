
    # ═══════════════════════════════════════════════════════════════════
    # Canvas Rendering
    # ═══════════════════════════════════════════════════════════════════

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
            fit = min(self._canvas_w / self.frame_w,
                      self._canvas_h / self.frame_h)
            self._fit_scale = fit
            self._scale = fit * self._zoom
            self._offset_x = ((self._canvas_w - self.frame_w * self._scale) / 2
                              + self._pan_x)
            self._offset_y = ((self._canvas_h - self.frame_h * self._scale) / 2
                              + self._pan_y)

    # _i2c / _c2i は触らない。当たり判定・ハンドル位置・描画がすべてここを
    # 通るので、ズーム/パンを _recompute_scale に閉じ込めれば全部が自動で正しくなる。
    def _i2c(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale + self._offset_x, y * self._scale + self._offset_y

    def _c2i(self, cx: float, cy: float) -> Tuple[float, float]:
        s = max(self._scale, 1e-9)
        return (cx - self._offset_x) / s, (cy - self._offset_y) / s

    # ─── Zoom / pan ──────────────────────────────────────────────────

    def _clamp_pan(self) -> None:
        """画像が画面外へ完全に逃げないよう、最低 64 canvas px は残す。"""
        if self.frame_w <= 0 or self.frame_h <= 0:
            return
        dw = self.frame_w * self._scale
        dh = self.frame_h * self._scale
        base_x = (self._canvas_w - dw) / 2
        base_y = (self._canvas_h - dh) / 2
        keep = float(_PAN_KEEP_PX)
        min_x = -base_x - dw + min(keep, dw)
        max_x = -base_x + self._canvas_w - min(keep, dw)
        min_y = -base_y - dh + min(keep, dh)
        max_y = -base_y + self._canvas_h - min(keep, dh)
        self._pan_x = max(min_x, min(self._pan_x, max_x))
        self._pan_y = max(min_y, min(self._pan_y, max_y))

    def fit_view(self) -> None:
        """F: ズームと平行移動をリセットしてウィンドウに合わせる。"""
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._recompute_scale()
        self._render_canvas()
        self._refresh_zoom_label()
        self._status("View fitted to the window.")

    def zoom_by(self, factor: float, anchor: Optional[Tuple[float, float]] = None) -> None:
        """キャンバス座標 ``anchor`` を固定したまま拡大縮小する。"""
        if self.cap is None:
            return
        if anchor is None:
            anchor = (self._canvas_w / 2.0, self._canvas_h / 2.0)
        cx, cy = anchor
        ix, iy = self._c2i(cx, cy)
        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom * float(factor)))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        self._zoom = new_zoom
        self._recompute_scale()
        nx, ny = self._i2c(ix, iy)
        self._pan_x += cx - nx
        self._pan_y += cy - ny
        self._clamp_pan()
        self._recompute_scale()
        self._render_canvas()
        self._refresh_zoom_label()

    def _refresh_zoom_label(self) -> None:
        if dpg.does_item_exist("tb_zoom_label"):
            dpg.set_value("tb_zoom_label", f"Zoom {self._zoom * 100:.0f}%")

    # ─── Layers ──────────────────────────────────────────────────────

    def _ensure_layer(self, tag: str) -> bool:
        """drawlist の中の描画レイヤを用意する (無ければ作る)。"""
        if not dpg.does_item_exist("canvas"):
            return False
        if not dpg.does_item_exist(tag):
            dpg.add_draw_layer(tag=tag, parent="canvas")
        return True

    def _render_canvas(self) -> None:
        if not dpg.does_item_exist("canvas"):
            return
        self._recompute_scale()
        if not self._ensure_layer("layer_image"):
            return
        dpg.delete_item("layer_image", children_only=True)
        cw, ch = self._canvas_w, self._canvas_h

        if self._current_bgr is None:
            dpg.draw_text((max(cw // 2 - 200, 10), ch // 2),
                          "Open a video to start (File > Open video, Ctrl+O)",
                          color=theme.TEXT_FAINT, size=16, parent="layer_image")
            self._render_overlay()
            return

        # Frame image
        if dpg.does_item_exist(_FRAME_TEX):
            dw = self.frame_w * self._scale
            dh = self.frame_h * self._scale
            dpg.draw_image(_FRAME_TEX,
                           pmin=(self._offset_x, self._offset_y),
                           pmax=(self._offset_x + dw, self._offset_y + dh),
                           parent="layer_image")

        # Annotated boxes (candidates last so they sit on top of accepted work)
        objs = self._visible_objects(self.current_frame_idx)
        for obj in sorted(objs, key=lambda o: 1 if o.is_candidate else 0):
            self._draw_box(obj, selected=(obj is self._selected))

        self._render_overlay()

    def _visible_objects(self, frame_idx: int) -> List[AnnObject]:
        """描画対象の一覧。当たり判定もこれと同じリストを逆順に走査する
        (見えているのに掴めない / 見えないのに選べる、を構造的に防ぐ)。"""
        return self.annotations.get(frame_idx, [])

    def _render_overlay(self) -> None:
        """
        選択ハンドル・ラバーバンド・ツール枠・十字線だけを描き直す軽量パス。

        マウス移動のたびにシーン全体を作り直さないためのレイヤ分割。
        """
        if not self._ensure_layer("layer_overlay"):
            return
        dpg.delete_item("layer_overlay", children_only=True)
        L = "layer_overlay"
        cw, ch = self._canvas_w, self._canvas_h

        # hovered box highlight
        if (self._hover_obj is not None
                and self._hover_obj is not self._selected
                and self._hover_obj in self._visible_objects(self.current_frame_idx)):
            hx1, hy1, hx2, hy2 = self._hover_obj.bbox
            dpg.draw_rectangle(self._i2c(hx1, hy1), self._i2c(hx2, hy2),
                               color=(255, 255, 255, 120),
                               fill=(255, 255, 255, 60), thickness=1.0, parent=L)

        # selection handles
        if self._selected is not None and self._selected in self._visible_objects(
                self.current_frame_idx):
            self._draw_selection_handles(self._selected, L)

        # rubber band while drawing
        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([self._draw_start[0], self._draw_end[0]])
            ys = sorted([self._draw_start[1], self._draw_end[1]])
            p1 = self._i2c(xs[0], ys[0])
            p2 = self._i2c(xs[1], ys[1])
            dpg.draw_rectangle(p1, p2, color=(255, 255, 255, 220),
                               fill=(255, 255, 255, 30),
                               thickness=1.5, parent=L)

        # tool frame + caption + crosshair
        if self._tool != "select":
            col = theme.rgba(theme.DANGER if self._tool == "delete" else theme.ACCENT)
            dpg.draw_rectangle((2, 2), (max(cw - 3, 4), max(ch - 3, 4)),
                               color=col, thickness=2.0, parent=L)
            caption = _TOOL_CAPTIONS.get(self._tool, "")
            if caption:
                dpg.draw_text((12, 10), caption, color=col, size=13, parent=L)
            if (self._tool in ("draw", "obb", "delete")
                    and self._cursor_canvas is not None
                    and self._is_over_canvas()):
                gx, gy = self._cursor_canvas
                guide = (col[0], col[1], col[2], 90)
                dpg.draw_line((0, gy), (cw, gy), color=guide, thickness=1.0,
                              parent=L)
                dpg.draw_line((gx, 0), (gx, ch), color=guide, thickness=1.0,
                              parent=L)

    def _draw_selection_handles(self, obj: AnnObject, parent: str) -> None:
        col = self._box_color(obj)
        if obj.has_obb:
            for px, py in obj.corners():
                hx, hy = self._i2c(px, py)
                dpg.draw_circle((hx, hy), _HANDLE_R, color=(255, 255, 255, 220),
                                fill=theme.rgba(theme.DANGER), parent=parent)
            rot = self._rotation_handle_pos(obj)
            ctr = self._i2c(obj.obb[0], obj.obb[1])
            dpg.draw_line(ctr, rot, color=(255, 230, 0, 200), thickness=1.5,
                          parent=parent)
            dpg.draw_circle(rot, _HANDLE_R, color=(255, 255, 255, 220),
                            fill=(255, 230, 0, 200), parent=parent)
        else:
            x1, y1, x2, y2 = obj.bbox
            c1 = self._i2c(x1, y1)
            c2 = self._i2c(x2, y2)
            for hx, hy in (c1, (c2[0], c1[1]), (c1[0], c2[1]), c2):
                dpg.draw_circle((hx, hy), _HANDLE_R, color=(255, 255, 255, 220),
                                fill=theme.rgba(theme.DANGER), parent=parent)
        _ = col

    # ─── Colors ──────────────────────────────────────────────────────

    def class_color(self, class_id: int) -> Tuple[int, int, int]:
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        return _palette_color(class_id)[:3]

    def action_color(self, action_id: int) -> Tuple[int, int, int]:
        if action_id < 0:
            return theme.TEXT_FAINT
        if action_id in self.action_colors:
            return self.action_colors[action_id]
        # 行動色はトラック色と紛れないよう、パレットの後半から取る
        return _palette_color(action_id + 4)[:3]

    def _box_color(self, obj: AnnObject, alpha: int = 255) -> Tuple[int, int, int, int]:
        """輪郭の色。'Color boxes by' の設定に従う (表示のみ、保存には影響しない)。"""
        if obj.is_candidate:
            return theme.rgba(theme.PENDING, alpha)
        if self._color_by == "class":
            return theme.rgba(self.class_color(obj.class_id), alpha)
        if self._color_by == "action":
            return theme.rgba(self.action_color(obj.action_id), alpha)
        return theme.rgba(_track_color(obj.track_id)[:3], alpha)

    # ─── Box drawing ─────────────────────────────────────────────────

    def _draw_dashed_rect(self, c1: Tuple[float, float], c2: Tuple[float, float],
                          color, thickness: float = 1.5,
                          dash: float = 7.0, gap: float = 5.0,
                          parent: str = "layer_image") -> None:
        """破線の矩形を drawlist に描く (dpg に破線 API が無いため線分で近似)。"""
        x1, y1 = c1
        x2, y2 = c2
        for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            self._draw_dashed_line(a, b, color, thickness, dash, gap, parent)

    def _draw_dashed_line(self, a, b, color, thickness: float = 1.5,
                          dash: float = 7.0, gap: float = 5.0,
                          parent: str = "layer_image") -> None:
        """破線の線分 (dpg に破線 API が無いため線分で近似)。"""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1.0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        d = 0.0
        while d < length:
            e = min(d + dash, length)
            dpg.draw_line((a[0] + ux * d, a[1] + uy * d),
                          (a[0] + ux * e, a[1] + uy * e),
                          color=color, thickness=thickness, parent=parent)
            d += dash + gap

    def _draw_box(self, obj: AnnObject, selected: bool = False) -> None:
        """
        provenance で描き分ける:
          human / accepted        実線 2px   トラック色      "T3 | walking"
          machine / pending       破線 7,5   PENDING amber   "T? | ? 0.82"
          machine / accepted      実線 2px   トラック色 + 左に ACCEPTED の帯
          interpolated / tracked  破線 4,4   INTERPOLATED    薄く
        """
        L = "layer_image"
        x1, y1, x2, y2 = obj.bbox
        c1 = self._i2c(x1, y1)
        c2 = self._i2c(x2, y2)
        pending = obj.is_candidate
        interp = (not pending) and (not obj.keyframe or obj.source in ("track", "interp"))

        if pending:
            col = theme.rgba(theme.PENDING)
            col_fill = theme.rgba(theme.PENDING, 28)
            thick = 2.0
        elif interp:
            col = theme.rgba(theme.INTERPOLATED)
            col_fill = theme.rgba(theme.INTERPOLATED, 32)
            thick = 1.5
        else:
            col = self._box_color(obj)
            col_fill = self._box_color(obj, alpha=50)
            thick = 2.0
        if selected:
            thick += 1.0

        if obj.has_obb:
            pts = [self._i2c(px, py) for px, py in obj.corners()]
            if pending:
                for i in range(4):
                    self._draw_dashed_line(pts[i], pts[(i + 1) % 4], col, thick,
                                           7.0, 5.0, L)
            elif interp:
                for i in range(4):
                    self._draw_dashed_line(pts[i], pts[(i + 1) % 4], col, thick,
                                           4.0, 4.0, L)
            else:
                dpg.draw_polygon(pts, color=col, fill=col_fill,
                                 thickness=thick, parent=L)
            # 体軸 (w 方向) を示す線 — 向きが一目で判るようにする
            mid_front = ((pts[1][0] + pts[2][0]) / 2, (pts[1][1] + pts[2][1]) / 2)
            center = self._i2c(obj.obb[0], obj.obb[1])
            dpg.draw_line(center, mid_front, color=col, thickness=thick, parent=L)
        elif pending:
            self._draw_dashed_rect(c1, c2, col, thick, 7.0, 5.0, L)
            dpg.draw_rectangle(c1, c2, color=(0, 0, 0, 0), fill=col_fill,
                               thickness=0.0, parent=L)
        elif interp:
            self._draw_dashed_rect(c1, c2, col, thick, 4.0, 4.0, L)
        else:
            dpg.draw_rectangle(c1, c2, color=col, fill=col_fill,
                               thickness=thick, parent=L)

        # 機械が出して人が確定した box は左端に ACCEPTED の帯を出す
        if (not pending) and obj.source != "human":
            dpg.draw_line((c1[0], c1[1]), (c1[0], c2[1]),
                          color=theme.rgba(theme.ACCEPTED), thickness=3.0, parent=L)

        # Label
        if pending:
            action_s = "?" if obj.action_id < 0 else self._action_name(obj.action_id)
            tid_s = f"T{obj.track_id}" if obj.track_id >= 0 else "T?"
            score_s = f" {obj.score:.2f}" if obj.score is not None else ""
            label = f"{tid_s} | {action_s}{score_s}"
        else:
            action_s = self._action_name(obj.action_id)
            tid_s = f"T{obj.track_id}" if obj.track_id >= 0 else "T?"
            label = f"{tid_s} | {action_s}"
        dpg.draw_text((c1[0] + 3, c1[1] + 2), label,
                      color=(0, 0, 0, 200), size=13, parent=L)
        dpg.draw_text((c1[0] + 2, c1[1] + 1), label,
                      color=(255, 255, 255, 230), size=13, parent=L)

        if obj.occluded:
            dpg.draw_text((c2[0] - 30, c1[1] + 2), "OCC",
                          color=theme.rgba(theme.PENDING), size=11, parent=L)

        # Track indicator (colored dot)
        has_tracker = obj.track_id in self._trackers
        dot_col = theme.rgba(theme.ACCEPTED) if has_tracker else col
        dpg.draw_circle((c1[0] + 6, c2[1] - 6), 5,
                        color=dot_col, fill=dot_col, parent=L)

    def _action_name(self, action_id: int) -> str:
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        return "-"

    def _rotation_handle_pos(self, obj: AnnObject) -> Tuple[float, float]:
        """回転ハンドルの canvas 座標 (体軸 +w 方向へ少し離した位置)。"""
        cx, cy, w, h, th = obj.get_obb()
        # 箱のサイズに依らず掴みやすいよう、最低でも一定距離だけ外に出す
        dist = max(w / 2 + _ROT_HANDLE_OFFSET / max(self._scale, 1e-6), w / 2 + 1)
        return self._i2c(cx + dist * math.cos(th), cy + dist * math.sin(th))
