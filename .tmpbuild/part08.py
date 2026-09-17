
    # ═══════════════════════════════════════════════════════════════════
    # Mouse Interaction
    # ═══════════════════════════════════════════════════════════════════

    def _is_over_canvas(self) -> bool:
        """
        マウスが描画キャンバス上にあるか。

        child_window の ``is_item_hovered()`` は ImGui の仕様上、マウスボタンを
        押している間 (active item がいる間) は False を返す。これをドラッグ判定に
        使うとボタンを押した瞬間に False になり、ドラッグが一切届かなくなる
        (= ボックスが描けない)。drawlist は押下中も True を返すので、そちらを見る。
        """
        if dpg.does_item_exist("canvas") and dpg.is_item_hovered("canvas"):
            return True
        # ドラッグ中はキャンバスの外へはみ出しても操作を続けられるようにする
        # (座標は frame 境界へクランプ済みなので安全)。
        if self._mode != "idle" and dpg.is_mouse_button_down(0):
            return True
        return dpg.does_item_exist("canvas_win") and dpg.is_item_hovered("canvas_win")

    def _clamped_image_point(self, cx: float, cy: float) -> Tuple[float, float]:
        ix, iy = self._c2i(cx, cy)
        return (max(0.0, min(ix, float(self.frame_w))),
                max(0.0, min(iy, float(self.frame_h))))

    def _on_mouse_press(self, sender=None, app_data=None) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        self._moved = False

        # Pan tool: 左ドラッグで画面を動かす
        if self._tool == "pan":
            self._mode = "pan"
            self._pan_origin = (cx, cy, self._pan_x, self._pan_y)
            return

        # Delete tool (X): クリックした箱を 1 つ消す (粘着、Esc で抜ける)
        if self._tool == "delete":
            hit = self._hit_object(ix, iy)
            if hit is None:
                self._status("No box there. Click on a box to remove it.")
                return
            self._push_undo([self.current_frame_idx])
            self._remove_object(hit)
            self._after_mutation(f"Deleted 1 box. {self._undo_hint()}")
            return

        # Draw tools (W / E): 既存の箱には触れず、必ず新しい箱を描き始める。
        # これにより既存ボックスの内側や重なった位置にも描ける (LabelImg と同じ)。
        if self._tool in ("draw", "obb"):
            self.deselect(render=False)
            self._mode = "draw"
            self._draw_start = self._clamped_image_point(cx, cy)
            self._draw_end = self._draw_start
            return

        # --- Select tool ------------------------------------------------
        # Resize / rotate handle?
        if self._selected is not None:
            h = self._hit_handle(cx, cy, self._selected)
            if h is not None:
                self._pending_snap = self._snapshot([self.current_frame_idx])
                self._mode = "rotate" if h == "rot" else "resize"
                self._resize_handle = h
                self._drag_start = (ix, iy)
                self._drag_bbox_orig = list(self._selected.bbox)
                self._drag_obb_orig = (
                    list(self._selected.obb) if self._selected.has_obb else None
                )
                return

        # Existing box?
        hit = self._hit_object(ix, iy)
        if hit is not None:
            if hit.is_candidate:
                # 候補は左クリックで確定 (これがレビューの主操作)
                self._push_undo([self.current_frame_idx])
                hit.accept()
                hit.keyframe = True
                self._select(hit)
                self._after_mutation(f"Accepted: track {hit.track_id}.")
                return
            self._pending_snap = self._snapshot([self.current_frame_idx])
            self._select(hit)
            self._mode = "move"
            self._drag_start = (ix, iy)
            self._drag_bbox_orig = list(hit.bbox)
            self._drag_obb_orig = list(hit.obb) if hit.has_obb else None
            return

        # Draw new box on empty canvas
        self.deselect(render=False)
        self._mode = "draw"
        self._draw_start = self._clamped_image_point(cx, cy)
        self._draw_end = self._draw_start

    def _on_mouse_drag(self, sender=None, app_data=None) -> None:
        if self._mode == "idle" or not self._is_over_canvas():
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        ixc, iyc = self._clamped_image_point(cx, cy)
        self._moved = True

        if self._mode == "pan":
            if self._pan_origin is not None:
                sx, sy, px, py = self._pan_origin
                self._pan_x = px + (cx - sx)
                self._pan_y = py + (cy - sy)
                self._clamp_pan()
                self._render_canvas()
            return

        if self._mode == "draw":
            self._draw_end = (ixc, iyc)
            self._render_overlay()
            return

        if self._mode == "move" and self._selected is not None:
            dx = ix - self._drag_start[0]
            dy = iy - self._drag_start[1]
            if self._drag_obb_orig is not None:
                # OBB は中心を動かすだけ (w/h/theta は保つ)
                ocx, ocy, ow, oh, oth = self._drag_obb_orig
                self._selected.obb = [
                    max(0.0, min(ocx + dx, float(self.frame_w))),
                    max(0.0, min(ocy + dy, float(self.frame_h))),
                    ow, oh, oth,
                ]
                self._selected.sync_bbox()
            else:
                ox1, oy1, ox2, oy2 = self._drag_bbox_orig
                w, h = ox2 - ox1, oy2 - oy1
                nx1 = int(max(0.0, min(ox1 + dx, self.frame_w - w)))
                ny1 = int(max(0.0, min(oy1 + dy, self.frame_h - h)))
                self._selected.bbox = [nx1, ny1, nx1 + w, ny1 + h]
            self._selected.keyframe = True  # 手動移動でキーフレーム化 (補間アンカー)
            # Re-init tracker on manual move
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        elif self._mode == "rotate" and self._selected is not None:
            # 中心からマウスへの向きをそのまま体軸方向にする (直感的な回転)
            ocx, ocy = self._selected.obb[0], self._selected.obb[1]
            self._selected.obb[4] = math.atan2(iy - ocy, ix - ocx)
            self._selected.sync_bbox()
            self._selected.keyframe = True
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        elif self._mode == "resize" and self._selected is not None:
            if self._drag_obb_orig is not None:
                self._resize_obb(ixc, iyc)
            else:
                ox1, oy1, ox2, oy2 = self._drag_bbox_orig
                ixi, iyi = int(ixc), int(iyc)
                h_name = self._resize_handle
                if h_name == "tl":
                    nx1, ny1, nx2, ny2 = ixi, iyi, ox2, oy2
                elif h_name == "tr":
                    nx1, ny1, nx2, ny2 = ox1, iyi, ixi, oy2
                elif h_name == "bl":
                    nx1, ny1, nx2, ny2 = ixi, oy1, ox2, iyi
                else:
                    nx1, ny1, nx2, ny2 = ox1, oy1, ixi, iyi
                nx1, nx2 = sorted([nx1, nx2])
                ny1, ny2 = sorted([ny1, ny2])
                if nx2 - nx1 >= _MIN_BOX_PX and ny2 - ny1 >= _MIN_BOX_PX:
                    self._selected.bbox = [nx1, ny1, nx2, ny2]
                    self._selected.keyframe = True  # 手動リサイズでキーフレーム化
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        self._render_canvas()

    def _resize_obb(self, ix: float, iy: float) -> None:
        """
        回転矩形の隅ハンドルドラッグ。

        マウス位置を box ローカル座標へ戻し、対角の隅を固定したまま w/h を決める。
        こうすると「掴んだ隅だけが動く」という軸平行の操作感が、回転していても
        そのまま成り立つ。
        """
        obj = self._selected
        ocx, ocy, ow, oh, oth = self._drag_obb_orig
        idx = int(self._resize_handle[1])           # "c0".."c3"
        c, s = math.cos(-oth), math.sin(-oth)

        # マウスをローカル座標へ
        dx, dy = ix - ocx, iy - ocy
        lx = dx * c - dy * s
        ly = dx * s + dy * c

        # 対角の固定点 (ローカル座標)
        signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        fx = -signs[idx][0] * ow / 2
        fy = -signs[idx][1] * oh / 2

        new_w = abs(lx - fx)
        new_h = abs(ly - fy)
        if new_w < _MIN_BOX_PX or new_h < _MIN_BOX_PX:
            return

        # 新しい中心 = 固定点と可動点の中点 (ローカル) → グローバルへ戻す
        mid_lx = (lx + fx) / 2
        mid_ly = (ly + fy) / 2
        cf, sf = math.cos(oth), math.sin(oth)
        obj.obb = [
            ocx + mid_lx * cf - mid_ly * sf,
            ocy + mid_lx * sf + mid_ly * cf,
            new_w, new_h, oth,
        ]
        obj.sync_bbox()
        obj.keyframe = True

    def _on_mouse_release(self, sender=None, app_data=None) -> None:
        if self._mode == "pan":
            self._mode = "idle"
            self._pan_origin = None
            self._pending_snap = None
            return

        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([int(self._draw_start[0]), int(self._draw_end[0])])
            ys = sorted([int(self._draw_start[1]), int(self._draw_end[1])])
            if xs[1] - xs[0] >= _MIN_BOX_PX and ys[1] - ys[0] >= _MIN_BOX_PX:
                self._push_undo([self.current_frame_idx])
                obj = self._create_object([xs[0], ys[0], xs[1], ys[1]])
                # Initialize tracker in auto-track mode
                if self._auto_track and self._current_bgr is not None:
                    self._init_tracker(obj.track_id, self.current_frame_idx, obj.bbox)
                self._dirty = True
                if self._tool in ("draw", "obb") and not self._sticky_draw:
                    # LabelImg と同じく 1 個描いたら選択ツールへ戻る
                    self.set_tool("select")
                else:
                    self._status("Box created. Press W to draw another one.")
            else:
                self._status(
                    f"The box was too small. Drag an area of at least "
                    f"{_MIN_BOX_PX} x {_MIN_BOX_PX} pixels."
                )
        elif self._mode in ("move", "resize", "rotate") and self._selected is not None:
            if self._moved and self._pending_snap is not None:
                self._commit_snapshot(self._pending_snap)
                self._dirty = True
            if self._auto_track and self._current_bgr is not None:
                self._init_tracker(self._selected.track_id,
                                   self.current_frame_idx, self._selected.bbox)

        self._pending_snap = None
        self._moved = False
        self._mode = "idle"
        self._draw_start = self._draw_end = None
        self._drag_start = self._drag_bbox_orig = None
        self._drag_obb_orig = None
        self._resize_handle = None
        self._render_canvas()
        self._refresh_sidebar()
        self._refresh_status_bar()

    # ─── Middle-drag pan (available from every tool) ─────────────────

    def _on_middle_press(self, sender=None, app_data=None) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        self._pan_origin = (cx, cy, self._pan_x, self._pan_y)

    def _on_middle_drag(self, sender=None, app_data=None) -> None:
        if self._pan_origin is None or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        sx, sy, px, py = self._pan_origin
        self._pan_x = px + (cx - sx)
        self._pan_y = py + (cy - sy)
        self._clamp_pan()
        self._render_canvas()

    def _on_middle_release(self, sender=None, app_data=None) -> None:
        self._pan_origin = None

    # ─── Wheel zoom / hover ──────────────────────────────────────────

    def _on_mouse_wheel(self, sender=None, app_data=0) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        try:
            notches = float(app_data)
        except (TypeError, ValueError):
            notches = 0.0
        if notches == 0.0:
            return
        self.zoom_by(1.25 if notches > 0 else 0.8, anchor=(cx, cy))

    def _on_mouse_move(self, sender=None, app_data=None) -> None:
        if self.cap is None:
            return
        if not (dpg.does_item_exist("canvas") and dpg.is_item_hovered("canvas")):
            if self._cursor_canvas is not None or self._hover_obj is not None:
                self._cursor_canvas = None
                self._hover_obj = None
                self._render_overlay()
            return
        cx, cy = self._get_canvas_mouse()
        self._cursor_canvas = (cx, cy)
        ix, iy = self._c2i(cx, cy)
        hover = self._hit_object(ix, iy) if self._mode == "idle" else None
        changed = hover is not self._hover_obj
        self._hover_obj = hover
        if changed or self._tool in ("draw", "obb", "delete"):
            self._render_overlay()

    def _on_right_click(self, sender=None, app_data=None) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        # 描画モード中の右クリックは描きかけのキャンセル (メニューは出さない)
        if self._tool in ("draw", "obb") and self._mode == "draw":
            self._mode = "idle"
            self._draw_start = self._draw_end = None
            self._render_canvas()
            self._status("Cancelled.")
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        hit = self._hit_object(ix, iy)
        if hit is None:
            return
        if hit.is_candidate:
            # 候補は右クリックで即削除 (レビューの reject)
            self._push_undo([self.current_frame_idx])
            self._remove_object(hit)
            self._after_mutation("Discarded 1 candidate.")
            return
        self._select(hit)
        self._right_click_obj = hit
        self._show_context_menu(hit)

    def _show_context_menu(self, obj: AnnObject) -> None:
        if dpg.does_item_exist("ctx_menu"):
            dpg.delete_item("ctx_menu")
        mx, my = dpg.get_mouse_pos(local=False)
        with dpg.window(tag="ctx_menu", popup=True, no_title_bar=True,
                        no_move=True, pos=(mx, my), min_size=(180, 10)):
            dpg.add_text(f"Track {obj.track_id}", color=theme.ACCENT)
            dpg.add_text(f"Source: {obj.source_label()}", color=theme.TEXT_DIM)
            dpg.add_separator()
            dpg.add_menu_item(label="Set track ID...",
                              callback=self._prompt_track_id_dialog)
            dpg.add_menu_item(label="Track forward from here (T)",
                              callback=self.track_forward_selected)
            dpg.add_separator()
            # --- OBB 操作 ---
            if obj.has_obb:
                angle_deg = math.degrees(obj.obb[4])
                dpg.add_text(f"Rotated box, angle {angle_deg:+.1f} deg",
                             color=theme.PENDING)
                dpg.add_menu_item(label="Rotate +15 deg",
                                  callback=lambda: self.rotate_selected(15))
                dpg.add_menu_item(label="Rotate -15 deg",
                                  callback=lambda: self.rotate_selected(-15))
                dpg.add_menu_item(label="Convert to axis-aligned box",
                                  callback=self.convert_selected_to_aabb)
            else:
                dpg.add_menu_item(label="Convert to rotated box (OBB)",
                                  callback=self.convert_selected_to_obb)
            dpg.add_separator()
            for i, name in enumerate(self.action_names):
                dpg.add_menu_item(
                    label=f"Action: {name}",
                    callback=lambda s, a, u: self.set_action(u),
                    user_data=i,
                )
            dpg.add_separator()
            if self._auto_track:
                dpg.add_menu_item(label="Initialize tracker for this box",
                                  callback=lambda: self._init_tracker_from_selected())
                dpg.add_menu_item(label="Remove tracker for this track",
                                  callback=lambda: self._reset_selected_tracker())
                dpg.add_separator()
            dpg.add_menu_item(label="Delete box", callback=self.delete_selected)

    def _hit_handle(self, cx: float, cy: float, obj: AnnObject) -> Optional[str]:
        """
        ハンドル判定。OBB のときは回転ハンドルを**先に**見る。

        回転ハンドルは体軸の外側にあり隅ハンドルと重なりうるので、優先順を
        固定しておかないと掴む対象が状況で変わってしまう。
        """
        if obj.has_obb:
            rx, ry = self._rotation_handle_pos(obj)
            if abs(cx - rx) <= _HANDLE_R + 3 and abs(cy - ry) <= _HANDLE_R + 3:
                return "rot"
            names = ("c0", "c1", "c2", "c3")
            for name, (px, py) in zip(names, obj.corners()):
                hx, hy = self._i2c(px, py)
                if abs(cx - hx) <= _HANDLE_R + 3 and abs(cy - hy) <= _HANDLE_R + 3:
                    return name
            return None

        x1, y1, x2, y2 = obj.bbox
        corners = {
            "tl": self._i2c(x1, y1), "tr": self._i2c(x2, y1),
            "bl": self._i2c(x1, y2), "br": self._i2c(x2, y2),
        }
        for name, (hx, hy) in corners.items():
            if abs(cx - hx) <= _HANDLE_R + 3 and abs(cy - hy) <= _HANDLE_R + 3:
                return name
        return None

    @staticmethod
    def _point_in_obb(px: float, py: float, obj: AnnObject) -> bool:
        """点が回転矩形の内部にあるか (box ローカル座標へ戻して判定)。"""
        cx, cy, w, h, th = obj.get_obb()
        c, s = math.cos(-th), math.sin(-th)
        dx, dy = px - cx, py - cy
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        return abs(lx) <= w / 2 and abs(ly) <= h / 2

    def _hit_object(self, ix: float, iy: float) -> Optional[AnnObject]:
        """画像座標 (ix, iy) にある最前面のオブジェクトを返す。

        描画と同じ ``_visible_objects`` を逆順に走査するので、見えている box は
        必ず掴め、見えない box は決して選ばれない。
        """
        for obj in reversed(self._visible_objects(self.current_frame_idx)):
            if obj.has_obb:
                if self._point_in_obb(ix, iy, obj):
                    return obj
            else:
                bx1, by1, bx2, by2 = obj.bbox
                if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                    return obj
        return None
