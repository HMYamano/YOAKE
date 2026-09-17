
    # ═══════════════════════════════════════════════════════════════════
    # Undo / Redo
    # ═══════════════════════════════════════════════════════════════════

    def _snapshot(self, frames) -> Dict[int, List[AnnObject]]:
        """指定フレームのオブジェクト列を丸ごと複製する (取り消し用)。"""
        snap: Dict[int, List[AnnObject]] = {}
        for f in frames:
            f = int(f)
            snap[f] = [copy.deepcopy(o) for o in self.annotations.get(f, [])]
        return snap

    def _commit_snapshot(self, snap: Dict[int, List[AnnObject]]) -> None:
        self._undo.append(snap)
        if len(self._undo) > _UNDO_DEPTH:
            del self._undo[0]
        self._redo.clear()

    def _push_undo(self, frames) -> None:
        """変更を加える**前**に呼ぶ。範囲操作は触る全フレームを渡すこと。"""
        self._commit_snapshot(self._snapshot(frames))

    def _restore(self, snap: Dict[int, List[AnnObject]]) -> Dict[int, List[AnnObject]]:
        """snap を適用し、適用前の状態を返す (redo 用の逆スナップショット)。"""
        inverse = self._snapshot(snap.keys())
        for f, objs in snap.items():
            if objs:
                self.annotations[int(f)] = [copy.deepcopy(o) for o in objs]
            else:
                self.annotations.pop(int(f), None)
        return inverse

    def undo(self) -> None:
        if not self._undo:
            self._status("Nothing to undo.")
            return
        snap = self._undo.pop()
        self._redo.append(self._restore(snap))
        if len(self._redo) > _UNDO_DEPTH:
            del self._redo[0]
        # インデックスは意味を失うので選択は捨てる
        self._selected = None
        self._hover_obj = None
        self._dirty = True
        self._after_mutation(f"Undone. {len(self._undo)} steps left.")

    def redo(self) -> None:
        if not self._redo:
            self._status("Nothing to redo.")
            return
        snap = self._redo.pop()
        self._undo.append(self._restore(snap))
        if len(self._undo) > _UNDO_DEPTH:
            del self._undo[0]
        self._selected = None
        self._hover_obj = None
        self._dirty = True
        self._after_mutation(f"Redone. {len(self._redo)} steps left.")

    def _undo_hint(self) -> str:
        return "Ctrl+Z can undo this."

    def _after_mutation(self, message: str = "") -> None:
        """変更後の共通処理: 再描画・サイドバー更新・ステータス更新。"""
        self._render_canvas()
        self._refresh_sidebar()
        self._refresh_auto_panel()
        self._refresh_status_bar()
        if message:
            self._status(message)

    def _remove_object(self, obj: AnnObject) -> None:
        objs = self.annotations.get(self.current_frame_idx, [])
        if obj in objs:
            objs.remove(obj)
        if not objs:
            self.annotations.pop(self.current_frame_idx, None)
        if obj is self._selected:
            self._selected = None
        if obj is self._hover_obj:
            self._hover_obj = None
        self._dirty = True

    # ═══════════════════════════════════════════════════════════════════
    # Video Loading
    # ═══════════════════════════════════════════════════════════════════

    def _open_video_dialog(self) -> None:
        path = _tk_open_file(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path is not None:
            self._load_video(path)

    def _load_video(self, path: Path) -> None:
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._status(f"Cannot open the video: {path.name}. "
                         "Check its path and format.")
            return

        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.annotations = {}
        self.current_frame_idx = 0
        self._selected = None
        self._hover_obj = None
        self._trackers.clear()
        self._undo.clear()
        self._redo.clear()
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._dirty = False
        self.annotation_path = path.parent / (path.stem + "_annotations.json")

        # Create frame texture
        if dpg.does_item_exist(_FRAME_TEX):
            dpg.delete_item(_FRAME_TEX)
        blank = np.zeros(self.frame_w * self.frame_h * 4, dtype=np.float32)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                dpg.add_raw_texture(self.frame_w, self.frame_h, blank,
                                    tag=_FRAME_TEX,
                                    format=dpg.mvFormat_Float_rgba)
        else:
            dpg.push_container_stack("tex_reg")
            dpg.add_raw_texture(self.frame_w, self.frame_h, blank,
                                tag=_FRAME_TEX,
                                format=dpg.mvFormat_Float_rgba)
            dpg.pop_container_stack()

        if dpg.does_item_exist("frame_input"):
            dpg.configure_item("frame_input", max_value=max(self.num_frames - 1, 0))
        if dpg.does_item_exist("total_frames_text"):
            dpg.set_value("total_frames_text", f"/ {max(self.num_frames - 1, 0)}")
        for _rtag in ("range_start", "range_end"):
            if dpg.does_item_exist(_rtag):
                dpg.configure_item(_rtag, max_value=max(self.num_frames - 1, 0))
                dpg.set_value(_rtag, 0)

        dpg.set_viewport_title(f"YOAKE Annotation Tool - {path.name}")
        self.goto_frame(0)
        self._refresh_zoom_label()
        self._status(f"Loaded {path.name} - {self.num_frames} frames, "
                     f"{self.fps:.1f} fps, {self.frame_w} x {self.frame_h}")

    def _read_frame_bgr(self, idx: int) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _worker_reader(self):
        """
        ワーカースレッド用の独立したフレームリーダ。

        ``cv2.VideoCapture`` はスレッド安全ではないので、UI スレッドが使っている
        ``self.cap`` には触れず、専用のキャプチャを開く。前方への移動は
        ``grab()`` の連打で進める (``set(POS_FRAMES)`` は非整数 fps で 1 枚ずれる)。
        """
        if self.video_path is None:
            return (lambda i: None), None
        cap = cv2.VideoCapture(str(self.video_path))
        state = {"pos": -1}

        def read(i: int):
            i = int(i)
            if not cap.isOpened():
                return None
            if i < state["pos"] or i > state["pos"] + 96:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                state["pos"] = i - 1
            while state["pos"] < i - 1:
                if not cap.grab():
                    return None
                state["pos"] += 1
            ok, frame = cap.read()
            state["pos"] = i
            return frame if ok else None

        return read, cap

    # ═══════════════════════════════════════════════════════════════════
    # Frame Navigation
    # ═══════════════════════════════════════════════════════════════════

    def goto_frame(self, idx: int) -> None:
        if self.cap is None:
            return
        idx = max(0, min(idx, self.num_frames - 1))

        # Auto-track: run trackers when advancing one frame
        if self._auto_track and idx == self.current_frame_idx + 1 and self._trackers:
            frame_bgr = self._read_frame_bgr(idx)
            if frame_bgr is not None:
                self._run_trackers_for_frame(idx, frame_bgr)

        self.current_frame_idx = idx
        frame_bgr = self._read_frame_bgr(idx)
        self._current_bgr = frame_bgr

        if frame_bgr is not None and dpg.does_item_exist(_FRAME_TEX):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
            data = (rgba.astype(np.float32) / 255.0).flatten()
            dpg.set_value(_FRAME_TEX, data)

        self.deselect(render=False)
        self._hover_obj = None
        if dpg.does_item_exist("frame_input"):
            dpg.set_value("frame_input", idx)

        self._render_canvas()
        self._refresh_sidebar()
        self._refresh_auto_panel()

        annotated = len(self.annotations)
        pct = annotated / max(self.num_frames, 1)
        if dpg.does_item_exist("progress_bar"):
            dpg.set_value("progress_bar", pct)
        self._refresh_status_bar()
        self._status(
            f"Frame {idx} / {max(self.num_frames - 1, 0)}  -  "
            f"{annotated} of {self.num_frames} frames annotated "
            f"({pct * 100:.0f}%)"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Selection
    # ═══════════════════════════════════════════════════════════════════

    def _select(self, obj: AnnObject) -> None:
        self._selected = obj
        self._refresh_sidebar()
        self._render_canvas()
        self._refresh_status_bar()

    def deselect(self, render: bool = True) -> None:
        self._selected = None
        self._refresh_sidebar()
        if render:
            self._render_canvas()
        self._refresh_status_bar()

    def cycle_selection(self) -> None:
        """Q: このフレームの box を順に選ぶ (候補を先に)。"""
        objs = self._visible_objects(self.current_frame_idx)
        if not objs:
            self._status("No boxes in this frame.")
            return
        order = [o for o in objs if o.is_candidate] + [o for o in objs if not o.is_candidate]
        if self._selected in order:
            nxt = order[(order.index(self._selected) + 1) % len(order)]
        else:
            nxt = order[0]
        self._select(nxt)
        kind = "candidate" if nxt.is_candidate else "box"
        self._status(f"Selected {kind}: track {nxt.track_id}.")

    # ═══════════════════════════════════════════════════════════════════
    # Annotation Management
    # ═══════════════════════════════════════════════════════════════════

    def _create_object(self, bbox: List[int]) -> AnnObject:
        obj = AnnObject(bbox=bbox, track_id=self.active_track_id, action_id=-1)
        if self._tool == "obb" or self._obb_mode:
            # 回転ツールでは theta=0 の回転矩形として作る。以後は回転ハンドルで
            # 向きを付けられる (最初から回転させる必要はない)。
            obj.make_obb()
        self.annotations.setdefault(self.current_frame_idx, []).append(obj)
        self._select(obj)
        self._dirty = True
        return obj

    # ─── Tool mode (select / draw / obb / pan / delete) ──────────────

    def set_tool(self, tool: str) -> None:
        """ツールを切り替える (select / draw / obb / pan / delete)。"""
        self._tool = tool if tool in _TOOLS else "select"
        self._obb_mode = (self._tool == "obb")
        if self._mode == "draw":
            self._mode = "idle"
            self._draw_start = self._draw_end = None
        self._status(_TOOL_CHIPS[self._tool])
        self._refresh_tool_ui()
        self._render_canvas()
        self._refresh_status_bar()

    def toggle_tool(self, tool: str) -> None:
        """同じツールのキーをもう一度押したら Select に戻す (トグル)。"""
        self.set_tool("select" if self._tool == tool else tool)

    def toggle_draw_tool(self) -> None:
        """W キー: 描画モードのオン / オフ (LabelImg の 'Create RectBox')。"""
        self.toggle_tool("draw")

    def _on_sticky_draw_toggle(self, sender=None, app_data=None) -> None:
        self._sticky_draw = bool(app_data)
        for tag in ("sticky_draw_cb", "tb_sticky_draw"):
            if dpg.does_item_exist(tag) and dpg.get_value(tag) != self._sticky_draw:
                dpg.set_value(tag, self._sticky_draw)
        self._status(
            "Keep drawing is on - the drawing tool stays active after each box."
            if self._sticky_draw else
            "Keep drawing is off - the tool returns to Select after each box."
        )

    def _on_color_by_change(self, sender=None, app_data=None) -> None:
        self._color_by = str(app_data or "track")
        self._render_canvas()
        self._status(f"Boxes are colored by {self._color_by}. "
                     "Display only - saved labels are unaffected.")

    def _refresh_tool_ui(self) -> None:
        """
        ツールの見え方を 3 系統そろえる: ボタンの塗り / 文章チップ /
        キャンバス内の枠と見出し (十字線は _render_overlay 側)。
        """
        for tool, tag in _TOOL_TAGS.items():
            widgets.set_toolbar_active(tag, tool == self._tool)
        chip = _TOOL_CHIPS.get(self._tool, "")
        for tag in ("tb_tool_chip", "tool_mode_text"):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, chip)
                dpg.configure_item(
                    tag, color=(theme.TEXT_DIM if self._tool == "select"
                                else (theme.DANGER if self._tool == "delete"
                                      else theme.ACCENT)))
        if dpg.does_item_exist("sb_tool"):
            dpg.set_value("sb_tool", _TOOL_NAMES[self._tool])

    def _escape(self) -> None:
        """Esc: 描きかけを捨て、Select へ戻り、選択も解除する。"""
        if self._mode == "draw":
            self._mode = "idle"
            self._draw_start = self._draw_end = None
        self._mode = "idle"
        if self._tool != "select":
            self.set_tool("select")
        self.deselect()

    def toggle_obb_mode(self) -> None:
        """後方互換: 旧 R キーの「OBB モード」は Draw Rotated Box ツールになった。"""
        self.toggle_tool("obb")

    def convert_selected_to_obb(self) -> None:
        """選択中のボックスを回転矩形へ昇格させる (theta=0 から開始)。"""
        if self._selected is None:
            self._status("Select a box first.")
            return
        self._push_undo([self.current_frame_idx])
        self._selected.make_obb()
        self._selected.keyframe = True
        self._dirty = True
        self._after_mutation("Converted to a rotated box. "
                             "Turn the yellow handle to set its direction.")

    def convert_selected_to_aabb(self) -> None:
        """選択中の回転矩形を軸平行ボックスへ戻す (外接矩形になる)。"""
        if self._selected is None or not self._selected.has_obb:
            self._status("Select a rotated box first.")
            return
        self._push_undo([self.current_frame_idx])
        self._selected.drop_obb()
        self._selected.keyframe = True
        self._dirty = True
        self._after_mutation("Reverted to an axis-aligned box.")

    def rotate_selected(self, degrees: float) -> None:
        """選択中の OBB を相対角度で回す (キーボードからの微調整用)。"""
        if self._selected is None:
            self._status("Select a box first.")
            return
        self._push_undo([self.current_frame_idx])
        self._selected.make_obb()
        self._selected.obb[4] += math.radians(degrees)
        self._selected.sync_bbox()
        self._selected.keyframe = True
        self._dirty = True
        self._after_mutation(
            f"Rotated {degrees:+.0f} deg to "
            f"{math.degrees(self._selected.obb[4]):.1f} deg.")

    def delete_selected(self) -> None:
        if self._selected is None:
            self._status("Select a box first.")
            return
        self._push_undo([self.current_frame_idx])
        tid = self._selected.track_id
        if tid in self._trackers:
            del self._trackers[tid]
        self._remove_object(self._selected)
        self._after_mutation(f"Deleted the box on track {tid}. {self._undo_hint()}")

    def new_track(self) -> None:
        self.next_track_id += 1
        self.active_track_id = self.next_track_id
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.next_track_id)
        if dpg.does_item_exist("range_track"):
            dpg.set_value("range_track", self.active_track_id)
        self._update_active_track_color()
        self._status(f"New track ID: {self.next_track_id}")

    def propagate_from_prev(self) -> None:
        if self.cap is None:
            self._status("Open a video first.")
            return
        prev = self.current_frame_idx - 1
        if prev < 0:
            self._status("No previous frame to copy from.")
            return
        prev_objs = [o for o in self.annotations.get(prev, []) if o.accepted]
        if not prev_objs:
            self._status("The previous frame has no boxes.")
            return
        existing_tids = {o.track_id for o in
                         self.annotations.get(self.current_frame_idx, [])}
        self._push_undo([self.current_frame_idx])
        count = 0
        for obj in prev_objs:
            if obj.track_id not in existing_tids:
                global _gid
                new_obj = copy.deepcopy(obj)
                _gid += 1
                new_obj.object_id = _gid
                self.annotations.setdefault(self.current_frame_idx, []).append(new_obj)
                existing_tids.add(obj.track_id)
                count += 1
        self._dirty = True
        self._after_mutation(f"Copied {count} boxes from frame {prev}. "
                             f"{self._undo_hint()}")
