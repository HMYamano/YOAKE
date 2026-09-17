
    # ═══════════════════════════════════════════════════════════════════
    # Label Management
    # ═══════════════════════════════════════════════════════════════════

    def _add_class_label_dialog(self) -> None:
        self._label_dialog("Add class label", self._on_add_class)

    def _add_action_label_dialog(self) -> None:
        self._label_dialog("Add action label", self._on_add_action)

    def _label_dialog(self, title: str, confirm_cb) -> None:
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        with dpg.window(label=title, tag="label_dlg", modal=True,
                        width=320, height=130,
                        pos=[max((vw - 320) // 2, 0), max((vh - 130) // 2, 0)],
                        no_resize=True):
            dpg.add_input_text(tag="label_dlg_input", width=-1,
                               hint="New label name")
            widgets.gap(2)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", width=110, callback=confirm_cb)
                dpg.add_button(label="Cancel", width=110,
                               callback=lambda: dpg.delete_item("label_dlg"))

    def _on_add_class(self) -> None:
        name = dpg.get_value("label_dlg_input").strip()
        if name and name not in self.class_names:
            self.class_names.append(name)
            self._update_action_buttons()
            self._refresh_labels_display()
            self._dirty = True
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")

    def _on_add_action(self) -> None:
        name = dpg.get_value("label_dlg_input").strip()
        if name and name not in self.action_names:
            self.action_names.append(name)
            self._update_action_buttons()
            self._refresh_labels_display()
            self._dirty = True
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")

    # ─── Full label manager: add / rename / recolour / delete ────────

    def open_label_manager(self) -> None:
        if dpg.does_item_exist("label_mgr"):
            dpg.delete_item("label_mgr")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        w, h = 640, 560
        with dpg.window(label="Manage labels", tag="label_mgr", modal=True,
                        width=w, height=h,
                        pos=[max((vw - w) // 2, 0), max((vh - h) // 2, 0)]):
            widgets.section_header(
                "Classes and actions",
                "A class is the object category (fly, worm). An action is the "
                "behavior on a box (walking, grooming). Colors are used by the "
                "canvas when \"Color boxes by\" is set to class or action.",
                wrap=w - 40)
            dpg.add_text("Classes", color=theme.ACCENT)
            dpg.add_group(tag="lm_classes")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="lm_new_class", width=260,
                                   hint="New class name")
                dpg.add_button(label="Add class", width=110,
                               callback=lambda: self._lm_add("class"))
            widgets.gap(3)
            dpg.add_separator()
            widgets.gap(1)
            dpg.add_text("Actions", color=theme.ACCENT)
            dpg.add_group(tag="lm_actions")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="lm_new_action", width=260,
                                   hint="New action name")
                dpg.add_button(label="Add action", width=110,
                               callback=lambda: self._lm_add("action"))
            widgets.gap(3)
            dpg.add_separator()
            widgets.gap(1)
            dpg.add_button(label="Close", width=-1, height=30,
                           callback=lambda: dpg.delete_item("label_mgr"))
        self._refresh_label_manager()

    def _refresh_label_manager(self) -> None:
        if not dpg.does_item_exist("label_mgr"):
            return
        for kind, parent, names in (("class", "lm_classes", self.class_names),
                                    ("action", "lm_actions", self.action_names)):
            dpg.delete_item(parent, children_only=True)
            if not names:
                dpg.add_text("None defined yet.", parent=parent,
                             color=theme.TEXT_FAINT)
                continue
            for i, name in enumerate(names):
                col = (self.class_color(i) if kind == "class"
                       else self.action_color(i))
                row = dpg.add_group(horizontal=True, parent=parent)
                dpg.add_text(f"[{i}]", parent=row, color=theme.TEXT_FAINT)
                dpg.add_input_text(tag=f"lm_{kind}_name_{i}", default_value=name,
                                   width=190, parent=row)
                dpg.add_color_edit(
                    tag=f"lm_{kind}_col_{i}",
                    default_value=(col[0], col[1], col[2], 255),
                    no_inputs=True, no_alpha=True, width=40, parent=row,
                    callback=(lambda s, a, u, k=kind, idx=i:
                              self._lm_set_color(k, idx, a)))
                dpg.add_button(label="Rename", width=84, parent=row,
                               callback=(lambda s, a, u, k=kind, idx=i:
                                         self._lm_rename(k, idx)))
                dpg.add_button(label="Delete", width=78, parent=row,
                               tag=f"lm_{kind}_del_{i}",
                               callback=(lambda s, a, u, k=kind, idx=i:
                                         self._lm_delete(k, idx)))
                theme.bind(f"lm_{kind}_del_{i}", theme.danger_theme())

    def _lm_add(self, kind: str) -> None:
        tag = "lm_new_class" if kind == "class" else "lm_new_action"
        name = (dpg.get_value(tag) or "").strip()
        if not name:
            self._status("Type a name first.")
            return
        names = self.class_names if kind == "class" else self.action_names
        if name in names:
            self._status(f"There is already a {kind} called {name}.")
            return
        names.append(name)
        dpg.set_value(tag, "")
        self._dirty = True
        self._update_action_buttons()
        self._refresh_label_manager()
        self._refresh_labels_display()
        self._status(f"Added {kind} {name}.")

    def _lm_rename(self, kind: str, index: int) -> None:
        tag = f"lm_{kind}_name_{index}"
        if not dpg.does_item_exist(tag):
            return
        new = (dpg.get_value(tag) or "").strip()
        names = self.class_names if kind == "class" else self.action_names
        if not new or index >= len(names):
            self._status("Type a name first.")
            return
        old = names[index]
        names[index] = new
        self._dirty = True
        self._update_action_buttons()
        self._refresh_label_manager()
        self._refresh_labels_display()
        self._render_canvas()
        self._status(f"Renamed {kind} {old} to {new}. "
                     "Boxes keep their numeric id, so nothing is relabelled.")

    def _lm_set_color(self, kind: str, index: int, value) -> None:
        # dpg は色を 0..1 の float で返す
        try:
            rgb = tuple(int(round(float(c) * 255)) if float(c) <= 1.0 else int(c)
                        for c in value[:3])
        except (TypeError, ValueError):
            return
        store = self.class_colors if kind == "class" else self.action_colors
        store[int(index)] = rgb
        self._dirty = True
        self._update_action_buttons()
        self._render_canvas()

    def _lm_delete(self, kind: str, index: int) -> None:
        names = self.class_names if kind == "class" else self.action_names
        if index >= len(names):
            return
        name = names[index]
        used = sum(1 for objs in self.annotations.values() for o in objs
                   if (o.class_id if kind == "class" else o.action_id) == index)

        def _do_delete():
            del names[index]
            store = self.class_colors if kind == "class" else self.action_colors
            store.pop(index, None)
            # 後ろの id を 1 つ詰める。参照していた box は -1 (class は 0) に戻す。
            for objs in self.annotations.values():
                for o in objs:
                    cur = o.class_id if kind == "class" else o.action_id
                    if cur == index:
                        new = 0 if kind == "class" else -1
                    elif cur > index:
                        new = cur - 1
                    else:
                        continue
                    if kind == "class":
                        o.class_id = new
                    else:
                        o.action_id = new
            self._dirty = True
            self._update_action_buttons()
            self._refresh_label_manager()
            self._refresh_labels_display()
            self._render_canvas()
            self._status(f"Deleted {kind} {name}.")

        if used:
            widgets.confirm_modal(
                f"Delete the {kind} \"{name}\"?",
                f"{used} boxes use it. They keep their geometry and their track; "
                f"only the {kind} is cleared. Every higher id shifts down by one.",
                confirm_label="Delete", on_confirm=lambda: _do_delete(),
                danger=True, preserved="Boxes and tracks are kept.")
        else:
            _do_delete()

    def _refresh_labels_display(self) -> None:
        if dpg.does_item_exist("classes_display"):
            dpg.set_value("classes_display",
                          "\n".join(f"  [{i}] {n}"
                                    for i, n in enumerate(self.class_names))
                          or "  (none)")
        if dpg.does_item_exist("actions_display"):
            dpg.set_value("actions_display",
                          "\n".join(f"  [{i}] {n}"
                                    for i, n in enumerate(self.action_names))
                          or "  (none)")

    # ═══════════════════════════════════════════════════════════════════
    # Sidebar Refresh
    # ═══════════════════════════════════════════════════════════════════

    def _object_row(self, obj: AnnObject, index: int) -> str:
        """listbox の 1 行。先頭の記号が provenance を表す。"""
        mark = "?" if obj.is_candidate else ("~" if not obj.keyframe else "*")
        action_s = self._action_name(obj.action_id)
        score_s = f"  {obj.score:.2f}" if (obj.is_candidate and obj.score is not None) else ""
        occ = " [OCC]" if obj.occluded else ""
        return f"{index}. {mark} T{obj.track_id}  {action_s}{score_s}{occ}"

    def _refresh_sidebar(self) -> None:
        if not dpg.does_item_exist("obj_listbox"):
            return

        objects = self._visible_objects(self.current_frame_idx)
        items = []
        sel_idx = None
        for i, obj in enumerate(objects):
            items.append(self._object_row(obj, i))
            if obj is self._selected:
                sel_idx = i
        # 行のテキストではなく行番号で対応づける (同じ表示の行があっても迷わない)
        self._listbox_rows = list(items)

        dpg.configure_item("obj_listbox", items=items)
        if sel_idx is not None and sel_idx < len(items):
            dpg.set_value("obj_listbox", items[sel_idx])

        # Selected box props
        if self._selected is not None:
            dpg.set_value("sel_tid", str(self._selected.track_id))
            action_val = (self.action_names[self._selected.action_id]
                          if 0 <= self._selected.action_id < len(self.action_names)
                          else "(none)")
            dpg.set_value("sel_action", action_val)
            if dpg.does_item_exist("sel_class"):
                cls_val = (self.class_names[self._selected.class_id]
                           if 0 <= self._selected.class_id < len(self.class_names)
                           else "(none)")
                dpg.set_value("sel_class", cls_val)
            dpg.set_value("sel_occluded", self._selected.occluded)
            dpg.set_value("sel_crowd", self._selected.is_crowd)
            if dpg.does_item_exist("sel_source_text"):
                state = "candidate" if self._selected.is_candidate else "accepted"
                dpg.set_value("sel_source_text",
                              f"Source: {self._selected.source_label()}  ({state})")
        else:
            dpg.set_value("sel_tid", "")
            dpg.set_value("sel_action", "(none)")
            if dpg.does_item_exist("sel_class"):
                dpg.set_value("sel_class", "(none)")
            dpg.set_value("sel_occluded", False)
            dpg.set_value("sel_crowd", False)
            if dpg.does_item_exist("sel_source_text"):
                dpg.set_value("sel_source_text", "Source: -")

        # Frame info
        annotated = len(self.annotations)
        if self.cap is not None:
            info = (
                f"Frame: {self.current_frame_idx} / {max(self.num_frames-1,0)}\n"
                f"Objects: {len(objects)}  (candidates {len(self._pending_in(self.current_frame_idx))})\n"
                f"Annotated: {annotated} / {self.num_frames}\n"
                f"Size: {self.frame_w} x {self.frame_h}\n"
                f"FPS: {self.fps:.1f}\n"
                f"Zoom: {self._zoom * 100:.0f}%\n"
                f"Trackers: {len(self._trackers)} active"
            )
        else:
            info = "No video loaded."
        if dpg.does_item_exist("frame_info_text"):
            dpg.set_value("frame_info_text", info)

        self._refresh_labels_display()
        self._update_active_track_color()

    def _refresh_status_bar(self) -> None:
        if dpg.does_item_exist("sb_frame"):
            if self.cap is None:
                dpg.set_value("sb_frame", "No video")
            else:
                dpg.set_value("sb_frame",
                              f"Frame {self.current_frame_idx} / "
                              f"{max(self.num_frames - 1, 0)}")
        if dpg.does_item_exist("sb_tool"):
            dpg.set_value("sb_tool", _TOOL_NAMES.get(self._tool, "Select"))
        if dpg.does_item_exist("sb_sel"):
            if self._selected is None:
                dpg.set_value("sb_sel", "No selection")
                dpg.configure_item("sb_sel", color=theme.TEXT_DIM)
            else:
                state = "candidate" if self._selected.is_candidate else "box"
                dpg.set_value("sb_sel",
                              f"Selected {state}: track {self._selected.track_id}"
                              f" | {self._action_name(self._selected.action_id)}")
                dpg.configure_item(
                    "sb_sel",
                    color=theme.PENDING if self._selected.is_candidate else theme.TEXT)
        if dpg.does_item_exist("sb_dirty"):
            pending = self.pending_total()
            if self._dirty:
                dpg.set_value("sb_dirty", "Unsaved changes")
                dpg.configure_item("sb_dirty", color=theme.PENDING)
            elif pending:
                dpg.set_value("sb_dirty", f"{pending} candidates pending")
                dpg.configure_item("sb_dirty", color=theme.PENDING)
            else:
                dpg.set_value("sb_dirty", "Saved")
                dpg.configure_item("sb_dirty", color=theme.ACCEPTED)
        self._refresh_zoom_label()

    def _update_active_track_color(self) -> None:
        if not dpg.does_item_exist("active_track_color_label"):
            return
        r, g, b, _a = _track_color(self.active_track_id)
        dpg.configure_item("active_track_color_label",
                           color=(r, g, b, 255),
                           default_value=f"  Track {self.active_track_id}")

    def _on_active_tid_change(self, sender=None, app_data=None) -> None:
        self.active_track_id = max(0, int(app_data or 0))
        self.next_track_id = max(self.next_track_id, self.active_track_id)
        if dpg.does_item_exist("range_track"):
            dpg.set_value("range_track", self.active_track_id)
        self._update_active_track_color()

    def _on_list_select(self, sender=None, app_data=None) -> None:
        """行番号で対応づける (表示文字列の一致に頼らない)。"""
        if app_data is None:
            return
        rows = getattr(self, "_listbox_rows", [])
        objects = self._visible_objects(self.current_frame_idx)
        try:
            idx = rows.index(app_data)
        except ValueError:
            idx = -1
        if 0 <= idx < len(objects):
            self._select(objects[idx])

    def _on_sel_tid_change(self, sender=None, app_data=None) -> None:
        if self._selected is None:
            return
        try:
            tid = int(app_data)
        except (TypeError, ValueError):
            self._status("Track IDs must be whole numbers.")
            return
        self._push_undo([self.current_frame_idx])
        self._selected.track_id = tid
        self.next_track_id = max(self.next_track_id, tid)
        self._dirty = True
        self._after_mutation(f"Track ID set to {tid}.")

    def _on_sel_action_change(self, sender=None, app_data=None) -> None:
        if self._selected is None:
            return
        self._push_undo([self.current_frame_idx])
        if app_data in self.action_names:
            self._selected.action_id = self.action_names.index(app_data)
        elif app_data == "(none)":
            self._selected.action_id = -1
        self._dirty = True
        self._after_mutation("")

    def _on_sel_class_change(self, sender=None, app_data=None) -> None:
        if self._selected is None:
            return
        if app_data in self.class_names:
            self._push_undo([self.current_frame_idx])
            self._selected.class_id = self.class_names.index(app_data)
            self._dirty = True
            self._after_mutation("")

    def _on_sel_flags_change(self, sender=None, app_data=None) -> None:
        if self._selected is None:
            return
        if dpg.does_item_exist("sel_occluded"):
            self._selected.occluded = dpg.get_value("sel_occluded")
        if dpg.does_item_exist("sel_crowd"):
            self._selected.is_crowd = dpg.get_value("sel_crowd")
        self._dirty = True
        self._render_canvas()

    def _prompt_track_id_dialog(self) -> None:
        if self._selected is None:
            return
        if dpg.does_item_exist("tid_dlg"):
            dpg.delete_item("tid_dlg")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        current = self._selected.track_id
        with dpg.window(label="Set track ID", tag="tid_dlg", modal=True,
                        width=280, height=110,
                        pos=[max((vw - 280) // 2, 0), max((vh - 110) // 2, 0)],
                        no_resize=True):
            dpg.add_input_int(tag="tid_dlg_input", default_value=current,
                              min_value=0, max_value=9999, width=-1)
            widgets.gap(1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Set", width=110,
                               callback=self._on_track_id_dialog_confirm)
                dpg.add_button(label="Cancel", width=110,
                               callback=lambda: dpg.delete_item("tid_dlg"))

    def _on_track_id_dialog_confirm(self) -> None:
        if self._selected is None or not dpg.does_item_exist("tid_dlg_input"):
            return
        tid = dpg.get_value("tid_dlg_input")
        self._push_undo([self.current_frame_idx])
        self._selected.track_id = tid
        self.next_track_id = max(self.next_track_id, tid)
        dpg.delete_item("tid_dlg")
        self._dirty = True
        self._after_mutation(f"Track ID set to {tid}.")
