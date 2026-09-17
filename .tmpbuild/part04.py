
    # ═══════════════════════════════════════════════════════════════════
    # Main Window
    # ═══════════════════════════════════════════════════════════════════

    def _build_main_window(self) -> None:
        if dpg.does_item_exist("main_win"):
            dpg.delete_item("main_win")

        self._cap_info = self._capability()

        with dpg.window(tag="main_win", label="YOAKE Annotation Tool",
                        no_title_bar=True, no_move=True, no_resize=True,
                        no_scrollbar=True):
            self._build_menubar()
            self._build_toolbar()

            # ── horizontal split: canvas | sidebar ──────────────────
            with dpg.group(horizontal=True):
                self._build_canvas_panel()
                self._build_sidebar()

            # ── navigation bar ────────────────────────────────────────
            self._build_navbar()

        dpg.set_primary_window("main_win", True)
        self._build_key_handlers()
        self._build_mouse_handlers()

        # texture registry (created once)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                pass

        self._refresh_tool_ui()
        self._refresh_auto_panel()
        self._refresh_labels_display()
        self._refresh_status_bar()
        self._arm_pump()

    def _capability(self) -> auto_label.Capability:
        """torch / パッケージ / OpenCV トラッカの有無を 1 回だけ調べる。"""
        if self._cap_info is None:
            try:
                self._cap_info = auto_label.probe(_repo_root())
            except Exception:
                self._cap_info = auto_label.Capability(
                    torch=False, package=False, package_via="", opencv=False,
                    tracker="none",
                    torch_reason=auto_label.TORCH_MISSING_MESSAGE,
                    package_reason="", tracker_reason="",
                )
        return self._cap_info

    # ─── Menu bar ────────────────────────────────────────────────────

    def _build_menubar(self) -> None:
        def k(action: str) -> str:
            return shortcuts.label_for(action, scope="annotator")

        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label=f"Open video...  {k('file.open_video')}",
                                  callback=self._open_video_dialog)
                dpg.add_menu_item(label="Load annotation...",
                                  callback=self._load_annotation_dialog)
                dpg.add_menu_item(label="Import predictions as candidates...",
                                  callback=self._import_predictions_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Set project folder...",
                                  callback=self._set_project_dir_dialog)
                dpg.add_menu_item(label=f"Save  {k('file.save')}",
                                  callback=self.save_annotation)
                dpg.add_menu_item(label=f"Save as...  {k('file.save_as')}",
                                  callback=self._save_as_dialog)
                dpg.add_menu_item(label="Save and export frames...",
                                  callback=self._save_and_export_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="New project...",
                                  callback=self._show_setup_screen)

            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label=f"Undo  {k('edit.undo')}",
                                  callback=self.undo)
                dpg.add_menu_item(label="Redo  Ctrl+Y", callback=self.redo)
                dpg.add_separator()
                dpg.add_menu_item(label=f"Delete selected  {k('edit.delete')}",
                                  callback=self.delete_selected)
                dpg.add_menu_item(label=f"Copy from previous frame  {k('edit.propagate')}",
                                  callback=self.propagate_from_prev)

            with dpg.menu(label="Review"):
                dpg.add_menu_item(label="Auto-Annotate...  "
                                        f"{k('auto.annotate')}",
                                  callback=self.open_autolabel_dialog)
                dpg.add_menu_item(label=f"Track forward  {k('auto.track_forward')}",
                                  callback=self.track_forward_selected)
                dpg.add_separator()
                dpg.add_menu_item(label=f"Accept all candidates in frame  {k('review.accept_all')}",
                                  callback=self.accept_all_in_frame)
                dpg.add_menu_item(label=f"Discard all candidates in frame  {k('review.discard_all')}",
                                  callback=self.discard_all_in_frame)
                dpg.add_menu_item(label=f"Next frame with candidates  {k('review.next_pending_frame')}",
                                  callback=self.goto_next_pending_frame)
                dpg.add_menu_item(label=f"Previous frame with candidates  {k('review.prev_pending_frame')}",
                                  callback=self.goto_prev_pending_frame)

            with dpg.menu(label="Labels"):
                dpg.add_menu_item(label="Add class label...",
                                  callback=self._add_class_label_dialog)
                dpg.add_menu_item(label="Add action label...",
                                  callback=self._add_action_label_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Manage labels...",
                                  callback=self.open_label_manager)

            with dpg.menu(label="Help"):
                dpg.add_menu_item(label=f"Keyboard shortcuts  {k('help.shortcuts')}",
                                  callback=self._show_help)

    # ─── Toolbar ─────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        """
        画面最上部の 40px ツールバー。

        ツールボタン / 表示操作 / レビュー操作 / 保存を 1 列に並べる。ラベルと
        ツールチップのショートカット表記は shortcuts.py から生成するので、
        ヘルプ画面と食い違うことがない。
        """
        vw = dpg.get_viewport_width()
        with dpg.child_window(tag="toolbar_win", width=-1, height=42,
                              no_scrollbar=True, border=False):
            theme.bind("toolbar_win", theme.toolbar_theme())
            with dpg.group(horizontal=True, tag="toolbar"):
                self._toolbar_tool("select", "tool.select", 76)
                self._toolbar_tool("draw", "tool.draw_box", 104)
                self._toolbar_tool("obb", "tool.draw_rotated", 152)
                self._toolbar_tool("pan", "tool.pan", 62)
                self._toolbar_tool("delete", "tool.delete", 110)

                dpg.add_checkbox(label="Keep drawing", tag="tb_sticky_draw",
                                 default_value=self._sticky_draw,
                                 callback=self._on_sticky_draw_toggle)
                widgets.tooltip("Stay in the drawing tool after each box. "
                                "Off means one box, then back to Select.")

                dpg.add_text("|", color=theme.BORDER_SUBTLE)

                # ── view group ──────────────────────────────────────
                widgets.toolbar_button(
                    "Fit", lambda: self._run_action("view.fit"), tag="tb_fit",
                    shortcut=shortcuts.label_for("view.fit", scope="annotator"),
                    help_text="Fit the frame to the window", width=64)
                widgets.toolbar_button(
                    "-", lambda: self._run_action("view.zoom_out"),
                    tag="tb_zoom_out", help_text="Zoom out", width=30)
                dpg.add_text("Zoom 100%", tag="tb_zoom_label",
                             color=theme.TEXT_DIM)
                widgets.toolbar_button(
                    "+", lambda: self._run_action("view.zoom_in"),
                    tag="tb_zoom_in", help_text="Zoom in", width=30)

                dpg.add_text("|", color=theme.BORDER_SUBTLE)

                # ── review group ────────────────────────────────────
                widgets.toolbar_button(
                    "Auto-Annotate...", self.open_autolabel_dialog,
                    tag="tb_autolabel",
                    shortcut=shortcuts.label_for("auto.annotate", scope="annotator"),
                    help_text="Run the detector and bring the results back as candidates",
                    width=170)
                widgets.toolbar_button(
                    "Track forward", self.track_forward_selected, tag="tb_track",
                    shortcut=shortcuts.label_for("auto.track_forward",
                                                 scope="annotator"),
                    help_text="Propagate the selected box forward with OpenCV",
                    width=150)
                widgets.toolbar_button(
                    "Accept all", self.accept_all_in_frame, tag="tb_accept_all",
                    shortcut=shortcuts.label_for("review.accept_all",
                                                 scope="annotator"),
                    help_text="Accept every candidate in this frame", width=112)
                widgets.toolbar_button(
                    "Discard all", self.discard_all_in_frame, tag="tb_discard_all",
                    shortcut=shortcuts.label_for("review.discard_all",
                                                 scope="annotator"),
                    help_text="Discard every candidate in this frame. "
                              "Boxes you drew by hand are kept.", width=118)
                widgets.status_pill("tb_cand_pill", "No candidates", "idle")

                dpg.add_text("|", color=theme.BORDER_SUBTLE)

                # ── edit / file group ───────────────────────────────
                widgets.toolbar_button(
                    "Undo", self.undo, tag="tb_undo",
                    shortcut=shortcuts.label_for("edit.undo", scope="annotator"),
                    help_text="Undo the last change (50 steps)", width=92)
                widgets.toolbar_button(
                    "Redo", self.redo, tag="tb_redo", shortcut="Ctrl+Y",
                    help_text="Redo the change you just undid", width=92)
                widgets.toolbar_button(
                    "Copy from previous", self.propagate_from_prev,
                    tag="tb_propagate",
                    shortcut=shortcuts.label_for("edit.propagate",
                                                 scope="annotator"),
                    help_text="Copy every box from the previous frame into this one",
                    width=190)
                widgets.toolbar_button(
                    "Interpolate", self._apply_range_ui, tag="tb_interpolate",
                    help_text="Apply the range below: fill the frames between "
                              "your keyframes for this track", width=120)
                widgets.toolbar_button(
                    "Save", self.save_annotation, tag="tb_save",
                    shortcut=shortcuts.label_for("file.save", scope="annotator"),
                    help_text="Write annotations.json", width=100)
                widgets.toolbar_button(
                    "Shortcuts", self._show_help, tag="tb_help",
                    shortcut=shortcuts.label_for("help.shortcuts",
                                                 scope="annotator"),
                    help_text="Show the full keyboard and mouse reference",
                    width=118)

        with dpg.group(horizontal=True):
            dpg.add_text("", tag="tb_tool_chip", color=theme.TEXT_DIM)

    def _toolbar_tool(self, tool: str, action: str, width: int) -> None:
        widgets.toolbar_button(
            _TOOL_NAMES[tool], (lambda s=None, a=None, u=None, t=tool: self.set_tool(t)),
            tag=_TOOL_TAGS[tool],
            shortcut=shortcuts.label_for(action, scope="annotator"),
            help_text=shortcuts.help_for(action, scope="annotator"),
            active=(self._tool == tool), width=width)

    # ─── Canvas panel ────────────────────────────────────────────────

    def _build_canvas_panel(self) -> None:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        cw = max(vw - _SIDEBAR_W - 20, 400)
        ch = max(vh - _CHROME_H, 300)

        with dpg.child_window(tag="canvas_win", width=cw, height=ch,
                              no_scrollbar=True, border=False):
            dpg.add_drawlist(tag="canvas", width=cw, height=ch)
