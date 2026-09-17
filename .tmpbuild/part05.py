
    # ─── Sidebar ─────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        vh = dpg.get_viewport_height()
        sh = max(vh - _CHROME_H, 300)
        wrap = _SIDEBAR_W - 26

        with dpg.child_window(tag="sidebar_win", width=_SIDEBAR_W, height=sh,
                              no_scrollbar=False, border=True):
            dpg.add_text("YOAKE Annotator", color=theme.ACCENT)
            theme.bind_h2(dpg.last_item())
            dpg.add_text("Draw, track, and label behavior", color=theme.TEXT_DIM)
            dpg.add_separator()
            widgets.gap(1)

            # 1 ── Quick start ────────────────────────────────────────
            with dpg.collapsing_header(label="Quick start", default_open=True):
                dpg.add_text(
                    "1. Open a video (Ctrl+O).\n"
                    "2. Press W, then drag on the image to draw a box.\n"
                    "3. Press T to track it forward, then Y to accept the result.\n"
                    "4. A and D move between frames. Ctrl+S saves.",
                    color=theme.TEXT_DIM, wrap=wrap)
                widgets.gap(1)
                dpg.add_text("Tip: press F1 for the full shortcut list.",
                             color=theme.TEXT_FAINT, wrap=wrap)

            widgets.gap(1)

            # 2 ── Active track ───────────────────────────────────────
            with dpg.collapsing_header(label="Active track", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:")
                    dpg.add_input_int(tag="active_tid", default_value=1, width=70,
                                      min_value=0, max_value=9999,
                                      callback=self._on_active_tid_change)
                    dpg.add_button(label="New track", width=92,
                                   callback=self.new_track)
                    widgets.tooltip("Start a new track ID "
                                    f"({shortcuts.label_for('edit.new_track', scope='annotator')})")
                widgets.gap(0)
                dpg.add_text("", tag="active_track_color_label")

            widgets.gap(1)

            # 3 ── Auto-Annotate ──────────────────────────────────────
            with dpg.collapsing_header(label="Auto-Annotate", default_open=True):
                dpg.add_text("", tag="al_cap_line", color=theme.TEXT_DIM, wrap=wrap)
                dpg.add_text("", tag="al_cap_detail", color=theme.TEXT_FAINT,
                             wrap=wrap)
                widgets.gap(1)
                dpg.add_button(label="Auto-Annotate... (Ctrl+R)",
                               tag="al_run_btn", width=-1, height=30,
                               callback=self.open_autolabel_dialog)
                dpg.add_button(label="Import predictions as candidates...",
                               tag="al_import_btn", width=-1, height=28,
                               callback=self._import_predictions_dialog)
                widgets.tooltip("Load a predictions.json produced anywhere - "
                                "including on another machine with PyTorch - and "
                                "review it here. Nothing is saved until you accept it.")
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                dpg.add_text("Track forward from this frame", color=theme.TEXT)
                with dpg.group(horizontal=True):
                    dpg.add_text("Frames:")
                    dpg.add_input_int(tag="al_track_frames", default_value=30,
                                      width=70, min_value=1, max_value=100000)
                    dpg.add_text("", tag="al_tracker_name", color=theme.TEXT_DIM)
                dpg.add_button(label="Track forward (T)", tag="al_track_btn",
                               width=-1, height=28,
                               callback=self.track_forward_selected)
                dpg.add_text("", tag="al_tracker_note", color=theme.TEXT_FAINT,
                             wrap=wrap)
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                dpg.add_text("", tag="al_review_line", color=theme.PENDING,
                             wrap=wrap)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Accept all (Y)", width=(_SIDEBAR_W - 40) // 2,
                                   tag="al_accept_btn",
                                   callback=self.accept_all_in_frame)
                    dpg.add_button(label="Discard all (N)",
                                   width=(_SIDEBAR_W - 40) // 2,
                                   tag="al_discard_btn",
                                   callback=self.discard_all_in_frame)
                theme.bind("al_discard_btn", theme.danger_theme())
                widgets.gap(1)
                dpg.add_text("", tag="al_job_line", color=theme.TEXT_DIM, wrap=wrap)
                dpg.add_button(label="Stop", tag="al_stop_btn", width=-1,
                               height=26, show=False, callback=self.stop_job)

            widgets.gap(1)

            # 4 ── Range labeling ─────────────────────────────────────
            with dpg.collapsing_header(label="Range labeling", default_open=True):
                dpg.add_text(
                    "Select a track, choose a frame span, and apply track or "
                    "action changes across that range. Ctrl+Z can undo this.",
                    color=theme.TEXT_DIM, wrap=wrap)
                with dpg.group(horizontal=True):
                    dpg.add_text("Start:", indent=4)
                    dpg.add_input_int(tag="range_start", default_value=0, width=70,
                                      min_value=0, max_value=0)
                    dpg.add_button(label="Use current", width=94,
                                   callback=lambda: dpg.set_value(
                                       "range_start", self.current_frame_idx))
                with dpg.group(horizontal=True):
                    dpg.add_text("End:", indent=4)
                    dpg.add_input_int(tag="range_end", default_value=0, width=70,
                                      min_value=0, max_value=0)
                    dpg.add_button(label="Use current", width=94,
                                   callback=lambda: dpg.set_value(
                                       "range_end", self.current_frame_idx))
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:", indent=4)
                    dpg.add_input_int(tag="range_track", default_value=1, width=80,
                                      min_value=0, max_value=9999)
                with dpg.group(horizontal=True):
                    dpg.add_text("Action:", indent=4)
                    dpg.add_combo(items=["(no change)"], tag="range_action",
                                  default_value="(no change)", width=-1)
                dpg.add_checkbox(label="Interpolate boxes", tag="range_interp",
                                 default_value=True)
                widgets.tooltip("Fill the frames between your keyframes by "
                                "moving the box linearly. Filled boxes are drawn "
                                "dashed until you correct one.")
                dpg.add_button(label="Apply to range", width=-1,
                               height=30, callback=self._apply_range_ui)
                dpg.add_button(label="Remove track from range", width=-1,
                               tag="range_clear_btn", callback=self._clear_range_ui)
                theme.bind("range_clear_btn", theme.danger_theme())

            widgets.gap(1)

            # 5 ── Objects in current frame ───────────────────────────
            with dpg.collapsing_header(label="Objects in current frame",
                                       default_open=True):
                dpg.add_listbox(items=[], tag="obj_listbox", num_items=6,
                                width=-1, callback=self._on_list_select)
                dpg.add_text("Marks: accepted = *, candidate = ?, "
                             "interpolated = ~", color=theme.TEXT_FAINT,
                             wrap=wrap)

            widgets.gap(1)

            # 6 ── Selected object ────────────────────────────────────
            with dpg.collapsing_header(label="Selected object", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:", indent=4)
                    dpg.add_input_text(tag="sel_tid", width=60,
                                       callback=self._on_sel_tid_change,
                                       on_enter=True)
                widgets.gap(0)
                dpg.add_text("Class:", indent=4)
                dpg.add_combo(items=["(none)"], tag="sel_class", width=-1,
                              callback=self._on_sel_class_change)
                widgets.gap(0)
                dpg.add_text("Action:", indent=4)
                dpg.add_combo(items=["(none)"], tag="sel_action", width=-1,
                              callback=self._on_sel_action_change)
                widgets.gap(0)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Occluded", tag="sel_occluded",
                                     callback=self._on_sel_flags_change)
                    dpg.add_checkbox(label="Is crowd", tag="sel_crowd",
                                     callback=self._on_sel_flags_change)
                widgets.gap(0)
                dpg.add_text("Source: -", tag="sel_source_text",
                             color=theme.TEXT_DIM, wrap=wrap)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="To rotated (OBB)", width=(_SIDEBAR_W - 40) // 2,
                                   callback=self.convert_selected_to_obb)
                    dpg.add_button(label="To axis-aligned",
                                   width=(_SIDEBAR_W - 40) // 2,
                                   callback=self.convert_selected_to_aabb)

            widgets.gap(1)

            # 7 ── Actions (0-9) ──────────────────────────────────────
            with dpg.collapsing_header(label="Actions (0-9)", default_open=True):
                dpg.add_group(tag="action_buttons_group")

            widgets.gap(1)

            # 8 ── Tools ──────────────────────────────────────────────
            with dpg.collapsing_header(label="Tools", default_open=False):
                dpg.add_text("Active tool: Select", tag="tool_mode_text",
                             color=theme.TEXT_DIM)
                widgets.gap(1)
                dpg.add_checkbox(label="Keep drawing after each box",
                                 tag="sticky_draw_cb",
                                 callback=self._on_sticky_draw_toggle)
                dpg.add_text(
                    "Press W to start a new box anywhere - even on top of an "
                    "existing one. Drag on the image, then release to create it.",
                    color=theme.TEXT_FAINT, wrap=wrap)
                widgets.gap(1)
                dpg.add_text("Color boxes by", color=theme.TEXT)
                dpg.add_combo(items=["track", "class", "action"],
                              default_value="track", tag="color_by_combo",
                              width=-1, callback=self._on_color_by_change)
                widgets.tooltip("Which label decides a box outline color. "
                                "Display only - saved labels are unaffected.")

            widgets.gap(1)

            # 9 ── Auto-track ─────────────────────────────────────────
            with dpg.collapsing_header(label="Auto-track", default_open=False):
                dpg.add_checkbox(label="Enable auto-track mode (Ctrl+T)",
                                 tag="auto_track_cb",
                                 callback=self._on_auto_track_toggle)
                dpg.add_text(
                    "When on, stepping to the next frame moves every initialized "
                    "tracker forward. Tracked boxes appear dashed - correct one "
                    "and it becomes a keyframe again.",
                    color=theme.TEXT_FAINT, wrap=wrap)
                widgets.gap(1)
                dpg.add_text("", tag="auto_track_backend", color=theme.TEXT_DIM,
                             wrap=wrap)
                dpg.add_button(label="Reset all trackers",
                               width=-1, callback=self._reset_all_trackers)
                dpg.add_button(label="Reset selected tracker",
                               width=-1, callback=self._reset_selected_tracker)

            widgets.gap(1)

            # 10 ── Labels ────────────────────────────────────────────
            with dpg.collapsing_header(label="Labels", default_open=False):
                dpg.add_text("Classes", color=theme.ACCENT)
                dpg.add_text("", tag="classes_display", wrap=wrap)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add class", width=(_SIDEBAR_W - 40) // 2,
                                   callback=self._add_class_label_dialog)
                    dpg.add_button(label="Manage...", width=(_SIDEBAR_W - 40) // 2,
                                   callback=self.open_label_manager)
                widgets.gap(1)
                dpg.add_text("Actions", color=theme.ACCENT)
                dpg.add_text("", tag="actions_display", wrap=wrap)
                dpg.add_button(label="Add action label", width=-1,
                               callback=self._add_action_label_dialog)

            widgets.gap(1)

            # 11 ── Frame info ────────────────────────────────────────
            with dpg.collapsing_header(label="Frame info", default_open=True):
                dpg.add_text("", tag="frame_info_text", color=theme.TEXT_DIM)
                widgets.gap(1)
                dpg.add_text("Project folder", color=theme.ACCENT)
                dpg.add_text("(not set)", tag="project_dir_text",
                             color=theme.TEXT_FAINT, wrap=wrap)
                dpg.add_button(label="Change...", width=-1,
                               callback=self._set_project_dir_dialog)

    def _update_action_buttons(self) -> None:
        if not dpg.does_item_exist("action_buttons_group"):
            return
        dpg.delete_item("action_buttons_group", children_only=True)
        for i, name in enumerate(self.action_names):
            col = self.action_color(i)
            key = f"[{i}] " if i <= 9 else ""
            btn = dpg.add_button(label=f"{key}{name}", width=-1, height=26,
                                 parent="action_buttons_group",
                                 callback=lambda s, a, u: self.set_action(u),
                                 user_data=i)
            theme.bind(btn, theme.chip_theme(col))

        # Update combos that list actions / classes
        if dpg.does_item_exist("sel_action"):
            dpg.configure_item("sel_action",
                               items=["(none)"] + list(self.action_names))
        if dpg.does_item_exist("sel_class"):
            dpg.configure_item("sel_class",
                               items=(list(self.class_names) or ["(none)"]))
        if dpg.does_item_exist("range_action"):
            dpg.configure_item("range_action",
                               items=["(no change)", "(none)"] + list(self.action_names))

    # ─── Navigation bar + status bar ─────────────────────────────────

    def _build_navbar(self) -> None:
        with dpg.group(horizontal=True, tag="navbar"):
            dpg.add_button(label="First", width=54,
                           callback=lambda: self.goto_frame(0))
            dpg.add_button(label="Prev", width=52,
                           callback=lambda: self.goto_frame(self.current_frame_idx - 1))
            dpg.add_text("Frame:")
            dpg.add_input_int(tag="frame_input", default_value=0, width=80,
                              min_value=0, max_value=0,
                              callback=lambda s, a: self.goto_frame(a),
                              on_enter=True)
            dpg.add_text("/ 0", tag="total_frames_text")
            dpg.add_button(label="Next", width=52,
                           callback=lambda: self.goto_frame(self.current_frame_idx + 1))
            dpg.add_button(label="Last", width=54,
                           callback=lambda: self.goto_frame(self.num_frames - 1))
            dpg.add_spacer(width=10)
            dpg.add_button(label="< Candidates", width=118,
                           callback=self.goto_prev_pending_frame)
            widgets.tooltip("Jump to the previous frame that still has candidates "
                            f"({shortcuts.label_for('review.prev_pending_frame', scope='annotator')})")
            dpg.add_button(label="Candidates >", width=118,
                           callback=self.goto_next_pending_frame)
            widgets.tooltip("Jump to the next frame that still has candidates "
                            f"({shortcuts.label_for('review.next_pending_frame', scope='annotator')})")
            dpg.add_spacer(width=10)
            dpg.add_progress_bar(tag="progress_bar", default_value=0.0,
                                 width=-10, height=16)

        # Persistent status bar: position | tool | selection | unsaved | result
        with dpg.group(horizontal=True, tag="statusbar"):
            dpg.add_text("Frame -/-", tag="sb_frame", color=theme.TEXT)
            dpg.add_text("|", color=theme.BORDER_SUBTLE)
            dpg.add_text("Select", tag="sb_tool", color=theme.ACCENT)
            dpg.add_text("|", color=theme.BORDER_SUBTLE)
            dpg.add_text("No selection", tag="sb_sel", color=theme.TEXT_DIM)
            dpg.add_text("|", color=theme.BORDER_SUBTLE)
            dpg.add_text("Saved", tag="sb_dirty", color=theme.ACCEPTED)
            dpg.add_text("|", color=theme.BORDER_SUBTLE)
            dpg.add_text("", tag="status_text", color=theme.TEXT_DIM)
