
    # ═══════════════════════════════════════════════════════════════════
    # File I/O
    # ═══════════════════════════════════════════════════════════════════

    def save_annotation(self) -> None:
        if self.cap is None:
            self._status("Open a video first.")
            return
        # プロジェクトディレクトリが指定されていれば、静かに annotations.json を上書き。
        if self.project_dir is not None:
            path = self.project_dir / "annotations.json"
            self._write_annotation(path)
            return
        # 未設定なら従来通りダイアログで保存先を選ばせる。
        initial = (self.annotation_path.name
                   if self.annotation_path else "annotations.json")
        path = _tk_save_file(
            title="Save Annotation",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _save_as_dialog(self) -> None:
        """プロジェクトディレクトリを設定していても、明示的にダイアログで保存先を選ぶ。"""
        if self.cap is None:
            self._status("Open a video first.")
            return
        initial = (self.annotation_path.name
                   if self.annotation_path else "annotations.json")
        path = _tk_save_file(
            title="Save Annotation As",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _set_project_dir_dialog(self) -> None:
        """メニューからプロジェクト保存ディレクトリを変更する。"""
        p = _tk_ask_dir(title="Select Project Directory")
        if p is None:
            return
        self.project_dir = p
        self._refresh_project_dir_display()
        self._status(f"Project folder set to {p}")

    def _refresh_project_dir_display(self) -> None:
        if dpg.does_item_exist("project_dir_text"):
            dpg.set_value(
                "project_dir_text",
                str(self.project_dir) if self.project_dir else "(not set)",
            )

    def _write_annotation(self, path: Path) -> None:
        """
        確定済み (accepted) の box だけを書き出す。

        レビュー待ちの候補はディスクへ出さない。これにより検出のやり直しは常に
        安全で、未レビューの実行が学習データを汚すことがない。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        video_id = self.video_path.stem if self.video_path else "video_001"

        pending = self.pending_total()
        frames = []
        saved_boxes = 0
        for fi in sorted(self.annotations.keys()):
            objs = [o for o in self.annotations[fi] if o.accepted]
            if not objs:
                continue
            saved_boxes += len(objs)
            frames.append({
                "frame_index": fi,
                "image_path": f"images/{video_id}/{fi:06d}.jpg",
                "objects": [o.to_dict() for o in objs],
            })

        # OBB を 1 つでも含むなら v2.0、含まなければ v1.1 のまま書く。
        # OBB を使っていないプロジェクトの出力は旧ツールでもそのまま読める。
        has_obb = any(
            o.has_obb for objs in self.annotations.values()
            for o in objs if o.accepted
        )
        data = {
            "meta": {
                "version": "2.0" if has_obb else "1.1",
                "description": f"Annotated with YOAKE Annotation Tool - {self.project_name}",
                "created": datetime.now().isoformat(timespec="seconds"),
                "fps_default": self.fps,
                "image_root": "",
            },
            "class_names": list(self.class_names),
            "action_names": list(self.action_names),
            "videos": [{
                "video_id": video_id,
                "fps": self.fps,
                "width": self.frame_w,
                "height": self.frame_h,
                "num_frames": self.num_frames,
                "frames": frames,
            }],
        }
        # 既定色から動かしたものだけ書く (触っていないプロジェクトは従来と同じ形)
        if self.class_colors:
            data["class_colors"] = {str(k): list(v)
                                    for k, v in sorted(self.class_colors.items())}
        if self.action_colors:
            data["action_colors"] = {str(k): list(v)
                                     for k, v in sorted(self.action_colors.items())}

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            self._status(f"Save error: {exc}")
            return

        self.annotation_path = path
        self._dirty = False
        self._refresh_status_bar()
        msg = f"Saved {saved_boxes} boxes across {len(frames)} frames to {path.name}."
        if pending:
            msg += (f" {pending} candidates are still pending and were not saved "
                    "as labels. Press Y to accept them, or N to discard them.")
        self._status(msg)
        widgets.toast(msg, kind="ok" if not pending else "warn")

    def _save_and_export_dialog(self) -> None:
        if self.cap is None:
            self._status("Open a video first.")
            return
        # プロジェクトディレクトリが設定されていればそこへ出力、無ければダイアログ。
        out = self.project_dir
        if out is None:
            out = _tk_ask_dir(title="Select Output Directory")
            if out is None:
                return
        video_id = self.video_path.stem if self.video_path else "video_001"
        img_dir = out / "images" / video_id
        img_dir.mkdir(parents=True, exist_ok=True)

        self._write_annotation(out / "annotations.json")

        frame_indices = sorted(fi for fi, objs in self.annotations.items()
                               if any(o.accepted for o in objs))
        self._export_frames_original(frame_indices, img_dir)
        self._status(f"Exported {len(frame_indices)} frames to {img_dir}")

    def _export_frames_original(self, frames: List[int], img_dir: Path) -> None:
        """元のフレーム番号のまま JPEG 出力する (annotations.json と対応)。"""
        if not frames:
            return
        wanted = set(int(f) for f in frames)
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            idx, last = 0, max(wanted)
            while idx <= last:
                if not cap.grab():
                    break
                if idx in wanted:
                    ok, frame = cap.retrieve()
                    if ok:
                        cv2.imwrite(str(img_dir / f"{idx:06d}.jpg"), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                idx += 1
        finally:
            cap.release()

    def _load_annotation_dialog(self) -> None:
        path = _tk_open_file(
            title="Load Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is not None:
            self._load_annotation_file(path)

    def _load_annotation_file(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._status(f"Load error: {exc}")
            return

        # Restore labels from JSON
        if "class_names" in data:
            self.class_names = list(data["class_names"])
        if "action_names" in data:
            self.action_names = list(data["action_names"])
        for key, store in (("class_colors", self.class_colors),
                           ("action_colors", self.action_colors)):
            store.clear()
            for k, v in (data.get(key) or {}).items():
                try:
                    store[int(k)] = tuple(int(c) for c in v[:3])
                except (TypeError, ValueError):
                    continue
        if dpg.does_item_exist("action_buttons_group"):
            self._update_action_buttons()

        self.annotations = {}
        for video in data.get("videos", []):
            for frame in video.get("frames", []):
                fi = frame["frame_index"]
                self.annotations[fi] = [
                    AnnObject.from_dict(o) for o in frame.get("objects", [])
                ]

        max_tid = max(
            (obj.track_id for objs in self.annotations.values()
             for obj in objs if obj.track_id >= 0),
            default=0,
        )
        self.next_track_id = max_tid + 1
        self.active_track_id = self.next_track_id
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.next_track_id)

        self._undo.clear()
        self._redo.clear()
        self._dirty = False
        self._after_mutation(f"Loaded {path.name}.")
        self.annotation_path = path
        # 読み込んだ JSON の親ディレクトリを既定のプロジェクトディレクトリにする。
        # (既に手動で別ディレクトリを設定していた場合は上書きしない。)
        if self.project_dir is None:
            self.project_dir = path.parent
            self._refresh_project_dir_display()

    # ═══════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _status(self, msg: str) -> None:
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", msg)

    def _ctrl_s_handler(self) -> None:
        """後方互換のためのフック (キー配線は shortcuts.py から生成される)。"""
        if dpg.is_key_down(_KEY_CTRL):
            self.save_annotation()

    # ─── Help / shortcuts window ─────────────────────────────────────

    def _changed_note(self) -> str:
        rows = shortcuts.changed_rows()
        lines = ["Changed in this version:"]
        for was, now, why in rows:
            lines.append(f"  - {was}  ->  {now}   ({why})")
        lines.append("Unchanged on purpose: " + "; ".join(
            shortcuts.UNCHANGED_ON_PURPOSE))
        return "\n".join(lines)

    def _show_help(self) -> None:
        if dpg.does_item_exist("help_win"):
            dpg.delete_item("help_win")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        w, h = 720, 640
        with dpg.window(label="Keyboard shortcuts", tag="help_win",
                        width=w, height=h, modal=False,
                        pos=[max((vw - w) // 2, 0), max((vh - h) // 2, 0)]):
            with dpg.child_window(height=h - 96, border=False):
                widgets.shortcut_table(shortcuts.help_groups("annotator"),
                                       note=self._changed_note())
                widgets.section_header(
                    "Annotator extras",
                    "Bindings that live only in the annotator.")
                widgets.kbd_table([(label, helptext)
                                   for _a, _k, _m, label, helptext
                                   in _EXTRA_BINDINGS])
                widgets.section_header("Mouse")
                widgets.kbd_table([
                    ("Drag", "Draw a box (Draw Box / Draw Rotated Box), or move "
                             "and resize the selected box in Select"),
                    ("Click a box", "Select it. A candidate is accepted by the click"),
                    ("Right-click a box", "Open the context menu. On a candidate, "
                                          "discard it immediately"),
                    ("Drag a corner dot", "Resize the box"),
                    ("Drag the yellow dot", "Turn a rotated box"),
                    ("Wheel", "Zoom at the cursor"),
                    ("Middle-drag", "Pan, from any tool"),
                ])
                widgets.section_header(
                    "Candidates",
                    "A candidate is a machine proposal that has not been "
                    "confirmed. Candidates are drawn dashed in amber and are "
                    "never written to disk. Accept one by clicking it, a whole "
                    "frame with Y, or discard a frame with N.")
                widgets.section_header(
                    "Rotated boxes (OBB)",
                    "Press E to draw a rotated box, or convert an existing box "
                    "from the right-click menu. Drag the yellow handle, or use "
                    "[ and ] for five degrees at a time. Projects without any "
                    "rotated box are still saved as schema v1.1.")
            widgets.gap(2)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Don't show this again",
                                 tag="help_dont_show",
                                 default_value=self._help_seen,
                                 callback=self._on_help_dont_show)
                dpg.add_button(label="Close", width=140,
                               callback=lambda: dpg.delete_item("help_win"))

    def _on_help_dont_show(self, sender=None, app_data=None) -> None:
        self._help_seen = bool(app_data)

    def _maybe_show_help_once(self) -> None:
        """キー配置が変わったので、プロジェクトにつき 1 回だけ自動で開く。"""
        if not self._help_seen:
            self._help_seen = True
            self._show_help()

    def _on_viewport_resize(self) -> None:
        """Called when the viewport is resized - update canvas and sidebar sizes."""
        if not dpg.does_item_exist("canvas_win"):
            return
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        cw = max(vw - _SIDEBAR_W - 20, 400)
        ch = max(vh - _CHROME_H, 300)
        dpg.configure_item("canvas_win", width=cw, height=ch)
        dpg.configure_item("canvas", width=cw, height=ch)
        if dpg.does_item_exist("sidebar_win"):
            dpg.configure_item("sidebar_win", height=ch)
        self._render_canvas()

    # ═══════════════════════════════════════════════════════════════════
    # Run
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_viewport(self, show: bool = True) -> None:
        dpg.create_viewport(title="YOAKE Annotation Tool",
                            width=1420, height=860,
                            min_width=980, min_height=640)
        dpg.setup_dearpygui()
        if show:
            dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

    def run(self, video: Optional[Path] = None,
            load_json: Optional[Path] = None,
            import_predictions: Optional[Path] = None) -> None:
        self._prepare_viewport()

        # Show project setup screen first
        self._show_setup_screen()

        # Handle CLI arguments after first render
        if video or load_json or import_predictions:
            def _deferred():
                if dpg.does_item_exist("setup_win"):
                    # auto-confirm setup with defaults
                    if not self.class_names:
                        self.class_names = list(_DEFAULT_CLASSES)
                    if not self.action_names:
                        self.action_names = list(_DEFAULT_ACTIONS)
                    dpg.delete_item("setup_win")
                    self._build_main_window()
                    self._update_action_buttons()
                if video:
                    self._load_video(video)
                if load_json:
                    self._load_annotation_file(load_json)
                if import_predictions:
                    self.import_predictions(import_predictions)
            dpg.set_frame_callback(2, _deferred)

        dpg.start_dearpygui()
        self._shutdown()

    def _shutdown(self) -> None:
        self._pump_armed = False
        self._track_stop = True
        if self._job is not None:
            self._force_kill(self._job)
            self._job = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        dpg.destroy_context()
        try:
            theme.reset_cache()
        except Exception:
            pass
