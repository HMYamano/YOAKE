
    # ─── Mouse handlers ──────────────────────────────────────────────

    def _build_mouse_handlers(self) -> None:
        # dearpygui 2.x に add_mouse_press_handler は無い。押下相当は
        # add_mouse_click_handler (押した瞬間に発火) を使う。
        if dpg.does_item_exist("mouse_handlers"):
            dpg.delete_item("mouse_handlers")
        with dpg.handler_registry(tag="mouse_handlers"):
            dpg.add_mouse_click_handler(button=0, callback=self._on_mouse_press)
            dpg.add_mouse_drag_handler(button=0, threshold=1.0,
                                       callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=0,
                                          callback=self._on_mouse_release)
            dpg.add_mouse_click_handler(button=1, callback=self._on_right_click)
            # 中ボタンドラッグはどのツールでもパンになる (常に使えるジェスチャ)
            dpg.add_mouse_click_handler(button=2, callback=self._on_middle_press)
            dpg.add_mouse_drag_handler(button=2, threshold=1.0,
                                       callback=self._on_middle_drag)
            dpg.add_mouse_release_handler(button=2,
                                          callback=self._on_middle_release)
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)

    # ─── Key handlers ────────────────────────────────────────────────

    def _build_action_table(self) -> None:
        """action id -> 実行する関数。shortcuts.actions() の全項目を必ず埋める。"""
        a: Dict[str, Callable[[], None]] = {
            "tool.select": lambda: self.set_tool("select"),
            "tool.draw_box": self.toggle_draw_tool,
            "tool.draw_rotated": lambda: self.toggle_tool("obb"),
            "tool.pan": lambda: self.toggle_tool("pan"),
            "tool.delete": lambda: self.toggle_tool("delete"),
            "tool.cancel": self._escape,

            "view.fit": self.fit_view,
            "view.zoom_in": lambda: self.zoom_by(1.25),
            "view.zoom_out": lambda: self.zoom_by(0.8),

            "nav.prev": lambda: self.goto_frame(self.current_frame_idx - 1),
            "nav.next": lambda: self.goto_frame(self.current_frame_idx + 1),
            "nav.first": lambda: self.goto_frame(0),
            "nav.last": lambda: self.goto_frame(self.num_frames - 1),

            "review.next_pending_frame": self.goto_next_pending_frame,
            "review.prev_pending_frame": self.goto_prev_pending_frame,
            "review.cycle": self.cycle_selection,
            "review.accept_all": self.accept_all_in_frame,
            "review.discard_all": self.discard_all_in_frame,

            "auto.track_forward": self.track_forward_selected,
            "auto.annotate": self.open_autolabel_dialog,
            "auto.toggle_autotrack": self._toggle_auto_track_key,

            "edit.delete": self.delete_selected,
            "edit.propagate": self.propagate_from_prev,
            "edit.undo": self.undo,
            "edit.redo": self.redo,
            "edit.new_track": self.new_track,

            "obb.rotate_ccw": lambda: self.rotate_selected(-5),
            "obb.rotate_cw": lambda: self.rotate_selected(5),

            "file.save": self.save_annotation,
            "file.save_as": self._save_as_dialog,
            "file.open_video": self._open_video_dialog,
            "help.shortcuts": self._show_help,
        }
        for i in range(10):
            a[f"action.set_{i}"] = (lambda idx=i: self.set_action(idx))
        self._ACTIONS = a

        missing = [act for act in shortcuts.actions("annotator") if act not in a]
        if missing:
            # 未実装のバインドは「押しても何も起きないキー」になるので起動時に落とす。
            raise AssertionError(
                "Shortcut actions without an implementation: " + ", ".join(missing))

    def _binding_entries(self) -> Dict[str, List[Tuple[Tuple[str, ...], str]]]:
        """key suffix -> [(mods, action), ...]  (共有テーブル + アノテータ拡張)。"""
        entries: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
        for b in shortcuts.for_scope("annotator"):
            entries.setdefault(b.key, []).append((tuple(b.mods), b.action))
        for action, key, mods, _label, _help in _EXTRA_BINDINGS:
            entries.setdefault(key, []).append((tuple(mods), action))
        return entries

    def _build_key_handlers(self) -> None:
        """
        キーバインドは gui/shortcuts.py が唯一の定義元。

        1 つのキーに修飾違いの複数バインドがぶら下がる (R / Ctrl+R, T / Ctrl+T,
        N / Ctrl+N, Right / Ctrl+Right, Ctrl+S / Ctrl+Shift+S) ので、
        **キーごとに 1 ハンドラ**を登録し、押下時に実際に押さえている修飾と
        完全一致するバインドだけを実行する。
        """
        self._build_action_table()
        if dpg.does_item_exist("key_handlers"):
            dpg.delete_item("key_handlers")

        entries = self._binding_entries()
        with dpg.handler_registry(tag="key_handlers"):
            for key, binds in entries.items():
                code = self._resolve_key(key)
                if code is None:
                    continue
                dpg.add_key_press_handler(code, callback=self._make_key_cb(binds))

    @staticmethod
    def _resolve_key(key: str) -> Optional[int]:
        """shortcuts.resolve() 経由で dpg の定数へ。存在しない綴りは None。"""
        probe = shortcuts.Binding(action="_probe", key=key, mods=(),
                                  scope="annotator", group="Tools")
        try:
            return shortcuts.resolve(dpg, probe)
        except AttributeError:
            return None

    def _make_key_cb(self, binds: List[Tuple[Tuple[str, ...], str]]):
        def cb(sender=None, app_data=None, user_data=None):
            held = set()
            try:
                if dpg.is_key_down(_KEY_CTRL):
                    held.add("Ctrl")
                if dpg.is_key_down(_KEY_SHIFT):
                    held.add("Shift")
            except Exception:
                pass
            action = None
            for mods, act in binds:
                if set(mods) == held:
                    action = act
                    break
            if action is None:
                return
            if action not in shortcuts.ALWAYS_ALLOWED and self._typing():
                return
            fn = self._ACTIONS.get(action)
            if fn is None:
                return
            try:
                fn()
            except Exception as exc:
                self._status(f"{action} failed: {exc}")
        return cb

    def _typing(self) -> bool:
        """
        入力欄を編集中か。

        ``is_item_focused`` ではなく ``is_item_active`` を見る。ウィンドウ内の
        最初の入力欄は自動でフォーカスを持つので、focus だけを見ると「常に入力中」
        になり、すべてのショートカットが死ぬ。
        """
        try:
            focused = dpg.get_focused_item()
        except Exception:
            return False
        if not focused:
            return False
        try:
            kind = dpg.get_item_type(focused)
        except Exception:
            return False
        if not any(kind.endswith(k) for k in shortcuts.EDITING_TYPES):
            return False
        try:
            return bool(dpg.is_item_active(focused))
        except Exception:
            return False

    def _run_action(self, action: str) -> None:
        fn = self._ACTIONS.get(action)
        if fn is not None:
            fn()
