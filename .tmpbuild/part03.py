

# ─── Application ──────────────────────────────────────────────────────────────

class AnnotationApp:

    def __init__(self) -> None:
        # ── project config ───────────────────────────────────────────
        self.class_names: List[str] = []
        self.action_names: List[str] = []
        self.project_name: str = "untitled"
        # ラベルごとの表示色。既定 (パレット順) から動かしたものだけ保存する。
        self.class_colors: Dict[int, Tuple[int, int, int]] = {}
        self.action_colors: Dict[int, Tuple[int, int, int]] = {}
        self._color_by: str = "track"        # track | class | action

        # ── video state ──────────────────────────────────────────────
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.num_frames: int = 0
        self.fps: float = 25.0
        self.frame_w: int = 640
        self.frame_h: int = 640
        self.current_frame_idx: int = 0
        self._current_bgr: Optional[np.ndarray] = None

        # ── annotation state ─────────────────────────────────────────
        self.annotations: Dict[int, List[AnnObject]] = {}
        self.next_track_id: int = 1
        self.active_track_id: int = 1
        self._dirty: bool = False

        # ── range labeling state ─────────────────────────────────────
        self.range_start: int = 0
        self.range_end: int = 0

        # ── canvas interaction state ──────────────────────────────────
        self._mode: str = "idle"        # idle | draw | move | resize | rotate | pan
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_end: Optional[Tuple[float, float]] = None
        self._selected: Optional[AnnObject] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_bbox_orig: Optional[List[int]] = None
        self._drag_obb_orig: Optional[List[float]] = None
        self._resize_handle: Optional[str] = None
        self._right_click_obj: Optional[AnnObject] = None
        # OBB モード: 新規に描く箱を回転矩形として作るか (Draw Rotated Box ツール)
        self._obb_mode: bool = False
        # ツールモード (LabelImg 準拠):
        #   "select" = 既存ボックスの選択 / 移動 / リサイズ
        #   "draw"   = クリック&ドラッグで必ず新規ボックスを描く (W キー)
        #   "obb"    = 同上だが回転矩形として作る (E / R キー)
        #   "pan"    = 画面を動かす (H キー)
        #   "delete" = クリックした箱を消す (X キー、粘着)
        # 既存ボックスの中や上に重ねて描けるようにするためのモード分離。
        self._tool: str = "select"
        # LabelImg は 1 個描くと編集モードへ戻る。連続で描きたい場合のための粘着指定。
        self._sticky_draw: bool = False
        self._hover_obj: Optional[AnnObject] = None
        self._cursor_canvas: Optional[Tuple[float, float]] = None
        self._moved: bool = False

        # ── display geometry ──────────────────────────────────────────
        self._scale: float = 1.0
        self._fit_scale: float = 1.0
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._pan_origin: Optional[Tuple[float, float, float, float]] = None
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._canvas_w: int = 800
        self._canvas_h: int = 600

        # ── undo / redo ───────────────────────────────────────────────
        self._undo: List[Dict[int, List[AnnObject]]] = []
        self._redo: List[Dict[int, List[AnnObject]]] = []
        self._pending_snap: Optional[Dict[int, List[AnnObject]]] = None

        # ── auto-tracking ────────────────────────────────────────────
        self._auto_track: bool = False
        self._trackers: Dict[int, object] = {}   # track_id -> OpenCV tracker
        self._tracker_factory = None
        self._tracker_backend = "none"

        # ── auto-annotation job ──────────────────────────────────────
        self._cap_info: Optional[auto_label.Capability] = None
        self._job: Optional[subprocess.Popen] = None
        self._job_queue: "queue.Queue" = queue.Queue()
        self._job_started: float = 0.0
        self._job_stopped: bool = False
        self._job_stop_at: float = 0.0
        self._job_kind: str = ""
        self._job_ctx: Dict[str, object] = {}
        self._job_log: List[str] = []
        self._track_thread: Optional[threading.Thread] = None
        self._track_stop: bool = False
        self._track_result: "queue.Queue" = queue.Queue()
        self._pump_armed: bool = False

        # ── annotation save path ─────────────────────────────────────
        self.annotation_path: Optional[Path] = None
        # プロジェクト保存ディレクトリ。設定されていれば Ctrl+S はここへ
        # annotations.json を静かに書き込む (ダイアログを開かない)。
        self.project_dir: Optional[Path] = None
        self._help_seen: bool = False

        self._ACTIONS: Dict[str, Callable[[], None]] = {}

        dpg.create_context()
        self._setup_theme()

    # ═══════════════════════════════════════════════════════════════════
    # Theme
    # ═══════════════════════════════════════════════════════════════════

    def _setup_theme(self) -> None:
        """Hub と同じデザイントークン / フォントを使う (gui/theme.py が唯一の定義元)。"""
        try:
            theme.setup()
        except Exception as exc:          # フォント欠落などでも起動は止めない
            print(f"Theme setup skipped: {exc}")

    # ═══════════════════════════════════════════════════════════════════
    # Project Setup Screen
    # ═══════════════════════════════════════════════════════════════════

    def _show_setup_screen(self) -> None:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        w, h = 560, 580

        if dpg.does_item_exist("setup_win"):
            dpg.delete_item("setup_win")

        with dpg.window(label="YOAKE - Project Setup", tag="setup_win",
                        width=w, height=h, no_resize=False, no_close=True,
                        pos=[max((vw - w) // 2, 0), max((vh - h) // 2, 0)]):

            widgets.page_title("New project",
                               "Name the project, define the labels, and pick "
                               "the folder Save writes into.", wrap=w - 40)
            dpg.add_separator()
            widgets.gap(2)
            dpg.add_text(
                "Create a project, define labels, and choose a folder. "
                "Save (Ctrl+S) then writes annotations.json there without a dialog.",
                color=theme.TEXT_DIM,
                wrap=w - 40,
            )
            widgets.gap(2)

            dpg.add_text("Project name", color=theme.TEXT)
            dpg.add_input_text(tag="setup_proj_name", default_value="my_project",
                               width=-1)
            widgets.gap(2)

            dpg.add_text("Class labels (comma-separated, for example: fly,worm)",
                         color=theme.TEXT)
            dpg.add_input_text(tag="setup_classes",
                               default_value=", ".join(_DEFAULT_CLASSES), width=-1)
            widgets.gap(2)

            dpg.add_text("Action labels (comma-separated, in display order)",
                         color=theme.TEXT)
            dpg.add_input_text(tag="setup_actions",
                               default_value=", ".join(_DEFAULT_ACTIONS),
                               width=-1, multiline=False)
            widgets.gap(2)

            widgets.path_field(
                "Project folder", "setup_proj_dir",
                field_key="annotator_project_dir",
                hint="Save (Ctrl+S) writes annotations.json here",
                browse=lambda initialdir="": str(
                    _tk_ask_dir("Select Project Directory", initialdir) or ""),
                tooltip_text="Optional. With a folder set, Save writes "
                             "annotations.json here without opening a dialog.",
                width=w - 230,
            )
            widgets.gap(3)
            dpg.add_separator()
            widgets.gap(2)

            widgets.primary_button("Create project",
                                   self._on_new_project_confirmed, height=36)

            widgets.gap(3)
            dpg.add_text("Or continue from an existing annotation file",
                         color=theme.TEXT_FAINT)
            widgets.gap(1)
            widgets.secondary_button("Load existing annotation JSON...",
                                     self._on_load_project_clicked, height=34)

    def _on_setup_pick_proj_dir(self) -> None:
        p = _tk_ask_dir(title="Select Project Directory")
        if p is not None and dpg.does_item_exist("setup_proj_dir"):
            dpg.set_value("setup_proj_dir", str(p))

    def _on_new_project_confirmed(self) -> None:
        self.project_name = dpg.get_value("setup_proj_name").strip() or "untitled"
        raw_cls = dpg.get_value("setup_classes")
        raw_act = dpg.get_value("setup_actions")
        self.class_names = [s.strip() for s in raw_cls.split(",") if s.strip()]
        self.action_names = [s.strip() for s in raw_act.split(",") if s.strip()]
        if not self.class_names:
            self.class_names = ["fly"]
        if not self.action_names:
            self.action_names = list(_DEFAULT_ACTIONS)
        raw_dir = dpg.get_value("setup_proj_dir").strip()
        self.project_dir = Path(raw_dir) if raw_dir else None
        dpg.delete_item("setup_win")
        self._build_main_window()
        self._update_action_buttons()
        self._refresh_project_dir_display()
        self._maybe_show_help_once()

    def _on_load_project_clicked(self) -> None:
        path = _tk_open_file(
            title="Load Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is None:
            return
        dpg.delete_item("setup_win")
        self._build_main_window()
        self._load_annotation_file(path)
