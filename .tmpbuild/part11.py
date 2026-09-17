
    # ═══════════════════════════════════════════════════════════════════
    # Auto-Annotation
    # ═══════════════════════════════════════════════════════════════════

    def _arm_pump(self) -> None:
        """
        サブプロセスの出力とワーカースレッドの結果を**メインスレッドで**取り込む。

        dpg の呼び出しはメインスレッド以外から行ってはいけないので、
        自分で再武装するフレームコールバックにポンプ役をさせる。
        """
        self._pump_armed = True
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + 6, self._pump)
        except Exception:
            self._pump_armed = False

    def _pump(self, sender=None, app_data=None) -> None:
        try:
            widgets.tick_toasts()
        except Exception:
            pass
        try:
            self._pump_job()
            self._pump_tracking()
        except Exception as exc:
            self._status(f"Background update failed: {exc}")
        if self._pump_armed:
            try:
                dpg.set_frame_callback(dpg.get_frame_count() + 6, self._pump)
            except Exception:
                self._pump_armed = False

    # ─── Capability + panel state ────────────────────────────────────

    def _refresh_auto_panel(self) -> None:
        cap = self._capability()
        if dpg.does_item_exist("al_cap_line"):
            if cap.torch:
                dpg.set_value("al_cap_line", "Model detection: available")
                dpg.configure_item("al_cap_line", color=theme.ACCEPTED)
                dpg.set_value("al_cap_detail",
                              "PyTorch was found in this Python environment.")
            else:
                dpg.set_value("al_cap_line", "Model detection: unavailable")
                dpg.configure_item("al_cap_line", color=theme.PENDING)
                dpg.set_value(
                    "al_cap_detail",
                    "PyTorch is not installed in this Python environment, so the "
                    "detector cannot run here. Importing predictions produced "
                    "elsewhere and tracking forward below both still work.")

        for tag in ("al_run_btn", "tb_autolabel"):
            if cap.torch:
                widgets.unlock(tag, "Run the detector and bring the results back "
                                    "as candidates")
            else:
                widgets.locked(tag, cap.torch_reason or
                               auto_label.TORCH_MISSING_MESSAGE)

        _factory, backend = self._tracker_chain()
        if dpg.does_item_exist("al_tracker_name"):
            dpg.set_value("al_tracker_name",
                          f"Tracker: {auto_label.TRACKER_LABELS.get(backend, backend)}")
        if dpg.does_item_exist("al_tracker_note"):
            dpg.set_value("al_tracker_note", cap.tracker_reason or "")
        if dpg.does_item_exist("auto_track_backend"):
            dpg.set_value("auto_track_backend",
                          f"Tracker: {auto_label.TRACKER_LABELS.get(backend, backend)}. "
                          + (cap.tracker_reason or ""))

        pend = len(self._pending_in(self.current_frame_idx))
        total = self.pending_total()
        if dpg.does_item_exist("al_review_line"):
            if pend:
                dpg.set_value(
                    "al_review_line",
                    f"{pend} candidates in this frame - click one to accept it, "
                    f"right-click to delete. {total} pending in the whole video.")
            elif total:
                dpg.set_value("al_review_line",
                              f"{total} candidates elsewhere - press Ctrl+Right "
                              "to jump to the next frame that has some.")
            else:
                dpg.set_value("al_review_line", "No candidates to review.")
            dpg.configure_item("al_review_line",
                               color=theme.PENDING if total else theme.TEXT_FAINT)
        if dpg.does_item_exist("tb_cand_pill"):
            widgets.status_pill(
                "tb_cand_pill",
                f"{pend} candidates" if pend else "No candidates",
                "pending" if pend else "idle")
        for tag in ("tb_accept_all", "tb_discard_all", "al_accept_btn",
                    "al_discard_btn"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=bool(pend))

    # ─── Dialog ──────────────────────────────────────────────────────

    def open_autolabel_dialog(self) -> None:
        cap = self._capability()
        if not cap.torch:
            self._status("Automatic detection needs PyTorch, which is not "
                         "installed here. Use \"Import predictions as "
                         "candidates...\" instead.")
            widgets.toast("Automatic detection needs PyTorch. "
                          "Install it with: pip install torch torchvision",
                          kind="warn", duration=7.0)
            return
        if self.cap is None or self.video_path is None:
            self._status("Open a video first.")
            return
        if dpg.does_item_exist("autolabel_dlg"):
            dpg.delete_item("autolabel_dlg")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        w, h = 600, 470
        with dpg.window(label="Auto-Annotate", tag="autolabel_dlg", modal=True,
                        width=w, height=h, no_resize=False,
                        pos=[max((vw - w) // 2, 0), max((vh - h) // 2, 0)]):
            widgets.section_header(
                "Run the detector on this video",
                "Results arrive as candidates. Candidates are not saved as "
                "labels. Review them, then accept the ones you want.", wrap=w - 40)
            widgets.path_field(
                "Detector checkpoint (.pth or .pt)", "al_ckpt",
                field_key="checkpoint", required=True,
                hint="A trained YOAKE checkpoint",
                browse=lambda initialdir="": str(_tk_open_file(
                    "Select Checkpoint",
                    [("Checkpoints", "*.pth *.pt"), ("All files", "*.*")],
                    initialdir) or ""),
                width=w - 230)
            widgets.gap(1)
            with widgets.advanced_section("Config (optional)"):
                widgets.path_field(
                    "Config (.yaml)", "al_cfg", field_key="config",
                    hint="Only when the checkpoint has no embedded config",
                    browse=lambda initialdir="": str(_tk_open_file(
                        "Select Config",
                        [("YAML", "*.yaml *.yml"), ("All files", "*.*")],
                        initialdir) or ""),
                    tooltip_text="Only needed when the checkpoint was saved "
                                 "without its training configuration. If a run "
                                 "fails with a message about checkpoint metadata, "
                                 "point this at the config used for training.",
                    width=w - 260)
            widgets.gap(1)
            widgets.labeled_slider("Score threshold", "al_score", default=0.30,
                                   min_value=0.05, max_value=0.95, kind="float",
                                   width=260, fmt="%.2f",
                                   help_text="Detections below this confidence are "
                                             "dropped (default 0.30). The "
                                             "detector's confidence is not a "
                                             "calibrated probability.")
            widgets.gap(1)
            dpg.add_text("Apply to", color=theme.TEXT)
            dpg.add_radio_button(
                items=["This frame", "This frame and the next N", "Whole video"],
                tag="al_scope", default_value="This frame and the next N",
                callback=self._on_al_scope_change)
            with dpg.group(horizontal=True, tag="al_frames_row"):
                dpg.add_text("Frames (N):", indent=4)
                dpg.add_input_int(tag="al_frames", default_value=60, width=90,
                                  min_value=1, max_value=100000)
            widgets.gap(1)
            dpg.add_checkbox(label="Skip frames that already have boxes I drew",
                             tag="al_skip", default_value=True)
            widgets.gap(1)
            dpg.add_text("Candidates are not saved as labels. Review them, then "
                         "accept the ones you want.", color=theme.TEXT_DIM,
                         wrap=w - 40)
            widgets.gap(2)
            dpg.add_separator()
            widgets.gap(1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run", width=140, height=32,
                               tag="al_dlg_run", callback=self._start_autolabel)
                theme.bind("al_dlg_run", theme.accent_theme())
                dpg.add_button(label="Cancel", width=140, height=32,
                               callback=lambda: dpg.delete_item("autolabel_dlg"))

    def _on_al_scope_change(self, sender=None, app_data=None) -> None:
        if dpg.does_item_exist("al_frames_row"):
            dpg.configure_item("al_frames_row",
                               show=(app_data == "This frame and the next N"))

    # ─── Run ─────────────────────────────────────────────────────────

    def _autolabel_cache_dir(self) -> Path:
        base = self.project_dir or (self.video_path.parent if self.video_path
                                    else Path.cwd())
        return Path(base) / ".cache" / "autolabel"

    def _start_autolabel(self) -> None:
        if self._job is not None:
            self._status("An auto-annotation run is already in progress.")
            return
        weights = (dpg.get_value("al_ckpt") or "").strip()
        if not weights or not Path(weights).exists():
            self._status("Detector checkpoint not found. Pick a .pth or .pt file.")
            return
        config = (dpg.get_value("al_cfg") or "").strip() if dpg.does_item_exist("al_cfg") else ""
        score = float(dpg.get_value("al_score"))
        if not (0.05 <= score <= 0.95):
            self._status("Score threshold must be between 0.05 and 0.95.")
            return
        scope = dpg.get_value("al_scope")
        skip_human = bool(dpg.get_value("al_skip"))

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        out_dir = self._autolabel_cache_dir() / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        index_map: Dict[str, int] = {}
        if scope == "Whole video" and self.current_frame_idx == 0:
            source = str(self.video_path)
            max_frames = None
        else:
            count = (1 if scope == "This frame"
                     else (int(dpg.get_value("al_frames"))
                           if scope == "This frame and the next N"
                           else self.num_frames))
            plan = auto_label.export_plan(self.current_frame_idx, count,
                                          self.num_frames, job_id)
            if not plan.frames:
                self._status("There are no frames in that range.")
                return
            frames_dir = out_dir / "frames"
            n = self._export_frames(plan.frames, frames_dir)
            if n == 0:
                self._status("Could not read frames from the video for export.")
                return
            auto_label.write_index_map(plan, out_dir)
            index_map = dict(plan.index_map)
            source = str(frames_dir)
            max_frames = None

        req = auto_label.PredictRequest(
            source=source, weights=weights, output_dir=str(out_dir),
            score_threshold=score, max_frames=max_frames, config=config)
        argv = [sys.executable] + auto_label.predict_argv(req)

        root = _repo_root()
        creation = 0
        if os.name == "nt":
            creation = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            proc = subprocess.Popen(
                argv, cwd=str(root), env=auto_label.child_env(root),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=creation)
        except OSError as exc:
            self._status(f"Auto-annotation could not start: {exc}")
            return

        self._job = proc
        self._job_kind = "autolabel"
        self._job_started = time.time()
        self._job_stopped = False
        self._job_stop_at = 0.0
        self._job_log = []
        self._job_queue = queue.Queue()
        self._job_ctx = {"out_dir": out_dir, "index_map": index_map,
                         "min_score": score, "skip_human": skip_human}

        threading.Thread(target=self._drain_process, args=(proc, self._job_queue),
                         daemon=True).start()
        if dpg.does_item_exist("autolabel_dlg"):
            dpg.delete_item("autolabel_dlg")
        if dpg.does_item_exist("al_stop_btn"):
            dpg.configure_item("al_stop_btn", show=True)
        self._status("Running auto-annotation...")

    def _export_frames(self, frames: List[int], out_dir: Path) -> int:
        """
        指定フレームを JPEG に書き出す。

        ``cap.set(POS_FRAMES)`` は非整数 fps で 1 枚ずれるうえ frame 0 では
        一致するため気付きにくい。順に ``grab()`` し、必要な枚だけ ``retrieve()``
        する。index_map.json と組にしないと候補が全部別フレームに乗る。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        wanted = {int(f): i for i, f in enumerate(frames)}
        if not wanted:
            return 0
        last = max(wanted)
        cap = cv2.VideoCapture(str(self.video_path))
        written = 0
        try:
            idx = 0
            while idx <= last:
                if not cap.grab():
                    break
                if idx in wanted:
                    ok, frame = cap.retrieve()
                    if ok:
                        cv2.imwrite(str(out_dir / f"{wanted[idx]:06d}.jpg"), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        written += 1
                idx += 1
        finally:
            cap.release()
        return written

    @staticmethod
    def _drain_process(proc: subprocess.Popen, q: "queue.Queue") -> None:
        """子プロセスの出力を読み続けるだけのスレッド (dpg には一切触らない)。"""
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    q.put(line.rstrip("\n"))
        except Exception as exc:
            q.put(f"[reader stopped: {exc}]")

    def stop_job(self) -> None:
        """Stop は停止であって失敗ではない。子プロセス木ごと確実に止める。"""
        if self._job is None:
            self._status("No auto-annotation is running.")
            return
        self._job_stopped = True
        self._job_stop_at = time.time()
        try:
            self._job.terminate()
        except Exception:
            pass
        self._status("Stop requested...")

    def _pump_job(self) -> None:
        if self._job is None:
            return
        while True:
            try:
                line = self._job_queue.get_nowait()
            except queue.Empty:
                break
            self._job_log.append(line)
            if len(self._job_log) > 400:
                del self._job_log[0]

        elapsed = time.time() - self._job_started
        tail = self._job_log[-1] if self._job_log else ""
        if dpg.does_item_exist("al_job_line"):
            mins, secs = divmod(int(elapsed), 60)
            dpg.set_value("al_job_line",
                          f"Running auto-annotation... {mins}:{secs:02d} elapsed\n"
                          f"{tail[:120]}")

        # Stop したのに残っていれば、DataLoader/CUDA の子ごと強制終了する
        if (self._job_stopped and self._job_stop_at
                and time.time() - self._job_stop_at > 3.0
                and self._job.poll() is None):
            self._force_kill(self._job)
            self._job_stop_at = 0.0

        code = self._job.poll()
        if code is None:
            return
        proc, self._job = self._job, None
        if dpg.does_item_exist("al_stop_btn"):
            dpg.configure_item("al_stop_btn", show=False)
        if dpg.does_item_exist("al_job_line"):
            dpg.set_value("al_job_line", "")
        self._finish_autolabel(code)
        _ = proc

    @staticmethod
    def _force_kill(proc: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, check=False)
                return
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass

    def _finish_autolabel(self, code: int) -> None:
        ctx = self._job_ctx or {}
        out_dir = Path(str(ctx.get("out_dir", "")))
        pred = auto_label.predictions_path_for(out_dir)
        if self._job_stopped and not pred.exists():
            self._status("Auto-annotation stopped. Nothing was produced.")
            return
        if not pred.exists():
            tail = " | ".join(self._job_log[-3:])
            self._status("The run produced no predictions.json. "
                         f"Check the log for the reason. {tail}")
            return
        try:
            cands = auto_label.load_candidates(
                pred, min_score=float(ctx.get("min_score", 0.0)))
        except (OSError, ValueError) as exc:
            self._status(f"Load error: {exc}")
            return
        index_map = ctx.get("index_map") or {}
        if index_map:
            cands = auto_label.remap_frame_indices(cands, index_map)
        policy = auto_label.MergePolicy(
            skip_frames_with_accepted=bool(ctx.get("skip_human", True)),
            min_score=float(ctx.get("min_score", 0.0)))
        self._ingest_candidates(cands, policy, stopped=self._job_stopped)

    def _obj_from_candidate(self, cand) -> AnnObject:
        obj = AnnObject(
            bbox=[int(v) for v in cand.bbox],
            track_id=int(cand.track_id),
            action_id=int(cand.action_id),
            class_id=int(cand.class_id),
            keyframe=False,
            obb=list(cand.obb) if cand.obb is not None else None,
            accepted=False,
            source=str(cand.source or "auto"),
            score=float(cand.score),
        )
        return obj

    def _ingest_candidates(self, cands, policy, *, stopped: bool = False) -> None:
        """候補を注釈へ取り込む。人の box は決して上書きしない。"""
        frames = sorted(set(list(cands.keys())
                            + [f for f, objs in self.annotations.items()
                               if any(o.is_candidate for o in objs)]))
        if frames:
            self._push_undo(frames)
        merged, result = auto_label.merge_candidates(
            self.annotations, cands, policy,
            is_accepted=lambda o: o.accepted,
            is_human=lambda o: o.source == "human",
            bbox_of=lambda o: o.bbox,
            obb_of=lambda o: o.obb,
            make_obj=self._obj_from_candidate,
        )
        self.annotations = {int(k): list(v) for k, v in merged.items() if v}
        self._after_mutation(auto_label.describe(result, stopped=stopped))
        if result.added:
            widgets.toast(f"{result.added} candidates are waiting for review.",
                          kind="warn", duration=5.0)

    # ─── Import a predictions.json produced elsewhere ────────────────

    def _import_predictions_dialog(self) -> None:
        path = _tk_open_file(
            title="Import Predictions as Candidates",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is not None:
            self.import_predictions(path)

    def import_predictions(self, path: Path, *, min_score: float = 0.0) -> None:
        """predictions.json (streaming / batch / pipeline) を候補として読み込む。"""
        try:
            cands = auto_label.load_candidates(Path(path), min_score=min_score)
        except (OSError, ValueError) as exc:
            self._status(f"Load error: {exc}")
            return
        if not cands:
            self._status(f"{Path(path).name} holds no detections.")
            return
        self._ingest_candidates(cands, auto_label.MergePolicy(min_score=min_score))

    # ─── Track forward (torch-free fallback) ─────────────────────────

    def track_forward_selected(self) -> None:
        """T: 選択中の box を OpenCV で前方へ伝播し、候補として積む。"""
        if self.cap is None or self.video_path is None:
            self._status("Open a video first.")
            return
        if self._track_thread is not None and self._track_thread.is_alive():
            self._track_stop = True
            self._status("Stop requested...")
            return
        obj = self._selected
        if obj is None:
            accepted = [o for o in self.annotations.get(self.current_frame_idx, [])
                        if o.accepted]
            if len(accepted) == 1:
                obj = accepted[0]
        if obj is None or not obj.accepted:
            self._status("Draw at least one box first. Tracking extends boxes you "
                         "have already drawn - it cannot find animals on its own.")
            return

        n = 30
        if dpg.does_item_exist("al_track_frames"):
            n = max(1, int(dpg.get_value("al_track_frames")))
        start = self.current_frame_idx
        box = list(obj.bbox)
        track_id = obj.track_id
        action_id = obj.action_id
        frame_w, frame_h = self.frame_w, self.frame_h
        self._track_stop = False
        read, cap = self._worker_reader()

        def worker():
            try:
                boxes, reason = auto_label.track_forward(
                    read, start, box, n, frame_w, frame_h,
                    should_stop=lambda: self._track_stop)
            except Exception as exc:
                boxes, reason = [], f"error: {exc}"
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            self._track_result.put((track_id, action_id, boxes, reason))

        self._track_thread = threading.Thread(target=worker, daemon=True)
        self._track_thread.start()
        self._status(f"Tracking... 0/{n} frames")

    def _pump_tracking(self) -> None:
        try:
            track_id, action_id, boxes, reason = self._track_result.get_nowait()
        except queue.Empty:
            return
        if not boxes:
            self._status(f"Tracking produced nothing ({reason or 'no motion'}). "
                         "Try a larger box or a shorter range.")
            return
        frames = [f for f, _b in boxes]
        self._push_undo(frames)
        added = 0
        for fidx, bbox in boxes:
            # そのトラックの box が既にあるフレームは絶対に触らない
            if any(o.track_id == track_id for o in self.annotations.get(fidx, [])):
                continue
            self.annotations.setdefault(fidx, []).append(AnnObject(
                bbox=[int(v) for v in bbox], track_id=track_id,
                action_id=action_id, keyframe=False,
                accepted=False, source="track", score=1.0))
            added += 1
        note = {
            "": "",
            "lost": f" Tracking lost for track {track_id}.",
            "too_small": " The box shrank below the minimum size.",
            "out_of_frame": " The box left the frame.",
            "stopped": " Stopped by user.",
            "end_of_video": " Reached the end of the video.",
        }.get(reason, f" {reason}")
        self._after_mutation(
            f"Track forward done: {added} candidates on track {track_id}."
            f"{note} Press Ctrl+Right to review them, Y to accept a frame.")
