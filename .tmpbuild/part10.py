
    # ═══════════════════════════════════════════════════════════════════
    # Range Labeling  (範囲ラベリング)
    # ═══════════════════════════════════════════════════════════════════

    def _track_obj(self, frame_idx: int, track_id: int) -> Optional[AnnObject]:
        """指定フレームで track_id を持つ最初のオブジェクトを返す (無ければ None)。"""
        for obj in self.annotations.get(frame_idx, []):
            if obj.track_id == track_id:
                return obj
        return None

    def apply_range(self, track_id: int, action_id: Optional[int],
                    f_start: int, f_end: int, interpolate: bool) -> None:
        """区間 [f_start, f_end] の track に行動/bbox を一括適用する。

        - ``interpolate=True`` かつ区間内にキーフレーム box が 2 個以上あれば、隣接
          キーフレーム間を線形補間して中間フレームの box を生成する (端の外側は
          最も近いキーフレームを保持)。キーフレームが 1 個なら全域その box を保持。
        - ``action_id`` が ``None`` の場合は行動を変更しない。box が無く geometry も
          得られないフレームはスキップ (既存 box があれば行動だけ更新)。
        保存形式は v1.1 のまま (生成 box は通常の per-frame object)。

        1 クリックで数千の box を書き換えられる唯一の操作なので、触る全フレームを
        まとめて 1 ステップとして undo に積む。
        """
        if self.cap is None:
            self._status("Open a video first.")
            return
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        f_start = max(0, f_start)
        f_end = min(self.num_frames - 1, f_end) if self.num_frames else f_end

        self._push_undo(range(f_start, f_end + 1))

        anchors: List[Tuple[int, List[int]]] = []
        for f in range(f_start, f_end + 1):
            obj = self._track_obj(f, track_id)
            if obj is not None and obj.keyframe:
                anchors.append((f, list(obj.bbox)))

        # 補間 ON かつアンカーがあれば box_map を生成 (純関数 resolve_range_boxes)
        box_map = (resolve_range_boxes(anchors, f_start, f_end)
                   if interpolate and anchors else {})
        anchor_frames = {f for f, _ in anchors}
        count = 0
        for f in range(f_start, f_end + 1):
            obj = self._track_obj(f, track_id)
            box = box_map.get(f)
            if obj is None:
                if box is None:
                    continue  # 配置すべき box が無い → スキップ
                placed = round_bbox(clamp_bbox(box, self.frame_w, self.frame_h))
                obj = AnnObject(bbox=placed, track_id=track_id, action_id=-1,
                                keyframe=False, source="interp")
                self.annotations.setdefault(f, []).append(obj)
            elif box is not None and f not in anchor_frames:
                obj.bbox = round_bbox(clamp_bbox(box, self.frame_w, self.frame_h))
                obj.keyframe = False
            if action_id is not None:
                obj.action_id = action_id
            count += 1

        self.next_track_id = max(self.next_track_id, track_id)
        self._dirty = True
        note = ""
        if interpolate and len(anchors) < 2:
            note = ("(keyframe held)" if anchors
                    else "(no box - action only)")
        self._after_mutation(
            f"Range {f_start}-{f_end}, track {track_id}: applied to {count} "
            f"frames {note}. {self._undo_hint()}")

    def clear_range(self, track_id: int, f_start: int, f_end: int) -> None:
        """区間 [f_start, f_end] の指定 track のオブジェクトを削除する。"""
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        self._push_undo(range(f_start, f_end + 1))
        removed = 0
        for f in range(f_start, f_end + 1):
            objs = self.annotations.get(f, [])
            keep = [o for o in objs if o.track_id != track_id]
            removed += len(objs) - len(keep)
            if keep:
                self.annotations[f] = keep
            elif f in self.annotations:
                del self.annotations[f]
        self._selected = None
        self._dirty = True
        self._after_mutation(
            f"Range {f_start}-{f_end}, track {track_id}: removed {removed} boxes. "
            f"{self._undo_hint()}")

    def _range_action_id(self) -> Optional[int]:
        val = dpg.get_value("range_action") if dpg.does_item_exist("range_action") else None
        if val in (None, "(no change)"):
            return None
        if val == "(none)":
            return -1
        return self.action_names.index(val) if val in self.action_names else None

    def _apply_range_ui(self) -> None:
        if not dpg.does_item_exist("range_track"):
            return
        self.apply_range(
            track_id=int(dpg.get_value("range_track")),
            action_id=self._range_action_id(),
            f_start=int(dpg.get_value("range_start")),
            f_end=int(dpg.get_value("range_end")),
            interpolate=bool(dpg.get_value("range_interp")),
        )

    def _clear_range_ui(self) -> None:
        if not dpg.does_item_exist("range_track"):
            return
        self.clear_range(
            track_id=int(dpg.get_value("range_track")),
            f_start=int(dpg.get_value("range_start")),
            f_end=int(dpg.get_value("range_end")),
        )

    def set_action(self, action_id: int) -> None:
        if self._selected is None:
            self._status(f"No box selected. Action {action_id} was ignored.")
            return
        if action_id >= len(self.action_names):
            self._status(f"There is no action {action_id}. "
                         f"Define one in the Labels panel first.")
            return
        self._push_undo([self.current_frame_idx])
        self._selected.action_id = action_id
        self._dirty = True
        self._after_mutation(f"Action set to {self.action_names[action_id]}.")

    # ═══════════════════════════════════════════════════════════════════
    # Review flow (candidates)
    # ═══════════════════════════════════════════════════════════════════

    def _pending_in(self, frame_idx: int) -> List[AnnObject]:
        return [o for o in self.annotations.get(frame_idx, []) if o.is_candidate]

    def pending_total(self) -> int:
        return sum(1 for objs in self.annotations.values()
                   for o in objs if o.is_candidate)

    def accept_all_in_frame(self) -> None:
        pend = self._pending_in(self.current_frame_idx)
        if not pend:
            self._status("No candidates in this frame.")
            return
        self._push_undo([self.current_frame_idx])
        for o in pend:
            o.accept()
            o.keyframe = True
        self._dirty = True
        self._after_mutation(
            f"Accepted {len(pend)} candidates in frame {self.current_frame_idx}. "
            f"{self._undo_hint()}")

    def discard_all_in_frame(self) -> None:
        pend = self._pending_in(self.current_frame_idx)
        if not pend:
            self._status("No candidates in this frame.")
            return
        self._push_undo([self.current_frame_idx])
        keep = [o for o in self.annotations.get(self.current_frame_idx, [])
                if not o.is_candidate]
        if keep:
            self.annotations[self.current_frame_idx] = keep
        else:
            self.annotations.pop(self.current_frame_idx, None)
        if self._selected is not None and self._selected.is_candidate:
            self._selected = None
        self._after_mutation(
            f"Discarded {len(pend)} candidates. Boxes you drew by hand are kept. "
            f"{self._undo_hint()}")

    def accept_above_threshold(self, threshold: float) -> None:
        """Batch accept: 全フレームで score >= threshold の候補を確定する。"""
        frames = [f for f, objs in self.annotations.items()
                  if any(o.is_candidate for o in objs)]
        if not frames:
            self._status("There are no candidates to review.")
            return
        self._push_undo(frames)
        n = 0
        for f in frames:
            for o in self.annotations.get(f, []):
                if o.is_candidate and (o.score is None or o.score >= threshold):
                    o.accept()
                    o.keyframe = True
                    n += 1
        self._dirty = True
        self._after_mutation(
            f"Accepted {n} candidates at or above {threshold:.2f}. "
            f"{self.pending_total()} candidates are still pending.")

    def discard_all_candidates(self) -> None:
        frames = [f for f, objs in self.annotations.items()
                  if any(o.is_candidate for o in objs)]
        if not frames:
            self._status("There are no candidates to discard.")
            return
        self._push_undo(frames)
        n = 0
        for f in list(frames):
            objs = self.annotations.get(f, [])
            keep = [o for o in objs if not o.is_candidate]
            n += len(objs) - len(keep)
            if keep:
                self.annotations[f] = keep
            else:
                self.annotations.pop(f, None)
        if self._selected is not None and self._selected.is_candidate:
            self._selected = None
        self._after_mutation(f"Discarded {n} candidates on {len(frames)} frames. "
                             "Boxes you drew by hand are kept.")

    def goto_next_pending_frame(self) -> None:
        frames = sorted(f for f, objs in self.annotations.items()
                        if any(o.is_candidate for o in objs))
        nxt = next((f for f in frames if f > self.current_frame_idx), None)
        if nxt is None:
            self._status("No candidates after this frame.")
            return
        self.goto_frame(nxt)
        self._status(f"Frame {nxt}: {len(self._pending_in(nxt))} candidates "
                     "- Y accepts them all, N discards them.")

    def goto_prev_pending_frame(self) -> None:
        frames = sorted(f for f, objs in self.annotations.items()
                        if any(o.is_candidate for o in objs))
        prev = next((f for f in reversed(frames) if f < self.current_frame_idx), None)
        if prev is None:
            self._status("No candidates before this frame.")
            return
        self.goto_frame(prev)
        self._status(f"Frame {prev}: {len(self._pending_in(prev))} candidates "
                     "- Y accepts them all, N discards them.")

    # ═══════════════════════════════════════════════════════════════════
    # Auto-Tracking
    # ═══════════════════════════════════════════════════════════════════

    def _tracker_chain(self):
        """OpenCV トラッカの工場を 1 回だけ解決する (CSRT -> legacy -> MIL -> LK)。"""
        if self._tracker_backend == "none":
            try:
                self._tracker_factory, self._tracker_backend = auto_label.make_tracker()
            except Exception:
                self._tracker_factory, self._tracker_backend = None, "lk"
        return self._tracker_factory, self._tracker_backend

    def _init_tracker(self, track_id: int, frame_idx: int,
                      bbox: List[int]) -> bool:
        """Initialize an OpenCV tracker for the given track and bounding box.

        ``cv2.TrackerCSRT_create`` は OpenCV 4.13 (contrib なし) には存在しない。
        gui/auto_label.make_tracker() が実際にある実装 (MIL など) を返す。
        """
        factory, backend = self._tracker_chain()
        if factory is None:
            self._status("No OpenCV tracker is available. Install "
                         "opencv-contrib-python for CSRT, or use Track forward, "
                         "which falls back to optical flow.")
            return False
        frame_bgr = self._read_frame_bgr(frame_idx)
        if frame_bgr is None:
            return False
        try:
            tracker = factory()
            x1, y1, x2, y2 = bbox
            tracker.init(frame_bgr, (int(x1), int(y1),
                                     int(x2 - x1), int(y2 - y1)))
        except Exception as exc:
            self._status(f"Tracking failed: {exc}")
            return False
        self._trackers[track_id] = tracker
        self._status(f"Tracker initialized for track {track_id} "
                     f"({auto_label.TRACKER_LABELS.get(backend, backend)}).")
        return True

    def _init_tracker_from_selected(self) -> None:
        if self._selected is None:
            self._status("Select a box first.")
            return
        self._init_tracker(self._selected.track_id,
                           self.current_frame_idx, self._selected.bbox)
        self._render_canvas()

    def _run_trackers_for_frame(self, frame_idx: int,
                                frame_bgr: np.ndarray) -> None:
        """Run all active trackers on frame_bgr and create annotation objects."""
        existing_tids = {o.track_id for o in
                         self.annotations.get(frame_idx, [])}
        alive = 0
        for track_id, tracker in list(self._trackers.items()):
            if track_id in existing_tids:
                alive += 1
                continue
            try:
                success, xywh = tracker.update(frame_bgr)
            except Exception:
                success, xywh = False, (0, 0, 0, 0)
            if success:
                alive += 1
                x, y, w, h = [int(v) for v in xywh]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(self.frame_w, x + w)
                y2 = min(self.frame_h, y + h)
                if x2 - x1 >= _MIN_BOX_PX and y2 - y1 >= _MIN_BOX_PX:
                    # 前フレームから action と「回転矩形かどうか」を引き継ぐ。
                    # トラッカは軸平行の箱しか返さないので、角度は前フレームの値を
                    # そのまま保持する (AI が提案し、人が必要に応じて直す方針)。
                    prev_action = -1
                    prev_obb = None
                    for obj in self.annotations.get(frame_idx - 1, []):
                        if obj.track_id == track_id:
                            prev_action = obj.action_id
                            prev_obb = list(obj.obb) if obj.has_obb else None
                            break
                    new_obj = AnnObject(bbox=[x1, y1, x2, y2],
                                        track_id=track_id,
                                        action_id=prev_action,
                                        keyframe=False,
                                        source="track")
                    if prev_obb is not None:
                        # 中心とサイズはトラッカの結果に合わせ、角度だけ引き継ぐ
                        new_obj.obb = [
                            (x1 + x2) / 2, (y1 + y2) / 2,
                            float(x2 - x1), float(y2 - y1), prev_obb[4],
                        ]
                        new_obj.sync_bbox()
                    self.annotations.setdefault(frame_idx, []).append(new_obj)
                    self._dirty = True
            else:
                total = len(self._trackers)
                self._status(
                    f"Tracking lost for track {track_id} at frame {frame_idx} - "
                    f"{alive} of {total} tracks are still following. "
                    "Draw the box again to re-anchor it.")

    def _reset_all_trackers(self) -> None:
        self._trackers.clear()
        self._status("All trackers reset.")
        self._render_canvas()

    def _reset_selected_tracker(self) -> None:
        if self._selected is None:
            self._status("Select a box first.")
            return
        tid = self._selected.track_id
        if tid in self._trackers:
            del self._trackers[tid]
            self._status(f"Tracker for track {tid} removed.")
        self._render_canvas()

    def _on_auto_track_toggle(self, sender=None, app_data=None) -> None:
        self._auto_track = bool(app_data)
        if not self._auto_track:
            self._trackers.clear()
        self._status(f"Auto-track is {'on' if self._auto_track else 'off'}.")

    def _on_obb_mode_toggle(self, sender=None, app_data=None) -> None:
        """後方互換のためのフック (チェックボックスはツールバーに置き換わった)。"""
        self.set_tool("obb" if app_data else "select")

    def _toggle_auto_track_key(self) -> None:
        new_val = not self._auto_track
        if dpg.does_item_exist("auto_track_cb"):
            dpg.set_value("auto_track_cb", new_val)
        self._auto_track = new_val
        if not self._auto_track:
            self._trackers.clear()
        self._status(f"Auto-track is {'on' if self._auto_track else 'off'}.")
