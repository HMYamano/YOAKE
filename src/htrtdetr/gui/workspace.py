"""Project orchestration for the existing annotator canvas.

All GUI mutations and VideoSession mutations happen on the main thread. Workers
receive detached inputs and return results tagged with project/video identity.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from ..annotation.layout import PROJECT_FILE, ProjectLayout
from ..annotation.project import ProjectSession, save_snapshot, scan_folder
from ..annotation.session import VideoSession
from .workspace_operations import WorkspaceOperations


class WorkspaceMixin(WorkspaceOperations):
    def _ensure_project(self):
        if not hasattr(self, "project"):
            self.project = ProjectSession()
            self._workspace_save_thread = None
            self._workspace_save_error = None
            self._workspace_revision = 0
        return self.project

    def _sync_project(self):
        project = self._ensure_project()
        # Older integrations set video_path/annotations before asking to save.
        if self.video_path is not None and not self._video_session.video_id:
            registered = project.add_video(self.video_path)
            self._video_session.video_id = registered.video_id
            project.videos[registered.video_id] = self._video_session
        project.name = self.project_name
        project.class_names = list(self.class_names)
        project.action_names = list(self.action_names)
        project.directory = self.project_dir
        from dataclasses import asdict

        project.settings["quality"] = asdict(self.quality_config)
        project.settings["click_backend"] = self._click_backend
        project.settings["sticky_draw"] = self._sticky_draw
        if self._video_session.video_id:
            project.last_video = self._video_session.video_id
            self._store_video_ui()
        return project

    def _store_video_ui(self):
        video = self._video_session
        video.ui.update(
            zoom=self._zoom,
            pan_x=self._pan_x,
            pan_y=self._pan_y,
            selected_object=getattr(self._selected, "object_id", None),
        )

    def _load_video(self, path: Path):
        project = self._ensure_project()
        video = project.add_video(path)
        return self.switch_video(video.video_id)

    def switch_video(self, video_id: str) -> bool:
        import cv2
        import dearpygui.dearpygui as dpg

        project = self._ensure_project()
        video = project.videos[video_id]
        if video is self._video_session and self.cap is not None:
            return True
        if video.missing:
            self._status(f"Missing video: {video.video_id}. Use Project Videos > Locate file.")
            self._refresh_workspace()
            return False
        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
            cap.release()
            self._status(f"Cannot decode {video.path.name}. Check the file or codec.")
            return False
        self._store_video_ui()
        if self.review_mode.active:
            self.review_mode.close()
        # Old native tracker and SAM prompt contexts cannot follow a new capture.
        self._trackers.clear()
        self._sam_jobs.clear()
        self._sam_embed_due = None
        self._propagate_stop = True
        if self.cap is not None:
            self.cap.release()
        self._video_session = video
        self._trackers.clear()
        self._mode = "idle"
        self._pending_snap = None
        project.last_video = video_id
        self._install_video_capture(cap)
        self._zoom = float(video.ui.get("zoom", 1.0))
        self._pan_x = float(video.ui.get("pan_x", 0.0))
        self._pan_y = float(video.ui.get("pan_y", 0.0))
        selected = video.ui.get("selected_object")
        self._selected = next(
            (
                o
                for o in self.annotations.get(self.current_frame_idx, [])
                if o.object_id == selected
            ),
            None,
        )
        self._recompute_scale()
        self.inbox.set_queue(self._queue, fps=self.fps)
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.active_track_id)
        self._refresh_history_ui()
        self._refresh_sidebar()
        self._render_canvas()
        self._render_timeline()
        self._refresh_workspace()
        return True

    def _visit_frame(self, index: int):
        video = self._video_session
        video.visited.add(index)
        # Session position also needs an autosave, even without a box edit.
        self._workspace_revision = getattr(self, "_workspace_revision", 0) + 1
        if getattr(self, "_save_due", None) is None:
            self._save_due = time.time() + 4.0

    def _advance_at_end(self, index: int) -> bool:
        project = self._ensure_project()
        if index < self.num_frames or project.settings.get("end_of_video") != "next_unfinished":
            return False
        return self.adjacent_video(1, unfinished=True)

    def adjacent_video(self, direction: int, unfinished=False):
        videos = list(self._ensure_project().videos.values())
        ids = [v.video_id for v in videos]
        if not ids:
            return False
        current = (
            ids.index(self._video_session.video_id) if self._video_session.video_id in ids else -1
        )
        indices = range(current + 1, len(ids)) if direction > 0 else range(current - 1, -1, -1)
        for index in indices:
            if unfinished and videos[index].display_status == "done":
                continue
            if self.switch_video(ids[index]):
                return True
        self._status("No more unfinished videos." if unfinished else "End of project video list.")
        return False

    def _preserve_project(self):
        project = self._sync_project()
        if not project.videos:
            return
        worker = getattr(self, "_workspace_save_thread", None)
        if worker is not None and worker.is_alive():
            worker.join()
        # Opening another project must not silently discard an unnamed session.
        directory = (
            project.directory
            or Path(tempfile.gettempdir()) / "YOAKE" / "recovery" / project.identity
        )
        project.save(directory / PROJECT_FILE)

    def open_project(self, path: Path):
        try:
            worker = getattr(self, "_workspace_save_thread", None)
            if worker is not None and worker.is_alive():
                worker.join()
            project = ProjectSession.open(path)  # validate before replacing anything
            old_path = self.video_path
            if old_path is not None:
                unlinked = project.videos.get(old_path.stem)
                if unlinked is not None and unlinked.path is None:
                    unlinked.path = old_path
            # Do not overwrite the very file about to be loaded with the old session.
            self._preserve_project_to_recovery()
            self._adopt_project(project)
            self.annotation_path = ProjectLayout(project.directory).annotations
            target = project.videos.get(project.last_video)
            if target is not None and not target.missing:
                self.switch_video(target.video_id)
            else:
                self._refresh_workspace()
                self._status("Project opened. Locate missing files in Project Videos.")
            if project.relocated:
                # The folder moved and the videos were found again by name. Say
                # so: silently opening a different file with the right name is
                # the one failure the search fallback could cause.
                names = ", ".join(project.relocated[:3])
                more = f" and {len(project.relocated) - 3} more" if len(project.relocated) > 3 else ""
                self._status(f"Relocated {names}{more} by file name. Check they are the right ones.")
                self._save_due = time.time() + 2
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._status(f"Could not open project: {exc}")
            return False
        return True

    def _preserve_project_to_recovery(self):
        project = self._sync_project()
        if project.videos:
            folder = Path(tempfile.gettempdir()) / "YOAKE" / "recovery" / project.identity
            project.save(folder / PROJECT_FILE)

    def _adopt_project(self, project):
        self.stop_project_batch()
        if hasattr(self, "_fast_track_stop"):
            self._fast_track_stop.set()
        if hasattr(self, "smart_dialog"):
            self.smart_dialog.selection = None
            if getattr(self, "_smart_running", False):
                self.smart_dialog.stopped(
                    "Project changed. Completed results stay in their project."
                )
            self.smart_dialog.hide()
        self._smart_selections = {}
        self._smart_running = False
        self._project_review_key = None
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self._analysis_stop = True
        self._propagate_stop = True
        self._sam_jobs.clear()
        self._video_session = VideoSession()
        self.project = project
        self.project_name = project.name
        self.class_names = list(project.class_names)
        self.action_names = list(project.action_names)
        self.project_dir = project.directory
        self._dirty = False
        from ..annotation.quality import QualityConfig

        self.quality_config = QualityConfig(**project.settings.get("quality", {}))
        self._click_backend = project.settings.get("click_backend", "auto")
        self._sticky_draw = project.settings.get("sticky_draw", False)
        self._update_action_buttons()
        self._refresh_labels_display()
        self._refresh_project_dir_display()
        # The new project decides whether boxes carry track ids, so the track
        # controls have to follow it rather than the one just closed.
        self._refresh_track_ui()
        # The checkpoint belongs to the project, so the chip that reports it
        # has to change with the project rather than with the dialog.
        self._refresh_model_ui()
        self._refresh_sidebar()
        self._render_canvas()
        self._render_timeline()

    def _save_workspace(self, path: Path):
        project = self._sync_project()
        worker = getattr(self, "_workspace_save_thread", None)
        if worker is not None and worker.is_alive():
            worker.join()
        from ..data.annotation import save_annotation

        save_annotation(project.dataset(path.parent), str(path), preserve_derived=True)
        # A Save As still includes a manifest beside the exported dataset.
        project.save(path.parent / PROJECT_FILE)
        self.annotation_path = path
        self.project_dir = path.parent
        self._dirty = False
        self._save_due = None
        self._last_saved = time.time()
        self._refresh_save_chip()
        self._refresh_workspace()
        self._status(f"Saved {len(project.videos)} videos to {path.parent}")

    def _autosave_workspace(self):
        if self._save_due is None or time.time() < self._save_due:
            return
        project = self._sync_project()
        if not project.videos or self.project_dir is None:
            return
        thread = getattr(self, "_workspace_save_thread", None)
        if thread is not None and thread.is_alive():
            return
        path = self.project_dir / PROJECT_FILE
        # The open video is always re-detached: ``visited``, ``last_frame`` and
        # ``ui`` change as the user merely looks around, and deliberately do
        # not mark the video dirty. Every other video's row comes from the
        # cache unless something called ``touch()`` on it. Without this, a
        # twenty-video study with a detector run in it spent three seconds of
        # frozen window every four seconds re-serialising candidates nobody
        # had touched.
        data, state = project.snapshot(
            path.parent, always=(self._video_session.video_id,)
        )
        revisions = {v.video_id: v.revision for v in project.videos.values()}
        position_revision = self._workspace_revision
        self._save_due = None

        def work():
            try:
                save_snapshot(path, data, state)
            except Exception as exc:  # noqa: BLE001 - report worker failures on the UI thread.
                message = str(exc)
                self._post(lambda: self._status(f"Autosave failed: {message}. Use Save As."))
                self._post(lambda: setattr(self, "_save_due", time.time() + 15))
                return

            def finish():
                if self.project is not project:
                    return
                for vid, revision in revisions.items():
                    video = project.videos[vid]
                    if video.revision == revision:
                        video.dirty = False
                self._dirty = any(v.dirty for v in project.videos.values())
                if self._dirty or position_revision != self._workspace_revision:
                    self._save_due = time.time() + 4
                self._last_saved = time.time()
                self._refresh_save_chip()

            self._post(finish)

        self._workspace_save_thread = threading.Thread(
            target=work, daemon=True, name="yoake-autosave"
        )
        self._workspace_save_thread.start()

    def _refresh_workspace(self):
        import dearpygui.dearpygui as dpg

        project = self._ensure_project()
        ids = list(project.videos)
        current = self._video_session.video_id
        position = ids.index(current) + 1 if current in ids else 0
        name = self.video_path.name if self.video_path else "No video open"
        if dpg.does_item_exist("workspace_identity"):
            dpg.set_value(
                "workspace_identity",
                f"Project: {self.project_name} | Video: {name} | {position} / {len(ids)}",
            )
        browser = getattr(self, "video_browser", None)
        if browser is not None:
            browser.refresh(project, current)

    def show_video_browser(self):
        from .video_browser import VideoBrowser

        if not hasattr(self, "video_browser"):
            self.video_browser = VideoBrowser(self)
        self.video_browser.show()
        self._refresh_workspace()

    def add_video_files(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            files = filedialog.askopenfilenames(
                title="Add project videos",
                filetypes=[
                    ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                    ("All files", "*.*"),
                ],
            )
        finally:
            root.destroy()
        self._add_videos([Path(f) for f in files])

    def add_video_folder(self, recursive=False):
        from .file_dialogs import ask_dir as _tk_ask_dir

        folder = _tk_ask_dir("Add a folder of videos")
        if folder is not None:
            self._add_videos(scan_folder(folder, recursive=recursive))

    def _add_videos(self, paths):
        project = self._ensure_project()
        added = [project.add_video(path) for path in paths]
        if added:
            self._dirty = True
            self._save_due = time.time() + 4
            if self.cap is None:
                self.switch_video(added[0].video_id)
        self._refresh_workspace()

    def collect_videos(self):
        """Copy every referenced capture into ``<project>/videos/``.

        Relative paths already survive moving the *whole* tree, and the
        recorded absolute path survives moving the project alone. Copying the
        captures in is what survives both at once - a folder handed to someone
        else, or restored onto a machine that never had the capture drive.
        """
        project = self._sync_project()
        if project.directory is None:
            self._status("Set a project folder first (File > Set Project Folder).")
            return
        outside = [
            v
            for v in project.videos.values()
            if v.path is not None and not ProjectLayout(project.directory).contains(v.path)
        ]
        if not outside:
            self._status("Every video is already inside the project folder.")
            return
        self._status(f"Copying {len(outside)} video(s) into the project folder...")

        def work():
            try:
                imported, failures = project.import_media()
            except (OSError, ValueError) as exc:  # reported on the UI thread
                message = str(exc)
                self._post(lambda: self._status(f"Could not copy videos: {message}"))
                return
            note = f"Copied {imported} video(s) into the project folder."
            if failures:
                note += " Failed: " + "; ".join(failures[:3])

            def finish():
                self._status(note)
                self._save_due = time.time() + 1
                self._refresh_workspace()

            self._post(finish)

        self._collect_thread = threading.Thread(
            target=work, daemon=True, name="yoake-collect-videos"
        )
        self._collect_thread.start()

    def locate_video(self, video_id):
        from .file_dialogs import open_file as _tk_open_file

        path = _tk_open_file("Locate video", [("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")])
        if path is not None:
            try:
                self.project.relink(video_id, path)
                self._save_due = time.time() + 4
                self.switch_video(video_id)
            except (ValueError, OSError) as exc:
                self._status(str(exc))

    def export_project(self, out):
        # Export creates a new image layout, including for imported datasets.
        # Commit matching paths before writing images so later autosaves agree.
        project = self._sync_project()
        project.metadata["image_root"] = ""
        for video in project.videos.values():
            for index in video.annotations:
                video.frame_metadata.setdefault(index, {})[
                    "image_path"
                ] = f"images/{video.video_id}/{index:06d}.jpg"
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # The export is a self-contained copy: it carries its own manifest so
        # it can be opened on its own. It must NOT become the live project's
        # home, or exporting into <project>/datasets would quietly move the
        # study one level down and leave the real folder behind.
        if self.project_dir is not None:
            self._save_workspace(ProjectLayout(self.project_dir).annotations)
        # A deliberate export, like Ctrl+S, is never anything but a full
        # detach. (The differing ``base`` would miss the row cache anyway;
        # saying so beats relying on it.)
        data, state = project.snapshot(out, always=tuple(project.videos))
        save_snapshot(out / PROJECT_FILE, data, state)
        self._status(f"Exporting frames to {out}")
        videos = [(v.video_id, v.path, sorted(v.annotations)) for v in project.videos.values()]

        def work():
            import cv2

            total = 0
            failed = []
            for vid, path, frames in videos:
                if not frames:
                    continue
                cap = cv2.VideoCapture(str(path)) if path is not None else None
                try:
                    if cap is None or not cap.isOpened():
                        raise OSError("Missing or unreadable video")
                    folder = out / "images" / vid
                    folder.mkdir(parents=True, exist_ok=True)
                    for index in frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        if not ok or not cv2.imwrite(str(folder / f"{index:06d}.jpg"), frame):
                            raise OSError(f"Could not export frame {index}")
                        total += 1
                except Exception as exc:  # noqa: BLE001 - isolate export failures per video.
                    failed.append(f"{vid}: {exc}")
                finally:
                    if cap is not None:
                        cap.release()
            message = f"Exported {total} frames." + (
                " Failed: " + "; ".join(failed) if failed else ""
            )
            self._post(lambda: self._status(message))

        self._export_thread = threading.Thread(target=work, daemon=True, name="yoake-export")
        self._export_thread.start()
