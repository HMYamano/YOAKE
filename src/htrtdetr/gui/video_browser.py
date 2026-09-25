"""Video browser: explicit status, selection and project operation entry points."""

from __future__ import annotations

import time

import dearpygui.dearpygui as dpg

from ..annotation.project import STATUS_LABELS
from . import theme, widgets


class VideoBrowser:
    #: Preferred size. The table wants width, but 990 px is wider than the
    #: annotator's own minimum window, so it is a preference that
    #: :func:`widgets.place_dialog` shrinks rather than a size that can push
    #: the Locate and Mark Done buttons off the right-hand edge.
    WIDTH, HEIGHT = 990, 570

    def __init__(self, app):
        self.app = app
        self.tag = "project_videos"

    def show(self):
        if not dpg.does_item_exist(self.tag):
            with dpg.window(
                tag=self.tag,
                label="Project Videos",
                width=self.WIDTH,
                height=self.HEIGHT,
                show=True,
            ):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Add Files...", callback=lambda: self.app.add_video_files()
                    )
                    dpg.add_button(
                        label="Add Folder...",
                        callback=lambda: self.app.add_video_folder(
                            dpg.get_value("browser_recursive")
                        ),
                    )
                    dpg.add_checkbox(label="Include subfolders", tag="browser_recursive")
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        ["All", *STATUS_LABELS.values()],
                        tag="browser_filter",
                        default_value="All",
                        width=150,
                        callback=lambda: self.app._refresh_workspace(),
                    )
                    dpg.add_input_text(
                        tag="browser_search",
                        hint="Search filename",
                        width=280,
                        callback=lambda: self.app._refresh_workspace(),
                    )
                    dpg.add_button(label="Select visible", callback=self._select_visible)
                    dpg.add_button(label="Clear selection", callback=self._clear_selection)
                with dpg.group(horizontal=True):
                    # One button, not two. "Pre-label with the model..." next
                    # to it opened the same window with one checkbox already
                    # ticked - and with no checkpoint, that checkbox is hidden
                    # before the window is drawn, so the two were the same
                    # click under two names on the day it mattered most.
                    dpg.add_button(
                        label="Pick frames to label...",
                        callback=lambda: self.app.open_smart_annotate("project"),
                    )
                    dpg.add_button(
                        label="Quality Check",
                        callback=lambda: self.app.run_project_quality("Selected videos"),
                    )
                    dpg.add_button(label="Mark Done", callback=lambda: self._mark("done"))
                    dpg.add_button(
                        label="Mark In progress", callback=lambda: self._mark("in_progress")
                    )
                dpg.add_text(
                    "Visited frames describe navigation, not annotation completeness. "
                    "Done is a manual status.",
                    color=theme.TEXT_DIM,
                    wrap=-1,
                )
                with dpg.child_window(tag="browser_rows", height=-1):
                    pass
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.show_item(self.tag)

    def visible(self, project):
        status = dpg.get_value("browser_filter")
        search = (dpg.get_value("browser_search") or "").casefold()
        return [
            v
            for v in project.videos.values()
            if (status == "All" or STATUS_LABELS[v.display_status] == status)
            and search in (v.path.name if v.path else v.video_id).casefold()
        ]

    def refresh(self, project, current):
        if not dpg.does_item_exist("browser_rows") or not dpg.is_item_shown(self.tag):
            return
        dpg.delete_item("browser_rows", children_only=True)
        with dpg.table(
            parent="browser_rows",
            header_row=True,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            for name in (
                "Select",
                "Video",
                "Status",
                "Visited",
                "Candidates / issues",
                "Last frame",
                "Duration",
                "File",
            ):
                dpg.add_table_column(label=name)
            for video in self.visible(project):
                vid = video.video_id
                with dpg.table_row():
                    dpg.add_checkbox(
                        default_value=vid in project.selected_videos,
                        callback=self._select,
                        user_data=vid,
                    )
                    label = ("> " if vid == current else "") + (
                        video.path.name if video.path else vid
                    )
                    dpg.add_button(
                        label=label + (" *" if video.dirty else ""),
                        callback=lambda s, a, u: self.app.switch_video(u),
                        user_data=vid,
                    )
                    dpg.add_text(STATUS_LABELS[video.display_status])
                    dpg.add_text(f"{len(video.visited)} / {video.num_frames or '?'}")
                    dpg.add_text(f"{video.pending} / {video.queue.progress().remaining}")
                    dpg.add_text(str(video.current_frame))
                    dpg.add_text(
                        f"{video.num_frames / video.fps:.1f} s"
                        if video.num_frames
                        else "Not opened"
                    )
                    if video.missing:
                        dpg.add_button(
                            label="Missing: Locate...",
                            callback=lambda s, a, u: self.app.locate_video(u),
                            user_data=vid,
                        )
                    else:
                        dpg.add_text("Available")

    def _select(self, sender, value, video_id):
        selected = self.app.project.selected_videos
        selected.add(video_id) if value else selected.discard(video_id)

    def _select_visible(self):
        self.app.project.selected_videos.update(v.video_id for v in self.visible(self.app.project))
        self.app._refresh_workspace()

    def _clear_selection(self):
        self.app.project.selected_videos.clear()
        self.app._refresh_workspace()

    def _mark(self, status):
        for vid in self.app.project.selected_videos:
            video = self.app.project.videos[vid]
            video.status = status
            video.touch()
        self.app._dirty = True
        self.app._save_due = time.time() + 4
        self.app._refresh_workspace()
