"""The five menus, built onto the open menu bar.

Lifted out of :mod:`htrtdetr.annotator` whole: the menu bar is a flat list of
destinations with no state of its own, which makes it the cheapest thing in
that module to read somewhere else. ``app`` is the :class:`AnnotationApp`; the
callbacks are its methods, unchanged.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from . import shortcuts


def build(app) -> None:
    """Five menus, and every row in them is a place, not a synonym.

    The count matters here more than anywhere else in the window: the
    menus were the single largest block of controls a first-time user
    could reach, and a third of them were second names for something
    already on screen. *Load Annotation...* was the same callback as
    *Open Project...*; *Pre-label videos with the model...* opened the
    same window as *Pick frames to label...* and, without a checkpoint,
    with the same checkbox hidden; the five Settings rows were tabs of a
    dialog that is one row above them. Nothing here lost a destination -
    the duplicates lost their second door.
    """
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            # Starting a study is the first row, not the last. It used to be
            # the twelfth, under Save and Export Frames, which is the one
            # place in this menu a person who has nothing yet will not look.
            dpg.add_menu_item(label="New Project...", callback=app._show_setup_screen)
            dpg.add_menu_item(label="Open Project...", callback=app._load_annotation_dialog)
            dpg.add_separator()
            dpg.add_menu_item(label="Add Video Files...", callback=app.add_video_files)
            dpg.add_menu_item(label="Add Video Folder...", callback=app.add_video_folder)
            dpg.add_menu_item(label="Open Video...  Ctrl+O", callback=app._open_video_dialog)
            dpg.add_separator()
            # Two rows that both opened a file picker for the same kind of
            # file and differed by a trailing phrase naming an internal
            # feature. They are one destination with two uses, so they are
            # one row with two uses under it, and each says what the file
            # becomes rather than which subsystem consumes it.
            with dpg.menu(label="Load Predictions"):
                dpg.add_menu_item(
                    label="As candidates to review...",
                    callback=app._import_predictions_dialog,
                )
                dpg.add_menu_item(
                    label="As boxes for Click animal...",
                    callback=app._load_predictions_dialog,
                )
            dpg.add_separator()
            dpg.add_menu_item(label="Save  Ctrl+S", callback=app.save_annotation)
            dpg.add_menu_item(label="Save As...", callback=app._save_as_dialog)
            dpg.add_menu_item(
                label="Save and Export Frames...", callback=app._save_and_export_dialog
            )
            # What the whole study is for, and a different thing from the row
            # above it: that one writes the frames you labelled, this one
            # writes every frame of every video you signed off.
            dpg.add_menu_item(
                label="Export the verified dataset...",
                callback=lambda: app.open_release_export(),
            )
            dpg.add_separator()
            dpg.add_menu_item(
                label="Set Project Folder...", callback=app._set_project_dir_dialog
            )
            dpg.add_menu_item(
                label="Open Project Folder", callback=app._open_project_folder
            )

        with dpg.menu(label="Edit"):
            dpg.add_menu_item(label=shortcuts.with_key("Undo", "edit.undo"), callback=app.undo)
            dpg.add_menu_item(label=shortcuts.with_key("Redo", "edit.redo"), callback=app.redo)
            dpg.add_menu_item(label="Delete Selected  Del", callback=app.delete_selected)
            # One verb for carrying a box forward. "Propagate from Prev"
            # was the only place in the app that used the word, and it
            # describes the mechanism rather than the result.
            dpg.add_menu_item(
                label="Copy Boxes from Previous Frame  Space",
                callback=app.propagate_from_prev,
            )

        with dpg.menu(label="Project"):
            dpg.add_menu_item(label="Videos...", callback=app.show_video_browser)
            # The scope is in the label, and the key is not. Ctrl+R and the
            # top-bar button open this same window on the *current video*;
            # this row opens it on the whole project. Both were labelled
            # "Pick frames to label..." and this one also advertised Ctrl+R,
            # so the menu promised a key that does something else.
            dpg.add_menu_item(
                label="Pick frames to label (whole project)...",
                callback=lambda: app.open_smart_annotate("project"),
            )
            dpg.add_menu_item(
                label="Check Project Quality",
                callback=lambda: app.run_project_quality("Entire project"),
            )
            dpg.add_menu_item(
                label="Copy Videos into Project Folder", callback=app.collect_videos
            )
            dpg.add_separator()
            # The other half of the loop, and it runs in this window. The
            # third row is the way out to the production run, which is a
            # different job: every stage, every label, once, at the end.
            dpg.add_menu_item(
                label="Train on these labels...",
                callback=lambda: app.open_train_round(),
            )
            dpg.add_menu_item(
                label="Find animals with this model...",
                callback=lambda: app.open_smart_annotate("project", "analyze"),
            )
            dpg.add_menu_item(
                label="Open the workflow window (full training)...",
                callback=lambda: app.open_workflow_page(),
            )
            dpg.add_separator()
            # The end of the loop, and the only row in this menu that is taken
            # once. It is not on the top bar's loop row for exactly that
            # reason: beside Analyze it would be pressed by habit on round
            # three, and it costs two hours when it is.
            dpg.add_menu_item(
                label="Check every frame (final pass)...",
                callback=lambda: app.open_final_pass(),
            )
            dpg.add_separator()
            # One row, not seven. The six that were here named the six tabs
            # of the dialog the seventh opens - and that dialog shows its tab
            # bar the moment it appears, so the menu was a copy of a
            # navigation control that is already visible on arrival.
            dpg.add_menu_item(label="Settings...", callback=lambda: app.open_settings())

        with dpg.menu(label="Review"):
            # Scope in the label, because the two inboxes differ only in
            # scope and used to differ only in whether the word "Project"
            # was in front. "Review Next across videos" joins them here
            # instead of living in the Project menu: all four rows are the
            # same job at three scopes.
            dpg.add_menu_item(label="Review Inbox (this video)", callback=app.toggle_inbox)
            dpg.add_menu_item(
                label="Review Inbox (all videos)", callback=app.show_project_review
            )
            dpg.add_menu_item(label="Review issues one by one", callback=app.open_review_mode)
            dpg.add_menu_item(
                label="Go to the next video needing review", callback=app.review_next_project
            )
            dpg.add_separator()
            # Clip propagation sits under Review, beside Smart Annotate,
            # because a clip is what Smart Annotate proposed and the boxes
            # it produces go straight into the Review Inbox. "Continue",
            # not "Follow": the rail's Continue button does this for one
            # animal on one frame, and these two do it for every animal
            # over a range. Same verb, wider scope.
            dpg.add_menu_item(
                label="Continue every animal across this clip",
                callback=app.propagate_clip_here,
            )
            dpg.add_menu_item(
                label="Continue every animal across all sampled clips",
                callback=app.propagate_sampled_clips,
            )
            dpg.add_separator()
            dpg.add_menu_item(label="Re-check everything", callback=app.rebuild_copilot)
            # "Expected number of animals..." used to be the eighth row here.
            # It is a project setting, it lives on the Settings dialog's
            # Project tab, and a settings field reachable from a menu called
            # Review is a field nobody finds twice.

        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="Guided Annotation", callback=app.show_onboarding)
            dpg.add_menu_item(label="Quick Start and Shortcuts", callback=app._show_help)
            dpg.add_separator()
            # The second door to the Activity window. The pill in the top bar
            # is the first and the one people use; this row is for the person
            # who has been told "send me the log" and is looking under Help
            # for it, which is where every other application keeps it.
            dpg.add_menu_item(
                label=shortcuts.with_key("Activity and logs", "view.activity"),
                callback=lambda: app.show_activity(),
            )
