"""
htrtdetr.annotation — the annotation workflow, as logic rather than as a GUI.

Everything an *Annotation Copilot* has to decide lives here, and nothing that
draws it does. The modules answer, in order:

===================  =====================================================
:mod:`status`        Is this box verified, predicted, or waiting for a human?
:mod:`tracklet`      Which candidates are one animal over one run of frames,
                     so that Accept costs one click and not three hundred?
:mod:`quality`       Which moments in this run would a careful human want to
                     see, and why?
:mod:`review_queue`  The worklist over those moments: filter, navigate, decide.
:mod:`intervals`     Behavior as spans of time, projected back onto the
                     per-frame ``action_id`` the file format already stores.
:mod:`keyframes`     The keyframe scheme: label one image (boxes and
                     behaviors, no track ids), take the time from the video.
:mod:`behaviors`     One behavior per animal, or one per group (multi-head).
:mod:`features`      Cheap per-frame descriptors — motion, difference, focus.
:mod:`sampling`      Given a budget, which frames and clips should a human
                     spend it on?
:mod:`propagation`   A correction on one frame, carried to its neighbours.
:mod:`cache`         None of the above computed twice for the same video.
===================  =====================================================

**Purity contract, and it is the reason this is a package and not part of the
annotator.** No module here imports ``dearpygui`` or ``torch``, at module scope
or inside a function. ``cv2`` and ``numpy`` are imported lazily, inside the two
functions in :mod:`features` that cannot avoid them. Consequences:

* Every rule is unit-testable on the GUI's Python, which has no torch
  (measured: 3.14.3 here has dearpygui and OpenCV, not torch), and on the
  training Python, which has no dearpygui.
* Heavy model work stays where it already belongs — a child process, through
  :mod:`htrtdetr.gui.sidecar` — so the GUI cannot be frozen or crashed by it.
* A headless script can build a review queue, and a test can build one with no
  video at all.
"""

from __future__ import annotations

__all__ = [
    "status", "tracklet", "quality", "review_queue", "intervals", "keyframes",
    "behaviors", "features", "sampling", "propagation", "cache",
]
