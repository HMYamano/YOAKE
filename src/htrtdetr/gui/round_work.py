"""The round's work list: which frames it asked for, and how many are done.

A round hands the user 50-200 frames and then has to answer one question over
and over until they are finished: *how many are left, and where is the next
one*. Nothing in the annotator could answer it before, because the selection
was a thing the timeline drew markers from rather than a list of work.

Three facts and two verbs, and they are here rather than in ``annotator.py``
because they are one responsibility with one source of truth - the selection,
counted, never remembered. Nothing in this module stores a count: every number
is derived on read, so there is no state to be stale and no second place for
"is this frame done" to disagree with ``annotations``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..annotation import finalpass
from ..annotation import round as round_mod
from ..annotation import sampling as sampling_mod
from . import shortcuts

__all__ = ["RoundWorkMixin"]


class RoundWorkMixin:
    """Mixed into ``AnnotationApp``. Reads its selection, annotations and inbox."""

    def chosen_frames(self) -> List[int]:
        """Every frame this round asks for in this video, in order.

        The chosen frames **and** the frames inside a chosen clip, which is
        what separates this from :meth:`sample_frames`: the timeline marks the
        anchors, and the work list is the spans.
        """
        selection = self._selection
        return sorted(selection.frame_set()) if selection is not None else []

    def round_progress(self) -> Any:
        """How far this video is through the round's frames."""
        return sampling_mod.progress(self._selection, self.annotations)

    def project_round_progress(self) -> Any:
        """The same, summed over every video in the project.

        Counted rather than assumed: a round spends its budget project-wide,
        so "31 of 200" is a different and more useful number than the one the
        open video can give, and it is the one that says whether the round is
        finished.
        """
        project = self._ensure_project()
        chosen = checked = 0
        for video in project.videos.values():
            part = sampling_mod.progress(
                getattr(video, "selection", None), video.annotations
            )
            chosen += part.chosen
            checked += part.checked
        return sampling_mod.SelectionProgress(chosen=chosen, checked=checked)

    def goto_next_chosen_frame(self) -> None:
        """Go to the next frame this round chose that has not been looked at.

        Wraps, because the work list is a set and not a path: a user who
        started in the middle of the video has frames behind them, and a
        button that goes quiet at the end of a list it has not finished is
        the one that gets reported as broken.
        """
        wanted = [f for f in self.chosen_frames() if f not in self.annotations]
        if not wanted:
            if self.chosen_frames():
                self._status(
                    "Every frame this round chose has been checked. Train on "
                    "these labels when you are ready."
                )
            else:
                self._status(
                    "This round chose nothing in this video. "
                    f"{shortcuts.label_for('auto.annotate')} chooses frames to label."
                )
            return
        nxt = next((f for f in wanted if f > self.current_frame_idx), wanted[0])
        self.goto_frame(int(nxt))
        done = self.round_progress()
        self._status(
            f"{done.checked} of {done.chosen} checked in this video. "
            f"Draw the animals, or press {shortcuts.label_for('review.mark_checked')} "
            "if there is nothing here."
        )

    def mark_frame_checked(self) -> None:
        """Record that this frame was looked at and holds nothing.

        An empty frame is a decision, not a gap. The round's counter asks
        whether a chosen frame has been *looked at*, and the only way a user
        can answer yes about a frame with no animal in it is to be able to say
        so - otherwise the hardest frames the sampler found, which are
        disproportionately the empty ones, pin the counter below its target
        forever. It is also a training sample: a detector that has never been
        shown an empty arena learns that there is always a fly in one.
        """
        frame = self.current_frame_idx
        if self.annotations.get(frame):
            self._status(
                "This frame has boxes on it, so it is already checked. Delete "
                "them first if it should be empty."
            )
            return
        if frame in self.annotations:
            self._status("Already marked as checked with nothing to box.")
            return
        self._push_undo()
        self.annotations[frame] = []
        self._mark_dirty()
        self._render_timeline()
        self._refresh_inspector()
        done = self.round_progress()
        self._status(
            "Nothing here - checked."
            + (f" {done.checked} of {done.chosen} in this video." if done.chosen else "")
        )


    def _examined_frames(self) -> Optional[List[int]]:
        """The frames the checks are entitled to have an opinion about.

        Every frame something looked at: the ones a human annotated, the ones
        a candidate is on, and the ones this round chose. Not
        ``range(num_frames)`` - that was the old argument and it is only
        correct after a dense pass. A round scans one frame in sixty, so
        claiming an opinion about all 108,000 declares 107,900 of them to be
        places the detector found nothing, and the inbox fills with a "missing
        animal" for each until :func:`quality.analyze`'s per-kind cap stops it.

        ``None`` when there is nothing to scope by, which is what
        :func:`quality.analyze` reads as "use the frames you were given".
        """
        frames = set(self.annotations)
        frames.update(self._candidate_map())
        selection = getattr(self._video_session, "selection", None) if self._video_session else None
        if selection is not None:
            frames.update(selection.frame_set())
        return sorted(frames) if frames else None


    def _refresh_round_block(self) -> None:
        """Keep the Review dock's round counter true.

        Called wherever the count can have changed, which is anywhere a frame
        gains its first boxes or a new selection arrives. It is a read of
        existing state - ``progress`` counts, it does not remember - so calling
        it too often is cheap and calling it too rarely shows a stale number.
        """
        inbox = getattr(self, "inbox", None)
        if inbox is None or not getattr(inbox, "_built", False):
            return
        # The final pass supersedes the round: once a video has been looked at
        # end to end, its round selection is history and the work in front of
        # the user is the walk through what the pass found. Two blocks both
        # claiming to be "what to do next" is one too many.
        sentence = self.final_progress_sentence()
        if sentence:
            inbox.set_round(None)
            _segments, stops = self.final_stops()
            state = finalpass.PassState.from_dict(self._video_session.final_pass)
            ready, _why = finalpass.can_sign_off(state, stops)
            inbox.set_final(sentence, ready=ready)
            return
        inbox.set_final("")
        rows = [r for r in (self._rounds_log().get("rounds") or []) if isinstance(r, dict)]
        inbox.set_round(
            self.round_progress(),
            project=self.project_round_progress(),
            note=round_mod.series_note(rows, self._next_round_number()),
            round_number=self._next_round_number(),
        )
