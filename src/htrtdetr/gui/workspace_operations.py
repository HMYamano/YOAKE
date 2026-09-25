"""Batch results, project review and guided continuation actions."""

from __future__ import annotations

import copy
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

from ..annotation import intervals, quality, review_queue, tracklet
from ..annotation.batch import BatchRunner, VideoInput, check_video
from . import auto_label, commands


class WorkspaceOperations:
    def _queue_tracker_step(self, frame_idx, frame_bgr):
        """Legacy A-mode tracker updates run off-thread and return candidates."""
        video, project = self._video_session, self.project
        if video.transient.get("tracker_busy"):
            self._status(
                "Tracker is still working. Wait for its candidates before stepping again."
            )
            return
        trackers = list(self._trackers.items())
        seeds = {}
        for tid, _tracker in trackers:
            if video.transient.get("tracker_frames", {}).get(tid) != frame_idx - 1:
                continue
            seed = video.transient.get("tracker_seeds", {}).get(tid)
            for obj in video.annotations.get(frame_idx - 1, []):
                if obj.track_id == tid:
                    seed = obj
                    break
            if seed is not None:
                seeds[tid] = copy.deepcopy(seed)
        image = frame_bgr.copy()
        video.transient["tracker_busy"] = True

        def work():
            candidates = []
            failed = []
            for tid, tracker in trackers:
                seed = seeds.get(tid)
                if seed is None:
                    failed.append(tid)
                    continue
                try:
                    ok, xywh = tracker.update(image)
                    if not ok:
                        failed.append(tid)
                        continue
                    x, y, w, h = (int(v) for v in xywh)
                    box = [max(0, x), max(0, y), min(video.width, x + w), min(video.height, y + h)]
                    if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                        failed.append(tid)
                        continue
                    obb = None
                    if seed.obb is not None:
                        obb = [
                            (box[0] + box[2]) / 2,
                            (box[1] + box[3]) / 2,
                            box[2] - box[0],
                            box[3] - box[1],
                            seed.obb[4],
                        ]
                    candidates.append(
                        auto_label.Candidate(
                            box,
                            obb=obb,
                            track_id=tid,
                            class_id=seed.class_id,
                            action_id=seed.action_id,
                            source="track",
                            provenance={
                                "operation": "next_frame_tracking",
                                "human_review": "pending",
                            },
                        )
                    )
                except Exception:  # noqa: BLE001 — native tracker failure stays local to its track.
                    failed.append(tid)

            def finish():
                video.transient["tracker_busy"] = False
                if self.project is not project:
                    return
                for tid in failed:
                    video.transient.get("_trackers", {}).pop(tid, None)
                for candidate in candidates:
                    video.transient.setdefault("tracker_frames", {})[candidate.track_id] = frame_idx
                self._merge_video_candidates(video, {frame_idx: candidates})
                video.touch()
                self._save_due = time.time() + 4
                if video is self._video_session:
                    self._refresh_candidate_ui()
                    self._render_canvas()
                    self._render_timeline()
                if failed:
                    self._status(f"Tracking stopped for {failed}; select a box and continue again.")

            self._post(finish)

        threading.Thread(target=work, daemon=True, name="yoake-next-frame-track").start()

    def _sam_frame_id(self, index):
        """A helper's frame cache is global; give each video a disjoint range."""
        project = self._ensure_project()
        namespaces = getattr(self, "_sam_namespaces", None)
        if namespaces is None:
            namespaces = self._sam_namespaces = {}
        key = (project.identity, self._video_session.video_id)
        namespace = namespaces.setdefault(key, len(namespaces) + 1)
        return (namespace << 32) + int(index)

    def _set_end_behavior(self, value):
        self.project.settings["end_of_video"] = (
            "next_unfinished" if value == "Open next unfinished video" else "stay"
        )
        self._mark_dirty()

    def _set_continue_method(self, value):
        self.project.settings["continue_method"] = value
        self._mark_dirty()

    # ── the way out to the production run ────────────────────────────
    #: ``page -> (button word, what that page is for)``. One entry now: the
    #: active-learning round and its inference pass happen in the annotator,
    #: and the workflow window is where the *final* model is made - every
    #: stage, every label, once, with the full controls.
    WORKFLOW_PAGES = {
        "workflow": (
            "Workflow",
            "The full training run: stages 1-4, the metrics, and the "
            "frame-by-frame review of a finished model. Done once, when the "
            "labels are finished - not every round.",
        ),
    }

    def open_workflow_page(self, page: str = "workflow"):
        """Open the workflow window on this project, for the production run.

        This used to be how the annotator reached training at all, which made
        the active-learning loop a thing you left the labelling window to
        continue. Training and inference for a round now happen here
        (:mod:`gui.train_operations`), and this is the door to the other job:
        the long run over finished labels.

        Two honest preconditions, both named rather than silently worked
        around: the workflow window works on a project folder, so there has to
        **be** one, and what is on disk has to be what is on screen - hence
        the save.
        """
        import subprocess
        import sys

        word, _purpose = self.WORKFLOW_PAGES.get(page, self.WORKFLOW_PAGES["workflow"])
        if self.project_dir is None:
            self._status(
                f"{word} works on a project folder, and this project has none "
                "yet. Settings > Project > Change folder... sets one."
            )
            self.open_settings("project")
            return
        if self._dirty:
            self.save_annotation()
        root = auto_label.repo_root_for(Path(__file__))
        cmd = [sys.executable, "-u"] + commands.workflow_command(page, str(self.project_dir))
        kwargs = {
            "cwd": str(root),
            "env": auto_label.child_env(root),
            "stdin": subprocess.DEVNULL,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen(cmd, **kwargs)
        except (OSError, ValueError) as exc:
            self._status(f"Could not open the workflow window: {exc}")
            return
        self._status(
            "Opening the workflow window. This annotator stays open - come "
            "back to it when the run is done."
        )

    def show_onboarding(self):
        from .onboarding import show_guide

        show_guide(self)

    def continue_animal(self, method=None):
        from ..annotation.objects import AnnObject

        obj = self._selected
        if obj is None or self.cap is None:
            self._status("Select an animal first. Press W to draw or C to click an animal.")
            return
        method = method or self.project.settings.get("continue_method", "Automatic")
        if method == "Automatic":
            method = (
                "Accurate propagation"
                if self.video_capability().available
                else "Fast tracking" if self._capability().can_track else "Copy to next frame"
            )
        # Fast tracking and accurate propagation both carry one animal's
        # identity forward, which a project without track ids has not claimed.
        # Copying the box to the next frame is the part that still means
        # something, so that is what Continue does there.
        if not self.use_track_ids and method != "Copy to next frame":
            method = "Copy to next frame"
        if method == "Accurate propagation":
            self.propagate_selection()
            return
        project, video = self.project, self._video_session
        seed = copy.deepcopy(obj)
        frame = self.current_frame_idx
        if method == "Copy to next frame":
            if frame + 1 >= self.num_frames:
                self._status("This animal is already on the last frame.")
                return
            existing = self.annotations.get(frame + 1, [])
            if seed.track_id >= 0 and any(o.track_id == seed.track_id for o in existing):
                self.goto_frame(frame + 1)
                self._status("This track already has an annotation on the next frame.")
                return
            self._push_undo(frames=[frame, frame + 1])
            record = seed.to_dict()
            record.pop("object_id", None)
            copied = AnnObject.from_dict(record)
            self.annotations.setdefault(frame + 1, []).append(copied)
            self.goto_frame(frame + 1)
            self._select(copied)
            self._mark_dirty()
            # Say which of the three methods ran. Continue resolves one of them
            # from what this machine can do, so pressing it and being moved a
            # frame with no word said leaves the user unable to tell a copy
            # from a tracking run that quietly did nothing.
            self._status(
                "Continue: copied this box to the next frame. Adjust it, or "
                "press Continue again."
            )
            return
        if method != "Fast tracking":
            self._status(f"Unknown continuation method: {method}")
            return
        running = getattr(self, "_fast_track_thread", None)
        if running is not None and running.is_alive():
            self._status("Fast tracking is already running.")
            return
        if not self._capability().can_track:
            self._show_unavailable("Fast tracking", self._capability().tracker_reason)
            return
        self._fast_track_stop = threading.Event()
        stop = self._fast_track_stop
        path, width, height, num_frames = video.path, video.width, video.height, video.num_frames

        def work():
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:

                def read(index):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read()
                    return image if ok else None

                rows, reason = auto_label.track_forward(
                    read,
                    frame,
                    seed.bbox,
                    min(100, num_frames - frame - 1),
                    width,
                    height,
                    should_stop=stop.is_set,
                )
                candidates = {}
                for index, box in rows:
                    obb = None
                    if seed.obb is not None:
                        x1, y1, x2, y2 = box
                        obb = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, seed.obb[4]]
                    candidates[index] = [
                        auto_label.Candidate(
                            box,
                            obb=obb,
                            track_id=seed.track_id,
                            class_id=seed.class_id,
                            action_id=seed.action_id,
                            source="track",
                            provenance={"operation": "fast_tracking", "human_review": "pending"},
                        )
                    ]

                def finish():
                    if self.project is not project:
                        return
                    self._merge_video_candidates(video, candidates)
                    self._check_session(video)
                    video.touch()
                    self._save_due = time.time() + 4
                    self._refresh_workspace()
                    if video is self._video_session:
                        self.inbox.set_queue(video.queue, fps=video.fps)
                        self._render_canvas()
                        self._render_timeline()
                        self._refresh_inspector()
                    self._status(
                        f"{len(rows)} tracking candidates for {video.video_id}. Review before accepting. {reason}"
                    )

                self._post(finish)
            except Exception as exc:  # noqa: BLE001 - isolate native tracking failures.
                message = str(exc)
                self._post(lambda: self._status(f"Fast tracking failed: {message}"))
            finally:
                cap.release()

        self._fast_track_thread = threading.Thread(
            target=work, daemon=True, name="yoake-fast-track"
        )
        self._fast_track_thread.start()
        self._status(
            "Continue: fast tracking this animal forward. The boxes arrive as "
            "candidates for you to accept."
        )

    def _finish_video_propagation(self, project, video, result, request):
        from .video_segment import build_candidates

        if self.project is not project:
            return
        candidates = build_candidates(result, track_id=request.track_id)
        for row in candidates.values():
            for candidate in row:
                candidate.class_id = request.class_id
                candidate.action_id = request.action_id
                candidate.provenance.update(
                    operation="accurate_propagation", human_review="pending"
                )
        self._merge_video_candidates(video, candidates)
        self._check_session(video)
        video.touch()
        self._save_due = time.time() + 4
        if video is self._video_session:
            self.inbox.set_queue(video.queue, fps=video.fps)
            self._refresh_candidate_ui()
            self._render_canvas()
            self._render_timeline()
        self._refresh_workspace()
        self._status(result.describe(request.track_id))

    # ─── Clips: one export, every animal in it ───────────────────────────
    #
    # Continue follows one animal. A clip is the span a human already agreed to
    # look at, and it usually holds every animal at once, so following them one
    # at a time re-exported and re-encoded the same frames per animal - and the
    # encode is the whole cost of a propagation. Measured on 40 frames with two
    # animals: 5.98 s for the first, 3.09 s for the second over the same
    # export. The saving lives entirely in exporting once, which is why one
    # worker owns the whole clip.

    def _clip_propagation_ready(self, retry=None) -> bool:
        """The same gate Continue's accurate path uses, with clip wording.

        ``retry`` is what the press meant. Starting the helper is asynchronous,
        so the gate cannot answer yes on the first press; it remembers the
        operation instead and replays it once the helper is up.
        """
        if not self.use_track_ids:
            # A propagation's output is a tracklet; a tracklet with no identity
            # cannot be accepted as one animal. Same reason continue_animal
            # drops to a plain copy here.
            self._status(
                "Following a clip needs track IDs, which this project has "
                "turned off. Turn them on in Project > Tracking, or use "
                "Continue to copy boxes forward one frame at a time."
            )
            return False
        if not self._sam_ready():
            self._start_sam(
                on_done=lambda ok: retry() if (ok and retry is not None) else None
            )
            if not self._sam_ready():
                self._status(
                    "Starting the segmentation helper. Following the clip "
                    "will begin as soon as it answers."
                )
                return False
        capability = self.video_capability()
        if not capability.available:
            self._show_unavailable("Clip propagation", capability.reason)
            return False
        if self.cap is None:
            self._status("Open a video first.")
            return False
        running = getattr(self, "_propagate_thread", None)
        if running is not None and running.is_alive():
            # Pressing again is how a minutes-long job stops (P6). Exact about
            # when, because the pass already inside the helper cannot be cut
            # short from here.
            self._propagate_stop = True
            self._status(
                "Stopping after the pass that is already running. Nothing "
                "after it will start."
            )
            return False
        return True

    def _clip_job(self, video, start, end, preferred=None):
        """``(first, last, requests)`` for one clip, or ``None`` with a reason."""
        from . import video_segment as vs

        seed = vs.clip_seed_frame(
            start, end, video.annotations,
            preferred=preferred, num_frames=video.num_frames,
        )
        if seed is None:
            return None, "no boxes inside it"
        prompts = vs.prompts_from_objects(
            video.annotations.get(seed, []),
            fallback_track=max(1, int(self.active_track_id)),
        )
        requests = vs.clip_requests(
            start, end, prompts, preferred_frame=seed, num_frames=video.num_frames,
        )
        if not requests:
            return None, "no box in it could prompt anything"
        first, last = vs.clip_window(start, end, video.num_frames)
        return (first, last, requests), ""

    def propagate_clip(self, start, end, prompt_frame=None):
        """Follow every animal on one frame of a clip across the whole clip."""
        # Resolved before the gate, so a replay seeds from the frame the user
        # was on when they pressed rather than wherever the playhead drifted.
        preferred = prompt_frame if prompt_frame is not None else self.current_frame_idx
        if not self._clip_propagation_ready(
            retry=lambda: self.propagate_clip(start, end, preferred)
        ):
            return False
        video = self._video_session
        job, why = self._clip_job(video, start, end, preferred)
        if job is None:
            self._status(
                f"This clip has {why}. Draw a box (W), click an animal (C), or "
                "use Find every animal in this frame on one frame of it, then try again."
            )
            return False
        return self._start_clip_jobs([job], clips=1)

    def clip_at(self, frame):
        """The sampled clip containing ``frame``, or ``None``.

        Sampled clips, not behavior intervals: a clip is a span Smart Annotate
        proposed and nobody has labelled yet, which is exactly what there is
        something to propose into. An interval is already a human's answer.
        """
        selection = getattr(self, "_selection", None)
        for clip in getattr(selection, "clips", ()) or ():
            if int(clip.start) <= int(frame) <= int(clip.end):
                return clip
        return None

    def propagate_clip_here(self):
        """The sampled clip under the playhead."""
        frame = self.current_frame_idx
        clip = self.clip_at(frame)
        if clip is None:
            self._status(
                "The playhead is not inside a sampled clip. Run Pick frames to label "
                "with a behavior or 'both' target, or use Continue to follow "
                "one animal from here."
            )
            return False
        return self.propagate_clip(clip.start, clip.end, frame)

    def propagate_sampled_clips(self):
        """Every clip Smart Annotate proposed for this video, in order."""
        if not self._clip_propagation_ready(retry=self.propagate_sampled_clips):
            return False
        selection = getattr(self, "_selection", None)
        clips = list(getattr(selection, "clips", ()) or ())
        if not clips:
            self._status(
                "No sampled clips yet. Run Pick frames to label with a behavior or "
                "'both' target first."
            )
            return False
        video = self._video_session
        jobs, skipped = [], 0
        for clip in clips:
            job, _why = self._clip_job(video, clip.start, clip.end)
            if job is None:
                skipped += 1
                continue
            jobs.append(job)
        if not jobs:
            self._status(
                f"None of the {len(clips)} sampled clips has a box to follow. "
                "Annotate one frame in a clip first - the propagation can only "
                "follow animals it was prompted with."
            )
            return False
        return self._start_clip_jobs(jobs, clips=len(jobs), skipped=skipped)

    def _start_clip_jobs(self, jobs, *, clips=1, skipped=0):
        self._propagate_stop = False
        context = (self.project, self._video_session, self._sam_client)
        self._propagate_thread = threading.Thread(
            target=self._clip_propagate_worker,
            args=(jobs, clips, skipped, context),
            name="yoake-clip-propagate",
            daemon=True,
        )
        self._propagate_thread.start()
        total = sum(len(requests) for _f, _l, requests in jobs)
        where = "this clip" if clips <= 1 else f"{clips} clips"
        self._status(f"Following {total} animal(s) across {where}...")
        return True

    def _clip_propagate_worker(self, jobs, clips, skipped, context):
        """One export per clip, reused by every animal in it. Off-thread."""
        import shutil

        import cv2

        from . import video_segment as vs

        project, video, client = context
        pairs = []
        cap = None
        try:
            # Its own capture: seeking one VideoCapture from two threads is not
            # safe, and sharing the annotator's would make the canvas jump.
            cap = cv2.VideoCapture(str(video.path))
            if not cap.isOpened():
                self._post(lambda: self._status("The video could not be opened."))
                return

            def read_frame(index):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                return frame if ok else None

            for ordinal, (first, last, requests) in enumerate(jobs, 1):
                if self._propagate_stop:
                    break
                folder = None
                try:
                    label = "" if clips <= 1 else f"clip {ordinal}/{clips}: "
                    folder, written = vs.export_window(
                        read_frame, first, last,
                        should_stop=lambda: self._propagate_stop,
                        on_progress=lambda done, total, lb=label: (
                            self._post(
                                lambda d=done, t=total, l=lb: self._status(
                                    f"{l}preparing frames {d}/{t}..."
                                )
                            )
                            if done % 25 == 0
                            else None
                        ),
                    )
                    if folder is None or written == 0:
                        continue
                    for index, request in enumerate(requests, 1):
                        if self._propagate_stop:
                            break
                        self._post(
                            lambda i=index, n=len(requests), t=request.track_id, lb=label:
                            self._status(f"{lb}following animal {i}/{n} (track {t})...")
                        )
                        forward = self._one_clip_pass(client, folder, first, request, False)
                        backward = None
                        if first < request.frame and not self._propagate_stop:
                            backward = self._one_clip_pass(client, folder, first, request, True)
                        if forward is None and backward is None:
                            pairs.append((request, None))
                        else:
                            # Forward first: both passes return the prompt
                            # frame, and the forward one should win it.
                            pairs.append((request, vs.merge_results(forward, backward)))
                finally:
                    if folder is not None:
                        shutil.rmtree(folder, ignore_errors=True)

            self._post(
                lambda p=pairs, c=clips, s=skipped: self._finish_clip_propagation(
                    project, video, p, c, s
                )
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            self._post(lambda m=message: self._status(f"Clip propagation failed. {m}"))
        finally:
            if cap is not None:
                cap.release()

    def _one_clip_pass(self, client, folder, first, request, backward):
        from . import video_segment as vs

        req_id = client.propagate(
            frames_dir=str(folder),
            first_frame=first,
            prompt_frame=request.frame,
            points=[list(p) for p in request.points] or None,
            box=list(request.box) if request.box else None,
            obj_id=max(1, int(request.track_id)),
            backward=backward,
        )
        reply = client.wait_for(req_id, timeout=600.0)
        return None if reply is None else vs.result_from_reply(reply)

    def _finish_clip_propagation(self, project, video, pairs, clips, skipped=0):
        from .video_segment import build_candidates, describe_clip_run

        if self.project is not project:
            return
        for request, result in pairs:
            if result is None or not result.count:
                continue
            candidates = build_candidates(
                result, track_id=request.track_id, action_id=request.action_id
            )
            for row in candidates.values():
                for candidate in row:
                    candidate.class_id = request.class_id
                    candidate.provenance.update(
                        operation="clip_propagation", human_review="pending"
                    )
            self._merge_video_candidates(video, candidates)
        # Once, not once per animal: the checks rebuild every tracklet and
        # re-run nine detectors over the whole pending set.
        self._check_session(video)
        video.touch()
        self._save_due = time.time() + 4
        if video is self._video_session:
            self.inbox.set_queue(video.queue, fps=video.fps)
            self._refresh_candidate_ui()
            self._render_canvas()
            self._render_timeline()
        self._refresh_workspace()
        self._status(describe_clip_run(pairs, clips=clips, skipped=skipped))

    def measure_videos(self, videos) -> int:
        """Fill in ``num_frames``/``fps`` for videos that have never been read.

        Length lives in the container header, so this is a metadata read and
        not a decode: measured on the twenty one-hour files this was written
        for, all twenty answered in 80 ms total, 4 ms each.

        It exists because the alternative was worse than slow, it was silent.
        A scope with one unmeasured video quoted no total at all, so the
        dialog that decides how much work to start opened on "Total length
        appears once every video in scope has been read." on the one day it
        matters - the day the project was created and nothing had been opened
        yet - and the run began with no number ever shown.

        Returns how many videos it measured, so a caller can tell whether
        anything changed.
        """
        try:
            import cv2
        except ImportError:  # pragma: no cover - the GUI cannot run without it
            return 0
        from .video_source import probe_capture

        done = 0
        for video in videos:
            if video.num_frames or video.path is None:
                continue
            try:
                if not Path(video.path).exists():
                    continue
                cap = cv2.VideoCapture(str(video.path))
                if not cap.isOpened():
                    cap.release()
                    continue
                probe = probe_capture(cap, default_fps=video.fps or 25.0)
                cap.release()
            except Exception:  # noqa: BLE001 - an unreadable file is not fatal
                continue
            if probe.num_frames > 0:
                video.num_frames = int(probe.num_frames)
                video.fps = float(probe.fps)
                video.width = int(probe.width) or video.width
                video.height = int(probe.height) or video.height
                done += 1
        return done

    def scope_summary(self, scope):
        """``(label, frames, fps, count)`` for what a scope covers.

        Smart Annotate asks for this rather than reading the project itself,
        which is what lets one dialog quote "courtship_01.mp4, 40,850 frames"
        and "6 videos, 241,900 frames" from the same header.
        """
        project = self._ensure_project()
        videos = list(project.scope(scope, self._video_session.video_id))
        if not videos:
            return ("No videos in this scope", 0, 25.0, 0)
        self.measure_videos(videos)
        # A video is measured when it is first read, so a scope can hold
        # videos with no frame count yet. Summing over them anyway quotes
        # "3 videos, 40 frames" for three 40-frame videos, which is worse
        # than quoting no total at all.
        measured = [video for video in videos if video.num_frames]
        frames = (
            sum(int(video.num_frames) for video in measured)
            if len(measured) == len(videos)
            else 0
        )
        rates = [float(video.fps) for video in videos if video.fps]
        fps = rates[0] if rates else 25.0
        if len(videos) == 1:
            first = videos[0]
            name = first.path.name if first.path else first.video_id
            return (name, frames, fps, 1)
        return (f"{len(videos)} videos", frames, fps, len(videos))

    def _batch_inputs(self, scope):
        result = []
        for video in self.project.scope(scope, self._video_session.video_id):
            candidates = (
                {f: [c.copy() for c in video.review.candidates(f)] for f in video.review.frames}
                if video.review is not None
                else {}
            )
            result.append(
                VideoInput(
                    video.video_id,
                    video.path,
                    candidates,
                    copy.deepcopy(video.annotations),
                    video.fps,
                    video.num_frames,
                )
            )
        return result

    def start_project_batch(self, scope, options):
        from .sam_session import repo_root as _repo_root

        thread = getattr(self, "_batch_thread", None)
        if thread is not None and thread.is_alive():
            self._status("A project operation is already running.")
            return False
        project = self._sync_project()
        videos = self._batch_inputs(scope)
        if not videos:
            self._status("No videos in this scope. Select videos in Project Videos first.")
            return False
        if options.detect:
            capability = auto_label.check_backend(_repo_root(), options.weights)
            if not capability.ok:
                self._show_unavailable("Automatic detection", capability.message)
                return False
        self._batch_video_index = None
        cache_dir = project.directory or Path(tempfile.gettempdir()) / "YOAKE" / project.identity
        runner = BatchRunner(
            _repo_root(), cache_dir, encoder=self.embedding_provider() if options.sample else None
        )
        self._batch_runner = runner

        def emit(event):
            self._post(lambda e=event: self._batch_event(project, e, options))

        self._batch_thread = threading.Thread(
            target=runner.run, args=(videos, options, emit), daemon=True, name="yoake-project-batch"
        )
        self._batch_thread.start()
        self.note_activity(
            f"Started {'finding animals in' if options.detect else 'scanning'} "
            f"{len(videos)} video{'s' if len(videos) != 1 else ''}."
        )
        return True

    def stop_project_batch(self):
        runner = getattr(self, "_batch_runner", None)
        if runner is not None:
            runner.cancel.set()

    def _batch_event(self, project, event, options):
        if self.project is not project:
            return  # Late results cannot land in another project.
        if event["kind"] == "progress":
            self._report_batch_progress(event)
        if event["kind"] == "video":
            result = event["result"]
            self.smart_dialog.add_result(
                f"{result.video_id}: {result.status}"
                + (f" - {result.error}" if result.error else "")
            )
            video = project.videos[result.video_id]
            if result.status == "failed":
                video.ui["last_failure"] = result.error
                # One failed video in a run of twenty is the failure most
                # likely to go unnoticed: the dialog keeps going, the other
                # nineteen succeed, and the only trace is a field on a record
                # nobody opens. So it is also said out loud, once, where the
                # Activity window will still have it tomorrow.
                self.note_activity(
                    f"{result.video_id} failed: {result.error}", failed=True
                )
                return
            video.ui.pop("last_failure", None)
            video.fps, video.num_frames = result.fps, result.num_frames
            video.width, video.height = result.width, result.height
            # A sampling run's boxes arrive on the ``finished`` event instead,
            # because until every video has been scanned the sampler has not
            # chosen and :func:`batch.keep_chosen` has nothing to trim against.
            # Merging here would put the whole scan in the Review Inbox and
            # then leave it there: nothing removes a candidate once merged.
            if not options.sample:
                if options.detect:
                    self._merge_video_candidates(video, result.candidates)
                self._check_session(video)
            video.touch()
            self._dirty = True
            self._save_due = time.time() + 4
            self._refresh_workspace()
            if video is self._video_session:
                self.inbox.set_queue(video.queue, fps=video.fps)
                self._refresh_candidate_ui()
                self._render_canvas()
                self._render_timeline()
        elif event["kind"] == "finished":
            for result in event["results"]:
                if result.selection is not None:
                    video = project.videos[result.video_id]
                    video.selection = result.selection
                    # The trimmed boxes, and the checks run over the frames
                    # they are actually on. Nothing was merged for this video
                    # while the batch was running (see the ``video`` branch).
                    if options.sample and not event["cancelled"]:
                        chosen = sorted(result.selection.frame_set())
                        if options.detect:
                            self._merge_video_candidates(video, result.candidates)
                        self._check_session(
                            video, frames=sorted(set(chosen) | set(video.annotations))
                        )
                    video.touch()
                    project.sampling_log.append(
                        {
                            **options.request.to_metadata(),
                            "video_id": video.video_id,
                            "frames": list(result.selection.frames),
                            "clips": [(c.start, c.end) for c in result.selection.clips],
                            "balanced": options.balanced,
                            "time": time.time(),
                        }
                    )
            # A stopped sampling run chose nothing, so nothing was trimmed and
            # nothing was merged. Saying "results were kept" would be a lie
            # about the Review Inbox; what was actually kept is the cached
            # scan, which is the thing that makes the next attempt fast.
            kept = (
                "The frames it already looked at are kept, so starting again "
                "is quick."
                if options.sample
                else "Results from completed videos were kept."
            )
            if getattr(self, "_smart_running", False):
                self._smart_running = False
                if event["cancelled"]:
                    self.smart_dialog.stopped(f"Stopped. {kept}")
                else:
                    self._report_batch_finished(event)
            self._save_due = time.time() + 4
            self._render_timeline()
            self._refresh_workspace()
            self._status(
                f"Project operation cancelled. {kept}"
                if event["cancelled"]
                else "Project operation finished. Review the candidates, or open a chosen frame."
            )

    def _report_batch_progress(self, event):
        """One event, one dialog.

        The steps describe the video being worked on, so they start over when
        the run moves to the next one; the second bar and the line above it
        are what say which video that is.
        """
        from .smart_annotate import step_for_operation

        if event["index"] != getattr(self, "_batch_video_index", None):
            self._batch_video_index = event["index"]
            self.smart_dialog.reset_steps()
        self.smart_dialog.advance_to_step(step_for_operation(event["operation"]))
        self.smart_dialog.set_progress(event["fraction"], event["operation"])
        self.smart_dialog.set_overall(
            event["overall"],
            f"Video {event['index']} of {event['total']}: {event['video_id']}",
        )

    def _report_batch_finished(self, event):
        """What the run produced, and the button that follows from it.

        A run that only pre-labelled has no selection, and used to be reported
        as "no readable videos" for want of one. Failure is now decided by
        whether any video was read, not by whether sampling was asked for.
        """
        from .smart_annotate import STEPS

        results = event["results"]
        read = [r for r in results if r.status != "failed"]
        failed = [r for r in results if r.status == "failed"]
        selected = [(r.video_id, r.selection) for r in results if r.selection is not None]
        self._smart_selections = dict(selected)
        for index in range(len(STEPS)):
            self.smart_dialog.set_step(index, "done")
        self.smart_dialog.set_progress(1.0, "Finished.")
        self.smart_dialog.set_overall(1.0)
        if not read:
            self.smart_dialog.fail(
                "No readable videos in the scope. Check Project Videos for missing files."
            )
            return

        first = None
        total_samples = None
        notes = []
        if selected:
            first = self._smart_selections.get(self._video_session.video_id, selected[0][1])
            if len(selected) > 1:
                total_samples = sum(selection.total for _, selection in selected)
                notes.append(
                    f"Across {len(selected)} videos. Details below show the starting video."
                )
        if failed:
            notes.append(f"{len(failed)} of {len(results)} videos failed; the list above says why.")
        candidates, issues = self.project_review_counts()
        self.smart_dialog.finish(
            first,
            total_samples=total_samples,
            summary=" ".join(notes),
            predictions=candidates,
            issues=issues,
        )

    def _merge_video_candidates(self, video, candidates):
        if video.review is None:
            video.review = auto_label.ReviewSession()
        for frame, row in candidates.items():
            existing = video.review.candidates(frame)
            verified = video.annotations.get(frame, [])
            additions = []
            for candidate in row:
                # Preserve prior Accept/Reject decisions on cache hits and reruns.
                duplicate = any(
                    c.class_id == candidate.class_id
                    and auto_label.iou_any(c.bbox, candidate.bbox, c.obb, candidate.obb) >= 0.95
                    for c in existing + additions
                )
                overlaps_verified = any(
                    o.class_id == candidate.class_id
                    and auto_label.iou_any(o.bbox, candidate.bbox, o.obb, candidate.obb) >= 0.5
                    for o in verified
                )
                if not duplicate and not overlaps_verified:
                    additions.append(candidate)
            video.review.add_frame(frame, additions)

    def _check_session(self, video, *, frames=None):
        """Re-run the checks over this video's pending candidates.

        ``frames`` is the list the checks are allowed to have an opinion
        about, and leaving it out means *all of them*. That default is right
        for a dense pass and wrong for everything else: a check that reports
        "the detector found nothing here" over the frames no detector ever ran
        on is not a finding, it is the stride. A round now looks at one frame
        in sixty and keeps a hundred of them, so the caller passes the frames
        the boxes are on, plus the ones a human has already annotated.
        """
        candidates = video.review.pending_candidates() if video.review is not None else {}
        parts = tracklet.build_tracklets(candidates, max_gap=self.quality_config.fragment_gap)
        items = check_video(
            candidates, video.annotations, self.quality_config, video.num_frames, frames=frames
        )
        items = [replace(i, video_id=video.video_id) for i in items]
        carried = {
            (video.video_id, *key) if len(key) == 4 else key: value
            for key, value in video.queue.decisions().items()
        }
        video.queue = review_queue.ReviewQueue(items)
        video.queue.carry_over(carried)
        video.tracklets = tracklet.TrackletIndex(parts)
        human = [i for i in video.intervals if i.source == "human"]
        derived = intervals.merge_to_intervals(candidates)
        video.intervals = intervals.IntervalTrack(
            human
            + [
                i
                for i in derived
                if not any(h.track_id == i.track_id and h.overlaps(i) for h in human)
            ]
        )

    def run_project_quality(self, scope="Entire project"):
        self._ensure_project()
        # Quality-only batches do not require the source video to be present.
        inputs = self._batch_inputs(scope)
        project, config = self.project, self.quality_config
        revisions = {v.video_id: v.revision for v in project.videos.values()}
        if not inputs:
            self._status("Select videos in Project Videos before checking quality.")
            return

        def work():
            rows = []
            for video in inputs:
                try:
                    items = check_video(
                        video.candidates, video.annotations, config, video.num_frames
                    )
                    rows.append((video.video_id, items, ""))
                except Exception as exc:  # noqa: BLE001 - isolate checks per video.
                    rows.append((video.video_id, [], str(exc)))

            def finish():
                if self.project is not project:
                    return
                for vid, items, error in rows:
                    video = project.videos[vid]
                    if video.revision != revisions[vid]:
                        self._check_session(video)
                        continue
                    if error:
                        video.ui["last_failure"] = error
                        continue
                    carried = video.queue.decisions()
                    video.queue = review_queue.ReviewQueue(
                        [replace(i, video_id=vid) for i in items]
                    )
                    video.queue.carry_over(carried)
                    video.touch()
                self.inbox.set_queue(self._queue, fps=self.fps)
                self._save_due = time.time() + 4
                self._refresh_workspace()
                self.show_project_review()

            self._post(finish)

        threading.Thread(target=work, daemon=True, name="yoake-project-quality").start()

    def project_review_items(self):
        items = []
        for video in self._ensure_project().videos.values():
            for item in video.queue.items:
                if not video.queue.decision(item).done:
                    items.append(replace(item, video_id=video.video_id))
            if video.review is not None:
                for frame in video.review.pending_frames():
                    row = video.review.candidates(frame)
                    ids = tuple(
                        sorted(
                            {
                                c.track_id
                                for i, c in enumerate(row)
                                if c.track_id >= 0
                                and video.review.state(frame, i) == auto_label.PENDING
                            }
                        )
                    )
                    items.append(
                        quality.ReviewItem(
                            "suggestion",
                            frame,
                            frame,
                            ids,
                            detail="Accept or reject the candidates on this frame.",
                            video_id=video.video_id,
                        )
                    )
        return items

    def project_review_counts(self):
        """``(candidates, issues)`` waiting across the project.

        One definition for both finished states: Smart Annotate's result block
        and the project batch's review button quote the same two numbers, so
        they cannot drift into disagreeing about what a run produced.
        """
        project = getattr(self, "project", None)
        if project is None:
            return (0, 0)
        return (
            sum(video.pending for video in project.videos.values()),
            sum(len(video.queue) for video in project.videos.values()),
        )

    def review_next_project(self):
        items = self.project_review_items()
        if not items:
            self._status("No pending project review items.")
            return False
        last = getattr(self, "_project_review_key", None)
        start = next((i + 1 for i, item in enumerate(items) if item.key == last), 0)
        for item in items[start:] + items[:start]:
            if self.switch_video(item.video_id):
                self._project_review_key = item.key
                if item.kind == "suggestion":
                    self.goto_frame(item.start)
                    self._sel_issue = None
                    self._refresh_inspector()
                else:
                    self.jump_to_issue(item)
                self._status(f"Review: {item.video_id} / frame {item.start} — {item.label}")
                return True
        self._status("Review items remain in missing videos. Use Locate file in Project Videos.")
        return False

    def show_project_review(self):
        import dearpygui.dearpygui as dpg

        from .widgets import place_dialog

        if dpg.does_item_exist("project_review"):
            dpg.delete_item("project_review")
        items = self.project_review_items()
        with dpg.window(tag="project_review", label="Project Review Inbox", width=760, height=480):
            dpg.add_text(f"{len(items)} items across {len(self.project.videos)} videos")
            # Review Next and Refresh sit above the list, not after it: a
            # project with 300 review items used to put them 300 buttons down.
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Review Next", callback=self.review_next_project, width=160, height=30
                )
                dpg.add_button(
                    label="Refresh", callback=self.show_project_review, width=140, height=30
                )
            dpg.add_separator()
            with dpg.child_window(width=-1, height=-1, border=False):
                for item in items:
                    dpg.add_button(
                        label=f"{item.video_id}: {item.describe()}",
                        callback=lambda s, a, i: self._open_project_issue(i),
                        user_data=item,
                        width=-1,
                    )
        place_dialog("project_review", 760, 480)

    def _open_project_issue(self, item):
        if self.switch_video(item.video_id):
            if item.kind == "suggestion":
                self.goto_frame(item.start)
            else:
                self.jump_to_issue(item)

    def _setup_add_files(self):
        import tkinter as tk
        from tkinter import filedialog

        import dearpygui.dearpygui as dpg

        root = tk.Tk()
        root.withdraw()
        try:
            paths = filedialog.askopenfilenames(
                title="Project videos",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")],
            )
        finally:
            root.destroy()
        self._setup_videos = list(
            dict.fromkeys(getattr(self, "_setup_videos", []) + [Path(p) for p in paths])
        )
        dpg.set_value("setup_video_count", f"{len(self._setup_videos)} videos selected")

    def _setup_add_folder(self):
        import dearpygui.dearpygui as dpg

        from ..annotation.project import scan_folder
        from .file_dialogs import ask_dir as _tk_ask_dir

        folder = _tk_ask_dir("Project videos folder")
        if folder is not None:
            paths = scan_folder(folder, recursive=dpg.get_value("setup_recursive"))
            self._setup_videos = list(dict.fromkeys(getattr(self, "_setup_videos", []) + paths))
            dpg.set_value("setup_video_count", f"{len(self._setup_videos)} videos selected")

    def _shutdown_workspace(self):
        self.stop_project_batch()
        batch = getattr(self, "_batch_thread", None)
        if batch is not None:
            batch.join(timeout=5)
        if hasattr(self, "_fast_track_stop"):
            self._fast_track_stop.set()
        self._propagate_stop = True
        self._analysis_stop = True
        thread = getattr(self, "_workspace_save_thread", None)
        if thread is not None:
            thread.join()
        self._pump_ui()
        project = self._sync_project()
        if project.videos:
            directory = (
                project.directory
                or Path(tempfile.gettempdir()) / "YOAKE" / "recovery" / project.identity
            )
            try:
                from ..annotation.layout import PROJECT_FILE

                project.save(directory / PROJECT_FILE)
            except OSError as exc:
                print(f"Could not save workspace on exit: {exc}")
