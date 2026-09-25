"""
sidecar.py — the segmentation helper process: protocol, spawn, client
=====================================================================

``click_segment.py`` may never spawn anything and may never import torch. A SAM
model needs both. This module is where that line is crossed, on purpose and in
one place: it owns the wire protocol, the child-process spec, and the client
that talks to a running helper. The child itself is ``tools/sam_sidecar.py``.

Design and rationale: ``docs/CLICK_SEGMENT.md`` §6.

Why a separate process at all, when the usual answer is "just import it":

* The annotator runs on a Python that **cannot have torch** (3.14 here). The
  helper therefore runs under a *different interpreter*, which is the one thing
  an in-process design can never do. This is also why the spec carries an
  explicit ``python`` path instead of using ``sys.executable`` the way
  ``auto_label.predict_argv`` does.
* A model that dies takes the helper with it, not the annotator and not the
  user's unsaved work.

Invariants (asserted by ``tests/test_gui_sidecar.py``):

* No ``dearpygui`` import. A worker thread never touches the GUI; it fills a
  queue that the main thread drains (GUI_DESIGN P9).
* No ``torch`` import. Capability detection runs in a throwaway child process.
* Standard library only at module scope, so the *child* can import this module
  for the codec without paying for numpy or OpenCV first.
* Every reply carries the id and the frame of the request that caused it, so a
  result for a frame the user has left can be dropped instead of applied.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PROTOCOL_VERSION", "OPS", "CAPABILITIES",
    "encode", "decode", "request_message",
    "SidecarSpec", "sidecar_script", "probe_argv", "probe_interpreters",
    "InterpreterInfo", "SidecarCapability", "locked_reason",
    "FrameStore", "SidecarClient", "SidecarError",
    "result_from_reply",
]

#: Bumped only when an existing message shape changes meaning. ``hello``
#: exchanges it so an old helper and a new GUI refuse each other loudly instead
#: of misreading fields.
PROTOCOL_VERSION = 1

#: Every operation the GUI may send.
#:
#: ``propagate`` and ``embed_frames`` are additive: a helper from an older
#: checkout answers ``unknown_op``, the GUI reports that the backend cannot do
#: video or embedding, and everything else keeps working. That is why adding
#: them does **not** bump :data:`PROTOCOL_VERSION` - no existing message shape
#: changed meaning.
OPS = ("hello", "embed", "point", "concept", "propagate", "embed_frames",
       "drop", "shutdown")

#: Every capability a helper may advertise. The GUI gates features on what came
#: back from ``hello`` — never on the backend's name, and never on a guess.
#:
#: ``video``      the helper can carry one prompt across many frames.
#: ``embedding``  the helper can turn frames into feature vectors.
CAPABILITIES = ("point", "box", "concept", "video", "embedding")


class SidecarError(RuntimeError):
    """The helper could not be started, or died while we were talking to it."""


# --------------------------------------------------------------------------- #
# 1. Codec — newline-delimited JSON
# --------------------------------------------------------------------------- #

def encode(message: Dict[str, Any]) -> str:
    """One message as one line. No newlines inside, ever."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def decode(line: str) -> Optional[Dict[str, Any]]:
    """Parse one line, or ``None`` if it is not a message.

    Returning ``None`` rather than raising is deliberate: a child's library can
    print a warning to stdout, and one stray line must not kill the session.
    """
    line = (line or "").strip()
    if not line or line[0] != "{":
        return None
    try:
        out = json.loads(line)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def request_message(op: str, req_id: int, **fields: Any) -> Dict[str, Any]:
    """Build a request. ``op`` is validated here so a typo fails at the source."""
    if op not in OPS:
        raise ValueError(f"unknown op: {op!r}")
    msg: Dict[str, Any] = {"op": op, "id": int(req_id)}
    msg.update({k: v for k, v in fields.items() if v is not None})
    return msg


# --------------------------------------------------------------------------- #
# 2. The child-process spec — data, not a spawn
# --------------------------------------------------------------------------- #

def sidecar_script(repo_root: Path) -> Path:
    return Path(repo_root) / "tools" / "sam_sidecar.py"


@dataclass(frozen=True)
class SidecarSpec:
    """Everything needed to start one helper.

    ``python`` is the interpreter to run it with, which is normally *not*
    ``sys.executable``: the GUI's Python has no torch, and the one that does is
    a different installation entirely.
    """

    python: str
    repo_root: str
    backend: str = "stub"
    model: str = ""
    checkpoint: str = ""
    #: ``auto`` and not ``cpu``: only the helper has torch, so only the helper
    #: can see a GPU. It resolves this and reports the answer through ``hello``.
    device: str = "auto"

    def argv(self) -> List[str]:
        """The full command line, including the interpreter.

        ``-u`` matters: without it the child's replies sit in a block buffer and
        every click looks like a hang.
        """
        argv = [
            self.python, "-u", str(sidecar_script(Path(self.repo_root))),
            "--backend", self.backend,
        ]
        if self.model:
            argv += ["--model", self.model]
        if self.checkpoint:
            argv += ["--checkpoint", self.checkpoint]
        if self.device:
            argv += ["--device", self.device]
        return argv

    def env(self, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Environment for the child: the repo on PYTHONPATH, utf-8, unbuffered.

        Delegates to ``auto_label.child_env`` so the src-layout rule has exactly
        one implementation.
        """
        from .auto_label import child_env
        return child_env(Path(self.repo_root), base)


# --------------------------------------------------------------------------- #
# 3. Finding an interpreter that can run a model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InterpreterInfo:
    """What one candidate interpreter can do."""

    python: str
    ok: bool
    torch: bool = False
    modules: Tuple[str, ...] = ()
    version: str = ""
    detail: str = ""

    @property
    def backends(self) -> Tuple[str, ...]:
        """Model backends this interpreter could run, best first.

        ``sam2`` needs ``huggingface_hub`` beside it and does not depend on it:
        ``from_pretrained`` is the only way the helper loads a SAM 2 checkpoint
        and it imports ``huggingface_hub`` internally. Listing ``sam2`` without
        it would start a helper that fails on the first click.
        """
        out: List[str] = []
        if "sam2" in self.modules and "huggingface_hub" in self.modules:
            out.append("sam2")
        if "segment_anything" in self.modules:
            out.append("segment_anything")
        if "transformers" in self.modules:
            out.append("transformers")
        return tuple(out)

    @property
    def sam2_needs_hub(self) -> bool:
        """``sam2`` is installed but cannot reach any weights. Says which."""
        return "sam2" in self.modules and "huggingface_hub" not in self.modules


#: Modules the probe reports on. ``torch`` gates everything; the rest name which
#: adapter can run.
PROBE_MODULES = ("torch", "sam2", "huggingface_hub", "segment_anything",
                 "transformers", "numpy")

_PROBE_SOURCE = (
    "import importlib.util as u, json, sys;"
    "print(json.dumps({'version': sys.version.split()[0], 'modules':"
    " [m for m in %r if u.find_spec(m) is not None]}))"
)


def probe_argv(python: str) -> List[str]:
    """Command that reports what a candidate interpreter has, as data.

    ``find_spec`` only: importing torch costs seconds and pulls GPU DLLs, and
    the answer we need is "is it there", not "does it work". The GUI's own
    process never imports any of this, which is the whole point.
    """
    return [python, "-c", _PROBE_SOURCE % (list(PROBE_MODULES),)]


def _run_probe(python: str, timeout: float) -> InterpreterInfo:
    try:
        proc = subprocess.run(
            probe_argv(python), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=_CREATION_FLAGS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return InterpreterInfo(python=python, ok=False, detail=str(exc))
    if proc.returncode != 0:
        return InterpreterInfo(python=python, ok=False,
                               detail=(proc.stderr or "").strip()[-200:])
    data = None
    for line in (proc.stdout or "").splitlines():
        data = decode(line) or data
    if not data:
        return InterpreterInfo(python=python, ok=False, detail="no probe output")
    modules = tuple(str(m) for m in data.get("modules", []))
    return InterpreterInfo(
        python=python, ok=True, torch="torch" in modules,
        modules=modules, version=str(data.get("version", "")),
    )


def candidate_interpreters(env: Optional[Dict[str, str]] = None,
                           home: Optional[Path] = None) -> List[str]:
    """Interpreters to probe, in priority order, without probing them.

    Order is "what the user said" before "what we guessed", and the guesses are
    only the two conda layouts that actually exist on this machine.
    """
    environ = os.environ if env is None else env
    out: List[str] = []

    def add(path: Optional[str]) -> None:
        if path and path not in out:
            out.append(path)

    add(environ.get("YOAKE_SAM_PYTHON"))
    add(sys.executable)
    root = Path(home) if home is not None else Path.home()
    for base in ("miniconda3", "anaconda3", "miniforge3"):
        envs = root / base / "envs"
        try:
            children = sorted(envs.iterdir())
        except OSError:
            continue
        for child in children:
            for name in ("python.exe", "bin/python", "python"):
                exe = child / name
                if exe.exists():
                    add(str(exe))
                    break
    return out


@dataclass(frozen=True)
class SidecarCapability:
    """Whether a helper can be started at all, and what to tell the user if not."""

    usable: bool
    interpreter: Optional[InterpreterInfo] = None
    checked: Tuple[InterpreterInfo, ...] = ()
    reason: str = ""

    @property
    def backends(self) -> Tuple[str, ...]:
        return self.interpreter.backends if self.interpreter else ()


def locked_reason(checked: Sequence[InterpreterInfo] = ()) -> str:
    """The three-part reason for a gated control (GUI_DESIGN §3.2).

    Cause, then a literal command, then the escape hatch — which here is the
    strongest part of the message, because the feature the user actually wanted
    is already working without any of this.
    """
    with_torch = [i for i in checked if i.ok and i.torch]
    half_installed = [i for i in with_torch if i.sam2_needs_hub]
    if half_installed:
        # The most confusing state there is: the package is there, so the user
        # believes the install worked, and every load fails inside sam2.
        cause = ("sam2 is installed next to PyTorch but cannot reach any weights: "
                 "huggingface_hub is missing, and sam2 does not depend on it.")
        fix = f"Finish the install with:  \"{half_installed[0].python}\" -m pip install huggingface_hub"
    elif with_torch:
        names = ", ".join(sorted({",".join(i.backends) or "none" for i in with_torch}))
        cause = ("The segmentation helper needs a SAM package next to PyTorch, and none "
                 f"was found (present: {names}).")
        fix = "Install one with:  pip install sam2 huggingface_hub"
    else:
        cause = ("The segmentation helper needs PyTorch, and no Python this GUI can "
                 "reach has it.")
        fix = ("Point YOAKE at one with:  set YOAKE_SAM_PYTHON="
               r"C:\path\to\env\python.exe")
    escape = ("Until then Click animal keeps using the Local (OpenCV) fit, which needs "
              "nothing installed and is already the default - press C and click an animal.")
    return f"{cause}\n{fix}\n{escape}"


def probe_interpreters(*, env: Optional[Dict[str, str]] = None,
                       home: Optional[Path] = None,
                       timeout: float = 5.0,
                       limit: int = 8) -> SidecarCapability:
    """Find an interpreter that could run a model backend.

    Stops at the first usable one. Every probe is a child process with a
    timeout, so a broken conda env costs seconds, not the session.
    """
    checked: List[InterpreterInfo] = []
    for python in candidate_interpreters(env, home)[:limit]:
        info = _run_probe(python, timeout)
        checked.append(info)
        if info.ok and info.torch and info.backends:
            return SidecarCapability(usable=True, interpreter=info,
                                     checked=tuple(checked))
    return SidecarCapability(usable=False, checked=tuple(checked),
                             reason=locked_reason(checked))


# --------------------------------------------------------------------------- #
# 4. Frame hand-off
# --------------------------------------------------------------------------- #

class FrameStore:
    """Decoded frames on disk for the helper to pick up.

    The GUI already holds the decoded frame, so it writes the array itself and
    sends a path. Two things this avoids, both deliberate:

    * base64 or JPEG in the pipe — 3 MB of text per frame, and a re-encode.
    * **the helper opening the video.** ``cap.set(POS_FRAMES, n)`` is off by one
      at non-integer fps and agrees at frame 0, so a helper that seeks would
      segment the neighbouring frame and every box would be one frame out, with
      nothing on screen to show it (GUI_DESIGN P15).
    """

    def __init__(self, root: Optional[Path] = None, keep: int = 8) -> None:
        import tempfile
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / f"yoake-sam-{os.getpid()}"
        self.keep = int(keep)
        self._order: deque = deque()
        self._paths: Dict[int, Path] = {}

    def put(self, frame_index: int, frame_bgr: Any) -> Path:
        import numpy as np
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"f{int(frame_index):08d}.npy"
        if frame_index not in self._paths:
            np.save(path, frame_bgr, allow_pickle=False)
            self._paths[frame_index] = path
            self._order.append(frame_index)
            while len(self._order) > self.keep:
                self.discard(self._order.popleft())
        return path

    def has(self, frame_index: int) -> bool:
        return frame_index in self._paths

    def discard(self, frame_index: int) -> None:
        path = self._paths.pop(frame_index, None)
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass

    def clear(self) -> None:
        for frame_index in list(self._paths):
            self.discard(frame_index)
        self._order.clear()
        try:
            self.root.rmdir()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 5. The client
# --------------------------------------------------------------------------- #

_CREATION_FLAGS = 0
if sys.platform == "win32":                      # pragma: no cover - platform
    _CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class _Pending:
    op: str
    frame: Optional[int]
    sent_at: float = field(default_factory=time.monotonic)


class SidecarClient:
    """Owns one helper process and the conversation with it.

    Threading contract (GUI_DESIGN P9): the reader thread only ever puts into a
    queue; :meth:`poll` drains it from whatever thread the GUI runs on. Nothing
    here touches dearpygui, so this class is testable headlessly and the GUI
    cannot be mutated from a worker.
    """

    #: A helper that dies is restarted this many times before the UI gives up
    #: and falls back. Once: a crash is worth retrying, a crash loop is not.
    MAX_RESTARTS = 1

    def __init__(self, spec: SidecarSpec,
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        self.spec = spec
        self.on_log = on_log
        self.capabilities: Tuple[str, ...] = ()
        self.backend_name: str = ""
        #: The whole ``hello`` reply, kept so consumers can gate on what the
        #: helper *said* rather than on its backend name. ``video_segment`` and
        #: ``embedding`` both read it; guessing from ``backend_name`` would
        #: claim SAM 2 can propagate on an install where it cannot.
        self.hello: Dict[str, Any] = {}
        self.restarts = 0
        self.frames = FrameStore()
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "Queue[Dict[str, Any]]" = Queue()
        self._inbox: List[Dict[str, Any]] = []
        self._pending: Dict[int, _Pending] = {}
        self._stderr: deque = deque(maxlen=40)
        self._next_id = 0
        self._write_lock = threading.Lock()
        self._exited = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Spawn the helper and complete the ``hello`` handshake.

        The timeout is generous because importing torch and loading weights is
        genuinely slow; the caller shows elapsed seconds while it waits, since
        there is no progress to show (GUI_DESIGN P20).
        """
        if self.is_running():
            return {"ok": True, "already": True}
        try:
            self._proc = subprocess.Popen(
                self.spec.argv(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1, cwd=self.spec.repo_root,
                env=self.spec.env(), creationflags=_CREATION_FLAGS,
            )
        except OSError as exc:
            raise SidecarError(f"could not start the helper: {exc}") from exc

        self._exited = False
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        req_id = self.send("hello", protocol=PROTOCOL_VERSION)
        reply = self.wait_for(req_id, timeout=timeout)
        if reply is None or not reply.get("ok"):
            detail = (reply or {}).get("detail") or self.stderr_tail()
            self.stop()
            raise SidecarError(detail or "the helper did not answer hello")
        self.hello = dict(reply)
        self.capabilities = tuple(str(c) for c in reply.get("capabilities", ()))
        self.backend_name = str(reply.get("backend", self.spec.backend))
        return reply

    def can(self, capability: str) -> bool:
        """Does the running helper advertise this capability?

        The single place a feature is gated. ``"video" in client.capabilities``
        would work too, but a method makes the not-running case one answer
        instead of two.
        """
        return self.is_running() and capability in self.capabilities

    def stop(self, timeout: float = 3.0) -> None:
        """Ask, then insist. Closing stdin is what a well-behaved child exits on."""
        proc, self._proc = self._proc, None
        self.frames.clear()
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    with self._write_lock:
                        if proc.stdin and not proc.stdin.closed:
                            proc.stdin.write(encode({"op": "shutdown"}) + "\n")
                            proc.stdin.flush()
                            proc.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:      # pragma: no cover
                        proc.kill()
        finally:
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:                            # pragma: no cover
                    pass

    # ── sending ──────────────────────────────────────────────────────────
    def send(self, op: str, *, frame: Optional[int] = None, **fields: Any) -> int:
        """Queue one request and return its id. Never blocks on a reply."""
        if not self.is_running():
            raise SidecarError("the helper is not running")
        self._next_id += 1
        req_id = self._next_id
        msg = request_message(op, req_id, frame=frame, **fields)
        try:
            with self._write_lock:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(encode(msg) + "\n")
                self._proc.stdin.flush()
        except (OSError, ValueError, AssertionError) as exc:
            raise SidecarError(f"the helper stopped listening: {exc}") from exc
        self._pending[req_id] = _Pending(op=op, frame=frame)
        return req_id

    def embed(self, frame_index: int, frame_bgr: Any) -> int:
        """Ask for a frame to be encoded now, so a later click is instant."""
        path = self.frames.put(frame_index, frame_bgr)
        return self.send("embed", frame=frame_index, path=str(path),
                         shape=list(getattr(frame_bgr, "shape", ())))

    def point(self, frame_index: int, points: Sequence[Sequence[float]],
              box: Optional[Sequence[float]] = None) -> int:
        return self.send("point", frame=frame_index,
                         points=[list(p) for p in points],
                         box=list(box) if box else None)

    def concept(self, frame_index: int, text: str, max_results: int = 32) -> int:
        return self.send("concept", frame=frame_index, text=text,
                         max=int(max_results))

    def propagate(self, *, frames_dir: str, first_frame: int, prompt_frame: int,
                  points: Optional[Sequence[Sequence[float]]] = None,
                  box: Optional[Sequence[float]] = None,
                  obj_id: int = 1, backward: bool = False,
                  max_frames: int = 0) -> int:
        """Ask the helper to carry one prompt across a window of frames.

        ``frames_dir`` holds the window already written to disk as
        ``00000.jpg`` upward, and ``first_frame`` is the real video index of
        ``00000.jpg``. Frames go through the filesystem rather than the pipe,
        for the same reason single frames already do: a 200-frame window is
        hundreds of megabytes and newline-delimited JSON is the wrong transport
        for it.

        The reply carries one **OBB per frame**, not one mask per frame. The
        helper reduces each mask with the GUI's own ``mask_to_obb``, so the fit
        is identical to Click to Box's and the pipe carries five floats instead
        of a megabyte.
        """
        return self.send("propagate", frame=prompt_frame,
                         frames_dir=str(frames_dir),
                         first_frame=int(first_frame),
                         points=[list(p) for p in points] if points else None,
                         box=list(box) if box else None,
                         obj_id=int(obj_id),
                         backward=bool(backward),
                         max_frames=int(max_frames))

    def embed_frames(self, *, frames_dir: str, first_frame: int,
                     batch_size: int = 16) -> int:
        """Ask the helper for one feature vector per frame in ``frames_dir``."""
        return self.send("embed_frames", frames_dir=str(frames_dir),
                         first_frame=int(first_frame),
                         batch_size=int(batch_size))

    # ── receiving ────────────────────────────────────────────────────────
    def poll(self) -> List[Dict[str, Any]]:
        """Every reply that has arrived, oldest first. Call from the GUI thread."""
        out, self._inbox = self._inbox, []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        for msg in out:
            self._pending.pop(int(msg.get("id", -1)), None)
        return out

    def wait_for(self, req_id: int, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Block until one specific reply arrives. Startup and tests only.

        Replies for other requests are kept, not dropped — otherwise waiting for
        ``hello`` would silently eat an ``embed`` that finished first.
        """
        deadline = time.monotonic() + timeout
        # Drain what is already in hand *completely* before deciding. Returning
        # from the middle of the drained batch would silently discard the
        # messages behind the match, and the next wait for one of those would
        # then block for its whole timeout — a 30 s stall with no error.
        found: Optional[Dict[str, Any]] = None
        for msg in self.poll():
            if found is None and int(msg.get("id", -1)) == req_id:
                found = msg
            else:
                self._inbox.append(msg)
        if found is not None:
            return found
        while time.monotonic() < deadline:
            try:
                msg = self._queue.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            except Empty:
                if self._exited and self._queue.empty():
                    return None
                continue
            if int(msg.get("id", -1)) == req_id:
                self._pending.pop(req_id, None)
                return msg
            self._inbox.append(msg)
        return None

    def pending_ops(self) -> Dict[int, str]:
        return {rid: p.op for rid, p in self._pending.items()}

    def oldest_pending_age(self) -> Optional[float]:
        """Seconds the user has been waiting, or ``None`` if nothing is in flight."""
        if not self._pending:
            return None
        return time.monotonic() - min(p.sent_at for p in self._pending.values())

    def stderr_tail(self, lines: int = 3) -> str:
        return "\n".join(list(self._stderr)[-lines:])

    # ── threads ──────────────────────────────────────────────────────────
    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                msg = decode(line)
                if msg is None:
                    self._log(f"[helper] {line.rstrip()}")
                elif msg.get("event") == "log":
                    self._log(f"[helper] {msg.get('text', '')}")
                else:
                    self._queue.put(msg)
        except (OSError, ValueError):                       # pragma: no cover
            pass
        finally:
            self._exited = True

    def _read_stderr(self) -> None:
        """Drain stderr forever. A full pipe would block the child mid-reply."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                text = line.rstrip()
                if text:
                    self._stderr.append(text)
                    self._log(f"[helper] {text}")
        except (OSError, ValueError):                       # pragma: no cover
            pass

    def _log(self, text: str) -> None:
        if self.on_log is not None:
            try:
                self.on_log(text)
            except Exception:                               # pragma: no cover
                pass


# --------------------------------------------------------------------------- #
# 6. Replies -> the engine's result type
# --------------------------------------------------------------------------- #

def result_from_reply(reply: Dict[str, Any], backend: str):
    """Turn one ``point`` reply into a :class:`click_segment.ClickResult`.

    Keeping the conversion here means the annotator handles one result type no
    matter which backend answered.
    """
    from . import click_segment as cs

    if not reply.get("ok"):
        diagnosis = str(reply.get("error") or cs.NOTHING_FOUND)
        detail = str(reply.get("detail") or cs.DIAGNOSES.get(diagnosis, "")) \
            or cs.DIAGNOSES[cs.NOTHING_FOUND]
        return cs.ClickResult(obb=None, bbox=None, score=None, backend=backend,
                              detail=detail, diagnosis=diagnosis)
    obb = tuple(float(v) for v in reply["obb"])
    bbox = reply.get("bbox")
    bbox = tuple(int(v) for v in bbox) if bbox else cs.obb_to_aabb(obb)
    score = reply.get("score")
    return cs.ClickResult(
        obb=obb, bbox=bbox,
        score=None if score is None else float(score),
        backend=backend,
        detail=str(reply.get("detail")
                   or f"Fitted a {obb[2]:.0f} x {obb[3]:.0f} px box from your click."),
    )
