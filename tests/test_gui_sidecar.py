"""
tests/test_gui_sidecar.py — the segmentation helper: protocol, spawn, client.

The interesting tests here spawn a **real child process** running
``tools/sam_sidecar.py --backend stub``. That is the point of the stub existing:
the pipe, the framing, the threading and the shutdown are exercised for real, on
a machine with no torch and no model, so these tests cannot decay into testing a
mock of themselves.

Design: ``docs/CLICK_SEGMENT.md`` §6.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from htrtdetr.gui import click_segment as cs
from htrtdetr.gui import sidecar as sc

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "src" / "htrtdetr" / "gui" / "sidecar.py"
CHILD_PATH = REPO_ROOT / "tools" / "sam_sidecar.py"


# --------------------------------------------------------------------------- #
# Module invariants
# --------------------------------------------------------------------------- #

def test_module_never_imports_dearpygui_or_torch():
    """This module spawns and threads; it must still never touch the GUI or torch.

    A worker thread that called into dearpygui would be the exact class of bug
    GUI_DESIGN P9 exists to prevent.
    """
    imported = set()
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "dearpygui" not in imported
    assert "torch" not in imported


def test_module_scope_imports_are_stdlib_only():
    """The child imports this module for the codec. It must cost nothing."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert "numpy" not in top and "cv2" not in top


# --------------------------------------------------------------------------- #
# Codec
# --------------------------------------------------------------------------- #

def test_encode_never_emits_a_newline():
    """One message is one line, or the framing is gone."""
    line = sc.encode({"op": "point", "detail": "two\nlines", "text": "日本語"})
    assert "\n" not in line
    assert sc.decode(line)["detail"] == "two\nlines"


def test_decode_ignores_noise_instead_of_raising():
    """A library printing to the child's stdout must not kill the session."""
    for junk in ("", "   ", "loading weights...", "[1/4] done", "null", "[1,2]", "{oops"):
        assert sc.decode(junk) is None


def test_request_message_rejects_an_unknown_op():
    with pytest.raises(ValueError):
        sc.request_message("segment_everything", 1)


def test_request_message_drops_none_fields():
    """`box=None` must not travel as a key the child then has to special-case."""
    msg = sc.request_message("point", 7, frame=3, box=None)
    assert msg == {"op": "point", "id": 7, "frame": 3}


# --------------------------------------------------------------------------- #
# The spec is data
# --------------------------------------------------------------------------- #

def test_argv_uses_the_given_interpreter_not_this_one():
    """The whole reason the helper exists: torch lives in a *different* Python."""
    spec = sc.SidecarSpec(python=r"C:\envs\yoru\python.exe", repo_root=str(REPO_ROOT))
    argv = spec.argv()
    assert argv[0] == r"C:\envs\yoru\python.exe"
    assert argv[0] != sys.executable or sys.executable == r"C:\envs\yoru\python.exe"
    assert "-u" in argv                      # or replies sit in a block buffer
    assert argv[2].endswith("sam_sidecar.py")


def test_argv_omits_empty_options():
    spec = sc.SidecarSpec(python="python", repo_root=str(REPO_ROOT), backend="stub")
    assert "--model" not in spec.argv()
    assert "--checkpoint" not in spec.argv()


def test_env_puts_src_on_pythonpath():
    """Without it the child dies with ModuleNotFoundError on a src-layout checkout."""
    env = sc.SidecarSpec(python="python", repo_root=str(REPO_ROOT)).env({})
    assert str(REPO_ROOT / "src") in env["PYTHONPATH"]
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_probe_argv_only_uses_find_spec():
    """Probing must not import torch — that is seconds and a pile of GPU DLLs."""
    source = sc.probe_argv("python")[2]
    assert "find_spec" in source
    assert "import torch" not in source


def test_probe_reports_this_interpreter_honestly():
    """A real probe of the running interpreter, which has no torch here."""
    info = sc._run_probe(sys.executable, timeout=30)
    assert info.ok
    assert info.version.startswith(str(sys.version_info[0]))
    assert info.torch is ("torch" in info.modules)


def test_probe_survives_a_missing_interpreter():
    info = sc._run_probe(str(REPO_ROOT / "no" / "such" / "python.exe"), timeout=5)
    assert not info.ok and info.detail


def test_candidate_order_puts_the_env_var_first(tmp_path):
    """What the user said beats what we guessed."""
    order = sc.candidate_interpreters({"YOAKE_SAM_PYTHON": "X:/chosen/python.exe"},
                                      home=tmp_path)
    assert order[0] == "X:/chosen/python.exe"


def test_candidate_finds_conda_envs(tmp_path):
    env = tmp_path / "miniconda3" / "envs" / "yoru"
    env.mkdir(parents=True)
    (env / "python.exe").write_text("")
    assert str(env / "python.exe") in sc.candidate_interpreters({}, home=tmp_path)


# --------------------------------------------------------------------------- #
# The locked reason (GUI_DESIGN §3.2)
# --------------------------------------------------------------------------- #

def test_locked_reason_has_cause_command_and_escape_hatch():
    reason = sc.locked_reason(())
    assert "PyTorch" in reason                        # the cause
    assert "YOAKE_SAM_PYTHON" in reason               # a literal, copyable fix
    assert "Local (OpenCV)" in reason                 # what works right now
    assert len(reason.splitlines()) == 3


def test_locked_reason_changes_when_torch_is_there_but_no_model_package():
    """A reason that names the wrong cause sends the user down the wrong path."""
    info = sc.InterpreterInfo(python="p", ok=True, torch=True, modules=("torch",))
    reason = sc.locked_reason([info])
    assert "SAM package" in reason
    assert "pip install" in reason


# --------------------------------------------------------------------------- #
# Frame hand-off
# --------------------------------------------------------------------------- #

def test_frame_store_round_trips_and_evicts(tmp_path):
    np = pytest.importorskip("numpy")
    store = sc.FrameStore(root=tmp_path / "frames", keep=2)
    paths = [store.put(i, np.full((4, 5, 3), i, np.uint8)) for i in range(3)]
    assert not paths[0].exists()                      # evicted, keep=2
    assert paths[2].exists()
    assert np.load(paths[2]).shape == (4, 5, 3)
    assert store.has(2) and not store.has(0)
    store.clear()
    assert not paths[2].exists()


def test_waiting_for_one_reply_keeps_the_others():
    """A measured bug: `wait_for` returned from the middle of the drained batch
    and dropped everything behind the match, so the next wait blocked for its
    whole timeout with no error to show for it."""
    client = sc.SidecarClient(sc.SidecarSpec(python="python", repo_root="."))
    for i in (1, 2, 3):
        client._queue.put({"id": i, "ok": True, "n": i})

    assert client.wait_for(1, timeout=0.5)["n"] == 1
    assert client.wait_for(3, timeout=0.5)["n"] == 3
    assert client.wait_for(2, timeout=0.5)["n"] == 2
    assert client.wait_for(9, timeout=0.1) is None


def test_poll_returns_replies_oldest_first():
    client = sc.SidecarClient(sc.SidecarSpec(python="python", repo_root="."))
    for i in (1, 2, 3):
        client._queue.put({"id": i})
    assert [m["id"] for m in client.poll()] == [1, 2, 3]
    assert client.poll() == []


# --------------------------------------------------------------------------- #
# A real child process
# --------------------------------------------------------------------------- #

@pytest.fixture
def helper():
    """A live ``--backend stub`` child, stopped no matter how the test ends."""
    spec = sc.SidecarSpec(python=sys.executable, repo_root=str(REPO_ROOT),
                          backend="stub")
    client = sc.SidecarClient(spec)
    client.start(timeout=60)
    try:
        yield client
    finally:
        client.stop()


def test_helper_starts_and_announces_what_it_can_do(helper):
    assert helper.is_running()
    assert helper.backend_name == "stub"
    assert "point" in helper.capabilities


def test_a_click_comes_back_as_a_click_result(helper):
    np = pytest.importorskip("numpy")
    frame = np.zeros((120, 200, 3), np.uint8)
    helper.wait_for(helper.embed(0, frame), timeout=30)

    reply = helper.wait_for(helper.point(0, [[100, 60, 1]]), timeout=30)
    assert reply["ok"] and reply["frame"] == 0

    res = sc.result_from_reply(reply, backend="sam:stub")
    assert isinstance(res, cs.ClickResult) and res.ok
    assert res.obb[0] == 100.0 and res.obb[2] == 40.0
    assert res.backend == "sam:stub"
    x1, y1, x2, y2 = res.bbox
    assert x1 < 100 < x2 and y1 < 60 < y2


def test_replies_carry_the_id_that_asked(helper):
    """Out-of-order replies must reach the right waiter, or a click lands on
    another click's box."""
    np = pytest.importorskip("numpy")
    helper.wait_for(helper.embed(0, np.zeros((120, 200, 3), np.uint8)), timeout=30)

    ids = [helper.point(0, [[x, 60, 1]]) for x in (20, 60, 100)]
    seen = {}
    for want in reversed(ids):                # ask for the last one first
        reply = helper.wait_for(want, timeout=30)
        assert reply is not None and reply["id"] == want
        seen[want] = reply["obb"][0]
    assert list(seen[i] for i in ids) == [20.0, 60.0, 100.0]


def test_an_unknown_op_is_an_error_not_a_dead_helper(helper):
    """The child must survive a message it does not understand."""
    helper._next_id += 1
    bad = helper._next_id
    with helper._write_lock:
        helper._proc.stdin.write(sc.encode({"op": "wat", "id": bad}) + "\n")
        helper._proc.stdin.flush()
    reply = helper.wait_for(bad, timeout=30)
    assert reply is not None and not reply["ok"] and reply["error"] == "unknown_op"
    assert helper.is_running()

    ok = helper.wait_for(helper.send("hello", protocol=sc.PROTOCOL_VERSION), timeout=30)
    assert ok["ok"]


def test_a_failing_request_is_a_reply_not_a_crash(helper):
    """A model that cannot segment one point must not cost the whole helper."""
    reply = helper.wait_for(helper.point(0, []), timeout=30)
    assert reply is not None and not reply["ok"] and reply["detail"]
    assert helper.is_running()


def test_point_on_an_unsent_frame_says_so(helper):
    reply = helper.wait_for(helper.point(99, [[10, 10, 1]]), timeout=30)
    assert not reply["ok"] and reply["error"] == "no_frame"


def test_concept_returns_several_candidates(helper):
    np = pytest.importorskip("numpy")
    helper.wait_for(helper.embed(0, np.zeros((120, 200, 3), np.uint8)), timeout=30)
    reply = helper.wait_for(helper.concept(0, "fly", 3), timeout=30)
    assert reply["ok"] and len(reply["items"]) == 3
    assert all(len(item["obb"]) == 5 for item in reply["items"])


def test_pending_tracks_what_the_user_is_waiting_for(helper):
    np = pytest.importorskip("numpy")
    req = helper.embed(0, np.zeros((60, 60, 3), np.uint8))
    assert helper.pending_ops().get(req) == "embed"
    assert helper.oldest_pending_age() is not None
    helper.wait_for(req, timeout=30)
    assert req not in helper.pending_ops()


def test_stop_is_idempotent_and_leaves_no_frames(helper):
    np = pytest.importorskip("numpy")
    path = helper.frames.put(0, np.zeros((8, 8, 3), np.uint8))
    assert path.exists()
    helper.stop()
    helper.stop()
    assert not helper.is_running()
    assert not path.exists()


def test_sending_to_a_stopped_helper_raises_sidecar_error(helper):
    helper.stop()
    with pytest.raises(sc.SidecarError):
        helper.point(0, [[1, 1, 1]])


def test_a_protocol_mismatch_is_refused_loudly():
    """Two checkouts that disagree must say so rather than misread fields."""
    spec = sc.SidecarSpec(python=sys.executable, repo_root=str(REPO_ROOT),
                          backend="stub")
    client = sc.SidecarClient(spec)
    client.start(timeout=60)
    try:
        req = client.send("hello", protocol=sc.PROTOCOL_VERSION + 99)
        reply = client.wait_for(req, timeout=30)
        assert not reply["ok"] and reply["error"] == "protocol_mismatch"
    finally:
        client.stop()


def test_starting_a_helper_that_cannot_run_raises():
    spec = sc.SidecarSpec(python=str(REPO_ROOT / "nope" / "python.exe"),
                          repo_root=str(REPO_ROOT))
    with pytest.raises(sc.SidecarError):
        sc.SidecarClient(spec).start(timeout=10)


def test_the_child_self_test_passes():
    """``--selftest`` is what a user runs when a click is not coming back."""
    import subprocess
    proc = subprocess.run([sys.executable, str(CHILD_PATH), "--selftest"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all checks passed" in proc.stdout


# --------------------------------------------------------------------------- #
# Reply -> result
# --------------------------------------------------------------------------- #

def test_a_failed_reply_still_carries_a_sentence():
    res = sc.result_from_reply({"ok": False, "error": cs.TWO_ANIMALS}, "sam:x")
    assert not res.ok
    assert res.detail == cs.DIAGNOSES[cs.TWO_ANIMALS]
    assert res.diagnosis == cs.TWO_ANIMALS


def test_a_reply_without_a_known_error_still_says_something():
    res = sc.result_from_reply({"ok": False, "error": "RuntimeError",
                                "detail": "CUDA out of memory"}, "sam:x")
    assert res.detail == "CUDA out of memory"


def test_a_reply_without_a_bbox_gets_one_derived():
    res = sc.result_from_reply({"ok": True, "obb": [10, 20, 8, 4, 0.0]}, "sam:x")
    assert res.bbox == cs.obb_to_aabb((10, 20, 8, 4, 0.0))
