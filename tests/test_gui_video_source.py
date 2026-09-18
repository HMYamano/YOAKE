"""
test_gui_video_source.py — annotator から切り出した動画読み出し層のテスト

``gui/video_source`` は dearpygui にも torch にも依存しないので、ウィンドウを
開かずに (``-m 'not gui'`` のまま) デコード位置の扱いを直接検証できる。
これは annotator.py と analysis_panel.py が共有する唯一の実装なので、
ここが正しければ両方の「次のフレーム」が正しい。
"""

from __future__ import annotations

import pytest

from htrtdetr.gui import video_source


class _FakeCapture:
    """seek / grab / read の回数を数える最小の capture。"""

    CAP_PROP_POS_FRAMES = 1

    def __init__(self, num_frames: int = 50, fail_at: int | None = None):
        self.num_frames = num_frames
        self.fail_at = fail_at
        self.pos = 0
        self.seeks = 0
        self.grabs = 0
        self.reads = 0

    def set(self, prop, value):
        self.seeks += 1
        self.pos = int(value)
        return True

    def grab(self):
        self.grabs += 1
        self.pos += 1
        return self.pos <= self.num_frames

    def read(self):
        self.reads += 1
        frame_index = self.pos
        self.pos += 1
        if frame_index >= self.num_frames or frame_index == self.fail_at:
            return False, None
        return True, f"frame-{frame_index}"


class TestReadFrameSequential:
    def test_none_capture_is_safe(self):
        assert video_source.read_frame_sequential(None, 0, None) == (None, None)

    def test_first_read_seeks(self):
        cap = _FakeCapture()
        frame, decoded = video_source.read_frame_sequential(cap, 0, None)
        assert frame == "frame-0" and decoded == 0
        assert cap.seeks == 1

    def test_next_frame_does_not_seek_or_grab(self):
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 0, None)
        seeks_after_first = cap.seeks

        for idx in range(1, 6):
            frame, decoded = video_source.read_frame_sequential(cap, idx, decoded)
            assert frame == f"frame-{idx}"
        assert cap.seeks == seeks_after_first, "1 フレーム前進で seek している"
        assert cap.grabs == 0, "1 フレーム前進で grab している"

    def test_short_forward_hop_grabs_instead_of_seeking(self):
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 0, None)
        seeks = cap.seeks

        frame, decoded = video_source.read_frame_sequential(cap, 4, decoded)
        assert frame == "frame-4" and decoded == 4
        assert cap.seeks == seeks, "短い前方ジャンプで seek した"
        assert cap.grabs == 3, "捨てるフレームぶんだけ grab するはず"

    def test_long_forward_jump_seeks(self):
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 0, None)
        seeks = cap.seeks

        frame, decoded = video_source.read_frame_sequential(
            cap, video_source.SEQ_SKIP_MAX + 5, decoded
        )
        assert frame is not None
        assert cap.seeks == seeks + 1
        assert cap.grabs == 0

    def test_backwards_seeks(self):
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 5, None)
        seeks = cap.seeks
        frame, decoded = video_source.read_frame_sequential(cap, 2, decoded)
        assert frame == "frame-2" and decoded == 2
        assert cap.seeks == seeks + 1

    def test_same_frame_again_seeks(self):
        """``idx == decoded_idx`` は前進ではないので seek し直す。"""
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 3, None)
        seeks = cap.seeks
        frame, decoded = video_source.read_frame_sequential(cap, 3, decoded)
        assert frame == "frame-3"
        assert cap.seeks == seeks + 1

    def test_failed_read_forgets_the_position(self):
        """読めなかったら位置は当てにならない → 次は必ず seek させる。"""
        cap = _FakeCapture(fail_at=3)
        _, decoded = video_source.read_frame_sequential(cap, 2, None)
        frame, decoded = video_source.read_frame_sequential(cap, 3, decoded)
        assert frame is None
        assert decoded is None

        seeks = cap.seeks
        video_source.read_frame_sequential(cap, 4, decoded)
        assert cap.seeks == seeks + 1

    def test_failed_grab_falls_back_to_seek(self):
        cap = _FakeCapture(num_frames=4)
        _, decoded = video_source.read_frame_sequential(cap, 0, None)
        seeks = cap.seeks
        video_source.read_frame_sequential(cap, 6, decoded)
        assert cap.seeks >= seeks, "grab が尽きたのに seek しなかった"

    def test_seq_skip_max_is_configurable(self):
        cap = _FakeCapture()
        _, decoded = video_source.read_frame_sequential(cap, 0, None)
        seeks = cap.seeks
        video_source.read_frame_sequential(cap, 3, decoded, seq_skip_max=1)
        assert cap.seeks == seeks + 1


class TestProbeCapture:
    class _Probe:
        def __init__(self, values):
            self.values = values

        def get(self, prop):
            return self.values.get(prop, 0)

    def test_probe_reads_basic_properties(self):
        import cv2

        cap = self._Probe({
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FRAME_COUNT: 300,
        })
        probe = video_source.probe_capture(cap)
        assert probe.fps == pytest.approx(30.0)
        assert (probe.width, probe.height) == (1920, 1080)
        assert probe.num_frames == 300
        assert probe.last_index == 299

    def test_zero_fps_falls_back_to_default(self):
        import cv2

        cap = self._Probe({cv2.CAP_PROP_FPS: 0.0})
        assert video_source.probe_capture(cap, default_fps=25.0).fps == pytest.approx(25.0)

    def test_empty_video_last_index_is_zero(self):
        import cv2

        cap = self._Probe({cv2.CAP_PROP_FRAME_COUNT: 0})
        assert video_source.probe_capture(cap).last_index == 0


class TestOpenCapture:
    def test_missing_file_returns_none(self, tmp_path):
        assert video_source.open_capture(tmp_path / "nope.mp4") is None


def test_module_does_not_import_dearpygui_or_torch():
    """torch なし GUI / ヘッドレステストのために、この層は依存を持たない。"""
    import ast
    import pathlib

    src = pathlib.Path(video_source.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "dearpygui" not in imported
    assert "torch" not in imported
