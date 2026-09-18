"""
test_annotator_structure.py — annotator.py の分割が後戻りしないようにする

``annotator.py`` は 5,900 行あり、責務を少しずつ外へ出している途中にある。
この移動で壊しやすいものが 3 つあるので、それぞれを固定する。

1. **import できる条件**。``htrtdetr.annotator`` は dearpygui の無い環境でも
   import できなければならない (アノテーション I/O だけを使うテストと、
   torch-free / GUI-free なパイプラインがそれに依存している)。切り出した先で
   ``theme`` のような GUI 必須モジュールをトップレベル import すると、ここが
   静かに壊れる — 実際に一度壊した。
2. **依存の向き**。``gui/*`` が ``annotator`` を import し返すと循環になり、
   関数内の遅延 import で誤魔化すことになる。向きは常に annotator → gui。
3. **公開名**。移動しただけのつもりでも、旧名で import していたコードが
   壊れれば移動ではなく破壊になる。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
GUI_DIR = SRC_ROOT / "htrtdetr" / "gui"

#: annotator.py から切り出したモジュール。どれも GUI 無しで import できること。
EXTRACTED = [
    "htrtdetr.gui.canvas_view",
    "htrtdetr.gui.file_dialogs",
    "htrtdetr.gui.history",
    "htrtdetr.gui.review_mode",
    "htrtdetr.gui.sam_session",
    "htrtdetr.gui.video_controller",
    "htrtdetr.gui.video_source",
]

#: ``AnnotationApp`` が組み立てられていることの確認に使う mixin 名。
EXPECTED_MIXINS = [
    "CanvasViewMixin",
    "EditHistoryMixin",
    "ReviewModeMixin",
    "SamSessionMixin",
    "VideoControllerMixin",
    "WorkspaceMixin",
]


def _run_without_gui(code: str) -> subprocess.CompletedProcess:
    """dearpygui と cv2 を潰した子プロセスでコードを走らせる。"""
    preamble = (
        "import sys\n"
        "sys.modules['dearpygui'] = None\n"
        "sys.modules['cv2'] = None\n"
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(SRC_ROOT), "PYTHONIOENCODING": "utf-8"},
    )


# --------------------------------------------------------------------------- #
# 1. GUI 無しで import できること
# --------------------------------------------------------------------------- #

class TestImportableWithoutGui:
    @pytest.mark.parametrize("module", EXTRACTED)
    def test_extracted_module_imports_without_dearpygui(self, module):
        r = _run_without_gui(f"import {module}\nprint('ok')")
        assert r.returncode == 0, (
            f"{module} が dearpygui 無しで import できない:\n{r.stderr}"
        )

    def test_annotator_still_imports_without_dearpygui(self):
        r = _run_without_gui(
            "from htrtdetr.annotator import AnnotationApp, AnnObject\n"
            "assert AnnObject([0, 0, 10, 10], class_id=2).class_id == 2\n"
            "print('ok')"
        )
        assert r.returncode == 0, r.stderr

    def test_starting_the_app_without_dearpygui_says_why(self):
        r = _run_without_gui(
            "from htrtdetr.annotator import AnnotationApp\n"
            "try:\n"
            "    AnnotationApp()\n"
            "except RuntimeError as exc:\n"
            "    assert 'DearPyGui' in str(exc), str(exc)\n"
            "    print('ok')\n"
            "else:\n"
            "    raise AssertionError('missing dependency was not reported')\n"
        )
        assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# 2. 依存の向き
# --------------------------------------------------------------------------- #

def _module_level_imports(path: Path) -> set:
    """トップレベル (関数の外) で import しているモジュール名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:                      # body だけ = トップレベルのみ
        stack = [node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.Import):
                names.update(a.name for a in cur.names)
            elif isinstance(cur, ast.ImportFrom):
                if cur.module:
                    names.add(cur.module)
            # try/except や if の中のトップレベル import も拾う
            for field in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(getattr(cur, field, []) or [])
    return names


class TestDependencyDirection:
    def test_no_gui_module_imports_the_annotator(self):
        """
        ``gui/*`` → ``annotator`` は循環。向きは annotator → gui だけ。

        以前は ``gui/workspace.py`` と ``gui/batch_progress.py`` が
        ``from ..annotator import _tk_open_file`` していて、循環を避けるために
        関数の中で遅延 import する必要があった。共有ヘルパーを
        ``gui/file_dialogs.py`` と ``gui/sam_session.py`` に移して解消済み。
        """
        offenders = []
        for path in sorted(GUI_DIR.glob("*.py")):
            # 文字列検索ではなく AST で見る。docstring の中で「昔はこう書いて
            # いた」と説明している行を違反として数えないため。
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):          # 関数の中の遅延 import も含む
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    relative = node.level > 0 and module == "annotator"
                    absolute = module == "htrtdetr.annotator"
                    if relative or absolute:
                        offenders.append(f"{path.name}:{node.lineno} from {module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[:2] == ["htrtdetr", "annotator"]:
                            offenders.append(f"{path.name}:{node.lineno} {alias.name}")
        assert offenders == [], (
            "gui モジュールが annotator を import し返している:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize("module", EXTRACTED)
    def test_extracted_modules_do_not_import_torch(self, module):
        """GUI 側は torch-free。切り出し先でも守る。"""
        path = SRC_ROOT / Path(module.replace(".", "/") + ".py")
        assert "torch" not in {n.split(".")[0] for n in _module_level_imports(path)}


# --------------------------------------------------------------------------- #
# 3. 公開名 / 組み立て
# --------------------------------------------------------------------------- #

class TestAnnotationAppComposition:
    @pytest.fixture(scope="class")
    def annotator(self):
        pytest.importorskip("dearpygui.dearpygui")
        pytest.importorskip("cv2")
        from htrtdetr import annotator as A

        return A

    @pytest.mark.parametrize("mixin", EXPECTED_MIXINS)
    def test_app_is_built_from_the_mixins(self, annotator, mixin):
        names = [c.__name__ for c in annotator.AnnotationApp.__mro__]
        assert mixin in names, f"{mixin} が AnnotationApp に組み込まれていない"

    @pytest.mark.parametrize(
        "name",
        ["_tk_open_file", "_tk_save_file", "_tk_ask_dir", "_repo_root",
         "_corners_of", "sidecar_mod"],
    )
    def test_legacy_module_level_names_still_resolve(self, annotator, name):
        """移動しても旧名で届くこと (外部コードとテストが使っている)。"""
        assert getattr(annotator, name, None) is not None

    @pytest.mark.parametrize(
        "name,expected",
        [("_ZOOM_MIN", 0.2), ("_ZOOM_MAX", 20.0), ("_PAN_KEEP", 64),
         ("_UNDO_DEPTH", 50), ("_AUTOSAVE_DEBOUNCE", 4.0),
         ("_SEQ_SKIP_MAX", 12), ("_FRAME_TEX", "yoake_frame_tex")],
    )
    def test_legacy_constants_keep_their_values(self, annotator, name, expected):
        assert getattr(annotator, name) == expected

    def test_methods_moved_to_mixins_are_still_on_the_app(self, annotator):
        """呼び出し名は 1 つも変えていない (移動であって改名ではない)。"""
        moved = [
            # canvas_view
            "_canvas_origin", "_recompute_scale", "fit_view", "zoom_by",
            "_i2c", "_c2i",
            # history
            "_snapshot", "_push_undo", "undo", "redo", "_mark_dirty",
            "_tick_autosave",
            # sam_session
            "_sam_ready", "set_click_backend", "_start_sam", "_stop_sam",
            "_pump_sidecar",
            # review_mode
            "open_review_mode", "swap_tracks", "_render_review_canvas",
            # video_controller
            "_open_video_dialog", "_install_video_capture", "_read_frame_bgr",
            "goto_frame",
        ]
        missing = [m for m in moved if not hasattr(annotator.AnnotationApp, m)]
        assert missing == [], f"移動で消えたメソッド: {missing}"


# --------------------------------------------------------------------------- #
# 4. サイズのラチェット
# --------------------------------------------------------------------------- #

#: ``annotator.py`` の行数の上限。分割を進めたら下げる。上げてはいけない
#: (上げたくなったら、それは新しい責務を足しているということ)。
_ANNOTATOR_LINE_BUDGET = 5100


def test_annotator_stays_within_its_line_budget():
    lines = (SRC_ROOT / "htrtdetr" / "annotator.py").read_text(
        encoding="utf-8"
    ).count("\n")
    assert lines <= _ANNOTATOR_LINE_BUDGET, (
        f"annotator.py が {lines} 行 (上限 {_ANNOTATOR_LINE_BUDGET})。"
        "責務を gui/ 配下へ切り出してください。"
    )
