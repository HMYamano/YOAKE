"""共通フィクスチャ"""
import sys
from pathlib import Path

import pytest

# リポジトリルートを sys.path に追加
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def _no_real_file_dialogs(monkeypatch):
    """
    テスト中に OS のファイルダイアログを開かせない。

    ``gui/file_dialogs`` は tkinter の modal ダイアログを出す。テストがどれか 1 つ
    でも stub し忘れると、pytest は**無言で永久にブロックする** — 失敗すらせず、
    どのテストで止まったかも分からない (実際にスイート全体が 82% で止まった)。

    ここで実装を「呼ばれたら落ちる」関数へ差し替えておけば、stub 忘れは
    ハングではなく明確な失敗として出る。ダイアログが要るテストは、この同じ
    属性を自分で monkeypatch すれば良い (後勝ちになる)。
    """
    try:
        from htrtdetr.gui import file_dialogs
    except Exception:  # pragma: no cover - GUI 依存が無い環境
        return

    def _blocked(name):
        def _raise(*args, **kwargs):
            raise AssertionError(
                f"file_dialogs.{name}() がテスト中に呼ばれました。"
                "OS のダイアログはテストを無言でハングさせます。"
                f"monkeypatch.setattr(file_dialogs, '{name}', ...) で差し替えてください。"
            )
        return _raise

    for name in ("open_file", "save_file", "ask_dir"):
        monkeypatch.setattr(file_dialogs, name, _blocked(name), raising=False)
