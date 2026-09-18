"""
file_dialogs.py — OS ネイティブのファイル選択ダイアログ (tkinter)

dearpygui 自身のファイルダイアログは OS の「最近使った場所」やネットワークドライブ
を扱えないので、ファイルを選ぶところだけ tkinter に任せる。

=== なぜ独立したモジュールか ===

これらは ``annotator.py`` のモジュール関数だったため、``gui/workspace.py`` や
``gui/batch_progress.py`` が ``from ..annotator import _tk_open_file`` という
**逆向きの import** をしていた (部品がアプリ本体を読む)。循環を避けるために関数の
中で遅延 import する必要もあった。ここへ移すと依存の向きが揃い、遅延 import も
要らなくなる。

``htrtdetr.annotator`` 側には旧名 (``_tk_open_file`` 等) を別名として残してある。
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional, Tuple

__all__ = ["open_file", "save_file", "ask_dir"]

FileTypes = List[Tuple[str, str]]


def _hidden_root() -> tk.Tk:
    """ダイアログだけを出すための、見えないルートウィンドウ。

    ``-topmost`` を立てないと、dearpygui のビューポートの**裏**に出て
    「クリックしても何も起きない」ように見える。
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def open_file(title: str, filetypes: FileTypes) -> Optional[Path]:
    """ファイルを 1 つ選ぶ。キャンセルなら ``None``。"""
    root = _hidden_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return Path(path) if path else None


def save_file(
    title: str, filetypes: FileTypes, initial_file: str = "annotations.json"
) -> Optional[Path]:
    """保存先を選ぶ。キャンセルなら ``None``。"""
    root = _hidden_root()
    path = filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        initialfile=initial_file,
        defaultextension=".json",
    )
    root.destroy()
    return Path(path) if path else None


def ask_dir(title: str) -> Optional[Path]:
    """フォルダを 1 つ選ぶ。キャンセルなら ``None``。"""
    root = _hidden_root()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(path) if path else None
