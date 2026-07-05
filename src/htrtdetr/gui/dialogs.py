"""
dialogs.py — ネイティブファイルダイアログ (tkinter)
==================================================

dearpygui にはネイティブなファイル選択が無いため、``tools/gui.py`` / ``annotator.py``
と同じく tkinter のダイアログを使う。tkinter は標準ライブラリで torch/dpg 非依存。
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def _dialog(kind: str, **kwargs) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "open":
            path = filedialog.askopenfilename(**kwargs)
        elif kind == "save":
            path = filedialog.asksaveasfilename(**kwargs)
        else:
            path = filedialog.askdirectory(**kwargs)
    finally:
        root.destroy()
    return path or ""


def pick_file(title: str = "ファイルを選択",
              filetypes: Optional[List[Tuple[str, str]]] = None) -> str:
    return _dialog("open", title=title, filetypes=filetypes or [("All", "*.*")])


def pick_dir(title: str = "フォルダを選択") -> str:
    return _dialog("dir", title=title)


def save_file(title: str = "保存先を選択", defaultextension: str = ".json",
              filetypes: Optional[List[Tuple[str, str]]] = None) -> str:
    return _dialog("save", title=title, defaultextension=defaultextension,
                   filetypes=filetypes or [("JSON", "*.json"), ("All", "*.*")])


VIDEO_TYPES = [("動画", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All", "*.*")]
JSON_TYPES = [("JSON", "*.json"), ("All", "*.*")]
CKPT_TYPES = [("PyTorch", "*.pt *.pth"), ("All", "*.*")]
