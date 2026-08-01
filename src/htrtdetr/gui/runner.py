"""
runner.py — subprocess 実行 + ログ配信 (dearpygui)
==================================================

``yoake`` CLI コマンドをバックグラウンドスレッドで実行し、stdout を dpg の
ログパネルへ逐次流し込む。``tools/gui.py`` の ``run_command`` と同じ責務を、
ハブの複数パネルで共有できる形にしたもの。
"""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import dearpygui.dearpygui as dpg


class CommandRunner:
    """1 本の subprocess を管理し、ログを ``log_tag`` の text へ追記する。"""

    MAX_LINES = 800

    def __init__(self, log_tag: str, status_tag: str = "", cwd: Optional[Path] = None) -> None:
        self.log_tag = log_tag
        self.status_tag = status_tag
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self._lines: List[str] = []
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    # ── ログ ─────────────────────────────────────────────────────────
    def log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._lines.append(f"[{ts}] {text}")
            if len(self._lines) > self.MAX_LINES:
                self._lines = self._lines[-self.MAX_LINES:]
            joined = "\n".join(self._lines)
        if dpg.does_item_exist(self.log_tag):
            dpg.set_value(self.log_tag, joined)

    def _status(self, text: str) -> None:
        if self.status_tag and dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, text)

    # ── 実行 ─────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def run(self, cmd: List[str], on_done: Optional[Callable[[int], None]] = None) -> None:
        """``cmd`` (例: ['-m','htrtdetr.cli','train',...]) を実行する。"""
        if self.is_running():
            self.log("[WARN] Another command is already running.")
            return

        def _worker() -> None:
            full = [sys.executable] + [str(c) for c in cmd]
            self.log("$ " + " ".join(full))
            self._status("Running...")
            try:
                proc = subprocess.Popen(
                    full,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self.cwd),
                    encoding="utf-8",
                    errors="replace",
                )
                self._proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log(line.rstrip())
                proc.wait()
                code = proc.returncode
                self.log(f"--- {'Done' if code == 0 else f'Error (code={code})'} ---")
                self._status("Done" if code == 0 else f"Error (code={code})")
                if on_done is not None:
                    try:
                        on_done(code)
                    except Exception:  # noqa: BLE001
                        pass
            except FileNotFoundError as exc:
                self.log(f"[ERROR] Command not found: {exc}")
                self.log("  Hint: install yoake with 'pip install -e .'")
                self._status("Error: command not found")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[ERROR] {exc}")
                self._status("Error")
            finally:
                self._proc = None

        threading.Thread(target=_worker, daemon=True).start()

    def stop(self) -> None:
        if self.is_running() and self._proc is not None:
            self._proc.terminate()
            self.log("--- Interrupted ---")
