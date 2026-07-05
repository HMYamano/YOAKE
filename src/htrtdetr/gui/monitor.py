"""
monitor.py — 学習ラン監視 (results.csv / metrics_latest.json)
=============================================================

``runs/train/stage{N}/results.csv`` と ``metrics_latest.json`` を読み取り、GUI の
ライブ学習モニタへ供給する。

パーサ (``read_results_csv`` / ``series`` / ``read_metrics_latest``) は dearpygui
非依存で単体テスト可能。``TrainingMonitor`` は daemon スレッドで定期ポーリングし、
コールバックに ``(rows, latest)`` を渡す (GUI 側で dpg プロットに反映する)。
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_results_csv(path: str) -> List[Dict[str, str]]:
    """results.csv を行 dict のリストとして読む。存在しなければ空リスト。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except OSError:
        return []


def series(rows: List[Dict[str, str]], x: str, y: str) -> Tuple[List[float], List[float]]:
    """列 ``x`` / ``y`` を数値ペア ``(xs, ys)`` にする。数値化できない行はスキップ。"""
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        xv = _to_float(row.get(x))
        yv = _to_float(row.get(y))
        if xv is not None and yv is not None:
            xs.append(xv)
            ys.append(yv)
    return xs, ys


def read_metrics_latest(path: str) -> Optional[dict]:
    """metrics_latest.json を読む。無い/壊れている場合は None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# 学習ステージ別のプロット対象 (y 列名, 表示ラベル)。列が無ければ自動で無視される。
STAGE_METRIC_COLUMNS: Dict[int, List[Tuple[str, str]]] = {
    1: [("ap50", "AP50")],
    2: [("macro_f1", "macro F1"), ("weighted_f1", "weighted F1")],
    3: [("idf1", "IDF1"), ("idsw", "ID switches")],
    4: [("composite_score", "composite"), ("ap50", "AP50"), ("macro_f1", "macro F1"), ("idf1", "IDF1")],
}


def metric_columns_for_stage(stage: int) -> List[Tuple[str, str]]:
    return STAGE_METRIC_COLUMNS.get(stage, STAGE_METRIC_COLUMNS[4])


class TrainingMonitor:
    """results.csv / metrics_latest.json を定期ポーリングするバックグラウンド監視。"""

    def __init__(
        self,
        run_dir: str,
        on_update: Callable[[List[Dict[str, str]], Optional[dict]], None],
        interval: float = 2.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.on_update = on_update
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> Tuple[List[Dict[str, str]], Optional[dict]]:
        rows = read_results_csv(str(self.run_dir / "results.csv"))
        latest = read_metrics_latest(str(self.run_dir / "metrics_latest.json"))
        return rows, latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            rows, latest = self.poll_once()
            try:
                self.on_update(rows, latest)
            except Exception:  # noqa: BLE001 — GUI コールバックの例外で監視を止めない
                pass
            self._stop.wait(self.interval)
