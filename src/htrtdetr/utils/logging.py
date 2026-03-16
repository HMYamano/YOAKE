"""
logging.py — 学習ログ管理

- Python 標準 logging への wrapper
- CSV / JSON 形式でのメトリクス保存
- wandb optional 対応
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_logger(
    name: str = "htrtdetr",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """フォーマット済み logger を返す"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 重複ハンドラを防ぐ
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricsLogger:
    """epoch / step ごとのメトリクスを CSV と JSON に保存するクラス"""

    def __init__(self, output_dir: str, prefix: str = "metrics") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{prefix}.csv"
        self.json_path = self.output_dir / f"{prefix}.json"
        self._records: List[Dict[str, Any]] = []
        self._csv_initialized = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """メトリクスを記録する"""
        record = dict(metrics)
        if step is not None:
            record["step"] = step
        self._records.append(record)
        self._write_csv(record)
        self._write_json()

    def _write_csv(self, record: Dict[str, Any]) -> None:
        fieldnames = list(record.keys())
        write_header = not self._csv_initialized
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(record)

    def _write_json(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, ensure_ascii=False)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._records)


class WandbLogger:
    """wandb を optional に使う wrapper。wandb がなければ no-op になる。"""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "ht-rtdetr",
        run_name: str = "run",
        config: Optional[Dict] = None,
    ) -> None:
        self.enabled = enabled
        self._run = None
        if enabled:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                )
            except ImportError:
                print("[WARN] wandb not installed. Logging disabled.")
                self.enabled = False
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
