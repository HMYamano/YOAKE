"""
Helpers shared by the GUI for building yoake CLI commands.

Contract notes:
- `data.train_root` / `data.val_root` may be either a dataset root directory or
  a direct `annotations.json` path.
- `checkpoint` is always a checkpoint file path.
- `analyze` modes are limited to: timeline, distribution, id_switches, confidence.
"""

from __future__ import annotations

from typing import Any


def _yoake(*args: str) -> list[str]:
    return ["-m", "htrtdetr.cli", *args]


def build_yoake_command(command: str, *args: str, **kwargs: Any) -> list[str]:
    parts = [command, *args]
    for key, value in kwargs.items():
        if value in ("", None):
            continue
        parts.append(f"{key}={value}")
    return _yoake(*parts)


def build_train_command(
    stage: int,
    train_root: str,
    val_root: str,
    output_dir: str,
    **overrides: Any,
) -> list[str]:
    return build_yoake_command(
        "train",
        f"stage={stage}",
        **{
            "data.train_root": train_root,
            "data.val_root": val_root,
            "train.output_dir": output_dir,
            **overrides,
        },
    )


def build_val_command(
    stage: int,
    checkpoint: str,
    val_root: str,
    output_dir: str,
    **overrides: Any,
) -> list[str]:
    return build_yoake_command(
        "val",
        f"stage={stage}",
        **{
            "checkpoint": checkpoint,
            "data.val_root": val_root,
            "output_dir": output_dir,
            **overrides,
        },
    )


def build_predict_command(
    source: str,
    weights: str,
    output_dir: str,
    **overrides: Any,
) -> list[str]:
    return build_yoake_command(
        "predict",
        **{
            "source": source,
            "weights": weights,
            "output_dir": output_dir,
            **overrides,
        },
    )


def build_analyze_command(
    mode: str,
    output_dir: str,
    predictions: str = "",
    annotation: str = "",
    **overrides: Any,
) -> list[str]:
    return build_yoake_command(
        "analyze",
        **{
            "mode": mode,
            "predictions": predictions,
            "annotation": annotation,
            "output_dir": output_dir,
            **overrides,
        },
    )


__all__ = [
    "_yoake",
    "build_yoake_command",
    "build_train_command",
    "build_val_command",
    "build_predict_command",
    "build_analyze_command",
]
