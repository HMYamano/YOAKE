"""
commands.py — yoake CLI コマンド生成 (dearpygui 非依存)
======================================================

GUI パネルとテストの両方で使う、config キーへのマッピングを 1 箇所に集約する。
``htrtdetr.gui_cli`` の低レベルビルダを包み、GUI 上のフィールド値を正しい
config オーバーライド (例: 「可視化 OFF」→ ``inference.show_*=false``) に変換する。

戻り値は ``["-m", "htrtdetr.cli", ...]`` 形式のリスト (先頭に ``sys.executable`` を
付けて subprocess 実行する)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..gui_cli import (
    build_analyze_command,
    build_predict_command,
    build_train_command,
    build_val_command,
)


def train_command(
    stage: int,
    train_root: str,
    val_root: str,
    output_dir: str,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    window_size: Optional[int] = None,
    imbalance_strategy: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """``yoake train stage=N`` コマンドを生成する。

    ``window_size`` は Stage 2–4 の sliding-window 学習に必須 (既定 16)。
    """
    overrides: Dict[str, Any] = {}
    if epochs is not None:
        overrides["train.max_epochs"] = epochs
    if batch_size is not None:
        overrides["data.batch_size"] = batch_size
    if lr is not None:
        overrides["optimizer.lr"] = lr
    if window_size is not None:
        overrides["data.window_size"] = window_size
    if imbalance_strategy:
        overrides["loss.imbalance_strategy"] = imbalance_strategy
    if extra:
        overrides.update(extra)
    return build_train_command(stage, train_root, val_root, output_dir, **overrides)


def val_command(
    stage: int,
    checkpoint: str,
    val_root: str,
    output_dir: str,
    **overrides: Any,
) -> List[str]:
    """``yoake val stage=N`` コマンドを生成する。"""
    return build_val_command(stage, checkpoint, val_root, output_dir, **overrides)


def predict_command(
    source: str,
    weights: str,
    output_dir: str,
    *,
    score_threshold: Optional[float] = None,
    window_size: Optional[int] = None,
    max_frames: Optional[int] = None,
    visualize: bool = True,
) -> List[str]:
    """``yoake predict`` コマンドを生成する。

    ``visualize=False`` のとき、存在しない ``visualize`` フィールドではなく実在する
    ``inference.show_bbox/show_id/show_action/show_confidence`` を false にする
    (旧 GUI の ``visualize=`` バグを修正)。
    """
    overrides: Dict[str, Any] = {}
    if score_threshold is not None:
        overrides["score_threshold"] = score_threshold
    if window_size is not None:
        overrides["window_size"] = window_size
    if max_frames is not None:
        overrides["max_frames"] = max_frames
    if not visualize:
        overrides["inference.show_bbox"] = False
        overrides["inference.show_id"] = False
        overrides["inference.show_action"] = False
        overrides["inference.show_confidence"] = False
    return build_predict_command(source, weights, output_dir, **overrides)


# analyze モード別の必須入力フィールド。
ANALYZE_REQUIRED: Dict[str, str] = {
    "timeline": "predictions",
    "id_switches": "predictions",
    "confidence": "predictions",
    "distribution": "annotation",
}


def analyze_missing_field(mode: str, predictions: str, annotation: str) -> Optional[str]:
    """analyze 実行前バリデーション。不足している必須フィールド名を返す (無ければ None)。"""
    req = ANALYZE_REQUIRED.get(mode)
    if req == "predictions" and not predictions:
        return "predictions"
    if req == "annotation" and not annotation:
        return "annotation"
    return None


def analyze_command(
    mode: str,
    output_dir: str,
    predictions: str = "",
    annotation: str = "",
) -> List[str]:
    """``yoake analyze mode=...`` コマンドを生成する。"""
    return build_analyze_command(
        mode, output_dir, predictions=predictions, annotation=annotation
    )
