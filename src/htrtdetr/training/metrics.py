"""
Metric selection helpers for best-model tracking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def _finite_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def compute_composite_score(
    ap50: float,
    macro_f1: float,
    idf1: float,
    w_ap50: float = 0.4,
    w_macro_f1: float = 0.3,
    w_idf1: float = 0.3,
) -> float:
    """Compute the stage-4 composite score with non-finite inputs sanitized."""
    ap50_v = _finite_or_default(ap50, 0.0)
    macro_f1_v = _finite_or_default(macro_f1, 0.0)
    idf1_v = _finite_or_default(idf1, 0.0)
    return w_ap50 * ap50_v + w_macro_f1 * macro_f1_v + w_idf1 * idf1_v


def select_best_metric(
    stage: int,
    val_metrics: Dict[str, Any],
    metrics_cfg=None,
) -> Tuple[float, str, bool]:
    """Return `(metric_value, metric_name, higher_is_better)` for a stage."""
    stage2_primary = "macro_f1"
    stage3_primary = "idf1"
    stage4_primary = "composite"
    composite_ap50 = 0.4
    composite_macro_f1 = 0.3
    composite_idf1 = 0.3

    if metrics_cfg is not None:
        stage2_primary = getattr(metrics_cfg, "stage2_primary", stage2_primary)
        stage3_primary = getattr(metrics_cfg, "stage3_primary", stage3_primary)
        stage4_primary = getattr(metrics_cfg, "stage4_primary", stage4_primary)
        composite_ap50 = getattr(metrics_cfg, "composite_ap50", composite_ap50)
        composite_macro_f1 = getattr(metrics_cfg, "composite_macro_f1", composite_macro_f1)
        composite_idf1 = getattr(metrics_cfg, "composite_idf1", composite_idf1)

    if stage == 1:
        val = _finite_or_default(
            val_metrics.get("val_AP50", val_metrics.get("AP50", val_metrics.get("ap50", 0.0))),
            0.0,
        )
        return val, "AP50", True

    if stage == 2:
        if stage2_primary == "macro_f1":
            val = _finite_or_default(val_metrics.get("macro_f1", 0.0), 0.0)
            return val, "macro_f1", True
        val = _finite_or_default(val_metrics.get("val_loss", float("inf")), float("inf"))
        return val, "val_loss", False

    if stage == 3:
        if stage3_primary == "idf1":
            val = _finite_or_default(val_metrics.get("idf1", val_metrics.get("IDF1", 0.0)), 0.0)
            return val, "idf1", True
        val = _finite_or_default(val_metrics.get("val_loss", float("inf")), float("inf"))
        return val, "val_loss", False

    if stage == 4:
        if stage4_primary == "composite":
            ap50 = _finite_or_default(
                val_metrics.get("val_AP50", val_metrics.get("AP50", val_metrics.get("ap50", 0.0))),
                0.0,
            )
            mf1 = _finite_or_default(val_metrics.get("macro_f1", 0.0), 0.0)
            idf1 = _finite_or_default(val_metrics.get("idf1", val_metrics.get("IDF1", 0.0)), 0.0)
            val = compute_composite_score(
                ap50,
                mf1,
                idf1,
                composite_ap50,
                composite_macro_f1,
                composite_idf1,
            )
            return val, "composite", True
        if stage4_primary == "ap50":
            val = _finite_or_default(
                val_metrics.get("val_AP50", val_metrics.get("AP50", val_metrics.get("ap50", 0.0))),
                0.0,
            )
            return val, "AP50", True
        val = _finite_or_default(val_metrics.get("val_loss", float("inf")), float("inf"))
        return val, "val_loss", False

    val = _finite_or_default(val_metrics.get("val_loss", float("inf")), float("inf"))
    return val, "val_loss", False


def is_better(new_val: float, best_val: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return new_val > best_val
    return new_val < best_val


def init_best_val(higher_is_better: bool) -> float:
    return float("-inf") if higher_is_better else float("inf")
