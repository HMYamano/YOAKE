"""
plots.py — run 出力の可視化ユーティリティ

学習・評価後に自動保存される画像・JSON・CSV を生成する関数群。
matplotlib がインストールされていない場合は画像保存をスキップし、
JSON / CSV は必ず保存する。

呼び出し元: trainer.py の _val_epoch 後
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# matplotlib はオプション依存
try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 不要のバックエンド
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> bool:
    """混同行列を PNG として保存する。

    Args:
        cm: 混同行列 (num_classes x num_classes) — リスト or ndarray
        class_names: クラス名リスト
        output_path: 出力パス (.png)
        normalize: True のとき行方向で正規化 (recall 表示)

    Returns:
        True if saved, False if matplotlib unavailable.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 常に JSON も保存
    json_path = str(output_path).replace(".png", ".json")
    try:
        Path(json_path).write_text(
            json.dumps({"confusion_matrix": [list(row) for row in cm],
                        "class_names": class_names}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    if not _HAS_MPL:
        warnings.warn(
            "matplotlib が見つかりません。confusion matrix 画像の保存をスキップします。"
            " pip install matplotlib で有効化できます。",
            UserWarning, stacklevel=2,
        )
        return False

    arr = np.array(cm, dtype=float)
    if normalize:
        row_sum = arr.sum(axis=1, keepdims=True)
        arr = np.where(row_sum > 0, arr / row_sum, 0.0)

    n = len(class_names)
    figsize = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=12)

    thresh = 0.5 if normalize else arr.max() / 2
    for i in range(n):
        for j in range(n):
            val = arr[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=7)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
    except Exception as e:
        warnings.warn(f"confusion matrix 保存失敗: {e}", UserWarning, stacklevel=2)
        return False
    finally:
        plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Per-class precision / recall / F1
# ---------------------------------------------------------------------------

def save_per_class_metrics(
    eval_result: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
) -> None:
    """ActionEvaluator.compute() の結果から per-class メトリクスを JSON + CSV に保存する。

    Args:
        eval_result: ActionEvaluator.compute() の返り値
        output_dir: 保存ディレクトリ
        prefix: ファイル名の接頭辞 (例: "epoch_005_")
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_class_f1: Dict[str, float] = eval_result.get("per_class_f1", {})
    cm: List[List[int]] = eval_result.get("confusion_matrix", [])
    class_names = list(per_class_f1.keys())
    n = len(class_names)

    # --- per-class precision / recall / F1 from confusion matrix ---
    rows = []
    for i, cls in enumerate(class_names):
        if i < len(cm) and cm:
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n) if j != i)
            fn = sum(cm[i][j] for j in range(n) if j != i)
            support = sum(cm[i])
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = per_class_f1.get(cls, 0.0)
        else:
            precision = recall = support = 0
            f1 = per_class_f1.get(cls, 0.0)
        rows.append({
            "class": cls,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })

    # summary rows
    macro_f1 = eval_result.get("macro_f1", 0.0)
    accuracy = eval_result.get("accuracy", 0.0)
    total_support = sum(r["support"] for r in rows)
    weighted_f1 = (
        sum(r["f1"] * r["support"] for r in rows) / max(total_support, 1)
    )

    fname = f"{prefix}per_class_metrics"
    json_data = {
        "per_class": rows,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "total_support": total_support,
    }
    (out / f"{fname}.json").write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # CSV
    with open(out / f"{fname}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"class": "macro", "f1": round(macro_f1, 4)})
        writer.writerow({"class": "weighted", "f1": round(weighted_f1, 4)})


# ---------------------------------------------------------------------------
# Metrics history plots (from results.csv)
# ---------------------------------------------------------------------------

def _read_results_csv(csv_path: str) -> Dict[str, List]:
    """results.csv を読んでカラム → 値リストの dict を返す。"""
    data: Dict[str, List] = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    data.setdefault(k, [])
                    try:
                        data[k].append(float(v))
                    except (ValueError, TypeError):
                        data[k].append(v)
    except FileNotFoundError:
        pass
    return data


def save_metrics_history(
    results_csv: str,
    output_dir: str,
    stage: int = 0,
) -> None:
    """results.csv から指標推移プロットを作成し plots/ に保存する。

    matplotlib がなければスキップ。
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _read_results_csv(results_csv)
    if not data or "epoch" not in data:
        return

    epochs = data["epoch"]

    if not _HAS_MPL:
        warnings.warn(
            "matplotlib が見つかりません。history plot をスキップします。",
            UserWarning, stacklevel=2,
        )
        return

    def _plot(keys_labels, title, fname, ylabel="value"):
        valid_pairs = [(k, lb) for k, lb in keys_labels if k in data and any(
            isinstance(v, float) for v in data[k])]
        if not valid_pairs:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, lb in valid_pairs:
            vals = [v if isinstance(v, float) else float("nan") for v in data[k]]
            ax.plot(epochs, vals, label=lb, marker=".")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.savefig(str(plots_dir / fname), dpi=100, bbox_inches="tight")
        finally:
            plt.close(fig)

    # --- Loss plot (all stages) ---
    _plot(
        [("train_loss", "train"), ("val_loss", "val")],
        "Loss History", "loss_history.png", ylabel="Loss",
    )

    if stage == 1 or stage == 4:
        _plot(
            [("ap50", "AP50"), ("val_AP50", "val AP50")],
            "AP50 History (Stage 1)", "ap50_history.png", ylabel="AP50",
        )

    if stage == 2 or stage == 4:
        _plot(
            [("macro_f1", "macro F1"), ("weighted_f1", "weighted F1")],
            "F1 History (Stage 2)", "f1_history.png", ylabel="F1",
        )

    if stage == 3 or stage == 4:
        _plot(
            [("idf1", "IDF1"), ("idp", "IDP"), ("idr", "IDR")],
            "IDF1 History (Stage 3)", "idf1_history.png", ylabel="Score",
        )
        _plot(
            [("idsw", "IDSW")],
            "Identity Switches (Stage 3)", "idsw_history.png", ylabel="Count",
        )

    if stage == 4:
        _plot(
            [("composite_score", "composite"),
             ("ap50", "AP50"), ("macro_f1", "macro F1"), ("idf1", "IDF1")],
            "Composite Score (Stage 4)", "composite_history.png", ylabel="Score",
        )

    # LR plot
    _plot(
        [("lr", "learning rate")],
        "Learning Rate Schedule", "lr_schedule.png", ylabel="LR",
    )


# ---------------------------------------------------------------------------
# metrics_latest.json
# ---------------------------------------------------------------------------

def save_metrics_latest(
    val_metrics: Dict[str, Any],
    output_dir: str,
    epoch: int,
    primary_metric_name: str,
    primary_metric_value: float,
    best_so_far: float,
    higher_is_better: bool,
    class_counts: Optional[List[int]] = None,
    imbalance_strategy: Optional[str] = None,
) -> None:
    """最新 val 指標と best 判定基準を metrics_latest.json に保存する。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "epoch": epoch,
        "primary_metric": primary_metric_name,
        "primary_metric_value": round(float(primary_metric_value), 6),
        "best_so_far": round(float(best_so_far), 6),
        "higher_is_better": higher_is_better,
        "metrics": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                    for k, v in val_metrics.items()},
    }
    if class_counts is not None:
        data["class_counts"] = class_counts
    if imbalance_strategy is not None:
        data["imbalance_strategy"] = imbalance_strategy
    (out / "metrics_latest.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
