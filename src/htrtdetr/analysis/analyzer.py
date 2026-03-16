"""
analyzer.py — 解析・可視化ツール

機能:
1. bbox + ID + action の動画可視化 (inferencer に委譲)
2. per-track の action timeline 可視化
3. confusion matrix 可視化
4. class distribution 集計
5. ID switch 解析
6. action confidence ヒストグラム
7. branch ablation 比較結果の保存

依存: matplotlib (optional), PIL

matplotlib がない場合は JSON での集計保存のみ行う。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# matplotlib helper
# ---------------------------------------------------------------------------

def _get_plt():
    """matplotlib を import して返す。なければ None"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Action Timeline Visualizer
# ---------------------------------------------------------------------------

class ActionTimelineVisualizer:
    """
    各 track の action timeline を横棒グラフで可視化する。

    Example:
        ID=1: [idle][idle][walk][walk][groom][groom][idle]
        ID=2: [walk][walk][walk][interact][interact][idle]
    """

    def __init__(
        self,
        action_names: List[str],
        fps: float = 30.0,
        cmap_name: str = "tab10",
    ):
        self.action_names = action_names
        self.fps = fps
        self.cmap_name = cmap_name

        # action_id → color
        plt = _get_plt()
        if plt is not None:
            cmap = plt.get_cmap(cmap_name)
            self.colors = [cmap(i) for i in np.linspace(0, 1, len(action_names))]
        else:
            self.colors = [(0.5, 0.5, 0.5, 1.0)] * len(action_names)

    def plot(
        self,
        track_timelines: Dict[int, List[int]],  # {track_id: [action_id_t0, ...]}
        save_path: str,
        title: str = "Action Timeline",
    ) -> None:
        """
        track_timelines: {track_id: List[action_id per frame]}
        """
        plt = _get_plt()
        if plt is None:
            # JSON で保存して終わり
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            json_path = Path(save_path).with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({str(k): v for k, v in track_timelines.items()}, f)
            return

        n_tracks = len(track_timelines)
        if n_tracks == 0:
            return

        fig, ax = plt.subplots(figsize=(max(12, len(list(track_timelines.values())[0]) / 5), n_tracks * 0.8 + 2))

        for row, (track_id, timeline) in enumerate(sorted(track_timelines.items())):
            T = len(timeline)
            for t, action_id in enumerate(timeline):
                if 0 <= action_id < len(self.action_names):
                    color = self.colors[action_id]
                    ax.barh(row, 1.0, left=t / self.fps, height=0.7, color=color, align="center")

        # 凡例
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color=self.colors[i], label=name)
            for i, name in enumerate(self.action_names)
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

        ax.set_yticks(range(n_tracks))
        ax.set_yticklabels([f"ID {tid}" for tid in sorted(track_timelines.keys())])
        ax.set_xlabel("Time (s)")
        ax.set_title(title)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion Matrix Visualizer
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    cm: (C, C) confusion matrix (row: GT, col: pred)
    """
    plt = _get_plt()

    # JSON 保存 (常に)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(save_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }, f)

    if plt is None:
        return

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(6, len(class_names))))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    ax.set_xticklabels(class_names, rotation=45, ha="right")

    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            text = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Class Distribution
# ---------------------------------------------------------------------------

def compute_class_distribution(
    annotation_json_path: str,
) -> Dict[str, Any]:
    """
    annotation JSON からクラス / action の分布を集計する。
    """
    with open(annotation_json_path, "r") as f:
        data = json.load(f)

    class_names = data.get("class_names", [])
    action_names = data.get("action_names", [])

    class_counts: Dict[str, int] = defaultdict(int)
    action_counts: Dict[str, int] = defaultdict(int)
    total_objects = 0
    total_frames = 0

    for video in data.get("videos", []):
        for frame in video.get("frames", []):
            total_frames += 1
            for obj in frame.get("objects", []):
                total_objects += 1
                cls_id = obj.get("class_id", 0)
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                class_counts[cls_name] += 1

                act_id = obj.get("action_id", -1)
                if act_id >= 0:
                    act_name = action_names[act_id] if act_id < len(action_names) else str(act_id)
                    action_counts[act_name] += 1

    return {
        "total_videos": len(data.get("videos", [])),
        "total_frames": total_frames,
        "total_objects": total_objects,
        "class_distribution": dict(class_counts),
        "action_distribution": dict(action_counts),
    }


def plot_distribution(
    distribution: Dict[str, int],
    save_path: str,
    title: str = "Distribution",
) -> None:
    """棒グラフで分布を可視化する"""
    plt = _get_plt()

    # CSV 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(save_path).with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("name,count\n")
        for name, count in sorted(distribution.items(), key=lambda x: -x[1]):
            f.write(f"{name},{count}\n")

    if plt is None:
        return

    names = list(distribution.keys())
    counts = [distribution[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names)), 4))
    bars = ax.bar(range(len(names)), counts)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Action Confidence Histogram
# ---------------------------------------------------------------------------

def plot_confidence_histogram(
    confidences: List[float],
    save_path: str,
    title: str = "Action Confidence",
    bins: int = 20,
) -> None:
    """confidence score のヒストグラムを描画する"""
    plt = _get_plt()

    stats = {
        "mean": float(np.mean(confidences)) if confidences else 0.0,
        "std": float(np.std(confidences)) if confidences else 0.0,
        "min": float(np.min(confidences)) if confidences else 0.0,
        "max": float(np.max(confidences)) if confidences else 0.0,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(save_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(stats, f)

    if plt is None or not confidences:
        return

    fig, ax = plt.subplots()
    ax.hist(confidences, bins=bins, edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\nmean={stats['mean']:.3f}, std={stats['std']:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ID Switch Analyzer
# ---------------------------------------------------------------------------

class IDSwitchAnalyzer:
    """
    予測結果の ID switch を検出・分析する。

    ID switch の定義:
    連続フレームで bbox が overlap している同一物体に
    異なる ID が割り当てられた場合
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._prev_detections: Optional[Dict] = None
        self.switch_events: List[Dict] = []

    def update(
        self,
        frame_idx: int,
        boxes_xyxy: np.ndarray,  # (N, 4)
        track_ids: np.ndarray,   # (N,)
    ) -> List[Dict]:
        """
        前フレームと比較して ID switch を検出する。
        Returns: このフレームで発生した switch イベントのリスト
        """
        events = []
        if self._prev_detections is not None and len(boxes_xyxy) > 0:
            prev_boxes = self._prev_detections["boxes"]
            prev_ids = self._prev_detections["track_ids"]

            if len(prev_boxes) > 0:
                from ..utils.misc import box_iou
                import torch
                curr_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
                prev_t = torch.tensor(prev_boxes, dtype=torch.float32)
                iou_mat = box_iou(curr_t, prev_t).numpy()  # (N_curr, N_prev)

                for i in range(len(boxes_xyxy)):
                    j = iou_mat[i].argmax()
                    if iou_mat[i, j] >= self.iou_threshold:
                        if track_ids[i] != prev_ids[j]:
                            event = {
                                "frame": frame_idx,
                                "prev_id": int(prev_ids[j]),
                                "curr_id": int(track_ids[i]),
                                "iou": float(iou_mat[i, j]),
                            }
                            events.append(event)
                            self.switch_events.append(event)

        self._prev_detections = {
            "boxes": boxes_xyxy.copy(),
            "track_ids": track_ids.copy(),
        }
        return events

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_id_switches": len(self.switch_events),
            "switch_events": self.switch_events[:100],  # 先頭100件のみ
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


# ---------------------------------------------------------------------------
# Ablation Study 結果保存
# ---------------------------------------------------------------------------

class AblationResultManager:
    """
    Ablation 実験の結果を管理・比較するクラス。

    使い方:
        mgr = AblationResultManager("outputs/ablation")
        mgr.add_result("full_model", metrics)
        mgr.add_result("no_long_branch", metrics)
        mgr.save_comparison()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def add_result(self, name: str, metrics: Dict[str, Any]) -> None:
        self.results[name] = metrics
        # 個別ファイルにも保存
        with open(self.output_dir / f"{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def save_comparison(self, filename: str = "ablation_comparison.json") -> None:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Ablation comparison saved to {path}")

    def print_comparison(self, metrics_to_show: Optional[List[str]] = None) -> None:
        """テーブル形式で比較表示する"""
        if not self.results:
            print("No results to compare")
            return

        # 表示するメトリクスを決める
        all_keys = set()
        for m in self.results.values():
            all_keys.update(m.keys())

        keys = metrics_to_show or sorted(all_keys)
        header = ["Model"] + keys
        print(" | ".join(f"{h:20s}" for h in header))
        print("-" * (22 * len(header)))

        for name, metrics in self.results.items():
            row = [name] + [str(metrics.get(k, "-"))[:20] for k in keys]
            print(" | ".join(f"{v:20s}" for v in row))
