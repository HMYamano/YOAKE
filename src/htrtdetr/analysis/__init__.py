from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .analyzer import (
    AblationResultManager,
    ActionTimelineVisualizer,
    IDSwitchAnalyzer,
    compute_class_distribution,
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_distribution,
)


def _load_prediction_frames(predictions_path: str) -> List[tuple[int, Dict]]:
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = [(int(frame_idx), payload) for frame_idx, payload in data.items()]
    frames.sort(key=lambda item: item[0])
    return frames


def run_timeline_analysis(predictions_path: str, output_dir: str) -> str:
    frames = _load_prediction_frames(predictions_path)
    track_timelines: Dict[int, List[int]] = {}
    max_action_id = 0
    for _, payload in frames:
        track_ids = payload.get("track_ids", [])
        action_ids = payload.get("action_ids", [])
        for track_id, action_id in zip(track_ids, action_ids):
            action_id_i = int(action_id)
            max_action_id = max(max_action_id, action_id_i)
            track_timelines.setdefault(int(track_id), []).append(action_id_i)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "timeline.png"
    ActionTimelineVisualizer(action_names=[str(i) for i in range(max_action_id + 1 or 1)]).plot(
        track_timelines,
        str(save_path),
    )
    return str(save_path)


def run_distribution_analysis(annotation_path: str, output_dir: str) -> str:
    stats = compute_class_distribution(annotation_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_plot = out / "class_distribution.png"
    plot_distribution(stats.get("class_distribution", {}), str(class_plot), title="Class Distribution")
    action_plot = out / "action_distribution.png"
    plot_distribution(stats.get("action_distribution", {}), str(action_plot), title="Action Distribution")
    (out / "distribution_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(class_plot)


def run_id_switch_analysis(predictions_path: str, output_dir: str) -> str:
    frames = _load_prediction_frames(predictions_path)
    analyzer = IDSwitchAnalyzer()
    for frame_idx, payload in frames:
        boxes = np.array(payload.get("boxes", []), dtype=np.float32)
        track_ids = np.array(payload.get("track_ids", []), dtype=np.int64)
        analyzer.update(frame_idx=frame_idx, boxes_xyxy=boxes, track_ids=track_ids)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "id_switches.json"
    analyzer.save(str(save_path))
    return str(save_path)


def run_confidence_analysis(predictions_path: str, output_dir: str) -> str:
    frames = _load_prediction_frames(predictions_path)
    confidences: List[float] = []
    for _, payload in frames:
        confidences.extend(float(score) for score in payload.get("scores", []))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "confidence_histogram.png"
    plot_confidence_histogram(confidences, str(save_path), title="Detection Confidence")
    return str(save_path)


__all__ = [
    "ActionTimelineVisualizer",
    "plot_confusion_matrix",
    "compute_class_distribution",
    "plot_distribution",
    "plot_confidence_histogram",
    "IDSwitchAnalyzer",
    "AblationResultManager",
    "run_timeline_analysis",
    "run_distribution_analysis",
    "run_id_switch_analysis",
    "run_confidence_analysis",
]
