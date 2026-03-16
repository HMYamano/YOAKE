"""
summarize_dataset.py — Dataset Statistics & Visualization

annotation JSON を読み込み、以下を集計・出力する:
  - 動画・track・frame 数の要約
  - action クラス分布（棒グラフ）
  - bbox サイズ分布（面積・縦横比ヒストグラム）
  - track 長さ分布
  - フレーム密度（動画ごとの track 数）
  - 全統計 JSON

使い方:
  python tools/summarize_dataset.py \
      anno=data/sample/annotations_train.json \
      output_dir=outputs/dataset_summary \
      plots=1

依存: matplotlib (optional — plots=0 でスキップ)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Statistics extraction
# ---------------------------------------------------------------------------

def _process_bbox(bbox, vid_w, vid_h,
                  bbox_areas, bbox_aspects, bbox_widths, bbox_heights):
    """Normalize and record a bbox (xyxy or xywh)."""
    if not bbox or len(bbox) < 4:
        return
    x1, y1, x2, y2 = bbox[:4]
    # Heuristic: if any coord > 2, treat as pixel
    if max(x2, y2) > 2.0:
        w = (x2 - x1) / max(vid_w, 1)
        h = (y2 - y1) / max(vid_h, 1)
    else:
        w = x2 - x1
        h = y2 - y1
    if w > 0 and h > 0:
        bbox_areas.append(w * h)
        bbox_aspects.append(w / h)
        bbox_widths.append(w)
        bbox_heights.append(h)


def _extract_stats(anno: dict) -> dict:
    """Walk annotation tree and collect all statistics.

    Supports two schemas:
    - Schema A: videos[].frames[].objects[] (per-frame objects with track_id)
    - Schema B: videos[].tracks[].frames[]  (per-track)
    """
    videos = anno.get("videos", anno.get("sequences", []))
    action_names: List[str] = anno.get("action_names", [])

    n_videos = len(videos)
    n_tracks_per_video: List[int] = []
    track_lengths: List[int] = []
    action_counter: Counter = Counter()
    id_counter: Counter = Counter()

    bbox_areas: List[float] = []
    bbox_aspects: List[float] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    frame_ids: List[int] = []

    for vid in videos:
        vid_w = vid.get("width", 1.0)
        vid_h = vid.get("height", 1.0)

        if "tracks" in vid:
            # Schema B: tracks with per-track frames
            tracks = vid.get("tracks", [])
            n_tracks_per_video.append(len(tracks))

            for track in tracks:
                frames = track.get("frames", [])
                track_lengths.append(len(frames))

                label = track.get("action_label", track.get("action", "unknown"))
                action_counter[label] += 1

                tid = track.get("track_id", track.get("id", -1))
                id_counter[str(tid)] += 1

                for frame in frames:
                    fid = frame.get("frame_id", frame.get("frame", -1))
                    frame_ids.append(int(fid))
                    _process_bbox(frame.get("bbox"), vid_w, vid_h,
                                  bbox_areas, bbox_aspects, bbox_widths, bbox_heights)
        else:
            # Schema A: per-frame objects
            frames = vid.get("frames", [])
            track_map: Dict[int, List] = {}

            for frame in frames:
                fid = frame.get("frame_index", frame.get("frame_id", frame.get("frame", -1)))
                frame_ids.append(int(fid))
                for obj in frame.get("objects", []):
                    tid = int(obj.get("track_id", obj.get("id", -1)))
                    if tid not in track_map:
                        track_map[tid] = []
                    track_map[tid].append(obj)

                    aid = obj.get("action_id", -1)
                    if aid >= 0 and action_names and aid < len(action_names):
                        action_counter[action_names[aid]] += 1
                    elif aid >= 0:
                        action_counter[str(aid)] += 1

                    _process_bbox(obj.get("bbox"), vid_w, vid_h,
                                  bbox_areas, bbox_aspects, bbox_widths, bbox_heights)
                    id_counter[str(tid)] += 1

            n_tracks_per_video.append(len(track_map))
            for tid, objs in track_map.items():
                track_lengths.append(len(objs))

    def _stats(vals: List[float]) -> dict:
        if not vals:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                    "p25": 0, "p50": 0, "p75": 0}
        n = len(vals)
        sorted_v = sorted(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        return {
            "count": n,
            "mean": round(mean, 6),
            "std": round(math.sqrt(variance), 6),
            "min": round(sorted_v[0], 6),
            "max": round(sorted_v[-1], 6),
            "p25": round(sorted_v[int(n * 0.25)], 6),
            "p50": round(sorted_v[int(n * 0.50)], 6),
            "p75": round(sorted_v[int(n * 0.75)], 6),
        }

    total_frames = sum(track_lengths)

    return {
        "n_videos": n_videos,
        "n_tracks": sum(n_tracks_per_video),
        "n_frames_total": total_frames,
        "action_distribution": dict(action_counter.most_common()),
        "track_length_stats": _stats(track_lengths),
        "tracks_per_video_stats": _stats([float(v) for v in n_tracks_per_video]),
        "bbox_area_stats": _stats(bbox_areas),
        "bbox_aspect_stats": _stats(bbox_aspects),
        "bbox_width_stats": _stats(bbox_widths),
        "bbox_height_stats": _stats(bbox_heights),
        "frame_id_range": {
            "min": min(frame_ids) if frame_ids else 0,
            "max": max(frame_ids) if frame_ids else 0,
        },
        "_raw": {
            "bbox_areas": bbox_areas,
            "bbox_aspects": bbox_aspects,
            "track_lengths": track_lengths,
            "n_tracks_per_video": n_tracks_per_video,
            "action_counter": dict(action_counter),
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_action_distribution(
    action_counts: Dict[str, int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(action_counts.keys())
    counts = [action_counts[l] for l in labels]
    colors = plt.cm.tab10.colors[: len(labels)]  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, padding=2)
    ax.set_title("Action Class Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Track Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def _plot_histogram(
    values: List[float],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 30,
    log_scale: bool = False,
) -> None:
    if not values:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if log_scale:
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def _plot_scatter(
    x: List[float],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    alpha: float = 0.3,
    max_points: int = 5000,
) -> None:
    if not x:
        return
    import matplotlib.pyplot as plt

    # Subsample if too many points
    if len(x) > max_points:
        import random
        idx = random.sample(range(len(x)), max_points)
        x = [x[i] for i in idx]
        y = [y[i] for i in idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=4, alpha=alpha, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def _plot_pie(
    action_counts: Dict[str, int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(action_counts.keys())
    sizes = [action_counts[l] for l in labels]
    colors = plt.cm.tab10.colors[: len(labels)]  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140
    )
    ax.set_title("Action Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def _print_summary(stats: dict) -> None:
    print("\n=== Dataset Summary ===")
    print(f"  Videos  : {stats['n_videos']}")
    print(f"  Tracks  : {stats['n_tracks']}")
    print(f"  Frames  : {stats['n_frames_total']}")

    print("\n  Action distribution:")
    for action, cnt in stats["action_distribution"].items():
        bar = "#" * min(40, cnt)
        print(f"    {action:<20} {cnt:>5}  {bar}")

    tl = stats["track_length_stats"]
    print(f"\n  Track length:  mean={tl['mean']:.1f}, std={tl['std']:.1f}, "
          f"min={tl['min']}, max={tl['max']}, p50={tl['p50']}")

    ba = stats["bbox_area_stats"]
    if ba["count"] > 0:
        print(f"  BBox area (normalized):  mean={ba['mean']:.4f}, "
              f"p50={ba['p50']:.4f}, max={ba['max']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse(sys.argv[1:])

    anno_path = args.get("anno", "data/sample/annotations_train.json")
    output_dir = Path(args.get("output_dir", "outputs/dataset_summary"))
    do_plots = args.get("plots", "1").strip().lower() not in ("0", "false", "no")

    print(f"Loading: {anno_path}")
    anno = _load(anno_path)
    stats = _extract_stats(anno)

    _print_summary(stats)

    # Save JSON (without _raw bulk data)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in stats.items() if k != "_raw"}
    json_path = output_dir / "dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Stats saved to: {json_path}")

    if not do_plots:
        print("  Plots skipped (plots=0).")
        return

    # Check matplotlib availability
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
    except ImportError:
        print("  WARNING: matplotlib not found. Skipping plots.")
        return

    raw = stats["_raw"]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    if raw["action_counter"]:
        _plot_action_distribution(raw["action_counter"], plots_dir / "action_distribution_bar.png")
        _plot_pie(raw["action_counter"], plots_dir / "action_distribution_pie.png")

    if raw["track_lengths"]:
        _plot_histogram(
            [float(v) for v in raw["track_lengths"]],
            title="Track Length Distribution",
            xlabel="Frames per Track",
            output_path=plots_dir / "track_length_hist.png",
        )

    if raw["bbox_areas"]:
        _plot_histogram(
            raw["bbox_areas"],
            title="BBox Area Distribution (normalized)",
            xlabel="w × h (normalized)",
            output_path=plots_dir / "bbox_area_hist.png",
        )
        _plot_histogram(
            raw["bbox_aspects"],
            title="BBox Aspect Ratio Distribution",
            xlabel="width / height",
            output_path=plots_dir / "bbox_aspect_hist.png",
        )
        _plot_scatter(
            raw["bbox_areas"],
            raw["bbox_aspects"],
            title="BBox Area vs Aspect Ratio",
            xlabel="area (normalized)",
            ylabel="aspect (w/h)",
            output_path=plots_dir / "bbox_area_vs_aspect.png",
        )

    if raw["n_tracks_per_video"]:
        _plot_histogram(
            [float(v) for v in raw["n_tracks_per_video"]],
            title="Tracks per Video",
            xlabel="Track count",
            output_path=plots_dir / "tracks_per_video_hist.png",
        )

    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
