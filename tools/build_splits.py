"""
build_splits.py — Train / Val / Test Split Generator

既存の annotation JSON を読み込み、video 単位または track 単位で
train / val / test に分割して別ファイルとして保存する。

使い方:
  python tools/build_splits.py \
      input=data/annotations_all.json \
      output_dir=data/splits \
      val_ratio=0.15 \
      test_ratio=0.15 \
      strategy=video \
      seed=42

strategy:
  video  — video 単位で split（デフォルト・推奨）
  track  — track 単位で split
  frame  — frame 単位で split（label leak に注意）

出力:
  output_dir/annotations_train.json
  output_dir/annotations_val.json
  output_dir/annotations_test.json
  output_dir/split_summary.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse(argv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path} ({len(data.get('videos', data.get('sequences', [])))} items)")


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def _split_by_video(
    anno: dict,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split at the video level (no label leakage)."""
    videos = anno.get("videos", anno.get("sequences", []))
    if not videos:
        raise ValueError("Annotation JSON has no 'videos' or 'sequences' key.")

    rng = random.Random(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Too few videos ({n}) for val_ratio={val_ratio}, test_ratio={test_ratio}."
        )

    train_vids = shuffled[:n_train]
    val_vids = shuffled[n_train : n_train + n_val]
    test_vids = shuffled[n_train + n_val :]
    return train_vids, val_vids, test_vids


def _split_by_track(
    anno: dict,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split at the track level within each video."""
    videos = anno.get("videos", anno.get("sequences", []))
    if not videos:
        raise ValueError("Annotation JSON has no 'videos' or 'sequences' key.")

    # Collect all tracks globally
    all_tracks: List[Tuple[int, int]] = []  # (video_idx, track_idx)
    for vi, vid in enumerate(videos):
        tracks = vid.get("tracks", [])
        for ti in range(len(tracks)):
            all_tracks.append((vi, ti))

    rng = random.Random(seed)
    shuffled = list(all_tracks)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    n_train = n - n_val - n_test

    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train : n_train + n_val])
    test_set = set(shuffled[n_train + n_val :])

    def _filter(video_list: list, track_keys: set) -> List[dict]:
        result = []
        for vi, vid in enumerate(video_list):
            local_tracks = [
                t
                for ti, t in enumerate(vid.get("tracks", []))
                if (vi, ti) in track_keys
            ]
            if local_tracks:
                new_vid = {k: v for k, v in vid.items() if k != "tracks"}
                new_vid["tracks"] = local_tracks
                result.append(new_vid)
        return result

    return (
        _filter(videos, train_set),
        _filter(videos, val_set),
        _filter(videos, test_set),
    )


def _split_by_frame(
    anno: dict,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split at the frame level (may cause label leakage across same track)."""
    videos = anno.get("videos", anno.get("sequences", []))
    if not videos:
        raise ValueError("Annotation JSON has no 'videos' or 'sequences' key.")

    # Collect all (video_idx, track_idx, frame_idx) triples
    all_frames: List[Tuple[int, int, int]] = []
    for vi, vid in enumerate(videos):
        for ti, track in enumerate(vid.get("tracks", [])):
            for fi in range(len(track.get("frames", []))):
                all_frames.append((vi, ti, fi))

    rng = random.Random(seed)
    shuffled = list(all_frames)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0

    train_set = set(shuffled[: n - n_val - n_test])
    val_set = set(shuffled[n - n_val - n_test : n - n_test])
    test_set = set(shuffled[n - n_test :])

    def _filter(frame_keys: set) -> List[dict]:
        result = []
        for vi, vid in enumerate(videos):
            new_tracks = []
            for ti, track in enumerate(vid.get("tracks", [])):
                new_frames = [
                    f
                    for fi, f in enumerate(track.get("frames", []))
                    if (vi, ti, fi) in frame_keys
                ]
                if new_frames:
                    new_t = {k: v for k, v in track.items() if k != "frames"}
                    new_t["frames"] = new_frames
                    new_tracks.append(new_t)
            if new_tracks:
                new_vid = {k: v for k, v in vid.items() if k != "tracks"}
                new_vid["tracks"] = new_tracks
                result.append(new_vid)
        return result

    return _filter(train_set), _filter(val_set), _filter(test_set)


# ---------------------------------------------------------------------------
# Build output annotation dicts
# ---------------------------------------------------------------------------

def _wrap(anno: dict, videos: List[dict], split_name: str) -> dict:
    """Wrap a list of videos back into a full annotation dict."""
    meta = {k: v for k, v in anno.items() if k not in ("videos", "sequences")}
    meta["split"] = split_name
    key = "videos" if "videos" in anno else "sequences"
    meta[key] = videos
    return meta


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _count(videos: List[dict]) -> dict:
    """Count statistics for either schema:
    - Schema A: videos[].frames[].objects[] (per-frame with track_id)
    - Schema B: videos[].tracks[].frames[]  (per-track)
    """
    n_frames = 0
    n_tracks = 0
    track_ids_seen: set = set()
    action_counts: Dict[str, int] = defaultdict(int)

    for v in videos:
        if "tracks" in v:
            # Schema B
            tracks = v.get("tracks", [])
            n_tracks += len(tracks)
            for t in tracks:
                n_frames += len(t.get("frames", []))
                label = t.get("action_label", t.get("action", "unknown"))
                action_counts[label] += 1
        else:
            # Schema A
            frames = v.get("frames", [])
            n_frames += len(frames)
            for fr in frames:
                for obj in fr.get("objects", []):
                    tid = obj.get("track_id", obj.get("id", -1))
                    vid_id = v.get("video_id", id(v))
                    key = (vid_id, tid)
                    if key not in track_ids_seen:
                        track_ids_seen.add(key)
                        n_tracks += 1

    return {
        "n_videos": len(videos),
        "n_tracks": n_tracks,
        "n_frames": n_frames,
        "action_counts": dict(action_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse(sys.argv[1:])

    input_path = args.get("input", "data/sample/annotations_train.json")
    output_dir = Path(args.get("output_dir", "data/splits"))
    val_ratio = float(args.get("val_ratio", "0.15"))
    test_ratio = float(args.get("test_ratio", "0.15"))
    strategy = args.get("strategy", "video")
    seed = int(args.get("seed", "42"))

    print(f"Loading: {input_path}")
    anno = _load(input_path)

    videos_all = anno.get("videos", anno.get("sequences", []))
    print(f"  Total videos: {len(videos_all)}, strategy={strategy}, seed={seed}")
    print(f"  val_ratio={val_ratio}, test_ratio={test_ratio}")

    if strategy == "video":
        train_vids, val_vids, test_vids = _split_by_video(anno, val_ratio, test_ratio, seed)
    elif strategy == "track":
        train_vids, val_vids, test_vids = _split_by_track(anno, val_ratio, test_ratio, seed)
    elif strategy == "frame":
        print("  WARNING: frame-level split may cause label leakage across same track.")
        train_vids, val_vids, test_vids = _split_by_frame(anno, val_ratio, test_ratio, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose: video | track | frame")

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_vids,
        "val": val_vids,
        "test": test_vids,
    }

    summary: Dict[str, dict] = {}
    for split_name, vids in splits.items():
        if not vids:
            print(f"  Skipping empty {split_name} split.")
            continue
        wrapped = _wrap(anno, vids, split_name)
        _save(wrapped, output_dir / f"annotations_{split_name}.json")
        summary[split_name] = _count(vids)

    # Print summary table
    print("\n=== Split Summary ===")
    header = f"{'split':<10} {'videos':>8} {'tracks':>8} {'frames':>8}"
    print(header)
    print("-" * len(header))
    for sname, info in summary.items():
        print(
            f"{sname:<10} {info['n_videos']:>8} {info['n_tracks']:>8} {info['n_frames']:>8}"
        )

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": input_path,
                "strategy": strategy,
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "splits": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
