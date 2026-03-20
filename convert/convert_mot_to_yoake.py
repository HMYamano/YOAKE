# convert_mot_to_yoake.py
# MOT17 / DanceTrack / AnimalTrack はすべて同一のテキスト形式:
# <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
# 実行例 (Anaconda PowerShell):
#   python convert/convert_mot_to_yoake.py `
#       --dataset_root C:/Users/utopi/YOAKE_pre-train/raw/dancetrack/train `
#       --image_root C:/Users/utopi/YOAKE_pre-train/raw/dancetrack `
#       --output C:/Users/utopi/YOAKE_pre-train/data/stage3/train/annotations.json `
#       --dataset_name dancetrack

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_mot_dataset(
    dataset_root: str,
    image_root: str,
    output_path: str,
    dataset_name: str = "mot",
    class_name: str = "object",
) -> None:
    """
    MOT 形式テキストアノテーションを YOAKE 形式に変換する。

    dataset_root 以下の各サブディレクトリが1動画に対応することを想定:
      dataset_root/
        seq01/
          gt/gt.txt
          img1/000001.jpg ...
        seq02/
          ...
    """
    sequences = sorted(p for p in Path(dataset_root).iterdir() if p.is_dir())
    videos = []

    for seq_dir in sequences:
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            # DanceTrack は annotations 以下の場合もある
            gt_path = seq_dir / "annotations.txt"
        if not gt_path.exists():
            continue

        # seqinfo.ini から映像情報を読む
        seqinfo_path = seq_dir / "seqinfo.ini"
        width, height, fps = 1920, 1080, 25.0
        img_dir = seq_dir / "img1"
        if seqinfo_path.exists():
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(str(seqinfo_path))
            si = cfg["Sequence"]
            width = int(si.get("imWidth", width))
            height = int(si.get("imHeight", height))
            fps = float(si.get("frameRate", fps))
            img_dir = seq_dir / si.get("imDir", "img1")

        # GT アノテーションを読む
        frame_objects = defaultdict(list)
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                if conf < 0.5:
                    continue  # visibility が低いものを除外
                frame_objects[frame_id].append({
                    "track_id": track_id,
                    "bbox": [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height],
                })

        if not frame_objects:
            continue

        frames = []
        for frame_id in sorted(frame_objects.keys()):
            img_name = f"{frame_id:06d}.jpg"
            img_path = str(img_dir / img_name)
            objects = []
            for oi, obj in enumerate(frame_objects[frame_id]):
                x1, y1, x2, y2 = obj["bbox"]
                objects.append({
                    "object_id": oi + 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "class_id": 0,
                    "track_id": obj["track_id"],
                    "action_id": -1,
                })
            frames.append({
                "frame_index": frame_id - 1,
                "image_path": img_path,
                "width": width,
                "height": height,
                "objects": objects,
            })

        videos.append({
            "video_id": f"{dataset_name}_{seq_dir.name}",
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": len(frames),
            "frames": frames,
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": f"{dataset_name} converted to YOAKE format",
            "created": "2026-03-18",
            "image_root": image_root,
        },
        "class_names": [class_name],
        "action_names": ["none"],
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} sequences → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_name", default="mot")
    parser.add_argument("--class_name", default="object")
    args = parser.parse_args()
    convert_mot_dataset(args.dataset_root, args.image_root, args.output,
                        args.dataset_name, args.class_name)
