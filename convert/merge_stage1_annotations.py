# merge_stage1_annotations.py
# 実行例 (Anaconda PowerShell):
#   python convert/merge_stage1_annotations.py `
#       --inputs `
#           C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations.json `
#           C:/Users/utopi/YOAKE_pre-train/data/stage1/train/ap10k_annotations.json `
#       --output C:/Users/utopi/YOAKE_pre-train/data/stage1/train/annotations_merged.json

import argparse
import json
from pathlib import Path


def merge_annotations(input_paths: list, output_path: str) -> None:
    all_videos = []
    all_classes = set()
    video_id_set = set()

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls in data.get("class_names", []):
            all_classes.add(cls)
        for v in data["videos"]:
            if v["video_id"] not in video_id_set:
                all_videos.append(v)
                video_id_set.add(v["video_id"])

    class_names = sorted(all_classes)
    result = {
        "meta": {
            "version": "1.1",
            "description": "COCO + AP-10K merged for Stage 1 pretraining",
            "created": "2026-03-18",
        },
        "class_names": class_names,
        "action_names": ["none"],
        "videos": all_videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(all_videos)} videos, {len(class_names)} classes → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_annotations(args.inputs, args.output)
