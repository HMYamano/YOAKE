# convert_ap10k_to_yoake.py
# AP-10K は COCO keypoint 形式 (bbox あり)
# 実行例 (Anaconda PowerShell):
#   python convert/convert_ap10k_to_yoake.py `
#       --ann C:/Users/utopi/YOAKE_pre-train/raw/ap10k/annotations/ap10k-train-split1.json `
#       --image_root C:/Users/utopi/YOAKE_pre-train/raw/ap10k/data `
#       --output C:/Users/utopi/YOAKE_pre-train/data/stage1/train/ap10k_annotations.json

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_ap10k(ann_path: str, image_root: str, output_path: str) -> None:
    with open(ann_path, "r") as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    cat_ids = sorted(c["id"] for c in data["categories"])
    cat_id_to_class = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]

    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    videos = []
    for img_id, img in img_info.items():
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        rel_path = str(Path(image_root) / img["file_name"])
        objects = []
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            objects.append({
                "object_id": i + 1,
                "bbox": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                "class_id": cat_id_to_class[ann["category_id"]],
                "track_id": -1,
                "action_id": -1,
            })

        videos.append({
            "video_id": f"ap10k_{img_id:08d}",
            "fps": 1.0,
            "width": img["width"],
            "height": img["height"],
            "num_frames": 1,
            "frames": [{
                "frame_index": 0,
                "image_path": rel_path,
                "width": img["width"],
                "height": img["height"],
                "objects": objects,
            }],
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": "AP-10K converted to YOAKE format",
            "created": "2026-03-18",
            "image_root": image_root,
        },
        "class_names": class_names,
        "action_names": ["none"],
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} entries → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_ap10k(args.ann, args.image_root, args.output)
