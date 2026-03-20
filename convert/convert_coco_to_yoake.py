import argparse
import json
from pathlib import Path


def convert_coco(coco_ann_path: str, image_root: str, output_path: str) -> None:
    with open(coco_ann_path, "r") as f:
        coco = json.load(f)

    # id → info マッピング
    img_info = {img["id"]: img for img in coco["images"]}
    # category id → 0-indexed class_id
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_id_to_class = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in sorted(coco["categories"], key=lambda x: x["id"])]

    # image_id → annotations
    from collections import defaultdict
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_img[ann["image_id"]].append(ann)

    videos = []
    for img_id, img in img_info.items():
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue  # 物体なし画像はスキップ

        rel_path = str(Path(image_root) / img["file_name"])
        objects = []
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            objects.append({
                "object_id": i + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "class_id": cat_id_to_class[ann["category_id"]],
                "track_id": -1,
                "action_id": -1,
                "is_crowd": bool(ann.get("iscrowd", 0)),
            })

        videos.append({
            "video_id": f"coco_{img_id:012d}",
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
            "description": "COCO 2017 converted to YOAKE format",
            "created": "2026-03-18",
            "fps_default": 1.0,
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
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_coco(args.coco_ann, args.image_root, args.output)