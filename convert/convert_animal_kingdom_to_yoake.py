# convert_animal_kingdom_to_yoake.py
# Animal Kingdom の action segment CSV/JSON をシーケンス形式に変換する
# 公式フォーマット: action_recognition/annotation/ 以下に CSV
# 実行例 (Anaconda PowerShell):
#   python convert/convert_animal_kingdom_to_yoake.py `
#       --ann_dir C:/Users/utopi/YOAKE_pre-train/raw/animal_kingdom/annotation/AR `
#       --video_dir C:/Users/utopi/YOAKE_pre-train/raw/animal_kingdom/dataset/AR `
#       --frame_dir C:/Users/utopi/YOAKE_pre-train/raw/animal_kingdom/frames `
#       --output C:/Users/utopi/YOAKE_pre-train/data/stage2/train/annotations.json `
#       --window_size 16 `
#       --split train

import argparse
import json
import os
from pathlib import Path


# Animal Kingdom の action label (一部。実際は 140 クラス)
AK_ACTIONS = [
    "eating", "running", "walking", "swimming", "flying",
    "jumping", "grooming", "fighting", "mating", "resting",
    # ... (実際の AK ラベルに合わせて拡張)
]


def convert_animal_kingdom(
    ann_dir: str,
    frame_dir: str,
    output_path: str,
    window_size: int = 16,
    split: str = "train",
) -> None:
    """
    Animal Kingdom の segment-level アノテーションを YOAKE 形式に変換する。

    Animal Kingdom は各クリップに行動ラベルがつくが bbox は提供されない場合がある。
    bbox なしの場合はフレーム全体を bbox として扱う (x1=0, y1=0, x2=W, y2=H)。
    学習では forward_geo_sequence を使う場合は bbox プロキシで十分。
    """
    ann_csv = Path(ann_dir) / f"{split}.csv"
    if not ann_csv.exists():
        # JSON 形式の場合
        ann_csv = Path(ann_dir) / f"{split}.json"

    if ann_csv.suffix == ".csv":
        import csv
        entries = []
        with open(ann_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    else:
        with open(ann_csv, "r") as f:
            entries = json.load(f)

    # action 名リスト構築 (実際のファイルから収集)
    action_set = set()
    for e in entries:
        action_set.add(e.get("action", e.get("label", "unknown")))
    action_names = sorted(action_set)
    action_to_id = {a: i for i, a in enumerate(action_names)}

    videos = []
    for e in entries:
        vid_id = e.get("video_id", e.get("filename", ""))
        action_name = e.get("action", e.get("label", "unknown"))
        action_id = action_to_id.get(action_name, -1)
        start_f = int(e.get("start_frame", 0))
        end_f = int(e.get("end_frame", start_f + window_size))
        W = int(e.get("width", 640))
        H = int(e.get("height", 480))

        # フレーム画像のパスを構築
        frames = []
        for fi in range(start_f, end_f):
            img_path = str(Path(frame_dir) / vid_id / f"{fi:06d}.jpg")
            frames.append({
                "frame_index": fi - start_f,
                "image_path": img_path,
                "width": W,
                "height": H,
                "objects": [{
                    "object_id": 1,
                    "bbox": [0.0, 0.0, float(W), float(H)],  # 全画面プロキシ
                    "class_id": 0,
                    "track_id": 1,  # 単一個体として仮定
                    "action_id": action_id,
                }],
            })

        videos.append({
            "video_id": f"ak_{vid_id}_{start_f}",
            "fps": float(e.get("fps", 30.0)),
            "width": W,
            "height": H,
            "num_frames": len(frames),
            "frames": frames,
        })

    result = {
        "meta": {
            "version": "1.1",
            "description": "Animal Kingdom converted to YOAKE format",
            "created": "2026-03-18",
        },
        "class_names": ["animal"],
        "action_names": action_names,
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} clips → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    convert_animal_kingdom(args.ann_dir, args.frame_dir, args.output,
                           args.window_size, args.split)
