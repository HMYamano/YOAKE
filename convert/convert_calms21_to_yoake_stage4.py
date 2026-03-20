# convert_calms21_to_yoake_stage4.py
# CalMS21 は HDF5 形式。行動ラベルと bbox (keypoints) を持つ。
# 連続フレームの bbox から track_id を Hungarian matching で自動生成する。
# 実行例 (Anaconda PowerShell):
#   python convert/convert_calms21_to_yoake_stage4.py `
#       --calms21_npy C:/Users/utopi/YOAKE_pre-train/raw/calms21/calms21_task1_train.npy `
#       --output C:/Users/utopi/YOAKE_pre-train/data/stage4/train/annotations.json `
#       --window_size 16

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


# CalMS21 task1 action labels
CALMS21_ACTIONS = ["attack", "investigation", "mount", "other"]


def bbox_iou(b1, b2):
    """[x1,y1,x2,y2] 形式の IoU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def assign_track_ids(bboxes_per_frame: list) -> list:
    """
    各フレームの bbox リストから IoU ベースで track_id を割り当てる。
    bboxes_per_frame: List[List[[x1,y1,x2,y2]]]
    returns: List[List[int]] (各フレームの各 bbox に対する track_id)
    """
    next_id = 0
    prev_boxes = []
    prev_ids = []
    all_track_ids = []

    for boxes in bboxes_per_frame:
        if not boxes:
            all_track_ids.append([])
            prev_boxes, prev_ids = [], []
            continue

        if not prev_boxes:
            # 最初のフレーム: 全て新規 ID
            ids = list(range(next_id, next_id + len(boxes)))
            next_id += len(boxes)
        else:
            # IoU コスト行列
            cost = np.zeros((len(boxes), len(prev_boxes)))
            for i, b in enumerate(boxes):
                for j, pb in enumerate(prev_boxes):
                    cost[i, j] = 1.0 - bbox_iou(b, pb)

            row_ind, col_ind = linear_sum_assignment(cost)
            ids = [-1] * len(boxes)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 0.7:  # IoU > 0.3 なら同一個体
                    ids[r] = prev_ids[c]
                else:
                    ids[r] = next_id
                    next_id += 1
            for i in range(len(boxes)):
                if ids[i] == -1:
                    ids[i] = next_id
                    next_id += 1

        all_track_ids.append(ids)
        prev_boxes = boxes
        prev_ids = ids

    return all_track_ids


def convert_calms21(npy_path: str, output_path: str, window_size: int = 16) -> None:
    """
    CalMS21 .npy ファイルを YOAKE Stage 4 形式に変換する。

    CalMS21 task1 形式:
      data[seq_name] = {
        "keypoints": np.ndarray (T, 2, 7, 2),  # 2 animals, 7 keypoints, xy
        "annotations": np.ndarray (T,),          # 0-3 behavior labels
        "metadata": {"fps": ..., ...}
      }
    """
    data = np.load(npy_path, allow_pickle=True).item()

    videos = []
    for seq_name, seq_data in data.items():
        keypoints = seq_data["keypoints"]  # (T, 2, 7, 2) [animal, kpt, xy]
        labels = seq_data.get("annotations", np.full(keypoints.shape[0], -1))
        T = keypoints.shape[0]
        n_animals = keypoints.shape[1]

        # bbox をキーポイントの bounding box として計算
        # keypoints: (T, n_animals, 7, 2) → bbox per animal per frame
        # 座標は pixel 単位と仮定 (実際の CalMS21 は 1024x570 等)
        W, H = 1024, 570  # CalMS21 の標準解像度

        bboxes_per_frame_per_animal = []
        for t in range(T):
            frame_boxes = []
            for a in range(n_animals):
                kpts = keypoints[t, a]  # (7, 2) xy
                x1 = float(np.min(kpts[:, 0])) - 10
                y1 = float(np.min(kpts[:, 1])) - 10
                x2 = float(np.max(kpts[:, 0])) + 10
                y2 = float(np.max(kpts[:, 1])) + 10
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                frame_boxes.append([x1, y1, x2, y2])
            bboxes_per_frame_per_animal.append(frame_boxes)

        # 各動物の track_id を割り当て (動物ごとに独立して処理)
        track_ids_per_animal = []
        for a in range(n_animals):
            boxes_a = [bboxes_per_frame_per_animal[t][a] for t in range(T)]
            tids = assign_track_ids([[b] for b in boxes_a])
            track_ids_per_animal.append([t[0] if t else -1 for t in tids])

        # window_size フレームずつ切り出して video を生成
        for start in range(0, T - window_size + 1, window_size // 2):
            end = start + window_size
            frames = []
            for fi, t in enumerate(range(start, end)):
                action_id = int(labels[t]) if labels[t] >= 0 else -1
                objects = []
                for a in range(n_animals):
                    x1, y1, x2, y2 = bboxes_per_frame_per_animal[t][a]
                    objects.append({
                        "object_id": a + 1,
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2, 2), round(y2, 2)],
                        "class_id": 0,
                        "track_id": track_ids_per_animal[a][t],
                        "action_id": action_id,
                    })
                frames.append({
                    "frame_index": fi,
                    "image_path": f"calms21/{seq_name}/{t:06d}.jpg",
                    "width": W,
                    "height": H,
                    "objects": objects,
                })

            videos.append({
                "video_id": f"calms21_{seq_name}_{start}",
                "fps": float(seq_data.get("metadata", {}).get("fps", 30.0)),
                "width": W,
                "height": H,
                "num_frames": window_size,
                "frames": frames,
            })

    result = {
        "meta": {
            "version": "1.1",
            "description": "CalMS21 with pseudo track_id for Stage 4",
            "created": "2026-03-18",
        },
        "class_names": ["mouse"],
        "action_names": CALMS21_ACTIONS,
        "videos": videos,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(videos)} windows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calms21_npy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    args = parser.parse_args()
    convert_calms21(args.calms21_npy, args.output, args.window_size)
