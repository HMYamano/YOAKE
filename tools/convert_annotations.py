"""
convert_annotations.py — 独自データ → YOAKE annotation format 変換

サポートする入力形式:
  1. CSV (bbox/track_id/action_id を含む)
  2. JSON (任意の外部 JSON スキーマ)
  3. 動画ファイル + CSV (フレーム抽出 + アノテーション)

出力:
  - project 標準 annotation JSON (version 1.1)

使い方:
  # CSV → annotation JSON
  python tools/convert_annotations.py \\
      mode=csv \\
      input=data/raw/annotations.csv \\
      output=data/converted/annotations.json \\
      video_width=640 video_height=640

  # JSON (外部スキーマ) → annotation JSON
  python tools/convert_annotations.py \\
      mode=json \\
      input=data/raw/tracking_results.json \\
      output=data/converted/annotations.json \\
      json_schema=sleap

  # 動画 + CSV → フレーム抽出 + annotation JSON
  python tools/convert_annotations.py \\
      mode=video_csv \\
      input=data/raw/experiment.mp4 \\
      csv=data/raw/tracks.csv \\
      output=data/converted/annotations.json \\
      frames_dir=data/frames/experiment \\
      extract_every=1

CSV 列名規約 (mode=csv):
  必須: video_id (or filename), frame_idx (or frame_index or frame), track_id
        x1, y1, x2, y2  (pixel 座標, または cx,cy,w,h — format で指定)
  省略可: class_id (デフォルト 0), action_id (デフォルト -1), image_path
  代替列名: 自動検出するが --column_map で上書き可能

label_map の指定:
  action_names=stationary,walking,grooming,courtship,aggression
  class_names=fly
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htrtdetr.data.annotation import (
    DatasetAnno, VideoAnno, FrameAnno, ObjectAnno,
    save_annotation, validate_annotation,
)


# ---------------------------------------------------------------------------
# CSV Column Auto-detection
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "video_id":  ["video_id", "video", "sequence", "seq_id", "clip_id", "filename"],
    "frame_idx": ["frame_idx", "frame_index", "frame", "t", "time", "frame_number"],
    "track_id":  ["track_id", "track", "id", "object_id", "identity"],
    "x1":        ["x1", "xmin", "left", "x_min"],
    "y1":        ["y1", "ymin", "top", "y_min"],
    "x2":        ["x2", "xmax", "right", "x_max"],
    "y2":        ["y2", "ymax", "bottom", "y_max"],
    "cx":        ["cx", "center_x", "x_center"],
    "cy":        ["cy", "center_y", "y_center"],
    "w":         ["w", "width", "bbox_width"],
    "h":         ["h", "height", "bbox_height"],
    "class_id":  ["class_id", "class", "category_id", "category", "species"],
    "action_id": ["action_id", "action", "behavior", "label", "behavior_id"],
    "image_path": ["image_path", "image", "frame_path", "file"],
}


def _detect_columns(header: List[str]) -> Dict[str, str]:
    """CSV ヘッダから列名マッピングを自動検出する"""
    header_lower = {h.lower(): h for h in header}
    mapping: Dict[str, str] = {}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in header_lower:
                mapping[field] = header_lower[alias]
                break
    return mapping


# ---------------------------------------------------------------------------
# CSV Converter
# ---------------------------------------------------------------------------

def convert_csv(
    csv_path: str,
    output_path: str,
    video_width: int = 640,
    video_height: int = 640,
    fps: float = 30.0,
    bbox_format: str = "xyxy",     # "xyxy" | "cxcywh" | "xywh"
    class_names: Optional[List[str]] = None,
    action_names: Optional[List[str]] = None,
    column_map: Optional[Dict[str, str]] = None,
    image_dir: str = "",
    normalize_coords: bool = False,
) -> DatasetAnno:
    """
    CSV ファイルを annotation JSON に変換する。

    Args:
        csv_path: 入力 CSV ファイルパス
        output_path: 出力 JSON パス
        video_width/height: 動画サイズ (正規化座標の場合は変換に使用)
        bbox_format: bbox の座標形式
        normalize_coords: True なら座標を [0,1] → pixel に変換
        column_map: 列名上書き {'video_id': 'actual_col_name', ...}
    """
    csv_path = Path(csv_path)
    print(f"Converting CSV: {csv_path}")

    # CSV 読み込み
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        col_map = _detect_columns(list(header))
        if column_map:
            col_map.update(column_map)
        for row in reader:
            rows.append(row)

    print(f"  Columns detected: {col_map}")
    print(f"  Total rows: {len(rows)}")

    _check_required(col_map, ["frame_idx", "track_id"])
    has_xyxy = "x1" in col_map and "y1" in col_map and "x2" in col_map and "y2" in col_map
    has_cxcywh = "cx" in col_map and "cy" in col_map and "w" in col_map and "h" in col_map

    if not (has_xyxy or has_cxcywh):
        raise ValueError("CSV must have (x1,y1,x2,y2) or (cx,cy,w,h) columns")

    # video_id が列にない場合は csv のファイル名を使う
    if "video_id" not in col_map:
        default_video_id = csv_path.stem
        print(f"  No video_id column → using '{default_video_id}' as video_id")
    else:
        default_video_id = None

    # 動画ごとにデータを整理: {video_id → {frame_idx → [row]}}
    video_frames: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        vid = row[col_map["video_id"]] if "video_id" in col_map else default_video_id
        fi = int(float(row[col_map["frame_idx"]]))
        video_frames[vid][fi].append(row)

    # DatasetAnno 構築
    videos = []
    global_obj_id = 0

    for vid, frames_dict in sorted(video_frames.items()):
        frame_annos = []
        for fi, frame_rows in sorted(frames_dict.items()):
            objects = []
            for row in frame_rows:
                # bbox 変換
                if has_xyxy:
                    x1 = float(row[col_map["x1"]])
                    y1 = float(row[col_map["y1"]])
                    x2 = float(row[col_map["x2"]])
                    y2 = float(row[col_map["y2"]])
                elif has_cxcywh:
                    cx = float(row[col_map["cx"]])
                    cy = float(row[col_map["cy"]])
                    w  = float(row[col_map["w"]])
                    h  = float(row[col_map["h"]])
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2

                if normalize_coords:
                    x1, x2 = x1 * video_width, x2 * video_width
                    y1, y2 = y1 * video_height, y2 * video_height

                # xywh 形式の変換
                if bbox_format == "xywh" and has_xyxy:
                    # x1,y1 はそのまま; x2,y2 は width/height として解釈し直す
                    w_, h_ = x2, y2  # 列名がずれている場合の再解釈
                    x2, y2 = x1 + w_, y1 + h_

                track_id = int(float(row[col_map["track_id"]]))
                class_id = int(float(row[col_map.get("class_id", "")])) \
                    if "class_id" in col_map else 0
                action_id = int(float(row[col_map.get("action_id", "")])) \
                    if "action_id" in col_map else -1

                global_obj_id += 1
                img_path = ""
                if "image_path" in col_map:
                    img_path = row.get(col_map["image_path"], "")
                elif image_dir:
                    img_path = str(Path(image_dir) / f"frame_{fi:06d}.jpg")

                objects.append(ObjectAnno(
                    object_id=global_obj_id,
                    bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    class_id=class_id,
                    track_id=track_id,
                    action_id=action_id,
                ))

            img_path_frame = img_path if not image_dir else \
                str(Path(image_dir) / f"frame_{fi:06d}.jpg")
            frame_annos.append(FrameAnno(
                frame_index=fi,
                image_path=img_path_frame,
                objects=objects,
            ))

        videos.append(VideoAnno(
            video_id=vid,
            fps=fps,
            width=video_width,
            height=video_height,
            frames=frame_annos,
        ))

    anno = DatasetAnno(
        class_names=class_names or ["fly"],
        action_names=action_names or ["stationary", "walking", "grooming", "courtship", "aggression"],
        videos=videos,
    )

    # 出力
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_annotation(anno, str(output_path))

    # 検証
    val_result = validate_annotation(anno)
    if not val_result.is_valid:
        print(f"  Warnings: {val_result.warnings}")

    n_frames = sum(len(v.frames) for v in videos)
    n_objects = sum(len(f.objects) for v in videos for f in v.frames)
    print(f"  Videos: {len(videos)}, Frames: {n_frames}, Objects: {n_objects}")
    print(f"  Saved: {output_path}")
    return anno


# ---------------------------------------------------------------------------
# Video + CSV → Frame extraction + annotation
# ---------------------------------------------------------------------------

def convert_video_csv(
    video_path: str,
    csv_path: str,
    output_path: str,
    frames_dir: str,
    extract_every: int = 1,
    img_size: Optional[Tuple[int, int]] = None,
    **csv_kwargs,
) -> DatasetAnno:
    """
    動画ファイルからフレームを抽出し、CSV と組み合わせて annotation を生成する。

    Args:
        video_path: 入力動画 (.mp4, .avi, etc.)
        csv_path: アノテーション CSV
        frames_dir: フレーム画像の保存先
        extract_every: N フレームごとに 1 フレーム抽出 (デフォルト: 全フレーム)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) が必要です: pip install opencv-python")

    video_path = Path(video_path)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tgt_w = img_size[1] if img_size else orig_w
    tgt_h = img_size[0] if img_size else orig_h

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % extract_every == 0:
            if img_size:
                frame = cv2.resize(frame, (tgt_w, tgt_h))
            out_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"  Extracted {saved} / {total} frames → {frames_dir}")

    return convert_csv(
        csv_path=csv_path,
        output_path=output_path,
        video_width=tgt_w,
        video_height=tgt_h,
        fps=fps / extract_every,
        image_dir=str(frames_dir),
        **csv_kwargs,
    )


# ---------------------------------------------------------------------------
# JSON schema converters
# ---------------------------------------------------------------------------

def convert_json_generic(
    json_path: str,
    output_path: str,
    schema: str = "generic",
    **kwargs,
) -> DatasetAnno:
    """
    外部 JSON スキーマを annotation JSON に変換する。

    schema:
      "generic"   — {"videos": [{"video_id": ..., "frames": [...]}]}
      "sleap"     — SLEAP tracking JSON
      "deeplabcut" — DeepLabCut output
      "trackmate" — TrackMate XML (JSON で書き出したもの)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if schema == "sleap":
        return _from_sleap(data, output_path, **kwargs)
    elif schema == "deeplabcut":
        return _from_deeplabcut(data, output_path, **kwargs)
    else:
        return _from_generic_json(data, output_path, **kwargs)


def _from_generic_json(data: dict, output_path: str, **kwargs) -> DatasetAnno:
    """
    汎用 JSON: {"videos": [{"video_id", "fps", "width", "height",
                            "frames": [{"frame_index", "image_path",
                                        "objects": [{"object_id", "bbox", "class_id",
                                                     "track_id", "action_id"}]}]}]}
    既に project format に近い形式を想定。
    """
    from htrtdetr.data.annotation import load_annotation
    # ほぼそのままロードを試みる
    try:
        anno = load_annotation(json_path=None, data=data)
    except Exception:
        raise ValueError(
            "generic JSON must match the project annotation format. "
            "Consider using mode=csv or providing explicit schema."
        )
    save_annotation(anno, output_path)
    print(f"Converted generic JSON → {output_path}")
    return anno


def _from_sleap(data: dict, output_path: str,
                action_names: Optional[List[str]] = None,
                **kwargs) -> DatasetAnno:
    """
    SLEAP tracking JSON の簡易変換。
    SLEAP は skeleton track を持つが、ここでは bbox を中心点 ± size で近似する。
    """
    videos = []
    bbox_half = kwargs.get("bbox_half_size", 25)  # pixel

    # SLEAP format varies by version, handle both common layouts
    labels = data.get("labels", [data])
    if isinstance(labels, dict):
        labels = [labels]

    for video_data in labels:
        vid = video_data.get("video", {}).get("filename", "unknown")
        video_id = Path(vid).stem if vid else "video_0"
        width = video_data.get("video", {}).get("width", 640)
        height = video_data.get("video", {}).get("height", 640)
        fps = video_data.get("video", {}).get("fps", 30.0)
        frames_map: Dict[int, List[ObjectAnno]] = defaultdict(list)
        obj_id = 0

        for lf in video_data.get("labeled_frames", []):
            fi = lf.get("frame_idx", lf.get("frame_index", 0))
            for inst in lf.get("instances", []):
                track_id = inst.get("track", 0)
                if isinstance(track_id, dict):
                    track_id = track_id.get("id", 0)
                # points: list of {x, y}
                pts = inst.get("points", [])
                if pts:
                    xs = [p["x"] for p in pts if p.get("x") is not None]
                    ys = [p["y"] for p in pts if p.get("y") is not None]
                    if xs and ys:
                        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                        x1, y1 = cx - bbox_half, cy - bbox_half
                        x2, y2 = cx + bbox_half, cy + bbox_half
                        obj_id += 1
                        frames_map[fi].append(ObjectAnno(
                            object_id=obj_id,
                            bbox=[max(0.0, x1), max(0.0, y1),
                                  min(float(width), x2), min(float(height), y2)],
                            class_id=0, track_id=int(track_id), action_id=-1,
                        ))

        frame_annos = [
            FrameAnno(frame_index=fi, image_path="", objects=objs)
            for fi, objs in sorted(frames_map.items())
        ]
        videos.append(VideoAnno(
            video_id=video_id, fps=fps, width=width, height=height, frames=frame_annos
        ))

    anno = DatasetAnno(
        class_names=["fly"],
        action_names=action_names or ["unknown"],
        videos=videos,
    )
    save_annotation(anno, output_path)
    print(f"Converted SLEAP JSON ({sum(len(v.frames) for v in videos)} frames) → {output_path}")
    return anno


def _from_deeplabcut(data: dict, output_path: str, **kwargs) -> DatasetAnno:
    """DeepLabCut JSON の簡易変換 (姿勢推定 → bbox 近似)"""
    # DLC stores per-individual keypoints
    bbox_half = kwargs.get("bbox_half_size", 30)
    videos = []
    obj_id = 0
    frames_map: Dict[int, List[ObjectAnno]] = defaultdict(list)

    scorer = data.get("scorer", "DLC")
    coords = data.get("coords", {})  # {frame_path: {individual: {bodypart: [x, y]}}}

    for frame_path, individuals in sorted(coords.items()):
        fi = int(Path(frame_path).stem.split("_")[-1]) if "_" in frame_path else obj_id
        for ind_id, (ind_name, keypoints) in enumerate(individuals.items()):
            xs, ys = [], []
            for bp_data in (keypoints.values() if isinstance(keypoints, dict) else []):
                if isinstance(bp_data, (list, tuple)) and len(bp_data) >= 2:
                    xs.append(float(bp_data[0]))
                    ys.append(float(bp_data[1]))
            if xs and ys:
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                obj_id += 1
                frames_map[fi].append(ObjectAnno(
                    object_id=obj_id,
                    bbox=[cx - bbox_half, cy - bbox_half, cx + bbox_half, cy + bbox_half],
                    class_id=0, track_id=ind_id, action_id=-1,
                ))

    frame_annos = [
        FrameAnno(frame_index=fi, image_path="", objects=objs)
        for fi, objs in sorted(frames_map.items())
    ]
    anno = DatasetAnno(
        class_names=["fly"],
        action_names=["unknown"],
        videos=[VideoAnno(video_id="dlc_video", fps=30.0, width=640, height=640,
                          frames=frame_annos)],
    )
    save_annotation(anno, output_path)
    print(f"Converted DLC JSON → {output_path}")
    return anno


# ---------------------------------------------------------------------------
# Label map builder
# ---------------------------------------------------------------------------

def build_label_map(
    anno: DatasetAnno,
    output_path: str = "",
) -> Dict[str, Any]:
    """annotation から label_map.json を生成する"""
    label_map = {
        "class_names": anno.class_names,
        "action_names": anno.action_names,
        "class_to_id": {name: i for i, name in enumerate(anno.class_names)},
        "action_to_id": {name: i for i, name in enumerate(anno.action_names)},
        "id_to_class": {i: name for i, name in enumerate(anno.class_names)},
        "id_to_action": {i: name for i, name in enumerate(anno.action_names)},
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"Label map saved: {out}")
    return label_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_required(col_map: Dict[str, str], required: List[str]) -> None:
    missing = [f for f in required if f not in col_map]
    if missing:
        raise ValueError(
            f"Required columns not found: {missing}. "
            "Check CSV header or use column_map to specify column names."
        )


def parse_overrides(argv: List[str]) -> Dict[str, str]:
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v
    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_overrides(sys.argv[1:])

    mode = args.get("mode", "csv")
    input_path = args.get("input", "")
    output_path = args.get("output", "data/converted/annotations.json")
    video_width = int(args.get("video_width", "640"))
    video_height = int(args.get("video_height", "640"))
    fps = float(args.get("fps", "30.0"))
    bbox_format = args.get("bbox_format", "xyxy")
    image_dir = args.get("image_dir", "")
    normalize = args.get("normalize_coords", "false").lower() == "true"
    label_map_path = args.get("label_map", "")

    class_names = [s.strip() for s in args.get("class_names", "fly").split(",")]
    action_names = [s.strip() for s in args.get(
        "action_names", "stationary,walking,grooming,courtship,aggression"
    ).split(",")]

    if not input_path:
        print("Usage: python tools/convert_annotations.py mode=csv input=<path> output=<path>")
        print("  mode: csv | json | video_csv")
        sys.exit(1)

    if mode == "csv":
        anno = convert_csv(
            csv_path=input_path,
            output_path=output_path,
            video_width=video_width,
            video_height=video_height,
            fps=fps,
            bbox_format=bbox_format,
            class_names=class_names,
            action_names=action_names,
            image_dir=image_dir,
            normalize_coords=normalize,
        )

    elif mode == "json":
        schema = args.get("json_schema", "generic")
        anno = convert_json_generic(
            json_path=input_path,
            output_path=output_path,
            schema=schema,
            action_names=action_names,
        )

    elif mode == "video_csv":
        csv_path = args.get("csv", "")
        frames_dir = args.get("frames_dir", f"data/frames/{Path(input_path).stem}")
        extract_every = int(args.get("extract_every", "1"))
        if not csv_path:
            raise ValueError("mode=video_csv requires csv=<path>")
        anno = convert_video_csv(
            video_path=input_path,
            csv_path=csv_path,
            output_path=output_path,
            frames_dir=frames_dir,
            extract_every=extract_every,
            video_width=video_width,
            video_height=video_height,
            fps=fps,
            class_names=class_names,
            action_names=action_names,
        )

    else:
        print(f"Unknown mode: {mode}. Use: csv | json | video_csv")
        sys.exit(1)

    # Label map
    build_label_map(anno, label_map_path or str(Path(output_path).parent / "label_map.json"))
    print("Done.")


if __name__ == "__main__":
    main()
