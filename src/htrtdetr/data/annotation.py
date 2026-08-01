"""
annotation.py — アノテーション仕様・検証・ユーティリティ

=== Annotation Format Specification (v1.1) ===

ファイル形式: JSON (UTF-8)

{
  "meta": {
    "version": "1.1",
    "description": "Drosophila melanogaster behavior dataset",
    "created": "YYYY-MM-DD",
    "fps_default": 25.0,
    "image_root": "images/"          // image_path はここからの相対パス
  },
  "class_names": ["fly"],            // class_id の対応表
  "action_names": ["idle", "walk", "groom", "court", "other"],  // action_id の対応表
  "videos": [
    {
      "video_id": "vid_001",
      "video_path": "videos/vid_001.mp4",   // optional
      "fps": 25.0,
      "width": 1024,
      "height": 1024,
      "num_frames": 300,
      "frames": [
        {
          "frame_index": 0,
          "image_path": "images/vid_001/000000.jpg",
          "width": 1024,     // optional, video レベルと同じなら省略可
          "height": 1024,
          "objects": [
            {
              "object_id": 1,           // フレーム内のユニーク番号
              "bbox": [x1, y1, x2, y2], // pixel coordinates, left-top / right-bottom
              "class_id": 0,
              "track_id": 1,            // 動画全体でユニーク
              "action_id": 0,           // -1 = unannotated
              "occluded": false,        // optional
              "is_crowd": false         // optional
            }
          ]
        }
      ]
    }
  ]
}

=== 単位の定義 ===
- bbox: pixel 座標 [x1, y1, x2, y2] (left-top, right-bottom), x は width 方向, y は height 方向
- class_id: 0-indexed, class_names のインデックス
- track_id: 動画内でユニーク (0-indexed 推奨, -1 は未追跡)
- action_id: 0-indexed, action_names のインデックス (-1 は未アノテーション)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 内部データ表現 (既存 dataset.py の ObjectAnnotation / FrameAnnotation と互換)
# ---------------------------------------------------------------------------

@dataclass
class ObjectAnno:
    """1 フレーム内の 1 個体アノテーション"""
    object_id: int
    bbox: List[float]          # [x1, y1, x2, y2] pixel coords
    class_id: int = 0
    track_id: int = -1
    action_id: int = -1
    occluded: bool = False
    is_crowd: bool = False

    def bbox_normalized(self, w: int, h: int) -> List[float]:
        """pixel → [0,1] 正規化 (xyxy のまま)"""
        x1, y1, x2, y2 = self.bbox
        return [x1 / w, y1 / h, x2 / w, y2 / h]

    def bbox_cxcywh(self) -> List[float]:
        """[x1,y1,x2,y2] → [cx, cy, w, h] (pixel)"""
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

    def bbox_cxcywh_normalized(self, w: int, h: int) -> List[float]:
        cx, cy, bw, bh = self.bbox_cxcywh()
        return [cx / w, cy / h, bw / w, bh / h]


@dataclass
class FrameAnno:
    """1 フレームのアノテーション"""
    frame_index: int
    image_path: str
    width: int = 0
    height: int = 0
    objects: List[ObjectAnno] = field(default_factory=list)


@dataclass
class VideoAnno:
    """1 動画のアノテーション"""
    video_id: str
    fps: float = 25.0
    width: int = 0
    height: int = 0
    num_frames: int = 0
    video_path: str = ""
    frames: List[FrameAnno] = field(default_factory=list)


@dataclass
class DatasetAnno:
    """データセット全体"""
    meta: Dict[str, Any] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=lambda: ["fly"])
    action_names: List[str] = field(
        default_factory=lambda: ["idle", "walk", "groom", "court", "other"]
    )
    videos: List[VideoAnno] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    def get_all_frames(self) -> List[Tuple[str, FrameAnno]]:
        """(video_id, FrameAnno) のフラットリストを返す"""
        result = []
        for video in self.videos:
            for frame in video.frames:
                result.append((video.video_id, frame))
        return result

    def get_video_size(self, video_id: str) -> Tuple[int, int]:
        """(width, height) を返す"""
        for v in self.videos:
            if v.video_id == video_id:
                return v.width, v.height
        return 0, 0


# ---------------------------------------------------------------------------
# JSON 読み込み / 保存
# ---------------------------------------------------------------------------

def load_annotation(path: str) -> DatasetAnno:
    """JSON アノテーションファイルを読み込み DatasetAnno を返す"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("meta", {})
    class_names = raw.get("class_names", ["fly"])
    action_names = raw.get("action_names", ["unknown"])

    videos = []
    for v in raw.get("videos", []):
        frames = []
        v_width = v.get("width", 0)
        v_height = v.get("height", 0)

        for fr in v.get("frames", []):
            objects = []
            f_width = fr.get("width", v_width)
            f_height = fr.get("height", v_height)

            for i, obj in enumerate(fr.get("objects", [])):
                objects.append(ObjectAnno(
                    object_id=obj.get("object_id", i),
                    bbox=obj["bbox"],
                    class_id=obj.get("class_id", 0),
                    track_id=obj.get("track_id", -1),
                    action_id=obj.get("action_id", -1),
                    occluded=obj.get("occluded", False),
                    is_crowd=obj.get("is_crowd", False),
                ))
            frames.append(FrameAnno(
                frame_index=fr["frame_index"],
                image_path=fr["image_path"],
                width=f_width,
                height=f_height,
                objects=objects,
            ))

        videos.append(VideoAnno(
            video_id=v["video_id"],
            fps=v.get("fps", meta.get("fps_default", 25.0)),
            width=v_width,
            height=v_height,
            num_frames=v.get("num_frames", len(frames)),
            video_path=v.get("video_path", ""),
            frames=frames,
        ))

    return DatasetAnno(
        meta=meta,
        class_names=class_names,
        action_names=action_names,
        videos=videos,
    )


_YOLO_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _read_yolo_classes(classes_file: Path) -> List[str]:
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _read_yolo_label(label_file: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    if not label_file.exists():
        return rows
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
            rows.append((cid, cx, cy, w, h))
    return rows


def _probe_image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image. Falls back to (0, 0) if unreadable."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


def _find_yolo_pairs(
    root: Path,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
) -> Tuple[Path, Path, Optional[Path]]:
    """Locate images/, labels/, and classes.txt inside a YOLO dataset root.

    Supports 3 layouts:
      1. root/images/*.jpg  + root/labels/*.txt
      2. root/*.jpg + root/*.txt (flat)
      3. root/<split>/images/... (returns as-is if caller passes the split dir)
    """
    img_dir = root / images_subdir
    lbl_dir = root / labels_subdir
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        # Flat layout: images and labels colocated in root
        img_dir = root
        lbl_dir = root

    classes_file: Optional[Path] = None
    for candidate in ("classes.txt", "obj.names", "labels.txt"):
        p = root / candidate
        if p.exists():
            classes_file = p
            break
    return img_dir, lbl_dir, classes_file


def load_yolo_annotation(
    dataset_dir: str,
    classes_file: str = "",
    default_class_names: Optional[List[str]] = None,
    video_id: Optional[str] = None,
    group_by_prefix: bool = True,
    fps: float = 30.0,
) -> DatasetAnno:
    """Load a YOLO-format detection dataset into a DatasetAnno.

    Args:
        dataset_dir: Root directory containing images and labels. Supports both
            ``root/images`` + ``root/labels`` layout and a flat layout.
        classes_file: Optional explicit path to ``classes.txt``. Falls back to
            ``<dataset_dir>/classes.txt`` (or ``obj.names`` / ``labels.txt``).
        default_class_names: Used if no classes file is found.
        video_id: Force a single video_id. Ignored when ``group_by_prefix=True``
            and the filenames encode a prefix (e.g. ``fly_copulation_01_5241.jpg``
            → video_id ``fly_copulation_01``).
        group_by_prefix: When True, group frames whose filename shares the prefix
            up to the last underscore-separated numeric segment into the same
            video. This treats sequential frames as a temporal sequence.
        fps: Frames per second recorded on the VideoAnno.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"YOLO dataset dir not found: {root}")

    img_dir, lbl_dir, auto_classes = _find_yolo_pairs(root)

    classes_path = Path(classes_file) if classes_file else auto_classes
    if classes_path and classes_path.exists():
        class_names = _read_yolo_classes(classes_path)
    else:
        class_names = list(default_class_names or ["fly"])

    # Gather image files
    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _YOLO_IMAGE_EXTS
    )
    if not image_files:
        raise FileNotFoundError(f"No YOLO images found in {img_dir}")

    def _group_key(stem: str) -> str:
        if not group_by_prefix:
            return video_id or "video_0"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            return parts[0]
        return stem

    def _frame_index(stem: str, fallback: int) -> int:
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            try:
                return int(parts[1])
            except ValueError:
                return fallback
        return fallback

    # {video_id: [(frame_index, image_path, label_path, width, height)]}
    groups: Dict[str, List[Tuple[int, Path, Path, int, int]]] = {}
    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        vid = video_id or _group_key(stem)
        fi = _frame_index(stem, i)
        lbl_path = lbl_dir / f"{stem}.txt"
        w, h = _probe_image_size(img_path)
        groups.setdefault(vid, []).append((fi, img_path, lbl_path, w, h))

    videos: List[VideoAnno] = []
    obj_uid = 0
    for vid, entries in sorted(groups.items()):
        entries.sort(key=lambda t: t[0])
        # Video-level size: majority of frames (defaults to first frame).
        first_w, first_h = entries[0][3], entries[0][4]
        frames: List[FrameAnno] = []
        for fi, img_path, lbl_path, w, h in entries:
            objects: List[ObjectAnno] = []
            fw = w or first_w or 0
            fh = h or first_h or 0
            for cid, cx, cy, bw, bh in _read_yolo_label(lbl_path):
                if fw <= 0 or fh <= 0:
                    # Store normalized boxes verbatim; consumer will scale later
                    x1 = cx - bw / 2.0
                    y1 = cy - bh / 2.0
                    x2 = cx + bw / 2.0
                    y2 = cy + bh / 2.0
                else:
                    x1 = (cx - bw / 2.0) * fw
                    y1 = (cy - bh / 2.0) * fh
                    x2 = (cx + bw / 2.0) * fw
                    y2 = (cy + bh / 2.0) * fh
                obj_uid += 1
                objects.append(ObjectAnno(
                    object_id=obj_uid,
                    bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    class_id=int(cid),
                    track_id=-1,
                    action_id=-1,
                ))
            try:
                rel_path = str(img_path.relative_to(root))
            except ValueError:
                rel_path = str(img_path)
            frames.append(FrameAnno(
                frame_index=int(fi),
                image_path=rel_path.replace("\\", "/"),
                width=fw,
                height=fh,
                objects=objects,
            ))
        videos.append(VideoAnno(
            video_id=vid,
            fps=fps,
            width=first_w,
            height=first_h,
            num_frames=len(frames),
            frames=frames,
        ))

    return DatasetAnno(
        meta={
            "version": "1.1",
            "source_format": "yolo",
            "image_root": str(img_dir.relative_to(root)).replace("\\", "/")
            if img_dir != root else "",
        },
        class_names=class_names,
        action_names=list(class_names),
        videos=videos,
    )


def save_annotation(anno: DatasetAnno, path: str) -> None:
    """DatasetAnno を JSON ファイルに保存する"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _frame_to_dict(fr: FrameAnno, include_size: bool = True) -> dict:
        d: Dict[str, Any] = {
            "frame_index": fr.frame_index,
            "image_path": fr.image_path,
        }
        if include_size and (fr.width > 0 or fr.height > 0):
            d["width"] = fr.width
            d["height"] = fr.height
        d["objects"] = [
            {
                "object_id": obj.object_id,
                "bbox": obj.bbox,
                "class_id": obj.class_id,
                "track_id": obj.track_id,
                "action_id": obj.action_id,
                **({"occluded": obj.occluded} if obj.occluded else {}),
                **({"is_crowd": obj.is_crowd} if obj.is_crowd else {}),
            }
            for obj in fr.objects
        ]
        return d

    out: Dict[str, Any] = {
        "meta": anno.meta,
        "class_names": anno.class_names,
        "action_names": anno.action_names,
        "videos": [],
    }
    for v in anno.videos:
        vd: Dict[str, Any] = {
            "video_id": v.video_id,
            "fps": v.fps,
            "width": v.width,
            "height": v.height,
            "num_frames": v.num_frames,
        }
        if v.video_path:
            vd["video_path"] = v.video_path
        vd["frames"] = [_frame_to_dict(fr) for fr in v.frames]
        out["videos"].append(vd)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# アノテーション検証
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Valid: {self.valid}"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_annotation(anno: DatasetAnno) -> ValidationResult:
    """アノテーションの整合性を検証する"""
    result = ValidationResult()

    if not anno.videos:
        result.errors.append("No videos in annotation")
        result.valid = False
        return result

    num_classes = len(anno.class_names)
    num_actions = len(anno.action_names)

    for v in anno.videos:
        if v.width <= 0 or v.height <= 0:
            result.warnings.append(f"Video {v.video_id}: missing/invalid size ({v.width}×{v.height})")

        frame_indices = set()
        for fr in v.frames:
            if fr.frame_index in frame_indices:
                result.errors.append(
                    f"Video {v.video_id}: duplicate frame_index {fr.frame_index}"
                )
                result.valid = False
            frame_indices.add(fr.frame_index)

            W, H = fr.width or v.width, fr.height or v.height

            for obj in fr.objects:
                x1, y1, x2, y2 = obj.bbox
                # bbox の validity check
                if x2 <= x1 or y2 <= y1:
                    result.errors.append(
                        f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                        f"Invalid bbox {obj.bbox}"
                    )
                    result.valid = False
                if W > 0:
                    if x1 < 0 or x2 > W:
                        result.warnings.append(
                            f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                            f"bbox x out of range [{x1}, {x2}] vs width {W}"
                        )
                if H > 0:
                    if y1 < 0 or y2 > H:
                        result.warnings.append(
                            f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                            f"bbox y out of range [{y1}, {y2}] vs height {H}"
                        )

                if not (0 <= obj.class_id < num_classes):
                    result.errors.append(
                        f"Video {v.video_id} Frame {fr.frame_index}: "
                        f"class_id {obj.class_id} out of range [0, {num_classes-1}]"
                    )
                    result.valid = False

                if obj.action_id != -1 and not (0 <= obj.action_id < num_actions):
                    result.errors.append(
                        f"Video {v.video_id} Frame {fr.frame_index}: "
                        f"action_id {obj.action_id} out of range [0, {num_actions-1}]"
                    )
                    result.valid = False

    return result


# ---------------------------------------------------------------------------
# サンプルアノテーション生成 (テスト・デバッグ用)
# ---------------------------------------------------------------------------

def generate_sample_annotation(
    num_videos: int = 3,
    num_frames_per_video: int = 60,
    num_flies: int = 5,
    image_width: int = 512,
    image_height: int = 512,
    fps: float = 25.0,
    fly_size: int = 20,       # approx bbox size in pixels
    seed: int = 42,
    action_names: Optional[List[str]] = None,
) -> DatasetAnno:
    """
    ショウジョウバエデータセットの合成サンプルアノテーションを生成する。
    実際の画像は生成しない (image_path は参照のみ)。

    各ハエは:
    - ランダムウォークで動く
    - 毎フレーム action をランダムに変化させる (遷移行列付き)
    - サイズはわずかに変動する
    """
    rng = random.Random(seed)

    if action_names is None:
        action_names = ["idle", "walk", "groom", "court", "other"]

    # 行動遷移行列 (action_id × action_id → 遷移確率)
    # idle は持続しやすい, walk も持続しやすい
    num_actions = len(action_names)
    action_transition = [
        [0.70, 0.15, 0.08, 0.04, 0.03],  # from idle
        [0.10, 0.65, 0.10, 0.10, 0.05],  # from walk
        [0.15, 0.10, 0.60, 0.10, 0.05],  # from groom
        [0.10, 0.15, 0.10, 0.60, 0.05],  # from court
        [0.25, 0.25, 0.20, 0.15, 0.15],  # from other
    ]
    # num_actions が 5 以外の場合は uniform にする
    if num_actions != 5:
        action_transition = [
            [1.0 / num_actions] * num_actions for _ in range(num_actions)
        ]

    videos = []
    for v_idx in range(num_videos):
        video_id = f"vid_{v_idx+1:03d}"

        # 各ハエの初期位置・状態
        fly_states = []
        for f_idx in range(num_flies):
            x = rng.uniform(fly_size, image_width - fly_size)
            y = rng.uniform(fly_size, image_height - fly_size)
            vx = rng.uniform(-2, 2)  # velocity
            vy = rng.uniform(-2, 2)
            action = rng.randint(0, num_actions - 1)
            fly_states.append({
                "track_id": f_idx + 1,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "action": action,
            })

        frames = []
        for fi in range(num_frames_per_video):
            objects = []
            for obj_i, state in enumerate(fly_states):
                # 位置更新 (random walk)
                state["vx"] += rng.gauss(0, 0.5)
                state["vy"] += rng.gauss(0, 0.5)
                state["vx"] = max(-5, min(5, state["vx"]))
                state["vy"] = max(-5, min(5, state["vy"]))
                state["x"] = max(fly_size, min(image_width - fly_size, state["x"] + state["vx"]))
                state["y"] = max(fly_size, min(image_height - fly_size, state["y"] + state["vy"]))

                # action 遷移
                probs = action_transition[state["action"]]
                r = rng.random()
                cumulative = 0.0
                for new_action, p in enumerate(probs):
                    cumulative += p
                    if r <= cumulative:
                        state["action"] = new_action
                        break

                # bbox (わずかにサイズ変動)
                bw = fly_size + rng.uniform(-2, 2)
                bh = fly_size + rng.uniform(-2, 2)
                x1 = max(0, state["x"] - bw / 2)
                y1 = max(0, state["y"] - bh / 2)
                x2 = min(image_width, state["x"] + bw / 2)
                y2 = min(image_height, state["y"] + bh / 2)

                objects.append(ObjectAnno(
                    object_id=obj_i + 1,
                    bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    class_id=0,
                    track_id=state["track_id"],
                    action_id=state["action"],
                ))

            frames.append(FrameAnno(
                frame_index=fi,
                image_path=f"images/{video_id}/{fi:06d}.jpg",
                width=image_width,
                height=image_height,
                objects=objects,
            ))

        videos.append(VideoAnno(
            video_id=video_id,
            fps=fps,
            width=image_width,
            height=image_height,
            num_frames=num_frames_per_video,
            frames=frames,
        ))

    return DatasetAnno(
        meta={
            "version": "1.1",
            "description": "Synthetic Drosophila sample annotation",
            "created": str(date.today()),
            "fps_default": fps,
            "image_root": "images/",
            "note": "Synthetically generated for testing. No actual images.",
        },
        class_names=["fly"],
        action_names=action_names,
        videos=videos,
    )


def print_annotation_stats(anno: DatasetAnno) -> None:
    """アノテーションの統計情報を表示する"""
    from collections import Counter

    total_frames = sum(len(v.frames) for v in anno.videos)
    total_objects = sum(
        len(fr.objects) for v in anno.videos for fr in v.frames
    )

    print(f"=== Annotation Statistics ===")
    print(f"Videos:       {len(anno.videos)}")
    print(f"Frames:       {total_frames}")
    print(f"Objects:      {total_objects}")
    print(f"Avg obj/frame: {total_objects / max(total_frames, 1):.1f}")
    print(f"Classes:      {anno.class_names}")
    print(f"Actions:      {anno.action_names}")

    # Action distribution
    action_counter: Counter = Counter()
    for v in anno.videos:
        for fr in v.frames:
            for obj in fr.objects:
                if obj.action_id >= 0:
                    name = anno.action_names[obj.action_id] \
                        if obj.action_id < len(anno.action_names) else str(obj.action_id)
                    action_counter[name] += 1

    if action_counter:
        print("Action distribution:")
        total = sum(action_counter.values())
        for name, cnt in sorted(action_counter.items(), key=lambda x: -x[1]):
            print(f"  {name:15s}: {cnt:5d} ({cnt/total*100:.1f}%)")
