"""
annotation.py — アノテーション仕様・検証・ユーティリティ

=== Annotation Format Specification (v2.0) ===

v2.0 の変更点 (すべて **後方互換**):
  * ``obb``       — 回転矩形 (任意)。dict / 5要素配列 / 4頂点8要素配列のいずれでも可。
  * ``keypoints`` — pose (任意)。[[x, y, v], ...] または COCO 互換のフラット配列。
  * ``keypoint_names`` / ``action_groups`` — データセットレベルの任意定義。

v1.1 のファイルはそのまま読める (obb なしとして扱われ、必要時に theta=0 で導出)。
OBB を持たないデータを保存すると v1.1 と同一の形になるので、旧ツールでも読める。
移行は ``migrate_annotation()`` / ``migrate_annotation_file()`` を使う。



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
              "is_crowd": false,        // optional
              // --- v2.0 optional ---
              "obb": {"cx": 100, "cy": 200, "w": 40, "h": 15, "angle": 0.52},
              "keypoints": [[100, 195, 2], [100, 205, 2]]
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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 内部データ表現 (既存 dataset.py の ObjectAnnotation / FrameAnnotation と互換)
# ---------------------------------------------------------------------------


@dataclass
class ObjectAnno:
    """
    1 フレーム内の 1 個体アノテーション。

    v2.0 で ``obb`` (回転矩形) と ``keypoints`` (任意 pose) を追加した。いずれも
    Optional なので、v1.1 の bbox のみのデータもそのまま読み書きできる。

    Attributes:
        bbox: [x1, y1, x2, y2] pixel — **常に存在する**。obb がある場合はその外接矩形。
        obb: [cx, cy, w, h, theta] pixel / rad。None なら回転アノテーションなし。
        obb_source: "annotated" = 人手/モデルによる真の回転矩形、
                    "derived"   = AABB から theta=0 で機械的に導出した代用値。
                    derived は**真の回転アノテーションではない**ため、学習時に
                    角度損失から除外され migration 時に警告される。
        keypoints: [[x, y, v], ...] pixel。v は 0=未ラベル, 1=隠れ, 2=可視。
                   None なら pose アノテーションなし (pose_enabled=False で動作)。
        action_ids: multi-head 行動 (``meta.labeling.behaviors = "multi_head"``)
                   のときだけ使う。行動グループごとに 1 つ、全グループ通しの行動番号
                   (-1 = そのグループは未判断)。``action_id`` は最初のグループの写し。
                   単一行動では空のまま書き出さない。docs/KEYFRAME_BEHAVIOR.md §15。
    """

    object_id: int
    bbox: List[float]  # [x1, y1, x2, y2] pixel coords
    class_id: int = 0
    track_id: int = -1
    action_id: int = -1
    occluded: bool = False
    is_crowd: bool = False
    obb: Optional[List[float]] = None  # [cx, cy, w, h, theta]
    obb_source: str = "none"  # "none" | "annotated" | "derived"
    keypoints: Optional[List[List[float]]] = None  # [[x, y, v], ...]

    source: str = "human"
    score: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    action_ids: List[int] = field(default_factory=list)

    @property
    def has_obb(self) -> bool:
        """真の回転アノテーションを持つか (derived は False)。"""
        return self.obb is not None and self.obb_source == "annotated"

    def get_obb(self) -> List[float]:
        """
        OBB を返す。未設定なら bbox から theta=0 で導出する (安全な fallback)。

        戻り値が真のアノテーションか導出値かは ``has_obb`` で判定すること。
        """
        if self.obb is not None:
            return list(self.obb)
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 0.0), max(y2 - y1, 0.0), 0.0]

    def obb_corners(self) -> List[List[float]]:
        """OBB の 4 頂点 [[x, y], ...] (反時計回り) を返す。GUI 描画用。"""
        cx, cy, w, h, th = self.get_obb()
        hw, hh = w / 2, h / 2
        cos_t, sin_t = math.cos(th), math.sin(th)
        return [
            [cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t]
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]

    def sync_bbox_from_obb(self) -> None:
        """obb の外接矩形で bbox を更新する (OBB 編集後の整合性維持)。"""
        if self.obb is None:
            return
        corners = self.obb_corners()
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        self.bbox = [min(xs), min(ys), max(xs), max(ys)]

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
    """
    データセット全体。

    v2.0 で ``keypoint_names`` (pose) と ``action_groups`` (階層/multi-head 行動定義)
    を追加した。いずれも空でよく、その場合は v1.1 と完全に同じ挙動になる。
    """

    meta: Dict[str, Any] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=lambda: ["fly"])
    action_names: List[str] = field(
        default_factory=lambda: ["idle", "walk", "groom", "court", "other"]
    )
    videos: List[VideoAnno] = field(default_factory=list)
    # v2.0: pose を使う場合のキーポイント名 (空なら pose 無し)
    keypoint_names: List[str] = field(default_factory=list)
    # v2.0: multi-head / hierarchical 行動分類の定義
    #   {"locomotion": ["stationary","walking","running"], "social": [...]}
    action_groups: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return str(self.meta.get("version", "1.1"))

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoint_names)

    def has_obb(self) -> bool:
        """真の回転アノテーションを 1 つでも含むか。"""
        return any(obj.has_obb for v in self.videos for fr in v.frames for obj in fr.objects)

    def obb_coverage(self) -> Tuple[int, int]:
        """(真の OBB を持つ物体数, 総物体数) を返す。"""
        total = annotated = 0
        for v in self.videos:
            for fr in v.frames:
                for obj in fr.objects:
                    total += 1
                    annotated += int(obj.has_obb)
        return annotated, total

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


def parse_obb_field(raw_obb: Any) -> Optional[List[float]]:
    """
    JSON の ``obb`` フィールドを [cx, cy, w, h, theta] に正規化する。

    受け付ける 3 形式 (どれで書かれていても読める):
      1. dict        {"cx":.., "cy":.., "w":.., "h":.., "angle":..}
                     ("theta" / "rotation" も angle の別名として受理)
      2. 5 要素配列   [cx, cy, w, h, theta]
      3. 8 要素配列   [x1,y1, x2,y2, x3,y3, x4,y4]  — 4 頂点表現

    Returns:
        [cx, cy, w, h, theta] または解釈できない場合 None
    """
    if raw_obb is None:
        return None

    if isinstance(raw_obb, dict):
        try:
            angle = raw_obb.get("angle", raw_obb.get("theta", raw_obb.get("rotation", 0.0)))
            return [
                float(raw_obb["cx"]),
                float(raw_obb["cy"]),
                float(raw_obb["w"]),
                float(raw_obb["h"]),
                float(angle),
            ]
        except (KeyError, TypeError, ValueError):
            return None

    if isinstance(raw_obb, (list, tuple)):
        vals = [float(v) for v in raw_obb]
        if len(vals) == 5:
            return vals
        if len(vals) == 8:
            return corners_to_obb_list(vals)
    return None


def corners_to_obb_list(flat: List[float]) -> List[float]:
    """
    4 頂点 [x1,y1,...,x4,y4] → [cx, cy, w, h, theta]。

    幅方向の辺 (頂点0→1) を theta とし、box ローカル軸への射影幅を w/h にする。
    厳密な長方形なら厳密に復元し、歪んだ四角形でも破綻せず妥当な OBB を返す。
    geometry/obb.py の ``corners_to_obb`` と同じ定義 (torch 非依存の純 Python 版)。
    """
    pts = [(flat[i], flat[i + 1]) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    theta = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    loc_x = [(p[0] - cx) * cos_t + (p[1] - cy) * sin_t for p in pts]
    loc_y = [-(p[0] - cx) * sin_t + (p[1] - cy) * cos_t for p in pts]
    w = max(loc_x) - min(loc_x)
    h = max(loc_y) - min(loc_y)
    return [cx, cy, w, h, normalize_theta(theta)]


def normalize_theta(theta: float) -> float:
    """theta を半開区間 [-pi/2, pi/2) に畳む (矩形の周期は pi)。"""
    out = (theta + math.pi / 2) % math.pi - math.pi / 2
    if out >= math.pi / 2 - 1e-6:
        out -= math.pi
    return out


def _parse_object(obj: Dict[str, Any], index: int) -> ObjectAnno:
    """1 個体の dict → ObjectAnno (v1.1 / v2.0 の両方を受理)。"""
    obb = parse_obb_field(obj.get("obb"))

    bbox = obj.get("bbox")
    if bbox is None and obb is not None:
        # OBB のみのアノテーション: 外接矩形を bbox として補う
        cx, cy, w, h, th = obb
        hw, hh = w / 2, h / 2
        cos_t, sin_t = math.cos(th), math.sin(th)
        xs, ys = [], []
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            xs.append(cx + lx * cos_t - ly * sin_t)
            ys.append(cy + lx * sin_t + ly * cos_t)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    elif bbox is None:
        raise ValueError(f"object #{index}: 'bbox' も 'obb' も存在しません")

    kpts = obj.get("keypoints")
    if kpts is not None:
        # [x,y,v,x,y,v,...] のフラット形式 (COCO 互換) も受理する
        if kpts and not isinstance(kpts[0], (list, tuple)):
            kpts = [list(kpts[i : i + 3]) for i in range(0, len(kpts) - 2, 3)]
        kpts = [[float(c) for c in kp] for kp in kpts]

    return ObjectAnno(
        object_id=obj.get("object_id", index),
        bbox=[float(v) for v in bbox],
        class_id=obj.get("class_id", 0),
        track_id=obj.get("track_id", -1),
        action_id=obj.get("action_id", -1),
        occluded=obj.get("occluded", False),
        is_crowd=obj.get("is_crowd", False),
        obb=obb,
        obb_source=obj.get("obb_source", "annotated" if obb is not None else "none"),
        keypoints=kpts,
        source=obj.get("source", "human"),
        score=obj.get("score"),
        provenance=dict(obj.get("provenance", {})),
        action_ids=_parse_action_ids(obj.get("action_ids")),
    )


def _parse_action_ids(raw: Any) -> List[int]:
    """``action_ids`` (multi-head) → ints. Anything unreadable counts as not judged."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            out.append(-1)
    return out


def load_annotation(path: str) -> DatasetAnno:
    """
    JSON アノテーションファイルを読み込み DatasetAnno を返す。

    v1.1 (bbox のみ) と v2.0 (obb / keypoints 付き) の両方を読める。v1.1 のファイルは
    obb=None のまま読み込まれ、``ObjectAnno.get_obb()`` が必要時に theta=0 で導出する。
    """
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
                objects.append(_parse_object(obj, i))
            frames.append(
                FrameAnno(
                    frame_index=fr["frame_index"],
                    image_path=fr.get(
                        "image_path", f"images/{v['video_id']}/{fr['frame_index']:06d}.jpg"
                    ),
                    width=f_width,
                    height=f_height,
                    objects=objects,
                )
            )

        videos.append(
            VideoAnno(
                video_id=v["video_id"],
                fps=v.get("fps", meta.get("fps_default", 25.0)),
                width=v_width,
                height=v_height,
                num_frames=v.get("num_frames", len(frames)),
                video_path=v.get("video_path", ""),
                frames=frames,
            )
        )

    return DatasetAnno(
        meta=meta,
        class_names=class_names,
        action_names=action_names,
        videos=videos,
        keypoint_names=raw.get("keypoint_names", []),
        action_groups=raw.get("action_groups", {}),
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
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
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
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _YOLO_IMAGE_EXTS
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
                objects.append(
                    ObjectAnno(
                        object_id=obj_uid,
                        bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        class_id=int(cid),
                        track_id=-1,
                        action_id=-1,
                    )
                )
            try:
                rel_path = str(img_path.relative_to(root))
            except ValueError:
                rel_path = str(img_path)
            frames.append(
                FrameAnno(
                    frame_index=int(fi),
                    image_path=rel_path.replace("\\", "/"),
                    width=fw,
                    height=fh,
                    objects=objects,
                )
            )
        videos.append(
            VideoAnno(
                video_id=vid,
                fps=fps,
                width=first_w,
                height=first_h,
                num_frames=len(frames),
                frames=frames,
            )
        )

    return DatasetAnno(
        meta={
            "version": "1.1",
            "source_format": "yolo",
            "image_root": (
                str(img_dir.relative_to(root)).replace("\\", "/") if img_dir != root else ""
            ),
        },
        class_names=class_names,
        action_names=list(class_names),
        videos=videos,
    )


def _object_to_dict(obj: ObjectAnno, *, preserve_derived: bool = False) -> Dict[str, Any]:
    """
    ObjectAnno → JSON dict。

    v1.1 のキーは常にそのまま出力し、v2.0 の追加キー (obb / keypoints) は値が
    あるときだけ足す。よって OBB を持たないデータを保存すると v1.1 と
    バイト単位で同じ形になり、旧ツールでも読める。
    ``derived`` な OBB (AABB からの代用値) は情報量ゼロなので書き出さない。
    """
    d: Dict[str, Any] = {
        "object_id": obj.object_id,
        "bbox": obj.bbox,
        "class_id": obj.class_id,
        "track_id": obj.track_id,
        "action_id": obj.action_id,
    }
    if obj.action_ids:
        # multi-head only: one answer per behavior group (docs/KEYFRAME_BEHAVIOR.md §15)
        d["action_ids"] = [int(a) for a in obj.action_ids]
    if obj.occluded:
        d["occluded"] = obj.occluded
    if obj.is_crowd:
        d["is_crowd"] = obj.is_crowd
    if obj.obb is not None and (obj.obb_source == "annotated" or preserve_derived):
        cx, cy, w, h, th = obj.obb
        d["obb"] = {
            "cx": round(float(cx), 3),
            "cy": round(float(cy), 3),
            "w": round(float(w), 3),
            "h": round(float(h), 3),
            "angle": round(float(th), 6),
        }
    if preserve_derived and obj.obb is not None:
        d["obb_source"] = obj.obb_source
    if obj.source != "human":
        d["source"] = obj.source
    if obj.score is not None:
        d["score"] = obj.score
    if obj.provenance:
        d["provenance"] = obj.provenance
    if obj.keypoints:
        d["keypoints"] = [[round(float(c), 2) for c in kp] for kp in obj.keypoints]
    return d


def save_annotation(anno: DatasetAnno, path: str, *, preserve_derived: bool = False) -> None:
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
            _object_to_dict(obj, preserve_derived=preserve_derived) for obj in fr.objects
        ]
        return d

    out: Dict[str, Any] = {
        "meta": anno.meta,
        "class_names": anno.class_names,
        "action_names": anno.action_names,
        "videos": [],
    }
    # v2.0 の任意フィールドは中身があるときだけ書き出す (v1.1 との差分を最小に保つ)
    if anno.keypoint_names:
        out["keypoint_names"] = anno.keypoint_names
    if anno.action_groups:
        out["action_groups"] = anno.action_groups
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

    from ..annotation.storage import atomic_json

    atomic_json(Path(path), out)


# ---------------------------------------------------------------------------
# スキーマ移行 (v1.1 → v2.0)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "2.0"


@dataclass
class MigrationReport:
    """migrate_annotation の結果サマリ。"""

    from_version: str
    to_version: str
    num_objects: int = 0
    num_obb_annotated: int = 0  # 元から真の OBB を持っていた物体
    num_obb_derived: int = 0  # AABB から theta=0 で導出した物体
    warnings: List[str] = field(default_factory=list)

    @property
    def obb_ratio(self) -> float:
        return self.num_obb_annotated / max(self.num_objects, 1)

    def __str__(self) -> str:
        lines = [
            f"Annotation schema {self.from_version} -> {self.to_version}",
            f"  objects:       {self.num_objects}",
            f"  true OBB:      {self.num_obb_annotated} ({self.obb_ratio*100:.1f}%)",
            f"  derived OBB:   {self.num_obb_derived} (theta=0 fallback)",
        ]
        lines.extend(f"  [WARN] {w}" for w in self.warnings)
        return "\n".join(lines)


def migrate_annotation(
    anno: DatasetAnno,
    derive_obb: bool = False,
    keypoint_names: Optional[List[str]] = None,
    action_groups: Optional[Dict[str, List[str]]] = None,
) -> MigrationReport:
    """
    アノテーションを v2.0 スキーマへ **in-place** で移行する。

    v1.1 → v2.0 の実体は「meta.version の更新」と「任意フィールドの追加」だけで、
    既存の bbox / class_id / track_id / action_id は一切変更しない。よって移行は
    常に可逆かつ非破壊で、移行しなくても新コードは v1.1 をそのまま読める
    (この関数は「明示的に v2.0 として保存したい」ときに使う)。

    Args:
        anno: 対象データセット (破壊的に更新される)
        derive_obb: True のとき、OBB を持たない物体に AABB 由来の theta=0 OBB を
            ``obb_source="derived"`` で書き込む。**真の回転アノテーションではない**
            ため学習時に角度損失から除外される。既定 False (必要時のみ明示的に)。
        keypoint_names: pose を使う場合のキーポイント名
        action_groups: multi-head / hierarchical 行動分類の定義

    Returns:
        MigrationReport — 統計と警告

    Example::

        anno = load_annotation("old.json")
        report = migrate_annotation(anno)
        print(report)
        save_annotation(anno, "new_v2.json")
    """
    report = MigrationReport(
        from_version=str(anno.meta.get("version", "1.1")),
        to_version=SCHEMA_VERSION,
    )

    for v in anno.videos:
        for fr in v.frames:
            for obj in fr.objects:
                report.num_objects += 1
                if obj.has_obb:
                    report.num_obb_annotated += 1
                elif derive_obb and obj.obb is None:
                    obj.obb = obj.get_obb()
                    obj.obb_source = "derived"
                    report.num_obb_derived += 1

    if keypoint_names is not None:
        anno.keypoint_names = list(keypoint_names)
    if action_groups is not None:
        anno.action_groups = dict(action_groups)

    anno.meta["version"] = SCHEMA_VERSION
    anno.meta.setdefault("migrated_from", report.from_version)

    if report.num_obb_derived > 0:
        report.warnings.append(
            f"{report.num_obb_derived} 個の物体に theta=0 の代用 OBB を付与しました。"
            "これは真の回転アノテーションではないため、OBB detector の角度損失からは "
            "除外されます (obb_source='derived')。回転性能を出すには実際の回転 "
            "アノテーションが必要です。"
        )
    if report.num_objects > 0 and report.num_obb_annotated == 0:
        report.warnings.append(
            "真の OBB アノテーションが 1 つもありません。detector.type='rtdetr_obb' で "
            "学習しても角度は学習されず、実質 AABB detector として振る舞います。"
        )
    return report


def migrate_annotation_file(
    src: str,
    dst: str,
    derive_obb: bool = False,
    **kwargs: Any,
) -> MigrationReport:
    """ファイル単位で v2.0 へ移行する (load → migrate → save)。"""
    anno = load_annotation(src)
    report = migrate_annotation(anno, derive_obb=derive_obb, **kwargs)
    save_annotation(anno, dst)
    return report


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
    num_keypoints = len(anno.keypoint_names)

    # multi-head behaviors: the groups are the vocabulary (docs/KEYFRAME_BEHAVIOR.md §15).
    from ..annotation import behaviors as _behaviors

    vocab = None
    if _behaviors.behavior_mode_of(anno) == _behaviors.MULTI_HEAD:
        try:
            vocab = _behaviors.vocabulary_of(anno)
        except ValueError as exc:
            result.errors.append(str(exc))
            result.valid = False
        else:
            num_actions = len(vocab)
            if list(anno.action_names) != list(vocab.names):
                result.warnings.append(
                    "action_names does not match action_groups; the groups are used "
                    f"({', '.join(vocab.names)})."
                )

    for v in anno.videos:
        if v.width <= 0 or v.height <= 0:
            result.warnings.append(
                f"Video {v.video_id}: missing/invalid size ({v.width}×{v.height})"
            )

        frame_indices = set()
        for fr in v.frames:
            if fr.frame_index in frame_indices:
                result.errors.append(f"Video {v.video_id}: duplicate frame_index {fr.frame_index}")
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

                if vocab is not None and obj.action_ids:
                    wrong = [
                        a for h, a in enumerate(obj.action_ids)
                        if a != -1 and (h >= len(vocab.groups) or vocab.group_of(a) != h)
                    ]
                    if wrong or len(obj.action_ids) > len(vocab.groups):
                        result.errors.append(
                            f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                            f"action_ids {obj.action_ids} do not fit the behavior groups "
                            f"{list(vocab.heads)} (one answer per group, in group order)"
                        )
                        result.valid = False

                # --- v2.0: OBB ---
                if obj.obb is not None:
                    if len(obj.obb) != 5:
                        result.errors.append(
                            f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                            f"obb must have 5 values [cx,cy,w,h,theta], got {len(obj.obb)}"
                        )
                        result.valid = False
                    else:
                        ow, oh = obj.obb[2], obj.obb[3]
                        if ow <= 0 or oh <= 0:
                            result.errors.append(
                                f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                                f"obb has non-positive size (w={ow}, h={oh})"
                            )
                            result.valid = False
                        if abs(obj.obb[4]) > 2 * math.pi:
                            result.warnings.append(
                                f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                                f"obb angle {obj.obb[4]} looks like degrees, not radians"
                            )

                # --- v2.0: keypoints ---
                if obj.keypoints is not None:
                    if num_keypoints and len(obj.keypoints) != num_keypoints:
                        result.warnings.append(
                            f"Video {v.video_id} Frame {fr.frame_index} Obj {obj.object_id}: "
                            f"{len(obj.keypoints)} keypoints but keypoint_names has "
                            f"{num_keypoints}"
                        )
                    for kp in obj.keypoints:
                        if len(kp) != 3:
                            result.errors.append(
                                f"Video {v.video_id} Frame {fr.frame_index} "
                                f"Obj {obj.object_id}: keypoint must be [x, y, v], got {kp}"
                            )
                            result.valid = False
                            break

    # OBB カバレッジの通知 (エラーではないが学習前に知りたい情報)
    annotated, total = anno.obb_coverage()
    if 0 < annotated < total:
        result.warnings.append(
            f"OBB coverage {annotated}/{total} ({annotated/total*100:.1f}%): "
            "OBB を持たない物体は角度損失から除外されます"
        )

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
    fly_size: int = 20,  # approx bbox size in pixels
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
        action_transition = [[1.0 / num_actions] * num_actions for _ in range(num_actions)]

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
            fly_states.append(
                {
                    "track_id": f_idx + 1,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "action": action,
                }
            )

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

                objects.append(
                    ObjectAnno(
                        object_id=obj_i + 1,
                        bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        class_id=0,
                        track_id=state["track_id"],
                        action_id=state["action"],
                    )
                )

            frames.append(
                FrameAnno(
                    frame_index=fi,
                    image_path=f"images/{video_id}/{fi:06d}.jpg",
                    width=image_width,
                    height=image_height,
                    objects=objects,
                )
            )

        videos.append(
            VideoAnno(
                video_id=video_id,
                fps=fps,
                width=image_width,
                height=image_height,
                num_frames=num_frames_per_video,
                frames=frames,
            )
        )

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
    total_objects = sum(len(fr.objects) for v in anno.videos for fr in v.frames)

    print("=== Annotation Statistics ===")
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
                    name = (
                        anno.action_names[obj.action_id]
                        if obj.action_id < len(anno.action_names)
                        else str(obj.action_id)
                    )
                    action_counter[name] += 1

    if action_counter:
        print("Action distribution:")
        total = sum(action_counter.values())
        for name, cnt in sorted(action_counter.items(), key=lambda x: -x[1]):
            print(f"  {name:15s}: {cnt:5d} ({cnt/total*100:.1f}%)")
