# Dataset Format — YOAKE Annotation Specification v1.1

This document describes the annotation format used by YOAKE, explains how to prepare your own data, and documents the CSV input format supported by the data preparation tools.

---

## Table of Contents

1. [JSON Annotation Format](#1-json-annotation-format)
2. [Field Reference](#2-field-reference)
3. [Coordinate System](#3-coordinate-system)
4. [Action Class Conventions](#4-action-class-conventions)
5. [CSV Input Format](#5-csv-input-format)
6. [Preparing Your Own Data — Step by Step](#6-preparing-your-own-data--step-by-step)
7. [Data Quality Guidelines](#7-data-quality-guidelines)

---

## 1. JSON Annotation Format

Annotations are stored as UTF-8 encoded JSON files. One file covers an entire split (train, val, or test) and may contain multiple videos. The loader is `src/htrtdetr/data/annotation.py::load_annotation()`.

### Complete Example

```json
{
  "meta": {
    "version": "1.1",
    "description": "Drosophila melanogaster courtship behavior dataset",
    "created": "2026-01-15",
    "fps_default": 25.0,
    "image_root": "images/"
  },
  "class_names": ["fly"],
  "action_names": ["idle", "walk", "groom", "court", "other"],
  "videos": [
    {
      "video_id": "vid_001",
      "video_path": "videos/vid_001.mp4",
      "fps": 25.0,
      "width": 1024,
      "height": 1024,
      "num_frames": 300,
      "frames": [
        {
          "frame_index": 0,
          "image_path": "images/vid_001/000000.jpg",
          "objects": [
            {
              "object_id": 1,
              "bbox": [112.0, 87.5, 132.0, 109.0],
              "class_id": 0,
              "track_id": 1,
              "action_id": 1
            },
            {
              "object_id": 2,
              "bbox": [450.3, 210.0, 468.7, 230.5],
              "class_id": 0,
              "track_id": 2,
              "action_id": 0
            }
          ]
        },
        {
          "frame_index": 1,
          "image_path": "images/vid_001/000001.jpg",
          "objects": [
            {
              "object_id": 1,
              "bbox": [113.5, 88.0, 133.5, 110.0],
              "class_id": 0,
              "track_id": 1,
              "action_id": 1,
              "occluded": false
            },
            {
              "object_id": 2,
              "bbox": [451.0, 211.2, 469.5, 231.8],
              "class_id": 0,
              "track_id": 2,
              "action_id": -1
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 2. Field Reference

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `meta` | object | Yes | Dataset metadata (see below) |
| `class_names` | string[] | Yes | Ordered list mapping `class_id` integers to names. Index 0 = first class. |
| `action_names` | string[] | Yes | Ordered list mapping `action_id` integers to names. Index 0 = first action. |
| `videos` | object[] | Yes | List of video annotation objects |

### `meta` fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Annotation format version. Must be `"1.1"` for this spec. |
| `description` | string | No | Free-text description of the dataset |
| `created` | string | No | Creation date in `YYYY-MM-DD` format |
| `fps_default` | float | No | Default FPS used when a video omits its `fps` field |
| `image_root` | string | No | Base directory for resolving relative `image_path` values |

### `videos[*]` fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_id` | string | Yes | Unique identifier for this video within the dataset |
| `fps` | float | Yes | Frames per second of the source video |
| `width` | int | Yes | Frame width in pixels |
| `height` | int | Yes | Frame height in pixels |
| `num_frames` | int | Yes | Total number of annotated frames in this video |
| `video_path` | string | No | Relative path to the source MP4 file |
| `frames` | object[] | Yes | Per-frame annotation list |

### `frames[*]` fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frame_index` | int | Yes | Zero-based frame number. Must be unique within a video. |
| `image_path` | string | Yes | Relative path to the extracted JPEG/PNG frame image |
| `width` | int | No | Frame width (if different from video-level width) |
| `height` | int | No | Frame height (if different from video-level height) |
| `objects` | object[] | Yes | List of annotated objects in this frame (may be empty) |

### `objects[*]` fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `object_id` | int | Yes | Per-frame unique integer for this detection (1-indexed by convention) |
| `bbox` | float[4] | Yes | Bounding box in pixel coordinates — see Coordinate System below |
| `class_id` | int | Yes | Index into `class_names`. For Drosophila data, always `0`. |
| `track_id` | int | Yes | Identity persistent across frames within the same video. Use `-1` for untracked objects. |
| `action_id` | int | Yes | Index into `action_names`. Use `-1` to indicate an unannotated frame. |
| `occluded` | bool | No | Whether the animal is partially occluded (default: `false`) |
| `is_crowd` | bool | No | Whether this annotation represents a crowd of overlapping animals (default: `false`). Crowd annotations are excluded from loss computation. |

---

## 3. Coordinate System

All bounding boxes use the **xyxy pixel coordinate system**:

```
bbox = [x1, y1, x2, y2]
```

- `x1`, `y1`: pixel coordinates of the **top-left corner** of the bounding box
- `x2`, `y2`: pixel coordinates of the **bottom-right corner** of the bounding box
- `x` increases to the right (width direction)
- `y` increases downward (height direction)
- Coordinates are in **absolute pixel units**, not normalized

**Valid constraints:**

```
0 <= x1 < x2 <= frame_width
0 <= y1 < y2 <= frame_height
```

The annotation validator in `src/htrtdetr/data/annotation.py::validate_annotation()` will report errors for boxes that violate `x2 <= x1` or `y2 <= y1`, and warnings for boxes extending outside the frame boundaries.

Internally, the model converts bounding boxes to normalized `cxcywh` format before computing losses. This conversion is handled automatically by the dataloader.

---

## 4. Action Class Conventions

Action labels are **0-indexed integers** referencing the position in the `action_names` list. The canonical five-class vocabulary used in the default configuration is:

| `action_id` | Canonical name | Typical behavioral description |
|-------------|----------------|--------------------------------|
| 0 | `idle` / `stationary` | Animal is stationary; no directed movement |
| 1 | `walk` / `walking` | Directed locomotion across the arena |
| 2 | `groom` / `grooming` | Self-grooming: cleaning legs, wings, abdomen, or head |
| 3 | `court` / `courtship` | Courtship display: wing extension, chasing, singing |
| 4 | `other` / `aggression` | Aggressive contact, unclassified interaction, or ambiguous behavior |

**Special value `-1`**: Indicates that the frame was not annotated for action. Loss functions use `ignore_index=-1` to exclude these instances from the action classification loss. Partial annotation (where only some frames carry action labels) is fully supported.

You may use a different vocabulary by updating `action_names` in your annotation JSON and setting `model.action_head.num_actions` in your YAML config accordingly.

---

## 5. CSV Input Format

If your existing annotations are in spreadsheet or CSV form, the tool `tools/convert_annotations.py` (when available) converts them to the JSON v1.1 format. The expected CSV columns are:

```
video_id, frame_idx, track_id, x1, y1, x2, y2, class_id, action_id
```

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `video_id` | string | Yes | Matches the video folder or file stem |
| `frame_idx` | int | Yes | Zero-based frame index |
| `track_id` | int | Yes | Persistent identity across frames; use `-1` if unknown |
| `x1` | float | Yes | Left edge of bounding box in pixels |
| `y1` | float | Yes | Top edge of bounding box in pixels |
| `x2` | float | Yes | Right edge of bounding box in pixels |
| `y2` | float | Yes | Bottom edge of bounding box in pixels |
| `class_id` | int | No | Animal class index (defaults to `0` if omitted) |
| `action_id` | int | No | Action label index (defaults to `-1` if omitted) |

Example CSV rows:

```csv
video_id,frame_idx,track_id,x1,y1,x2,y2,class_id,action_id
vid_001,0,1,112.0,87.5,132.0,109.0,0,1
vid_001,0,2,450.3,210.0,468.7,230.5,0,0
vid_001,1,1,113.5,88.0,133.5,110.0,0,1
vid_001,1,2,451.0,211.2,469.5,231.8,0,-1
```

The converter groups rows by `video_id` and `frame_idx` automatically. Video-level metadata (fps, width, height) can be supplied in a separate sidecar CSV or as command-line arguments.

---

## 6. Preparing Your Own Data — Step by Step

### Step 1: Extract frames from source video

Use `ffmpeg` or any video reader to extract frames as JPEG files. The naming convention expected by the dataloader is zero-padded six-digit indices:

```bash
mkdir -p data/my_dataset/images/vid_001
ffmpeg -i raw_videos/vid_001.mp4 \
    -q:v 2 \
    data/my_dataset/images/vid_001/%06d.jpg
```

Verify frame count:

```bash
ls data/my_dataset/images/vid_001/ | wc -l
```

### Step 2: Annotate bounding boxes, track IDs, and action labels

Use a tracking annotation tool (e.g., CVAT, VGG Image Annotator, or a custom tool) to produce per-frame bounding boxes with persistent track IDs. If the tool exports in a different format (MOT, COCO, CSV), convert to the YOAKE JSON format using the CSV pipeline or a custom converter.

For quick prototyping, you can generate synthetic annotations with the built-in generator:

```python
from htrtdetr.data.annotation import generate_sample_annotation, save_annotation

anno = generate_sample_annotation(
    num_videos=10,
    num_frames_per_video=300,
    num_flies=5,
    image_width=1024,
    image_height=1024,
    fps=25.0,
)
save_annotation(anno, "data/sample/annotations.json")
```

### Step 3: Validate your annotations

```python
from htrtdetr.data.annotation import load_annotation, validate_annotation

anno = load_annotation("data/my_dataset/annotations.json")
result = validate_annotation(anno)
print(result)
```

Fix any errors reported before proceeding. Warnings (e.g., bounding boxes slightly out of frame) are informational.

### Step 4: Split into train / val / test

Splits should be done at the **video level** to avoid temporal leakage. A recommended split is 70% train / 15% val / 15% test. Each split is saved as a separate JSON file:

```
data/
  train/annotations.json
  val/annotations.json
  test/annotations.json
```

### Step 5: Summarize and inspect

Print dataset statistics to verify class balance and track length distributions before training:

```python
from htrtdetr.data.annotation import load_annotation, print_annotation_stats

anno = load_annotation("data/train/annotations.json")
print_annotation_stats(anno)
```

---

## 7. Data Quality Guidelines

Following these guidelines will improve training stability and metric reliability.

### Annotation density

- **Recommended**: Annotate every frame (100% density). The model uses 16-frame temporal windows; large gaps in annotations create discontinuous sequences.
- **Minimum acceptable**: No more than one unannotated frame per 5 frames in any video. Use `action_id=-1` for unannotated action labels rather than skipping the frame entirely.

### Minimum track length

- Tracks shorter than 16 frames will not contribute to temporal module training because they are shorter than one full window. Aim for an average track length of at least 50 frames.
- During Stage 3 (ID head training), the trainer reindexes track IDs per-video. Videos with fewer than 2 unique tracks are skipped automatically.

### Bounding box quality

- Bounding boxes should tightly enclose the animal body. Loose boxes (large empty margins) increase false-negative IoU overlap and hurt detection AP.
- For occluded animals, annotate the **visible portion** of the body only and set `"occluded": true`.
- Avoid crowd (`is_crowd: true`) annotations except for genuinely inseparable pile-ups; crowd detections are excluded from both loss and evaluation.

### Class and action balance

- The action classification loss uses focal loss (`focal_gamma=2.0`) to down-weight easy/frequent classes. However, extremely imbalanced action distributions (>20:1 ratio) can still hurt minority-class F1. Consider augmenting underrepresented behaviors or adjusting focal alpha per class.
- A useful check: run `python tools/analyze.py mode=distribution annotation=data/train/annotations.json` and inspect `outputs/analysis/action_distribution.png`.

### Video-level metadata

- Always set `width` and `height` at the video level. Per-frame overrides are only needed if your video has changing resolution (rare).
- Set `fps` to the actual capture rate, not the display rate. This affects temporal window calculations.
- Consistent `fps` across all videos in a split is assumed. If your dataset mixes frame rates, normalize windows by temporal duration rather than frame count (requires config changes).

### Image file format

- JPEG is recommended for storage efficiency. Use quality setting ≥ 90 (`-q:v 2` in ffmpeg) to avoid compression artifacts affecting small animal textures.
- PNG is acceptable but increases disk usage significantly for 1024×1024 frame sequences.
- All images in a dataset should have the same resolution. Mixed resolutions require padding/resizing pre-processing before loading.
