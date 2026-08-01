from importlib.util import find_spec


_TORCH_DATA_EXPORTS = [
    "ObjectAnnotation",
    "FrameAnnotation",
    "VideoAnnotation",
    "load_annotations",
    "validate_image_paths",
    "SingleFrameDataset",
    "SlidingWindowDataset",
    "DummyDataset",
    "create_annotation_from_mot",
    "collate_single_frame",
    "collate_sequence",
    "collate_sequence_padded",
    "get_collate_fn",
]

# torch 依存の dataset/collate は、torch が無い環境 (GUI・ラベリング・テストの
# 純ロジック) では import しない。torch がある環境では ImportError を握りつぶさず、
# dataset/collate 側の実装不備がすぐ見えるようにする。
_TORCH_DATA = find_spec("torch") is not None
if _TORCH_DATA:
    from .dataset import (
        ObjectAnnotation,
        FrameAnnotation,
        VideoAnnotation,
        load_annotations,
        validate_image_paths,
        SingleFrameDataset,
        SlidingWindowDataset,
        DummyDataset,
        create_annotation_from_mot,
    )
    from .collate import (
        collate_single_frame,
        collate_sequence,
        collate_sequence_padded,
        get_collate_fn,
    )

# v1.1 アノテーション I/O は純 Python (torch 非依存)。常にエクスポートする。
from .annotation import (
    DatasetAnno,
    VideoAnno,
    FrameAnno,
    ObjectAnno,
    ValidationResult,
    load_annotation,
    load_yolo_annotation,
    save_annotation,
    validate_annotation,
    print_annotation_stats,
    generate_sample_annotation,
)

_PURE_EXPORTS = [
    # torch 非依存 (v1.1 I/O)
    "DatasetAnno",
    "VideoAnno",
    "FrameAnno",
    "ObjectAnno",
    "ValidationResult",
    "load_annotation",
    "load_yolo_annotation",
    "save_annotation",
    "validate_annotation",
    "print_annotation_stats",
    "generate_sample_annotation",
]

__all__ = [*_PURE_EXPORTS, *(_TORCH_DATA_EXPORTS if _TORCH_DATA else [])]
