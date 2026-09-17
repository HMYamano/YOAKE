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


def __getattr__(name):
    # Importing annotation storage must not load PyTorch in the GUI process.
    if name not in _TORCH_DATA_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(
        ".collate" if name.startswith("collate_") or name == "get_collate_fn" else ".dataset",
        __name__,
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


# v1.1 アノテーション I/O は純 Python (torch 非依存)。常にエクスポートする。
from .annotation import (
    DatasetAnno,
    FrameAnno,
    ObjectAnno,
    ValidationResult,
    VideoAnno,
    generate_sample_annotation,
    load_annotation,
    load_yolo_annotation,
    print_annotation_stats,
    save_annotation,
    validate_annotation,
)

__all__ = [
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

if _TORCH_DATA:
    __all__ += _TORCH_DATA_EXPORTS
