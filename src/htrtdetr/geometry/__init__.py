"""
htrtdetr.geometry — 箱の幾何に関する単一の真実 (single source of truth)

``obb.py`` が torch 実装 (モデル / 学習 / 評価 / GUI)。torch 非依存の段
(``stages/*``) が使う numpy サブセットは ``stages/_geometry.py`` にあり、
``tests/test_geometry_obb.py`` が両者の数値一致を固定している。
"""

from .obb import (  # noqa: F401
    aabb_to_obb,
    canonicalize_obb,
    corners_to_obb,
    cxcywh_to_obb,
    decode_angle,
    encode_angle,
    encoded_to_obb,
    expand_aabb,
    normalize_angle,
    obb_area,
    obb_to_aabb,
    obb_to_context_aabb,
    obb_to_corners,
    obb_to_cxcywh,
    obb_to_encoded,
    probiou,
    rotated_iou,
    rotated_nms,
)

__all__ = [
    "normalize_angle",
    "canonicalize_obb",
    "encode_angle",
    "decode_angle",
    "obb_to_encoded",
    "encoded_to_obb",
    "obb_to_corners",
    "corners_to_obb",
    "obb_to_aabb",
    "aabb_to_obb",
    "cxcywh_to_obb",
    "obb_to_cxcywh",
    "expand_aabb",
    "obb_to_context_aabb",
    "rotated_iou",
    "probiou",
    "rotated_nms",
    "obb_area",
]
