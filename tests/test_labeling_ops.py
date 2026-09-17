"""
test_labeling_ops.py — bbox 補間ユーティリティと JSON ラウンドトリップ

dearpygui / torch 非依存で実行できる (純関数 + data/annotation.py の I/O)。
AnnObject の dict 変換テストのみ dearpygui が必要なので importorskip でガードする。
"""

from __future__ import annotations

import pytest

from htrtdetr.gui.labeling_ops import (
    clamp_bbox,
    interpolate_bbox,
    interpolate_between,
    round_bbox,
)


# ─── interpolate_bbox / interpolate_between ────────────────────────────────

def test_interpolate_bbox_endpoints_and_midpoint():
    a = [0, 0, 10, 10]
    b = [100, 100, 110, 110]
    assert interpolate_bbox(a, b, 0.0) == [0, 0, 10, 10]
    assert interpolate_bbox(a, b, 1.0) == [100, 100, 110, 110]
    assert interpolate_bbox(a, b, 0.5) == [50, 50, 60, 60]  # 両端の平均


def test_interpolate_between_covers_all_frames_inclusive():
    pairs = interpolate_between(0, [0, 0, 10, 10], 4, [40, 40, 50, 50])
    frames = [f for f, _ in pairs]
    assert frames == [0, 1, 2, 3, 4]
    assert pairs[0][1] == [0, 0, 10, 10]      # 端点は厳密
    assert pairs[-1][1] == [40, 40, 50, 50]
    assert pairs[2][1] == [20, 20, 30, 30]    # 中点


def test_interpolate_between_reversed_and_single():
    # 逆順でも自動で入れ替わる
    fwd = interpolate_between(0, [0, 0, 10, 10], 4, [40, 40, 50, 50])
    rev = interpolate_between(4, [40, 40, 50, 50], 0, [0, 0, 10, 10])
    assert fwd == rev
    # 単一フレーム
    single = interpolate_between(3, [1, 2, 3, 4], 3, [1, 2, 3, 4])
    assert single == [(3, [1, 2, 3, 4])]


# ─── clamp / round ─────────────────────────────────────────────────────────

def test_clamp_and_round_bbox():
    clamped = clamp_bbox([-5, 3, 700, 20], width=640, height=480)
    assert clamped == [0.0, 3.0, 640.0, 20.0]
    # 順序が逆でも正規化
    assert clamp_bbox([50, 60, 10, 20], 640, 480) == [10.0, 20.0, 50.0, 60.0]
    assert round_bbox([1.4, 2.6, 3.5, 4.49]) == [1, 3, 4, 4]


# ─── v1.1 JSON ラウンドトリップ (補間 box を保存 → 読込 → 検証) ────────────

def test_interpolated_boxes_json_roundtrip(tmp_path):
    from htrtdetr.data.annotation import (
        DatasetAnno,
        FrameAnno,
        ObjectAnno,
        VideoAnno,
        load_annotation,
        save_annotation,
        validate_annotation,
    )

    W = H = 640
    boxes = dict(interpolate_between(0, [10, 10, 30, 30], 10, [110, 110, 130, 130]))

    frames = []
    for f in range(0, 11):
        bbox = round_bbox(clamp_bbox(boxes[f], W, H))
        obj = ObjectAnno(object_id=f + 1, bbox=bbox, class_id=0,
                         track_id=1, action_id=1)  # action 'walk'
        frames.append(FrameAnno(frame_index=f, image_path=f"images/vid/{f:06d}.jpg",
                                objects=[obj]))
    anno = DatasetAnno(
        meta={"version": "1.1", "fps_default": 25.0, "image_root": ""},
        class_names=["fly"],
        action_names=["idle", "walk", "groom", "court", "other"],
        videos=[VideoAnno(video_id="vid", fps=25.0, width=W, height=H,
                          num_frames=11, frames=frames)],
    )

    out = tmp_path / "annotations.json"
    save_annotation(anno, str(out))
    loaded = load_annotation(str(out))

    result = validate_annotation(loaded)
    assert result.valid, str(result)
    assert not result.errors

    lf = loaded.videos[0].frames
    assert len(lf) == 11
    # 中点フレーム (f=5) は両端の平均 → [60,60,80,80]
    mid = lf[5].objects[0].bbox
    assert mid == [60, 60, 80, 80]
    # 全フレームで action/track が保たれている
    assert all(fr.objects[0].track_id == 1 for fr in lf)
    assert all(fr.objects[0].action_id == 1 for fr in lf)


# ─── AnnObject の dict 変換 (keyframe は保存しない) ── dearpygui 必要 ───────

def test_annobject_dict_roundtrip_excludes_keyframe():
    pytest.importorskip("dearpygui")
    pytest.importorskip("cv2")
    from htrtdetr.annotator import AnnObject

    obj = AnnObject(bbox=[1, 2, 3, 4], track_id=5, action_id=2, keyframe=False)
    d = obj.to_dict()
    assert "keyframe" not in d          # v1.1 スキーマを汚さない
    assert d["bbox"] == [1, 2, 3, 4]
    assert d["track_id"] == 5 and d["action_id"] == 2 and d["class_id"] == 0

    restored = AnnObject.from_dict(d)
    assert restored.keyframe is True     # 読み込んだ box はキーフレーム扱い
    assert restored.bbox == [1, 2, 3, 4]
