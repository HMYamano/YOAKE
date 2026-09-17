"""
test_annotation_v2.py — アノテーション schema v2.0 と移行のテスト

最重要の検証: **old annotation → new code で読み込める**こと、そして
OBB を使わないデータの保存結果が v1.1 と**バイト等価**であること
(旧ツールが読めなくなる変更は後方互換の破壊そのものなので)。
"""

from __future__ import annotations

import json
import math

import pytest

from htrtdetr.data.annotation import (
    SCHEMA_VERSION,
    DatasetAnno,
    FrameAnno,
    ObjectAnno,
    VideoAnno,
    corners_to_obb_list,
    generate_sample_annotation,
    load_annotation,
    migrate_annotation,
    migrate_annotation_file,
    normalize_theta,
    parse_obb_field,
    save_annotation,
    validate_annotation,
)

_V11 = {
    "meta": {"version": "1.1"},
    "class_names": ["fly"],
    "action_names": ["idle", "walk"],
    "videos": [{
        "video_id": "v1", "fps": 25, "width": 100, "height": 100, "num_frames": 1,
        "frames": [{
            "frame_index": 0, "image_path": "a.jpg", "width": 100, "height": 100,
            "objects": [
                {"object_id": 1, "bbox": [10, 10, 30, 20],
                 "class_id": 0, "track_id": 1, "action_id": 0},
            ],
        }],
    }],
}


def _write(tmp_path, data, name="a.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 後方互換
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_v11_file_loads(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        obj = anno.videos[0].frames[0].objects[0]
        assert obj.bbox == [10, 10, 30, 20]
        assert obj.track_id == 1
        assert obj.obb is None
        assert obj.has_obb is False

    def test_v11_validates(self, tmp_path):
        assert validate_annotation(load_annotation(_write(tmp_path, _V11))).valid

    def test_v11_roundtrip_is_byte_identical(self, tmp_path):
        """
        OBB を持たないデータを保存すると v1.1 と同じ形になる。
        これが「旧ツールでも読める」ことの保証。
        """
        anno = load_annotation(_write(tmp_path, _V11))
        out = str(tmp_path / "out.json")
        save_annotation(anno, out)
        written = json.loads(open(out, encoding="utf-8").read())
        assert (written["videos"][0]["frames"][0]["objects"]
                == _V11["videos"][0]["frames"][0]["objects"])
        assert "obb" not in json.dumps(written)

    def test_get_obb_derives_from_bbox(self, tmp_path):
        """OBB が無くても get_obb() が theta=0 の OBB を返す (安全な fallback)。"""
        obj = load_annotation(_write(tmp_path, _V11)).videos[0].frames[0].objects[0]
        cx, cy, w, h, th = obj.get_obb()
        assert (cx, cy, w, h, th) == (20, 15, 20, 10, 0.0)

    def test_generate_sample_still_works(self):
        anno = generate_sample_annotation(num_videos=1, num_frames_per_video=3,
                                          num_flies=2)
        assert validate_annotation(anno).valid
        assert not anno.has_obb()


# ---------------------------------------------------------------------------
# OBB のパース (3 形式)
# ---------------------------------------------------------------------------

class TestOBBParsing:
    EXPECTED = [50.0, 50.0, 40.0, 10.0, 0.5]

    def test_dict_form(self):
        got = parse_obb_field({"cx": 50, "cy": 50, "w": 40, "h": 10, "angle": 0.5})
        assert got == pytest.approx(self.EXPECTED)

    def test_dict_theta_alias(self):
        got = parse_obb_field({"cx": 50, "cy": 50, "w": 40, "h": 10, "theta": 0.5})
        assert got == pytest.approx(self.EXPECTED)

    def test_five_element_list(self):
        assert parse_obb_field([50, 50, 40, 10, 0.5]) == pytest.approx(self.EXPECTED)

    def test_four_corner_form(self):
        th, hw, hh = 0.5, 20.0, 5.0
        c, s = math.cos(th), math.sin(th)
        flat = []
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            flat += [50 + lx * c - ly * s, 50 + lx * s + ly * c]
        assert parse_obb_field(flat) == pytest.approx(self.EXPECTED, abs=1e-4)

    def test_none_and_garbage(self):
        assert parse_obb_field(None) is None
        assert parse_obb_field({"nope": 1}) is None
        assert parse_obb_field([1, 2, 3]) is None

    def test_corners_to_obb_list_direct(self):
        assert corners_to_obb_list([0, 0, 10, 0, 10, 5, 0, 5]) == pytest.approx(
            [5.0, 2.5, 10.0, 5.0, 0.0]
        )

    def test_normalize_theta_range(self):
        for t in (-10.0, -1.6, 0.0, 1.6, 10.0):
            out = normalize_theta(t)
            assert -math.pi / 2 - 1e-6 <= out < math.pi / 2


# ---------------------------------------------------------------------------
# v2.0 の読み書き
# ---------------------------------------------------------------------------

class TestV2Roundtrip:
    def _v2(self):
        data = json.loads(json.dumps(_V11))
        data["meta"]["version"] = "2.0"
        data["videos"][0]["frames"][0]["objects"][0]["obb"] = {
            "cx": 20, "cy": 15, "w": 22, "h": 9, "angle": 0.3,
        }
        data["videos"][0]["frames"][0]["objects"][0]["keypoints"] = [
            [12, 12, 2], [28, 18, 2],
        ]
        data["keypoint_names"] = ["head", "abdomen"]
        data["action_groups"] = {"locomotion": ["stationary", "walking"]}
        return data

    def test_obb_and_keypoints_load(self, tmp_path):
        anno = load_annotation(_write(tmp_path, self._v2()))
        obj = anno.videos[0].frames[0].objects[0]
        assert obj.has_obb
        assert obj.obb == pytest.approx([20, 15, 22, 9, 0.3])
        assert obj.keypoints == [[12, 12, 2], [28, 18, 2]]
        assert anno.keypoint_names == ["head", "abdomen"]
        assert anno.action_groups == {"locomotion": ["stationary", "walking"]}

    def test_save_load_roundtrip(self, tmp_path):
        anno = load_annotation(_write(tmp_path, self._v2()))
        out = str(tmp_path / "out.json")
        save_annotation(anno, out)
        back = load_annotation(out)
        o1 = anno.videos[0].frames[0].objects[0]
        o2 = back.videos[0].frames[0].objects[0]
        assert o2.obb == pytest.approx(o1.obb, abs=1e-3)
        assert o2.keypoints == o1.keypoints
        assert back.keypoint_names == anno.keypoint_names

    def test_coco_style_flat_keypoints(self, tmp_path):
        data = self._v2()
        data["videos"][0]["frames"][0]["objects"][0]["keypoints"] = [12, 12, 2, 28, 18, 2]
        obj = load_annotation(_write(tmp_path, data)).videos[0].frames[0].objects[0]
        assert obj.keypoints == [[12, 12, 2], [28, 18, 2]]

    def test_obb_only_object_gets_bbox(self, tmp_path):
        """bbox が無く obb だけの物体は外接矩形で bbox が補われる。"""
        data = json.loads(json.dumps(_V11))
        data["videos"][0]["frames"][0]["objects"] = [{
            "object_id": 1, "class_id": 0, "track_id": 1, "action_id": 0,
            "obb": {"cx": 50, "cy": 50, "w": 10, "h": 10, "angle": 0.0},
        }]
        obj = load_annotation(_write(tmp_path, data)).videos[0].frames[0].objects[0]
        assert obj.bbox == pytest.approx([45, 45, 55, 55])

    def test_object_without_bbox_or_obb_raises(self, tmp_path):
        data = json.loads(json.dumps(_V11))
        data["videos"][0]["frames"][0]["objects"] = [{"object_id": 1, "class_id": 0}]
        with pytest.raises(ValueError, match="bbox"):
            load_annotation(_write(tmp_path, data))

    def test_sync_bbox_from_obb(self):
        obj = ObjectAnno(object_id=1, bbox=[0, 0, 10, 10],
                         obb=[5, 5, 10, 10, math.pi / 4], obb_source="annotated")
        obj.sync_bbox_from_obb()
        half = 10 * math.sqrt(2) / 2
        assert obj.bbox == pytest.approx([5 - half, 5 - half, 5 + half, 5 + half], abs=1e-3)

    def test_obb_corners(self):
        obj = ObjectAnno(object_id=1, bbox=[0, 0, 10, 4],
                         obb=[5, 2, 10, 4, 0.0], obb_source="annotated")
        corners = obj.obb_corners()
        assert len(corners) == 4
        assert corners[0] == pytest.approx([0.0, 0.0])
        assert corners[2] == pytest.approx([10.0, 4.0])


# ---------------------------------------------------------------------------
# 移行
# ---------------------------------------------------------------------------

class TestMigration:
    def test_version_bumped(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        report = migrate_annotation(anno)
        assert anno.meta["version"] == SCHEMA_VERSION
        assert report.from_version == "1.1"
        assert report.num_objects == 1

    def test_labels_are_untouched(self, tmp_path):
        """移行で bbox / track_id / action_id は一切変わらない。"""
        anno = load_annotation(_write(tmp_path, _V11))
        before = [
            (o.bbox[:], o.track_id, o.action_id, o.class_id)
            for v in anno.videos for f in v.frames for o in f.objects
        ]
        migrate_annotation(anno)
        after = [
            (o.bbox[:], o.track_id, o.action_id, o.class_id)
            for v in anno.videos for f in v.frames for o in f.objects
        ]
        assert before == after

    def test_derive_obb_off_by_default(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        report = migrate_annotation(anno)
        assert report.num_obb_derived == 0
        assert anno.videos[0].frames[0].objects[0].obb is None

    def test_derive_obb_marks_source_and_warns(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        report = migrate_annotation(anno, derive_obb=True)
        obj = anno.videos[0].frames[0].objects[0]
        assert obj.obb is not None
        assert obj.obb_source == "derived"
        assert obj.has_obb is False, "derived は真の OBB として扱わない"
        assert report.num_obb_derived == 1
        assert any("真の回転" in w for w in report.warnings)

    def test_derived_obb_is_not_saved(self, tmp_path):
        """derived な OBB は情報量ゼロなので書き出さない。"""
        anno = load_annotation(_write(tmp_path, _V11))
        migrate_annotation(anno, derive_obb=True)
        out = str(tmp_path / "out.json")
        save_annotation(anno, out)
        assert "obb" not in open(out, encoding="utf-8").read()

    def test_warns_when_no_true_obb(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        report = migrate_annotation(anno)
        assert any("1 つもありません" in w for w in report.warnings)

    def test_file_level_migration(self, tmp_path):
        src = _write(tmp_path, _V11)
        dst = str(tmp_path / "v2.json")
        report = migrate_annotation_file(src, dst, derive_obb=False)
        assert report.to_version == SCHEMA_VERSION
        back = load_annotation(dst)
        assert back.schema_version == SCHEMA_VERSION

    def test_can_add_keypoint_names_and_groups(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        migrate_annotation(anno, keypoint_names=["head", "tail"],
                           action_groups={"social": ["none", "court"]})
        assert anno.keypoint_names == ["head", "tail"]
        assert anno.num_keypoints == 2
        assert anno.action_groups["social"] == ["none", "court"]

    def test_report_str(self, tmp_path):
        anno = load_annotation(_write(tmp_path, _V11))
        s = str(migrate_annotation(anno, derive_obb=True))
        assert "2.0" in s and "derived OBB" in s


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------

class TestValidation:
    def _anno_with(self, **obj_kw):
        obj = ObjectAnno(object_id=1, bbox=[10, 10, 30, 20], **obj_kw)
        return DatasetAnno(
            meta={"version": "2.0"}, class_names=["fly"], action_names=["idle"],
            videos=[VideoAnno(video_id="v", width=100, height=100, frames=[
                FrameAnno(frame_index=0, image_path="a.jpg",
                          width=100, height=100, objects=[obj]),
            ])],
        )

    def test_bad_obb_length(self):
        r = validate_annotation(self._anno_with(obb=[1, 2, 3], obb_source="annotated"))
        assert not r.valid
        assert any("5 values" in e for e in r.errors)

    def test_non_positive_obb_size(self):
        r = validate_annotation(
            self._anno_with(obb=[20, 15, 0, 10, 0.1], obb_source="annotated")
        )
        assert not r.valid

    def test_degrees_instead_of_radians_warns(self):
        r = validate_annotation(
            self._anno_with(obb=[20, 15, 20, 10, 45.0], obb_source="annotated")
        )
        assert any("degrees" in w for w in r.warnings)

    def test_bad_keypoint_shape(self):
        r = validate_annotation(self._anno_with(keypoints=[[1, 2]]))
        assert not r.valid

    def test_partial_obb_coverage_warns(self):
        anno = self._anno_with(obb=[20, 15, 20, 10, 0.1], obb_source="annotated")
        anno.videos[0].frames[0].objects.append(
            ObjectAnno(object_id=2, bbox=[40, 40, 60, 60])
        )
        r = validate_annotation(anno)
        assert r.valid
        assert any("coverage" in w for w in r.warnings)

    def test_obb_coverage_counts(self):
        anno = self._anno_with(obb=[20, 15, 20, 10, 0.1], obb_source="annotated")
        anno.videos[0].frames[0].objects.append(
            ObjectAnno(object_id=2, bbox=[40, 40, 60, 60])
        )
        assert anno.obb_coverage() == (1, 2)
        assert anno.has_obb() is True
