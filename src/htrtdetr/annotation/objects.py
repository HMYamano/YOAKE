"""Editable annotation objects, independent of the GUI runtime."""

from __future__ import annotations

import math

_gid = 0


class AnnObject:
    """
    1 個体のアノテーション。

    v2.0 で ``obb`` (回転矩形) を任意フィールドとして追加した。``obb=None`` なら
    従来どおり軸平行 bbox のみのオブジェクトとして扱われ、保存 JSON も v1.1 と
    同じ形になる (旧ツールで読める)。

    ``bbox`` は obb があっても**常に**その外接矩形として保たれる。これにより
    bbox しか見ない既存コード (伝播 / エクスポート) が
    OBB の有無に関係なく動く。
    """

    def __init__(
        self,
        bbox: list[int],
        track_id: int = -1,
        action_id: int = -1,
        occluded: bool = False,
        is_crowd: bool = False,
        keyframe: bool = True,
        obb: list[float] | None = None,
        source: str = "human",
        score: float | None = None,
        class_id: int = 0,
        obb_source: str | None = None,
        provenance: dict | None = None,
        keypoints=None,
    ) -> None:
        global _gid
        _gid += 1
        self.object_id = _gid
        self.bbox = list(bbox)
        self.class_id = int(class_id)
        self.obb_source = obb_source or ("annotated" if obb is not None else "none")
        self.provenance = dict(provenance or {})
        self.keypoints = keypoints
        self.track_id = track_id
        self.action_id = action_id
        self.occluded = occluded
        self.is_crowd = is_crowd
        # keyframe: True = 手動で置いた box (補間のアンカー)、False = 補間/自動生成 box。
        # 編集用フラグで、保存 JSON には出力しない。
        self.keyframe = keyframe
        # obb: [cx, cy, w, h, theta] or None (= 軸平行のみ)
        self.obb = list(obb) if obb is not None else None
        # provenance: どこから来た box か ("human" / "wand" / "track" / "auto")。
        # 「最後に誰が触ったか」ではないので、人が編集しても変えない。
        self.source = source or "human"
        # 機械が付けた確信度。人が描いた box には無い (None)。
        self.score = None if score is None else float(score)

    # ---- OBB ヘルパー -------------------------------------------------

    @property
    def has_obb(self) -> bool:
        return self.obb is not None

    def get_obb(self) -> list[float]:
        """OBB を返す。未設定なら bbox から theta=0 で導出する。"""
        if self.obb is not None:
            return list(self.obb)
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 0), max(y2 - y1, 0), 0.0]

    def corners(self) -> list[tuple[float, float]]:
        """OBB の 4 頂点 (反時計回り)。描画とハンドル判定に使う。"""
        cx, cy, w, h, th = self.get_obb()
        hw, hh = w / 2, h / 2
        c, s = math.cos(th), math.sin(th)
        return [
            (cx + lx * c - ly * s, cy + lx * s + ly * c)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]

    def sync_bbox(self) -> None:
        """obb の外接矩形で bbox を更新する (OBB 編集後に必ず呼ぶ)。"""
        if self.obb is None:
            return
        pts = self.corners()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    def make_obb(self) -> None:
        """軸平行 box を theta=0 の OBB に昇格させる (回転編集を始めるとき)。"""
        if self.obb is None:
            self.obb = self.get_obb()
            self.obb_source = "annotated"

    def drop_obb(self) -> None:
        """OBB を捨てて軸平行 box に戻す (外接矩形を bbox として残す)。"""
        if self.obb is not None:
            self.sync_bbox()
            self.obb = None
            self.obb_source = "none"

    # ---- 直列化 -------------------------------------------------------

    def to_dict(self) -> dict:
        from ..data.annotation import _object_to_dict

        record = _object_to_dict(self.to_annotation(), preserve_derived=True)
        record.update(occluded=self.occluded, is_crowd=self.is_crowd)
        return record

    def to_annotation(self):
        """The GUI adapter: one canonical serializer owns the file format."""
        from ..data.annotation import ObjectAnno

        return ObjectAnno(
            object_id=self.object_id,
            bbox=list(self.bbox),
            class_id=self.class_id,
            track_id=self.track_id,
            action_id=self.action_id,
            occluded=self.occluded,
            is_crowd=self.is_crowd,
            obb=self.obb,
            obb_source=self.obb_source,
            keypoints=self.keypoints,
            source=self.source,
            score=self.score,
            provenance=dict(self.provenance),
        )

    @staticmethod
    def from_dict(d: dict) -> AnnObject:
        global _gid
        obj = AnnObject(
            bbox=d["bbox"],
            class_id=d.get("class_id", 0),
            obb_source=d.get("obb_source"),
            provenance=d.get("provenance", {}),
            keypoints=d.get("keypoints"),
            track_id=d.get("track_id", -1),
            action_id=d.get("action_id", -1),
            occluded=d.get("occluded", False),
            is_crowd=d.get("is_crowd", False),
            keyframe=d.get("keyframe", True),  # 読み込んだ box はキーフレーム扱い
            obb=_parse_obb(d.get("obb")),
            source=d.get("source", "human"),
            score=d.get("score"),
        )
        if "object_id" in d:
            obj.object_id = d["object_id"]
            _gid = max(_gid, d["object_id"])
        return obj


def _parse_obb(raw):
    from ..data.annotation import parse_obb_field

    return parse_obb_field(raw)
