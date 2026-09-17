

# ─── Data Model ───────────────────────────────────────────────────────────────

_gid = 0


class AnnObject:
    """
    1 個体のアノテーション。

    v2.0 で ``obb`` (回転矩形) を任意フィールドとして追加した。``obb=None`` なら
    従来どおり軸平行 bbox のみのオブジェクトとして扱われ、保存 JSON も v1.1 と
    同じ形になる (旧ツールで読める)。

    ``bbox`` は obb があっても**常に**その外接矩形として保たれる。これにより
    bbox しか見ない既存コード (補間 / 範囲ラベリング / エクスポート) が
    OBB の有無に関係なく動く。

    provenance (``accepted`` / ``source`` / ``score``) は自動アノテーション用。
    ``accepted=False`` の候補は**ディスクに書かれない**。``source="human"`` の
    box は to_dict() が追加キーを一切足さないので、人手だけのプロジェクトの
    JSON は従来とバイト等価のまま。
    """

    def __init__(self, bbox: List[int], track_id: int = -1, action_id: int = -1,
                 occluded: bool = False, is_crowd: bool = False,
                 keyframe: bool = True,
                 obb: Optional[List[float]] = None,
                 class_id: int = 0,
                 accepted: bool = True,
                 source: str = "human",
                 score: Optional[float] = None) -> None:
        global _gid
        _gid += 1
        self.object_id = _gid
        self.bbox = list(bbox)
        self.track_id = track_id
        self.action_id = action_id
        self.class_id = int(class_id)
        self.occluded = occluded
        self.is_crowd = is_crowd
        # keyframe: True = 手動で置いた box (補間のアンカー)、False = 補間/自動生成 box。
        # 編集用フラグで、保存 JSON には出力しない。
        self.keyframe = keyframe
        # obb: [cx, cy, w, h, theta] or None (= 軸平行のみ)
        self.obb = list(obb) if obb is not None else None
        # provenance
        self.accepted = bool(accepted)
        self.source = str(source or "human")
        self.score = None if score is None else float(score)

    # ---- provenance ヘルパー -------------------------------------------

    @property
    def is_candidate(self) -> bool:
        """レビュー待ちの機械提案か (= まだ保存されない box)。"""
        return not self.accepted

    @property
    def is_human(self) -> bool:
        return self.source == "human"

    def accept(self) -> None:
        """候補を確定させる。以後は通常の box として保存対象になる。"""
        self.accepted = True

    def source_label(self) -> str:
        """Sidebar 'Source:' row — human / auto (0.82) / tracked / interpolated."""
        if self.source == "human":
            return "human"
        name = {"auto": "auto", "track": "tracked", "interp": "interpolated"}.get(
            self.source, self.source)
        if self.score is not None:
            return f"{name} ({self.score:.2f})"
        return name

    # ---- OBB ヘルパー -------------------------------------------------

    @property
    def has_obb(self) -> bool:
        return self.obb is not None

    def get_obb(self) -> List[float]:
        """OBB を返す。未設定なら bbox から theta=0 で導出する。"""
        if self.obb is not None:
            return list(self.obb)
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 0), max(y2 - y1, 0), 0.0]

    def corners(self) -> List[Tuple[float, float]]:
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

    def drop_obb(self) -> None:
        """OBB を捨てて軸平行 box に戻す (外接矩形を bbox として残す)。"""
        if self.obb is not None:
            self.sync_bbox()
            self.obb = None

    # ---- 直列化 -------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "object_id": self.object_id,
            "bbox": [int(v) for v in self.bbox],
            "class_id": int(self.class_id),
            "track_id": self.track_id,
            "action_id": self.action_id,
            "occluded": self.occluded,
            "is_crowd": self.is_crowd,
        }
        # OBB は持っているときだけ書く → 無ければ v1.1 と同一形式
        if self.obb is not None:
            cx, cy, w, h, th = self.obb
            d["obb"] = {
                "cx": round(float(cx), 2), "cy": round(float(cy), 2),
                "w": round(float(w), 2), "h": round(float(h), 2),
                "angle": round(float(th), 6),
            }
        # source/score は human 以外のときだけ付く ({} が返るので human は不変)
        d.update(auto_label.provenance_fields(self.source, self.score))
        return d

    @staticmethod
    def from_dict(d: dict) -> "AnnObject":
        global _gid
        obj = AnnObject(
            bbox=d["bbox"],
            track_id=d.get("track_id", -1),
            action_id=d.get("action_id", -1),
            occluded=d.get("occluded", False),
            is_crowd=d.get("is_crowd", False),
            keyframe=d.get("keyframe", True),  # 読み込んだ box はキーフレーム扱い
            obb=_parse_obb(d.get("obb")),
            class_id=int(d.get("class_id", 0) or 0),
            accepted=True,                     # 保存された box は常に確定済み
            source=str(d.get("source", "human") or "human"),
            score=(float(d["score"]) if d.get("score") is not None else None),
        )
        if "object_id" in d:
            obj.object_id = d["object_id"]
            _gid = max(_gid, d["object_id"])
        return obj


def _parse_obb(raw) -> Optional[List[float]]:
    """JSON の obb フィールド (dict / 5要素 / 4頂点8要素) を [cx,cy,w,h,theta] に。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        try:
            ang = raw.get("angle", raw.get("theta", 0.0))
            return [float(raw["cx"]), float(raw["cy"]),
                    float(raw["w"]), float(raw["h"]), float(ang)]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)):
        v = [float(x) for x in raw]
        if len(v) == 5:
            return v
        if len(v) == 8:
            cx = sum(v[0::2]) / 4
            cy = sum(v[1::2]) / 4
            th = math.atan2(v[3] - v[1], v[2] - v[0])
            c, s = math.cos(th), math.sin(th)
            lx = [(v[i] - cx) * c + (v[i + 1] - cy) * s for i in range(0, 8, 2)]
            ly = [-(v[i] - cx) * s + (v[i + 1] - cy) * c for i in range(0, 8, 2)]
            return [cx, cy, max(lx) - min(lx), max(ly) - min(ly), th]
    return None
