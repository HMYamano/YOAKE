"""
silhouette.py — 検出ごとのシルエット幾何 (体軸・頭尾・形状・残差運動)

学習もラベルも要らない、推論の**後処理**。detector の箱 (OBB の 4 隅) を受け取り、
その周りの ROI を二値化して、箱からは分からない量を測る。モデルの出力は変えない。
設計と検証計画は docs/SILHOUETTE_GEOMETRY.md。

=== なぜ要るのか ===

行動ブランチの orientation / interaction 特徴は体軸を前提にしているが、既定の AABB
detector では θ=0 に固定され、OBB でも頭尾 (180°) は決まらない。ここでは

  1. 二値化 → 付属肢を落とした体の芯から PCA で体軸
     (``gui/click_segment.reduce_mask`` をそのまま使う。Click to Box と同じ縮約規則)
  2. 移動方向と時間的連続性で頭尾を決め、2π の向き (heading) にする
  3. 箱では見えない形: 凸性・側方への張り出し (片翅伸展)・前後の幅・芯の数 (交尾)
  4. 体に固定した座標でのフレーム差分 = 全身の移動を差し引いた残差運動の量

を出す。1 個体あたり CPU で数十〜数百 µs。

=== 何をしないか ===

骨格 (細線化) や Medial Axis は作らない。640px 画像で体長 15–40px のハエでは脚は
二値化で消え、骨格に残るのは体軸と翅だけで、それは 1 と 3 で取れる。Optical flow も
計算しない。全身の移動は kinematics が持っており、局所運動の「量」は 4 で足りる。

=== 状態 ===

``SilhouetteAnalyzer`` は track ごとの頭尾の状態と前フレームの濃淡画像を持つので、
フレームは時間順に渡すこと (realtime の geometry プロセスも順番を守る)。
"""

from __future__ import annotations

import contextlib
import csv
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GEOMETRY_FIELDS",
    "GEOMETRY_VERSION",
    "SilhouetteParams",
    "Silhouette",
    "SilhouetteAnalyzer",
    "measure",
    "geometry_meta",
    "rows_as_dicts",
    "draw_geometry",
    "GeometryCSV",
]

#: フィールドの意味を変えたら上げる。読む側は知らない版を拒否できる。
GEOMETRY_VERSION = 1

#: 1 検出 1 行の並び。predictions.json の ``geometry`` 行と geometry.csv の列はこの順。
#: 値が決まらないものは None (JSON の null)。
GEOMETRY_FIELDS: Tuple[str, ...] = (
    "valid",            # 1 = シルエットを測れた / 0 = 測れなかった (diagnosis に理由)
    "cx", "cy",         # 体の中心 [px]
    "axis",             # 体軸 [rad, -pi/2..pi/2)。頭尾なし
    "heading",          # 頭の向き [rad, -pi..pi)。一度も移動を見ていない track は null
    "heading_conf",     # 0..1。向きを支持した移動の量
    "length", "width",  # 体軸方向・直交方向の体の長さ [px] (細い付属肢を除く)
    "elongation",       # 体の芯の細長さ = sqrt(PCA 固有値比)。翅や脚に左右されない
    "area",             # 二値化した blob の面積 [px²] (付属肢を含む)
    "body_area",        # 細い付属肢を落とした体の面積 [px²]
    "appendage_ratio",  # (area - body_area) / body_area
    "solidity",         # 輪郭面積 / 凸包面積 (翅や脚が張り出すと下がる)
    "lateral_extent",   # 体軸の線から横へ最も遠い blob の点 [体長] (付属肢を含む)
    "lateral_asym",     # 体軸の左右での到達の差 [体長] (片翅伸展で大きい)
    "width_skew",       # (前 1/3 の幅 - 後 1/3 の幅) / 和。前 = heading 側。heading が無ければ null
    "twin_core",        # 1 = 同程度の芯が 2 つ (接触・交尾中の 2 個体が 1 つの blob)。
                        #     体軸は近い方の芯、length / width / cx / cy はペアのもの
    "motion_energy",    # 体固定のフレーム差分 / 前景背景の濃度差。前フレームが無ければ null
    "nn_dist",          # 最近傍個体までの距離 [体長]
    "facing_nn",        # heading と最近傍への方向のなす角 [rad, 0..pi]。小さいほど向いている
    "faced_by_nn",      # 最近傍の heading と自分への方向のなす角 [rad, 0..pi]
    "diagnosis",        # "" か、測れなかった理由 (click_segment.DIAGNOSES のキー)
)
_I: Dict[str, int] = {name: i for i, name in enumerate(GEOMETRY_FIELDS)}

_EPS = 1e-6


@dataclass
class SilhouetteParams:
    """シルエット測定の設定。既定値は 640px 画像で体長 15–40px のハエを想定している。

    速度は体長/秒で決めるので、解像度や種が変わってもそのまま使えることが多い。
    """

    #: ROI の半辺 = roi_scale × (箱の長辺) / 2。中心から体長 1 つ分で、翅を横に開いた
    #: 個体も収まる。はみ出したときは 1.6 倍に広げて 1 度だけ測り直す。
    roi_scale: float = 2.0
    #: これより大きい ROI は縮小してから測る。高解像度・大型動物でもコストを一定にする。
    max_roi_side: int = 160
    #: 芯の細長さがこれ未満だと体軸は丸すぎて信用しない (頭尾の更新を止める)。
    min_elongation: float = 1.3
    #: 頭尾の投票に使う「移動中」の下限 [体長/秒]。
    min_speed: float = 1.0
    #: 移動方向が体軸からこの角度以内のときだけ頭尾に投票する (横歩きは投票しない)。
    motion_alignment_deg: float = 45.0
    #: 確立した向きの反転に要る、逆向き移動の継続時間 [秒]。短い後退歩行では反転しない。
    flip_seconds: float = 0.25
    #: これより長く見失った track は向きを忘れる [フレーム]。
    max_gap_frames: int = 15
    #: heading_conf = 1 - exp(-support / support_scale)。support は向きを支持した移動フレーム数。
    support_scale: float = 5.0
    #: 投票の無いフレームごとに support に掛ける減衰。
    stationary_decay: float = 0.99
    #: 速度の指数平滑の係数 (1 = 平滑なし)。
    velocity_smoothing: float = 0.5
    #: motion_energy の体固定 crop の一辺 [px]。体長の 2 倍の正方形をこの解像度で見る。
    motion_crop: int = 32
    #: fps が分からないときの仮定値。
    default_fps: float = 30.0

    def __post_init__(self) -> None:
        if self.roi_scale <= 1.0:
            raise ValueError("SilhouetteParams.roi_scale must be > 1 (the ROI must contain the box)")
        if self.max_roi_side < 16:
            raise ValueError("SilhouetteParams.max_roi_side must be >= 16")
        if self.motion_crop < 8:
            raise ValueError("SilhouetteParams.motion_crop must be >= 8")
        if not 0.0 < self.velocity_smoothing <= 1.0:
            raise ValueError("SilhouetteParams.velocity_smoothing must be in (0, 1]")
        if not 0.0 < self.stationary_decay <= 1.0:
            raise ValueError("SilhouetteParams.stationary_decay must be in (0, 1]")


@dataclass
class Silhouette:
    """1 検出の測定結果。座標と長さはフレームの画素単位。"""

    centre: Tuple[float, float]
    #: 体軸の単位ベクトル。符号は任意 (頭尾は ``SilhouetteAnalyzer`` が決める)。
    axis: Tuple[float, float]
    length: float
    width: float
    #: 体の芯の sqrt(PCA 固有値比)。box の length / width と違い、開いた翅で下がらない。
    elongation: float
    area: float
    body_area: float
    solidity: float
    #: 体軸の線 (中心を通る) から blob が横へ届く距離。+v 側 / -v 側
    #: (v = axis を +90° 回したもの)。左右対称な姿勢では等しい。
    reach_pos: float
    reach_neg: float
    #: 体の +axis 側 / -axis 側 1/3 区間の平均幅。
    front_width: float
    back_width: float
    twin_core: bool
    #: 前景と背景の濃度差 (motion_energy の正規化)。
    contrast: float
    #: ROI 座標の blob (uint8 0/1) と、その座標系: local = (frame - (x0, y0)) * (sx, sy)。
    blob: Any = None
    roi: Tuple[int, int, float, float] = (0, 0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# 1 個体の測定 (状態なし)
# --------------------------------------------------------------------------- #

def measure(
    gray: np.ndarray,
    polygon: Any,
    others: Optional[Sequence[Any]] = None,
    params: Optional[SilhouetteParams] = None,
) -> Tuple[Optional[Silhouette], str]:
    """1 個体のシルエットを測る。

    Args:
        gray: フレーム全体の濃淡画像 (uint8, 2-D)
        polygon: この個体の箱の頂点 (K, 2) [px]。OBB なら 4 隅、AABB なら矩形の 4 隅
        others: 同じフレームの他個体の箱。ROI に入り込んだ他個体の画素を除くのに使う
        params: 設定
    Returns:
        ``(Silhouette, "")`` か ``(None, 診断)``。診断は ``click_segment.DIAGNOSES`` のキー
    """
    from ..gui.click_segment import NOTHING_FOUND, TOUCHES_EDGE

    p = params or SilhouetteParams()
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3 or not np.isfinite(poly).all():
        return None, NOTHING_FOUND
    centre = poly.mean(axis=0)
    extent = float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).max())
    if extent < 1.0:
        return None, NOTHING_FOUND
    rest = [np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in (others or [])]

    result: Optional[Silhouette] = None
    why = NOTHING_FOUND
    # 体が ROI からはみ出したら、1 度だけ広げて測り直す (click_segment と同じ考え方)。
    for grow in (1.0, 1.6):
        half = 0.5 * extent * p.roi_scale * grow
        result, why = _measure_in_roi(gray, poly, centre, half, rest, p)
        if why != TOUCHES_EDGE:
            break
    return result, why


def _measure_in_roi(
    gray: np.ndarray,
    poly: np.ndarray,
    centre: np.ndarray,
    half: float,
    others: List[np.ndarray],
    p: SilhouetteParams,
) -> Tuple[Optional[Silhouette], str]:
    import cv2

    from ..gui.click_segment import NOTHING_FOUND, reduce_mask, threshold_gray

    H, W = gray.shape[:2]
    x0 = max(0, int(math.floor(centre[0] - half)))
    y0 = max(0, int(math.floor(centre[1] - half)))
    x1 = min(W, int(math.ceil(centre[0] + half)) + 1)
    y1 = min(H, int(math.ceil(centre[1] + half)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, NOTHING_FOUND

    roi = gray[y0:y1, x0:x1]
    sx = sy = 1.0
    side = max(x1 - x0, y1 - y0)
    if side > p.max_roi_side:
        k = p.max_roi_side / side
        size = (max(4, int(round((x1 - x0) * k))), max(4, int(round((y1 - y0) * k))))
        roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    roi = np.ascontiguousarray(roi)

    offset = np.array([x0, y0], dtype=np.float64)
    scale = np.array([sx, sy], dtype=np.float64)

    def local_int(pts: np.ndarray) -> np.ndarray:
        return np.round((pts - offset) * scale).astype(np.int32)

    at = (float((centre[0] - x0) * sx), float((centre[1] - y0) * sy))
    mask = threshold_gray(roi, at)

    # 他個体の箱に属する画素は除く (自分の箱と重なる部分は残す)。接触中の 2 個体が
    # 1 つの blob になるのを、検出が 2 つある限りはここで切り離せる。
    theirs: Optional[np.ndarray] = None
    for q in others:
        if q[:, 0].max() < x0 or q[:, 1].max() < y0 or q[:, 0].min() >= x1 or q[:, 1].min() >= y1:
            continue
        if theirs is None:
            theirs = np.zeros_like(mask)
        cv2.fillPoly(theirs, [local_int(q)], 1)
    if theirs is not None:
        mine = np.zeros_like(mask)
        cv2.fillPoly(mine, [local_int(poly)], 1)
        mask[(theirs > 0) & (mine == 0)] = 0

    parts, why = reduce_mask(mask, at, allow_twins=True)
    if parts is None:
        return None, why

    blob, body = parts.blob, parts.body
    ux, uy = parts.axis
    vx, vy = -uy, ux
    mx, my = parts.centre
    u0, u1 = parts.extent[0], parts.extent[1]
    # local px / frame px。縮小の丸めで sx と sy は最大 1% 程度ずれるだけなので平均を使う。
    s = 0.5 * (sx + sy)

    solidity = 1.0
    contours, _hier = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        if hull_area > 0:
            solidity = min(1.0, float(cv2.contourArea(c)) / hull_area)

    # 横への到達: blob 全体 (付属肢を含む) の、体軸の線からの両側の最遠点。横に開いた
    # 翅は幅があるので opening では消えず body に残る — 「blob − body」では測れない。
    lys, lxs = np.nonzero(blob)
    pv = (lxs - mx) * vx + (lys - my) * vy
    reach_pos = max(0.0, float(pv.max()))
    reach_neg = max(0.0, -float(pv.min()))

    # 芯の細長さ。芯は距離変換で付属肢を落とした部分なので、翅を開いても丸くならない。
    cys, cxs = np.nonzero(parts.core)
    elongation = parts.obb[2] / max(parts.obb[3], _EPS)
    if cxs.size >= 3:
        evals = np.linalg.eigvalsh(np.cov(np.stack([cxs, cys]).astype(np.float64)))
        if evals[0] > _EPS:
            elongation = math.sqrt(float(evals[-1]) / float(evals[0]))

    # 前後 1/3 区間の平均幅 = 区間内の体の画素数 / 区間長。
    bys, bxs = np.nonzero(body)
    pu = (bxs - mx) * ux + (bys - my) * uy
    third = max((u1 - u0) / 3.0, 1.0)
    front = float(np.count_nonzero(pu > u1 - third)) / third
    back = float(np.count_nonzero(pu < u0 + third)) / third

    # 前景と背景の濃度差 (motion_energy を照明に依らない量にする正規化。平均で足りる)。
    fg = roi[blob > 0]
    background = mask == 0
    if theirs is not None:
        background &= theirs == 0
    bg = roi[background]
    contrast = abs(float(fg.mean()) - float(bg.mean())) if fg.size and bg.size else 0.0

    cx_l, cy_l, length_l, breadth_l, _theta = parts.obb
    return Silhouette(
        centre=(cx_l / sx + x0, cy_l / sy + y0),
        axis=(ux, uy),
        length=length_l / s,
        width=breadth_l / s,
        elongation=elongation,
        area=float(blob.sum()) / (sx * sy),
        body_area=float(body.sum()) / (sx * sy),
        solidity=solidity,
        reach_pos=reach_pos / s,
        reach_neg=reach_neg / s,
        front_width=front / s,
        back_width=back / s,
        twin_core=parts.twin_core,
        contrast=contrast,
        blob=blob,
        roi=(x0, y0, sx, sy),
    ), ""


# --------------------------------------------------------------------------- #
# 動画全体 (track ごとの頭尾と残差運動)
# --------------------------------------------------------------------------- #

@dataclass
class _Track:
    """1 track の頭尾の状態。heading は直前に報告した向き (符号を決めた体軸)。"""

    frame: int
    centre: Tuple[float, float]
    heading: float
    support: float = 0.0
    contra: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)


class SilhouetteAnalyzer:
    """フレームを時間順に受け取り、検出ごとの幾何を ``GEOMETRY_FIELDS`` の行で返す。

    === 頭尾の決め方 ===

    体軸 (mod π) の 2 つの読み方のうち、直前の向きに近い方を選ぶ (連続性)。体軸方向に
    ``min_speed`` 以上で移動していれば、進行方向を前とみなして投票する。確立した向きは、
    逆向きの移動が ``flip_seconds`` 続いたときだけ反転する (短い後退歩行で反転しない)。
    一度も移動を見ていない track は向きを報告しない (``heading`` = null)。

    Args:
        params: 設定
        fps: 動画の fps (体長/秒への換算と反転の継続時間に使う)。None なら
            ``params.default_fps``。後から ``analyzer.fps = ...`` で入れてもよい
    """

    def __init__(self, params: Optional[SilhouetteParams] = None, fps: Optional[float] = None):
        self.params = params or SilhouetteParams()
        self.fps: Optional[float] = float(fps) if fps else None
        self.reset()

    def reset(self) -> None:
        """新しい動画の開始時に状態を捨てる。"""
        self._tracks: Dict[int, _Track] = {}
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_index: Optional[int] = None
        self.seconds = 0.0
        self.frames = 0

    def timing(self) -> Dict[str, float]:
        """1 フレームあたりの平均処理時間。"""
        return {
            "frames": self.frames,
            "ms_per_frame": round(1000.0 * self.seconds / max(self.frames, 1), 3),
        }

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        boxes_xyxy: Any,
        corners: Optional[Any] = None,
        track_ids: Optional[Sequence[int]] = None,
    ) -> List[List[Any]]:
        """1 フレームを測る。

        Args:
            frame: (H, W, 3) RGB uint8 (2-D の濃淡画像も可)
            frame_index: フレーム番号 (欠番があってよい)
            boxes_xyxy: (N, 4) 画素座標の箱
            corners: (N, 4, 2) OBB の隅 [px]。None なら箱の 4 隅を使う
            track_ids: (N,) 頭尾と残差運動は track 単位で決めるので、-1 の検出は
                ``heading`` / ``motion_energy`` が null になる
        Returns:
            検出順に並んだ N 行。各行は ``GEOMETRY_FIELDS`` の順
        """
        t0 = time.perf_counter()
        p = self.params
        fi = int(frame_index)
        gray = _to_gray(frame)
        polys = _polygons(boxes_xyxy, corners)
        n = len(polys)
        tids = [int(t) for t in track_ids] if track_ids is not None else [-1] * n
        if len(tids) != n:
            raise ValueError(f"track_ids has {len(tids)} entries for {n} boxes")

        sils: List[Optional[Silhouette]] = []
        whys: List[str] = []
        with _opencv_single_thread():
            for i in range(n):
                sil, why = measure(gray, polys[i], polys[:i] + polys[i + 1:], p)
                sils.append(sil)
                whys.append(why)

        consecutive = (
            self._prev_gray is not None
            and self._prev_index is not None
            and fi - self._prev_index == 1
            and self._prev_gray.shape == gray.shape
        )
        headings: List[Optional[float]] = [None] * n
        confs = [0.0] * n
        energies: List[Optional[float]] = [None] * n
        for i, sil in enumerate(sils):
            if sil is None or tids[i] < 0:
                continue
            prev = self._tracks.get(tids[i])
            prev_pose = (prev.frame, prev.centre, prev.heading) if prev is not None else None
            heading, conf, aligned = self._resolve(tids[i], fi, sil)
            if conf > 0.0:
                headings[i], confs[i] = heading, conf
            if consecutive and prev_pose is not None and prev_pose[0] == fi - 1:
                assert self._prev_gray is not None  # consecutive が保証する
                energies[i] = _motion_energy(
                    self._prev_gray, gray, sil, prev_pose[1], prev_pose[2], aligned, p
                )

        stale = [t for t, st in self._tracks.items() if fi - st.frame > p.max_gap_frames]
        for t in stale:
            del self._tracks[t]
        self._prev_gray = gray
        self._prev_index = fi

        rows = _rows(polys, sils, whys, headings, confs, energies)
        self.seconds += time.perf_counter() - t0
        self.frames += 1
        return rows

    def _resolve(self, tid: int, frame: int, sil: Silhouette) -> Tuple[float, float, float]:
        """track の頭尾を 1 フレーム進める。

        Returns:
            (報告する向き, 確信度, 前フレームの向きに近い方の読み)。最後の値は
            motion_energy の crop を前フレームと同じ向きに揃えるのに使う (反転した
            フレームでも crop を 180° 回さない)。
        """
        p = self.params
        fps = self.fps or p.default_fps
        axis_angle = math.atan2(sil.axis[1], sil.axis[0])
        readings = (_wrap(axis_angle), _wrap(axis_angle + math.pi))
        st = self._tracks.get(tid)
        if st is None or frame <= st.frame or frame - st.frame > p.max_gap_frames:
            st = _Track(frame=frame, centre=sil.centre, heading=readings[0])
            self._tracks[tid] = st
            return st.heading, 0.0, st.heading

        dt = float(frame - st.frame)
        a = p.velocity_smoothing
        vel = (
            a * (sil.centre[0] - st.centre[0]) / dt + (1.0 - a) * st.velocity[0],
            a * (sil.centre[1] - st.centre[1]) / dt + (1.0 - a) * st.velocity[1],
        )
        elongation = sil.elongation
        previous = st.heading
        aligned = min(readings, key=lambda h: abs(_wrap(h - previous)))
        # 丸い姿勢 (グルーミング中など) の体軸は雑音なので、向きは前のまま持つ。
        chosen = aligned if elongation >= p.min_elongation else previous

        support, contra = st.support, st.contra
        voted = False
        moving = math.hypot(*vel) * fps / max(sil.length, _EPS) >= p.min_speed
        if moving and elongation >= p.min_elongation:
            direction = math.atan2(vel[1], vel[0])
            along = abs(math.cos(direction - axis_angle))
            if along >= math.cos(math.radians(p.motion_alignment_deg)):
                voted = True
                forward = readings[0] if math.cos(direction - readings[0]) > 0 else readings[1]
                if abs(_wrap(forward - chosen)) < math.pi / 2:
                    support += 1.0
                    contra = 0
                else:
                    contra += 1
                    # まだ根拠が無い向きは最初の投票で決める。根拠がある向きは、
                    # 逆向きの移動が続いたときだけ覆す。
                    needed = max(1, int(round(p.flip_seconds * fps))) if support > 0 else 1
                    if contra >= needed:
                        chosen = forward
                        support = float(contra)
                        contra = 0
        if not voted:
            support *= p.stationary_decay
            contra = 0

        st.frame, st.centre, st.heading = frame, sil.centre, chosen
        st.support, st.contra, st.velocity = support, contra, vel
        conf = 1.0 - math.exp(-support / p.support_scale) if support > 0 else 0.0
        return chosen, conf, aligned


def _rows(
    polys: List[np.ndarray],
    sils: List[Optional[Silhouette]],
    whys: List[str],
    headings: List[Optional[float]],
    confs: List[float],
    energies: List[Optional[float]],
) -> List[List[Any]]:
    """測定結果を ``GEOMETRY_FIELDS`` の行に並べる (最近傍と facing もここで決める)。"""
    n = len(polys)
    centres = [
        (s.centre if s is not None else (float(polys[i][:, 0].mean()), float(polys[i][:, 1].mean())))
        for i, s in enumerate(sils)
    ]
    rows: List[List[Any]] = []
    for i, sil in enumerate(sils):
        row: List[Any] = [None] * len(GEOMETRY_FIELDS)
        if sil is None:
            row[_I["valid"]] = 0
            row[_I["diagnosis"]] = whys[i]
            rows.append(row)
            continue

        L = max(sil.length, _EPS)
        axis_angle = math.atan2(sil.axis[1], sil.axis[0])
        row[_I["valid"]] = 1
        row[_I["cx"]] = round(sil.centre[0], 2)
        row[_I["cy"]] = round(sil.centre[1], 2)
        row[_I["axis"]] = round(_fold_pi(axis_angle), 4)
        row[_I["heading_conf"]] = round(confs[i], 4)
        row[_I["length"]] = round(sil.length, 2)
        row[_I["width"]] = round(sil.width, 2)
        row[_I["elongation"]] = round(sil.elongation, 4)
        row[_I["area"]] = round(sil.area, 1)
        row[_I["body_area"]] = round(sil.body_area, 1)
        row[_I["appendage_ratio"]] = round((sil.area - sil.body_area) / max(sil.body_area, 1.0), 4)
        row[_I["solidity"]] = round(sil.solidity, 4)
        row[_I["lateral_extent"]] = round(max(sil.reach_pos, sil.reach_neg) / L, 4)
        row[_I["lateral_asym"]] = round(abs(sil.reach_pos - sil.reach_neg) / L, 4)
        row[_I["twin_core"]] = 1 if sil.twin_core else 0
        row[_I["diagnosis"]] = ""

        heading = headings[i]
        if heading is not None:
            row[_I["heading"]] = round(heading, 4)
            if math.cos(heading - axis_angle) > 0:
                front, back = sil.front_width, sil.back_width
            else:
                front, back = sil.back_width, sil.front_width
            row[_I["width_skew"]] = round((front - back) / max(front + back, _EPS), 4)
        if energies[i] is not None:
            row[_I["motion_energy"]] = round(float(energies[i]), 4)  # type: ignore[arg-type]

        nearest, best = -1, math.inf
        for j in range(n):
            if j == i:
                continue
            d = math.hypot(centres[j][0] - centres[i][0], centres[j][1] - centres[i][1])
            if d < best:
                nearest, best = j, d
        if nearest >= 0:
            dx = centres[nearest][0] - centres[i][0]
            dy = centres[nearest][1] - centres[i][1]
            row[_I["nn_dist"]] = round(best / L, 4)
            if heading is not None:
                row[_I["facing_nn"]] = round(abs(_wrap(heading - math.atan2(dy, dx))), 4)
            other = headings[nearest]
            if other is not None:
                row[_I["faced_by_nn"]] = round(abs(_wrap(other - math.atan2(-dy, -dx))), 4)
        rows.append(row)
    return rows


def _motion_energy(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    sil: Silhouette,
    prev_centre: Tuple[float, float],
    prev_heading: float,
    heading: float,
    p: SilhouetteParams,
) -> Optional[float]:
    """体固定のフレーム差分。

    前後のフレームを、それぞれのフレームでの体の位置と向きで切り出して重ねるので、
    並進と回転 (全身の移動) は打ち消され、脚・翅・腹部の動きだけが差として残る。
    スケールは今のフレームの体長に揃える。差は今の blob (少し膨らませた) の中で平均し、
    前景と背景の濃度差で割って照明に依らない量にする。
    """
    import cv2

    S = int(p.motion_crop)
    if sil.blob is None:
        return None
    k = 2.0 * sil.length / S
    m_cur = _crop_affine(sil.centre, heading, k, S)
    m_prev = _crop_affine(prev_centre, prev_heading, k, S)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    cur = cv2.warpAffine(gray, m_cur, (S, S), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    old = cv2.warpAffine(prev_gray, m_prev, (S, S), flags=flags, borderMode=cv2.BORDER_REPLICATE)

    # crop → ROI 座標 (blob の座標系): local = (frame - (x0, y0)) * (sx, sy)
    x0, y0, sx, sy = sil.roi
    m_loc = np.array([
        [sx * m_cur[0, 0], sx * m_cur[0, 1], sx * (m_cur[0, 2] - x0)],
        [sy * m_cur[1, 0], sy * m_cur[1, 1], sy * (m_cur[1, 2] - y0)],
    ])
    region = cv2.warpAffine(
        sil.blob, m_loc, (S, S),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    region = cv2.dilate(region, np.ones((3, 3), np.uint8))
    selected = region > 0
    if not selected.any():
        return None
    diff = cv2.absdiff(cur, old)
    return float(diff[selected].mean()) / max(sil.contrast, 1.0)


def _crop_affine(
    centre: Tuple[float, float], phi: float, k: float, size: int
) -> np.ndarray:
    """crop の画素 (x', y') → フレーム (x, y)。x' 軸が向き phi に沿う。

        [x, y] = centre + R(phi) · k · ([x', y'] − o),   o = (size − 1) / 2
    """
    c, s = math.cos(phi), math.sin(phi)
    o = (size - 1) / 2.0
    return np.array([
        [k * c, -k * s, centre[0] - k * c * o + k * s * o],
        [k * s, k * c, centre[1] - k * s * o - k * c * o],
    ])


# --------------------------------------------------------------------------- #
# 入出力の補助
# --------------------------------------------------------------------------- #

def geometry_meta(params: Optional[SilhouetteParams] = None) -> Dict[str, Any]:
    """predictions.json の ``_meta["geometry"]``: 列の名前と、測ったときの設定。"""
    return {
        "version": GEOMETRY_VERSION,
        "fields": list(GEOMETRY_FIELDS),
        "params": asdict(params or SilhouetteParams()),
    }


def rows_as_dicts(
    rows: Sequence[Sequence[Any]], fields: Sequence[str] = GEOMETRY_FIELDS
) -> List[Dict[str, Any]]:
    """``geometry`` 行を {列名: 値} に直す (読む側の補助)。"""
    return [dict(zip(fields, row)) for row in rows]


def draw_geometry(
    frame: np.ndarray, rows: Sequence[Sequence[Any]], thickness: int = 1
) -> np.ndarray:
    """向きを ``frame`` に描く (in place、RGB)。

    heading がある個体は矢印 (確信度 0.5 以上は緑、未満は橙)、まだ向きの無い個体は
    体軸の線 (灰)。頭尾の決まり方を目で確かめるためのもの。
    """
    import cv2

    for row in rows:
        if not row or not row[_I["valid"]]:
            continue
        cx, cy = float(row[_I["cx"]]), float(row[_I["cy"]])
        L = float(row[_I["length"]])
        heading = row[_I["heading"]]
        if heading is None:
            a = float(row[_I["axis"]])
            dx, dy = 0.5 * L * math.cos(a), 0.5 * L * math.sin(a)
            cv2.line(
                frame,
                (int(round(cx - dx)), int(round(cy - dy))),
                (int(round(cx + dx)), int(round(cy + dy))),
                (200, 200, 200), thickness, cv2.LINE_AA,
            )
            continue
        h = float(heading)
        conf = float(row[_I["heading_conf"]] or 0.0)
        color = (0, 255, 0) if conf >= 0.5 else (255, 160, 0)
        tail = (int(round(cx - 0.4 * L * math.cos(h))), int(round(cy - 0.4 * L * math.sin(h))))
        tip = (int(round(cx + 0.75 * L * math.cos(h))), int(round(cy + 0.75 * L * math.sin(h))))
        cv2.arrowedLine(frame, tail, tip, color, thickness, cv2.LINE_AA, tipLength=0.3)
    return frame


class GeometryCSV:
    """geometry.csv を 1 フレームずつ書く (長い動画やリアルタイムでもメモリに溜めない)。

    列: frame, detection (そのフレームの検出番号), track_id, ``GEOMETRY_FIELDS``。
    """

    def __init__(self, path: str):
        self._handle = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(["frame", "detection", "track_id", *GEOMETRY_FIELDS])

    def write(
        self, frame_index: int, track_ids: Sequence[int], rows: Sequence[Sequence[Any]]
    ) -> None:
        for i, row in enumerate(rows):
            tid = int(track_ids[i]) if i < len(track_ids) else -1
            self._writer.writerow(
                [int(frame_index), i, tid, *("" if v is None else v for v in row)]
            )

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


# --------------------------------------------------------------------------- #
# 小物
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _opencv_single_thread() -> Iterator[None]:
    """測定の間だけ OpenCV を 1 スレッドにする。

    ROI は 1 個体あたり数千画素しかなく、OpenCV のスレッドプールは仕事より起動の方が
    高くつく (10 個体・640x480 で実測 6.8 → 3.9 ms/frame)。切り替えは 1 µs 未満で、
    抜けるときに元のスレッド数へ戻すので、同じプロセスの他の OpenCV 処理には影響しない。
    """
    import cv2

    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def _wrap(a: float) -> float:
    """角度を [-pi, pi) に畳む。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _fold_pi(a: float) -> float:
    """体軸の角度を [-pi/2, pi/2) に畳む (``click_segment.normalize_angle`` と同じ規約)。"""
    from ..gui.click_segment import normalize_angle

    return normalize_angle(a)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB / RGBA / 濃淡 → 濃淡 (uint8)。共有メモリのビューを持ち続けないよう必ずコピーする。"""
    import cv2

    arr = np.asarray(frame)
    if arr.ndim == 2:
        return np.array(arr, dtype=np.uint8, copy=True)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"expected an (H, W), (H, W, 3) or (H, W, 4) frame, got {arr.shape}")


def _polygons(boxes_xyxy: Any, corners: Optional[Any]) -> List[np.ndarray]:
    """各検出の箱の頂点 (K, 2)。OBB の隅があればそれ、無ければ AABB の 4 隅。"""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if corners is not None:
        pts = np.asarray(corners, dtype=np.float64)
        if pts.shape[0] != boxes.shape[0]:
            raise ValueError(f"{pts.shape[0]} corner sets for {boxes.shape[0]} boxes")
        return [pts[i].reshape(-1, 2) for i in range(pts.shape[0])]
    return [
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        for x1, y1, x2, y2 in boxes
    ]
