"""
obb.py — Oriented / Rotated Bounding Box の幾何ユーティリティ (torch 実装)

YOAKE の個体検出は「細長く・自由な向きで・互いに接近する」動物を分離するために
Oriented Bounding Box (OBB) を主表現に使う。本モジュールはその**唯一の幾何的真実**
であり、detector / tracking / behavior ROI / 評価 / GUI がすべてここを経由する。

=== 表現の定義 ===

1. ``obb``      (..., 5) = [cx, cy, w, h, theta]
   交換用の正準表現。theta は **ラジアン**、x 軸正方向から反時計回り (画像座標では
   y 下向きなので見た目は時計回り)。``canonicalize_obb`` で theta ∈ [-pi/2, pi/2) に畳む。

2. ``encoded``  (..., 6) = [cx, cy, w, h, sin(2*theta), cos(2*theta)]
   ネットワークが回帰する表現。**角度を 2 倍する**のが要点:

     矩形は theta と theta+pi が同一の箱なので、周期は pi であって 2pi ではない。
     生の (sin t, cos t) は周期 2pi なので、同じ箱に 2 つの正解が存在してしまい
     境界 (±pi/2) で損失が不連続に飛ぶ。2 倍角 (sin 2t, cos 2t) は周期 pi と一致する
     ため一意かつ連続で、これが周期境界問題の直接的な解になる。

     残る 180 度の曖昧さ (頭と尾の区別) は箱の幾何では原理的に決まらないので、
     ``models/pose_head`` の head-tail polarity head が担当する (責務の分離)。

3. ``corners``  (..., 4, 2) = 4 頂点 (反時計回り)。polygon 描画・厳密 IoU に使う。
4. ``aabb``     (..., 4) = [x1, y1, x2, y2] 軸平行。既存 YOAKE 資産との互換境界。

=== IoU の 2 系統 ===

* ``rotated_iou``  — 凸多角形クリッピングによる**厳密**な IoU。評価 / NMS / 追跡の
  association に使う。勾配を流す用途は想定しない。
* ``probiou``      — Gaussian 近似 (Bhattacharyya 距離) による**微分可能**な IoU。
  学習の回帰損失に使う。厳密 IoU は場合分けが不連続で勾配が壊れるため。
  参考: Llerena et al., "Gaussian Bounding Boxes and Probabilistic IoU" (2021)。

numpy 版 (torch 非依存の段が使う小サブセット) は ``stages/_geometry.py`` にあり、
``tests/test_geometry_obb.py`` が両者の数値一致を固定している。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

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

# 数値安定化用の共通イプシロン (幾何全体で一貫させる)
_EPS = 1e-7
_HALF_PI = math.pi / 2.0
# 半開区間 [-pi/2, pi/2) の上端スナップ幅。float32 の丸め (~2e-7 @ 1.57) より十分大きく、
# 幾何的に無視できる大きさに取る。
_ANGLE_SNAP = 1e-6


# ---------------------------------------------------------------------------
# 角度の正規化 / 符号化
# ---------------------------------------------------------------------------

def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    """
    theta を半開区間 [-pi/2, pi/2) に畳む (矩形の周期は pi)。

    float32 では ``(theta + pi/2) % pi`` が pi にごく近い値へ丸められ、結果が
    +pi/2 側に張り付くことがある (theta = -pi/2 が +pi/2 として返るなど)。
    幾何的には同一の箱だが正準形が 2 つになってしまうので、上端近傍は pi を
    引いて畳み直し、表現を一意に保つ。畳み直しによる誤差は 1e-6 rad 未満。
    """
    out = (theta + _HALF_PI) % math.pi - _HALF_PI
    return torch.where(out >= _HALF_PI - _ANGLE_SNAP, out - math.pi, out)


def canonicalize_obb(obb: torch.Tensor) -> torch.Tensor:
    """
    OBB を正準形に整える。

    - w, h を非負にクランプ
    - theta を [-pi/2, pi/2) に畳む

    w/h の入れ替え (長辺を w に固定する) は **行わない**。長辺固定は w<->h の
    境界でサイズ回帰が不連続に飛ぶうえ、動物の体軸は「長辺」とは限らないため。
    向きの一意性は 2 倍角符号化が担保する。

    Args:
        obb: (..., 5) [cx, cy, w, h, theta]
    Returns:
        (..., 5) 正準化された OBB
    """
    cx, cy, w, h, theta = obb.unbind(-1)
    return torch.stack(
        [cx, cy, w.clamp(min=0), h.clamp(min=0), normalize_angle(theta)], dim=-1
    )


def encode_angle(theta: torch.Tensor) -> torch.Tensor:
    """theta (...,) → (..., 2) = [sin(2 theta), cos(2 theta)]。"""
    two = 2.0 * theta
    return torch.stack([torch.sin(two), torch.cos(two)], dim=-1)


def decode_angle(sincos: torch.Tensor) -> torch.Tensor:
    """
    (..., 2) = [sin(2 theta), cos(2 theta)] → theta ∈ [-pi/2, pi/2)。

    ネットワーク出力は正規化されていない (ノルムが 1 とは限らない) ので atan2 を使う。
    atan2 は方向のみを見るためノルムに不変で、追加の正規化は不要。
    """
    s, c = sincos.unbind(-1)
    return normalize_angle(0.5 * torch.atan2(s, c))


def obb_to_encoded(obb: torch.Tensor) -> torch.Tensor:
    """(..., 5) [cx,cy,w,h,theta] → (..., 6) [cx,cy,w,h,sin2t,cos2t]。"""
    return torch.cat([obb[..., :4], encode_angle(obb[..., 4])], dim=-1)


def encoded_to_obb(encoded: torch.Tensor) -> torch.Tensor:
    """(..., 6) [cx,cy,w,h,sin2t,cos2t] → (..., 5) [cx,cy,w,h,theta]。"""
    theta = decode_angle(encoded[..., 4:6])
    return torch.cat([encoded[..., :4], theta.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# 表現間の変換
# ---------------------------------------------------------------------------

def obb_to_corners(obb: torch.Tensor) -> torch.Tensor:
    """
    OBB → 4 頂点。

    Args:
        obb: (..., 5) [cx, cy, w, h, theta]
    Returns:
        (..., 4, 2) 頂点座標。順序は box ローカル座標の
        (-w/2,-h/2), (+w/2,-h/2), (+w/2,+h/2), (-w/2,+h/2) を回転したもの
        (theta=0 なら tl, tr, br, bl に対応)。
    """
    cx, cy, w, h, theta = obb.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    # ローカル座標の 4 頂点 (..., 4)
    lx = torch.stack([-hw, hw, hw, -hw], dim=-1)
    ly = torch.stack([-hh, -hh, hh, hh], dim=-1)

    # 回転 + 平行移動
    cos_t = cos_t.unsqueeze(-1)
    sin_t = sin_t.unsqueeze(-1)
    gx = cx.unsqueeze(-1) + lx * cos_t - ly * sin_t
    gy = cy.unsqueeze(-1) + lx * sin_t + ly * cos_t
    return torch.stack([gx, gy], dim=-1)


def corners_to_obb(corners: torch.Tensor) -> torch.Tensor:
    """
    4 頂点 → OBB。

    入力が厳密な長方形であれば厳密に復元する。GUI の頂点編集などで平行四辺形に
    歪んだ場合は、辺 0→1 の向きを theta とし、各軸への射影幅を w/h とする
    最小二乗的な当てはめになる (破綻せず常に妥当な OBB を返す)。

    Args:
        corners: (..., 4, 2)
    Returns:
        (..., 5) [cx, cy, w, h, theta]
    """
    center = corners.mean(dim=-2)                       # (..., 2)
    edge = corners[..., 1, :] - corners[..., 0, :]      # (..., 2) 幅方向の辺
    theta = torch.atan2(edge[..., 1], edge[..., 0])

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rel = corners - center.unsqueeze(-2)                # (..., 4, 2)
    # box ローカル軸への射影
    local_x = rel[..., 0] * cos_t.unsqueeze(-1) + rel[..., 1] * sin_t.unsqueeze(-1)
    local_y = -rel[..., 0] * sin_t.unsqueeze(-1) + rel[..., 1] * cos_t.unsqueeze(-1)

    w = local_x.max(dim=-1).values - local_x.min(dim=-1).values
    h = local_y.max(dim=-1).values - local_y.min(dim=-1).values

    obb = torch.stack([center[..., 0], center[..., 1], w, h, theta], dim=-1)
    return canonicalize_obb(obb)


def obb_to_aabb(obb: torch.Tensor) -> torch.Tensor:
    """
    OBB → それを囲う最小の軸平行矩形 (..., 4) [x1, y1, x2, y2]。

    行動認識の context ROI はここを起点に ``expand_aabb`` で広げる。
    """
    corners = obb_to_corners(obb)                        # (..., 4, 2)
    lo = corners.min(dim=-2).values                      # (..., 2)
    hi = corners.max(dim=-2).values
    return torch.cat([lo, hi], dim=-1)


def aabb_to_obb(aabb: torch.Tensor) -> torch.Tensor:
    """
    軸平行矩形 → theta=0 の OBB。

    OBB アノテーションを持たない旧データセットの安全な fallback。**真の回転
    アノテーションではない**ので、学習側 (``data/annotation.py`` の migration と
    ``training/losses.py``) はこれを角度損失から除外・警告する。
    """
    x1, y1, x2, y2 = aabb.unbind(-1)
    return torch.stack([
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
        (x2 - x1).clamp(min=0),
        (y2 - y1).clamp(min=0),
        torch.zeros_like(x1),
    ], dim=-1)


def cxcywh_to_obb(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → theta=0 の OBB (..., 5)。"""
    return torch.cat([boxes, torch.zeros_like(boxes[..., :1])], dim=-1)


def obb_to_cxcywh(obb: torch.Tensor) -> torch.Tensor:
    """
    OBB → 軸平行 [cx, cy, w, h]。

    既存の AABB パイプライン (損失 / 評価 / JSON) へ戻す互換境界。外接矩形を使うため
    theta != 0 では w, h が元の OBB より大きくなる点に注意。
    """
    aabb = obb_to_aabb(obb)
    x1, y1, x2, y2 = aabb.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def obb_area(obb: torch.Tensor) -> torch.Tensor:
    """OBB の面積 (..., )。"""
    return obb[..., 2].clamp(min=0) * obb[..., 3].clamp(min=0)


# ---------------------------------------------------------------------------
# 行動認識用の context box
# ---------------------------------------------------------------------------

def expand_aabb(
    aabb: torch.Tensor,
    scale: float,
    image_size: Tuple[int, int] | None = None,
    square: bool = False,
) -> torch.Tensor:
    """
    AABB を中心を保ったまま ``scale`` 倍に広げる (行動認識の context ROI)。

    Args:
        aabb: (..., 4) [x1, y1, x2, y2]
        scale: 拡大率。1.0 でそのまま、2.0 で縦横 2 倍。
        image_size: (H, W) を渡すと画像内にクリップする。None ならクリップしない。
        square: True のとき長辺に合わせて正方形にする (ROIAlign の歪みを避ける)。
    Returns:
        (..., 4)
    """
    x1, y1, x2, y2 = aabb.unbind(-1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0) * scale
    h = (y2 - y1).clamp(min=0) * scale

    if square:
        side = torch.maximum(w, h)
        w, h = side, side

    out = torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)

    if image_size is not None:
        H, W = image_size
        lo = out.new_tensor([0.0, 0.0, 0.0, 0.0])
        hi = out.new_tensor([W, H, W, H])
        out = torch.maximum(out, lo)
        out = torch.minimum(out, hi)
    return out


def obb_to_context_aabb(
    obb: torch.Tensor,
    scale: float = 2.0,
    image_size: Tuple[int, int] | None = None,
    square: bool = False,
) -> torch.Tensor:
    """
    個体の tight OBB → 行動認識用の広い context AABB。

    YOAKE の設計原則: 個体検出は tight OBB、行動認識は expanded AABB。
    前者は個体を分離するため細粒度・orientation-sensitive、後者は他個体や周辺状況
    (social context) を視野に入れるため粗粒度・軸平行でよい。
    """
    return expand_aabb(obb_to_aabb(obb), scale, image_size=image_size, square=square)


# ---------------------------------------------------------------------------
# 厳密な rotated IoU (凸多角形の交差面積)
# ---------------------------------------------------------------------------

def _polygon_area(poly: torch.Tensor) -> torch.Tensor:
    """shoelace 公式で凸多角形の面積を返す。poly: (..., K, 2) → (...,)"""
    x, y = poly[..., 0], poly[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=-1)
    y_next = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * (x * y_next - x_next * y).sum(dim=-1).abs()


def _points_in_box(points: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """
    点が凸多角形の内部にあるか判定する。

    Args:
        points:  (N, M, K, 2)
        corners: (N, M, 4, 2) — 反時計回り (obb_to_corners の順序)
    Returns:
        (N, M, K) bool
    """
    # 各辺 (a→b) について外積符号を見る。凸かつ一貫した巻き方向なら全辺同符号。
    a = corners                                    # (N, M, 4, 2)
    b = torch.roll(corners, shifts=-1, dims=-2)    # (N, M, 4, 2)
    edge = b - a                                   # (N, M, 4, 2)

    # (N, M, K, 1, 2) - (N, M, 1, 4, 2) → (N, M, K, 4, 2)
    rel = points.unsqueeze(-2) - a.unsqueeze(-3)
    cross = edge.unsqueeze(-3)[..., 0] * rel[..., 1] - edge.unsqueeze(-3)[..., 1] * rel[..., 0]

    # 巻き方向は obb_to_corners が保証するが、退化 (w or h = 0) に備えて両符号を許す
    return (cross >= -_EPS).all(dim=-1) | (cross <= _EPS).all(dim=-1)


def _edge_intersections(c1: torch.Tensor, c2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2 つの 4 角形の全辺ペア (4x4=16) の交点を求める。

    Args:
        c1, c2: (N, M, 4, 2)
    Returns:
        points: (N, M, 16, 2)
        valid:  (N, M, 16) bool — 両線分の内部で交わるペアのみ True
    """
    p = c1                                       # (N, M, 4, 2) 線分1 始点
    r = torch.roll(c1, shifts=-1, dims=-2) - p   # 線分1 方向
    q = c2                                       # 線分2 始点
    s = torch.roll(c2, shifts=-1, dims=-2) - q   # 線分2 方向

    # 全ペアへブロードキャスト: (N, M, 4, 4, 2)
    p_ = p.unsqueeze(-2)
    r_ = r.unsqueeze(-2)
    q_ = q.unsqueeze(-3)
    s_ = s.unsqueeze(-3)

    def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    denom = _cross(r_, s_)                       # (N, M, 4, 4)
    qp = q_ - p_
    t = _cross(qp, s_) / torch.where(denom.abs() < _EPS, torch.full_like(denom, _EPS), denom)
    u = _cross(qp, r_) / torch.where(denom.abs() < _EPS, torch.full_like(denom, _EPS), denom)

    valid = (denom.abs() >= _EPS) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts = p_ + t.unsqueeze(-1) * r_              # (N, M, 4, 4, 2)

    N, M = c1.shape[0], c1.shape[1]
    return pts.reshape(N, M, 16, 2), valid.reshape(N, M, 16)


def _convex_intersection_area(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    2 つの凸 4 角形の交差面積 (N, M)。

    手順 (交差多角形の頂点は必ずこの 3 種のいずれか):
      1. c1 の頂点で c2 の内部にあるもの
      2. c2 の頂点で c1 の内部にあるもの
      3. 辺どうしの交点
    候補を集め、重心まわりの偏角でソートして shoelace で面積を出す。
    無効な候補は「最初の有効点」で埋めると退化三角形になり面積 0 に寄与しないため、
    可変頂点数をマスクだけで扱える (分岐なしでベクトル化できる)。
    """
    in1 = _points_in_box(c1, c2)                 # (N, M, 4)
    in2 = _points_in_box(c2, c1)                 # (N, M, 4)
    inter_pts, inter_valid = _edge_intersections(c1, c2)   # (N, M, 16, *)

    pts = torch.cat([c1, c2, inter_pts], dim=-2)           # (N, M, 24, 2)
    valid = torch.cat([in1, in2, inter_valid], dim=-1)     # (N, M, 24)

    has_any = valid.any(dim=-1)                            # (N, M)
    # 有効点が 3 未満なら面積 0 (点/線分の交差)
    enough = valid.sum(dim=-1) >= 3

    # 重心 (有効点のみ)
    cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    masked = pts * valid.unsqueeze(-1)
    centroid = masked.sum(dim=-2) / cnt                    # (N, M, 2)

    rel = pts - centroid.unsqueeze(-2)
    ang = torch.atan2(rel[..., 1], rel[..., 0])            # (N, M, 24)
    # 無効点は最後に送る
    ang = torch.where(valid, ang, torch.full_like(ang, 1e4))
    order = ang.argsort(dim=-1)

    pts_sorted = torch.gather(pts, -2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
    valid_sorted = torch.gather(valid, -1, order)

    # 無効スロットを「先頭の有効点」で埋める (退化して面積 0)
    first = pts_sorted[..., :1, :].expand_as(pts_sorted)
    pts_sorted = torch.where(valid_sorted.unsqueeze(-1), pts_sorted, first)

    area = _polygon_area(pts_sorted)
    return torch.where(has_any & enough, area, torch.zeros_like(area))


def rotated_iou(obb1: torch.Tensor, obb2: torch.Tensor) -> torch.Tensor:
    """
    厳密な rotated IoU 行列。

    評価 (rotated AP) / rotated NMS / tracking association 用。凸多角形クリッピングに
    よる厳密解なので、学習損失には ``probiou`` を使うこと (こちらは場合分けが
    不連続で勾配が信頼できない)。

    Args:
        obb1: (N, 5) [cx, cy, w, h, theta]
        obb2: (M, 5)
    Returns:
        (N, M) IoU ∈ [0, 1]
    """
    if obb1.numel() == 0 or obb2.numel() == 0:
        return obb1.new_zeros((obb1.shape[0], obb2.shape[0]))

    N, M = obb1.shape[0], obb2.shape[0]
    c1 = obb_to_corners(obb1).unsqueeze(1).expand(N, M, 4, 2).contiguous()
    c2 = obb_to_corners(obb2).unsqueeze(0).expand(N, M, 4, 2).contiguous()

    inter = _convex_intersection_area(c1, c2)              # (N, M)
    a1 = obb_area(obb1).unsqueeze(1)                       # (N, 1)
    a2 = obb_area(obb2).unsqueeze(0)                       # (1, M)
    union = (a1 + a2 - inter).clamp(min=_EPS)
    return (inter / union).clamp(0.0, 1.0)


def rotated_nms(
    obb: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    rotated IoU ベースの greedy NMS。

    Args:
        obb: (N, 5), scores: (N,)
    Returns:
        keep: (K,) long — スコア降順の残存インデックス
    """
    if obb.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=obb.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = rotated_iou(obb[i: i + 1], obb[rest])[0]
        order = rest[ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=obb.device)


# ---------------------------------------------------------------------------
# 微分可能な IoU (ProbIoU / Gaussian Bhattacharyya)
# ---------------------------------------------------------------------------

def _obb_to_gaussian(obb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    OBB を 2D ガウス分布の共分散成分 (a, b, c) に変換する。

    一様分布する矩形の共分散は Sigma = R diag(w^2/12, h^2/12) R^T。
    戻り値は Sigma = [[a, c], [c, b]] の 3 成分。
    """
    w, h, theta = obb[..., 2], obb[..., 3], obb[..., 4]
    w2 = (w.clamp(min=_EPS) ** 2) / 12.0
    h2 = (h.clamp(min=_EPS) ** 2) / 12.0
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos2, sin2 = cos_t ** 2, sin_t ** 2

    a = w2 * cos2 + h2 * sin2
    b = w2 * sin2 + h2 * cos2
    c = (w2 - h2) * cos_t * sin_t
    return a, b, c


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    微分可能な OBB 類似度 (ProbIoU)。要素ごとに対応する 1 対 1 版。

    OBB を 2D ガウスとみなし、Bhattacharyya 距離 BD から
    Hellinger 距離 HD = sqrt(1 - exp(-BD)) を求め、``1 - HD`` を返す。
    中心・サイズ・角度のすべてに滑らかな勾配が流れるので、L1 + rotated-IoU の
    代わりに単独で回帰損失として使える。角度の周期性も共分散経由で自然に扱われる。

    参考: Llerena et al. (2021), "Gaussian Bounding Boxes and Probabilistic IoU"。

    Args:
        obb1, obb2: (..., 5) 同一 shape。
    Returns:
        (...,) ∈ [0, 1] — 1 が完全一致。
    """
    x1, y1 = obb1[..., 0], obb1[..., 1]
    x2, y2 = obb2[..., 0], obb2[..., 1]
    a1, b1, c1 = _obb_to_gaussian(obb1)
    a2, b2, c2 = _obb_to_gaussian(obb2)

    a_s, b_s, c_s = a1 + a2, b1 + b2, c1 + c2
    det_s = (a_s * b_s - c_s ** 2).clamp(min=eps)
    dx, dy = x1 - x2, y1 - y2

    t1 = 0.25 * (a_s * dy ** 2 + b_s * dx ** 2) / det_s
    t2 = 0.5 * (c_s * dx * (-dy)) / det_s
    det1 = (a1 * b1 - c1 ** 2).clamp(min=0)
    det2 = (a2 * b2 - c2 ** 2).clamp(min=0)
    t3 = 0.5 * torch.log(det_s / (4.0 * torch.sqrt(det1 * det2) + eps) + eps)

    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = torch.sqrt((1.0 - torch.exp(-bd)).clamp(min=0.0))
    return 1.0 - hd
