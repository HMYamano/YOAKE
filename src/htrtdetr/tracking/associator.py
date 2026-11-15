"""
associator.py — OBB 対応の detection ↔ track 関連付け

複数の手がかりを重み付き和で 1 つの類似度にまとめ、Hungarian 法で最適割当を行う:

    similarity = w_iou    * IoU(rotated or axis-aligned)
               + w_app    * cosine(appearance embedding)
               + w_motion * 等速度予測との一致度
               + w_orient * OBB 向きの一致度

各重みは ``TrackingConfig`` から来る。重みを 0 にするとその項の計算自体を飛ばすので、
ablation がそのまま計算量の削減にもなる。既定値 (iou=0.5, appearance=0.5, 他 0) は
既存 ``MemoryIDHead`` の挙動 (0.5*sim + 0.5*IoU) をそのまま再現する。

回転 IoU は OBB がある場合だけ使い、無ければ軸平行 IoU へ自動 fallback する
(AABB detector でも設定を変えずに動く)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from ..geometry.obb import normalize_angle, obb_to_aabb, rotated_iou
from ..utils.misc import box_iou

__all__ = ["AssociationResult", "OBBAssociator"]


@dataclass
class AssociationResult:
    """
    関連付けの結果。

    Attributes:
        matches: (detection index, track index) のペア列
        unmatched_detections: どの track にも割り当たらなかった検出 index
        unmatched_tracks: 検出が割り当たらなかった track index
        scores: matches と同じ並びの類似度スコア
        cost_terms: 各手がかりの寄与 (デバッグ / 可視化用)。
                    {"iou": (D, T), "appearance": (D, T), ...}
    """
    matches: List[Tuple[int, int]]
    unmatched_detections: List[int]
    unmatched_tracks: List[int]
    scores: List[float]
    cost_terms: dict


class OBBAssociator:
    """
    OBB 対応の association。

    Args:
        iou_weight / appearance_weight / motion_weight / orientation_weight:
            各手がかりの重み。0 なら計算しない。
        use_rotated_iou: OBB がある場合に rotated IoU を使う (無ければ軸平行に fallback)
        match_threshold: この類似度未満の割当は棄却して「未マッチ」にする
        max_center_distance: 中心距離がこれを超える組は候補から外す (0 で無効)。
            正規化座標なら 0.2 程度、画素座標なら実距離で指定する。
    """

    def __init__(
        self,
        iou_weight: float = 0.5,
        appearance_weight: float = 0.5,
        motion_weight: float = 0.0,
        orientation_weight: float = 0.0,
        use_rotated_iou: bool = True,
        match_threshold: float = 0.5,
        max_center_distance: float = 0.0,
    ):
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)
        self.motion_weight = float(motion_weight)
        self.orientation_weight = float(orientation_weight)
        self.use_rotated_iou = bool(use_rotated_iou)
        self.match_threshold = float(match_threshold)
        self.max_center_distance = float(max_center_distance)

        total = (self.iou_weight + self.appearance_weight
                 + self.motion_weight + self.orientation_weight)
        if total <= 0:
            raise ValueError("association の重みが全て 0 です (少なくとも 1 つは > 0)")
        self._weight_sum = total

    @classmethod
    def from_config(cls, cfg) -> "OBBAssociator":
        """``TrackingConfig`` から生成する。"""
        return cls(
            iou_weight=cfg.iou_weight,
            appearance_weight=cfg.appearance_weight,
            motion_weight=cfg.motion_weight,
            orientation_weight=cfg.orientation_weight,
            use_rotated_iou=cfg.use_rotated_iou,
            match_threshold=cfg.match_threshold,
            max_center_distance=cfg.max_center_distance,
        )

    # ------------------------------------------------------------------ #
    # 類似度行列
    # ------------------------------------------------------------------ #

    def similarity(
        self,
        det_obb: torch.Tensor,                       # (D, 5)
        track_obb: torch.Tensor,                     # (T, 5)
        det_embed: Optional[torch.Tensor] = None,    # (D, E) L2 正規化済み
        track_embed: Optional[torch.Tensor] = None,  # (T, E) L2 正規化済み
        track_velocity: Optional[torch.Tensor] = None,  # (T, 2)
        has_obb: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        (D, T) の類似度行列と、手がかりごとの内訳を返す。

        Args:
            has_obb: False なら回転情報が無いとみなし、IoU は軸平行、
                     orientation 項は 0 になる (AABB detector 用の fallback)。
        Returns:
            (sim, terms) — sim は**実際に計算した**各項の重み付き平均 ∈ [0, 1] 近傍

        正規化の分母は「計算できた手がかりの重みの和」であって、設定された重みの
        総和ではない。以前は後者で割っていたため、embedding を渡さない呼び出し
        (学習時の内部 tracker、パイプラインの一部の段) では類似度が
        ``iou_weight / 総和 × IoU`` = 既定で ``0.5 × IoU`` に頭打ちになり、閾値 0.5 を
        決して超えなかった。結果として学習中はどの検出もフレーム間で連結されず、
        track 単位の履歴は実質 1 フレームだった (推論は embedding を渡すので連結され、
        学習と推論で時間特徴の意味がずれていた)。全手がかりが揃う場合の値は従来と同一。
        """
        D, T = det_obb.shape[0], track_obb.shape[0]
        device = det_obb.device
        terms: dict = {}
        sim = torch.zeros(D, T, device=device, dtype=det_obb.dtype)

        if D == 0 or T == 0:
            return sim, terms

        used_weight = 0.0

        # --- IoU ---
        if self.iou_weight > 0:
            if self.use_rotated_iou and has_obb:
                iou = rotated_iou(det_obb, track_obb)
            else:
                iou = box_iou(obb_to_aabb(det_obb), obb_to_aabb(track_obb))
            terms["iou"] = iou
            sim = sim + self.iou_weight * iou
            used_weight += self.iou_weight

        # --- Appearance ---
        if self.appearance_weight > 0 and det_embed is not None and track_embed is not None:
            # 入力は L2 正規化済みなので内積 = cosine。[-1,1] を [0,1] へ写す。
            cos = torch.mm(det_embed, track_embed.t())
            app = (cos + 1.0) * 0.5
            terms["appearance"] = app
            sim = sim + self.appearance_weight * app
            used_weight += self.appearance_weight

        # --- Motion (等速度予測との一致) ---
        if self.motion_weight > 0 and track_velocity is not None:
            # track の次フレーム予測中心と検出中心の距離を、track のサイズで正規化。
            pred_center = track_obb[:, :2] + track_velocity          # (T, 2)
            dist = torch.cdist(det_obb[:, :2], pred_center)          # (D, T)
            scale = track_obb[:, 2:4].mean(dim=1).clamp(min=1e-6)    # (T,)
            motion = torch.exp(-dist / scale.unsqueeze(0))
            terms["motion"] = motion
            sim = sim + self.motion_weight * motion
            used_weight += self.motion_weight

        # --- Orientation ---
        if self.orientation_weight > 0 and has_obb:
            # 矩形の周期は pi なので差を [-pi/2, pi/2] に畳んでから正規化する。
            dtheta = normalize_angle(
                det_obb[:, 4].unsqueeze(1) - track_obb[:, 4].unsqueeze(0)
            )
            orient = 1.0 - dtheta.abs() / (torch.pi / 2)
            terms["orientation"] = orient
            sim = sim + self.orientation_weight * orient
            used_weight += self.orientation_weight

        if used_weight <= 0:
            # 有効な手がかりが 1 つも無い (例: appearance だけを重み付けしたのに
            # embedding が無い)。類似度ゼロ = 全員新規として扱い、0 除算はしない。
            return sim, terms
        sim = sim / used_weight

        # --- 中心距離ゲート ---
        if self.max_center_distance > 0:
            dist = torch.cdist(det_obb[:, :2], track_obb[:, :2])
            sim = sim.masked_fill(dist > self.max_center_distance, -1.0)

        return sim, terms

    # ------------------------------------------------------------------ #
    # 割当
    # ------------------------------------------------------------------ #

    def associate(
        self,
        det_obb: torch.Tensor,
        track_obb: torch.Tensor,
        det_embed: Optional[torch.Tensor] = None,
        track_embed: Optional[torch.Tensor] = None,
        track_velocity: Optional[torch.Tensor] = None,
        has_obb: bool = True,
    ) -> AssociationResult:
        """
        Hungarian 法で最適割当を行う (scipy が無ければ greedy に fallback)。

        Returns:
            AssociationResult
        """
        D, T = det_obb.shape[0], track_obb.shape[0]
        if D == 0 or T == 0:
            return AssociationResult(
                matches=[],
                unmatched_detections=list(range(D)),
                unmatched_tracks=list(range(T)),
                scores=[],
                cost_terms={},
            )

        sim, terms = self.similarity(
            det_obb, track_obb, det_embed, track_embed, track_velocity, has_obb
        )
        rows, cols = _optimal_assignment(-sim)

        matches: List[Tuple[int, int]] = []
        scores: List[float] = []
        used_d, used_t = set(), set()
        for r, c in zip(rows, cols):
            s = float(sim[r, c])
            if s < self.match_threshold:
                continue  # 閾値未満は棄却 (新規 ID になる)
            matches.append((int(r), int(c)))
            scores.append(s)
            used_d.add(int(r))
            used_t.add(int(c))

        return AssociationResult(
            matches=matches,
            unmatched_detections=[i for i in range(D) if i not in used_d],
            unmatched_tracks=[j for j in range(T) if j not in used_t],
            scores=scores,
            cost_terms=terms,
        )


def _optimal_assignment(cost: torch.Tensor) -> Tuple[Sequence[int], Sequence[int]]:
    """Hungarian 法 (scipy)。無ければ greedy に fallback する。"""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        return list(r), list(c)
    except ImportError:
        return _greedy_assignment(cost)


def _greedy_assignment(cost: torch.Tensor) -> Tuple[List[int], List[int]]:
    """コスト最小のペアから貪欲に割り当てる (scipy 非依存 fallback)。"""
    D, T = cost.shape
    flat = cost.flatten().argsort()
    used_r, used_c = set(), set()
    rows: List[int] = []
    cols: List[int] = []
    for idx in flat.tolist():
        r, c = divmod(idx, T)
        if r in used_r or c in used_c:
            continue
        rows.append(r)
        cols.append(c)
        used_r.add(r)
        used_c.add(c)
        if len(rows) == min(D, T):
            break
    return rows, cols
