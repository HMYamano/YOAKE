"""
losses.py — 損失関数群

実装する損失:
1. DetectionLoss (Hungarian matching + class + bbox_l1 + giou)
2. ActionLoss (focal cross-entropy)
3. IDLoss (cross-entropy + optional metric loss)
4. TemporalSmoothingLoss
5. CombinedLoss (上記をまとめて重み付き合算)

設計方針:
- Hungarian matching は scipy で行う (torchvision 依存を最小化)
- focal loss を class / action / ID に適用可能にする
- loss weight は config から参照する
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.config import LossConfig
from ..geometry.obb import (
    aabb_to_obb,
    canonicalize_obb,
    cxcywh_to_obb,
    encode_angle,
    normalize_angle,
    probiou,
    rotated_iou,
)
from ..utils.misc import generalized_box_iou, xyxy_to_cxcywh, cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,   # (N, C) logits
    targets: torch.Tensor,  # (N,) class indices
    num_classes: int,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid Focal Loss (RetinaNet 論文から)。
    multi-class 版: one-hot に展開して各クラスで BCE+focal を計算。
    """
    N = inputs.shape[0]
    if N == 0:
        return inputs.sum() * 0.0

    # one-hot encoding
    target_onehot = torch.zeros_like(inputs)  # (N, C)
    valid_mask = (targets >= 0) & (targets < num_classes)
    target_onehot[valid_mask, targets[valid_mask]] = 1.0

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction="none")
    p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)
    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def softmax_focal_loss(
    inputs: torch.Tensor,   # (N, C) logits
    targets: torch.Tensor,  # (N,) class indices
    gamma: float = 2.0,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Softmax Focal Loss (通常の cross-entropy に focal weighting を追加)。
    gamma=0 で通常の cross-entropy になる。
    """
    valid = targets != ignore_index
    if valid.sum() == 0:
        return inputs.sum() * 0.0

    log_probs = F.log_softmax(inputs[valid], dim=-1)
    nll = -log_probs.gather(1, targets[valid].unsqueeze(1)).squeeze(1)  # (N_valid,)

    if gamma > 0:
        probs = F.softmax(inputs[valid], dim=-1)
        p_t = probs.gather(1, targets[valid].unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** gamma
        nll = focal_weight * nll

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """
    DETR スタイルの Hungarian matcher。
    予測と GT の二部マッチングを行う。
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_rotated: bool = False,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # OBB 検出時に localization cost を rotated IoU にするか。
        # 接近して並ぶ細長い個体では、軸平行 GIoU だと隣の個体に引っ張られた
        # マッチングになりやすいので、回転を考慮した重なりで評価する。
        self.use_rotated = bool(use_rotated)

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, C+1)
        pred_boxes: torch.Tensor,   # (B, Q, 4) [cx, cy, w, h]
        targets: List[Dict],        # B 個の dict: {boxes: (N_i, 4), class_ids: (N_i,)}
        pred_obb: Optional[torch.Tensor] = None,  # (B, Q, 5)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            indices: List[(row_idx, col_idx)] × B
                     row_idx: 予測のインデックス
                     col_idx: GT のインデックス
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("scipy が必要です: pip install scipy")

        B, Q = pred_logits.shape[:2]
        indices = []
        rotated = self.use_rotated and pred_obb is not None

        for b in range(B):
            gt_boxes = targets[b]["boxes"]    # (N_i, 4)
            gt_cls = targets[b]["class_ids"]  # (N_i,)
            N_i = gt_boxes.shape[0]

            if N_i == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # Class cost: negative softmax probability of GT class
            # background クラスを除く
            probs = F.softmax(pred_logits[b], dim=-1)  # (Q, C+1)
            cost_class = -probs[:, gt_cls]  # (Q, N_i)

            # BBox L1 cost — pred is cxcywh, GT is xyxy → convert GT to cxcywh
            gt_cxcywh = xyxy_to_cxcywh(gt_boxes)
            cost_bbox = torch.cdist(
                pred_boxes[b], gt_cxcywh, p=1
            )  # (Q, N_i)

            if rotated:
                # rotated IoU cost — GT に OBB があればそれを、無ければ theta=0 を使う
                gt_obb = targets[b].get("obb")
                gt_obb = aabb_to_obb(gt_boxes) if gt_obb is None else gt_obb
                cost_giou = -rotated_iou(pred_obb[b], gt_obb)   # (Q, N_i)
            else:
                # GIoU cost — both in xyxy
                pred_xyxy = cxcywh_to_xyxy(pred_boxes[b])  # (Q, 4)
                gt_xyxy = gt_boxes                           # (N_i, 4) already xyxy
                giou = generalized_box_iou(pred_xyxy, gt_xyxy)  # (Q, N_i)
                cost_giou = -giou

            # Combined cost matrix
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )  # (Q, N_i)

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices


# ---------------------------------------------------------------------------
# Detection Loss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """
    DETR スタイルの detection loss。
    - class loss (focal cross-entropy)
    - bbox L1 loss
    - GIoU loss
    """

    def __init__(self, cfg: LossConfig, num_classes: int = 1, obb_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.obb_cfg = obb_cfg
        self.obb_enabled = bool(obb_cfg is not None and obb_cfg.enabled)
        self.matcher = HungarianMatcher(
            cost_class=cfg.w_class,
            cost_bbox=cfg.w_bbox_l1,
            cost_giou=cfg.w_bbox_giou,
            use_rotated=self.obb_enabled and obb_cfg.use_rotated_matching,
        )
        self.obb_loss = OBBLoss(cfg, obb_cfg) if self.obb_enabled else None

    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, C+1)
        pred_boxes: torch.Tensor,   # (B, Q, 4) [cx, cy, w, h]
        targets: List[Dict],        # B 個の dict
        pred_obb: Optional[torch.Tensor] = None,     # (B, Q, 5)
        pred_angles: Optional[torch.Tensor] = None,  # (B, Q, 2)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            losses: Dict[str, Tensor]
        """
        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, targets, pred_obb=pred_obb)

        # Class loss
        loss_class = self._class_loss(pred_logits, targets, indices)

        # BBox L1 loss
        loss_bbox = self._bbox_l1_loss(pred_boxes, targets, indices)

        # GIoU loss
        loss_giou = self._giou_loss(pred_boxes, targets, indices)

        total = (
            self.cfg.w_class * loss_class
            + self.cfg.w_bbox_l1 * loss_bbox
            + self.cfg.w_bbox_giou * loss_giou
        )

        losses = {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

        # --- OBB losses ---
        if self.obb_loss is not None and pred_obb is not None and pred_angles is not None:
            obb_losses = self._obb_losses(pred_obb, pred_angles, targets, indices)
            losses.update(obb_losses)
            total = total + obb_losses["loss_obb"]

        losses["loss_detection"] = total
        return losses

    def _obb_losses(
        self,
        pred_obb: torch.Tensor,      # (B, Q, 5)
        pred_angles: torch.Tensor,   # (B, Q, 2)
        targets: List[Dict],
        indices: List[Tuple],
    ) -> Dict[str, torch.Tensor]:
        """マッチした予測と GT の OBB 損失を集める。"""
        pred_list, angle_list, gt_list, valid_list = [], [], [], []

        for b, (row_ind, col_ind) in enumerate(indices):
            if len(row_ind) == 0:
                continue
            pred_list.append(pred_obb[b][row_ind])
            angle_list.append(pred_angles[b][row_ind])

            gt_obb = targets[b].get("obb")
            if gt_obb is None:
                # 真の回転 GT が無い: AABB から theta=0 で代用しつつ、角度損失は無効化。
                # ここで角度まで学習させると「すべて水平」を教え込むことになる。
                gt_obb = aabb_to_obb(targets[b]["boxes"])
                valid = torch.zeros(
                    len(col_ind), dtype=torch.bool, device=pred_obb.device
                )
            else:
                gt_obb = gt_obb.to(pred_obb.device)
                av = targets[b].get("obb_valid")
                valid = (
                    torch.ones(len(col_ind), dtype=torch.bool, device=pred_obb.device)
                    if av is None else av.to(pred_obb.device)[col_ind].bool()
                )
            gt_list.append(gt_obb[col_ind])
            valid_list.append(valid)

        if not pred_list:
            zero = pred_obb.sum() * 0.0
            return {
                "loss_obb_angle": zero, "loss_obb_probiou": zero, "loss_obb": zero,
            }

        return self.obb_loss(
            torch.cat(pred_list, 0),
            torch.cat(angle_list, 0),
            torch.cat(gt_list, 0),
            torch.cat(valid_list, 0),
        )

    def _class_loss(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Background を "num_classes" クラスとして扱う。
        matched queries には GT class, 残りは background を割り当てる。
        """
        B, Q = pred_logits.shape[:2]
        device = pred_logits.device

        # background class index = num_classes
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=device
        )

        for b, (row_ind, col_ind) in enumerate(indices):
            if len(row_ind) == 0:
                continue
            gt_cls = targets[b]["class_ids"][col_ind]  # (N_matched,)
            target_classes[b, row_ind] = gt_cls

        # focal cross-entropy
        # (B, Q, C+1) → (B*Q, C+1)
        logits_flat = pred_logits.reshape(-1, self.num_classes + 1)
        targets_flat = target_classes.reshape(-1)

        loss = softmax_focal_loss(
            logits_flat, targets_flat,
            gamma=self.cfg.focal_gamma,
        )
        return loss

    def _bbox_l1_loss(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> torch.Tensor:
        pred_list, gt_list = [], []
        for b, (row_ind, col_ind) in enumerate(indices):
            if len(row_ind) == 0:
                continue
            pred_list.append(pred_boxes[b][row_ind])
            gt_list.append(targets[b]["boxes"][col_ind])

        if not pred_list:
            return pred_boxes.sum() * 0.0

        preds = torch.cat(pred_list, dim=0)          # cxcywh
        gts = xyxy_to_cxcywh(torch.cat(gt_list, dim=0))  # GT xyxy → cxcywh
        return F.l1_loss(preds, gts, reduction="mean")

    def _giou_loss(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> torch.Tensor:
        pred_list, gt_list = [], []
        for b, (row_ind, col_ind) in enumerate(indices):
            if len(row_ind) == 0:
                continue
            pred_list.append(pred_boxes[b][row_ind])
            gt_list.append(targets[b]["boxes"][col_ind])

        if not pred_list:
            return pred_boxes.sum() * 0.0

        preds = torch.cat(pred_list, dim=0)
        gts = torch.cat(gt_list, dim=0)  # already xyxy

        pred_xyxy = cxcywh_to_xyxy(preds)
        gt_xyxy = gts  # GT boxes are already xyxy normalized

        # 対角要素が各 match の GIoU
        giou = generalized_box_iou(pred_xyxy, gt_xyxy)
        diag = giou.diag()
        return (1 - diag).mean()


# ---------------------------------------------------------------------------
# OBB Loss
# ---------------------------------------------------------------------------

class OBBLoss(nn.Module):
    """
    Oriented Bounding Box の回帰損失。

    2 つの補完的な項を足す:

    1. **角度損失** — 予測 [sin2t, cos2t] と GT [sin2t, cos2t] の L1。
       2 倍角なので矩形の周期 pi と一致し、±pi/2 の境界で損失が飛ばない。
       ネットワーク出力は正規化されていないので、GT の単位ベクトルへ向かうように
       L2 正規化してから比較する (ノルムは角度に無関係なので学習対象にしない)。

    2. **ProbIoU 損失** — 中心・サイズ・角度を 1 つの微分可能な IoU で同時に最適化。
       厳密な rotated IoU は場合分けが不連続で勾配が壊れるため、ガウス近似を使う。

    === derived な GT の扱い ===

    AABB から theta=0 で機械的に導出した GT (``obb_source="derived"``) は
    真の回転情報を持たない。これを角度損失に含めると「すべて水平」という誤った
    事前分布を学習してしまうので、``angle_valid`` マスクで**角度損失からのみ**除外する
    (ProbIoU は位置とサイズの情報を持つので除外しない)。
    """

    def __init__(self, cfg: LossConfig, obb_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.angle_weight = cfg.w_obb_angle
        self.probiou_weight = cfg.w_obb_probiou
        if obb_cfg is not None:
            self.angle_weight *= obb_cfg.angle_loss_weight
            self.probiou_weight *= obb_cfg.probiou_loss_weight

    def forward(
        self,
        pred_obb: torch.Tensor,         # (N, 5) [cx, cy, w, h, theta]
        pred_angles: torch.Tensor,      # (N, 2) [sin2t, cos2t] 未正規化
        gt_obb: torch.Tensor,           # (N, 5)
        angle_valid: Optional[torch.Tensor] = None,  # (N,) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {"loss_obb_angle", "loss_obb_probiou", "loss_obb"}
        """
        if pred_obb.shape[0] == 0:
            zero = pred_obb.sum() * 0.0
            return {
                "loss_obb_angle": zero,
                "loss_obb_probiou": zero,
                "loss_obb": zero,
            }

        # --- 角度損失 ---
        if angle_valid is None:
            angle_valid = torch.ones(
                pred_obb.shape[0], dtype=torch.bool, device=pred_obb.device
            )
        if angle_valid.any():
            gt_enc = encode_angle(gt_obb[angle_valid, 4])            # (M, 2) 単位ベクトル
            pred_enc = F.normalize(pred_angles[angle_valid], dim=-1)
            loss_angle = F.l1_loss(pred_enc, gt_enc, reduction="mean")
        else:
            loss_angle = pred_angles.sum() * 0.0

        # --- ProbIoU 損失 ---
        iou = probiou(pred_obb, gt_obb)
        loss_probiou = (1.0 - iou).mean()

        total = self.angle_weight * loss_angle + self.probiou_weight * loss_probiou
        return {
            "loss_obb_angle": loss_angle,
            "loss_obb_probiou": loss_probiou,
            "loss_obb": total,
        }


# ---------------------------------------------------------------------------
# Pose Loss (任意)
# ---------------------------------------------------------------------------

class PoseLoss(nn.Module):
    """
    キーポイントと head-tail polarity の損失 (どちらも任意)。

    pose アノテーションが無いデータセットでは呼ばれないので、
    ``pose_enabled=False`` の既存ワークフローには一切影響しない。
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred_keypoints: Optional[torch.Tensor],   # (N, K, 3) [x, y, vis_logit]
        gt_keypoints: Optional[torch.Tensor],     # (N, K, 3) [x, y, v]
        pred_polarity: Optional[torch.Tensor] = None,  # (N, 2)
        gt_polarity: Optional[torch.Tensor] = None,    # (N,) 0/1、-1 は ignore
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = None

        if pred_keypoints is not None and gt_keypoints is not None \
                and pred_keypoints.shape[0] > 0:
            # v > 0 のキーポイントだけを座標回帰の対象にする
            vis = (gt_keypoints[..., 2] > 0).float()               # (N, K)
            coord_err = F.l1_loss(
                pred_keypoints[..., :2], gt_keypoints[..., :2], reduction="none"
            ).sum(dim=-1)                                          # (N, K)
            denom = vis.sum().clamp(min=1.0)
            loss_kp = (coord_err * vis).sum() / denom
            losses["loss_pose_keypoint"] = loss_kp
            total = self.cfg.w_pose_keypoint * loss_kp

        if pred_polarity is not None and gt_polarity is not None \
                and pred_polarity.shape[0] > 0:
            loss_pol = F.cross_entropy(pred_polarity, gt_polarity, ignore_index=-1)
            losses["loss_pose_polarity"] = loss_pol
            term = self.cfg.w_pose_polarity * loss_pol
            total = term if total is None else total + term

        if total is None:
            ref = pred_keypoints if pred_keypoints is not None else pred_polarity
            total = ref.sum() * 0.0 if ref is not None else torch.zeros(())
        losses["loss_pose"] = total
        return losses


# ---------------------------------------------------------------------------
# Multi-head / Hierarchical Behavior Loss
# ---------------------------------------------------------------------------

class MultiHeadActionLoss(nn.Module):
    """
    multi-head 行動分類の損失。

    head ごとに独立した cross-entropy を計算し、平均する。GT が欠けている head は
    -1 (ignore) で飛ばせるので、「locomotion だけラベルがある」ような部分
    アノテーションのデータセットでも学習できる。
    """

    def __init__(self, cfg: LossConfig, head_names: Optional[List[str]] = None):
        super().__init__()
        self.cfg = cfg
        self.head_names = list(head_names) if head_names else []

    def forward(
        self,
        head_logits: Dict[str, torch.Tensor],   # head 名 → (N, C_h)
        head_targets: Dict[str, torch.Tensor],  # head 名 → (N,)、-1 は ignore
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        terms: List[torch.Tensor] = []

        for name, logits in head_logits.items():
            target = head_targets.get(name)
            if target is None or logits.shape[0] == 0:
                continue
            if (target != -1).sum() == 0:
                continue
            loss = softmax_focal_loss(
                logits, target, gamma=self.cfg.focal_gamma, ignore_index=-1
            )
            losses[f"loss_action_{name}"] = loss
            terms.append(loss)

        if terms:
            total = torch.stack(terms).mean() * self.cfg.w_action
        else:
            any_logits = next(iter(head_logits.values()), None)
            total = any_logits.sum() * 0.0 if any_logits is not None else torch.zeros(())
        losses["loss_action"] = total
        return losses


# ---------------------------------------------------------------------------
# Action Loss
# ---------------------------------------------------------------------------

class ActionLoss(nn.Module):
    """
    行動分類の損失。

    imbalance_strategy:
      "none"         — 重み無し softmax focal loss
      "class_weight" — Laplace 平滑化逆頻度重みを適用した cross-entropy
      "focal"        — softmax focal loss (gamma でクラス不均衡を緩和)
    """

    def __init__(self, cfg: LossConfig, num_actions: int = 5):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        # class_weight モード用: set_class_weights() で設定
        self._class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """外部から逆頻度重みを注入する。compute_class_weights() で生成するのが典型。"""
        self._class_weights = class_weights

    @contextmanager
    def temporarily_disable_class_weights(self):
        """Temporarily disable class weights, e.g. for validation-time loss."""
        prev = self._class_weights
        self._class_weights = None
        try:
            yield
        finally:
            self._class_weights = prev

    @staticmethod
    def compute_class_weights(
        counts: List[int],
        smoothing: float = 1.0,
        clip_min: float = 0.1,
        clip_max: float = 10.0,
    ) -> torch.Tensor:
        """
        Laplace 平滑化逆頻度重みを計算する。

        Args:
            counts: クラスごとのサンプル数 (長さ = num_actions)
            smoothing: Laplace 平滑化係数 (デフォルト 1.0)
            clip_min / clip_max: 重みのクリップ範囲

        Returns:
            Tensor of shape (num_actions,)
        """
        counts_t = torch.tensor(counts, dtype=torch.float32)
        total = counts_t.sum() + smoothing * len(counts)
        weights = total / (counts_t + smoothing)
        weights = weights / weights.mean()  # 平均 1.0 に正規化
        return weights.clamp(clip_min, clip_max)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        action_logits: torch.Tensor,  # (N, num_actions)
        gt_action_ids: torch.Tensor,  # (N,) — -1 は unannotated
    ) -> torch.Tensor:
        if action_logits.shape[0] == 0:
            return action_logits.sum() * 0.0

        # 有効サンプルが 0 の場合は 0 を返す (F.cross_entropy は NaN になるため)
        if (gt_action_ids != -1).sum() == 0:
            return action_logits.sum() * 0.0

        strategy = getattr(self.cfg, "imbalance_strategy", "none")

        if strategy == "class_weight" and self._class_weights is not None:
            # 重み付き cross-entropy (-1 は ignore のまま保持する)
            w = self._class_weights.to(action_logits.device)
            loss = F.cross_entropy(
                action_logits,
                gt_action_ids,
                weight=w,
                ignore_index=-1,
            )
        else:
            # focal loss (strategy=="focal" または "none" または fallback)
            gamma = self.cfg.focal_gamma if strategy in ("focal", "none") else 0.0
            loss = softmax_focal_loss(
                action_logits,
                gt_action_ids,
                gamma=gamma,
                ignore_index=-1,
            )

        return loss * self.cfg.w_action


# ---------------------------------------------------------------------------
# ID Loss
# ---------------------------------------------------------------------------

class IDLoss(nn.Module):
    """
    ID 分類の損失。
    - cross-entropy (GT track ID を分類)
    - optional: triplet loss (metric learning)
    """

    def __init__(self, cfg: LossConfig, max_ids: int = 50):
        super().__init__()
        self.cfg = cfg
        self.max_ids = max_ids

    def forward(
        self,
        id_logits: torch.Tensor,        # (N, max_ids+1)
        gt_track_ids: torch.Tensor,     # (N,) — -1 は unknown
        embeddings: Optional[torch.Tensor] = None,  # (N, D) for metric loss
    ) -> Dict[str, torch.Tensor]:
        if id_logits.shape[0] == 0:
            return {"loss_id": id_logits.sum() * 0.0}

        # track_id を [0, max_ids) にクランプ (-1 は ignore)
        valid_ids = gt_track_ids.clamp(max=self.max_ids)

        loss_cls = softmax_focal_loss(
            id_logits,
            valid_ids,
            gamma=self.cfg.focal_gamma,
            ignore_index=-1,
        ) * self.cfg.w_id_cls

        losses = {"loss_id_cls": loss_cls, "loss_id": loss_cls}

        # Metric loss (triplet)
        if self.cfg.w_id_metric > 0 and embeddings is not None:
            loss_metric = self._triplet_loss(embeddings, gt_track_ids)
            losses["loss_id_metric"] = loss_metric
            losses["loss_id"] = loss_cls + self.cfg.w_id_metric * loss_metric

        return losses

    def _triplet_loss(
        self,
        embeddings: torch.Tensor,  # (N, D) L2 normalized
        labels: torch.Tensor,      # (N,)
        margin: float = 0.3,
    ) -> torch.Tensor:
        """Online hard triplet mining を使った triplet loss"""
        N = embeddings.shape[0]
        if N < 3:
            return embeddings.sum() * 0.0

        # Pairwise distance matrix
        dist_mat = 1 - torch.mm(embeddings, embeddings.t())  # cosine distance (N, N)

        valid = labels >= 0
        if valid.sum() < 3:
            return embeddings.sum() * 0.0

        total_loss = 0.0
        count = 0

        for i in range(N):
            if not valid[i]:
                continue
            label_i = labels[i]

            # Positive: same label, different index
            pos_mask = (labels == label_i) & valid
            pos_mask[i] = False
            if not pos_mask.any():
                continue

            # Negative: different label
            neg_mask = (labels != label_i) & valid

            if not neg_mask.any():
                continue

            # Hardest positive (最も遠い)
            ap = dist_mat[i][pos_mask].max()
            # Hardest negative (最も近い)
            an = dist_mat[i][neg_mask].min()

            loss_i = F.relu(ap - an + margin)
            total_loss += loss_i
            count += 1

        return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Temporal Smoothing Loss
# ---------------------------------------------------------------------------

class TemporalSmoothingLoss(nn.Module):
    """
    連続フレーム間の bbox / action prediction の滑らかさを促す loss。
    急激な変化にペナルティを与える。
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred_boxes_seq: torch.Tensor,    # (B, T, Q, 4)
        action_logits_seq: Optional[torch.Tensor] = None,  # (B, T, N, A)
    ) -> torch.Tensor:
        """連続フレーム間の差分の L2 ノルムを loss にする"""
        # bbox の時間方向の滑らかさ
        diff = pred_boxes_seq[:, 1:] - pred_boxes_seq[:, :-1]  # (B, T-1, Q, 4)
        loss = diff.pow(2).mean()

        if action_logits_seq is not None:
            probs = F.softmax(action_logits_seq, dim=-1)  # (B, T, N, A)
            diff_a = probs[:, 1:] - probs[:, :-1]  # (B, T-1, N, A)
            loss = loss + diff_a.pow(2).mean()

        return loss * self.cfg.w_temporal_smooth


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    全損失をまとめて計算する統合損失クラス。
    training_stage に応じて使う損失を切り替える。
    """

    def __init__(
        self,
        cfg: LossConfig,
        num_classes: int = 1,
        num_actions: int = 5,
        max_ids: int = 50,
        obb_cfg=None,
        head_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.stage = 1

        self.detection_loss = DetectionLoss(cfg, num_classes, obb_cfg=obb_cfg)
        self.action_loss = ActionLoss(cfg, num_actions)
        self.multi_head_action_loss = MultiHeadActionLoss(cfg, head_names)
        self.pose_loss = PoseLoss(cfg)
        self.id_loss = IDLoss(cfg, max_ids)
        self.temporal_smooth = TemporalSmoothingLoss(cfg)

    def set_stage(self, stage: int) -> None:
        self.stage = stage

    def forward(
        self,
        model_output,          # HTRTDETROutput
        targets: List,         # List[Dict] or List[List[Dict]]
        stage: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: HTRTDETROutput
            targets: stage に応じた GT データ
            stage: override training stage

        Returns:
            losses: Dict[str, Tensor]
        """
        s = stage or self.stage
        losses = {}

        # ----- Stage 1: Detection -----
        if s >= 1:
            # targets は List[Dict] (1 フレーム)
            frame_targets = targets if isinstance(targets[0], dict) else targets[-1]
            det_losses = self.detection_loss(
                model_output.pred_logits,
                model_output.pred_boxes,
                frame_targets,
                pred_obb=getattr(model_output, "pred_obb", None),
                pred_angles=getattr(model_output, "pred_angles", None),
            )
            losses.update(det_losses)

            # Auxiliary losses (各中間デコーダ層)
            aux_outputs = getattr(model_output, "aux_outputs", None)
            if aux_outputs:
                aux_total = model_output.pred_logits.new_zeros(())
                for i, aux in enumerate(aux_outputs):
                    aux_losses = self.detection_loss(
                        aux["pred_logits"],
                        aux["pred_boxes"],
                        frame_targets,
                        pred_obb=aux.get("pred_obb"),
                        pred_angles=aux.get("pred_angles"),
                    )
                    aux_total = aux_total + aux_losses["loss_detection"]
                    losses[f"aux_det_layer{i}"] = aux_losses["loss_detection"]  # ログ用 (合計には含まない)
                losses["loss_detection_aux"] = aux_total

            # Pose losses (pose head 有効時のみ)
            pred_kp = getattr(model_output, "pred_keypoints", None)
            pred_pol = getattr(model_output, "pred_polarity", None)
            if pred_kp is not None or pred_pol is not None:
                pose_losses = self._pose_losses(model_output, frame_targets)
                if pose_losses:
                    losses.update(pose_losses)

        # ----- Stage 2: Action -----
        if s in (2, 4):
            frame_targets = targets if isinstance(targets[0], dict) else targets[-1]
            behavior = getattr(model_output, "behavior", None)

            if behavior is not None and behavior.head_logits:
                # multi-head: head ごとの GT を targets から引く
                head_targets = self._collect_multi_head_targets(
                    frame_targets, model_output, list(behavior.head_logits)
                )
                if head_targets:
                    losses.update(
                        self.multi_head_action_loss(behavior.head_logits, head_targets)
                    )
            elif model_output.action_logits is not None:
                # single / hierarchical: どちらも (N, num_actions) の logit
                gt_action = self._collect_gt_actions(frame_targets, model_output)
                if gt_action is not None:
                    losses["loss_action"] = self.action_loss(
                        model_output.action_logits, gt_action
                    )

        # ----- Stage 3: ID -----
        if s in (3, 4) and model_output.id_logits is not None:
            frame_targets = targets if isinstance(targets[0], dict) else targets[-1]
            gt_track = self._collect_gt_track_ids(frame_targets, model_output)
            if gt_track is not None:
                id_losses = self.id_loss(
                    model_output.id_logits,
                    gt_track,
                    model_output.id_embeddings,
                )
                losses.update(id_losses)

        # Total loss — sum only the top-level aggregate losses to avoid
        # double-counting (loss_class/bbox/giou are already included in loss_detection,
        # loss_obb_angle/probiou in loss_obb which is inside loss_detection)
        _AGGREGATE_KEYS = (
            "loss_detection", "loss_detection_aux", "loss_action", "loss_id", "loss_pose",
        )
        total = sum(losses[k] for k in _AGGREGATE_KEYS if k in losses)
        losses["total_loss"] = total

        return losses

    def _pose_losses(self, model_output, targets) -> Dict[str, torch.Tensor]:
        """
        pose head の損失。GT が無い (pose 未アノテーション) なら空を返す。

        pose アノテーションを持たない既存データセットでも学習が止まらないよう、
        「GT が無ければ損失を作らない」を徹底する。
        """
        if model_output.det_results is None:
            return {}
        has_kp = any("keypoints" in t for t in targets)
        has_pol = any("polarity" in t for t in targets)
        if not has_kp and not has_pol:
            return {}

        from .matching import collect_matched_gt_tensor

        pred_kp = getattr(model_output, "pred_keypoints", None)
        pred_pol = getattr(model_output, "pred_polarity", None)

        det_kp = self._gather_det_field(model_output, "keypoints")
        det_pol = self._gather_det_field(model_output, "polarity")

        gt_kp = collect_matched_gt_tensor(
            model_output.det_results, targets, "keypoints"
        ) if has_kp else None
        gt_pol = collect_matched_gt(
            model_output.det_results, targets, key="polarity"
        ) if has_pol else None

        return self.pose_loss(det_kp, gt_kp, det_pol, gt_pol)

    @staticmethod
    def _gather_det_field(model_output, key: str) -> Optional[torch.Tensor]:
        """det_results から指定フィールドを全バッチ分 concat する。"""
        if model_output.det_results is None:
            return None
        parts = [d[key] for d in model_output.det_results if key in d]
        return torch.cat(parts, dim=0) if parts else None

    def _collect_multi_head_targets(
        self, targets, model_output, head_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        multi-head 用の GT を targets から集める。

        targets の各 dict が ``action_<head名>`` というキーを持つことを期待する。
        欠けている head は学習対象から外れる (部分アノテーション対応)。
        """
        from .matching import collect_matched_gt

        out: Dict[str, torch.Tensor] = {}
        for name in head_names:
            key = f"action_{name}"
            if not any(key in t for t in targets):
                continue
            gt = collect_matched_gt(model_output.det_results, targets, key=key)
            if gt.numel() > 0:
                out[name] = gt
        return out

    def _collect_gt_actions(self, targets, model_output) -> Optional[torch.Tensor]:
        """det_results の各 bbox と GT annotation を IoU マッチングして action_id を収集する。

        matching.py の collect_matched_gt を利用して greedy IoU ベースの割り当てを行う。
        IoU < threshold の未マッチ検出は -1 (ActionLoss で ignore される)。
        """
        if model_output.det_results is None:
            return None
        from .matching import collect_matched_gt
        gt = collect_matched_gt(model_output.det_results, targets, key="action_ids")
        return gt if gt.numel() > 0 else None

    def _collect_gt_track_ids(self, targets, model_output) -> Optional[torch.Tensor]:
        """det_results の各 bbox と GT annotation を IoU マッチングして track_id を収集する。

        matching.py の collect_matched_gt を利用して greedy IoU ベースの割り当てを行う。
        IoU < threshold の未マッチ検出は -1 (IDLoss で ignore される)。
        """
        if model_output.det_results is None:
            return None
        from .matching import collect_matched_gt
        gt = collect_matched_gt(model_output.det_results, targets, key="track_ids")
        return gt if gt.numel() > 0 else None
