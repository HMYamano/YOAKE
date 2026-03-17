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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.config import LossConfig
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
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, C+1)
        pred_boxes: torch.Tensor,   # (B, Q, 4) [cx, cy, w, h]
        targets: List[Dict],        # B 個の dict: {boxes: (N_i, 4), class_ids: (N_i,)}
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

    def __init__(self, cfg: LossConfig, num_classes: int = 1):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=cfg.w_class,
            cost_bbox=cfg.w_bbox_l1,
            cost_giou=cfg.w_bbox_giou,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, C+1)
        pred_boxes: torch.Tensor,   # (B, Q, 4) [cx, cy, w, h]
        targets: List[Dict],        # B 個の dict
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            losses: Dict[str, Tensor]
        """
        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, targets)

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

        return {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_detection": total,
        }

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
# Action Loss
# ---------------------------------------------------------------------------

class ActionLoss(nn.Module):
    """行動分類の損失 (focal cross-entropy)"""

    def __init__(self, cfg: LossConfig, num_actions: int = 5):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions

    def forward(
        self,
        action_logits: torch.Tensor,  # (N, num_actions)
        gt_action_ids: torch.Tensor,  # (N,) — -1 は unannotated
    ) -> torch.Tensor:
        if action_logits.shape[0] == 0:
            return action_logits.sum() * 0.0

        loss = softmax_focal_loss(
            action_logits,
            gt_action_ids,
            gamma=self.cfg.focal_gamma,
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
    ):
        super().__init__()
        self.cfg = cfg
        self.stage = 1

        self.detection_loss = DetectionLoss(cfg, num_classes)
        self.action_loss = ActionLoss(cfg, num_actions)
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
                    )
                    aux_total = aux_total + aux_losses["loss_detection"]
                    losses[f"aux_det_layer{i}"] = aux_losses["loss_detection"]  # ログ用 (合計には含まない)
                losses["loss_detection_aux"] = aux_total

        # ----- Stage 2: Action -----
        if s in (2, 4) and model_output.action_logits is not None:
            # GT action_id を収集
            frame_targets = targets if isinstance(targets[0], dict) else targets[-1]
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

        # Total loss
        total = sum(v for k, v in losses.items() if k.startswith("loss_"))
        losses["total_loss"] = total

        return losses

    def _collect_gt_actions(self, targets, model_output) -> Optional[torch.Tensor]:
        """det_results と GT annotation を照合して action_id を収集する"""
        # 簡略実装: det_results のインデックスと GT の対応を確認
        if model_output.det_results is None:
            return None

        gt_list = []
        for b, det in enumerate(model_output.det_results):
            n = det["features"].shape[0]
            if n == 0:
                continue
            # 実際には matcher で対応をとる必要があるが、
            # ここでは -1 (ignore) を返して注意を促す
            # TODO: proper GT matching
            gt_list.append(torch.full((n,), -1, dtype=torch.long,
                                      device=det["features"].device))

        return torch.cat(gt_list) if gt_list else None

    def _collect_gt_track_ids(self, targets, model_output) -> Optional[torch.Tensor]:
        """det_results と GT annotation を照合して track_id を収集する"""
        if model_output.det_results is None:
            return None

        gt_list = []
        for b, det in enumerate(model_output.det_results):
            n = det["features"].shape[0]
            if n == 0:
                continue
            gt_list.append(torch.full((n,), -1, dtype=torch.long,
                                      device=det["features"].device))

        return torch.cat(gt_list) if gt_list else None
