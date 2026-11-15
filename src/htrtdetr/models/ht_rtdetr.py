"""
ht_rtdetr.py — YOAKE: Hierarchical Temporal RT-DETR (統合モデル)

全モジュールを統合した unified model。

アーキテクチャ概要:
  Input frames (B, T, 3, H, W)
    ↓ frame-wise
  [Spatial Detector] → (B, T, Q, D) query features + (B, T, Q, 4) boxes
    ↓
  [Hierarchical Temporal Module] → (B, T, Q, out_dim) temporal features
    ↓ per-frame
  [Multi-Head Feature Router]
    ├─ [Memory-based ID Head] → track_ids, id_confidence
    └─ [Action Head] + [Interaction] → action_labels, action_confidence

出力 (推論時):
  各フレーム各個体について:
  - bbox: [x1, y1, x2, y2]
  - detection_score
  - class_id
  - track_id
  - id_confidence
  - action_id
  - action_confidence

Stage による動作の違い:
  Stage 1: detector のみ active → detection loss のみ
  Stage 2: detector + HTM + action head → action loss
  Stage 3: detector + HTM + ID head → ID loss
  Stage 4: 全モジュール → 統合 loss
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.config import ModelConfig
from .detector import DetectionOutput, build_detector
from .temporal import HierarchicalTemporalModule
from .id_head import MemoryIDHead, IdentityMemory
from .action_head import ActionHead, InteractionFeatureComputer
from .fusion import (
    DetectionFeatureAdapter,
    MultiHeadFeatureRouter,
    QueryTemporalFusion,
    TrackTemporalFusion,
)
from .behavior import BehaviorBranch, BehaviorPrediction
from ..geometry.obb import cxcywh_to_obb
from ..tracking.assembly import FrameDetections, build_track_aligned_history
from ..utils.misc import freeze_module, unfreeze_module
from ..data.feature_builder import GEO_FEAT_DIM


def _one_to_one_queries(iou: torch.Tensor) -> torch.Tensor:
    """(Q, N) の IoU から、GT ごとに互いに異なる query を貪欲に選ぶ (N,)。

    IoU の高い組から確定させる。query が GT より少ない (Q < N) 異常系では
    残りの GT に最良の query を重複して割り当てる (落とすより安全)。
    """
    Q, N = iou.shape
    chosen = torch.full((N,), -1, dtype=torch.long, device=iou.device)
    if N == 0:
        return chosen
    if Q == 0:
        raise ValueError("the detector produced no queries to anchor the keyframe labels to")
    flat = iou.flatten().argsort(descending=True).tolist()
    used_q: set = set()
    for k in flat:
        q, n = divmod(int(k), N)
        if chosen[n] >= 0 or q in used_q:
            continue
        chosen[n] = q
        used_q.add(q)
        if len(used_q) == min(Q, N):
            break
    best = iou.argmax(dim=0)
    return torch.where(chosen >= 0, chosen, best)


# ---------------------------------------------------------------------------
# Unified Model Output
# ---------------------------------------------------------------------------

class HTRTDETROutput:
    """
    YOAKE の forward 出力をまとめるクラス。
    損失計算と推論で共通に使う。
    """

    def __init__(
        self,
        # Detection
        pred_logits: torch.Tensor,          # (B, Q, C+1)
        pred_boxes: torch.Tensor,           # (B, Q, 4)
        query_features: torch.Tensor,       # (B, Q, D)
        # Temporal (None if stage 1)
        temporal_features: Optional[torch.Tensor] = None,  # (B, T, Q, out_dim)
        # ID Head (None if stage 1 or 2)
        id_logits: Optional[torch.Tensor] = None,           # (B*N_det, max_ids+1)
        id_embeddings: Optional[torch.Tensor] = None,       # (B*N_det, emb_dim)
        # Action Head (None if stage 1 or 3)
        action_logits: Optional[torch.Tensor] = None,       # (B*N_det, num_actions)
        action_probs: Optional[torch.Tensor] = None,        # (B*N_det, num_actions)
        # 有効な検出の情報
        det_results: Optional[List[Dict]] = None,           # B 個の dict
        # 中間デコーダ層の出力 (auxiliary loss 用)
        aux_outputs: Optional[List[Dict]] = None,
        # --- OBB / pose (detector が対応している場合のみ) ---
        pred_obb: Optional[torch.Tensor] = None,            # (B, Q, 5)
        pred_angles: Optional[torch.Tensor] = None,         # (B, Q, 2)
        pred_keypoints: Optional[torch.Tensor] = None,      # (B, Q, K, 3)
        pred_polarity: Optional[torch.Tensor] = None,       # (B, Q, 2)
        # --- Behavior branch (evidence 分解つき) ---
        behavior: Optional["BehaviorPrediction"] = None,
        # track-aligned な行動予測を検出順へ戻すための対応表
        track_ids: Optional[List[int]] = None,
        behavior_rows: Optional[List[List[int]]] = None,
        # multi-head 行動: head 名 → (B*N_det, C_head)。検出順 (action_logits と同じ行)。
        action_head_logits: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.pred_logits = pred_logits
        self.pred_boxes = pred_boxes
        self.query_features = query_features
        self.temporal_features = temporal_features
        self.id_logits = id_logits
        self.id_embeddings = id_embeddings
        self.action_logits = action_logits
        self.action_probs = action_probs
        self.det_results = det_results
        self.aux_outputs = aux_outputs
        self.pred_obb = pred_obb
        self.pred_angles = pred_angles
        self.pred_keypoints = pred_keypoints
        self.pred_polarity = pred_polarity
        self.behavior = behavior
        self.track_ids = track_ids
        self.behavior_rows = behavior_rows
        self.action_head_logits = action_head_logits

    @property
    def has_obb(self) -> bool:
        return self.pred_obb is not None

    def explain_detection(self, batch_index: int, det_index: int) -> Dict[str, float]:
        """
        1 検出の行動判断根拠 (evidence 寄与) を返す。

        GUI の「個体をクリック → 判断根拠を表示」に対応する API。
        evidence 分解が無効な構成や、その検出に track 履歴が無い場合は空 dict。

        Returns:
            {"actor": 0.32, "motion": 0.54, "interaction": 1.10, ...}
            値は logit 単位で、足し合わせると最終 logit になる。
        """
        if self.behavior is None or self.behavior_rows is None:
            return {}
        if batch_index >= len(self.behavior_rows):
            return {}
        rows = self.behavior_rows[batch_index]
        if det_index >= len(rows) or rows[det_index] < 0:
            return {}
        return self.behavior.explain(rows[det_index])

    def behavior_gates(self, batch_index: int, det_index: int) -> Dict[str, float]:
        """1 検出の feature gate 重み (重要度) を返す。寄与とは別物。"""
        if self.behavior is None or self.behavior_rows is None:
            return {}
        if batch_index >= len(self.behavior_rows):
            return {}
        rows = self.behavior_rows[batch_index]
        if det_index >= len(rows) or rows[det_index] < 0:
            return {}
        return self.behavior.gate_weights(rows[det_index])

    def iter_detection_batches(self):
        """
        Yield `(slice, det_result)` pairs for the valid detections of each image.
        """
        if self.det_results is None:
            return

        ptr = 0
        for det in self.det_results:
            boxes = det.get("boxes") if isinstance(det, dict) else None
            n_det = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0
            yield slice(ptr, ptr + n_det), det
            ptr += n_det

    def split_flattened_tensor(
        self,
        tensor: Optional[torch.Tensor],
    ) -> List[Optional[torch.Tensor]]:
        """
        Split a flat `(sum_i N_i, ...)` tensor into per-image chunks.
        """
        if self.det_results is None:
            return []
        if tensor is None:
            return [None for _ in self.det_results]
        return [tensor[slc] for slc, _ in self.iter_detection_batches()]


# ---------------------------------------------------------------------------
# YOAKE: Unified Model
# ---------------------------------------------------------------------------

class HTRTDETR(nn.Module):
    """
    YOAKE: Hierarchical Temporal RT-DETR

    学習時は training_stage に応じてモジュールを有効/無効にする。
    推論時は全モジュールを使って統合出力を生成する。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- Spatial Detector ---
        self.detector = build_detector(cfg.detector)

        hidden_dim = self.detector.get_hidden_dim()

        # --- Hierarchical Temporal Module ---
        self.temporal = HierarchicalTemporalModule(cfg.temporal)

        # --- Feature Fusion ---
        # 既定は track 単位。query 単位 (旧挙動) は後方互換の ablation としてのみ残す。
        self.track_aligned_temporal = bool(
            getattr(cfg, "track_aligned_temporal", True)
        )
        self.track_temporal_fusion = TrackTemporalFusion(hidden_dim)
        # 旧 checkpoint / 旧 ablation 用 (パラメータを持たないので構築コストは実質ゼロ)
        self.query_temporal_fusion = QueryTemporalFusion(hidden_dim)
        self.det_adapter = DetectionFeatureAdapter(hidden_dim)
        self.feature_router = MultiHeadFeatureRouter(
            feature_dim=cfg.temporal.output_dim,
            temporal_dim=cfg.temporal.output_dim,
            output_dim=hidden_dim,
        )

        # --- Memory-based ID Head ---
        self.id_head = MemoryIDHead(cfg.id_head)

        # --- Action Head ---
        self.action_head = ActionHead(cfg.action_head)

        # --- Interaction Feature Computer ---
        self.interaction = InteractionFeatureComputer(cfg.action_head)

        # --- Behavior Branch (actor + context 2-stream, evidence 分解) ---
        # 既存 ActionHead は互換のため常に残し、BehaviorBranch を追加で持つ。
        # 行動出力は BehaviorBranch があればそちらを優先し、無ければ ActionHead。
        behavior_cfg = getattr(cfg, "behavior", None)
        self.behavior_enabled = behavior_cfg is not None
        if self.behavior_enabled:
            self.behavior = BehaviorBranch(
                cfg=behavior_cfg,
                num_actions=cfg.action_head.num_actions,
                feature_channels=cfg.detector.fpn.out_channels,
                detector_dim=hidden_dim,
                temporal_dim=cfg.temporal.output_dim,
                num_keypoints=cfg.detector.pose.num_keypoints
                if cfg.detector.pose.enabled else 0,
                explain_cfg=getattr(cfg, "explainability", None),
            )
            assert behavior_cfg is not None  # この分岐に入る条件そのもの
            self.behavior_window = int(behavior_cfg.temporal_window)
        else:
            self.behavior_window = 16

        # --- Geo feature projector (Stage 2/3: geo-only track window training) ---
        temporal_feat_dim = cfg.temporal.feature_dim  # 256
        self.geo_projector = nn.Linear(GEO_FEAT_DIM, temporal_feat_dim)
        # Adapter to produce the "spatial_feat" proxy expected by ActionHead
        self.geo_action_adapter = nn.Linear(
            cfg.temporal.output_dim, cfg.action_head.feature_dim
        )
        # Adapter to produce the feature expected by MemoryIDHead
        self.geo_id_adapter = nn.Linear(
            cfg.temporal.output_dim, cfg.id_head.feature_dim
        )

        # --- Stage 設定 ---
        self.training_stage = cfg.training_stage
        self._configure_stage(cfg.training_stage)

    def _configure_stage(self, stage: int) -> None:
        """
        各 training stage に応じてモジュールの freeze / unfreeze を設定する。
        """
        self.training_stage = stage

        # behavior ブランチは action head と同じ stage で学習する
        # (どちらも「行動を当てる」責務なので freeze の単位を揃える)
        behavior = getattr(self, "behavior", None) if self.behavior_enabled else None

        if stage == 1:
            # Stage 1: detector のみ学習
            unfreeze_module(self.detector)
            freeze_module(self.temporal)
            freeze_module(self.id_head)
            freeze_module(self.action_head)
            if behavior is not None:
                freeze_module(behavior)

        elif stage == 2:
            # Stage 2: temporal + action/behavior head を学習 (detector は freeze)
            freeze_module(self.detector)
            unfreeze_module(self.temporal)
            freeze_module(self.id_head)
            unfreeze_module(self.action_head)
            unfreeze_module(self.interaction)
            if behavior is not None:
                unfreeze_module(behavior)

        elif stage == 3:
            # Stage 3: temporal + ID head を学習 (detector は freeze)
            freeze_module(self.detector)
            unfreeze_module(self.temporal)
            unfreeze_module(self.id_head)
            freeze_module(self.action_head)
            if behavior is not None:
                freeze_module(behavior)

        elif stage == 4:
            # Stage 4: 全モジュールを fine-tune
            unfreeze_module(self)

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def set_stage(self, stage: int) -> None:
        """外部から stage を変更する"""
        self._configure_stage(stage)
        if self.training:
            self.train(True)

    def train(self, mode: bool = True) -> "HTRTDETR":
        """Stage 2/3 では凍結した detector を eval のまま保つ。

        ``freeze_module`` は ``requires_grad`` を落とすだけなので、Trainer が
        ``model.train()`` を呼ぶと凍結した detector の BatchNorm も train モードに
        なり、(1) 各時刻の小さなバッチ (B 枚) の統計で正規化された特徴と検出で
        学習し、(2) running statistics が文脈フレームで書き換わる。推論は eval
        (running statistics) なので学習時と特徴の意味がずれる。凍結は凍結として扱う。
        """
        super().train(mode)
        if mode and self.training_stage in (2, 3):
            self.detector.eval()
        return self

    # ------------------------------------------------------------------ #
    # Forward (シーケンス入力)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: torch.Tensor,   # (B, T, 3, H, W)
        memory_list: Optional[List[IdentityMemory]] = None,  # 推論時のみ使用
        track_ids_per_frame: Optional[List[List[Sequence[int]]]] = None,
        keyframe_targets: Optional[List[Dict]] = None,
    ) -> HTRTDETROutput:
        """
        窓単位の forward。

        データフロー (原則 3: 時間モデルは query 単位ではなく track 単位):

            frames → detector (各フレーム 1 回) → detections
                   → tracking → track_id
                   → track-aligned history → HTM
                   → behavior branch (actor + context + kinematics + interaction)

        Args:
            images: (B, T, 3, H, W) — シーケンス入力
            memory_list: バッチ内各シーケンスの IdentityMemory (推論時)
            track_ids_per_frame: teacher forcing 用の track_id。
                ``track_ids_per_frame[b][t]`` がバッチ要素 b・フレーム t の
                各検出に対応する track_id。None なら内部 tracker が付与する。
            keyframe_targets: キーフレーム方式の学習 (Stage 2/4) で、最終フレーム
                (= キーフレーム) の GT。与えると、キーフレームの検出集合を**ラベル
                された個体そのもの**に置き換える (:meth:`_anchor_keyframe`)。
                AVA 系の行動検出が GT 箱で行動分類器を学習するのと同じ考え方で、
                detector が見落とした個体のラベルも行動の学習に必ず使われる。
                文脈フレームは従来どおり detector + tracker で扱う。

        Returns:
            HTRTDETROutput
        """
        B, T, C, H, W = images.shape

        # ---------------------------------------------------------------- #
        # Step 1: Frame-wise detection — 各フレームはちょうど 1 回だけ通す
        # ---------------------------------------------------------------- #
        frame_outs: List[DetectionOutput] = [
            self.detector(images[:, t]) for t in range(T)
        ]
        last_out = frame_outs[-1]
        pred_logits = last_out.pred_logits   # (B, Q, C+1)
        pred_boxes = last_out.pred_boxes     # (B, Q, 4)
        query_feats = last_out.query_features  # (B, Q, D)

        # Stage 1: detection のみ
        if self.training_stage == 1:
            return HTRTDETROutput(
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                query_features=query_feats,
                aux_outputs=last_out.aux_outputs,
                pred_obb=last_out.pred_obb,
                pred_angles=last_out.pred_angles,
                pred_keypoints=last_out.pred_keypoints,
                pred_polarity=last_out.pred_polarity,
            )

        # ---------------------------------------------------------------- #
        # Step 2: 各フレームの有効な検出を抽出
        # ---------------------------------------------------------------- #
        thr = self.cfg.detector.score_threshold
        per_frame: List[List[Dict]] = [
            self.det_adapter.extract_valid_detections(
                fo.pred_logits, fo.pred_boxes, fo.query_features,
                score_threshold=thr,
                pred_obb=fo.pred_obb,
                pred_keypoints=fo.pred_keypoints,
                pred_polarity=fo.pred_polarity,
            )
            for fo in frame_outs
        ]
        if keyframe_targets is not None:
            per_frame[-1] = self._anchor_keyframe(last_out, keyframe_targets)
        det_results = per_frame[-1]   # 現フレーム (main output)

        # ---------------------------------------------------------------- #
        # Step 3: Temporal — track 単位 (既定) / query 単位 (旧・非推奨)
        # ---------------------------------------------------------------- #
        if self.track_aligned_temporal:
            aligned = self._build_track_aligned(
                per_frame, B, T, images.device, track_ids_per_frame
            )
            temporal_in = self.track_temporal_fusion.pack_from_cache(
                aligned.features, aligned.mask
            )
            if temporal_in.shape[0] > 0:
                temporal_seq = self.temporal(temporal_in)     # (M, T, out_dim)
                temporal_last = temporal_seq[:, -1]           # (M, out_dim)
            else:
                out_dim = self.cfg.temporal.output_dim
                temporal_seq = temporal_in.new_zeros(0, T, out_dim)
                temporal_last = temporal_in.new_zeros(0, out_dim)
            temporal_features_out = temporal_seq
            # 現フレームの各検出に対応する temporal 特徴を引き当てる
            per_det_temporal = self._gather_by_rows(
                temporal_last, aligned.current_rows, det_results
            )
            behavior_rows = aligned.current_rows
            track_ids_out = aligned.track_ids
        else:
            aligned = None
            behavior_rows = None
            track_ids_out = None
            Q = query_feats.shape[1]
            packed = self.query_temporal_fusion.pack_for_temporal(
                [fo.query_features for fo in frame_outs]
            )
            temporal_feats = self.temporal(packed)            # (B*Q, T, out_dim)
            temporal_4d = self.query_temporal_fusion.unpack_from_temporal(
                temporal_feats, B, Q
            )
            temporal_features_out = temporal_4d
            last_by_query = temporal_4d[:, -1]                # (B, Q, out_dim)
            per_det_temporal = torch.cat([
                last_by_query[b][det_results[b]["query_indices"]]
                for b in range(B)
            ], dim=0) if any(
                d["features"].shape[0] > 0 for d in det_results
            ) else query_feats.new_zeros(0, self.cfg.temporal.output_dim)

        # ---------------------------------------------------------------- #
        # Step 4: 検出特徴を平坦化 (ID / action head 用)
        # ---------------------------------------------------------------- #
        flat = self._flatten_detections(det_results, images.device)
        all_spatial_cat = flat["features"]
        all_boxes_cat = flat["boxes"]

        if all_spatial_cat.shape[0] > 0:
            routed = self.feature_router(all_spatial_cat, per_det_temporal)
            id_feat = routed["id_feat"]
            action_spatial = routed["action_spatial_feat"]
        else:
            hidden_dim = self.detector.get_hidden_dim()
            id_feat = all_spatial_cat.new_zeros(0, hidden_dim)
            action_spatial = all_spatial_cat.new_zeros(0, hidden_dim)

        # ---------------------------------------------------------------- #
        # Step 5: Behavior — actor + context 2-stream / evidence 分解
        # ---------------------------------------------------------------- #
        action_logits: Optional[torch.Tensor] = None
        action_probs: Optional[torch.Tensor] = None
        action_head_logits: Optional[Dict[str, torch.Tensor]] = None
        behavior_pred: Optional[BehaviorPrediction] = None

        if self.training_stage in (2, 4):
            if self.behavior_enabled and self.track_aligned_temporal and aligned is not None:
                behavior_pred = self._run_behavior(
                    aligned, last_out, temporal_last, det_results,
                    frame_outs=frame_outs,
                )
                if behavior_pred is not None and behavior_pred.logits is not None:
                    # track 単位の予測を現フレームの検出順に並べ直す
                    action_logits = self._gather_by_rows(
                        behavior_pred.logits, aligned.current_rows, det_results
                    )
                    if behavior_pred.head_logits:
                        # multi-head: 平坦な logit を並べ直してから head に切り分ける。
                        # head_logits は track 行の順なので、そのまま損失に渡すと
                        # 検出順の GT と行がずれる (docs/KEYFRAME_BEHAVIOR.md §15.6)。
                        names = list(behavior_pred.head_logits)
                        parts = action_logits.split(list(behavior_pred.head_sizes), dim=-1)
                        action_head_logits = dict(zip(names, parts))
                        action_probs = torch.cat(
                            [F.softmax(p, dim=-1) for p in parts], dim=-1
                        )
                    else:
                        action_probs = F.softmax(action_logits, dim=-1)

            if action_logits is None:
                # 旧 ActionHead 経路 (behavior 無効 / query-aligned 時)
                velocities = torch.zeros_like(all_boxes_cat[:, :2])
                int_feat = self.interaction(all_boxes_cat, velocities)
                act_out = self.action_head(action_spatial, per_det_temporal, int_feat)
                action_logits = act_out["action_logits"]
                action_probs = act_out["action_probs"]

        # ---------------------------------------------------------------- #
        # Step 6: ID Head
        # ---------------------------------------------------------------- #
        id_logits = None
        id_embeddings = None

        if self.training_stage in (3, 4):
            id_logits, id_embeddings = self._run_id_head(
                det_results, id_feat, all_boxes_cat, memory_list, images.device, B
            )

        return HTRTDETROutput(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            query_features=query_feats,
            temporal_features=temporal_features_out,
            id_logits=id_logits,
            id_embeddings=id_embeddings,
            action_logits=action_logits,
            action_probs=action_probs,
            det_results=det_results,
            aux_outputs=last_out.aux_outputs,
            pred_obb=last_out.pred_obb,
            pred_angles=last_out.pred_angles,
            pred_keypoints=last_out.pred_keypoints,
            pred_polarity=last_out.pred_polarity,
            behavior=behavior_pred,
            track_ids=track_ids_out,
            behavior_rows=behavior_rows,
            action_head_logits=action_head_logits,
        )

    # ------------------------------------------------------------------ #
    # forward のヘルパー
    # ------------------------------------------------------------------ #

    def _anchor_keyframe(
        self, out: DetectionOutput, targets: List[Dict]
    ) -> List[Dict]:
        """キーフレームの検出を、ラベルされた個体 (GT 箱) に置き換える。

        各 GT 箱について、予測箱が最もよく重なる query を 1 対 1 で選び、その
        query 特徴 (detector が見たもの) と GT の箱を組にする。箱は GT なので
        行動損失の IoU マッチングは必ず成立し、**ラベルを付けた個体はすべて行動の
        学習に使われる**。detector がまだ弱い初期ラウンドでも行動の学習が止まらない。

        回転が付いていない (derived) GT は角度だけ予測値を使う。theta=0 で
        actor crop を切ると、斜めの個体が斜めのまま写るため。
        """
        from ..geometry.obb import aabb_to_obb
        from ..utils.misc import box_iou, cxcywh_to_xyxy, xyxy_to_cxcywh

        device = out.pred_boxes.device
        dtype = out.pred_boxes.dtype
        probs = F.softmax(out.pred_logits, dim=-1)[..., :-1]
        results: List[Dict] = []
        for b in range(out.pred_boxes.shape[0]):
            tgt = targets[b] if b < len(targets) else {}
            gt = tgt.get("boxes")
            if gt is None or gt.shape[0] == 0:
                idx = torch.zeros(0, dtype=torch.long, device=device)
                q_boxes = None
            else:
                gt = gt.to(device=device, dtype=dtype)
                iou = box_iou(cxcywh_to_xyxy(out.pred_boxes[b].detach()), gt)  # (Q, N)
                idx = _one_to_one_queries(iou)
                q_boxes = xyxy_to_cxcywh(gt)
            entry: Dict[str, torch.Tensor] = {
                "boxes": q_boxes if q_boxes is not None else out.pred_boxes[b][idx],
                "scores": (torch.ones(idx.shape[0], device=device, dtype=dtype)
                           if q_boxes is not None else probs[b].max(dim=-1).values[idx]),
                "class_ids": (tgt["class_ids"].to(device).long()
                              if q_boxes is not None and "class_ids" in tgt
                              else torch.zeros(idx.shape[0], dtype=torch.long, device=device)),
                "features": out.query_features[b][idx],
                "query_indices": idx,
            }
            if out.pred_obb is not None:
                if q_boxes is None:
                    entry["obb"] = out.pred_obb[b][idx]
                else:
                    gt_obb = tgt.get("obb")
                    obb = (aabb_to_obb(gt) if gt_obb is None
                           else gt_obb.to(device=device, dtype=dtype).clone())
                    valid = tgt.get("obb_valid")
                    if valid is None and gt_obb is None:
                        valid = torch.zeros(obb.shape[0], dtype=torch.bool, device=device)
                    if valid is not None:
                        valid = valid.to(device).bool()
                        pred_theta = out.pred_obb[b][idx][:, 4].detach().to(dtype)
                        obb[~valid, 4] = pred_theta[~valid]
                    entry["obb"] = obb
            if out.pred_keypoints is not None:
                entry["keypoints"] = out.pred_keypoints[b][idx]
            if out.pred_polarity is not None:
                entry["polarity"] = out.pred_polarity[b][idx]
            results.append(entry)
        return results

    def _det_obb(self, det: Dict) -> torch.Tensor:
        """
        検出 dict から OBB (N, 5) を取り出す。

        OBB detector でない場合は theta=0 の OBB に昇格させる。これにより
        下流 (tracking / behavior) は常に OBB を前提に書ける。
        """
        if "obb" in det:
            return det["obb"]
        return cxcywh_to_obb(det["boxes"])

    def _build_track_aligned(
        self,
        per_frame: List[List[Dict]],
        B: int,
        T: int,
        device: torch.device,
        track_ids_per_frame: Optional[List[List[Sequence[int]]]],
    ):
        """
        窓内の検出列に track_id を付け、track 単位の履歴テンソルへ組み立てる。

        ``track_ids_per_frame`` が与えられればそれを使う (学習時の teacher forcing)。
        無ければ内部 tracker が幾何 + 外観で関連付ける。tracker は窓ごとに新規作成
        するので、この forward は窓に対して自己完結する (streaming 推論は
        ``StreamingInferencer`` が持続的な tracker を保持する)。
        """
        from ..tracking import OnlineTracker

        sequences: List[List[FrameDetections]] = []
        for b in range(B):
            tracker = None
            if track_ids_per_frame is None:
                tracker = OnlineTracker.from_config(
                    self.cfg.tracking,
                    feat_dim=self.detector.get_hidden_dim(),
                    window=T,
                    device=device,
                )
            frames: List[FrameDetections] = []
            for t in range(T):
                det = per_frame[t][b]
                feats = det["features"]
                obb = self._det_obb(det)
                if track_ids_per_frame is not None:
                    tids = list(track_ids_per_frame[b][t])
                else:
                    # tracking は離散的な割当なので勾配は流さない (特徴側には流れる)
                    assert tracker is not None  # track_ids_per_frame is None のとき必ず作る
                    tracked = tracker.update(
                        frame_index=t,
                        det_obb=obb.detach(),
                        det_features=feats.detach(),
                        has_obb="obb" in det,
                    )
                    tids = [0] * feats.shape[0]
                    for obj in tracked:
                        tids[obj.detection_index] = obj.track_id
                frames.append(FrameDetections(feats, obb, tids))
            sequences.append(frames)

        return build_track_aligned_history(
            sequences,
            window=T,
            feat_dim=self.detector.get_hidden_dim(),
            device=device,
            dtype=per_frame[-1][0]["features"].dtype,
        )

    @staticmethod
    def _gather_by_rows(
        source: torch.Tensor,
        rows: List[List[int]],
        det_results: List[Dict],
    ) -> torch.Tensor:
        """
        track 単位のテンソル (M, D) を、現フレームの検出順 (N_total, D) に並べ直す。

        track 履歴が無い検出 (rows が -1) はゼロ行になる。
        """
        flat_rows: List[int] = []
        for b, det in enumerate(det_results):
            n = det["features"].shape[0]
            r = rows[b] if b < len(rows) else []
            for i in range(n):
                flat_rows.append(r[i] if i < len(r) else -1)

        if not flat_rows:
            return source.new_zeros(0, source.shape[-1])

        idx = torch.tensor(flat_rows, dtype=torch.long, device=source.device)
        valid = idx >= 0
        out = source.new_zeros(len(flat_rows), source.shape[-1])
        if valid.any() and source.shape[0] > 0:
            out[valid] = source[idx[valid]]
        return out

    @staticmethod
    def _flatten_detections(
        det_results: List[Dict], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """バッチ内の検出を 1 本のテンソルに平坦化する。"""
        feats, boxes, bidx = [], [], []
        for b, det in enumerate(det_results):
            n = det["features"].shape[0]
            if n == 0:
                continue
            feats.append(det["features"])
            boxes.append(det["boxes"])
            bidx.extend([b] * n)

        if not feats:
            ref = det_results[0]["features"] if det_results else torch.zeros(0, 1)
            return {
                "features": ref.new_zeros(0, ref.shape[-1] if ref.ndim > 1 else 1),
                "boxes": ref.new_zeros(0, 4),
                "batch_index": torch.zeros(0, dtype=torch.long, device=device),
            }
        return {
            "features": torch.cat(feats, dim=0),
            "boxes": torch.cat(boxes, dim=0),
            "batch_index": torch.tensor(bidx, dtype=torch.long, device=device),
        }

    def _run_behavior(
        self,
        aligned,
        last_out: DetectionOutput,
        temporal_last: torch.Tensor,
        det_results: List[Dict],
        frame_outs: Optional[List[DetectionOutput]] = None,
    ) -> Optional[BehaviorPrediction]:
        """
        BehaviorBranch を track 単位で走らせる。

        actor / context crop は detector の最高解像度 FPN レベルから取る
        (P2 有効なら stride 4、無効なら stride 8)。細かい姿勢を見る actor stream に
        とって解像度が効くため。clip tube も同じレベルを、窓の全フレームについて
        使う (フレームごとに 1 回だけ縮約してから tube を切り出す)。
        """
        if aligned.num_tracks == 0:
            return None

        cfg_b = self.cfg.behavior
        spatial = None
        if cfg_b.use_actor or cfg_b.use_context:
            enc = last_out.encoder_features
            if not enc:
                return None
            spatial = enc[0]   # 最高解像度レベル

        detector_feats = None
        if cfg_b.use_detector_feat:
            # track ごとの「最後に観測された」検出特徴 = 履歴の最終フレーム
            detector_feats = aligned.features[:, -1]

        keypoints = None
        if "pose" in self.behavior.group_dims:
            keypoints = self._scatter_to_tracks(
                det_results, aligned, key="keypoints",
                num_rows=aligned.num_tracks,
            )
            if keypoints is None:
                # pose head が無効 / このフレームに検出が無い等でキーポイントが
                # 得られない場合はゼロで埋める (行動推論を止めない)。
                K = self.cfg.detector.pose.num_keypoints
                keypoints = aligned.obb.new_zeros(aligned.num_tracks, K, 3)

        clip_frames = None
        if getattr(self.behavior, "use_clip", False):
            outs = frame_outs if frame_outs else [last_out]
            if any(not fo.encoder_features for fo in outs):
                return None
            clip_frames = [
                self.behavior.clip_encoder.reduce_frame(fo.encoder_features[0])
                for fo in outs
            ]

        return self.behavior(
            obb_history=aligned.obb,
            history_mask=aligned.mask,
            spatial_features=spatial,
            batch_index=aligned.batch_index,
            detector_features=detector_feats,
            temporal_features=temporal_last if cfg_b.use_temporal else None,
            keypoints=keypoints,
            clip_frames=clip_frames,
        )

    @staticmethod
    def _scatter_to_tracks(
        det_results: List[Dict],
        aligned,
        key: str,
        num_rows: int,
    ) -> Optional[torch.Tensor]:
        """
        現フレームの検出ごとの値を track 行へ配置する (検出順 → track 順)。

        現フレームに現れない track の行はゼロのまま残る。pose のように
        「検出にはあるが track 履歴には持っていない」値を behavior へ渡すのに使う。
        """
        parts = [d[key] for d in det_results if key in d and d[key].shape[0] > 0]
        if not parts:
            return None

        sample = parts[0]
        out = sample.new_zeros((num_rows,) + tuple(sample.shape[1:]))
        for b, det in enumerate(det_results):
            if key not in det:
                continue
            rows = aligned.current_rows[b] if b < len(aligned.current_rows) else []
            values = det[key]
            for i in range(min(len(rows), values.shape[0])):
                if rows[i] >= 0:
                    out[rows[i]] = values[i]
        return out

    def _run_id_head(
        self,
        det_results: List[Dict],
        id_feat: torch.Tensor,
        all_boxes_cat: torch.Tensor,
        memory_list: Optional[List[IdentityMemory]],
        device: torch.device,
        B: int,
    ):
        """
        ID head (学習: GT ベース / 推論: memory マッチング)。

        学習分岐がゼロ hidden から GRU を回すのは**意図的**で、書き忘れではない。
        この経路の時間的な文脈は ``self.temporal`` (HTM) が窓全体から作った
        track-aligned な特徴として既に ``id_feat`` に入っている。GRU をここで
        再帰させると同じ履歴を二重に積むことになり、窓ごとに独立という学習の
        前提 (バッチ内のシーケンスは互いに独立) も崩れる。

        フレーム間で持ち越す GRU memory が意味を持つのは推論時だけなので、
        ``IdentityMemory`` を使うのは下の推論分岐 (``forward_inference``) に限る。
        """
        if self.training:
            id_logits_list, id_emb_list = [], []
            ptr = 0
            for b in range(B):
                n = det_results[b]["features"].shape[0]
                if n == 0:
                    continue
                feat_b = id_feat[ptr:ptr + n]
                box_b = all_boxes_cat[ptr:ptr + n]
                vel_b = torch.zeros(n, 2, device=feat_b.device, dtype=feat_b.dtype)
                x_b = self.id_head._build_input(feat_b, box_b, vel_b)
                h = torch.zeros(
                    n, self.cfg.id_head.memory_dim,
                    device=feat_b.device, dtype=feat_b.dtype,
                )
                _, emb = self.id_head.memory_net(x_b, h)
                id_logits_list.append(self.id_head.id_classifier(emb))
                id_emb_list.append(emb)
                ptr += n

            if id_logits_list:
                return torch.cat(id_logits_list, 0), torch.cat(id_emb_list, 0)
            return None, None

        # --- 推論: memory を使って ID を割り当てる ---
        if memory_list is None:
            memory_list = [self.id_head.create_memory(device) for _ in range(B)]
        id_emb_list = []
        ptr = 0
        for b in range(B):
            n = det_results[b]["features"].shape[0]
            feat_b = id_feat[ptr:ptr + n]
            box_b = all_boxes_cat[ptr:ptr + n]
            inf_out = self.id_head.forward_inference(feat_b, box_b, memory_list[b])
            det_results[b]["track_ids"] = inf_out["track_ids"]
            det_results[b]["id_scores"] = inf_out["id_scores"]
            id_emb_list.append(inf_out["embeddings"])
            ptr += n

        if id_emb_list:
            return None, torch.cat(id_emb_list, 0)
        return None, None

    # ------------------------------------------------------------------ #
    # Forward (単一フレーム — Stage 1 用)
    # ------------------------------------------------------------------ #

    def forward_single_frame(self, images: torch.Tensor) -> HTRTDETROutput:
        """
        Stage 1 の学習で使う単フレーム forward。
        images: (B, 3, H, W)
        """
        det_out = self.detector(images)
        return HTRTDETROutput(
            pred_logits=det_out.pred_logits,
            pred_boxes=det_out.pred_boxes,
            query_features=det_out.query_features,
            aux_outputs=det_out.aux_outputs,
            pred_obb=det_out.pred_obb,
            pred_angles=det_out.pred_angles,
            pred_keypoints=det_out.pred_keypoints,
            pred_polarity=det_out.pred_polarity,
        )

    # ------------------------------------------------------------------ #
    # Forward (Geo-feature sequence — Stage 2/3 用)
    # ------------------------------------------------------------------ #

    def forward_geo_sequence(
        self,
        geo_feats: torch.Tensor,  # (B, T, GEO_FEAT_DIM)
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Stage 2/3 で使う geo-feature-only forward。
        視覚的な detector を使わず、bbox 軌跡の geometric / motion features から
        直接 HTM → action / ID heads に通す。

        Args:
            geo_feats: (B, T, GEO_FEAT_DIM) — GeometricFeatureBuilder の出力

        Returns:
            dict with keys:
              'action_logits'   (B, num_actions)  — Stage 2/4 のみ
              'id_logits'       (B, max_ids+1)    — Stage 3/4 のみ
              'id_embeddings'   (B, embedding_dim)
        """
        B, T, _ = geo_feats.shape
        device = geo_feats.device

        # (B, T, GEO_FEAT_DIM) → (B, T, temporal_feat_dim)
        x = self.geo_projector(geo_feats)

        # HTM: expects (B, T, D) or (B*Q, T, D) — here B plays the role of B*Q
        temporal_out = self.temporal(x)         # (B, T, output_dim)
        last_feat = temporal_out[:, -1]         # (B, output_dim) — last time step

        result: Dict[str, Optional[torch.Tensor]] = {
            "action_logits": None,
            "id_logits": None,
            "id_embeddings": None,
        }

        if self.training_stage in (2, 4):
            # spatial proxy: project temporal feat to feature_dim
            spatial_proxy = self.geo_action_adapter(last_feat)  # (B, feature_dim)
            act_out = self.action_head(spatial_proxy, last_feat)
            result["action_logits"] = act_out["action_logits"]   # (B, num_actions)

        if self.training_stage in (3, 4):
            id_feat = self.geo_id_adapter(last_feat)  # (B, feature_dim)
            # geo_feats already encode bbox ([:4]) and velocity ([:,5:7]);
            # pass them as proxies so MemoryIDHead._build_input gets correct dims
            # geo_feats[:, 0:4] = [cx_n, cy_n, w_n, h_n] ≈ cxcywh bbox proxy
            # geo_feats[:, 5:7] = [dx, dy] velocity proxy (last frame)
            bbox_proxy = geo_feats[:, -1, :4]           # (B, 4)
            vel_proxy  = geo_feats[:, -1, 5:7]          # (B, 2)
            id_in = self.id_head._build_input(id_feat, bbox_proxy, vel_proxy)
            # ゼロ hidden は意図的。各サンプルは既に T フレームぶんの窓であり、
            # その時間情報は self.temporal が last_feat に畳み込んでいる。ここで
            # さらに GRU を再帰させると履歴を二重に数えることになるうえ、
            # バッチ内のサンプルは別 track / 別動画なので引き継ぐ相手がいない。
            # フレームをまたぐ GRU memory は推論時 (forward_inference) の話。
            h0 = torch.zeros(B, self.cfg.id_head.memory_dim, device=device)
            _, emb = self.id_head.memory_net(id_in, h0)
            logits = self.id_head.id_classifier(emb)
            result["id_logits"] = logits      # (B, max_ids+1)
            result["id_embeddings"] = emb     # (B, embedding_dim)

        return result

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> Dict[str, int]:
        """各モジュールのパラメータ数を返す"""
        return {
            "detector": sum(p.numel() for p in self.detector.parameters()),
            "temporal": sum(p.numel() for p in self.temporal.parameters()),
            "id_head": sum(p.numel() for p in self.id_head.parameters()),
            "action_head": sum(p.numel() for p in self.action_head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def create_memory_list(
        self, batch_size: int, device: torch.device
    ) -> List[IdentityMemory]:
        """推論時の memory リストを作成する"""
        return [self.id_head.create_memory(device) for _ in range(batch_size)]
