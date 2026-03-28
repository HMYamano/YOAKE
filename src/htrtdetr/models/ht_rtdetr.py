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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.config import ModelConfig
from .detector import RTDETRDetector, DetectionOutput, build_detector
from .temporal import HierarchicalTemporalModule
from .id_head import MemoryIDHead, IdentityMemory
from .action_head import ActionHead, InteractionFeatureComputer
from .fusion import QueryTemporalFusion, DetectionFeatureAdapter, MultiHeadFeatureRouter
from ..utils.misc import freeze_module, unfreeze_module, cxcywh_to_xyxy
from ..data.feature_builder import GEO_FEAT_DIM


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

        if stage == 1:
            # Stage 1: detector のみ学習
            unfreeze_module(self.detector)
            freeze_module(self.temporal)
            freeze_module(self.id_head)
            freeze_module(self.action_head)

        elif stage == 2:
            # Stage 2: temporal + action head を学習 (detector は freeze)
            freeze_module(self.detector)
            unfreeze_module(self.temporal)
            freeze_module(self.id_head)
            unfreeze_module(self.action_head)
            unfreeze_module(self.interaction)

        elif stage == 3:
            # Stage 3: temporal + ID head を学習 (detector は freeze)
            freeze_module(self.detector)
            unfreeze_module(self.temporal)
            unfreeze_module(self.id_head)
            freeze_module(self.action_head)

        elif stage == 4:
            # Stage 4: 全モジュールを fine-tune
            unfreeze_module(self)

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def set_stage(self, stage: int) -> None:
        """外部から stage を変更する"""
        self._configure_stage(stage)

    # ------------------------------------------------------------------ #
    # Forward (シーケンス入力)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: torch.Tensor,   # (B, T, 3, H, W)
        memory_list: Optional[List[IdentityMemory]] = None,  # 推論時のみ使用
    ) -> HTRTDETROutput:
        """
        Args:
            images: (B, T, 3, H, W) — シーケンス入力
            memory_list: バッチ内各シーケンスの IdentityMemory (推論時)

        Returns:
            HTRTDETROutput
        """
        B, T, C, H, W = images.shape

        # ---------------------------------------------------------------- #
        # Step 1: Frame-wise detection
        # ---------------------------------------------------------------- #
        all_pred_logits = []   # List[(B, Q, C+1)], length T
        all_pred_boxes = []    # List[(B, Q, 4)], length T
        all_query_feats = []   # List[(B, Q, D)], length T

        for t in range(T):
            det_out = self.detector(images[:, t])  # DetectionOutput
            all_pred_logits.append(det_out.pred_logits)
            all_pred_boxes.append(det_out.pred_boxes)
            all_query_feats.append(det_out.query_features)

        # 最終フレームを main output に使う
        pred_logits = all_pred_logits[-1]   # (B, Q, C+1)
        pred_boxes = all_pred_boxes[-1]     # (B, Q, 4)
        query_feats = all_query_feats[-1]   # (B, Q, D)

        # Stage 1: detection のみ
        if self.training_stage == 1:
            return HTRTDETROutput(
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                query_features=query_feats,
            )

        # ---------------------------------------------------------------- #
        # Step 2: Hierarchical Temporal Module
        # ---------------------------------------------------------------- #
        # query features を temporal 次元に変換: (B*Q, T, D)
        Q = query_feats.shape[1]
        packed = self.query_temporal_fusion.pack_for_temporal(all_query_feats)
        # packed: (B*Q, T, D)

        temporal_feats = self.temporal(packed)  # (B*Q, T, out_dim)

        # (B, T, Q, out_dim) に戻す
        temporal_feats_4d = self.query_temporal_fusion.unpack_from_temporal(
            temporal_feats, B, Q
        )

        # 最終フレームの temporal feature: (B, Q, out_dim)
        temporal_feats_last = temporal_feats_4d[:, -1]  # (B, Q, out_dim)

        # ---------------------------------------------------------------- #
        # Step 3: 有効な検出を抽出
        # ---------------------------------------------------------------- #
        det_results = self.det_adapter.extract_valid_detections(
            pred_logits, pred_boxes, query_feats,
            score_threshold=self.cfg.detector.score_threshold,
        )

        # ---------------------------------------------------------------- #
        # Step 4: Feature routing (ID head / Action head 用)
        # ---------------------------------------------------------------- #
        # 各バッチの valid detection に対して feature を集める
        all_spatial = []
        all_temporal = []
        all_boxes = []
        all_batch_idx = []

        for b in range(B):
            det = det_results[b]
            N_b = det["features"].shape[0]
            if N_b == 0:
                continue
            q_idx = det["query_indices"]

            # spatial feature (query feature)
            sf = det["features"]                      # (N_b, D)
            # temporal feature (対応する query の temporal)
            tf = temporal_feats_last[b][q_idx]        # (N_b, out_dim)

            all_spatial.append(sf)
            all_temporal.append(tf)
            all_boxes.append(det["boxes"])
            all_batch_idx.extend([b] * N_b)

        # concat して一括処理
        if all_spatial:
            all_spatial_cat = torch.cat(all_spatial, dim=0)    # (N_total, D)
            all_temporal_cat = torch.cat(all_temporal, dim=0)  # (N_total, out_dim)
            all_boxes_cat = torch.cat(all_boxes, dim=0)         # (N_total, 4)
        else:
            D = query_feats.shape[-1]
            out_dim = temporal_feats_last.shape[-1]
            all_spatial_cat = torch.zeros(0, D, device=images.device)
            all_temporal_cat = torch.zeros(0, out_dim, device=images.device)
            all_boxes_cat = torch.zeros(0, 4, device=images.device)

        # Feature routing
        if all_spatial_cat.shape[0] > 0:
            routed = self.feature_router(all_spatial_cat, all_temporal_cat)
            id_feat = routed["id_feat"]             # (N_total, hidden_dim)
            action_spatial = routed["action_spatial_feat"]  # (N_total, hidden_dim)
        else:
            hidden_dim = self.detector.get_hidden_dim()
            id_feat = torch.zeros(0, hidden_dim, device=images.device)
            action_spatial = torch.zeros(0, hidden_dim, device=images.device)

        # ---------------------------------------------------------------- #
        # Step 5: Action Head
        # ---------------------------------------------------------------- #
        action_logits = None
        action_probs = None

        if self.training_stage in (2, 4):
            # interaction feature
            velocities = torch.zeros_like(all_boxes_cat[:, :2])  # (N_total, 2)
            int_feat = self.interaction(all_boxes_cat, velocities)

            act_out = self.action_head(action_spatial, all_temporal_cat, int_feat)
            action_logits = act_out["action_logits"]  # (N_total, num_actions)
            action_probs = act_out["action_probs"]

        # ---------------------------------------------------------------- #
        # Step 6: ID Head
        # ---------------------------------------------------------------- #
        id_logits = None
        id_embeddings = None

        if self.training_stage in (3, 4):
            if self.training:
                # 学習時は GT track_id を使う (損失計算側で処理)
                # ここでは embedding のみ計算
                # (実際の ID 損失は trainer で計算する)
                id_logits_list = []
                id_emb_list = []
                ptr = 0
                for b in range(B):
                    N_b = det_results[b]["features"].shape[0]
                    if N_b == 0:
                        continue
                    feat_b = id_feat[ptr:ptr + N_b]
                    box_b = all_boxes_cat[ptr:ptr + N_b]
                    # geometry/motion を concat して input_proj の次元に合わせる
                    vel_b = torch.zeros(N_b, 2, device=feat_b.device)
                    x_b = self.id_head._build_input(feat_b, box_b, vel_b)
                    h = torch.zeros(N_b, self.cfg.id_head.memory_dim, device=feat_b.device)
                    _, emb = self.id_head.memory_net(x_b, h)
                    logits = self.id_head.id_classifier(emb)
                    id_logits_list.append(logits)
                    id_emb_list.append(emb)
                    ptr += N_b

                if id_logits_list:
                    id_logits = torch.cat(id_logits_list, dim=0)
                    id_embeddings = torch.cat(id_emb_list, dim=0)
            else:
                # 推論時は memory を使って ID を割り当て
                if memory_list is None:
                    memory_list = [
                        self.id_head.create_memory(images.device) for _ in range(B)
                    ]
                id_emb_list = []
                ptr = 0
                for b in range(B):
                    N_b = det_results[b]["features"].shape[0]
                    feat_b = id_feat[ptr:ptr + N_b]
                    box_b = all_boxes_cat[ptr:ptr + N_b]
                    inf_out = self.id_head.forward_inference(
                        feat_b, box_b, memory_list[b]
                    )
                    # track_ids を det_results に書き込む
                    det_results[b]["track_ids"] = inf_out["track_ids"]
                    det_results[b]["id_scores"] = inf_out["id_scores"]
                    id_emb_list.append(inf_out["embeddings"])
                    ptr += N_b

                if id_emb_list:
                    id_embeddings = torch.cat(id_emb_list, dim=0)

        return HTRTDETROutput(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            query_features=query_feats,
            temporal_features=temporal_feats_4d,
            id_logits=id_logits,
            id_embeddings=id_embeddings,
            action_logits=action_logits,
            action_probs=action_probs,
            det_results=det_results,
        )

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
