"""
memory_id_head.py — Memory-based ID Head

設計方針:
- 各個体の appearance / geometry / motion / action 情報を GRU memory に蓄積
- 新しい detection query との類似度を cosine similarity で計算
- 既存 ID の再同定 / 新規 ID の割り当てを行う
- IoU-based assignment (Hungarian matching) と similarity matching を組み合わせる
- 将来的に metric learning loss (triplet / contrastive) を追加しやすい構造

ID 割り当てアルゴリズム:
1. 現フレームの detection feature から ID embedding を計算
2. memory 内の各 ID の embedding との cosine similarity を計算
3. IoU (bbox overlap) と similarity の組み合わせでコスト行列を構築
4. Hungarian algorithm で最適割り当て
5. threshold 以下の場合は新規 ID を割り当て
6. unmatched な memory は TTL を減らし、0 になれば削除
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config.config import MemoryIDConfig
from ...utils.misc import box_iou, cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# ID Embedding Network
# ---------------------------------------------------------------------------

class IDEmbeddingNet(nn.Module):
    """
    Detection feature (query feature + temporal feature) から
    ID embedding を計算するネットワーク。
    """

    def __init__(self, feature_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, feature_dim)
        Returns:
            embedding: (N, embedding_dim) — L2 normalized
        """
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Identity Memory
# ---------------------------------------------------------------------------

class IdentityMemory:
    """
    1 動画シーケンスのすべての ID の memory を管理するクラス。

    各 ID に対して以下を保持:
    - gru_hidden: GRU の hidden state (memory の実体)
    - last_bbox: 最後に見た bbox [cx, cy, w, h]
    - last_embedding: 最後の ID embedding
    - velocity: フレーム間の移動速度 [dx, dy]
    - ttl: あと何フレーム残っているか
    - frame_count: 何フレーム観測されたか
    """

    def __init__(
        self,
        memory_dim: int,
        embedding_dim: int,
        max_ids: int,
        memory_ttl: int,
        device: torch.device,
    ):
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.max_ids = max_ids
        self.memory_ttl = memory_ttl
        self.device = device

        # track_id → state のマッピング
        self.tracks: Dict[int, Dict] = {}
        self._next_id = 0

    def get_active_ids(self, species_class: Optional[int] = None) -> List[int]:
        """TTL > 0 の active な ID リストを返す。
        species_class を指定した場合はその種のみを返す。"""
        if species_class is None:
            return [tid for tid, s in self.tracks.items() if s["ttl"] > 0]
        return [
            tid for tid, s in self.tracks.items()
            if s["ttl"] > 0 and s.get("species_class", 0) == species_class
        ]

    def get_embeddings(self, ids: List[int]) -> Optional[torch.Tensor]:
        """指定 ID の embedding を (N, D) で返す"""
        if not ids:
            return None
        embs = [self.tracks[tid]["last_embedding"] for tid in ids]
        return torch.stack(embs, dim=0)  # (N, D)

    def get_bboxes(self, ids: List[int]) -> Optional[torch.Tensor]:
        """指定 ID の bbox を (N, 4) で返す"""
        if not ids:
            return None
        boxes = [self.tracks[tid]["last_bbox"] for tid in ids]
        return torch.stack(boxes, dim=0)  # (N, 4)

    def get_hidden_states(self, ids: List[int]) -> Optional[torch.Tensor]:
        """指定 ID の GRU hidden states を (N, memory_dim) で返す"""
        if not ids:
            return None
        hs = [self.tracks[tid]["gru_hidden"] for tid in ids]
        return torch.stack(hs, dim=0)  # (N, memory_dim)

    def update_track(
        self,
        track_id: int,
        embedding: torch.Tensor,       # (embedding_dim,)
        bbox: torch.Tensor,            # (4,) [cx, cy, w, h]
        gru_hidden: torch.Tensor,      # (memory_dim,)
        action_summary: Optional[torch.Tensor] = None,
    ) -> None:
        """既存 track を更新する"""
        if track_id not in self.tracks:
            raise KeyError(f"track_id {track_id} not in memory")
        s = self.tracks[track_id]

        # velocity 計算
        prev_bbox = s["last_bbox"]
        velocity = bbox[:2] - prev_bbox[:2]  # [dx, dy]

        s["last_embedding"] = embedding.detach()
        s["last_bbox"] = bbox.detach()
        s["gru_hidden"] = gru_hidden.detach()
        s["velocity"] = velocity.detach()
        s["ttl"] = self.memory_ttl
        s["frame_count"] += 1
        if action_summary is not None:
            s["action_summary"] = action_summary.detach()

    def add_new_track(
        self,
        embedding: torch.Tensor,
        bbox: torch.Tensor,
        gru_hidden: torch.Tensor,
        species_class: int = 0,
    ) -> int:
        """新規 track を追加し、track_id を返す。
        species_class: 種クラスID（単一種の場合は 0 のまま）"""
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {
            "last_embedding": embedding.detach(),
            "last_bbox": bbox.detach(),
            "gru_hidden": gru_hidden.detach(),
            "velocity": torch.zeros(2, device=self.device),
            "ttl": self.memory_ttl,
            "frame_count": 1,
            "action_summary": None,
            "species_class": species_class,
        }
        return tid

    def decrement_ttl(self, matched_ids: List[int]) -> None:
        """matched されなかった track の TTL を減らす"""
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]["ttl"] -= 1

    def remove_dead_tracks(self) -> None:
        """TTL <= 0 の track を削除する"""
        dead = [tid for tid, s in self.tracks.items() if s["ttl"] <= 0]
        for tid in dead:
            del self.tracks[tid]

    def reset(self) -> None:
        """メモリをリセットする (新しいシーケンス開始時)"""
        self.tracks.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Memory Update Network (GRU ベース)
# ---------------------------------------------------------------------------

class MemoryUpdateNet(nn.Module):
    """
    GRU を使って ID の memory を更新するネットワーク。
    入力: 現在の detection feature
    状態: 過去の hidden state
    出力: 更新された hidden state + ID embedding
    """

    def __init__(
        self,
        input_dim: int,   # feature_dim + optional geometry/motion dim
        memory_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        # 入力を memory_dim に射影
        self.input_proj = nn.Linear(input_dim, memory_dim)

        # GRU cell
        self.gru_cell = nn.GRUCell(memory_dim, memory_dim)

        # GRU hidden → ID embedding
        self.embedding_head = IDEmbeddingNet(memory_dim, embedding_dim)

    def forward(
        self,
        features: torch.Tensor,        # (N, input_dim)
        hidden_states: torch.Tensor,   # (N, memory_dim) — 0 for new tracks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            new_hidden: (N, memory_dim)
            embedding: (N, embedding_dim) — L2 normalized
        """
        x = self.input_proj(features)          # (N, memory_dim)
        new_hidden = self.gru_cell(x, hidden_states)  # (N, memory_dim)
        embedding = self.embedding_head(new_hidden)    # (N, embedding_dim)
        return new_hidden, embedding


# ---------------------------------------------------------------------------
# Interaction Feature Extractor (Optional)
# ---------------------------------------------------------------------------

class InteractionFeatureExtractor(nn.Module):
    """
    bbox から個体間の interaction feature を計算する。
    使用する特徴:
    - nearest neighbor との距離
    - relative angle
    - relative velocity
    - overlap (IoU)
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()
        # 距離 + angle + velocity_x + velocity_y + iou = 5 次元
        raw_dim = 5
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        bboxes: torch.Tensor,      # (N, 4) [cx, cy, w, h] normalized
        velocities: torch.Tensor,  # (N, 2) [dx, dy]
    ) -> torch.Tensor:
        """
        Returns: (N, output_dim)
        """
        N = bboxes.shape[0]
        if N == 0:
            return torch.zeros(0, self.proj[0].out_features, device=bboxes.device)
        if N == 1:
            return self.proj(torch.zeros(1, 5, device=bboxes.device))

        centers = bboxes[:, :2]  # (N, 2)

        # Pairwise distances
        dists = torch.cdist(centers, centers)  # (N, N)
        # 自分自身は除外して最近傍距離を取る
        dists.fill_diagonal_(float("inf"))
        nn_dist, nn_idx = dists.min(dim=1)  # (N,)

        # Relative angle to nearest neighbor
        nn_centers = centers[nn_idx]  # (N, 2)
        delta = nn_centers - centers  # (N, 2)
        angle = torch.atan2(delta[:, 1], delta[:, 0])  # (N,)

        # Relative velocity
        rel_vel = velocities - velocities[nn_idx]  # (N, 2)

        # IoU with nearest neighbor
        # cxcywh → xyxy に変換
        boxes_xyxy = torch.cat([
            centers - bboxes[:, 2:] / 2,
            centers + bboxes[:, 2:] / 2,
        ], dim=1)  # (N, 4)
        nn_boxes = boxes_xyxy[nn_idx]  # (N, 4)
        iou_vals = box_iou(boxes_xyxy, nn_boxes).diagonal()  # (N,) ... 厳密には異なるが近似

        # 特徴を結合
        feats = torch.stack([
            nn_dist,
            angle,
            rel_vel[:, 0],
            rel_vel[:, 1],
            iou_vals,
        ], dim=1)  # (N, 5)

        # NaN / Inf をゼロに
        feats = feats.nan_to_num(0.0).clamp(-10.0, 10.0)

        return self.proj(feats)  # (N, output_dim)


# ---------------------------------------------------------------------------
# Memory-based ID Head (メインクラス)
# ---------------------------------------------------------------------------

class MemoryIDHead(nn.Module):
    """
    Memory-based ID Head

    各フレームの detection feature に対して:
    1. geometry/motion feature を追加して入力を構築
    2. MemoryUpdateNet で GRU memory を更新
    3. 既存 memory との similarity から ID を決定 (推論時)
       もしくは GT ID を使ってクロスエントロピー損失を計算 (学習時)

    学習時は track_id の GT が与えられることを前提にする。
    推論時は Hungarian matching で ID を割り当てる。
    """

    def __init__(self, cfg: MemoryIDConfig):
        super().__init__()
        self.cfg = cfg

        # 入力次元の計算
        input_dim = cfg.feature_dim
        if cfg.use_geometry:
            input_dim += 4   # bbox [cx, cy, w, h]
        if cfg.use_motion:
            input_dim += 2   # velocity [dx, dy]

        # Memory update network
        self.memory_net = MemoryUpdateNet(
            input_dim=input_dim,
            memory_dim=cfg.memory_dim,
            embedding_dim=cfg.embedding_dim,
        )

        # ID classification head (for training with GT IDs)
        # max_ids 個の track + 1 (new/unknown) を分類
        self.id_classifier = nn.Linear(cfg.embedding_dim, cfg.max_ids + 1)

        # Optional: interaction feature extractor
        self.interaction = InteractionFeatureExtractor(output_dim=64)

        # メモリはインスタンスに属さず、外部から注入する設計にすること
        # (batch 内の複数シーケンスはそれぞれ独立した memory を持つ)

    def _build_input(
        self,
        features: torch.Tensor,     # (N, feature_dim)
        bboxes: Optional[torch.Tensor] = None,    # (N, 4) [cx, cy, w, h]
        velocities: Optional[torch.Tensor] = None, # (N, 2)
    ) -> torch.Tensor:
        """geometry / motion を feature に concat して入力を構築する"""
        parts = [features]
        if self.cfg.use_geometry and bboxes is not None:
            parts.append(bboxes)
        if self.cfg.use_motion and velocities is not None:
            parts.append(velocities)
        return torch.cat(parts, dim=-1)

    def forward_train(
        self,
        features: torch.Tensor,         # (N, feature_dim)
        bboxes: torch.Tensor,            # (N, 4) [cx, cy, w, h]
        gt_track_ids: torch.Tensor,      # (N,) GT track IDs
        memory: IdentityMemory,
    ) -> Dict[str, torch.Tensor]:
        """
        学習時の forward。

        Returns:
            id_logits: (N, max_ids+1) — ID 分類スコア
            embeddings: (N, embedding_dim) — ID embedding
            new_hidden: (N, memory_dim) — 更新された hidden state
        """
        N = features.shape[0]
        device = features.device

        # velocity を memory から計算
        velocities = torch.zeros(N, 2, device=device)
        hidden_states = torch.zeros(N, self.cfg.memory_dim, device=device)

        for i, tid in enumerate(gt_track_ids.tolist()):
            if tid >= 0 and tid in memory.tracks:
                velocities[i] = memory.tracks[tid]["velocity"]
                hidden_states[i] = memory.tracks[tid]["gru_hidden"]

        # 入力構築
        x = self._build_input(features, bboxes, velocities)

        # GRU memory 更新
        new_hidden, embeddings = self.memory_net(x, hidden_states)

        # ID 分類
        id_logits = self.id_classifier(embeddings)  # (N, max_ids+1)

        return {
            "id_logits": id_logits,
            "embeddings": embeddings,
            "new_hidden": new_hidden,
        }

    @torch.no_grad()
    def forward_inference(
        self,
        features: torch.Tensor,                      # (N, feature_dim)
        bboxes: torch.Tensor,                        # (N, 4) [cx, cy, w, h]
        memory: IdentityMemory,
        species_classes: Optional[torch.Tensor] = None,  # (N,) int, 種クラスID
    ) -> Dict[str, torch.Tensor]:
        """
        推論時の forward。
        memory と照合して track_id を割り当て、memory を更新する。

        species_classes が指定され use_species_separated_pools=True の場合、
        同種の memory のみとマッチングを行う（異種間の ID 混同を防ぐ）。
        species_classes=None または use_species_separated_pools=False の場合は
        従来の一括マッチングと同一の挙動になる。

        Returns:
            track_ids: (N,) 割り当てた track ID
            id_scores: (N,) 最高 similarity スコア
            embeddings: (N, embedding_dim)
        """
        N = features.shape[0]
        device = features.device

        if N == 0:
            return {
                "track_ids": torch.zeros(0, dtype=torch.long, device=device),
                "id_scores": torch.zeros(0, device=device),
                "embeddings": torch.zeros(0, self.cfg.embedding_dim, device=device),
            }

        # velocity はゼロ初期化（memory から取得する拡張は forward_train と共通化可能）
        velocities = torch.zeros(N, 2, device=device)
        x = self._build_input(features, bboxes, velocities)
        hidden_states = torch.zeros(N, self.cfg.memory_dim, device=device)

        # GRU で embedding を計算
        new_hidden, embeddings = self.memory_net(x, hidden_states)

        track_ids = torch.full((N,), -1, dtype=torch.long, device=device)
        id_scores = torch.zeros(N, device=device)

        use_separation = (
            self.cfg.use_species_separated_pools
            and species_classes is not None
        )

        if use_separation:
            # --- 種ごとに独立してマッチング ---
            for sp in species_classes.unique().tolist():
                sp = int(sp)
                det_indices = (species_classes == sp).nonzero(as_tuple=True)[0]
                sp_active_ids = memory.get_active_ids(species_class=sp)
                if not sp_active_ids:
                    continue  # この種の memory がなければスキップ（全て新規IDに）

                sp_embs = embeddings[det_indices]          # (n_sp, D)
                sp_bboxes = bboxes[det_indices]            # (n_sp, 4)
                mem_embs = memory.get_embeddings(sp_active_ids)  # (M_sp, D)
                mem_boxes = memory.get_bboxes(sp_active_ids)     # (M_sp, 4)

                sim = torch.mm(sp_embs, mem_embs.t())            # (n_sp, M_sp)
                det_xyxy = cxcywh_to_xyxy(sp_bboxes)
                mem_xyxy = cxcywh_to_xyxy(mem_boxes)
                iou_mat = box_iou(det_xyxy, mem_xyxy)            # (n_sp, M_sp)
                cost_mat = -(0.5 * sim + 0.5 * iou_mat)

                assigned = self._hungarian_match(cost_mat, threshold=-self.cfg.new_id_threshold)
                for local_i, mem_j in assigned:
                    global_i = det_indices[local_i].item()
                    if mem_j >= 0:
                        track_ids[global_i] = sp_active_ids[mem_j]
                        id_scores[global_i] = sim[local_i, mem_j]
        else:
            # --- 従来の一括マッチング (backward compatible) ---
            active_ids = memory.get_active_ids()
            if active_ids:
                mem_embs = memory.get_embeddings(active_ids)  # (M, D)
                mem_boxes = memory.get_bboxes(active_ids)     # (M, 4)
                sim = torch.mm(embeddings, mem_embs.t())      # (N, M)
                det_xyxy = cxcywh_to_xyxy(bboxes)
                mem_xyxy = cxcywh_to_xyxy(mem_boxes)
                iou_mat = box_iou(det_xyxy, mem_xyxy)         # (N, M)
                cost_mat = -(0.5 * sim + 0.5 * iou_mat)
                assigned = self._hungarian_match(cost_mat, threshold=-self.cfg.new_id_threshold)
                for det_i, mem_j in assigned:
                    if mem_j >= 0:
                        track_ids[det_i] = active_ids[mem_j]
                        id_scores[det_i] = sim[det_i, mem_j]

        # 未割り当ての detection に新規 ID を割り当て
        matched_ids = []
        for i in range(N):
            sp_class = int(species_classes[i].item()) if species_classes is not None else 0
            if track_ids[i].item() == -1:
                new_id = memory.add_new_track(
                    embeddings[i], bboxes[i], new_hidden[i],
                    species_class=sp_class,
                )
                track_ids[i] = new_id
                id_scores[i] = 1.0
                matched_ids.append(new_id)
            else:
                tid = track_ids[i].item()
                matched_ids.append(tid)
                memory.update_track(tid, embeddings[i], bboxes[i], new_hidden[i])

        # TTL 更新
        memory.decrement_ttl(matched_ids)
        memory.remove_dead_tracks()

        return {
            "track_ids": track_ids,
            "id_scores": id_scores,
            "embeddings": embeddings,
        }

    @staticmethod
    def _hungarian_match(
        cost_mat: torch.Tensor,
        threshold: float = -0.5,
    ) -> List[Tuple[int, int]]:
        """
        Hungarian algorithm で最適割り当てを行う。
        cost > threshold の割り当ては無効 (新規 ID として扱う)。

        Returns: List[(det_idx, mem_idx)] — mem_idx=-1 は新規 ID
        """
        N, M = cost_mat.shape
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(
                cost_mat.cpu().numpy()
            )
        except ImportError:
            # scipy がない場合のグリーディな fallback
            row_ind, col_ind = _greedy_match(cost_mat)

        result = [(-1, -1)] * N
        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] <= threshold:
                result[r] = (r, c)
            else:
                result[r] = (r, -1)  # 新規 ID

        return [(r, c) for r, (r, c) in enumerate(result)]

    def create_memory(self, device: torch.device) -> IdentityMemory:
        """新しい IdentityMemory を生成するファクトリメソッド"""
        return IdentityMemory(
            memory_dim=self.cfg.memory_dim,
            embedding_dim=self.cfg.embedding_dim,
            max_ids=self.cfg.max_ids,
            memory_ttl=self.cfg.memory_ttl,
            device=device,
        )


def _greedy_match(
    cost_mat: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    """scipy なしのグリーディマッチング (fallback)"""
    N, M = cost_mat.shape
    used_cols = set()
    row_ind, col_ind = [], []
    for r in range(N):
        best_c, best_cost = -1, float("inf")
        for c in range(M):
            if c not in used_cols and cost_mat[r, c] < best_cost:
                best_c = c
                best_cost = cost_mat[r, c].item()
        if best_c >= 0:
            row_ind.append(r)
            col_ind.append(best_c)
            used_cols.add(best_c)
    return row_ind, col_ind
