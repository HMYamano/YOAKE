"""
rtdetr_wrapper.py — RT-DETR 風 Detector 実装

設計方針:
- RT-DETR (arXiv:2304.08069) の思想をベースに、
  from-scratch で自己完結した実装を提供する
- Backbone: torchvision の ResNet (BSDライセンス、商用利用可)
- Neck: シンプルな FPN
- Encoder: AIFI 風の intra-scale attention + cross-scale feature fusion
- Decoder: Deformable 風の cross-attention (簡略化して標準 attention で実装)
- 将来的に torchvision の DETR や RT-DETR 公式実装への差し替えも容易

注: 本実装は研究プロトタイプとして RT-DETR の key idea を再現したものであり、
    論文著者の公式実装ではありません。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights

from .base import BaseDetector, DetectionOutput
from ...config.config import DetectorConfig
from ...geometry.obb import decode_angle, obb_to_cxcywh


# ---------------------------------------------------------------------------
# Position Encoding
# ---------------------------------------------------------------------------

class SinePositionEncoding2D(nn.Module):
    """2D Sinusoidal Position Encoding (DETR 原著から)"""

    def __init__(self, hidden_dim: int = 256, temperature: float = 10000.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim は偶数でなければなりません"
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            mask: (B, H, W) — True: padded position (optional)
        Returns:
            pos: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        if mask is None:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        not_mask = ~mask  # (B, H, W)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # (B, H, W)

        # 正規化
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * 2 * math.pi

        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, C//2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, C//2)

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4)

        pos_x = pos_x.flatten(3)  # (B, H, W, C//2)
        pos_y = pos_y.flatten(3)  # (B, H, W, C//2)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)
        return pos


# ---------------------------------------------------------------------------
# FPN (Feature Pyramid Network)
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """
    シンプルな top-down FPN。
    backbone の C3, C4, C5 を受け取り、同一チャネル数の feature map を返す。
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # lateral conv (1x1 で統一チャネルに変換)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        # output conv (3x3 で詳細を整える)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [C3, C4, C5] — stride 8, 16, 32
        Returns:
            [P3, P4, P5] — 同じ stride で out_channels チャネル
        """
        # lateral
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # output
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


# ---------------------------------------------------------------------------
# AIFI: Attention-based Intra-scale Feature Interaction
# ---------------------------------------------------------------------------

class AIFI(nn.Module):
    """
    RT-DETR の AIFI モジュール。
    単一スケールで self-attention を適用し、特徴を強化する。
    計算量削減のため C5 (最低解像度) のみに適用する。
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, d_model),
        )
        self.pos_enc = SinePositionEncoding2D(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        pos = self.pos_enc(x)  # (B, C, H, W)

        # flatten: (B, H*W, C)
        x_flat = (x + pos).flatten(2).permute(0, 2, 1)

        # self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        # reshape: (B, C, H, W)
        return x_flat.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# DETR-like Decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """DETR デコーダの 1 層"""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        # Cross-attention: queries → encoder features
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,      # (B, Q, C)
        memory: torch.Tensor,       # (B, HW, C)
        query_pos: torch.Tensor,    # (B, Q, C) — learnable query embeddings
    ) -> torch.Tensor:
        # Self-attention
        q = queries + query_pos
        sa_out, _ = self.self_attn(q, q, queries)
        queries = self.norm1(queries + self.dropout(sa_out))

        # Cross-attention
        q = queries + query_pos
        ca_out, _ = self.cross_attn(q, memory, memory)
        queries = self.norm2(queries + self.dropout(ca_out))

        # FFN
        queries = self.norm3(queries + self.dropout(self.ffn(queries)))
        return queries


class TransformerDecoder(nn.Module):
    """複数層の DETR デコーダ"""

    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Returns: 各層の出力のリスト (auxiliary loss のため)
        """
        outputs = []
        for layer in self.layers:
            queries = layer(queries, memory, query_pos)
            outputs.append(queries)
        return outputs


# ---------------------------------------------------------------------------
# Detection Head (bbox + class)
# ---------------------------------------------------------------------------

class DetectionPredHead(nn.Module):
    """
    Query → bbox + class logits (+ 任意で angle / pose) の予測ヘッド。

    角度・pose のブランチは ``obb=True`` / ``num_keypoints>0`` のときだけ作られる。
    作らなければ state_dict にキー自体が現れないので、AABB 学習済み checkpoint との
    互換が完全に保たれる (逆方向は新規 head のみ missing key になる)。
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 1,
        obb: bool = False,
        num_keypoints: int = 0,
        predict_polarity: bool = False,
    ):
        super().__init__()
        self.obb = bool(obb)
        self.num_keypoints = int(num_keypoints)
        self.predict_polarity = bool(predict_polarity)

        # bbox head: 3-layer MLP → [cx, cy, w, h] in [0,1]
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        # class head: 線形分類器 (+ 1 for "no object" / background)
        self.class_head = nn.Linear(d_model, num_classes + 1)

        # angle head: → [sin(2*theta), cos(2*theta)] (sigmoid を通さない生の 2 次元)
        # 2 倍角にすることで矩形の周期 pi と一致し、角度の境界不連続が原理的に消える。
        if self.obb:
            self.angle_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 2),
            )

        # pose head (任意): キーポイント座標 + 可視性
        if self.num_keypoints > 0:
            self.keypoint_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, self.num_keypoints * 3),
            )
        # head-tail polarity (任意): OBB の向きを 180 度反転すべきかの 2 値分類
        if self.predict_polarity:
            self.polarity_head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, Q, d_model)
        Returns:
            dict:
              logits:    (B, Q, num_classes + 1)
              boxes:     (B, Q, 4) — sigmoid でクランプ済み [cx, cy, w, h]
              angles:    (B, Q, 2) — [sin2t, cos2t] 未正規化 (obb=False なら None)
              keypoints: (B, Q, K, 3) — [x, y, logit] (pose 無効なら None)
              polarity:  (B, Q, 2) — head-tail 反転 logit (無効なら None)
        """
        out: Dict[str, Optional[torch.Tensor]] = {
            "logits": self.class_head(x),
            "boxes": self.bbox_head(x).sigmoid(),
            "angles": None,
            "keypoints": None,
            "polarity": None,
        }
        if self.obb:
            out["angles"] = self.angle_head(x)
        if self.num_keypoints > 0:
            B, Q = x.shape[:2]
            kp = self.keypoint_head(x).reshape(B, Q, self.num_keypoints, 3)
            # x, y は [0,1] 正規化座標、3 列目は可視性 logit (sigmoid は損失側で適用)
            out["keypoints"] = torch.cat([kp[..., :2].sigmoid(), kp[..., 2:]], dim=-1)
        if self.predict_polarity:
            out["polarity"] = self.polarity_head(x)
        return out


# ---------------------------------------------------------------------------
# ResNet Backbone wrapper
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """
    torchvision の ResNet から C3, C4, C5 (+ 任意で C2) を取り出す wrapper。

    ResNet-18/34: C2=64ch,  C3=128ch, C4=256ch, C5=512ch
    ResNet-50:    C2=256ch, C3=512ch, C4=1024ch, C5=2048ch

    ``return_c2=True`` のとき stride 4 の C2 を先頭に含めて返す。小さい個体・
    密集個体の検出を強化するための高解像度レベルで、VRAM と latency の増加と
    引き換えになる (config の ``detector.fpn.use_p2`` で切り替える)。
    """

    # stride 8/16/32 (C3, C4, C5) の出力チャネル
    CHANNEL_MAP = {
        "resnet18": [128, 256, 512],
        "resnet34": [128, 256, 512],
        "resnet50": [512, 1024, 2048],
    }
    # stride 4 (C2 = layer1 出力) の出力チャネル
    C2_CHANNELS = {"resnet18": 64, "resnet34": 64, "resnet50": 256}

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        freeze_bn: bool = False,
        return_c2: bool = False,
    ):
        super().__init__()
        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        elif name == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)
        elif name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {name}")

        # stem + layer1 → stride 4 (C2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4 → stride 4  (C2)
        self.layer2 = backbone.layer2  # stride 4 → stride 8  (C3)
        self.layer3 = backbone.layer3  # stride 8 → stride 16 (C4)
        self.layer4 = backbone.layer4  # stride 16 → stride 32 (C5)

        self.return_c2 = bool(return_c2)
        self.out_channels = list(self.CHANNEL_MAP[name])
        if self.return_c2:
            self.out_channels = [self.C2_CHANNELS[name]] + self.out_channels

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            return_c2=False: [C3, C4, C5] — stride 8, 16, 32
            return_c2=True:  [C2, C3, C4, C5] — stride 4, 8, 16, 32
        """
        x = self.stem(x)
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return [c2, c3, c4, c5] if self.return_c2 else [c3, c4, c5]

    def get_out_strides(self) -> List[int]:
        """各出力レベルの stride を返す。"""
        return [4, 8, 16, 32] if self.return_c2 else [8, 16, 32]


# ---------------------------------------------------------------------------
# RT-DETR 風 Detector (メインクラス)
# ---------------------------------------------------------------------------

class RTDETRDetector(BaseDetector):
    """
    RT-DETR 風の Detector 実装。

    アーキテクチャ:
        ResNet Backbone
        ↓
        FPN (C3, C4, C5 → P3, P4, P5)
        ↓
        AIFI (P5 に self-attention で特徴強化)
        ↓
        memory として P3+P4+P5 を flatten して concat
        ↓
        Learnable queries
        ↓
        Transformer Decoder (4 層)
        ↓
        Detection Head (bbox + class)
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        hcfg = cfg.head

        # --- OBB / pose / P2 の有効化フラグ (config が唯一の真実) ---
        self.obb_enabled = bool(getattr(cfg, "obb", None) and cfg.obb.enabled)
        pose_cfg = getattr(cfg, "pose", None)
        self.pose_enabled = bool(pose_cfg and pose_cfg.enabled)
        self.num_keypoints = int(pose_cfg.num_keypoints) if self.pose_enabled else 0
        self.predict_polarity = bool(
            self.pose_enabled and pose_cfg.predict_polarity
        )
        self.use_p2 = bool(getattr(cfg.fpn, "use_p2", False))
        self.p2_in_memory = bool(getattr(cfg.fpn, "p2_in_memory", True))

        # Backbone
        self.backbone = ResNetBackbone(
            name=cfg.backbone.name,
            pretrained=cfg.backbone.pretrained,
            freeze_bn=cfg.backbone.freeze_bn,
            return_c2=self.use_p2,
        )

        # FPN (P2 有効時は 4 レベル)
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=cfg.fpn.out_channels,
        )

        # AIFI (最高レベルの feature map に適用)
        self.aifi = AIFI(
            d_model=cfg.fpn.out_channels,
            num_heads=hcfg.num_heads,
            ffn_dim=hcfg.ffn_dim,
        )

        # Encoder feature → decoder memory に変換する linear projection
        self.enc_proj = nn.Linear(cfg.fpn.out_channels, hcfg.hidden_dim)

        # Learnable query embeddings
        self.query_embed = nn.Embedding(hcfg.num_queries, hcfg.hidden_dim)

        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=hcfg.num_decoder_layers,
            d_model=hcfg.hidden_dim,
            num_heads=hcfg.num_heads,
            ffn_dim=hcfg.ffn_dim,
            dropout=hcfg.dropout,
        )

        # Detection head (最終層 + 各中間層用)
        def _make_head() -> DetectionPredHead:
            return DetectionPredHead(
                hcfg.hidden_dim, hcfg.num_classes,
                obb=self.obb_enabled,
                num_keypoints=self.num_keypoints,
                predict_polarity=self.predict_polarity,
            )

        self.pred_head = _make_head()
        # auxiliary loss 用 head (各デコーダ層)
        self.aux_heads = nn.ModuleList([
            _make_head() for _ in range(hcfg.num_decoder_layers - 1)
        ])

        # Positional encoding (query に使わないが、encoder 側で使用)
        self.pos_enc = SinePositionEncoding2D(cfg.fpn.out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        """bbox head の bias を初期化 (DETR 論文の prior 推奨)"""
        # cx, cy: 0.0 → sigmoid = 0.5 (center of image)
        # w, h: -2.0 → sigmoid ≈ 0.12 (small initial box size, ~12% of image)
        # MOT17 の歩行者は画像幅の ~3-10%、高さの ~10-25% 程度なので、
        # 初期ボックスを画像全体の50%に設定すると収束が遅くなる
        bias_init = torch.zeros(4)
        bias_init[2] = -2.0  # w
        bias_init[3] = -2.0  # h
        with torch.no_grad():
            self.pred_head.bbox_head[-1].bias.copy_(bias_init)
            for aux_head in self.aux_heads:
                aux_head.bbox_head[-1].bias.copy_(bias_init)
        # class head の bias 初期化 (focal loss 推奨)
        # foreground クラスだけを低確率に設定し、background (最後のクラス) は 0 のまま
        # こうすることで P(foreground) ≈ prior_prob << P(background) となり
        # class imbalance に対する focal loss の補正が正しく機能する
        prior_prob = 0.01
        fg_bias = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.6
        with torch.no_grad():
            self.pred_head.class_head.bias[:-1] = fg_bias   # foreground classes
            self.pred_head.class_head.bias[-1]  = 0.0       # background
            for aux_head in self.aux_heads:
                aux_head.class_head.bias[:-1] = fg_bias
                aux_head.class_head.bias[-1]  = 0.0

            # angle head: theta=0 (sin2t=0, cos2t=1) を初期姿勢にする。
            # 回転アノテーションが無い / 少ないデータでも AABB 相当から学習が始まる。
            if self.obb_enabled:
                for head in [self.pred_head, *self.aux_heads]:
                    head.angle_head[-1].weight.zero_()
                    head.angle_head[-1].bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, images: torch.Tensor) -> DetectionOutput:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            DetectionOutput
        """
        B = images.shape[0]

        # 1. Backbone
        features = self.backbone(images)  # [C3, C4, C5]

        # 2. FPN
        fpn_features = self.fpn(features)  # [P3, P4, P5]

        # 3. AIFI on P5
        fpn_features[-1] = self.aifi(fpn_features[-1])

        # 4. flatten して concat → memory
        # encoder_features は Hierarchical Temporal Module / behavior ROI に渡す
        encoder_features = fpn_features  # List[(B, C, H_i, W_i)]

        # P2 (stride 4) は解像度が高くトークン数が支配的になる。p2_in_memory=False なら
        # FPN の特徴強化だけに使い、decoder の cross-attention memory からは外して
        # 計算量を抑える (精度と latency のトレードオフを config で選べるようにする)。
        mem_sources = fpn_features
        if self.use_p2 and not self.p2_in_memory:
            mem_sources = fpn_features[1:]

        mem_list = []
        for f in mem_sources:
            pos = self.pos_enc(f)
            f_pos = (f + pos).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            mem_list.append(f_pos)
        memory = torch.cat(mem_list, dim=1)  # (B, sum(H_i*W_i), C)
        memory = self.enc_proj(memory)       # (B, sum(H_i*W_i), hidden_dim)

        # 5. Decoder
        # query_embed: (Q, hidden_dim) → (B, Q, hidden_dim)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = torch.zeros_like(query_pos)

        decoder_outputs = self.decoder(queries, memory, query_pos)

        # 6. 最終予測
        final_out = decoder_outputs[-1]                  # (B, Q, hidden_dim)
        head_out = self.pred_head(final_out)

        pred_obb = None
        pred_boxes = head_out["boxes"]
        if self.obb_enabled:
            # angle と tight な w/h から OBB を組み、その外接矩形で pred_boxes を
            # 置き換える。こうすると pred_boxes は OBB 有効時も「軸平行な外接矩形」
            # という一貫した意味を保ち、既存の AABB 経路がそのまま動く。
            pred_obb = self._assemble_obb(head_out["boxes"], head_out["angles"])
            pred_boxes = obb_to_cxcywh(pred_obb)

        # auxiliary 予測 (損失計算時に使用)
        aux_outputs = []
        for i, layer_out in enumerate(decoder_outputs[:-1]):
            aux = self.aux_heads[i](layer_out)
            aux_entry = {"pred_logits": aux["logits"], "pred_boxes": aux["boxes"]}
            if self.obb_enabled:
                aux_obb = self._assemble_obb(aux["boxes"], aux["angles"])
                aux_entry["pred_obb"] = aux_obb
                aux_entry["pred_angles"] = aux["angles"]
                aux_entry["pred_boxes"] = obb_to_cxcywh(aux_obb)
            aux_outputs.append(aux_entry)

        return DetectionOutput(
            pred_logits=head_out["logits"],     # (B, Q, num_classes+1)
            pred_boxes=pred_boxes,              # (B, Q, 4) [cx, cy, w, h] 外接矩形
            query_features=final_out,           # (B, Q, hidden_dim)
            encoder_features=encoder_features,  # List[(B, C, H_i, W_i)]
            aux_outputs=aux_outputs,            # List[{pred_logits, pred_boxes, ...}]
            pred_angles=head_out["angles"],     # (B, Q, 2) or None
            pred_obb=pred_obb,                  # (B, Q, 5) or None
            pred_keypoints=head_out["keypoints"],
            pred_polarity=head_out["polarity"],
        )

    @staticmethod
    def _assemble_obb(boxes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        bbox 回帰 (cx, cy, w, h) と角度符号 (sin2t, cos2t) から OBB を組む。

        Args:
            boxes:  (B, Q, 4) sigmoid 済み [cx, cy, w, h] ∈ [0,1]
            angles: (B, Q, 2) 未正規化 [sin(2t), cos(2t)]
        Returns:
            (B, Q, 5) [cx, cy, w, h, theta]
        """
        theta = decode_angle(angles)                    # (B, Q)
        return torch.cat([boxes, theta.unsqueeze(-1)], dim=-1)

    def get_hidden_dim(self) -> int:
        return self.cfg.head.hidden_dim

    def get_num_queries(self) -> int:
        return self.cfg.head.num_queries

    def get_encoder_channels(self) -> List[int]:
        return [self.cfg.fpn.out_channels] * len(self.backbone.out_channels)

    def get_encoder_strides(self) -> List[int]:
        """encoder_features の各レベルの stride (behavior ROIAlign が使う)。"""
        return self.backbone.get_out_strides()

    def supports_obb(self) -> bool:
        return self.obb_enabled
