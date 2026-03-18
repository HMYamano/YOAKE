"""
visualize_architecture.py — YOAKE アーキテクチャ可視化スクリプト

以下の図を生成する:
  1. YOAKE 全体パイプライン (ブロック図)
  2. RT-DETR Detector 詳細
  3. Hierarchical Temporal Module (HTM) 詳細
  4. Memory-based ID Head 詳細
  5. Action Head 詳細
  6. Stage 学習パイプライン (フリーズ状態)
  7. torchinfo によるモデルサマリー (テキスト保存)

出力先: outputs/architecture_diagrams/
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # GUI なし環境向け
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# torchinfo
import torch
import torch.nn as nn
from torchinfo import summary

# YOAKE モジュール
from htrtdetr.config.config import (
    ModelConfig, DetectorConfig, HierarchicalTemporalConfig,
    MemoryIDConfig, ActionHeadConfig,
)
from htrtdetr.models.detector.rtdetr_wrapper import (
    RTDETRDetector, ResNetBackbone, FPN, AIFI,
    TransformerDecoder, DetectionPredHead,
)
from htrtdetr.models.temporal.hierarchical_temporal import (
    HierarchicalTemporalModule, TemporalBranch, DepthwiseSeparableTemporalConv,
)
from htrtdetr.models.id_head.memory_id_head import MemoryIDHead, IDEmbeddingNet
from htrtdetr.models.action_head.action_head import ActionHead
from htrtdetr.models.ht_rtdetr import HTRTDETR

# ---------------------------------------------------------------------------
# 出力ディレクトリ
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "outputs" / "architecture_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# カラーパレット
# ---------------------------------------------------------------------------
C = {
    "detector":  "#4C72B0",
    "temporal":  "#DD8452",
    "id_head":   "#55A868",
    "action":    "#C44E52",
    "fusion":    "#8172B2",
    "input":     "#3A7EB5",
    "output":    "#2CA02C",
    "frozen":    "#AAAAAA",
    "active":    "#E8A838",
    "bg":        "#F8F8F8",
    "arrow":     "#333333",
    "border":    "#333333",
    "text":      "#111111",
    "subtext":   "#555555",
}

DPI = 150


def draw_block(ax, x, y, w, h, label, sublabel="", color="#4C72B0",
               fontsize=10, subfontsize=8, alpha=0.92, text_color="white",
               style="round,pad=0.05"):
    """ブロックを描画するヘルパー"""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style,
        linewidth=1.5,
        edgecolor=C["border"],
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(box)
    ty = y + (h * 0.15 if sublabel else 0)
    ax.text(x, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=text_color, alpha=0.85)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#333333", lw=1.5, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label, fontsize=7.5, color=C["subtext"],
                ha="left", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))


# ===========================================================================
# 図1: YOAKE 全体パイプライン
# ===========================================================================

def fig_overall_pipeline():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("YOAKE — Overall Pipeline", fontsize=16, fontweight="bold",
                 pad=14, color=C["text"])

    # --- 入力 ---
    draw_block(ax, 2, 8, 3.2, 0.7,
               "Input Frames", "(B, T, 3, H, W)",
               color=C["input"], fontsize=11, subfontsize=9)

    # --- Detector ---
    draw_block(ax, 2, 6.4, 3.2, 1.2,
               "Spatial Detector", "RT-DETR (ResNet + FPN + AIFI + Decoder)",
               color=C["detector"], fontsize=11, subfontsize=8)

    # Detector 出力ラベル
    ax.text(2, 5.55, "pred_logits (B,Q,C+1)  pred_boxes (B,Q,4)  query_feats (B,Q,D)",
            ha="center", fontsize=7.5, color=C["subtext"])

    # --- Stage1 出力 ---
    draw_block(ax, 6.5, 6.4, 1.6, 0.55,
               "Stage 1", "Detection only",
               color=C["detector"], fontsize=9, subfontsize=7.5)

    # --- HTM ---
    draw_block(ax, 2, 4.5, 3.2, 1.1,
               "Hierarchical Temporal Module",
               "Short(d=1) | Mid(d=2) | Long(d=4) → Fusion",
               color=C["temporal"], fontsize=10, subfontsize=8)

    ax.text(2, 3.8, "temporal_feats (B, T, Q, out_dim)",
            ha="center", fontsize=7.5, color=C["subtext"])

    # --- Stage2 出力 ---
    draw_block(ax, 6.5, 4.5, 1.6, 0.55,
               "Stage 2", "Action only",
               color=C["action"], fontsize=9, subfontsize=7.5)

    # --- Feature Router ---
    draw_block(ax, 2, 3.1, 3.2, 0.75,
               "Multi-Head Feature Router",
               "spatial + temporal → id_feat / action_feat",
               color=C["fusion"], fontsize=10, subfontsize=8)

    # --- ID Head ---
    draw_block(ax, 0.8, 1.6, 2.2, 1.0,
               "Memory-based ID Head",
               "GRU + Hungarian Matching",
               color=C["id_head"], fontsize=10, subfontsize=8)

    # --- Action Head ---
    draw_block(ax, 3.2, 1.6, 2.2, 1.0,
               "Action Head",
               "Spatial + Temporal + Interaction → MLP",
               color=C["action"], fontsize=10, subfontsize=8)

    # --- Stage3 ---
    draw_block(ax, 6.5, 1.8, 1.6, 0.55,
               "Stage 3", "ID only",
               color=C["id_head"], fontsize=9, subfontsize=7.5)

    # --- Stage4 ---
    draw_block(ax, 8.5, 1.8, 1.6, 0.55,
               "Stage 4", "Full model",
               color=C["active"], fontsize=9, subfontsize=7.5, text_color=C["text"])

    # --- 出力 ---
    draw_block(ax, 2, 0.55, 3.2, 0.7,
               "Output per Detection",
               "bbox | score | class | track_id | id_conf | action | action_conf",
               color=C["output"], fontsize=10, subfontsize=7.5)

    # --- Interaction ---
    draw_block(ax, 6.5, 3.1, 1.6, 0.55,
               "Interaction\nFeatureComputer",
               "dist / angle / velocity",
               color=C["action"], fontsize=8.5, subfontsize=7.5)

    # Arrows
    draw_arrow(ax, 2, 7.65, 2, 7.0)             # input → detector
    draw_arrow(ax, 2, 5.8, 2, 5.05)             # detector → HTM
    draw_arrow(ax, 2, 3.95, 2, 3.47)            # HTM → router
    draw_arrow(ax, 0.9, 2.72, 0.9, 2.1)         # router → id head
    draw_arrow(ax, 3.1, 2.72, 3.1, 2.1)         # router → action head
    draw_arrow(ax, 0.9, 1.1, 0.9, 0.9)          # id → output
    draw_arrow(ax, 3.1, 1.1, 3.1, 0.9)          # action → output
    # Stage分岐
    draw_arrow(ax, 3.6, 6.4, 5.7, 6.4, style="->")
    draw_arrow(ax, 3.6, 4.5, 5.7, 4.5, style="->")
    draw_arrow(ax, 3.6, 3.1, 5.7, 3.1, style="->")
    draw_arrow(ax, 3.6, 1.6, 5.7, 1.6, style="->")
    draw_arrow(ax, 6.5, 2.73, 6.5, 2.07, style="->")  # interaction → action
    # Stage 4
    draw_arrow(ax, 6.5, 1.52, 7.7, 1.52, style="->")

    # Stage ラベル
    for sx, sy, label, col in [
        (8.5, 6.4, "Detection\nStage 1", C["detector"]),
        (8.5, 4.5, "Action\nStage 2", C["action"]),
        (8.5, 3.1, "ID\nStage 3", C["id_head"]),
    ]:
        draw_block(ax, sx, sy, 1.5, 0.55, label, color=col, fontsize=8.5)

    # データフロー注釈
    ax.text(9.6, 7.5, "Data Flow", fontsize=12, fontweight="bold", color=C["text"])
    for y_pos, dim_str, col in [
        (7.1, "(B, T, 3, H, W)", C["input"]),
        (6.05, "(B, Q, D)", C["detector"]),
        (5.1, "(B, T, Q, out_dim)", C["temporal"]),
        (3.55, "id_feat / action_feat", C["fusion"]),
        (1.7, "track_ids, action_logits", C["output"]),
    ]:
        ax.text(9.2, y_pos, dim_str, fontsize=9, color=col,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor=col,
                          boxstyle="round,pad=0.2"))

    plt.tight_layout()
    path = OUT_DIR / "01_overall_pipeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# 図2: RT-DETR Detector 詳細
# ===========================================================================

def fig_detector():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("RT-DETR Detector — Architecture Detail", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    # 入力
    draw_block(ax, 3.5, 9.3, 4, 0.7, "Input Image", "(B, 3, H, W)",
               color=C["input"], fontsize=11)

    # Backbone
    draw_block(ax, 3.5, 8.0, 4, 1.0,
               "ResNet Backbone",
               "stem→layer1→layer2→layer3→layer4\n(ImageNet pretrained)",
               color=C["detector"], fontsize=11, subfontsize=8.5)

    # C3/C4/C5
    for xpos, label, stride in [(1.5, "C3\n(stride 8)", 8),
                                  (3.5, "C4\n(stride 16)", 16),
                                  (5.5, "C5\n(stride 32)", 32)]:
        draw_block(ax, xpos, 6.7, 1.3, 0.7, label, f"ch={128 if xpos<4 else 512}",
                   color=C["detector"], fontsize=9, subfontsize=7.5)

    ax.text(3.5, 6.15, "C3: 128ch    C4: 256ch    C5: 512ch  [ResNet-18]",
            ha="center", fontsize=8, color=C["subtext"])

    # FPN
    draw_block(ax, 3.5, 5.35, 5.5, 0.85,
               "FPN (Feature Pyramid Network)",
               "Lateral Conv (1×1) + Top-Down Fusion + Output Conv (3×3, GN, ReLU)",
               color=C["detector"], fontsize=10.5, subfontsize=8.5)

    # P3/P4/P5
    for xpos, label in [(1.5, "P3\n256ch"), (3.5, "P4\n256ch"), (5.5, "P5\n256ch")]:
        draw_block(ax, xpos, 4.35, 1.3, 0.65, label, color=C["detector"], fontsize=9)

    # AIFI
    draw_block(ax, 5.5, 3.35, 1.8, 0.75,
               "AIFI",
               "Self-Attn on P5\n(8-head, 2D SinePE)",
               color=C["detector"], fontsize=9.5, subfontsize=8)

    # Memory concat
    draw_block(ax, 3.5, 2.4, 5.5, 0.75,
               "Encoder Memory",
               "Flatten [P3+P4+P5] → concat → Linear → (B, ΣH_i·W_i, 256)",
               color="#6A8EC9", fontsize=10, subfontsize=8.5)

    # Learnable queries
    draw_block(ax, 0.9, 2.4, 1.4, 0.75,
               "Query\nEmbeddings",
               "100 × 256d",
               color=C["detector"], fontsize=9, subfontsize=8)

    # Transformer Decoder
    draw_block(ax, 3.5, 1.35, 5.5, 0.85,
               "Transformer Decoder (4 layers)",
               "Self-Attn (queries) → Cross-Attn (memory) → FFN → LayerNorm",
               color=C["detector"], fontsize=10.5, subfontsize=8.5)

    # Detection Head
    draw_block(ax, 2.0, 0.45, 2.5, 0.7,
               "Bbox Head (MLP)",
               "3-layer MLP → sigmoid → [cx,cy,w,h]",
               color="#2B7CBB", fontsize=9, subfontsize=7.5)
    draw_block(ax, 5.0, 0.45, 2.5, 0.7,
               "Class Head (Linear)",
               "Linear → C+1 logits",
               color="#2B7CBB", fontsize=9, subfontsize=7.5)
    draw_block(ax, 8.0, 0.45, 2.2, 0.7,
               "Query Features",
               "(B, Q, 256) → HTM",
               color=C["temporal"], fontsize=9, subfontsize=7.5)

    # Arrows
    draw_arrow(ax, 3.5, 8.95, 3.5, 8.5)
    draw_arrow(ax, 3.5, 7.5, 1.5, 7.05)
    draw_arrow(ax, 3.5, 7.5, 3.5, 7.05)
    draw_arrow(ax, 3.5, 7.5, 5.5, 7.05)
    for xpos in [1.5, 3.5, 5.5]:
        draw_arrow(ax, xpos, 6.35, xpos, 5.77)
    for xpos in [1.5, 3.5, 5.5]:
        draw_arrow(ax, xpos, 4.93, xpos, 4.67)
    draw_arrow(ax, 5.5, 4.03, 5.5, 3.73)
    for xpos in [1.5, 3.5, 5.5]:
        draw_arrow(ax, xpos, 4.03, 3.5, 2.77)
    draw_arrow(ax, 3.5, 2.02, 3.5, 1.77)
    draw_arrow(ax, 0.9, 2.02, 3.5, 1.77)  # query → decoder
    draw_arrow(ax, 2.0, 0.93, 2.0, 0.8)
    draw_arrow(ax, 5.0, 0.93, 5.0, 0.8)
    draw_arrow(ax, 3.5, 0.93, 8.0, 0.8)

    # 注釈: Auxiliary Loss
    ax.text(9.5, 1.35, "Auxiliary outputs\nfrom each decoder layer\n(intermediate heads)",
            fontsize=8, color=C["subtext"], ha="left", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=C["subtext"],
                      boxstyle="round,pad=0.3"))

    plt.tight_layout()
    path = OUT_DIR / "02_detector.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# 図3: Hierarchical Temporal Module
# ===========================================================================

def fig_htm():
    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("Hierarchical Temporal Module (HTM)", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    # 入力
    draw_block(ax, 6.5, 9.0, 4, 0.7,
               "Input: Query Feature Sequence",
               "(B×Q, T, D=256)  ← packed per detection query",
               color=C["input"], fontsize=11, subfontsize=9)

    # Input Proj
    draw_block(ax, 6.5, 7.9, 4, 0.7,
               "Input Projection",
               "Linear(D→D) + LayerNorm",
               color=C["temporal"], fontsize=10, subfontsize=9)
    draw_arrow(ax, 6.5, 8.65, 6.5, 8.25)

    ax.text(6.5, 7.45, "permute → (B×Q, D, T)", ha="center", fontsize=8, color=C["subtext"])
    draw_arrow(ax, 6.5, 7.55, 6.5, 7.15)

    # 3 branches
    branches = [
        (2.0,  "Short Branch",  "dilation=1,  num_frames=3",  "T_short=3",  "~Fine-grained motion"),
        (6.5,  "Mid Branch",    "dilation=2,  num_frames=8",  "T_mid=8",    "~Mid-term pattern"),
        (11.0, "Long Branch",   "dilation=4,  num_frames=16", "T_long=16",  "~Long-term behavior"),
    ]
    branch_y_top = 6.9

    for bx, bname, bparams, bframes, bjp in branches:
        # 分岐矢印
        draw_arrow(ax, 6.5, 7.15, bx, branch_y_top + 0.05)

        # Branch 外枠
        rect = FancyBboxPatch(
            (bx - 1.5, 3.7), 3.0, 3.15,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=C["temporal"], facecolor="#FFF3E6", alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(bx, branch_y_top - 0.05, bname, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C["temporal"])
        ax.text(bx, branch_y_top - 0.3, bparams, ha="center", fontsize=8,
                color=C["subtext"])
        ax.text(bx, branch_y_top - 0.52, bjp, ha="center", fontsize=8,
                color="#888888")

        # pad/truncate
        draw_block(ax, bx, 6.1, 2.5, 0.55,
                   f"Pad/Truncate to {bframes}",
                   color="#D9ECFF", fontsize=8.5, text_color=C["text"])

        # DepthwiseSeparable Conv
        draw_block(ax, bx, 5.3, 2.5, 0.65,
                   "DepthwiseSeparable\nTemporal Conv",
                   color=C["temporal"], fontsize=9, subfontsize=7.5)

        # Channel Attention (optional)
        draw_block(ax, bx, 4.5, 2.5, 0.6,
                   "Channel Attention Gate\n(SE-like, optional)",
                   color="#FFAA66", fontsize=8.5, subfontsize=7, text_color=C["text"])

        # Residual
        draw_block(ax, bx, 3.85, 2.5, 0.5,
                   "+ Residual Connection",
                   color="#EEE", fontsize=8.5, text_color=C["text"])

        # Pad to T
        draw_block(ax, bx, 3.1, 2.5, 0.55,
                   "Pad/Interpolate to T",
                   color="#D9ECFF", fontsize=8.5, text_color=C["text"])

        # 矢印
        draw_arrow(ax, bx, 5.82, bx, 5.62)
        draw_arrow(ax, bx, 4.97, bx, 4.8)
        draw_arrow(ax, bx, 4.2, bx, 4.1)
        draw_arrow(ax, bx, 3.6, bx, 3.37)

    # BranchFusion
    draw_block(ax, 6.5, 2.25, 7.0, 0.8,
               "BranchFusion",
               "concat([short, mid, long]) (B, 3D, T) → Linear → LayerNorm → ReLU → (B, out_dim, T)",
               color=C["temporal"], fontsize=10.5, subfontsize=8.5)

    for bx in [2.0, 6.5, 11.0]:
        draw_arrow(ax, bx, 2.82, 6.5, 2.65)

    # Output Norm
    draw_block(ax, 6.5, 1.3, 4.0, 0.65,
               "LayerNorm → Output",
               "permute → (B×Q, T, out_dim=256)",
               color=C["output"], fontsize=10, subfontsize=9)
    draw_arrow(ax, 6.5, 1.85, 6.5, 1.62)

    # DepthwiseSep 詳細インセット
    inset_x, inset_y = 10.2, 5.9
    ax.text(inset_x, inset_y + 0.15, "DepthwiseSep Conv detail:", fontsize=8,
            fontweight="bold", color=C["subtext"])
    for i, (lbl, dim) in enumerate([
        ("Depthwise Conv1d\n(groups=C, dilation=d)", "(B,C,T)→(B,C,T)"),
        ("Pointwise Conv1d\n(kernel=1)", "(B,C,T)→(B,C,T)"),
        ("BatchNorm1d + ReLU", "(B,C,T)"),
    ]):
        yy = inset_y - 0.05 - i * 0.62
        draw_block(ax, inset_x, yy, 2.3, 0.5, lbl, dim,
                   color="#AACBEE", fontsize=7.5, subfontsize=7, text_color=C["text"])
        if i < 2:
            draw_arrow(ax, inset_x, yy - 0.25, inset_x, yy - 0.37)

    plt.tight_layout()
    path = OUT_DIR / "03_htm.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# 図4: Memory-based ID Head
# ===========================================================================

def fig_id_head():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("Memory-based ID Head", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    # 入力
    draw_block(ax, 3.5, 8.4, 5.5, 0.7,
               "Input Features",
               "id_feat (N, D)  +  bbox (N, 4)  +  velocity (N, 2)",
               color=C["input"], fontsize=11, subfontsize=9)

    # Build Input
    draw_block(ax, 3.5, 7.45, 5.5, 0.7,
               "_build_input()",
               "concat[feature, bbox, velocity] → input_proj (Linear → ReLU)",
               color=C["id_head"], fontsize=10, subfontsize=8.5)
    draw_arrow(ax, 3.5, 8.05, 3.5, 7.8)

    # MemoryUpdateNet
    rect = FancyBboxPatch((0.5, 5.5), 6, 1.6,
                           boxstyle="round,pad=0.1",
                           linewidth=1.5, edgecolor=C["id_head"], facecolor="#E6F4EA")
    ax.add_patch(rect)
    ax.text(3.5, 6.9, "MemoryUpdateNet", ha="center", fontsize=11,
            fontweight="bold", color=C["id_head"])
    draw_block(ax, 2.0, 6.35, 2.2, 0.6,
               "GRU Cell",
               "hidden_state update",
               color=C["id_head"], fontsize=9.5, subfontsize=8)
    draw_block(ax, 5.0, 6.35, 2.2, 0.6,
               "Embedding Head",
               "Linear → L2-normalize",
               color=C["id_head"], fontsize=9.5, subfontsize=8)

    draw_arrow(ax, 3.5, 7.1, 2.0, 6.65)   # input → GRU
    draw_arrow(ax, 3.5, 7.1, 5.0, 6.65)   # input → Emb
    draw_arrow(ax, 2.0, 6.05, 5.0, 6.05)  # GRU → Emb

    # Memory Bank
    rect2 = FancyBboxPatch((7.5, 5.0), 4.8, 2.5,
                            boxstyle="round,pad=0.1",
                            linewidth=1.5, edgecolor="#888", facecolor="#F5F5F5")
    ax.add_patch(rect2)
    ax.text(9.9, 7.25, "IdentityMemory (per sequence)", ha="center",
            fontsize=10, fontweight="bold", color="#555")
    for i, item in enumerate(["gru_hidden (memory_dim)", "last_bbox [cx,cy,w,h]",
                               "last_embedding (emb_dim)", "velocity [dx, dy]",
                               "TTL (time-to-live)"]):
        ax.text(7.9, 6.9 - i * 0.36, f"• {item}", fontsize=8.5, color=C["subtext"])

    draw_arrow(ax, 5.5, 7.45, 7.5, 6.5, label="update hidden state")

    # Training vs Inference 分岐
    draw_block(ax, 3.5, 4.65, 2.5, 0.65,
               "Training",
               "GT track_id → ID classifier\n→ CE loss + optional triplet",
               color="#5A9E6F", fontsize=9.5, subfontsize=8)
    draw_block(ax, 8.5, 4.65, 2.5, 0.65,
               "Inference",
               "Cosine Similarity + IoU\n→ Hungarian Matching",
               color="#E07030", fontsize=9.5, subfontsize=8)

    draw_arrow(ax, 2.5, 5.5, 2.5, 4.97)
    draw_arrow(ax, 8.5, 5.5, 8.5, 4.97)

    ax.text(6.0, 5.2, "Branch by\ntraining / inference",
            ha="center", fontsize=9, color=C["subtext"])

    # ID Classifier
    draw_block(ax, 3.5, 3.6, 5.5, 0.75,
               "ID Classifier (Linear)",
               "(N, embedding_dim) → (N, max_ids+1) logits",
               color=C["id_head"], fontsize=10.5, subfontsize=8.5)
    draw_arrow(ax, 3.5, 4.32, 3.5, 3.97)

    # Cost Matrix detail
    rect3 = FancyBboxPatch((6.8, 2.5), 5.5, 1.7,
                            boxstyle="round,pad=0.1",
                            linewidth=1.2, edgecolor="#E07030", facecolor="#FFF3EC")
    ax.add_patch(rect3)
    ax.text(9.55, 3.95, "Hungarian Matching (inference)", ha="center",
            fontsize=9.5, fontweight="bold", color="#E07030")
    ax.text(7.1, 3.65,
            "Cost = −(0.5 × cosine_similarity + 0.5 × IoU)\n"
            "scipy.linear_sum_assignment\n"
            "New ID assigned if cost > new_id_threshold",
            fontsize=8.5, color=C["subtext"], va="top")

    # 出力
    draw_block(ax, 3.5, 1.95, 5.5, 0.7,
               "Output",
               "id_logits (N, max_ids+1)  |  id_embeddings (N, emb_dim)  |  track_ids",
               color=C["output"], fontsize=10, subfontsize=9)
    draw_arrow(ax, 3.5, 3.22, 3.5, 2.3)

    plt.tight_layout()
    path = OUT_DIR / "04_id_head.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# 図5: Action Head
# ===========================================================================

def fig_action_head():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("Action Head — Behavior Classification", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    # 3 入力
    src_y = 8.3
    for x, lbl, sub, col in [
        (2.0, "Spatial Feature", "spatial_feat\n(N, feature_dim)", C["detector"]),
        (6.0, "Temporal Feature", "temporal_feat\n(N, temporal_dim)", C["temporal"]),
        (10.0, "Interaction Feature", "interaction_feat\n(N, interaction_dim)", "#9467BD"),
    ]:
        draw_block(ax, x, src_y, 3.2, 0.9, lbl, sub,
                   color=col, fontsize=10, subfontsize=8.5)
        draw_arrow(ax, x, src_y - 0.45, 6.0, 7.05)

    # Concat / Gate
    draw_block(ax, 6.0, 6.55, 7.0, 0.75,
               "Feature Fusion",
               "concat(spatial, temporal, interaction) → Linear(in_dim→hidden) → LayerNorm → ReLU",
               color=C["action"], fontsize=10, subfontsize=8.5)

    # MLP layers
    mlp_colors = ["#D62728", "#C44E52", "#E07070", "#F09090"]
    for i, (lbl, sub) in enumerate([
        ("MLP Layer 1", "Linear → BatchNorm → ReLU → Dropout"),
        ("MLP Layer 2", "Linear → BatchNorm → ReLU → Dropout"),
        ("MLP Layer 3", "Linear → BatchNorm → ReLU"),
    ]):
        yy = 5.45 - i * 1.0
        draw_block(ax, 6.0, yy, 7.0, 0.75, lbl, sub,
                   color=mlp_colors[i], fontsize=10, subfontsize=8.5)
        draw_arrow(ax, 6.0, yy + 0.37, 6.0, yy + 0.37 - 0.25)

    draw_arrow(ax, 6.0, 6.17, 6.0, 5.82)

    # Output linear
    draw_block(ax, 6.0, 2.05, 7.0, 0.75,
               "Output Linear",
               "Linear(hidden_dim → num_actions)",
               color=C["action"], fontsize=10.5, subfontsize=9)
    draw_arrow(ax, 6.0, 2.8, 6.0, 2.42)

    # Outputs
    draw_block(ax, 3.5, 1.1, 3.5, 0.75,
               "action_logits",
               "(N, num_actions) — raw logits",
               color=C["output"], fontsize=10, subfontsize=8.5)
    draw_block(ax, 8.5, 1.1, 3.5, 0.75,
               "action_probs",
               "(N, num_actions) — softmax probs",
               color=C["output"], fontsize=10, subfontsize=8.5)
    draw_arrow(ax, 5.0, 1.67, 3.5, 1.47)
    draw_arrow(ax, 7.0, 1.67, 8.5, 1.47)

    # InteractionFeatureComputer インセット
    inset_box = FancyBboxPatch((8.5, 5.8), 3.0, 2.2,
                                boxstyle="round,pad=0.1",
                                linewidth=1.2, edgecolor="#9467BD", facecolor="#F5EEF8")
    ax.add_patch(inset_box)
    ax.text(10.0, 7.75, "InteractionFeatureComputer", ha="center",
            fontsize=8.5, fontweight="bold", color="#9467BD")
    for i, item in enumerate([
        "• Nearest neighbor distance",
        "• Relative angle",
        "• Relative velocity (2D)",
        "• IoU with nearest neighbor",
    ]):
        ax.text(8.65, 7.4 - i * 0.35, item, fontsize=8, color=C["subtext"])

    ax.text(10.0, 5.92, "→ interaction_feat (N, 4)",
            ha="center", fontsize=8, color="#9467BD")

    plt.tight_layout()
    path = OUT_DIR / "05_action_head.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# 図6: Stage 学習パイプライン
# ===========================================================================

def fig_stage_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("YOAKE — Staged Training Pipeline", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    modules = ["Detector\n(RT-DETR)", "HTM\n(Temporal)", "ID Head\n(GRU+Match)", "Action Head\n(MLP)"]
    module_x = [2.5, 5.5, 8.5, 11.5]
    module_colors = [C["detector"], C["temporal"], C["id_head"], C["action"]]

    stages = [
        ("Stage 1", "Detection\nLoss only",      [True, False, False, False], "detector_best.pth"),
        ("Stage 2", "Action\nLoss only",          [False, True, False, True],  "action_best.pth"),
        ("Stage 3", "ID\nLoss only",              [False, True, True, False],  "id_best.pth"),
        ("Stage 4", "Full\nUnified Loss",         [True, True, True, True],    "full_model_best.pth"),
    ]

    # Module header
    ax.text(0.5, 6.3, "Stage", fontsize=10, fontweight="bold", color=C["text"], ha="center")
    ax.text(0.5, 5.9, "Loss", fontsize=9, color=C["subtext"], ha="center")
    for mx, mlbl, mc in zip(module_x, modules, module_colors):
        draw_block(ax, mx, 6.2, 2.2, 0.65, mlbl, color=mc, fontsize=10)

    # Legend
    ax.text(13.5, 6.6, "Active", ha="center", fontsize=9, color="white",
            bbox=dict(facecolor=C["active"], pad=3, edgecolor="none"))
    ax.text(13.5, 6.2, "Frozen", ha="center", fontsize=9, color="white",
            bbox=dict(facecolor=C["frozen"], pad=3, edgecolor="none"))

    # Stage rows
    for si, (sname, sloss, active_list, save_file) in enumerate(stages):
        yy = 5.0 - si * 1.1

        # Stage label
        draw_block(ax, 0.5, yy, 0.8, 0.65, sname,
                   color=C["active"] if any(active_list) else C["frozen"],
                   fontsize=9, text_color=C["text"])
        ax.text(0.5, yy - 0.42, sloss, ha="center", fontsize=7.5, color=C["subtext"])

        # Module boxes
        for mx, active, mc in zip(module_x, active_list, module_colors):
            col = mc if active else C["frozen"]
            lbl = "ACTIVE" if active else "FROZEN"
            draw_block(ax, mx, yy, 2.2, 0.65, lbl,
                       color=col, fontsize=9.5, alpha=0.88)

        # Loss description
        ax.text(13.4, yy + 0.05, save_file, ha="center", fontsize=8,
                color=C["subtext"],
                bbox=dict(facecolor="white", edgecolor=C["subtext"],
                          boxstyle="round,pad=0.2", alpha=0.8))

        # Arrows between stages
        if si < len(stages) - 1:
            ax.annotate("", xy=(2.5, yy - 0.33), xytext=(2.5, yy - 0.62),
                        arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.2))

    # Column labels under modules
    for mx, desc in zip(module_x, [
        "ResNet+FPN\n+AIFI+Decoder",
        "Short/Mid/Long\nBranch Fusion",
        "GRU Memory\n+Hungarian",
        "Spatial+Temporal\n+Interaction MLP",
    ]):
        ax.text(mx, 0.55, desc, ha="center", fontsize=7.5, color=C["subtext"])

    # Progress arrow
    ax.annotate("", xy=(0.5, 1.78), xytext=(0.5, 5.35),
                arrowprops=dict(arrowstyle="->, head_width=0.2", color=C["arrow"],
                                lw=2.0, connectionstyle="arc3,rad=0"))
    ax.text(0.12, 3.55, "Training\nProgress", ha="center", fontsize=8.5,
            color=C["arrow"], rotation=90)

    plt.tight_layout()
    path = OUT_DIR / "06_stage_pipeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# torchinfo サマリー
# ===========================================================================

def run_torchinfo_summaries():
    """各モジュールの torchinfo summary を生成してテキストファイルに保存"""
    device = torch.device("cpu")

    summary_path = OUT_DIR / "07_torchinfo_summary.txt"
    lines = []

    def add_section(title, model, input_data=None, input_size=None, col_names=None):
        lines.append("=" * 80)
        lines.append(f"  {title}")
        lines.append("=" * 80)
        try:
            kw = dict(
                verbose=0,
                col_names=col_names or ["input_size", "output_size", "num_params"],
                row_settings=["var_names"],
            )
            if input_data is not None:
                s = summary(model, input_data=input_data, **kw)
            elif input_size is not None:
                s = summary(model, input_size=input_size, **kw)
            else:
                s = summary(model, **kw)
            lines.append(str(s))
        except Exception as e:
            lines.append(f"  [Error during summary: {e}]")
        lines.append("")

    # --- 1. Detector ---
    det_cfg = DetectorConfig()
    detector = RTDETRDetector(det_cfg).to(device).eval()
    x_img = torch.zeros(1, 3, 640, 640).to(device)
    add_section("RT-DETR Detector (ResNet-18, 640×640)", detector, input_data=x_img)

    # --- 2. Backbone ---
    backbone = ResNetBackbone(name="resnet18", pretrained=False).to(device).eval()
    add_section("ResNet-18 Backbone", backbone, input_size=(1, 3, 640, 640))

    # --- 3. FPN ---
    fpn = FPN([128, 256, 512], 256).to(device).eval()
    c3 = torch.zeros(1, 128, 80, 80).to(device)
    c4 = torch.zeros(1, 256, 40, 40).to(device)
    c5 = torch.zeros(1, 512, 20, 20).to(device)
    add_section("FPN (Feature Pyramid Network)", fpn, input_data=[[c3, c4, c5]])

    # --- 4. AIFI ---
    aifi = AIFI(d_model=256, num_heads=8, ffn_dim=1024).to(device).eval()
    x_p5 = torch.zeros(1, 256, 20, 20).to(device)
    add_section("AIFI (Intra-scale Attention)", aifi, input_data=x_p5)

    # --- 5. Transformer Decoder ---
    decoder = TransformerDecoder(4, 256, 8, 1024).to(device).eval()
    queries = torch.zeros(1, 100, 256).to(device)
    memory = torch.zeros(1, 2100, 256).to(device)
    query_pos = torch.zeros(1, 100, 256).to(device)
    add_section("Transformer Decoder (4 layers)", decoder,
                input_data=[queries, memory, query_pos])

    # --- 6. HTM ---
    htm_cfg = HierarchicalTemporalConfig()
    htm = HierarchicalTemporalModule(htm_cfg).to(device).eval()
    x_seq = torch.zeros(4, 16, 256).to(device)  # B=4 (B*Q), T=16, D=256
    add_section("Hierarchical Temporal Module (HTM)", htm, input_data=x_seq)

    # --- 7. ID Head ---
    id_cfg = MemoryIDConfig()
    id_head = MemoryIDHead(id_cfg).to(device).eval()
    add_section("Memory-based ID Head", id_head,
                col_names=["kernel_size", "num_params", "trainable"])

    # --- 8. Action Head ---
    act_cfg = ActionHeadConfig()
    act_head = ActionHead(act_cfg).to(device).eval()
    sf = torch.zeros(8, act_cfg.feature_dim).to(device)
    tf = torch.zeros(8, act_cfg.temporal_dim).to(device)
    add_section("Action Head", act_head, input_data=[sf, tf])

    # --- 9. Full Model (Stage 4) ---
    try:
        model_cfg = ModelConfig()
        full_model = HTRTDETR(model_cfg).to(device).eval()
        # パラメータ数のみ
        params = full_model.num_parameters()
        lines.append("=" * 80)
        lines.append("  HTRTDETR Full Model — Parameter Count")
        lines.append("=" * 80)
        for k, v in params.items():
            lines.append(f"  {k:<20}: {v:>12,} params")
        lines.append("")
    except Exception as e:
        lines.append(f"  [HTRTDETR full model error: {e}]")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {summary_path.name}")


# ===========================================================================
# 図7: データフロー次元図
# ===========================================================================

def fig_data_flow():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("YOAKE — Tensor Dimension Flow", fontsize=15, fontweight="bold",
                 color=C["text"], pad=14)

    steps = [
        (1.0, "Input\nFrames",    "(B, T, 3, H, W)",       C["input"]),
        (3.0, "Detector\n(×T)",   "(B, Q, D)\n(B, Q, 4)",  C["detector"]),
        (5.2, "Pack for\nHTM",    "(B·Q, T, D)",            C["fusion"]),
        (7.4, "HTM\nOutput",      "(B·Q, T, out)",          C["temporal"]),
        (9.3, "Unpack\n+Extract", "(N_total, D)\n(N_total, out)", C["fusion"]),
        (11.2, "Router\nOutput",  "id_feat\naction_feat",   C["fusion"]),
        (13.0, "Outputs",         "track_ids\naction_logits", C["output"]),
    ]

    y_main = 3.8
    bw, bh = 1.55, 1.1

    for sx, slbl, sdim, scol in steps:
        draw_block(ax, sx, y_main, bw, bh, slbl, sdim,
                   color=scol, fontsize=9.5, subfontsize=8.5)

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + bw / 2
        x2 = steps[i + 1][0] - bw / 2
        draw_arrow(ax, x1, y_main, x2, y_main)

    # 下段: geo-only flow
    geo_steps = [
        (2.0, "Geo Features\n(bbox+motion)",  "(B, T, GEO_FEAT_DIM)", C["input"]),
        (5.2, "geo_projector\n(Linear)",       "(B, T, D)",            "#9467BD"),
        (7.4, "HTM\nOutput",                   "(B, T, out_dim)",      C["temporal"]),
        (10.0, "Stage2/3\nHeads",              "action / id\nlogits",  C["action"]),
    ]
    y_geo = 1.5
    ax.text(0.1, y_geo, "Geo-only\nPath\n(Stage2/3)", fontsize=8.5,
            color=C["subtext"], ha="left", va="center")
    for sx, slbl, sdim, scol in geo_steps:
        draw_block(ax, sx, y_geo, 1.55, 0.9, slbl, sdim,
                   color=scol, fontsize=8.5, subfontsize=7.5)
    for i in range(len(geo_steps) - 1):
        x1 = geo_steps[i][0] + 0.78
        x2 = geo_steps[i + 1][0] - 0.78
        draw_arrow(ax, x1, y_geo, x2, y_geo)

    ax.axhline(y=2.6, xmin=0.08, xmax=0.92, color="#BBBBBB", lw=1, ls="--")
    ax.text(7.0, 2.7, "forward_geo_sequence()", fontsize=8, color="#AAAAAA", ha="center")
    ax.text(7.0, 5.8, "forward() / forward_single_frame()", fontsize=8,
            color="#AAAAAA", ha="center")

    plt.tight_layout()
    path = OUT_DIR / "07_data_flow_dims.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"\nYOAKE Architecture Visualization")
    print(f"Output directory: {OUT_DIR}\n")

    print("[1/7] Overall Pipeline...")
    fig_overall_pipeline()

    print("[2/7] RT-DETR Detector...")
    fig_detector()

    print("[3/7] Hierarchical Temporal Module...")
    fig_htm()

    print("[4/7] Memory-based ID Head...")
    fig_id_head()

    print("[5/7] Action Head...")
    fig_action_head()

    print("[6/7] Stage Training Pipeline...")
    fig_stage_pipeline()

    print("[7/7] Tensor Dimension Flow...")
    fig_data_flow()

    print("\n[Summary] Running torchinfo summaries...")
    run_torchinfo_summaries()

    print(f"\nAll diagrams saved to: {OUT_DIR}")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")) :
        print(f"  {f.name}")
    txt = list(OUT_DIR.glob("*.txt"))
    for f in txt:
        print(f"  {f.name}")
