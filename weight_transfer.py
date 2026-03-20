# weight_transfer.py
# Stage N の checkpoint から Stage N+1 モデルに重みを転送する
# 実行例: python weight_transfer.py --src stage3_best.pth --dst stage4_init.pth

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import torch
from pathlib import Path

from htrtdetr.config.config import get_variant_config
from htrtdetr.models import build_model

WORK = "C:/Users/utopi/YOAKE_pre-train"


def transfer_weights(src_path: str, dst_stage: int, dst_path: str) -> None:
    """
    src_path の checkpoint を dst_stage のモデルにロードし、dst_path に保存する。

    引き継ぐレイヤー / 初期化するレイヤー:
      Stage 1 → Stage 2: detector (引き継ぎ), temporal/action head (新規初期化)
      Stage 2 → Stage 3: detector + temporal (引き継ぎ), id_head (新規初期化)
      Stage 3 → Stage 4: 全モジュール (引き継ぎ)
    """
    # 転送先モデルを作成
    cfg = get_variant_config("large", stage=dst_stage)
    model = build_model(cfg.model)

    # 転送元 checkpoint を読み込む
    ckpt = torch.load(src_path, map_location="cpu")
    src_state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # strict=False: 形状が一致しないキーはスキップ
    missing, unexpected = model.load_state_dict(src_state, strict=False)
    print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # 保存
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": dst_stage,
        "source": src_path,
    }, dst_path)
    print(f"Saved: {dst_path}")


def show_transferable_keys(src_path: str, dst_stage: int) -> None:
    """転送可能なキーと初期化されるキーを表示する"""
    cfg = get_variant_config("large", stage=dst_stage)
    model = build_model(cfg.model)

    ckpt = torch.load(src_path, map_location="cpu")
    src_state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    dst_keys = set(model.state_dict().keys())
    src_keys = set(src_state.keys())

    transferable = dst_keys & src_keys
    new_keys = dst_keys - src_keys
    removed_keys = src_keys - dst_keys

    # モジュール別に集計
    modules = ["detector", "temporal", "id_head", "action_head", "interaction",
               "query_temporal_fusion", "det_adapter", "feature_router",
               "geo_projector", "geo_action_adapter", "geo_id_adapter"]

    print("\n=== Weight Transfer Summary ===")
    for mod in modules:
        t = [k for k in transferable if k.startswith(mod)]
        n = [k for k in new_keys if k.startswith(mod)]
        print(f"  {mod:30s}: {len(t):4d} transferred, {len(n):4d} re-initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source checkpoint path")
    parser.add_argument("--dst_stage", type=int, required=True, help="Destination stage (1-4)")
    parser.add_argument("--dst", required=True, help="Destination checkpoint path")
    parser.add_argument("--show_keys", action="store_true")
    args = parser.parse_args()

    if args.show_keys:
        show_transferable_keys(args.src, args.dst_stage)
    else:
        transfer_weights(args.src, args.dst_stage, args.dst)
