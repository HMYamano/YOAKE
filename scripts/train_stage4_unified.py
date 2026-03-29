"""
train_stage4_unified.py — Stage 4: Unified Fine-tuning (All Modules)  [DEPRECATED]

.. deprecated::
   このスクリプトは stage_trainers.py の Stage4Trainer を使う旧系統です。
   正式な学習経路は以下を使用してください:

     yoake train stage=4 [key=value ...]
     # または
     python scripts/train_stage4.py [key=value ...]

   このファイルは互換性のために残してありますが、今後削除される予定です。
"""

from __future__ import annotations

import warnings
warnings.warn(
    "train_stage4_unified.py は非推奨です。"
    "代わりに 'yoake train stage=4' または 'python scripts/train_stage4.py' を使用してください。",
    DeprecationWarning,
    stacklevel=1,
)
# 入力は full-scene シーケンス (T フレームの visual + geo features)。

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import get_stage4_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.training.stage_trainers import Stage4Trainer
from htrtdetr.utils.misc import set_seed

_YOAKE_TRYAL = str(Path(__file__).resolve().parent.parent)


def parse_overrides(argv) -> dict:
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k] = v
    return overrides


def main():
    overrides = parse_overrides(sys.argv[1:])

    cfg = get_stage4_config()

    train_anno_path = overrides.pop("train_anno", "data/sample/annotations_train.json")
    val_anno_path = overrides.pop("val_anno", "data/sample/annotations_val.json")
    stage1_ckpt = overrides.pop("stage1_ckpt", None)
    stage2_ckpt = overrides.pop("stage2_ckpt", None)
    stage3_ckpt = overrides.pop("stage3_ckpt", None)
    root = overrides.pop("root", _YOAKE_TRYAL)
    output_dir = overrides.pop("output_dir", f"{root}/outputs/stage4")
    num_epochs = int(overrides.pop("num_epochs", 30))
    batch_size = int(overrides.pop("batch_size", cfg.data.batch_size))
    resume = overrides.pop("resume", None)
    seed = int(overrides.pop("seed", 42))

    if overrides:
        cfg = cfg.merge(overrides)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 4 — Unified fine-tuning (all modules)")
    print(f"  Stage1 ckpt: {stage1_ckpt}")
    print(f"  Stage2 ckpt: {stage2_ckpt}")
    print(f"  Stage3 ckpt: {stage3_ckpt}")
    print(f"  Output: {output_dir}")

    train_anno = load_annotation(train_anno_path)
    val_anno = load_annotation(val_anno_path) if Path(val_anno_path).exists() else None

    cfg.data.batch_size = batch_size
    loaders = build_dataloaders(
        train_anno=train_anno,
        val_anno=val_anno,
        stage=4,
        cfg=cfg.data,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    model = build_model(cfg.model)
    model.to(device)

    param_info = model.num_parameters()
    print(f"Parameters: total={param_info['total']:,}")

    trainer = Stage4Trainer(cfg, model, device, output_dir)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        resume=resume,
        stage1_ckpt=stage1_ckpt,
        stage2_ckpt=stage2_ckpt,
        stage3_ckpt=stage3_ckpt,
    )

    print("Stage 4 training complete.")
    print(f"Best checkpoint: {output_dir}/checkpoint_best.pth")


if __name__ == "__main__":
    main()
