"""
train_stage2_action.py — Stage 2: Action Head Training (Detector Frozen)  [DEPRECATED]

.. deprecated::
   このスクリプトは stage_trainers.py の Stage2Trainer を使う旧系統です。
   正式な学習経路は以下を使用してください:

     yoake train stage=2 [key=value ...]
     # または
     python scripts/train_stage2.py [key=value ...]

   このファイルは互換性のために残してありますが、今後削除される予定です。
"""

import warnings
warnings.warn(
    "train_stage2_action.py は非推奨です。"
    "代わりに 'yoake train stage=2' または 'python scripts/train_stage2.py' を使用してください。",
    DeprecationWarning,
    stacklevel=1,
)

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import HTRTDETRConfig, get_stage2_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.training.stage_trainers import Stage2Trainer
from htrtdetr.utils.misc import set_seed, load_model_weights

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

    cfg = get_stage2_config()

    train_anno_path = overrides.pop("train_anno", "data/sample/annotations_train.json")
    val_anno_path = overrides.pop("val_anno", "data/sample/annotations_val.json")
    stage1_ckpt = overrides.pop("stage1_ckpt", None)
    root = overrides.pop("root", _YOAKE_TRYAL)
    output_dir = overrides.pop("output_dir", f"{root}/outputs/stage2")
    num_epochs = int(overrides.pop("num_epochs", 30))
    batch_size = int(overrides.pop("batch_size", cfg.data.batch_size))
    resume = overrides.pop("resume", None)
    seed = int(overrides.pop("seed", 42))

    if overrides:
        cfg = cfg.merge(overrides)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 2 — Action Head training (detector frozen)")
    print(f"Stage1 checkpoint: {stage1_ckpt}")
    print(f"Output: {output_dir}")

    train_anno = load_annotation(train_anno_path)
    val_anno = load_annotation(val_anno_path) if Path(val_anno_path).exists() else None

    cfg.data.batch_size = batch_size
    loaders = build_dataloaders(
        train_anno=train_anno,
        val_anno=val_anno,
        stage=2,
        cfg=cfg.data,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    model = build_model(cfg.model)
    model.to(device)

    # Stage 1 checkpoint から detector 重みをロード
    if stage1_ckpt and Path(stage1_ckpt).exists():
        load_model_weights(model, stage1_ckpt, strict=False)
        print(f"Loaded stage1 weights: {stage1_ckpt}")
    else:
        if stage1_ckpt:
            print(f"Warning: stage1 checkpoint not found: {stage1_ckpt}")

    trainer = Stage2Trainer(cfg, model, device, output_dir)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        resume=resume,
    )

    print("Stage 2 training complete.")
    print(f"Best checkpoint: {output_dir}/checkpoint_best.pth")


if __name__ == "__main__":
    main()
