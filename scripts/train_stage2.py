"""
train_stage2.py — Stage 2: Action Head Pretraining

Stage 1 で学習した detector の重みをロードし、
Hierarchical Temporal Module + Action Head を学習する。

使い方:
    python scripts/train_stage2.py [config.yaml] [key=value ...]

    例:
    python scripts/train_stage2.py \\
        train.resume=outputs/stage1/stage1_best.pth \\
        train.max_epochs=80
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from typing import Dict, Any

from torch.utils.data import DataLoader

from htrtdetr.config import get_stage2_config, HTRTDETRConfig
from htrtdetr.data import (
    DummyDataset, SlidingWindowDataset, load_annotations, get_collate_fn
)
from htrtdetr.models import build_model
from htrtdetr.training import Trainer
from htrtdetr.utils import set_seed, get_logger, load_model_weights


def parse_argv(argv):
    config_path, overrides = "", {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try: v = int(v)
            except:
                try: v = float(v)
                except: pass
            keys = k.split(".")
            d = overrides
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
        elif not config_path and arg.endswith((".yaml", ".yml")):
            config_path = arg
    return config_path, overrides


def main() -> None:
    config_path, overrides = parse_argv(sys.argv)

    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = get_stage2_config(overrides if overrides else None)

    logger = get_logger(
        "stage2",
        log_file=str(Path(cfg.train.output_dir) / "stage2.log"),
    )
    logger.info("Stage 2: Action Head Training")

    set_seed(cfg.train.seed)

    # ----- Dataset (sequence mode) -----
    use_dummy = not (
        Path(cfg.data.train_root).exists() and
        any(Path(cfg.data.train_root).iterdir())
    )

    if use_dummy:
        logger.warning("Using DummyDataset (sequence mode)")
        train_dataset = DummyDataset(
            n_samples=100, window_size=cfg.data.window_size,
            image_size=tuple(cfg.data.image_size),
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="sequence",
        )
        val_dataset = DummyDataset(
            n_samples=20, window_size=cfg.data.window_size,
            image_size=tuple(cfg.data.image_size),
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="sequence",
        )
    else:
        train_ann = str(Path(cfg.data.train_root) / "annotations.json")
        val_ann = str(Path(cfg.data.val_root) / "annotations.json")
        train_videos, _, _ = load_annotations(train_ann)
        val_videos, _, _ = load_annotations(val_ann)

        train_dataset = SlidingWindowDataset(
            train_videos, window_size=cfg.data.window_size,
            stride=cfg.data.window_stride, image_size=tuple(cfg.data.image_size),
            augment=cfg.data.augment_train, require_action=True,
            data_root=cfg.data.train_root,
        )
        val_dataset = SlidingWindowDataset(
            val_videos, window_size=cfg.data.window_size,
            stride=cfg.data.window_size, image_size=tuple(cfg.data.image_size),
            augment=False, data_root=cfg.data.val_root,
        )

    collate_fn = get_collate_fn("sequence")
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    # ----- Model -----
    model = build_model(cfg.model)
    model.set_stage(2)

    # Stage 1 の detector 重みをロード
    stage1_path = "outputs/stage1/stage1_best.pth"
    if Path(stage1_path).exists():
        logger.info(f"Loading Stage 1 weights from {stage1_path}")
        load_model_weights(stage1_path, model, strict=False, prefix_to_remove="")
    else:
        logger.warning(f"Stage 1 weights not found: {stage1_path}. Training from scratch.")

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader,
        train_cfg=cfg.train, loss_cfg=cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
    )
    trainer.train()
    logger.info(f"Stage 2 complete. Best: {cfg.train.output_dir}/stage2_best.pth")


if __name__ == "__main__":
    main()
