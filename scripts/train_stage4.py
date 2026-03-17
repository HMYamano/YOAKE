"""
train_stage4.py — Stage 4: Unified Fine-tuning

全モジュールを統合して fine-tune する。

使い方:
    python scripts/train_stage4.py [config.yaml] [key=value ...]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

from torch.utils.data import DataLoader

from htrtdetr.config import get_stage4_config, HTRTDETRConfig
from htrtdetr.data import DummyDataset, SlidingWindowDataset, load_annotations, get_collate_fn
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
        cfg = get_stage4_config(overrides if overrides else None)

    logger = get_logger("stage4", log_file=str(Path(cfg.train.output_dir) / "stage4.log"))
    logger.info("Stage 4: Unified Fine-tuning")
    set_seed(cfg.train.seed)

    use_dummy = not (
        Path(cfg.data.train_root).exists() and
        any(Path(cfg.data.train_root).iterdir())
    )

    if use_dummy:
        logger.warning("Using DummyDataset")
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
            augment=cfg.data.augment_train, data_root=cfg.data.train_root,
        )
        val_dataset = SlidingWindowDataset(
            val_videos, window_size=cfg.data.window_size,
            stride=cfg.data.window_size, image_size=tuple(cfg.data.image_size),
            augment=False, data_root=cfg.data.val_root,
        )

    collate_fn = get_collate_fn("sequence")
    _nw = cfg.data.num_workers
    _loader_kwargs = dict(
        pin_memory=cfg.data.pin_memory,
        persistent_workers=(_nw > 0 and cfg.data.persistent_workers),
        prefetch_factor=(cfg.data.prefetch_factor if _nw > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=_nw, collate_fn=collate_fn, drop_last=True, **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=_nw, collate_fn=collate_fn, **_loader_kwargs,
    )

    model = build_model(cfg.model)
    model.set_stage(4)

    # Stage 3 の重みをロード
    for ckpt_path in [
        "outputs/stage3/stage3_best.pth",
        "outputs/stage2/stage2_best.pth",
        "outputs/stage1/stage1_best.pth",
    ]:
        if Path(ckpt_path).exists():
            logger.info(f"Loading weights from {ckpt_path}")
            load_model_weights(ckpt_path, model, strict=False)
            break

    trainer = Trainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader,
        train_cfg=cfg.train, loss_cfg=cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
    )
    trainer.train()

    logger.info("Stage 4 complete. Full model saved.")
    logger.info("=== Staged Training Complete ===")


if __name__ == "__main__":
    main()
