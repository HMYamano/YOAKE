"""
train_stage1.py — Stage 1: Detector Pretraining / Finetuning

使い方:
    # デフォルト設定で学習
    python scripts/train_stage1.py

    # config yaml を指定
    python scripts/train_stage1.py configs/stage1_detector.yaml

    # yaml を起点にいくつかの値を上書き
    python scripts/train_stage1.py configs/stage1_detector.yaml \
        train.max_epochs=50 data.batch_size=8

argparse は使わない。sys.argv からシンプルに設定ファイルと上書き値を読む。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from typing import Dict, Any

from torch.utils.data import DataLoader

from htrtdetr.config import get_stage1_config, HTRTDETRConfig
from htrtdetr.data import DummyDataset, SingleFrameDataset, load_annotations, get_collate_fn
from htrtdetr.models import build_model
from htrtdetr.training import Trainer
from htrtdetr.utils import set_seed, get_logger


# ---------------------------------------------------------------------------
# argv パーサー (argparse なし)
# ---------------------------------------------------------------------------

def parse_argv(argv: list) -> tuple[str, Dict[str, Any]]:
    """
    argv を解析して (config_path, overrides) を返す。

    例:
        train_stage1.py                         → ("", {})
        train_stage1.py my.yaml                 → ("my.yaml", {})
        train_stage1.py my.yaml key=value       → ("my.yaml", {"key": value})
    """
    config_path = ""
    overrides: Dict[str, Any] = {}

    args = argv[1:]
    for arg in args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # 簡易型変換
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            # ネストキー ("train.max_epochs") を dict に変換
            keys = k.split(".")
            d = overrides
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
        elif not config_path and arg.endswith((".yaml", ".yml")):
            config_path = arg

    return config_path, overrides


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path, overrides = parse_argv(sys.argv)

    # Config 構築
    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = get_stage1_config(overrides if overrides else None)

    logger = get_logger(
        "stage1",
        log_file=str(Path(cfg.train.output_dir) / "stage1.log"),
    )
    logger.info(f"Stage 1: Detector Training | config: {config_path or 'default'}")
    logger.info(f"Output: {cfg.train.output_dir}")

    set_seed(cfg.train.seed, cfg.train.deterministic)

    # ----- Dataset -----
    use_dummy = not (
        Path(cfg.data.train_root).exists() and
        any(Path(cfg.data.train_root).iterdir())
    )

    if use_dummy:
        logger.warning(
            f"Train data not found at '{cfg.data.train_root}'. "
            "Using DummyDataset for smoke-test."
        )
        train_dataset = DummyDataset(
            n_samples=100,
            window_size=1,
            image_size=tuple(cfg.data.image_size),
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="single",
        )
        val_dataset = DummyDataset(
            n_samples=20,
            window_size=1,
            image_size=tuple(cfg.data.image_size),
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="single",
        )
    else:
        # 実データ読み込み
        train_ann = str(Path(cfg.data.train_root) / "annotations.json")
        val_ann = str(Path(cfg.data.val_root) / "annotations.json")

        train_videos, _, _ = load_annotations(train_ann)
        val_videos, _, _ = load_annotations(val_ann)

        train_dataset = SingleFrameDataset(
            train_videos,
            image_size=tuple(cfg.data.image_size),
            augment=cfg.data.augment_train,
            data_root=cfg.data.train_root,
        )
        val_dataset = SingleFrameDataset(
            val_videos,
            image_size=tuple(cfg.data.image_size),
            augment=False,
            data_root=cfg.data.val_root,
        )

    collate_fn = get_collate_fn("single")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # ----- Model -----
    model = build_model(cfg.model)
    logger.info(f"Model parameters: {model.num_parameters()}")

    # Stage 1 設定
    model.set_stage(1)

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.train,
        loss_cfg=cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
    )

    trainer.train()

    logger.info(f"Stage 1 complete. Best model: {cfg.train.output_dir}/stage1_best.pth")


if __name__ == "__main__":
    main()
