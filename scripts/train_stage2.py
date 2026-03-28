"""
train_stage2.py — Stage 2: Action Head Pretraining  [DEPRECATED]

.. deprecated::
   このスクリプトは非推奨です。代わりに yoake CLI を使用してください:

     yoake train stage=2

   出力は runs/train/stage2/ に保存されます。

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

from htrtdetr.config import get_stage2_config, HTRTDETRConfig, clamp_temporal_branches
from htrtdetr.data import (
    DummyDataset, SlidingWindowDataset, load_annotations, get_collate_fn
)
from htrtdetr.models import build_model
from htrtdetr.training import Trainer
from htrtdetr.utils import set_seed, get_logger, load_model_weights

_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)


def _ask_confirm(message: str) -> bool:
    try:
        resp = input(f"{message} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return resp in ("y", "yes")


def parse_argv(argv):
    config_path, overrides = "", {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
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


def _find_checkpoint(root: str, stage: int) -> str:
    """outputs 以下を再帰検索して stage{N}_best.pth を見つける。

    優先順位:
      1. outputs/*/stage{N}/stage{N}_best.pth  (バリアント別: large / medium / small)
      2. outputs/stage{N}/stage{N}_best.pth     (フラット構造・旧形式)

    複数候補がある場合は最終更新日時が最新のものを返す。
    見つからなければ空文字列を返す。
    """
    fname = f"stage{stage}_best.pth"
    candidates = sorted(
        [
            *Path(root).glob(f"runs/train/*/stage{stage}/{fname}"),
            *Path(root).glob(f"runs/train/stage{stage}/{fname}"),
            *Path(root).glob(f"outputs/*/stage{stage}/{fname}"),  # legacy
            *Path(root).glob(f"outputs/stage{stage}/{fname}"),    # legacy
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return ""


def main() -> None:
    config_path, overrides = parse_argv(sys.argv)

    root = overrides.pop("root", _DEFAULT_ROOT)

    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = get_stage2_config(overrides if overrides else None)

    if not overrides.get("train", {}).get("output_dir"):
        cfg.train.output_dir = f"{root}/runs/train/stage2"

    # window_size が small な場合に temporal branch の num_frames を自動クリップ
    clamp_temporal_branches(cfg)

    logger = get_logger(
        "stage2",
        log_file=str(Path(cfg.train.output_dir) / "stage2.log"),
    )
    logger.info("Stage 2: Action Head Training")
    logger.info(f"Root   : {root}")

    set_seed(cfg.train.seed)

    # ----- Dataset (sequence mode) -----
    use_dummy = not (
        Path(cfg.data.train_root).exists() and
        any(Path(cfg.data.train_root).iterdir())
    )

    if use_dummy:
        logger.warning(
            f"Train data not found: {cfg.data.train_root}\n"
            f"  指定方法: python scripts/train_stage2.py data.train_root=<path/to/train>"
        )
        if not _ask_confirm("Continue with DummyDataset for smoke-test?"):
            logger.error("Aborted. Please specify: data.train_root=<path/to/train>")
            sys.exit(1)
        logger.warning("Using DummyDataset (sequence mode). メトリクスは意味を持ちません。")
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

    # ----- Model -----
    model = build_model(cfg.model)
    model.set_stage(2)

    # Stage 1 の detector 重みをロード
    # train.resume が指定されていれば Trainer が処理するためここではスキップ
    if not cfg.train.resume:
        stage1_path = _find_checkpoint(root, stage=1)
        if stage1_path:
            logger.info(f"Loading Stage 1 weights from {stage1_path}")
            load_model_weights(stage1_path, model, strict=False, prefix_to_remove="")
        else:
            logger.warning("Stage 1 weights not found. Training from scratch.")

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader,
        train_cfg=cfg.train, loss_cfg=cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        use_dummy=use_dummy,
    )
    trainer.train()
    logger.info(f"Stage 2 complete. Best: {cfg.train.output_dir}/stage2_best.pth")


if __name__ == "__main__":
    main()
