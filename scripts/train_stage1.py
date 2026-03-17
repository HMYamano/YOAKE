"""
train_stage1.py — Stage 1: Detector Pretraining / Finetuning

使い方:
    # デフォルト設定で学習 (YOAKE_tryal のデータを自動検出)
    python scripts/train_stage1.py

    # アノテーションファイルを直接指定
    python scripts/train_stage1.py \
        train_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/train/annotations.json \
        val_anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/val/annotations.json

    # その他のパラメータを上書き
    python scripts/train_stage1.py \
        train_anno=path/to/train.json \
        val_anno=path/to/val.json \
        train.max_epochs=50 data.batch_size=8

argparse は使わない。sys.argv からシンプルに設定ファイルと上書き値を読む。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from typing import Dict, Any, Optional

from torch.utils.data import DataLoader

from htrtdetr.config import get_stage1_config, HTRTDETRConfig
from htrtdetr.data import DummyDataset, SingleFrameDataset, load_annotations, get_collate_fn
from htrtdetr.models import build_model
from htrtdetr.training import Trainer
from htrtdetr.utils import set_seed, get_logger


# ---------------------------------------------------------------------------
# デフォルトデータパス
# ---------------------------------------------------------------------------

_YOAKE_TRYAL = "C:/Users/hayam/Desktop/YOAKE_tryal"
DEFAULT_TRAIN_ANNO = f"{_YOAKE_TRYAL}/data/train/annotations.json"
DEFAULT_VAL_ANNO   = f"{_YOAKE_TRYAL}/data/val/annotations.json"


# ---------------------------------------------------------------------------
# argv パーサー (argparse なし)
# ---------------------------------------------------------------------------

def parse_argv(argv: list) -> tuple[str, Dict[str, Any]]:
    """
    argv を解析して (config_path, overrides) を返す。

    train_anno= / val_anno= はスクリプトレベルのキーとして特別扱い。
    それ以外の key=value はネストキー ("train.max_epochs") として config に渡す。
    """
    config_path = ""
    overrides: Dict[str, Any] = {}

    args = argv[1:]
    for arg in args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # train_anno / val_anno は文字列のまま保持
            if k in ("train_anno", "val_anno"):
                overrides[k] = v
                continue
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

    # スクリプトレベルのキーを先に取り出す
    train_anno_path: Optional[str] = overrides.pop("train_anno", None)
    val_anno_path:   Optional[str] = overrides.pop("val_anno",   None)

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

    # ----- アノテーションパスの解決 -----
    # 優先順位: コマンドライン引数 > デフォルトパス (YOAKE_tryal) > DummyDataset
    if train_anno_path is None and Path(DEFAULT_TRAIN_ANNO).exists():
        train_anno_path = DEFAULT_TRAIN_ANNO
    if val_anno_path is None and Path(DEFAULT_VAL_ANNO).exists():
        val_anno_path = DEFAULT_VAL_ANNO

    # ----- Dataset -----
    _img_size = cfg.data.image_size
    if isinstance(_img_size, int):
        _img_size = (_img_size, _img_size)
    else:
        _img_size = tuple(_img_size)

    use_dummy = train_anno_path is None or not Path(train_anno_path).exists()

    if use_dummy:
        logger.warning(
            "Train annotation not found. Using DummyDataset for smoke-test.\n"
            f"  指定方法: python scripts/train_stage1.py train_anno=<path/to/train.json>"
        )
        train_dataset = DummyDataset(
            n_samples=100,
            window_size=1,
            image_size=_img_size,
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="single",
        )
        val_dataset = DummyDataset(
            n_samples=20,
            window_size=1,
            image_size=_img_size,
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="single",
        )
    else:
        logger.info(f"Train annotation: {train_anno_path}")

        train_videos, _, _ = load_annotations(train_anno_path)

        # val アノテーションがなければ train を流用 (過学習確認用)
        if val_anno_path and Path(val_anno_path).exists():
            logger.info(f"Val   annotation: {val_anno_path}")
            val_videos, _, _ = load_annotations(val_anno_path)
        else:
            logger.warning("Val annotation not found. Using train data as val.")
            val_videos = train_videos

        # アノテーション内の image_path が絶対パスの場合は data_root="" でよい
        first_path = train_videos[0].frames[0].image_path if train_videos else ""
        data_root = "" if Path(first_path).is_absolute() else cfg.data.train_root

        train_dataset = SingleFrameDataset(
            train_videos,
            image_size=_img_size,
            augment=cfg.data.augment_train,
            data_root=data_root,
        )
        val_dataset = SingleFrameDataset(
            val_videos,
            image_size=_img_size,
            augment=False,
            data_root=data_root,
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
