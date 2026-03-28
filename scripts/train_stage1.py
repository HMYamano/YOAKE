"""
train_stage1.py — Stage 1: Detector Pretraining / Finetuning  [DEPRECATED]

.. deprecated::
   このスクリプトは非推奨です。代わりに yoake CLI を使用してください:

     yoake train stage=1 data.train_root=data/train data.val_root=data/val

   出力は runs/train/stage1/ に保存されます。

使い方:
    # デフォルト設定で学習 (YOAKE_tryal のデータを自動検出)
    python scripts/train_stage1.py

    # アノテーションファイルを直接指定
    python scripts/train_stage1.py \
        train_anno=C:/Users/utopi/Desktop/YOAKE_tryal/data/train/annotations.json \
        val_anno=C:/Users/utopi/Desktop/YOAKE_tryal/data/val/annotations.json

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
from htrtdetr.data import (
    DummyDataset, SingleFrameDataset, load_annotations,
    validate_image_paths, get_collate_fn,
)
from htrtdetr.models import build_model
from htrtdetr.training import Trainer
from htrtdetr.utils import set_seed, get_logger


# ---------------------------------------------------------------------------
# デフォルトデータパス
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _ask_confirm(message: str) -> bool:
    """
    コマンドプロンプトで Yes / No をユーザーに尋ねる。
    y / yes のときのみ True を返す。EOF / Ctrl-C は No 扱い。
    """
    try:
        resp = input(f"{message} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return resp in ("y", "yes")


def _resolve_anno_path(
    explicit: Optional[str],
    candidates: list,
    split: str,
    logger,
) -> Optional[str]:
    """
    アノテーションパスを解決する。

    1. explicit が指定されていればそのまま返す (存在チェックあり)
    2. candidates を順番に試して最初に見つかったものを返す
    3. 見つからなければ None を返す

    Args:
        explicit:   CLI で明示されたパス (None のとき候補を探す)
        candidates: 自動探索するパス候補のリスト
        split:      "train" / "val" 等のラベル (ログ用)
        logger:     ロガー

    Returns:
        解決されたパス、または None
    """
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            logger.info(f"[{split}] annotation: {explicit}")
            return explicit
        logger.warning(f"[{split}] Specified annotation not found: {explicit}")
        return None

    for candidate in candidates:
        if Path(candidate).exists():
            logger.info(f"[{split}] annotation (auto-detected): {candidate}")
            return candidate

    return None


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
            # 文字列のまま保持するスクリプトレベルキー
            if k in ("train_anno", "val_anno", "root"):
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
    root:            str           = overrides.pop("root",       _DEFAULT_ROOT)
    train_anno_path: Optional[str] = overrides.pop("train_anno", None)
    val_anno_path:   Optional[str] = overrides.pop("val_anno",   None)

    # Config 構築
    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = get_stage1_config(overrides if overrides else None)

    # output_dir が overrides で明示されていなければ root から構築
    # ただし config_path が指定されている場合は yaml の値を優先する
    if not overrides.get("train", {}).get("output_dir"):
        if not config_path or not cfg.train.output_dir:
            cfg.train.output_dir = f"{root}/runs/train/stage1"

    logger = get_logger(
        "stage1",
        log_file=str(Path(cfg.train.output_dir) / "stage1.log"),
    )
    logger.info(f"Stage 1: Detector Training | config: {config_path or 'default'}")
    logger.info(f"Root   : {root}")
    logger.info(f"Output : {cfg.train.output_dir}")

    set_seed(cfg.train.seed, cfg.train.deterministic)

    # ----- アノテーションパスの解決 -----
    # 優先順位: コマンドライン引数 > root 配下のデフォルトパス > cfg.data.*_root 配下
    # どこにも見つからない場合はユーザーに確認を求める

    train_anno_path = _resolve_anno_path(
        explicit=train_anno_path,
        candidates=[
            f"{root}/data/train/annotations.json",
            str(Path(cfg.data.train_root) / "annotations.json"),
        ],
        split="train",
        logger=logger,
    )
    val_anno_path = _resolve_anno_path(
        explicit=val_anno_path,
        candidates=[
            f"{root}/data/val/annotations.json",
            str(Path(cfg.data.val_root) / "annotations.json"),
        ],
        split="val",
        logger=logger,
    )

    # ----- DummyDataset 確認 -----
    use_dummy = train_anno_path is None

    if use_dummy:
        logger.warning(
            f"Train annotation not found.\n"
            f"  Tried: {root}/data/train/annotations.json\n"
            f"         {Path(cfg.data.train_root) / 'annotations.json'}\n"
            f"  指定方法: python scripts/train_stage1.py train_anno=<path/to/train.json>"
        )
        if not _ask_confirm("Continue with DummyDataset for smoke-test?"):
            logger.error("Aborted. Please specify: train_anno=<path/to/annotations.json>")
            sys.exit(1)

    if val_anno_path is None and not use_dummy:
        logger.warning(
            f"Val annotation not found.\n"
            f"  Tried: {root}/data/val/annotations.json\n"
            f"         {Path(cfg.data.val_root) / 'annotations.json'}"
        )
        if not _ask_confirm("Continue without val annotation (use train data as val)?"):
            logger.error("Aborted. Please specify: val_anno=<path/to/annotations.json>")
            sys.exit(1)

    # ----- Dataset -----
    _img_size = cfg.data.image_size
    if isinstance(_img_size, int):
        _img_size = (_img_size, _img_size)
    else:
        _img_size = tuple(_img_size)

    if use_dummy:
        logger.warning("Using DummyDataset (smoke-test mode). AP50 は意味を持ちません。")
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
        train_videos, _, _ = load_annotations(train_anno_path)

        # val アノテーション: 見つかれば読込、なければ train を流用 (確認済み)
        _val_uses_train_data = False
        if val_anno_path is not None:
            val_videos, _, _ = load_annotations(val_anno_path)
        else:
            logger.warning("Val annotation not found. Using train data as val.")
            val_videos = train_videos
            _val_uses_train_data = True

        # ----- data_root の解決 (train / val それぞれ独立して決める) -----
        # アノテーション内の image_path が絶対パスなら data_root="" でよい
        first_train_path = train_videos[0].frames[0].image_path if train_videos else ""
        train_data_root = (
            "" if Path(first_train_path).is_absolute() else cfg.data.train_root
        )

        if _val_uses_train_data:
            # val が train データを流用している場合は train の root を使う
            val_data_root = train_data_root
        else:
            first_val_path = val_videos[0].frames[0].image_path if val_videos else ""
            val_data_root = (
                "" if Path(first_val_path).is_absolute() else cfg.data.val_root
            )

        logger.info(f"train_data_root: '{train_data_root}'")
        logger.info(f"val_data_root  : '{val_data_root}'")

        # ----- 画像パスの事前検証 -----
        _FAIL_THRESH = 0.5   # 50% 以上欠損でエラー
        _CHECK_N     = 30    # 先頭 30 件チェック

        logger.info("[train] Validating image paths ...")
        n_found, n_checked, missing = validate_image_paths(
            train_videos, train_data_root, split="train",
            check_n=_CHECK_N, fail_threshold=_FAIL_THRESH,
        )
        logger.info(f"[train] {n_found}/{n_checked} images found")
        if missing:
            logger.warning(f"[train] {len(missing)} missing (first: {missing[0]})")

        if not _val_uses_train_data:
            logger.info("[val] Validating image paths ...")
            n_found_v, n_checked_v, missing_v = validate_image_paths(
                val_videos, val_data_root, split="val",
                check_n=_CHECK_N, fail_threshold=_FAIL_THRESH,
            )
            logger.info(f"[val] {n_found_v}/{n_checked_v} images found")
            if missing_v:
                logger.warning(f"[val] {len(missing_v)} missing (first: {missing_v[0]})")

        train_dataset = SingleFrameDataset(
            train_videos,
            image_size=_img_size,
            augment=cfg.data.augment_train,
            data_root=train_data_root,
            strict=True,
        )
        val_dataset = SingleFrameDataset(
            val_videos,
            image_size=_img_size,
            augment=False,
            data_root=val_data_root,
            strict=True,
        )

    collate_fn = get_collate_fn("single")
    _nw = cfg.data.num_workers
    _loader_kwargs = dict(
        pin_memory=cfg.data.pin_memory,
        persistent_workers=(_nw > 0 and cfg.data.persistent_workers),
        prefetch_factor=(cfg.data.prefetch_factor if _nw > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=_nw,
        collate_fn=collate_fn,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=_nw,
        collate_fn=collate_fn,
        **_loader_kwargs,
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
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        use_dummy=use_dummy,
    )

    trainer.train()

    logger.info(f"Stage 1 complete. Best model: {cfg.train.output_dir}/stage1_best.pth")


if __name__ == "__main__":
    main()
