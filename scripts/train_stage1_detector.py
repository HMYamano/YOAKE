"""
train_stage1_detector.py — Stage 1: Spatial Detector Training  [DEPRECATED]

.. deprecated::
   このスクリプトは stage_trainers.py の Stage1Trainer を使う旧系統です。
   正式な学習経路は以下を使用してください:

     yoake train stage=1 [key=value ...]
     # または
     python scripts/train_stage1.py [key=value ...]

   このファイルは互換性のために残してありますが、今後削除される予定です。

使い方 (旧):
  python scripts/train_stage1_detector.py \
      train_anno=data/sample/annotations_train.json \
      val_anno=data/sample/annotations_val.json \
      output_dir=runs/train/stage1 \
      num_epochs=50 \
      batch_size=8 \
      resume=runs/train/stage1/checkpoint_last.pth
"""

from __future__ import annotations

import warnings
warnings.warn(
    "train_stage1_detector.py は非推奨です。"
    "代わりに 'yoake train stage=1' または 'python scripts/train_stage1.py' を使用してください。",
    DeprecationWarning,
    stacklevel=1,
)

import sys
from pathlib import Path

# project root を sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import HTRTDETRConfig, get_stage1_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.training.stage_trainers import Stage1Trainer
from htrtdetr.utils.misc import set_seed

_YOAKE_TRYAL = str(Path(__file__).resolve().parent.parent)


def parse_overrides(argv) -> dict:
    """key=value 形式の引数を dict に変換する"""
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # 数値の自動変換
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass  # 文字列のまま
            overrides[k] = v
    return overrides


def main():
    overrides = parse_overrides(sys.argv[1:])

    # Config 構築
    cfg = get_stage1_config()

    # コマンドライン上書き
    train_anno_path = overrides.pop("train_anno", "data/sample/annotations_train.json")
    val_anno_path = overrides.pop("val_anno", "data/sample/annotations_val.json")
    root = overrides.pop("root", _YOAKE_TRYAL)
    output_dir = overrides.pop("output_dir", f"{root}/outputs/stage1")
    num_epochs = int(overrides.pop("num_epochs", 50))
    batch_size = int(overrides.pop("batch_size", cfg.data.batch_size))
    resume = overrides.pop("resume", None)
    seed = int(overrides.pop("seed", 42))

    # 残りは cfg.merge() で適用
    if overrides:
        cfg = cfg.merge(overrides)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 1 — Detector training")
    print(f"Train: {train_anno_path}")
    print(f"Val:   {val_anno_path}")
    print(f"Output: {output_dir}")

    # アノテーション読み込み
    train_anno = load_annotation(train_anno_path)
    val_anno = load_annotation(val_anno_path) if Path(val_anno_path).exists() else None

    # DataLoader
    cfg.data.batch_size = batch_size
    loaders = build_dataloaders(
        train_anno=train_anno,
        val_anno=val_anno,
        stage=1,
        cfg=cfg.data,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    # モデル構築
    model = build_model(cfg.model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    # Trainer
    trainer = Stage1Trainer(cfg, model, device, output_dir)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        resume=resume,
    )

    print("Stage 1 training complete.")
    print(f"Best checkpoint: {output_dir}/checkpoint_best.pth")


if __name__ == "__main__":
    main()
