"""
train_stage1_detector.py — Stage 1: Spatial Detector Training

使い方:
  python scripts/train_stage1_detector.py \
      train_anno=data/sample/annotations_train.json \
      val_anno=data/sample/annotations_val.json \
      output_dir=outputs/stage1 \
      num_epochs=50 \
      batch_size=8 \
      resume=outputs/stage1/checkpoint_last.pth

すべての設定は dataclass (HTRTDETRConfig) で管理する。
コマンドライン引数は key=value 形式で config を上書きできる。
"""

from __future__ import annotations

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
    output_dir = overrides.pop("output_dir", "outputs/stage1")
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
