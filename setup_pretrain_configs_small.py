# setup_pretrain_configs_small.py
# yoake-small (ResNet-18 backbone, feature_dim=128) 用の設定ファイルを生成する
# 実行: python C:/Users/utopi/YOAKE/setup_pretrain_configs_small.py
# 生成先: C:/Users/utopi/YOAKE_small_pretrain/configs/

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from htrtdetr.config.config import get_variant_config

WORK = "C:/Users/utopi/YOAKE_small_pretrain"  # ← 自分の環境に合わせて変更
VARIANT = "small"                          # ResNet-18, feature_dim=128, ~15-20M params


def make_stage1_config(num_classes: int = 125):
    """
    Stage 1: 物体検出事前学習
      - データ: COCO (330K) + AP-10K (10K) を YOAKE 形式にマージしたもの
      - 入力: 単フレーム (B, 3, H, W)
      - 学習対象: backbone / FPN / decoder
      - 凍結: temporal / ID head / action head
      - 損失: cls_loss + bbox_l1 + giou

    num_classes: マージ後の実際のクラス数。
                 merge_stage1_annotations.py 実行後に確認して更新すること。
                 python -c "import json; d=json.load(open(r'C:/Users/utopi/YOAKE_small_pretrain/data/stage1/train/annotations.json')); print(len(d['class_names']))"
    """
    cfg = get_variant_config(VARIANT, stage=1, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage1/train",
            "val_root":   f"{WORK}/data/stage1/val",
            "batch_size": 64,          # small は large の倍 (RTX 8000 で余裕あり)
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 1,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 1,
            "max_epochs": 24,
            "early_stopping_patience": 10,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/small/stage1",
            "log_interval": 100,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.1,   # backbone は 1/10 の lr
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 24,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 1,
            "detector": {
                "head": {
                    "num_classes": num_classes,  # COCO(80) + AP-10K 追加クラス
                    "num_queries": 300,
                },
            },
            "action_head": {"num_actions": 5},
            "id_head": {"max_ids": 50},
        },
        "loss": {
            "w_class": 2.0,
            "w_bbox_l1": 5.0,
            "w_bbox_giou": 2.0,
            "w_action": 0.0,   # Stage 1: action loss 無効
            "w_id_cls": 0.0,   # Stage 1: ID loss 無効
        },
    })
    cfg.save_yaml(f"{WORK}/configs/small_stage1.yaml")
    print(f"Saved: {WORK}/configs/small_stage1.yaml")


def make_stage2_config(num_actions: int = 140):
    """
    Stage 2: 行動認識事前学習
      - データ: Animal Kingdom (140 actions) + CalMS21
      - 入力: シーケンス (B, T, 3, H, W), T=16
      - 学習対象: temporal module + action head
      - 凍結: detector (backbone_lr_factor=0.0)
      - 損失: action_loss

    num_actions: Animal Kingdom の実際のアクションクラス数。
                 convert 後に確認すること。
                 python -c "import json; d=json.load(open(r'C:/Users/utopi/YOAKE_small_pretrain/data/stage2/train/annotations.json')); print(len(d['action_names']))"
    """
    cfg = get_variant_config(VARIANT, stage=2, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage2/train",
            "val_root":   f"{WORK}/data/stage2/val",
            "batch_size": 16,          # small: large(8) の倍
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 2,
            "max_epochs": 15,
            "early_stopping_patience": 8,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/small/stage2",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.0,   # detector は凍結
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 2,
            "total_epochs": 15,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 2,
            "detector": {
                "head": {
                    "num_classes": 1,    # Stage 2: 行動学習なので汎用1クラス
                    "num_queries": 100,
                },
            },
            "action_head": {
                "num_actions": num_actions,   # Animal Kingdom の全行動クラス数
            },
            "id_head": {"max_ids": 50},
        },
        "loss": {
            "w_class": 0.0,       # Stage 2: detection loss 無効
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
            "w_action": 1.0,
            "w_id_cls": 0.0,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/small_stage2.yaml")
    print(f"Saved: {WORK}/configs/small_stage2.yaml")


def make_stage3_config():
    """
    Stage 3: トラッキング事前学習
      - データ: DanceTrack + AnimalTrack (+ BEE23 オプション)
      - 入力: シーケンス (B, T, 3, H, W), T=16
      - 学習対象: temporal module + ID head
      - 凍結: detector
      - 損失: id_cls + metric_loss
    """
    cfg = get_variant_config(VARIANT, stage=3, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage3/train",
            "val_root":   f"{WORK}/data/stage3/val",
            "batch_size": 16,          # small: large(8) の倍
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 3,
            "max_epochs": 30,
            "early_stopping_patience": 15,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/small/stage3",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.1,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "total_epochs": 30,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 3,
            "detector": {
                "head": {"num_classes": 1, "num_queries": 100},
            },
            "action_head": {"num_actions": 5},
            "id_head": {
                "max_ids": 100,
                "use_metric_loss": True,
                "metric_loss_margin": 0.3,
                "memory_ttl": 30,
            },
        },
        "loss": {
            "w_class": 0.0,
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
            "w_action": 0.0,
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/small_stage3.yaml")
    print(f"Saved: {WORK}/configs/small_stage3.yaml")


def make_stage4_config(num_actions: int = 4):
    """
    Stage 4: 統合ファインチューン
      - データ: CalMS21 (擬似ラベル付き, 4クラス)
      - 入力: シーケンス (B, T, 3, H, W), T=16
      - 学習対象: 全モジュール (unfreeze all)
      - 損失: cls + bbox + action + id + metric

    num_actions: CalMS21 は 4 クラス (attack / investigation / mount / other)
    """
    cfg = get_variant_config(VARIANT, stage=4, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage4/train",
            "val_root":   f"{WORK}/data/stage4/val",
            "batch_size": 8,           # small: large(4) の倍
            "num_workers": 8,
            "image_size": [640, 640],
            "window_size": 16,
            "window_stride": 8,
            "augment_train": True,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "train": {
            "stage": 4,
            "max_epochs": 20,
            "early_stopping_patience": 10,
            "use_amp": True,
            "output_dir": f"{WORK}/outputs/small/stage4",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-5,              # 全モジュール fine-tune: 小さい lr
            "backbone_lr_factor": 0.1,
            "weight_decay": 1e-4,
            "grad_clip_norm": 0.1,
        },
        "scheduler": {
            "scheduler": "cosine",
            "warmup_epochs": 2,
            "total_epochs": 20,
            "eta_min": 1e-7,
        },
        "model": {
            "training_stage": 4,
            "detector": {
                "head": {"num_classes": 1, "num_queries": 100},
            },
            "action_head": {"num_actions": num_actions},   # CalMS21: 4クラス
            "id_head": {
                "max_ids": 50,
                "use_metric_loss": True,
                "use_action_summary": True,
                "memory_ttl": 30,
            },
        },
        "loss": {
            "w_class": 1.0,
            "w_bbox_l1": 2.0,
            "w_bbox_giou": 1.0,
            "w_action": 1.0,
            "w_id_cls": 1.0,
            "w_id_metric": 0.5,
            "w_temporal_smooth": 0.1,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/small_stage4.yaml")
    print(f"Saved: {WORK}/configs/small_stage4.yaml")


if __name__ == "__main__":
    import pathlib
    pathlib.Path(f"{WORK}/configs").mkdir(parents=True, exist_ok=True)
    make_stage1_config()
    make_stage2_config()
    make_stage3_config()
    make_stage4_config()
    print("\nAll yoake-small configs generated.")
    print(f"  Stage 1: {WORK}/configs/small_stage1.yaml")
    print(f"  Stage 2: {WORK}/configs/small_stage2.yaml")
    print(f"  Stage 3: {WORK}/configs/small_stage3.yaml")
    print(f"  Stage 4: {WORK}/configs/small_stage4.yaml")
