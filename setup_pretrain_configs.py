# setup_pretrain_configs.py  ← リポジトリには含まれていない。以下の内容をコピーして作成すること。
# 実行: python C:/Users/utopi/YOAKE/setup_pretrain_configs.py
# 生成先: C:/Users/utopi/YOAKE_pre-train/configs/

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from htrtdetr.config.config import get_variant_config

WORK = "C:/Users/utopi/YOAKE_pre-train"  # ← 自分の環境に合わせて変更


def make_stage1_config():
    cfg = get_variant_config("large", stage=1, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage1/train",
            "val_root": f"{WORK}/data/stage1/val",
            "batch_size": 32,
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
            "output_dir": f"{WORK}/outputs/large/stage1",
            "log_interval": 100,
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
            "total_epochs": 24,
            "eta_min": 1e-6,
        },
        "model": {
            "training_stage": 1,
            "detector": {
                "head": {
                    "num_classes": 84,   # COCO(80) + AP-10K 追加クラス
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
    cfg.save_yaml(f"{WORK}/configs/large_stage1.yaml")
    print(f"Saved: {WORK}/configs/large_stage1.yaml")


def make_stage2_config():
    cfg = get_variant_config("large", stage=2, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage2/train",
            "val_root": f"{WORK}/data/stage2/val",
            "batch_size": 8,
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
            "output_dir": f"{WORK}/outputs/large/stage2",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "backbone_lr_factor": 0.0,   # detector は freeze_module() で制御
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
                    "num_classes": 1,    # Stage 2: 行動学習なので汎用1クラスでよい
                    "num_queries": 100,
                },
            },
            "action_head": {
                "num_actions": 140,      # Animal Kingdom の全行動クラス数 (実際に合わせること)
            },
            "id_head": {"max_ids": 50},
        },
        "loss": {
            "w_class": 0.0,     # Stage 2: detection loss 無効
            "w_bbox_l1": 0.0,
            "w_bbox_giou": 0.0,
            "w_action": 1.0,
            "w_id_cls": 0.0,
        },
    })
    cfg.save_yaml(f"{WORK}/configs/large_stage2.yaml")
    print(f"Saved: {WORK}/configs/large_stage2.yaml")


def make_stage3_config():
    cfg = get_variant_config("large", stage=3, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage3/train",
            "val_root": f"{WORK}/data/stage3/val",
            "batch_size": 8,
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
            "output_dir": f"{WORK}/outputs/large/stage3",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-4,
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
    cfg.save_yaml(f"{WORK}/configs/large_stage3.yaml")
    print(f"Saved: {WORK}/configs/large_stage3.yaml")


def make_stage4_config():
    cfg = get_variant_config("large", stage=4, overrides={
        "data": {
            "train_root": f"{WORK}/data/stage4/train",
            "val_root": f"{WORK}/data/stage4/val",
            "batch_size": 4,
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
            "output_dir": f"{WORK}/outputs/large/stage4",
            "log_interval": 50,
            "val_interval": 1,
            "save_best": True,
            "save_last": True,
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": 1e-5,           # 全モジュール fine-tune: 小さい lr
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
            "action_head": {"num_actions": 4},  # CalMS21 の4クラス
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
    cfg.save_yaml(f"{WORK}/configs/large_stage4.yaml")
    print(f"Saved: {WORK}/configs/large_stage4.yaml")


if __name__ == "__main__":
    import pathlib
    pathlib.Path(f"{WORK}/configs").mkdir(parents=True, exist_ok=True)
    make_stage1_config()
    make_stage2_config()
    make_stage3_config()
    make_stage4_config()
    print("All configs generated.")
