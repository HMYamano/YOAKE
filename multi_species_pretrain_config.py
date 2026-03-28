# multi_species_pretrain_config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from htrtdetr.config.config import get_variant_config

# デフォルトはリポジトリの隣のディレクトリ。環境に合わせて変更可。
WORK = str(Path(__file__).resolve().parent.parent / "YOAKE_pre-train")
N_SPECIES = 3  # 例: ショウジョウバエ / マウス / 魚 の3種混在

cfg = get_variant_config("large", stage=4, overrides={
    "model": {
        "training_stage": 4,
        "detector": {
            "head": {
                "num_classes": N_SPECIES,   # 種ごとにクラスを分ける
                "num_queries": 300,
            }
        },
        "id_head": {
            "max_ids": 100,
            "use_species_separated_pools": True,
            "num_species": N_SPECIES,       # num_classes と一致させること
            "use_metric_loss": True,
        },
        "action_head": {
            "num_actions": 5,
        },
    },
    "data": {
        "batch_size": 4,
        "window_size": 16,
        "num_workers": 8,
    },
    "train": {
        "stage": 4,
        "use_amp": True,
        "max_epochs": 20,
        "output_dir": f"{WORK}/outputs/large/stage4_multispecies",
    },
})

cfg.save_yaml(f"{WORK}/configs/large_stage4_multispecies.yaml")
print("Saved multi-species config.")
