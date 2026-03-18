"""
eval_stage3.py — Stage 3 Evaluation: ID / Tracking Metrics

embedding の分離度 (within-class vs between-class cosine similarity)、
ID classification accuracy を計算する。

使い方:
  python scripts/eval_stage3.py \
      anno=data/sample/annotations_val.json \
      checkpoint=outputs/stage3/stage3_best.pth \
      output_dir=outputs/eval_stage3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from htrtdetr.config.config import get_stage3_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.utils.misc import load_checkpoint


_YOAKE_TRYAL = "C:/Users/hayam/Desktop/YOAKE_tryal"


def parse_overrides(argv) -> dict:
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v
    return overrides


@torch.no_grad()
def main():
    overrides = parse_overrides(sys.argv[1:])

    anno_path = overrides.get("anno", "data/sample/annotations_val.json")
    root = overrides.pop("root", _YOAKE_TRYAL)
    checkpoint_path = overrides.get("checkpoint", f"{root}/outputs/stage3/stage3_best.pth")
    output_dir = Path(overrides.get("output_dir", f"{root}/outputs/eval_stage3"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage 3 (ID) on: {anno_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = get_stage3_config()
    anno = load_annotation(anno_path)

    loaders = build_dataloaders(train_anno=anno, val_anno=None, stage=3, cfg=cfg.data)
    loader = loaders["train"]

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(3)

    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint_path}")

    model.eval()

    all_embeddings = []
    all_local_ids = []
    correct = 0
    total = 0

    for batch in loader:
        geo = batch["geo_features"].to(device)          # (B, T, 10)
        gt_ids = batch["local_track_ids"]                # (B,) cpu

        out = model.forward_geo_sequence(geo)
        emb = out["id_embeddings"].cpu()                # (B, emb_dim)
        logits = out["id_logits"].cpu()                 # (B, max_ids+1)

        valid = gt_ids >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            correct += (preds == gt_ids[valid]).sum().item()
            total += valid.sum().item()

            all_embeddings.append(emb[valid])
            all_local_ids.append(gt_ids[valid])

    # ID classification accuracy
    id_acc = correct / max(total, 1)

    # Embedding quality: within-class vs between-class similarity
    if all_embeddings:
        embs = torch.cat(all_embeddings, dim=0)          # (N, D)
        ids = torch.cat(all_local_ids, dim=0)             # (N,)

        embs_norm = F.normalize(embs, p=2, dim=-1)
        sim_mat = embs_norm @ embs_norm.t()               # (N, N)

        same_id = (ids.unsqueeze(0) == ids.unsqueeze(1))  # (N, N) bool
        same_id.fill_diagonal_(False)
        diff_id = ~same_id
        diff_id.fill_diagonal_(False)

        within_sim = sim_mat[same_id].mean().item() if same_id.any() else 0.0
        between_sim = sim_mat[diff_id].mean().item() if diff_id.any() else 0.0
    else:
        within_sim = 0.0
        between_sim = 0.0

    results = {
        "id_accuracy": id_acc,
        "within_class_similarity": within_sim,
        "between_class_similarity": between_sim,
        "similarity_gap": within_sim - between_sim,
        "total_samples": total,
    }

    print("\n=== Stage 3 ID Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = output_dir / "results_stage3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
