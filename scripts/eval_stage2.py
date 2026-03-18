"""
eval_stage2.py — Stage 2 Evaluation: Action Classification Metrics

accuracy, per-class F1, macro F1, confusion matrix を計算する。

使い方:
  python scripts/eval_stage2.py \
      anno=data/sample/annotations_val.json \
      checkpoint=outputs/stage2/checkpoint_best.pth \
      output_dir=outputs/eval_stage2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import get_stage2_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.evaluation.evaluator import ActionEvaluator
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
    checkpoint_path = overrides.get("checkpoint", f"{root}/outputs/stage2/checkpoint_best.pth")
    output_dir = Path(overrides.get("output_dir", f"{root}/outputs/eval_stage2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage 2 (Action) on: {anno_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = get_stage2_config()
    anno = load_annotation(anno_path)

    loaders = build_dataloaders(train_anno=anno, val_anno=None, stage=2, cfg=cfg.data)
    loader = loaders["train"]

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(2)

    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint_path}")

    model.eval()
    num_actions = cfg.model.action_head.num_actions
    evaluator = ActionEvaluator(num_classes=num_actions)

    for batch in loader:
        geo = batch["geo_features"].to(device)         # (B, T, 10)
        gt_actions = batch["action_ids"]                 # (B,) cpu

        out = model.forward_geo_sequence(geo)
        preds = out["action_logits"].argmax(dim=-1).cpu()  # (B,)
        scores = out["action_logits"].softmax(dim=-1).cpu()

        for i in range(geo.shape[0]):
            if gt_actions[i] < 0:
                continue
            evaluator.update(
                pred_labels=preds[i:i+1],
                gt_labels=gt_actions[i:i+1],
                pred_scores=scores[i:i+1],
            )

    results = evaluator.compute()

    print("\n=== Stage 2 Action Results ===")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = output_dir / "results_stage2.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            serializable[k] = float(v)
        elif hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = str(v)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
