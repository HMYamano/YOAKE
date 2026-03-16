"""
eval_stage1.py — Stage 1 Evaluation: Detection Metrics

AP50, AP75, recall@50, mean center error を計算する。

使い方:
  python scripts/eval_stage1.py \
      anno=data/sample/annotations_val.json \
      checkpoint=outputs/stage1/checkpoint_best.pth \
      output_dir=outputs/eval_stage1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import get_stage1_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.evaluation.evaluator import DetectionEvaluator
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy


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
    checkpoint_path = overrides.get("checkpoint", "outputs/stage1/checkpoint_best.pth")
    output_dir = Path(overrides.get("output_dir", "outputs/eval_stage1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage 1 (Detection) on: {anno_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = get_stage1_config()
    anno = load_annotation(anno_path)

    loaders = build_dataloaders(train_anno=anno, val_anno=None, stage=1, cfg=cfg.data)
    # val がないので train を使って評価 (eval 専用 anno を想定)
    loader = loaders["train"]

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)

    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint_path}")

    model.eval()
    evaluator = DetectionEvaluator(
        num_classes=cfg.model.detector.head.num_classes,
        iou_thresholds=[0.5, 0.75],
    )

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"]

        out = model.forward_single_frame(images)

        # (B, Q, C+1) → scores, (B, Q, 4) cxcywh → xyxy
        probs = out.pred_logits.softmax(dim=-1)[:, :, :-1]  # remove background
        scores, labels = probs.max(dim=-1)  # (B, Q)
        pred_xyxy = cxcywh_to_xyxy(out.pred_boxes)  # (B, Q, 4)

        B = images.shape[0]
        for b in range(B):
            # Filter by score
            mask = scores[b] > 0.3
            pred_b = {
                "boxes": pred_xyxy[b][mask].cpu(),
                "scores": scores[b][mask].cpu(),
                "labels": labels[b][mask].cpu(),
            }
            gt_b = {
                "boxes": targets[b]["boxes"].cpu(),
                "labels": targets[b]["class_ids"].cpu(),
            }
            evaluator.update(pred_b, gt_b)

    results = evaluator.compute()

    print("\n=== Stage 1 Detection Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    out_path = output_dir / "results_stage1.json"
    with open(out_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
