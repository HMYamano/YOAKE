"""
eval_unified.py — Unified Evaluation: Detection + Tracking + Action  [DEPRECATED]

.. deprecated::
   このスクリプトは非推奨です。代わりに yoake CLI を使用してください:

     yoake val stage=4 data.val_root=data/test

   出力は runs/val/stage4/ に保存されます。

AP50, IDF1, Action F1 をまとめて計算する。
Stage 4 の統合モデル評価用。

使い方:
  python scripts/eval_unified.py \
      anno=data/sample/annotations_val.json \
      checkpoint=outputs/stage4/checkpoint_best.pth \
      output_dir=outputs/eval_unified \
      score_threshold=0.3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from htrtdetr.config.config import get_stage4_config
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.evaluation.evaluator import DetectionEvaluator, ActionEvaluator, TrackingEvaluator
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy


_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)


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
    root = overrides.pop("root", _DEFAULT_ROOT)
    checkpoint_path = overrides.get("checkpoint", f"{root}/runs/train/stage4/stage4_best.pth")
    output_dir = Path(overrides.get("output_dir", f"{root}/runs/val/stage4"))
    score_threshold = float(overrides.get("score_threshold", "0.3"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unified evaluation on: {anno_path}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = get_stage4_config()
    anno = load_annotation(anno_path)

    loaders = build_dataloaders(train_anno=anno, val_anno=None, stage=4, cfg=cfg.data)
    loader = loaders["train"]

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(4)

    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint_path}")

    model.eval()

    det_evaluator = DetectionEvaluator(
        num_classes=cfg.model.detector.head.num_classes,
        iou_thresholds=[0.5, 0.75],
    )
    act_evaluator = ActionEvaluator(cfg.model.action_head.num_actions)
    trk_evaluator = TrackingEvaluator()

    memory_list = None

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)     # (B, T, 3, H, W)
        targets = batch["targets"]               # List[Dict]

        B = images.shape[0]

        # Create fresh memory for each batch (or persist within video for tracking)
        memory_list = model.create_memory_list(B, device)

        out = model(images, memory_list)

        # Detection evaluation (last frame)
        probs = out.pred_logits.softmax(dim=-1)[:, :, :-1]
        scores, labels = probs.max(dim=-1)
        pred_xyxy = cxcywh_to_xyxy(out.pred_boxes)

        for b in range(B):
            mask = scores[b] > score_threshold
            pred_b = {
                "boxes": pred_xyxy[b][mask].cpu(),
                "scores": scores[b][mask].cpu(),
                "labels": labels[b][mask].cpu(),
            }
            gt_b = {
                "boxes": targets[b]["boxes"].cpu(),
                "labels": targets[b]["class_ids"].cpu(),
            }
            det_evaluator.update(pred_b, gt_b)

        # Action evaluation (from det_results)
        if out.action_logits is not None and out.det_results is not None:
            ptr = 0
            for b in range(B):
                N_b = out.det_results[b]["features"].shape[0]
                if N_b == 0:
                    continue
                act_logits_b = out.action_logits[ptr:ptr + N_b]
                preds_b = act_logits_b.argmax(dim=-1).cpu()
                scores_b = act_logits_b.softmax(dim=-1).cpu()

                gt_actions = targets[b].get("action_ids", None)
                if gt_actions is not None and len(gt_actions) > 0:
                    # Truncate to min length (rough alignment)
                    min_n = min(N_b, len(gt_actions))
                    act_evaluator.update(
                        preds_b[:min_n].tolist(),
                        gt_actions[:min_n].cpu().tolist(),
                    )
                ptr += N_b

        # Tracking evaluation (from det_results)
        for b in range(B):
            det = out.det_results[b]
            pred_boxes = det.get("boxes", torch.zeros(0, 4)).cpu()
            pred_track_ids = det.get("track_ids", torch.zeros(0, dtype=torch.long)).cpu()
            gt_boxes = targets[b].get("boxes", torch.zeros(0, 4)).cpu()
            gt_track_ids = targets[b].get("track_ids", torch.zeros(0, dtype=torch.long)).cpu()

            trk_evaluator.update(
                pred_boxes=pred_boxes,
                pred_track_ids=pred_track_ids,
                gt_boxes=gt_boxes,
                gt_track_ids=gt_track_ids,
            )

    det_results = det_evaluator.compute()
    act_results = act_evaluator.compute()
    trk_results = trk_evaluator.compute()

    all_results = {
        **{f"det_{k}": v for k, v in det_results.items()},
        **{f"act_{k}": v for k, v in act_results.items() if isinstance(v, (int, float))},
        **{f"trk_{k}": v for k, v in trk_results.items()},
    }

    print("\n=== Unified Evaluation Results ===")
    print("\n--- Detection ---")
    for k, v in det_results.items():
        print(f"  {k}: {v:.4f}")
    print("\n--- Action ---")
    for k, v in act_results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print("\n--- Tracking ---")
    for k, v in trk_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = output_dir / "results_unified.json"
    serializable = {}
    for k, v in all_results.items():
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
