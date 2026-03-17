"""
eval_stage1.py — Stage 1 Evaluation: Detection Metrics

AP50, AP75, recall@50, mean center error を計算する。

使い方:
  # デフォルト (YOAKE_tryal の val データを自動検出)
  python scripts/eval_stage1.py

  # 直接指定
  python scripts/eval_stage1.py \
      anno=C:/Users/hayam/Desktop/YOAKE_tryal/data/val/annotations.json \
      checkpoint=outputs/stage1/stage1_best.pth \
      output_dir=outputs/eval_stage1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from htrtdetr.config.config import get_stage1_config
from htrtdetr.models import build_model
from htrtdetr.data import load_annotations, SingleFrameDataset, get_collate_fn
from htrtdetr.evaluation.evaluator import DetectionEvaluator
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy


_YOAKE_TRYAL = "C:/Users/hayam/Desktop/YOAKE_tryal"
DEFAULT_ANNO  = f"{_YOAKE_TRYAL}/data/val/annotations.json"


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

    anno_path       = overrides.get("anno",        DEFAULT_ANNO)
    checkpoint_path = overrides.get("checkpoint",  "outputs/stage1/stage1_best.pth")
    output_dir      = Path(overrides.get("output_dir", "outputs/eval_stage1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage 1 (Detection) on: {anno_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")

    # ----- Config -----
    cfg = get_stage1_config()
    if "num_actions" in overrides:
        cfg.model.action_head.num_actions = int(overrides["num_actions"])

    # ----- Dataset (学習と同じ SingleFrameDataset を使用) -----
    if not Path(anno_path).exists():
        print(f"ERROR: annotation not found: {anno_path}")
        sys.exit(1)

    videos, _, _ = load_annotations(anno_path)

    # image_path が絶対パスかどうかで data_root を決める
    first_path = videos[0].frames[0].image_path if videos else ""
    data_root = "" if Path(first_path).is_absolute() else str(Path(anno_path).parent)

    _img_size = cfg.data.image_size
    if isinstance(_img_size, int):
        _img_size = (_img_size, _img_size)
    else:
        _img_size = tuple(_img_size)

    dataset = SingleFrameDataset(
        videos,
        image_size=_img_size,
        augment=False,
        data_root=data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=get_collate_fn("single"),
    )
    print(f"Eval samples: {len(dataset)}")

    # ----- Model -----
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)

    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found — using random weights: {checkpoint_path}")

    model.eval()
    evaluator = DetectionEvaluator(iou_thresholds=[0.5, 0.75])

    for batch in loader:
        images  = batch["images"].to(device)
        targets = batch["targets"]

        out = model.forward_single_frame(images)

        # (B, Q, C+1) → scores,  (B, Q, 4) cxcywh → xyxy
        probs      = out.pred_logits.softmax(dim=-1)[:, :, :-1]  # background 除去
        scores, labels = probs.max(dim=-1)                        # (B, Q)
        pred_xyxy  = cxcywh_to_xyxy(out.pred_boxes)              # (B, Q, 4)

        B = images.shape[0]
        for b in range(B):
            mask = scores[b] > 0.05
            evaluator.update(
                pred_xyxy[b][mask].cpu(),
                scores[b][mask].cpu(),
                labels[b][mask].cpu(),
                targets[b]["boxes"].cpu(),
                targets[b]["class_ids"].cpu(),
                box_format="xyxy",
            )

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
