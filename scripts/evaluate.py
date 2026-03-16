"""
evaluate.py — 統合評価スクリプト

使い方:
    python scripts/evaluate.py [config.yaml] [key=value ...]
    python scripts/evaluate.py eval.checkpoint=outputs/stage4/stage4_best.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from htrtdetr.config import HTRTDETRConfig
from htrtdetr.data import (
    DummyDataset, SlidingWindowDataset, SingleFrameDataset,
    load_annotations, get_collate_fn
)
from htrtdetr.models import build_model, HTRTDETR
from htrtdetr.evaluation import HTRTDETREvaluator
from htrtdetr.utils import (
    set_seed, load_checkpoint, get_logger, get_device,
    cxcywh_to_xyxy, move_batch_to_device, GPUMemoryTracker
)


def parse_argv(argv):
    config_path, overrides = "", {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try: v = int(v)
            except:
                try: v = float(v)
                except: pass
            keys = k.split(".")
            d = overrides
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
        elif not config_path and arg.endswith((".yaml", ".yml")):
            config_path = arg
    return config_path, overrides


def main() -> None:
    config_path, overrides = parse_argv(sys.argv)

    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = HTRTDETRConfig()
        if overrides:
            cfg = cfg.merge(overrides)

    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("evaluate", log_file=str(output_dir / "eval.log"))
    logger.info(f"Evaluating checkpoint: {cfg.eval.checkpoint}")

    device = get_device()
    set_seed(42)

    # ----- Dataset -----
    use_dummy = not (
        Path(cfg.data.val_root).exists() and
        any(Path(cfg.data.val_root).iterdir())
    )

    if use_dummy:
        logger.warning("Using DummyDataset for evaluation")
        val_dataset = DummyDataset(
            n_samples=50, window_size=cfg.data.window_size,
            image_size=tuple(cfg.data.image_size),
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            mode="sequence",
        )
        action_names = cfg.eval.action_names
    else:
        val_ann = str(Path(cfg.data.val_root) / "annotations.json")
        val_videos, _, action_names = load_annotations(val_ann)
        val_dataset = SlidingWindowDataset(
            val_videos, window_size=cfg.data.window_size,
            stride=cfg.data.window_size, image_size=tuple(cfg.data.image_size),
            augment=False, data_root=cfg.data.val_root,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=False,
        collate_fn=get_collate_fn("sequence"),
    )

    # ----- Model -----
    model = build_model(cfg.model)
    model = model.to(device)

    if Path(cfg.eval.checkpoint).exists():
        logger.info(f"Loading checkpoint: {cfg.eval.checkpoint}")
        load_checkpoint(cfg.eval.checkpoint, model, strict=False, map_location=str(device))
    else:
        logger.warning(f"Checkpoint not found: {cfg.eval.checkpoint}. Using random weights.")

    model.eval()

    # ----- Evaluator -----
    evaluator = HTRTDETREvaluator(
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        action_names=action_names[:cfg.model.action_head.num_actions],
        iou_thresholds=cfg.eval.iou_thresholds,
    )

    # ----- Eval loop -----
    GPUMemoryTracker.reset_peak()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = move_batch_to_device(batch, device)

            images = batch["images"]         # (B, T, 3, H, W)
            all_targets = batch["targets"]   # List[List[Dict]]

            # 推論時間計測
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with autocast(enabled=torch.cuda.is_available()):
                output = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start

            gpu_mb = GPUMemoryTracker.current_mb()
            evaluator.runtime.update(elapsed / images.shape[0], gpu_mb)

            # 最終フレームの GT と予測を比較
            B, T = images.shape[:2]
            for b in range(B):
                last_target = all_targets[b][-1] if isinstance(all_targets[b], list) \
                    else all_targets[b]

                gt_boxes = last_target["boxes"]          # (M, 4) normalized [cx,cy,w,h]
                gt_classes = last_target["class_ids"]    # (M,)
                gt_tracks = last_target.get("track_ids", torch.full_like(gt_classes, -1))
                gt_actions = last_target.get("action_ids", torch.full_like(gt_classes, -1))

                # Detection 評価
                det_results = output.det_results[b] if output.det_results else {}
                if det_results:
                    pred_boxes = det_results.get("boxes", torch.zeros(0, 4))
                    pred_scores = det_results.get("scores", torch.zeros(0))
                    pred_classes = det_results.get("class_ids", torch.zeros(0, dtype=torch.long))
                else:
                    pred_boxes = torch.zeros(0, 4)
                    pred_scores = torch.zeros(0)
                    pred_classes = torch.zeros(0, dtype=torch.long)

                evaluator.detection.update(
                    pred_boxes, pred_scores, pred_classes,
                    gt_boxes, gt_classes,
                    frame_id=batch_idx * B + b,
                    box_format="cxcywh",
                )

                # Action 評価
                if output.action_probs is not None and det_results:
                    N_det = det_results["features"].shape[0] if "features" in det_results else 0
                    if N_det > 0:
                        action_probs = output.action_probs[:N_det]
                        pred_actions = action_probs.argmax(dim=-1).cpu().tolist()
                        # GT との照合 (簡略: predict とGTの個数を合わせる)
                        gt_actions_list = gt_actions[:N_det].cpu().tolist()
                        evaluator.action.update(pred_actions, gt_actions_list)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Evaluated {batch_idx + 1}/{len(val_loader)} batches")

    # ----- Results -----
    results = evaluator.compute_all()
    logger.info("\n=== Evaluation Results ===")
    for domain, metrics in results.items():
        logger.info(f"\n[{domain.upper()}]")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if not isinstance(v, (list, dict)):
                    logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # JSON 保存
    results_path = str(output_dir / "eval_results.json")
    evaluator.save(results_path)
    logger.info(f"\nResults saved to: {results_path}")

    # Confusion matrix 可視化
    action_eval = results.get("action", {})
    cm = action_eval.get("confusion_matrix")
    if cm:
        from htrtdetr.analysis import plot_confusion_matrix
        import numpy as np
        plot_confusion_matrix(
            np.array(cm),
            action_names[:cfg.model.action_head.num_actions],
            str(output_dir / "confusion_matrix.png"),
            title="Action Confusion Matrix",
        )
        logger.info(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
