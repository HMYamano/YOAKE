"""
val_runner.py — yoake val の直接実行エンジン

eval_stage*.py の動的ロードに依存せず、共通ロジックを直接呼ぶ。
CLI の run_val() から呼ばれる。

使い方:
    from htrtdetr.runners.val_runner import run_val_stage
    run_val_stage(stage=1, argv=["checkpoint=runs/train/stage1/weights/best.pth"])
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.amp import autocast
from tqdm import tqdm


def run_val_stage(stage: int, argv: list) -> Dict[str, Any]:
    """Stage N の評価を実行して結果 dict を返す。

    Args:
        stage: 1 | 2 | 3 | 4
        argv:  [key=value ...] 形式の引数リスト

    Side effects:
        runs/val/stage{N}/ に評価結果を保存する。
    """
    from ..cli import parse_argv, find_checkpoint, resolve_annotation_path
    from ..config.config import (
        HTRTDETRConfig,
        get_stage1_config, get_stage2_config,
        get_stage3_config, get_stage4_config,
    )
    from ..models import build_model
    from ..training.losses import CombinedLoss
    from ..training.metrics import select_best_metric
    from ..training.plots import (
        save_confusion_matrix, save_per_class_metrics,
        save_metrics_latest,
    )
    from ..evaluation.evaluator import (
        DetectionEvaluator, ActionEvaluator, SimpleTrackingEvaluator,
    )
    from ..training.matching import assign_gt_to_detections
    from ..utils.misc import (
        load_checkpoint, AverageMeter, move_batch_to_device,
        cxcywh_to_xyxy,
    )
    from ..utils.logging import get_logger

    _cfg_builders = {
        1: get_stage1_config,
        2: get_stage2_config,
        3: get_stage3_config,
        4: get_stage4_config,
    }
    if stage not in _cfg_builders:
        raise ValueError(f"stage must be 1–4, got {stage}")

    config_path, overrides = parse_argv(argv)
    root = str(overrides.pop("root", str(Path.cwd())))

    # ----- Config -----
    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = _cfg_builders[stage](overrides if overrides else None)

    output_dir = str(overrides.get("output_dir", f"runs/val/stage{stage}"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- Logger -----
    logger = get_logger(
        f"val_stage{stage}",
        log_file=str(Path(output_dir) / f"val_stage{stage}.log"),
    )
    logger.info(f"=== yoake val stage={stage} ===")

    # ----- Checkpoint -----
    checkpoint = str(overrides.get(
        "checkpoint",
        find_checkpoint(root, stage)
    ))
    if not checkpoint or not Path(checkpoint).exists():
        # find_checkpoint が空文字を返した場合の追加探索
        cands = sorted(
            Path(root).glob(f"runs/train/stage{stage}/weights/best.pth"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        checkpoint = str(cands[0]) if cands else ""

    if not checkpoint or not Path(checkpoint).exists():
        searched = [
            f"runs/train/stage{stage}/weights/best.pth",
            f"runs/train/*/stage{stage}/weights/best.pth",
            f"runs/train/stage{stage}/stage{stage}_best.pth",
            f"outputs/stage{stage}/stage{stage}_best.pth",
        ]
        raise FileNotFoundError(
            f"Stage {stage} のチェックポイントが見つかりません。\n"
            f"  探索した root: {root}\n"
            f"  探索パターン: {', '.join(searched)}\n"
            f"  解決策: checkpoint= で明示してください。\n"
            f"  例: yoake val stage={stage} "
            f"checkpoint=runs/train/stage{stage}/weights/best.pth"
        )
    logger.info(f"checkpoint: {checkpoint}")

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Model -----
    model = build_model(cfg.model)
    model.set_stage(stage)
    ckpt = load_checkpoint(
        checkpoint, model, strict=False, map_location=str(device),
    )
    # EMA 重みがあれば優先使用
    if "extra" in ckpt and "ema_state" in ckpt["extra"]:
        from ..utils.misc import ModelEMA
        ema = ModelEMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["extra"]["ema_state"])
        eval_model = ema.ema
        logger.info("EMA weights loaded.")
    else:
        eval_model = model
    eval_model = eval_model.to(device).eval()

    # ----- DataLoader -----
    from ..cli import _build_dataloaders
    val_root = cfg.data.val_root if cfg.data.val_root else ""
    use_dummy = not resolve_annotation_path(str(val_root)).exists() if val_root else True
    _, val_loader = _build_dataloaders(cfg, stage, use_dummy)
    if use_dummy:
        logger.warning("DummyDataset を使用中。メトリクスは無意味です。")

    # ----- Loss -----
    criterion = CombinedLoss(
        cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
    )
    criterion.set_stage(stage)
    criterion.eval()

    num_actions = cfg.model.action_head.num_actions

    # ----- Evaluators -----
    det_eval = DetectionEvaluator(iou_thresholds=[0.5]) if stage in (1, 4) else None
    act_eval = ActionEvaluator(num_actions=num_actions) if stage in (2, 4) else None
    trk_eval = SimpleTrackingEvaluator() if stage in (3, 4) else None

    total_loss = AverageMeter("val_loss")
    t0 = time.time()

    val_bar = tqdm(val_loader, desc=f"val stage={stage}", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for batch in val_bar:
            batch = move_batch_to_device(batch, device)
            images = batch["images"]
            targets = batch["targets"]
            bs = images.shape[0]

            with autocast("cuda", enabled=torch.cuda.is_available()):
                if stage == 1:
                    output = eval_model.forward_single_frame(images)
                    last_targets = targets
                else:
                    # memory_list=None: バッチ評価ではウィンドウをまたいだ状態保持は行わず、
                    # バッチごとに新鮮なメモリを自動生成する (意図的)。
                    # 動画単位の逐次評価が必要な場合は Inferencer を使うこと。
                    output = eval_model(images)
                    last_targets = (
                        [t[-1] for t in targets]
                        if isinstance(targets[0], list) else targets
                    )
                loss_dict = criterion(output, last_targets, stage=stage)

            loss_val = loss_dict.get("total_loss", 0.0)
            total_loss.update(
                loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val,
                bs,
            )

            # Detection
            if det_eval is not None and output.pred_boxes is not None:
                probs = output.pred_logits.softmax(dim=-1)[:, :, :-1]
                scores, labels = probs.max(dim=-1)
                pred_xyxy = cxcywh_to_xyxy(output.pred_boxes)
                for b in range(bs):
                    mask = scores[b] > 0.05
                    det_eval.update(
                        pred_xyxy[b][mask].cpu(), scores[b][mask].cpu(),
                        labels[b][mask].cpu(),
                        last_targets[b]["boxes"].cpu(),
                        last_targets[b].get("class_ids", torch.zeros(0, dtype=torch.long)).cpu(),
                        box_format="xyxy",
                    )

            # Action / Tracking
            if act_eval is not None or trk_eval is not None:
                det_results = output.det_results or [
                    {"boxes": images.new_zeros((0, 4))} for _ in range(bs)
                ]
                action_batches = (
                    output.split_flattened_tensor(output.action_logits)
                    if output.det_results is not None
                    else [None] * bs
                )
                id_batches = (
                    output.split_flattened_tensor(output.id_logits)
                    if output.det_results is not None
                    else [None] * bs
                )

                for b in range(bs):
                    det_b = det_results[b] if b < len(det_results) else {"boxes": images.new_zeros((0, 4))}
                    pred_boxes_b = det_b.get("boxes", images.new_zeros((0, 4)))
                    asgn = assign_gt_to_detections(
                        pred_boxes_b, last_targets[b], iou_threshold=0.5
                    )
                    matched = asgn["matched_gt_idx"] >= 0
                    n_det = int(pred_boxes_b.shape[0])

                    if act_eval is not None and b < len(action_batches):
                        al = action_batches[b]
                        if al is not None:
                            pred_acts = al.argmax(dim=-1)
                            gt_acts = asgn.get(
                                "matched_action_ids",
                                torch.full((n_det,), -1, dtype=torch.long, device=al.device),
                            )
                            valid = matched & (gt_acts >= 0)
                            if valid.any():
                                act_eval.update(
                                    pred_acts[valid].cpu().tolist(),
                                    gt_acts[valid].cpu().tolist(),
                                )

                    if trk_eval is not None and b < len(id_batches):
                        il = id_batches[b]
                        if il is not None:
                            pred_ids = il.argmax(dim=-1)
                            gt_ids = asgn.get(
                                "matched_track_ids",
                                torch.full((n_det,), -1, dtype=torch.long, device=il.device),
                            )
                            valid = matched & (gt_ids >= 0)
                            if valid.any():
                                sequence_key = None
                                if "meta" in batch and b < len(batch["meta"]):
                                    sequence_key = batch["meta"][b].get("video_id")
                                trk_eval.update(
                                    pred_ids[valid].cpu().tolist(),
                                    gt_ids[valid].cpu().tolist(),
                                    sequence_key=sequence_key,
                                )

            val_bar.set_postfix({"loss": f"{total_loss.avg:.4f}"})

    val_bar.close()
    elapsed = time.time() - t0

    # ----- Aggregate results -----
    result: Dict[str, Any] = {
        "val_loss": total_loss.avg,
        "val_loss_kind": "unweighted",
        "elapsed_sec": elapsed,
    }

    if det_eval is not None:
        dm = det_eval.compute()
        result.update({k.lower(): v for k, v in dm.items()})
        result["val_AP50"] = dm.get("AP50", 0.0)

    if act_eval is not None:
        am = act_eval.compute()
        result["macro_f1"] = am.get("macro_f1", 0.0)
        result["accuracy"] = am.get("accuracy", 0.0)
        result["per_class_f1"] = am.get("per_class_f1", {})
        result["confusion_matrix"] = am.get("confusion_matrix", [])
        result["weighted_f1"] = sum(
            v * sum(row) for v, row in zip(am.get("per_class_f1", {}).values(),
                                           am.get("confusion_matrix", []))
        ) / max(sum(sum(row) for row in am.get("confusion_matrix", [])), 1)

    if trk_eval is not None:
        tm = trk_eval.compute()
        result["idf1"] = tm.get("idf1", 0.0)
        result["idp"] = tm.get("idp", 0.0)
        result["idr"] = tm.get("idr", 0.0)
        result["IDSW"] = tm.get("IDSW", 0)
        result["IDTP"] = tm.get("IDTP", 0)
        result["IDFP"] = tm.get("IDFP", 0)
        result["IDFN"] = tm.get("IDFN", 0)

    # primary metric
    metric_val, metric_name, higher = select_best_metric(stage, result)
    result["primary_metric"] = metric_name
    result["primary_metric_value"] = metric_val

    logger.info(f"val_loss={result['val_loss']:.4f}  {metric_name}={metric_val:.4f}  "
                f"elapsed={elapsed:.1f}s")

    # ----- Save results -----
    # main JSON
    json_safe = {k: v for k, v in result.items()
                 if isinstance(v, (int, float, str, bool, list, dict))}
    (Path(output_dir) / "val_results.json").write_text(
        json.dumps(json_safe, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Stage 2 extras: confusion matrix + per-class metrics
    if act_eval is not None and result.get("confusion_matrix"):
        action_names = cfg.model.action_head.action_names if hasattr(
            cfg.model.action_head, "action_names") else \
            [str(i) for i in range(num_actions)]
        save_confusion_matrix(
            result["confusion_matrix"], action_names,
            str(Path(output_dir) / "confusion_matrix.png"),
            title=f"Confusion Matrix — Stage {stage} val",
        )
        save_per_class_metrics(
            {"macro_f1": result["macro_f1"], "accuracy": result.get("accuracy", 0.0),
             "per_class_f1": result["per_class_f1"],
             "confusion_matrix": result["confusion_matrix"]},
            output_dir,
        )

    save_metrics_latest(
        val_metrics={k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))},
        output_dir=output_dir,
        epoch=ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
        primary_metric_name=metric_name,
        primary_metric_value=metric_val,
        best_so_far=metric_val,
        higher_is_better=higher,
    )

    print(f"\n[val stage={stage}] {metric_name}={metric_val:.4f}  "
          f"val_loss={result['val_loss']:.4f}")
    print(f"Results saved to: {output_dir}/val_results.json")

    return result
