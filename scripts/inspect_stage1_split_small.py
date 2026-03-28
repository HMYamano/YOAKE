from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.ops import nms

from htrtdetr.config.config import HTRTDETRConfig
from htrtdetr.data import SingleFrameDataset, get_collate_fn, load_annotations
from htrtdetr.evaluation.evaluator import DetectionEvaluator
from htrtdetr.models import build_model
from htrtdetr.utils.misc import cxcywh_to_xyxy, load_checkpoint


COLOR_GT = (220, 50, 50)
COLOR_PRED = (50, 200, 80)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_kv(argv):
    out = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            out[key] = value
    return out


def str_to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def denormalize_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image


def select_vis_predictions(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    image_size: tuple[int, int],
    score_thresh: float,
    pre_topk: int,
    post_topk: int,
    nms_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes_xyxy.numel() == 0 or scores.numel() == 0:
        return boxes_xyxy.new_zeros((0, 4)), scores.new_zeros((0,))

    order = torch.argsort(scores, descending=True)
    if pre_topk > 0:
        order = order[: min(pre_topk, order.numel())]

    boxes = boxes_xyxy[order]
    scores = scores[order]

    keep = scores > score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))

    h, w = image_size
    boxes_px = boxes.detach().cpu().clone()
    boxes_px[:, [0, 2]] *= w
    boxes_px[:, [1, 3]] *= h
    scores_cpu = scores.detach().cpu()

    keep_nms = nms(boxes_px, scores_cpu, nms_iou)
    if post_topk > 0:
        keep_nms = keep_nms[: min(post_topk, keep_nms.numel())]

    return boxes[keep_nms], scores[keep_nms]


def draw_boxes(
    image_tensor: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    pred_scores: torch.Tensor,
    image_size: tuple[int, int],
    title: str,
) -> Image.Image:
    image = denormalize_to_uint8(image_tensor)
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    h, w = image_size

    for box in gt_boxes_xyxy:
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        draw.rectangle(
            [x1 * w, y1 * h, x2 * w, y2 * h],
            outline=COLOR_GT,
            width=2,
        )

    for idx, box in enumerate(pred_boxes_xyxy):
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        x1 *= w
        y1 *= h
        x2 *= w
        y2 *= h
        score = float(pred_scores[idx].detach().cpu())

        draw.rectangle([x1, y1, x2, y2], outline=COLOR_PRED, width=2)
        label = f"{score:.2f}"
        tx = x1
        ty = max(0, y1 - 14)
        draw.rectangle([tx, ty, tx + 40, ty + 13], fill=COLOR_PRED)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    draw.rectangle([4, 4, 360, 24], fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255), font=font)
    return pil_img


@torch.no_grad()
def main():
    kv = parse_kv(sys.argv[1:])

    config_path = kv.get("config", "")
    anno_path = kv.get("anno", "")
    checkpoint_path = kv.get("checkpoint", "")
    output_dir = Path(kv.get("output_dir", "runs/inspect_stage1_split_small")).resolve()

    eval_score_thresh = float(kv.get("eval_score_thresh", "0.05"))
    vis_score_thresh = float(kv.get("vis_score_thresh", "0.10"))
    pre_topk = int(kv.get("pre_topk", "50"))
    vis_topk = int(kv.get("vis_topk", "5"))
    nms_iou = float(kv.get("nms_iou", "0.50"))
    max_images = int(kv.get("max_images", "50"))
    num_workers = int(kv.get("num_workers", "0"))
    save_images = str_to_bool(kv.get("save_images", "true"), default=True)
    split_name = kv.get("split_name", "")

    if not config_path:
        raise SystemExit("config=... is required")
    if not anno_path:
        raise SystemExit("anno=... is required")
    if not checkpoint_path:
        raise SystemExit("checkpoint=... is required")

    cfg = HTRTDETRConfig.from_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "images"
    if save_images:
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    videos, class_names, _ = load_annotations(anno_path)
    first_path = videos[0].frames[0].image_path if videos else ""
    data_root = "" if Path(first_path).is_absolute() else str(Path(anno_path).parent)

    image_size = cfg.data.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    if not split_name:
        split_name = Path(anno_path).parent.name or "split"

    dataset = SingleFrameDataset(
        videos,
        image_size=image_size,
        augment=False,
        data_root=data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=get_collate_fn("single"),
    )

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)
    load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
    model.eval()

    det_eval = DetectionEvaluator(iou_thresholds=[0.5, 0.75])
    all_scores = []
    records = []
    saved = 0

    print(f"Config           : {config_path}")
    print(f"Annotation       : {anno_path}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Output dir       : {output_dir}")
    print(f"Split            : {split_name}")
    print(f"Dataset size     : {len(dataset)}")
    print(f"Eval score thr   : {eval_score_thresh}")
    print(f"Vis score thr    : {vis_score_thresh}")
    print(f"Vis pre_topk     : {pre_topk}")
    print(f"Vis topk         : {vis_topk}")
    print(f"Vis nms_iou      : {nms_iou}")
    print("Running inference...")

    for idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = batch["targets"]
        meta = batch.get("meta", [{}])

        output = model.forward_single_frame(images)
        probs = output.pred_logits.softmax(dim=-1)[:, :, :-1]
        scores, labels = probs.max(dim=-1)
        pred_xyxy = cxcywh_to_xyxy(output.pred_boxes)

        for b in range(images.shape[0]):
            scores_b = scores[b]
            labels_b = labels[b]
            boxes_b = pred_xyxy[b]
            gt_boxes_b = targets[b]["boxes"]
            gt_classes_b = targets[b].get("class_ids", torch.zeros(0, dtype=torch.long))

            eval_mask = scores_b > eval_score_thresh
            det_eval.update(
                boxes_b[eval_mask].detach().cpu(),
                scores_b[eval_mask].detach().cpu(),
                labels_b[eval_mask].detach().cpu(),
                gt_boxes_b.detach().cpu(),
                gt_classes_b.detach().cpu(),
                frame_id=idx,
                box_format="xyxy",
            )

            vis_boxes_b, vis_scores_b = select_vis_predictions(
                boxes_xyxy=boxes_b.detach().cpu(),
                scores=scores_b.detach().cpu(),
                image_size=image_size,
                score_thresh=vis_score_thresh,
                pre_topk=pre_topk,
                post_topk=vis_topk,
                nms_iou=nms_iou,
            )

            frame_meta = meta[b] if b < len(meta) else {}
            image_path = str(frame_meta.get("image_path", ""))

            score_np = scores_b.detach().cpu().numpy()
            all_scores.append(score_np)
            records.append(
                {
                    "index": idx,
                    "image_path": image_path,
                    "n_gt": int(len(gt_boxes_b)),
                    "n_pred_eval": int(eval_mask.sum().item()),
                    "n_pred_vis": int(len(vis_boxes_b)),
                    "max_score": float(score_np.max()) if score_np.size else 0.0,
                    "mean_score": float(score_np.mean()) if score_np.size else 0.0,
                }
            )

            if save_images and saved < max_images:
                title = (
                    f"{split_name} | GT=red Pred=green | "
                    f"eval>{eval_score_thresh:.2f} vis>{vis_score_thresh:.2f}"
                )
                vis_img = draw_boxes(
                    image_tensor=images[b].detach().cpu(),
                    gt_boxes_xyxy=gt_boxes_b.detach().cpu(),
                    pred_boxes_xyxy=vis_boxes_b,
                    pred_scores=vis_scores_b,
                    image_size=image_size,
                    title=title,
                )
                vis_img.save(vis_dir / f"{saved:04d}.jpg", quality=90)
                saved += 1

    score_array = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    metrics = det_eval.compute()

    score_report = {
        "min": float(score_array.min()) if score_array.size else 0.0,
        "max": float(score_array.max()) if score_array.size else 0.0,
        "mean": float(score_array.mean()) if score_array.size else 0.0,
        "median": float(np.median(score_array)) if score_array.size else 0.0,
        "p90": float(np.percentile(score_array, 90)) if score_array.size else 0.0,
        "p99": float(np.percentile(score_array, 99)) if score_array.size else 0.0,
        "above_threshold": {
            f">{thr:.2f}": int((score_array > thr).sum()) if score_array.size else 0
            for thr in thresholds
        },
    }

    summary = {
        "config": config_path,
        "annotation": anno_path,
        "checkpoint": checkpoint_path,
        "output_dir": str(output_dir),
        "split_name": split_name,
        "dataset_size": len(dataset),
        "class_names": class_names,
        "eval_score_thresh": eval_score_thresh,
        "vis_score_thresh": vis_score_thresh,
        "pre_topk": pre_topk,
        "vis_topk": vis_topk,
        "nms_iou": nms_iou,
        "images_saved": saved,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "score_distribution": score_report,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "per_image.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"AP50             : {metrics.get('AP50', 0.0):.6f}")
    print(f"AP75             : {metrics.get('AP75', 0.0):.6f}")
    print(f"Recall@50        : {metrics.get('recall@50', 0.0):.6f}")
    print(f"Center error     : {metrics.get('center_error', 0.0):.6f}")
    print(f"Summary          : {output_dir / 'summary.json'}")
    print(f"Per-image        : {output_dir / 'per_image.json'}")
    if save_images:
        print(f"Images           : {vis_dir}")


if __name__ == "__main__":
    main()
