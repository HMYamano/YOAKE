from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from htrtdetr.config.config import HTRTDETRConfig
from htrtdetr.models import build_model
from htrtdetr.data import load_annotations, SingleFrameDataset, get_collate_fn
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy

from torchvision.ops import nms


COLOR_GT = (220, 50, 50)     # red
COLOR_PRED = (50, 200, 80)   # green


def parse_kv(argv):
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


def draw_boxes_on_image(image_tensor, pred_boxes_xyxy, pred_scores, gt_boxes_xyxy, image_size):
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    H, W = image_size

    # GT boxes: normalized xyxy -> pixel xyxy
    for box in gt_boxes_xyxy:
        b = box.cpu().numpy()
        x1 = float(b[0]) * W
        y1 = float(b[1]) * H
        x2 = float(b[2]) * W
        y2 = float(b[3]) * H
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_GT, width=2)

    # Pred boxes: normalized xyxy -> pixel xyxy
    for i, box in enumerate(pred_boxes_xyxy):
        b = box.cpu().numpy()
        x1 = float(b[0]) * W
        y1 = float(b[1]) * H
        x2 = float(b[2]) * W
        y2 = float(b[3]) * H
        score = float(pred_scores[i].cpu())

        draw.rectangle([x1, y1, x2, y2], outline=COLOR_PRED, width=2)
        label = f"{score:.2f}"
        tx = x1
        ty = max(0, y1 - 14)
        draw.rectangle([tx, ty, tx + 38, ty + 13], fill=COLOR_PRED)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    draw.text((5, 5), "GT=red / Pred=green", fill=(255, 255, 255), font=font)
    return pil_img


@torch.no_grad()
def main():
    kv = parse_kv(sys.argv[1:])

    config_path = kv.get("config", "")
    anno_path = kv.get("anno", "")
    checkpoint_path = kv.get("checkpoint", "")
    output_dir = Path(kv.get("output_dir", "runs/visualize_stage1_small")).resolve()
    score_thresh = float(kv.get("score_thresh", "0.30"))
    max_images = int(kv.get("max_images", "50"))

    if not config_path:
        raise SystemExit("config=... is required")
    if not anno_path:
        raise SystemExit("anno=... is required")
    if not checkpoint_path:
        raise SystemExit("checkpoint=... is required")

    cfg = HTRTDETRConfig.from_yaml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    videos, class_names, action_names = load_annotations(anno_path)
    first_path = videos[0].frames[0].image_path if videos else ""
    data_root = "" if Path(first_path).is_absolute() else str(Path(anno_path).parent)

    img_size = cfg.data.image_size
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    else:
        img_size = tuple(img_size)

    dataset = SingleFrameDataset(
        videos,
        image_size=img_size,
        augment=False,
        data_root=data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=get_collate_fn("single"),
    )

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)
    load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
    model.eval()

    H, W = img_size
    saved = 0
    records = []

    print(f"Config      : {config_path}")
    print(f"Annotation  : {anno_path}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Score thr   : {score_thresh}")
    print(f"Max images  : {max_images}")
    print(f"Dataset size: {len(dataset)}")
    print("Running inference...")

    for idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = batch["targets"]
        meta = batch.get("meta", [{}])

        out = model.forward_single_frame(images)
        
        probs = out.pred_logits.softmax(dim=-1)[:, :, :-1]
        scores, labels = probs.max(dim=-1)
        pred_xyxy = cxcywh_to_xyxy(out.pred_boxes)

        scores_b = scores[0]
        boxes_b = pred_xyxy[0]

        # まず score 上位だけ残す
        topk = min(50, scores_b.numel())
        topk_idx = torch.argsort(scores_b, descending=True)[:topk]
        scores_b = scores_b[topk_idx]
        boxes_b = boxes_b[topk_idx]

        # NMS 用に pixel 座標へ変換
        boxes_px = boxes_b.clone()
        boxes_px[:, [0, 2]] *= W
        boxes_px[:, [1, 3]] *= H

        # score threshold
        keep = scores_b > score_thresh
        scores_b = scores_b[keep]
        boxes_b = boxes_b[keep]
        boxes_px = boxes_px[keep]

        # NMS
        if len(scores_b) > 0:
            keep_nms = nms(boxes_px, scores_b, 0.5)
            keep_nms = keep_nms[:5]   # 最終的に上位5個だけ表示
            pred_boxes_b = boxes_b[keep_nms]
            pred_scores_b = scores_b[keep_nms]
        else:
            pred_boxes_b = boxes_b
            pred_scores_b = scores_b

        gt_boxes_b = targets[0]["boxes"]

        frame_meta = meta[0] if meta else {}
        image_path = str(frame_meta.get("image_path", ""))

        if saved < max_images:
            pil_img = draw_boxes_on_image(
                image_tensor=images[0].cpu(),
                pred_boxes_xyxy=pred_boxes_b.cpu(),
                pred_scores=pred_scores_b.cpu(),
                gt_boxes_xyxy=gt_boxes_b.cpu(),
                image_size=(H, W),
            )
            out_name = f"{saved:04d}.jpg"
            pil_img.save(vis_dir / out_name, quality=90)
            saved += 1

        records.append({
            "index": idx,
            "image_path": image_path,
            "n_gt": int(len(gt_boxes_b)),
            "n_pred": int(len(pred_boxes_b)),
            "max_score": float(scores[0].max().item()),
            "mean_score": float(scores[0].mean().item()),
        })

        if saved >= max_images and idx >= max_images - 1:
            continue

    summary = {
        "config": config_path,
        "annotation": anno_path,
        "checkpoint": checkpoint_path,
        "score_thresh": score_thresh,
        "max_images": max_images,
        "images_saved": saved,
        "dataset_size": len(dataset),
        "class_names": class_names,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Images      : {vis_dir}")
    print(f"Summary     : {output_dir / 'summary.json'}")
    print(f"Per-image   : {output_dir / 'per_image.json'}")


if __name__ == "__main__":
    main()
