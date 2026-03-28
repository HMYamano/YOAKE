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
from htrtdetr.data import load_annotations, SingleFrameDataset, get_collate_fn
from htrtdetr.utils.misc import cxcywh_to_xyxy


COLOR_GT = (220, 50, 50)


def parse_kv(argv):
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


def tensor_to_vis_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """
    見やすさ優先の簡易表示。
    正規化済み画像でも真っ黒/真っ白になりにくいよう、min-maxで 0-255 にします。
    """
    x = image_tensor.detach().cpu().float()
    x = x.permute(1, 2, 0).numpy()  # HWC
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def clamp_boxes_xyxy(boxes_xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes = boxes_xyxy.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, h - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, w - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, h - 1)
    return boxes


def convert_boxes(boxes_raw: torch.Tensor, mode: str, w: int, h: int) -> torch.Tensor:
    """
    mode:
      - pixel_xyxy
      - norm_xyxy
      - pixel_cxcywh
      - norm_cxcywh
    """
    boxes = boxes_raw.clone().float()

    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    if mode == "pixel_xyxy":
        boxes_xyxy = boxes

    elif mode == "norm_xyxy":
        boxes_xyxy = boxes
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h

    elif mode == "pixel_cxcywh":
        boxes_xyxy = cxcywh_to_xyxy(boxes)

    elif mode == "norm_cxcywh":
        boxes_xyxy = cxcywh_to_xyxy(boxes)
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h

    else:
        raise ValueError(f"Unknown mode: {mode}")

    boxes_xyxy = clamp_boxes_xyxy(boxes_xyxy, w, h)
    return boxes_xyxy


def draw_gt_only(image_tensor: torch.Tensor, gt_boxes_xyxy: torch.Tensor, title: str) -> Image.Image:
    img_np = tensor_to_vis_uint8(image_tensor)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    for box in gt_boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_GT, width=2)

    draw.rectangle([4, 4, 260, 22], fill=(0, 0, 0))
    draw.text((6, 6), f"GT only: {title}", fill=(255, 255, 255), font=font)
    return pil_img


@torch.no_grad()
def main():
    kv = parse_kv(sys.argv[1:])

    config_path = kv.get("config", "")
    anno_path = kv.get("anno", "")
    output_dir = Path(kv.get("output_dir", "runs/check_gt_boxes_stage1")).resolve()
    max_images = int(kv.get("max_images", "10"))

    if not config_path:
        raise SystemExit("config=... is required")
    if not anno_path:
        raise SystemExit("anno=... is required")

    cfg = HTRTDETRConfig.from_yaml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

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

    h, w = img_size

    print(f"Config      : {config_path}")
    print(f"Annotation  : {anno_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Image size  : {img_size}")

    records = []
    modes = ["pixel_xyxy", "norm_xyxy", "pixel_cxcywh", "norm_cxcywh"]

    for idx, batch in enumerate(loader):
        if idx >= max_images:
            break

        images = batch["images"]
        targets = batch["targets"]
        meta = batch.get("meta", [{}])

        image_tensor = images[0].cpu()
        gt_raw = targets[0]["boxes"].detach().cpu().float()

        frame_meta = meta[0] if meta else {}
        image_path = str(frame_meta.get("image_path", ""))

        print("=" * 80)
        print(f"idx={idx}")
        print(f"image_path={image_path}")
        print(f"n_gt={len(gt_raw)}")
        if len(gt_raw) > 0:
            print("gt_raw[:3] =")
            print(gt_raw[:3])
            print(f"gt_raw min={gt_raw.min().item():.6f}, max={gt_raw.max().item():.6f}")
        else:
            print("gt_raw is empty")

        saved_files = []

        for mode in modes:
            gt_xyxy = convert_boxes(gt_raw, mode, w, h)
            img = draw_gt_only(image_tensor, gt_xyxy, mode)
            out_name = f"{idx:04d}_{mode}.jpg"
            out_path = vis_dir / out_name
            img.save(out_path, quality=90)
            saved_files.append(str(out_path))

        records.append({
            "index": idx,
            "image_path": image_path,
            "n_gt": int(len(gt_raw)),
            "gt_raw_min": float(gt_raw.min().item()) if len(gt_raw) > 0 else None,
            "gt_raw_max": float(gt_raw.max().item()) if len(gt_raw) > 0 else None,
            "saved_files": saved_files,
        })

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": config_path,
            "annotation": anno_path,
            "dataset_size": len(dataset),
            "max_images": max_images,
            "class_names": class_names,
            "records": records,
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Images  : {vis_dir}")
    print(f"Summary : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()