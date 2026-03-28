"""
visualize_stage1_train.py — Stage 1 モデルで train データを解析・可視化

モデル予測 (緑) と GT (赤) を重ねて画像に描画して保存する。
メトリクス (AP50, recall など) も計算して出力する。

使い方:
  # デフォルト (YOAKE_tryal の train データを自動検出)
  python scripts/visualize_stage1_train.py

  # 直接指定
  python scripts/visualize_stage1_train.py \
      anno=C:/Users/utopi/Desktop/YOAKE_tryal/data/train/annotations.json \
      checkpoint=outputs/stage1/stage1_best.pth \
      output_dir=outputs/eval_stage1_train \
      score_thresh=0.3 \
      max_images=50
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from htrtdetr.config.config import get_stage1_config
from htrtdetr.models import build_model
from htrtdetr.data import load_annotations, SingleFrameDataset, get_collate_fn
from htrtdetr.evaluation.evaluator import DetectionEvaluator
from htrtdetr.utils.misc import load_checkpoint, cxcywh_to_xyxy


_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

# 描画色 (RGB)
COLOR_GT   = (220, 50,  50)   # 赤: GT
COLOR_PRED = (50,  200, 80)   # 緑: 予測
COLOR_TEXT_BG = (0, 0, 0)


def parse_overrides(argv) -> dict:
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v
    return overrides


def find_checkpoint(candidates: list[str]) -> str | None:
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def draw_boxes_on_image(
    image_tensor: torch.Tensor,   # (3, H, W) float [0,1]
    pred_boxes_xyxy: torch.Tensor,  # (N_pred, 4) pixel coords, xyxy
    pred_scores: torch.Tensor,      # (N_pred,)
    gt_boxes_xyxy: torch.Tensor,    # (N_gt, 4) pixel coords, xyxy
    image_size: tuple[int, int],    # (H, W) of the resized image
    orig_size: tuple[int, int] | None = None,  # (orig_H, orig_W) for scaling back
) -> Image.Image:
    """テンソルから PIL 画像を作り、GT と予測 BBox を描画して返す。"""
    # テンソル → numpy → PIL
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    H, W = image_size

    # 座標スケールが [0,1] 正規化なら pixel へ変換する必要はない
    # (eval_stage1.py と同様、cxcywh_to_xyxy → [0,1] 正規化 xyxy の場合)
    def to_pixel(box_xyxy_norm):
        """(x1,y1,x2,y2) 正規化 → ピクセル座標"""
        x1, y1, x2, y2 = box_xyxy_norm
        return (
            float(x1) * W,
            float(y1) * H,
            float(x2) * W,
            float(y2) * H,
        )

    # GT (赤)
    for box in gt_boxes_xyxy:
        b = box.cpu().numpy()
        # GT は pixel 座標で渡される (loader の targets は pixel coords)
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_GT, width=2)

    # 予測 (緑) — pred_boxes_xyxy は [0,1] 正規化
    for i, box in enumerate(pred_boxes_xyxy):
        b = box.cpu().numpy()
        x1, y1, x2, y2 = to_pixel(b)
        score = float(pred_scores[i].cpu())
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_PRED, width=2)
        label = f"{score:.2f}"
        # テキスト背景
        text_pos = (x1, max(0, y1 - 14))
        draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 36, text_pos[1] + 13],
                       fill=COLOR_PRED)
        draw.text(text_pos, label, fill=(0, 0, 0))

    return pil_img


@torch.no_grad()
def main():
    overrides = parse_overrides(sys.argv[1:])

    root            = overrides.pop("root", _DEFAULT_ROOT)
    anno_path       = overrides.get("anno",       f"{root}/data/train/annotations.json")
    checkpoint_path = overrides.get("checkpoint", None)
    output_dir      = Path(overrides.get("output_dir", f"{root}/runs/visualize/stage1_train"))
    score_thresh    = float(overrides.get("score_thresh", "0.05"))
    max_images      = int(overrides.get("max_images", "200"))  # 保存する最大画像数

    if checkpoint_path is None:
        checkpoint_path = find_checkpoint([
            f"{root}/outputs/stage1/stage1_best.pth",
            f"{root}/outputs/stage1/last.pth",
        ])

    # 絶対パスに解決してから表示・作成する
    output_dir = output_dir.resolve()
    vis_dir    = output_dir / "images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Analyzing Stage 1 on TRAIN data")
    print(f"  Annotation : {anno_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Output     : {output_dir}")
    print(f"  Images dir : {vis_dir}")
    print(f"  Device     : {device}")
    print(f"  Score thresh: {score_thresh}")
    print(f"  Max images : {max_images}")

    # ----- Config -----
    cfg = get_stage1_config()

    # ----- アノテーション読み込み -----
    if not Path(anno_path).exists():
        print(f"ERROR: annotation not found: {anno_path}")
        sys.exit(1)

    # アノテーション確認後にフォルダを作成する
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    videos, class_names, action_names = load_annotations(anno_path)
    print(f"\nDataset info:")
    print(f"  Videos       : {len(videos)}")
    total_frames = sum(len(v.frames) for v in videos)
    total_objects = sum(
        len(f.objects) for v in videos for f in v.frames
    )
    print(f"  Total frames : {total_frames}")
    print(f"  Total objects: {total_objects}")
    print(f"  Class names  : {class_names}")

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
        num_workers=0,
        pin_memory=False,
        collate_fn=get_collate_fn("single"),
    )
    print(f"  Dataset size : {len(dataset)} frames\n")

    # ----- Model -----
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: no checkpoint found — using random weights")

    model.eval()

    evaluator = DetectionEvaluator(iou_thresholds=[0.5, 0.75])
    all_scores_list = []

    # 統計集計用
    n_gt_total   = 0
    n_pred_total = 0
    images_saved = 0
    per_frame_stats = []

    H, W = _img_size  # (height, width)

    print("Running inference...")
    for batch_idx, batch in enumerate(loader):
        images  = batch["images"].to(device)
        targets = batch["targets"]
        meta    = batch.get("meta", [{}] * images.shape[0])

        out = model.forward_single_frame(images)

        # (B, Q, C+1) → スコア, (B, Q, 4) cxcywh → xyxy [0,1]
        probs         = out.pred_logits.softmax(dim=-1)[:, :, :-1]  # bg 除去
        scores, labels = probs.max(dim=-1)   # (B, Q)
        pred_xyxy_norm = cxcywh_to_xyxy(out.pred_boxes)            # (B, Q, 4) [0,1]

        B = images.shape[0]
        for b in range(B):
            s = scores[b].cpu().numpy()
            all_scores_list.append(s)

            mask = scores[b] > score_thresh
            pred_boxes_b = pred_xyxy_norm[b][mask]   # (N_pred, 4) [0,1]
            pred_scores_b = scores[b][mask]           # (N_pred,)
            pred_labels_b = labels[b][mask]           # (N_pred,)

            gt_boxes_b   = targets[b]["boxes"]        # (N_gt, 4) pixel xyxy
            gt_classes_b = targets[b]["class_ids"]    # (N_gt,)

            n_gt   = len(gt_boxes_b)
            n_pred = int(mask.sum().item())
            n_gt_total   += n_gt
            n_pred_total += n_pred

            # GT を [0,1] 正規化 xyxy に変換 (evaluator 用)
            gt_boxes_norm = gt_boxes_b.float().clone()
            gt_boxes_norm[:, [0, 2]] /= W
            gt_boxes_norm[:, [1, 3]] /= H

            evaluator.update(
                pred_boxes_b.cpu(),
                pred_scores_b.cpu(),
                pred_labels_b.cpu(),
                gt_boxes_norm.cpu(),
                gt_classes_b.cpu(),
                box_format="xyxy",
            )

            # フレームごと統計
            frame_meta = meta[b] if b < len(meta) else {}
            per_frame_stats.append({
                "frame_index": frame_meta.get("frame_index", batch_idx * B + b),
                "image_path" : str(frame_meta.get("image_path", "")),
                "n_gt"       : n_gt,
                "n_pred"     : n_pred,
                "max_score"  : float(scores[b].max().item()),
                "mean_score" : float(scores[b].mean().item()),
            })

            # ----- 可視化 -----
            if images_saved < max_images:
                pil_img = draw_boxes_on_image(
                    image_tensor    = images[b],
                    pred_boxes_xyxy = pred_boxes_b,
                    pred_scores     = pred_scores_b,
                    gt_boxes_xyxy   = gt_boxes_b,
                    image_size      = (H, W),
                )
                # 凡例テキスト
                draw = ImageDraw.Draw(pil_img)
                draw.text((5, 5),  f"GT({n_gt})   red",   fill=COLOR_GT)
                draw.text((5, 18), f"Pred({n_pred}) green", fill=COLOR_PRED)

                img_name = f"frame_{batch_idx:04d}_{b:02d}.jpg"
                pil_img.save(vis_dir / img_name, quality=90)
                images_saved += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] batches done")

    print(f"\nInference complete.")
    print(f"  Images saved   : {images_saved}")
    print(f"  Total GT boxes : {n_gt_total}")
    print(f"  Total pred boxes (thresh>{score_thresh}): {n_pred_total}")

    # ===== スコア分布 =====
    all_scores_np = np.concatenate(all_scores_list)
    thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n=== Score Distribution (foreground probability) ===")
    print(f"  total queries : {len(all_scores_np)}")
    print(f"  min   : {all_scores_np.min():.6f}")
    print(f"  max   : {all_scores_np.max():.6f}")
    print(f"  mean  : {all_scores_np.mean():.6f}")
    print(f"  median: {np.median(all_scores_np):.6f}")
    print(f"  p90   : {np.percentile(all_scores_np, 90):.6f}")
    print(f"  p99   : {np.percentile(all_scores_np, 99):.6f}")
    print("  --- queries above threshold ---")
    for thr in thresholds:
        n = (all_scores_np > thr).sum()
        pct = 100.0 * n / len(all_scores_np)
        print(f"  > {thr:.2f} : {n:6d} queries  ({pct:.2f}%)")

    # ===== Detection メトリクス =====
    results = evaluator.compute()
    print("\n=== Stage 1 Detection Metrics (TRAIN data) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ===== 保存 =====
    score_report = {
        "min"   : float(all_scores_np.min()),
        "max"   : float(all_scores_np.max()),
        "mean"  : float(all_scores_np.mean()),
        "median": float(np.median(all_scores_np)),
        "p90"   : float(np.percentile(all_scores_np, 90)),
        "p99"   : float(np.percentile(all_scores_np, 99)),
        "above_threshold": {
            f">{thr}": int((all_scores_np > thr).sum()) for thr in thresholds
        },
    }

    summary = {
        "dataset"       : "train",
        "annotation"    : str(anno_path),
        "checkpoint"    : str(checkpoint_path),
        "score_threshold": score_thresh,
        "n_frames"      : len(per_frame_stats),
        "n_gt_total"    : n_gt_total,
        "n_pred_total"  : n_pred_total,
        "metrics"       : {k: float(v) for k, v in results.items()},
        "score_distribution": score_report,
    }

    summary_path = output_dir / "summary_train.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved  : {summary_path}")

    per_frame_path = output_dir / "per_frame_stats.json"
    with open(per_frame_path, "w", encoding="utf-8") as f:
        json.dump(per_frame_stats, f, indent=2, ensure_ascii=False)
    print(f"Per-frame stats: {per_frame_path}")

    print(f"Visualizations : {vis_dir}  ({images_saved} images)")
    print("\nDone.")


if __name__ == "__main__":
    main()
