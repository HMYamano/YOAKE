"""
run_ablation.py — Ablation Experiment Runner

指定した ablation suite の全 variant を順番に eval し、
結果を outputs/ablation/<suite_name>.csv / .json に自動集計する。

使い方:
  python tools/run_ablation.py \
      suite=temporal_branch \
      stage=2 \
      anno=data/splits/annotations_val.json \
      checkpoint_dir=outputs/stage2 \
      output_dir=outputs/ablation \
      dry_run=0

suite:
  temporal_branch | dilation | clip_length | interaction | memory | fusion
  (または custom yaml を指定: suite=configs/my_ablation.yaml)

dry_run=1:
  チェックポイント不要で動作確認のみ実行し CSV に dummy 結果を書く

注意:
  各 variant のチェックポイントは checkpoint_dir/<variant_name>/checkpoint_best.pth
  を自動探索する。見つからない場合は base checkpoint_dir/checkpoint_best.pth を使う。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from htrtdetr.config.config import (
    HTRTDETRConfig,
    get_stage2_config,
    get_stage3_config,
)
from htrtdetr.config.ablation import (
    AblationSuite,
    AblationVariant,
    ResultsCollector,
    get_suite,
    list_suites,
)
from htrtdetr.models import build_model
from htrtdetr.data.fly_dataset import build_dataloaders
from htrtdetr.data.annotation import load_annotation
from htrtdetr.utils.misc import load_checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Eval helpers (reuse logic from eval_stage2 / eval_stage3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_stage2(model, loader, device) -> Dict[str, float]:
    """Action classification accuracy + macro-F1 (simplified)."""
    from htrtdetr.evaluation.evaluator import ActionEvaluator
    from htrtdetr.config.config import HTRTDETRConfig

    cfg = get_stage2_config()
    evaluator = ActionEvaluator(num_classes=cfg.model.action_head.num_actions)
    model.eval()

    for batch in loader:
        geo = batch["geo_features"].to(device)
        gt_actions = batch["action_ids"]
        out = model.forward_geo_sequence(geo)
        preds = out["action_logits"].argmax(dim=-1).cpu()
        scores = out["action_logits"].softmax(dim=-1).cpu()
        for i in range(geo.shape[0]):
            if gt_actions[i] < 0:
                continue
            evaluator.update(
                pred_labels=preds[i : i + 1],
                gt_labels=gt_actions[i : i + 1],
                pred_scores=scores[i : i + 1],
            )
    return evaluator.compute()


@torch.no_grad()
def _eval_stage3(model, loader, device) -> Dict[str, float]:
    """ID classification accuracy + embedding similarity gap."""
    all_embeddings = []
    all_local_ids = []
    correct = 0
    total = 0
    model.eval()

    for batch in loader:
        geo = batch["geo_features"].to(device)
        gt_ids = batch["local_track_ids"]
        out = model.forward_geo_sequence(geo)
        emb = out["id_embeddings"].cpu()
        logits = out["id_logits"].cpu()
        valid = gt_ids >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            correct += (preds == gt_ids[valid]).sum().item()
            total += valid.sum().item()
            all_embeddings.append(emb[valid])
            all_local_ids.append(gt_ids[valid])

    id_acc = correct / max(total, 1)
    within_sim = 0.0
    between_sim = 0.0
    if all_embeddings:
        embs = torch.cat(all_embeddings, dim=0)
        ids = torch.cat(all_local_ids, dim=0)
        embs_norm = F.normalize(embs, p=2, dim=-1)
        sim_mat = embs_norm @ embs_norm.t()
        same_id = (ids.unsqueeze(0) == ids.unsqueeze(1))
        same_id.fill_diagonal_(False)
        diff_id = ~same_id
        diff_id.fill_diagonal_(False)
        within_sim = sim_mat[same_id].mean().item() if same_id.any() else 0.0
        between_sim = sim_mat[diff_id].mean().item() if diff_id.any() else 0.0

    return {
        "id_accuracy": id_acc,
        "within_class_similarity": within_sim,
        "between_class_similarity": between_sim,
        "similarity_gap": within_sim - between_sim,
        "total_samples": total,
    }


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def _find_checkpoint(variant_name: str, checkpoint_dir: str) -> Optional[str]:
    """variant 固有 → base の順にチェックポイントを探す。"""
    candidates = [
        Path(checkpoint_dir) / variant_name / "checkpoint_best.pth",
        Path(checkpoint_dir) / variant_name / "checkpoint_last.pth",
        Path(checkpoint_dir) / "checkpoint_best.pth",
        Path(checkpoint_dir) / "checkpoint_last.pth",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _run_variant(
    variant: AblationVariant,
    base_cfg: HTRTDETRConfig,
    stage: int,
    anno_path: str,
    checkpoint_dir: str,
    device: torch.device,
    dry_run: bool,
) -> Dict[str, float]:
    """1 variant を eval して results dict を返す。"""
    print(f"\n  Variant: {variant.name} - {variant.description}")

    if dry_run:
        print("    [dry_run] Returning dummy results.")
        return {
            "action_accuracy": 0.0,
            "action_macro_f1": 0.0,
            "id_accuracy": 0.0,
            "similarity_gap": 0.0,
        }

    cfg = variant.apply(base_cfg)
    anno = load_annotation(anno_path)
    loaders = build_dataloaders(
        train_anno=anno, val_anno=None, stage=stage, cfg=cfg.data
    )
    loader = loaders["train"]

    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(stage)

    ckpt = _find_checkpoint(variant.name, checkpoint_dir)
    if ckpt:
        load_checkpoint(model, ckpt, device=device)
        print(f"    Loaded: {ckpt}")
    else:
        print(f"    WARNING: No checkpoint found for {variant.name!r} in {checkpoint_dir}")

    if stage == 2:
        return _eval_stage2(model, loader, device)
    elif stage == 3:
        return _eval_stage3(model, loader, device)
    else:
        raise ValueError(f"Ablation runner currently supports stage 2 or 3. Got: {stage}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse(sys.argv[1:])

    suite_name = args.get("suite", "temporal_branch")
    stage = int(args.get("stage", "2"))
    anno_path = args.get("anno", "data/splits/annotations_val.json")
    checkpoint_dir = args.get("checkpoint_dir", f"outputs/stage{stage}")
    output_dir = Path(args.get("output_dir", "outputs/ablation"))
    dry_run = args.get("dry_run", "0").strip().lower() in ("1", "true", "yes")
    variant_filter = args.get("variant", None)  # run only this variant

    print(f"=== Ablation Runner ===")
    print(f"  Suite       : {suite_name}")
    print(f"  Stage       : {stage}")
    print(f"  Anno        : {anno_path}")
    print(f"  Checkpoint  : {checkpoint_dir}")
    print(f"  Output      : {output_dir}")
    print(f"  Dry run     : {dry_run}")

    # List mode
    if suite_name == "list":
        print("\nAvailable suites:", ", ".join(list_suites()))
        return

    suite = get_suite(suite_name)

    if stage == 2:
        base_cfg = get_stage2_config()
    elif stage == 3:
        base_cfg = get_stage3_config()
    else:
        raise ValueError(f"Unsupported stage: {stage}. Use 2 or 3.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device      : {device}")

    collector = ResultsCollector(suite_name=suite_name)

    variants = suite.variants
    if variant_filter:
        variants = [v for v in variants if v.name == variant_filter]
        if not variants:
            print(f"  ERROR: Variant {variant_filter!r} not found in suite {suite_name!r}")
            return

    failed: List[str] = []
    for variant in variants:
        try:
            results = _run_variant(
                variant=variant,
                base_cfg=base_cfg,
                stage=stage,
                anno_path=anno_path,
                checkpoint_dir=checkpoint_dir,
                device=device,
                dry_run=dry_run,
            )
            results["stage"] = stage
            results["description"] = variant.description
            collector.add(variant.name, results)
        except Exception as exc:
            print(f"    ERROR running {variant.name}: {exc}")
            failed.append(variant.name)
            collector.add(variant.name, {"error": str(exc), "stage": stage})

    # Print and save
    collector.print_table()
    output_dir.mkdir(parents=True, exist_ok=True)
    collector.save(output_dir / f"{suite_name}_stage{stage}.csv")

    if failed:
        print(f"\nFailed variants: {failed}")
    else:
        print("\nAll variants completed successfully.")


if __name__ == "__main__":
    main()
