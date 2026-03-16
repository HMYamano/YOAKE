"""
analyze.py — 解析ツール

使い方:
    # Action timeline 可視化
    python tools/analyze.py mode=timeline predictions=outputs/inference/predictions.json

    # Class distribution 集計
    python tools/analyze.py mode=distribution annotation=data/train/annotations.json

    # Ablation 比較
    python tools/analyze.py mode=ablation results_dir=outputs/ablation

    # ID switch 解析
    python tools/analyze.py mode=id_switch predictions=outputs/inference/predictions.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path
from typing import Dict, List

from htrtdetr.analysis import (
    ActionTimelineVisualizer,
    plot_confusion_matrix,
    compute_class_distribution,
    plot_distribution,
    plot_confidence_histogram,
    IDSwitchAnalyzer,
    AblationResultManager,
)
from htrtdetr.utils import get_logger

logger = get_logger("analyze")


def parse_argv(argv) -> Dict:
    kwargs = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kwargs[k] = v
    return kwargs


# ---------------------------------------------------------------------------
# 各モード
# ---------------------------------------------------------------------------

def mode_timeline(predictions_path: str, output_dir: str, action_names: List[str]) -> None:
    """per-track action timeline を描画する"""
    with open(predictions_path) as f:
        preds = json.load(f)

    # {track_id: [action_id per frame]}
    track_timelines: Dict[int, List[int]] = {}
    for frame_key in sorted(preds.keys(), key=int):
        frame_data = preds[frame_key]
        track_ids = frame_data.get("track_ids", [])
        action_ids = frame_data.get("action_ids", [])
        for tid, aid in zip(track_ids, action_ids):
            if tid not in track_timelines:
                track_timelines[tid] = []
            track_timelines[tid].append(aid)

    if not track_timelines:
        logger.warning("No tracking data found in predictions")
        return

    vis = ActionTimelineVisualizer(action_names)
    save_path = str(Path(output_dir) / "action_timeline.png")
    vis.plot(track_timelines, save_path)
    logger.info(f"Action timeline saved: {save_path}")


def mode_distribution(annotation_path: str, output_dir: str) -> None:
    """データセット分布を集計・可視化する"""
    dist = compute_class_distribution(annotation_path)
    logger.info(f"Dataset stats: {dist}")

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # JSON 保存
    with open(output_dir_p / "distribution.json", "w") as f:
        json.dump(dist, f, indent=2)

    # グラフ
    if dist["class_distribution"]:
        plot_distribution(
            dist["class_distribution"],
            str(output_dir_p / "class_distribution.png"),
            title="Class Distribution",
        )
    if dist["action_distribution"]:
        plot_distribution(
            dist["action_distribution"],
            str(output_dir_p / "action_distribution.png"),
            title="Action Distribution",
        )
    logger.info(f"Distribution plots saved to: {output_dir}")


def mode_ablation(results_dir: str) -> None:
    """ablation 比較表を作成する"""
    results_dir_p = Path(results_dir)
    mgr = AblationResultManager(str(results_dir_p))

    # 各 json ファイルを読み込む
    for json_path in sorted(results_dir_p.glob("*.json")):
        if json_path.name == "ablation_comparison.json":
            continue
        with open(json_path) as f:
            metrics = json.load(f)
        mgr.add_result(json_path.stem, metrics)

    mgr.save_comparison()
    mgr.print_comparison()


def mode_id_switch(predictions_path: str, output_dir: str) -> None:
    """ID switch を解析する"""
    with open(predictions_path) as f:
        preds = json.load(f)

    import numpy as np
    analyzer = IDSwitchAnalyzer(iou_threshold=0.5)

    for frame_key in sorted(preds.keys(), key=int):
        frame_idx = int(frame_key)
        frame_data = preds[frame_key]
        boxes = np.array(frame_data.get("boxes", []), dtype=np.float32)
        track_ids = np.array(frame_data.get("track_ids", []), dtype=int)
        if len(boxes) > 0:
            analyzer.update(frame_idx, boxes, track_ids)

    summary = analyzer.get_summary()
    logger.info(f"ID Switches: {summary['total_id_switches']}")

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    analyzer.save(str(output_dir_p / "id_switch_analysis.json"))
    logger.info(f"ID switch analysis saved to: {output_dir}")


def mode_confidence(predictions_path: str, output_dir: str) -> None:
    """action confidence ヒストグラムを描画する"""
    with open(predictions_path) as f:
        preds = json.load(f)

    confidences = []
    for frame_data in preds.values():
        # action_probs がある場合: max prob を confidence として使う
        for probs in frame_data.get("action_probs", []):
            if isinstance(probs, list) and probs:
                confidences.append(max(probs))
        # scores がある場合
        for s in frame_data.get("scores", []):
            confidences.append(float(s))

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    if confidences:
        plot_confidence_histogram(
            confidences,
            str(output_dir_p / "confidence_histogram.png"),
        )
    else:
        logger.warning("No confidence data found in predictions. "
                       "Expected keys: 'action_probs' or 'scores'.")


def mode_confusion_matrix(
    eval_result_path: str,
    output_dir: str,
    action_names: List[str],
) -> None:
    """eval_stage2 の JSON 結果から confusion matrix を描画する。

    eval_result_path: outputs/eval_stage2/results_stage2.json
                      または confusion_matrix キーを持つ任意の JSON
    """
    import numpy as np

    with open(eval_result_path) as f:
        results = json.load(f)

    cm_data = results.get("confusion_matrix", None)
    if cm_data is None:
        logger.error("'confusion_matrix' key not found in eval result JSON.")
        logger.info("Run eval_stage2.py first and ensure it saves a confusion matrix.")
        return

    cm = np.array(cm_data, dtype=np.int64)
    n = cm.shape[0]
    names = action_names[:n] if len(action_names) >= n else action_names + [str(i) for i in range(len(action_names), n)]

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    save_path = str(output_dir_p / "confusion_matrix.png")
    plot_confusion_matrix(cm, names, save_path, normalize=True)
    logger.info(f"Confusion matrix saved: {save_path}")

    # Also save raw counts
    plot_confusion_matrix(
        cm, names,
        str(output_dir_p / "confusion_matrix_counts.png"),
        normalize=False,
        title="Confusion Matrix (counts)",
    )


def mode_branch_stats(
    anno_path: str,
    checkpoint: str,
    output_dir: str,
    stage: int = 2,
) -> None:
    """HTM の各 branch の feature activation 統計を計測して可視化する。

    model の forward_geo_sequence を hook して各 branch の出力分布を集計する。
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))

    import torch
    import numpy as np

    from htrtdetr.config.config import get_stage2_config, get_stage3_config
    from htrtdetr.models import build_model
    from htrtdetr.data.fly_dataset import build_dataloaders
    from htrtdetr.data.annotation import load_annotation
    from htrtdetr.utils.misc import load_checkpoint

    cfg = get_stage2_config() if stage == 2 else get_stage3_config()
    anno = load_annotation(anno_path)
    loaders = build_dataloaders(train_anno=anno, val_anno=None, stage=stage, cfg=cfg.data)
    loader = loaders["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(stage)

    if Path(checkpoint).exists():
        load_checkpoint(model, checkpoint, device=device)
        logger.info(f"Loaded checkpoint: {checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint} — using random weights")

    model.eval()

    # Register forward hooks on HTM branches
    branch_activations: Dict[str, List[np.ndarray]] = {
        "short": [], "mid": [], "long": []
    }
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            branch_activations[name].append(out.detach().cpu().numpy())
        return _hook

    temporal = model.temporal
    if hasattr(temporal, "short_branch"):
        hooks.append(temporal.short_branch.register_forward_hook(_make_hook("short")))
    if hasattr(temporal, "mid_branch"):
        hooks.append(temporal.mid_branch.register_forward_hook(_make_hook("mid")))
    if hasattr(temporal, "long_branch"):
        hooks.append(temporal.long_branch.register_forward_hook(_make_hook("long")))

    with torch.no_grad():
        for batch in loader:
            geo = batch["geo_features"].to(device)
            model.forward_geo_sequence(geo)

    for h in hooks:
        h.remove()

    # Compute statistics per branch
    stats: Dict[str, Dict] = {}
    for name, acts in branch_activations.items():
        if not acts:
            continue
        flat = np.concatenate([a.reshape(-1) for a in acts])
        stats[name] = {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "l2_norm_mean": float(np.sqrt((flat ** 2).mean())),
        }

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    json_path = output_dir_p / "branch_feature_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Branch stats saved: {json_path}")

    print("\n=== Branch Feature Statistics ===")
    for name, s in stats.items():
        print(f"  {name:6s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
              f"l2_norm={s['l2_norm_mean']:.4f}  [{s['min']:.3f}, {s['max']:.3f}]")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(stats.keys())
        metrics = ["mean", "std", "l2_norm_mean"]
        colors = ["steelblue", "coral", "seagreen"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric, color in zip(axes, metrics, colors):
            vals = [stats[n][metric] for n in names]
            ax.bar(names, vals, color=color)
            ax.set_title(metric)
            ax.set_xlabel("Branch")
        plt.suptitle("HTM Branch Feature Statistics")
        plt.tight_layout()
        plt.savefig(output_dir_p / "branch_feature_stats.png", dpi=150)
        plt.close()
        logger.info(f"Branch stats plot saved: {output_dir_p / 'branch_feature_stats.png'}")
    except ImportError:
        logger.warning("matplotlib not available — plot skipped")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    kwargs = parse_argv(sys.argv)
    mode = kwargs.get("mode", "distribution")
    output_dir = kwargs.get("output", "outputs/analysis")

    # デフォルトの action_names
    action_names = ["idle", "walk", "groom", "interact", "other"]
    ann_path = kwargs.get("annotation", "")
    if ann_path and Path(ann_path).exists():
        from htrtdetr.data import load_annotations
        try:
            _, _, action_names = load_annotations(ann_path)
        except Exception:
            pass

    if mode == "timeline":
        preds = kwargs.get("predictions", "outputs/inference/predictions.json")
        if not Path(preds).exists():
            logger.error(f"Predictions not found: {preds}")
            sys.exit(1)
        mode_timeline(preds, output_dir, action_names)

    elif mode == "distribution":
        if not ann_path:
            logger.error("annotation= path required for distribution mode")
            sys.exit(1)
        mode_distribution(ann_path, output_dir)

    elif mode == "ablation":
        results_dir = kwargs.get("results_dir", "outputs/ablation")
        mode_ablation(results_dir)

    elif mode == "id_switch":
        preds = kwargs.get("predictions", "outputs/inference/predictions.json")
        if not Path(preds).exists():
            logger.error(f"Predictions not found: {preds}")
            sys.exit(1)
        mode_id_switch(preds, output_dir)

    elif mode == "confidence":
        preds = kwargs.get("predictions", "outputs/inference/predictions.json")
        mode_confidence(preds, output_dir)

    elif mode == "confusion_matrix":
        eval_result = kwargs.get("eval_result", "outputs/eval_stage2/results_stage2.json")
        mode_confusion_matrix(eval_result, output_dir, action_names)

    elif mode == "branch_stats":
        anno_p = kwargs.get("annotation", "data/sample/annotations_val.json")
        ckpt = kwargs.get("checkpoint", "outputs/stage2/checkpoint_best.pth")
        stage_num = int(kwargs.get("stage", "2"))
        mode_branch_stats(anno_p, ckpt, output_dir, stage=stage_num)

    else:
        logger.error(f"Unknown mode: {mode}")
        logger.info(
            "Available modes: timeline, distribution, ablation, id_switch, "
            "confidence, confusion_matrix, branch_stats"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
