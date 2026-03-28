"""
Single-entry CLI for YOAKE.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _ask_confirm(message: str) -> bool:
    try:
        resp = input(f"{message} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return resp in {"y", "yes"}


def parse_argv(argv: list) -> Tuple[str, Dict[str, Any]]:
    config_path = ""
    overrides: Dict[str, Any] = {}

    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k == "config":
                config_path = v
                continue
            if v.lower() in {"true", "false"}:
                value: Any = v.lower() == "true"
            else:
                try:
                    value = int(v)
                except ValueError:
                    try:
                        value = float(v)
                    except ValueError:
                        value = v

            keys = k.split(".")
            dst = overrides
            for key in keys[:-1]:
                dst = dst.setdefault(key, {})
            dst[keys[-1]] = value
            continue

        if not config_path and isinstance(arg, str) and arg.endswith((".yaml", ".yml")):
            config_path = arg

    return config_path, overrides


def _extract_kv(argv: list, key: str, default: Any = None) -> Tuple[Any, list]:
    value = default
    remaining = []
    for arg in argv:
        if isinstance(arg, str) and arg.startswith(f"{key}="):
            value = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
    return value, remaining


def _is_annotation_file(path_str: str) -> bool:
    return Path(path_str).suffix.lower() == ".json"


def resolve_annotation_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if _is_annotation_file(path_str) else path / "annotations.json"


def resolve_dataset_root(path_str: str) -> str:
    path = Path(path_str)
    return str(path.parent if _is_annotation_file(path_str) else path)


def find_checkpoint(root: str, stage: int) -> str:
    root_path = Path(root)
    candidates = sorted(
        [
            *root_path.glob(f"runs/train/stage{stage}/weights/best.pth"),
            *root_path.glob(f"runs/train/*/stage{stage}/weights/best.pth"),
            *root_path.glob(f"runs/train/stage{stage}/stage{stage}_best.pth"),
            *root_path.glob(f"runs/train/*/stage{stage}/stage{stage}_best.pth"),
            *root_path.glob(f"outputs/stage{stage}/stage{stage}_best.pth"),
            *root_path.glob(f"outputs/*/stage{stage}/stage{stage}_best.pth"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


def _build_dataloaders(cfg, stage: int, use_dummy: bool):
    from torch.utils.data import DataLoader

    from .data import (
        DummyDataset,
        SingleFrameDataset,
        SlidingWindowDataset,
        get_collate_fn,
        load_annotations,
    )

    num_workers = cfg.data.num_workers
    loader_kwargs = dict(
        pin_memory=cfg.data.pin_memory,
        persistent_workers=(num_workers > 0 and cfg.data.persistent_workers),
        prefetch_factor=(cfg.data.prefetch_factor if num_workers > 0 else None),
    )
    img_size = (
        (cfg.data.image_size, cfg.data.image_size)
        if isinstance(cfg.data.image_size, int)
        else tuple(cfg.data.image_size)
    )

    if stage == 1:
        if use_dummy:
            train_ds = DummyDataset(
                n_samples=100,
                window_size=1,
                image_size=img_size,
                num_classes=cfg.model.detector.head.num_classes,
                num_actions=cfg.model.action_head.num_actions,
                mode="single",
            )
            val_ds = DummyDataset(
                n_samples=20,
                window_size=1,
                image_size=img_size,
                num_classes=cfg.model.detector.head.num_classes,
                num_actions=cfg.model.action_head.num_actions,
                mode="single",
            )
        else:
            train_ann = resolve_annotation_path(str(cfg.data.train_root))
            val_ann = resolve_annotation_path(str(cfg.data.val_root))
            train_root = resolve_dataset_root(str(cfg.data.train_root))
            val_root = resolve_dataset_root(str(cfg.data.val_root))
            train_videos, _, _ = load_annotations(str(train_ann))
            val_videos, _, _ = load_annotations(str(val_ann)) if val_ann.exists() else (train_videos, None, None)
            train_ds = SingleFrameDataset(
                train_videos,
                image_size=img_size,
                augment=cfg.data.augment_train,
                data_root=train_root,
            )
            val_ds = SingleFrameDataset(
                val_videos,
                image_size=img_size,
                augment=False,
                data_root=val_root,
            )
        mode = "single"
    else:
        if use_dummy:
            train_ds = DummyDataset(
                n_samples=100,
                window_size=cfg.data.window_size,
                image_size=img_size,
                num_classes=cfg.model.detector.head.num_classes,
                num_actions=cfg.model.action_head.num_actions,
                mode="sequence",
            )
            val_ds = DummyDataset(
                n_samples=20,
                window_size=cfg.data.window_size,
                image_size=img_size,
                num_classes=cfg.model.detector.head.num_classes,
                num_actions=cfg.model.action_head.num_actions,
                mode="sequence",
            )
        else:
            train_ann = resolve_annotation_path(str(cfg.data.train_root))
            val_ann = resolve_annotation_path(str(cfg.data.val_root))
            train_root = resolve_dataset_root(str(cfg.data.train_root))
            val_root = resolve_dataset_root(str(cfg.data.val_root))
            train_videos, _, _ = load_annotations(str(train_ann))
            val_videos, _, _ = load_annotations(str(val_ann))
            train_ds = SlidingWindowDataset(
                train_videos,
                window_size=cfg.data.window_size,
                stride=cfg.data.window_stride,
                image_size=img_size,
                augment=cfg.data.augment_train,
                data_root=train_root,
                require_action=(stage == 2),
            )
            val_ds = SlidingWindowDataset(
                val_videos,
                window_size=cfg.data.window_size,
                stride=cfg.data.window_size,
                image_size=img_size,
                augment=False,
                data_root=val_root,
            )
        mode = "sequence"

    collate_fn = get_collate_fn(mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader


def run_train(stage: int, argv: list) -> None:
    from .config import HTRTDETRConfig, clamp_temporal_branches
    from .config.config import (
        get_stage1_config,
        get_stage2_config,
        get_stage3_config,
        get_stage4_config,
    )
    from .models import build_model
    from .training import Trainer
    from .utils import get_logger, set_seed

    try:
        from .utils import load_model_weights
    except ImportError:
        load_model_weights = None

    builders = {
        1: get_stage1_config,
        2: get_stage2_config,
        3: get_stage3_config,
        4: get_stage4_config,
    }
    if stage not in builders:
        raise ValueError(f"stage must be 1-4, got {stage}")

    config_path, overrides = parse_argv(argv)
    root = overrides.pop("root", str(Path.cwd()))
    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
        if overrides:
            cfg = cfg.merge(overrides)
    else:
        cfg = builders[stage](overrides if overrides else None)

    output_dir = Path(str(cfg.train.output_dir)) if cfg.train.output_dir else None
    if not overrides.get("train", {}).get("output_dir") and (
        output_dir is None or not output_dir.is_absolute()
    ):
        cfg.train.output_dir = str(Path(root) / f"runs/train/stage{stage}")

    if stage > 1:
        clamp_temporal_branches(cfg)

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"stage{stage}", log_file=str(Path(cfg.train.output_dir) / f"stage{stage}.log"))
    logger.info("=== yoake train stage=%s ===", stage)
    logger.info("output : %s", cfg.train.output_dir)
    logger.info("root   : %s", root)

    set_seed(cfg.train.seed)
    train_ann = resolve_annotation_path(str(cfg.data.train_root)) if cfg.data.train_root else Path(".")
    use_dummy = not train_ann.exists()
    if use_dummy:
        logger.warning("Train data not found at %s.", cfg.data.train_root)
        if not _ask_confirm("Run a DummyDataset smoke test instead?"):
            logger.error("Aborted. Set data.train_root to a real dataset before training.")
            sys.exit(1)

    train_loader, val_loader = _build_dataloaders(cfg, stage, use_dummy)
    model = build_model(cfg.model)
    model.set_stage(stage)
    logger.info("Model parameters: %s", model.num_parameters())

    if not cfg.train.resume and stage > 1 and load_model_weights is not None:
        for prev in range(stage - 1, 0, -1):
            ckpt = find_checkpoint(root, prev)
            if ckpt:
                logger.info("Loading Stage %s weights: %s", prev, ckpt)
                load_model_weights(ckpt, model, strict=False)
                break

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.train,
        loss_cfg=cfg.loss,
        num_classes=cfg.model.detector.head.num_classes,
        num_actions=cfg.model.action_head.num_actions,
        max_ids=cfg.model.id_head.max_ids,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        use_dummy=use_dummy,
        metrics_cfg=getattr(cfg, "metrics", None),
    )
    trainer.train()


def run_val(stage: int, argv: list) -> None:
    from .runners.val_runner import run_val_stage

    try:
        run_val_stage(stage, argv)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] val stage={stage} failed: {exc}")
        raise


def run_predict(argv: list) -> None:
    _, overrides = parse_argv(argv)
    source = str(overrides.get("source", ""))
    weights = str(overrides.get("weights", ""))
    if not source:
        print("Error: source= is required.")
        sys.exit(1)
    if not weights:
        print("Error: weights= is required.")
        sys.exit(1)

    from .config.config import HTRTDETRConfig
    from .inference import Inferencer
    from .models import build_model
    from .utils.misc import load_model_weights

    cfg = HTRTDETRConfig()
    cfg.inference.input_path = source
    cfg.inference.checkpoint = weights
    output_dir = str(overrides.get("output_dir", "runs/predict"))
    cfg.inference.output_path = output_dir
    if "score_threshold" in overrides:
        cfg.model.detector.score_threshold = float(overrides["score_threshold"])

    model = build_model(cfg.model)
    model.set_stage(4)
    load_model_weights(weights, model, strict=False)
    img_size = (
        (cfg.data.image_size, cfg.data.image_size)
        if isinstance(cfg.data.image_size, int)
        else tuple(cfg.data.image_size)
    )
    inferencer = Inferencer(
        model=model,
        cfg=cfg.inference,
        action_names=[str(i) for i in range(cfg.model.action_head.num_actions)],
        class_names=[str(i) for i in range(cfg.model.detector.head.num_classes)],
        window_size=int(overrides.get("window_size", cfg.data.window_size)),
    )
    inferencer.run(
        input_path=source,
        output_dir=output_dir,
        image_size=img_size,
        max_frames=int(overrides["max_frames"]) if "max_frames" in overrides else None,
    )


def run_analyze(argv: list) -> None:
    _, overrides = parse_argv(argv)
    mode = str(overrides.get("mode", ""))
    if not mode:
        print("Error: mode= is required.")
        sys.exit(1)

    output_dir = str(overrides.get("output_dir", "runs/analyze"))

    if mode == "timeline":
        predictions = str(overrides.get("predictions", ""))
        if not predictions:
            print("Error: mode=timeline requires predictions=")
            sys.exit(1)
        from .analysis import run_timeline_analysis

        run_timeline_analysis(predictions, output_dir)
        return

    if mode == "distribution":
        annotation = str(overrides.get("annotation", ""))
        if not annotation:
            print("Error: mode=distribution requires annotation=")
            sys.exit(1)
        from .analysis import run_distribution_analysis

        run_distribution_analysis(annotation, output_dir)
        return

    if mode == "id_switches":
        predictions = str(overrides.get("predictions", ""))
        if not predictions:
            print("Error: mode=id_switches requires predictions=")
            sys.exit(1)
        from .analysis import run_id_switch_analysis

        run_id_switch_analysis(predictions, output_dir)
        return

    if mode == "confidence":
        predictions = str(overrides.get("predictions", ""))
        if not predictions:
            print("Error: mode=confidence requires predictions=")
            sys.exit(1)
        from .analysis import run_confidence_analysis

        run_confidence_analysis(predictions, output_dir)
        return

    print(
        f"Unknown analyze mode: {mode!r}. "
        "Supported: timeline | distribution | id_switches | confidence"
    )
    sys.exit(1)


_HELP = """\
yoake - YOAKE: Hierarchical Temporal RT-DETR for Animal Behavior Analysis

Usage:
  yoake <command> [options]

Commands:
  train    stage=<1-4> [config=path.yaml] [key=value ...]
  val      stage=<1-4> [checkpoint=path.pth] [key=value ...]
  predict  source=<video/dir> weights=<path.pth> [key=value ...]
  analyze  mode=<timeline|distribution|id_switches|confidence> [key=value ...]

Examples:
  yoake train stage=1
  yoake train stage=1 config=configs/stage1_small.yaml
  yoake val stage=2 checkpoint=runs/train/stage2/weights/best.pth
  yoake predict source=data/videos/video.mp4 weights=runs/train/stage4/weights/best.pth
  yoake analyze mode=timeline predictions=runs/predict/predictions.json

Primary metrics:
  Stage 1: AP50
  Stage 2: macro_F1
  Stage 3: IDF1
  Stage 4: composite = 0.4*AP50 + 0.3*macro_F1 + 0.3*IDF1

Outputs:
  runs/train/stage{N}/weights/best.pth
  runs/train/stage{N}/weights/last.pth
  runs/train/stage{N}/results.csv
  runs/train/stage{N}/metrics_latest.json
"""


def _print_help() -> None:
    try:
        print(_HELP)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(_HELP.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = argv[0]
    rest = argv[1:]

    if command == "train":
        stage_str, rest = _extract_kv(rest, "stage", default="1")
        run_train(int(stage_str), rest)
        return

    if command == "val":
        stage_str, rest = _extract_kv(rest, "stage", default="1")
        run_val(int(stage_str), rest)
        return

    if command == "predict":
        run_predict(rest)
        return

    if command == "analyze":
        run_analyze(rest)
        return

    print(f"yoake: unknown command '{command}'\n")
    _print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
