"""
Single-entry CLI for YOAKE.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


def _load_checkpoint_payload(path: str) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        return payload
    return {"model_state": payload}


def load_runtime_config_from_checkpoint(path: str):
    from .config.config import HTRTDETRConfig

    payload = _load_checkpoint_payload(path)
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        return None

    cfg_dict = extra.get("config")
    if isinstance(cfg_dict, dict):
        return HTRTDETRConfig.from_dict(cfg_dict)

    model_cfg = extra.get("model_cfg")
    if isinstance(model_cfg, dict):
        return HTRTDETRConfig.from_dict({"model": model_cfg})

    return None


def _yolo_dataset_to_videos(anno) -> list:
    """Adapt v1.1 ``DatasetAnno`` into the ``VideoAnnotation`` shape expected
    by ``SingleFrameDataset`` / ``SlidingWindowDataset``.
    """
    from .data.dataset import ObjectAnnotation, FrameAnnotation, VideoAnnotation

    videos = []
    for v in anno.videos:
        frames = []
        for fr in v.frames:
            objects = [
                ObjectAnnotation(
                    bbox=list(obj.bbox),
                    class_id=obj.class_id,
                    track_id=obj.track_id,
                    action_id=obj.action_id,
                )
                for obj in fr.objects
            ]
            frames.append(FrameAnnotation(
                frame_index=fr.frame_index,
                image_path=fr.image_path,
                objects=objects,
                width=fr.width,
                height=fr.height,
            ))
        videos.append(VideoAnnotation(
            video_id=v.video_id,
            fps=v.fps,
            frames=frames,
        ))
    return videos


def _load_dataset_videos(path_str: str, cfg) -> list:
    """Load a training/val dataset by format.

    Selects between YOLO and JSON based on ``cfg.data.annotation_format``.
    When ``format=yolo``, ``path_str`` should point to a YOLO dataset root
    (``images/`` + ``labels/`` or a flat directory with a ``classes.txt``).
    """
    fmt = str(getattr(cfg.data, "annotation_format", "json")).lower()
    if fmt == "yolo":
        from .data import load_yolo_annotation

        classes_file = str(getattr(cfg.data, "classes_file", "") or "")
        anno = load_yolo_annotation(path_str, classes_file=classes_file)
        return _yolo_dataset_to_videos(anno)

    from .data import load_annotations

    ann_path = resolve_annotation_path(path_str)
    videos, _, _ = load_annotations(str(ann_path))
    return videos


def _dataset_root_for(path_str: str, cfg) -> str:
    """Resolve the image root used to locate frame images on disk."""
    fmt = str(getattr(cfg.data, "annotation_format", "json")).lower()
    if fmt == "yolo":
        return str(Path(path_str))
    return resolve_dataset_root(path_str)


def _resolve_label_mode(cfg, train_ann: Path, stage: int, overrides: Dict[str, Any], logger) -> str:
    """Decide keyframe vs dense training, and write the decision into ``cfg``.

    The decision is written back (``data.label_mode``, the resolved clip, and
    ``window_size``) so it travels inside the checkpoint's config: inference
    reads it to feed the temporal model history at the spacing it trained on.
    """
    from .annotation import keyframes as kf
    from .data.annotation import load_annotation

    try:
        anno = load_annotation(str(train_ann))
    except Exception as exc:  # noqa: BLE001 - the dataset build reports it properly
        logger.warning("Could not read %s to decide the label mode: %s", train_ann, exc)
        anno = None
    mode = kf.resolve_label_mode(str(getattr(cfg.data, "label_mode", "auto")), anno)
    cfg.data.label_mode = mode
    if mode != kf.KEYFRAME:
        return mode

    if stage == 3:
        logger.error(
            "This dataset is keyframe-labelled (meta.labeling.mode = keyframe). Stage 3 "
            "trains an identity classifier from track ids, and keyframe labels have none "
            "on purpose: identity comes from the tracker at analysis time. Train stages "
            "1, 2 and 4 instead. See docs/KEYFRAME_BEHAVIOR.md."
        )
        sys.exit(2)

    clip = kf.clip_spec_of(anno) if anno is not None else kf.ClipSpec()
    length = int(getattr(cfg.data, "clip_length", 0) or clip.length)
    stride = int(getattr(cfg.data, "clip_stride", 0) or clip.stride)
    clip = kf.ClipSpec(length=length, stride=stride)
    cfg.data.clip_length, cfg.data.clip_stride = clip.length, clip.stride
    cfg.data.window_size = clip.length
    behavior = getattr(cfg.model, "behavior", None)
    if behavior is not None:
        behavior.temporal_window = clip.length
        if getattr(behavior, "use_clip", None) is None:
            behavior.use_clip = True
    names_from_data = list(anno.action_names) if anno is not None else []
    if anno is not None:
        names_from_data = _resolve_behavior_heads(cfg, anno, overrides, logger)
    if anno is not None and names_from_data:
        # The behaviors are the dataset's, not the config default's five. A
        # mismatch is the num_classes trap again (annotation/round.py): a
        # behavior numbered past the head's width cannot be learned at all.
        names = [str(n) for n in names_from_data]
        explicit = ((overrides.get("model") or {}).get("action_head") or {})
        if "num_actions" not in explicit and cfg.model.action_head.num_actions != len(names):
            logger.info(
                "behaviors: %d (%s) - model.action_head.num_actions set to match",
                len(names), ", ".join(names),
            )
            cfg.model.action_head.num_actions = len(names)
        if len(names) == cfg.model.action_head.num_actions:
            cfg.eval.action_names = names
    logger.info(
        "label mode: keyframe - clip %s; track ids are not used (identity comes from tracking)",
        kf.describe_clip(clip),
    )
    if anno is not None:
        flat = [str(n) for n in (names_from_data or anno.action_names)]
        logger.info("  %s", kf.summarize(anno, flat).sentence(flat))
        for warning in kf.check(anno, flat, clip):
            logger.warning("  %s", warning)
    return mode


def _resolve_behavior_heads(cfg, anno, overrides: Dict[str, Any], logger) -> list:
    """One behavior per animal, or one per group: the dataset decides (§15).

    A file that declares ``meta.labeling.behaviors = "multi_head"`` trains the
    behavior branch as ``classifier_type = multi_head`` with the file's groups;
    any other file trains a single softmax. The choice is written into ``cfg``
    so it travels inside the checkpoint and ``predict`` builds the same heads.
    A config that names a classifier the labels contradict stops the run: a
    multi-head model on single labels (or the reverse) would train nothing.
    Returns the flat behavior names.
    """
    from .annotation import behaviors

    try:
        vocab = behaviors.vocabulary_of(anno)
    except ValueError as exc:
        logger.error("%s See docs/KEYFRAME_BEHAVIOR.md §15.", exc)
        sys.exit(2)
    behavior = getattr(cfg.model, "behavior", None)
    explicit = ((overrides.get("model") or {}).get("behavior") or {}).get("classifier_type")
    wanted = "multi_head" if vocab.multi else "single"
    if explicit and explicit != wanted and explicit != "hierarchical":
        logger.error(
            "model.behavior.classifier_type=%s, but the labels have %s. Remove the "
            "override, or relabel. See docs/KEYFRAME_BEHAVIOR.md §15.",
            explicit,
            "one behavior per group (multi-head)" if vocab.multi else "one behavior per animal",
        )
        sys.exit(2)
    if behavior is not None and not explicit:
        if vocab.multi:
            behavior.classifier_type = "multi_head"
            behavior.heads = vocab.as_groups()
        elif behavior.classifier_type == "multi_head":
            behavior.classifier_type = "single"
            behavior.heads = {}
    if vocab.multi:
        logger.info(
            "behaviors: multi-head - %s",
            "; ".join(f"{h}: {', '.join(b)}" for h, b in vocab.groups),
        )
    return list(vocab.names)


def _build_keyframe_dataloaders(cfg, stage: int, img_size, loader_kwargs):
    """Loaders over :class:`KeyframeClipDataset` - one sample per keyframe.

    Stage 1 sees the keyframe alone; stages 2 and 4 see the clip ending at it.
    Stage 2 skips keyframes with no behavior label at all: the detector is
    frozen there, so such a clip would cost a full forward pass for no loss.
    """
    from torch.utils.data import DataLoader

    from .annotation.keyframes import ClipSpec
    from .data.keyframe_clip import (
        KeyframeClipDataset,
        collate_keyframe_clips,
        collate_keyframe_single,
    )

    clip = ClipSpec(length=int(cfg.data.clip_length), stride=int(cfg.data.clip_stride))
    single = stage == 1
    train_path = resolve_annotation_path(str(cfg.data.train_root))
    val_path = resolve_annotation_path(str(cfg.data.val_root))
    if not val_path.exists():
        val_path = train_path
    common = dict(
        clip=clip,
        image_size=img_size,
        frame_source=str(getattr(cfg.data, "frame_source", "auto") or "auto"),
        single_frame=single,
    )
    train_ds = KeyframeClipDataset(
        str(train_path),
        augment=cfg.data.augment_train,
        hflip=cfg.data.aug_hflip,
        vflip=cfg.data.aug_vflip,
        brightness=cfg.data.aug_brightness,
        contrast=cfg.data.aug_contrast,
        saturation=cfg.data.aug_saturation,
        require_behavior=(stage == 2),
        **common,
    )
    val_ds = KeyframeClipDataset(
        str(val_path), augment=False, require_behavior=(stage == 2), **common
    )
    collate_fn = collate_keyframe_single if single else collate_keyframe_clips
    num_workers = cfg.data.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=len(train_ds) > cfg.data.batch_size,
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


def _build_dataloaders(cfg, stage: int, use_dummy: bool):
    from torch.utils.data import DataLoader

    from .data import (
        DummyDataset,
        SingleFrameDataset,
        SlidingWindowDataset,
        get_collate_fn,
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

    if not use_dummy and str(getattr(cfg.data, "label_mode", "")) == "keyframe":
        return _build_keyframe_dataloaders(cfg, stage, img_size, loader_kwargs)

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
            train_root_str = str(cfg.data.train_root)
            val_root_str = str(cfg.data.val_root)
            train_videos = _load_dataset_videos(train_root_str, cfg)
            try:
                val_videos = _load_dataset_videos(val_root_str, cfg)
            except FileNotFoundError:
                val_videos = train_videos
            train_ds = SingleFrameDataset(
                train_videos,
                image_size=img_size,
                augment=cfg.data.augment_train,
                data_root=_dataset_root_for(train_root_str, cfg),
            )
            val_ds = SingleFrameDataset(
                val_videos,
                image_size=img_size,
                augment=False,
                data_root=_dataset_root_for(val_root_str, cfg),
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
            train_root_str = str(cfg.data.train_root)
            val_root_str = str(cfg.data.val_root)
            train_videos = _load_dataset_videos(train_root_str, cfg)
            val_videos = _load_dataset_videos(val_root_str, cfg)
            train_ds = SlidingWindowDataset(
                train_videos,
                window_size=cfg.data.window_size,
                stride=cfg.data.window_stride,
                image_size=img_size,
                augment=cfg.data.augment_train,
                data_root=_dataset_root_for(train_root_str, cfg),
                require_action=(stage == 2),
            )
            val_ds = SlidingWindowDataset(
                val_videos,
                window_size=cfg.data.window_size,
                stride=cfg.data.window_size,
                image_size=img_size,
                augment=False,
                data_root=_dataset_root_for(val_root_str, cfg),
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
    # Weights-only warm start, popped before the config is built for the same
    # reason ``root`` is: it is an instruction to this function, not a config
    # field, and the config merge drops unknown keys in silence.
    #
    # This is what the annotator's active-learning rounds use to carry on from
    # the previous round instead of re-learning the species from ImageNet every
    # time. ``train.resume`` cannot serve: it restores the optimizer, the
    # scheduler (so the learning rate resumes at its decayed floor) and the
    # previous run's ``best_metric``, and it sets ``start_epoch`` past
    # ``max_epochs`` so a finished run resumes into zero epochs.
    init_weights = str(overrides.pop("init_weights", "") or "")
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
    fmt = str(getattr(cfg.data, "annotation_format", "json")).lower()
    if fmt == "yolo":
        # YOLO 形式は train_root がデータセットディレクトリそのもの
        use_dummy = not Path(str(cfg.data.train_root)).exists()
    else:
        train_ann = resolve_annotation_path(str(cfg.data.train_root)) if cfg.data.train_root else Path(".")
        use_dummy = not train_ann.exists()
    if use_dummy:
        logger.warning("Train data not found at %s.", cfg.data.train_root)
        if not _ask_confirm("Run a DummyDataset smoke test instead?"):
            logger.error("Aborted. Set data.train_root to a real dataset before training.")
            sys.exit(1)
    elif fmt == "json":
        # Keyframe or dense? Decided from the annotation file (data.label_mode=auto)
        # before the loaders and the model are built, because both depend on it.
        _resolve_label_mode(cfg, train_ann, stage, overrides, logger)
        if stage > 1:
            clamp_temporal_branches(cfg)

    train_loader, val_loader = _build_dataloaders(cfg, stage, use_dummy)
    if hasattr(train_loader.dataset, "report"):
        logger.info("train: %s", train_loader.dataset.report())
    model = build_model(cfg.model)
    model.set_stage(stage)
    logger.info("Model parameters: %s", model.num_parameters())

    if init_weights and load_model_weights is not None:
        if not Path(init_weights).is_file():
            logger.error("init_weights not found: %s", init_weights)
            sys.exit(1)
        logger.info("Initialising from %s (weights only)", init_weights)
        load_model_weights(init_weights, model, strict=False)
    elif not cfg.train.resume and stage > 1 and load_model_weights is not None:
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
        full_cfg=cfg,
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
    config_path, overrides = parse_argv(argv)
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

    restored_from_checkpoint = None
    if config_path:
        cfg = HTRTDETRConfig.from_yaml(config_path)
    else:
        restored_from_checkpoint = load_runtime_config_from_checkpoint(weights)
        cfg = restored_from_checkpoint if restored_from_checkpoint is not None else HTRTDETRConfig()
    if overrides:
        cfg = cfg.merge(overrides)

    cfg.inference.input_path = source
    cfg.inference.checkpoint = weights
    output_dir = str(overrides.get("output_dir", cfg.inference.output_path))
    cfg.inference.output_path = output_dir
    if "score_threshold" in overrides:
        cfg.model.detector.score_threshold = float(overrides["score_threshold"])
    cfg.model.detector.backbone.pretrained = False
    # シルエット幾何とリアルタイムは後処理と実行方式の切り替えで、モデルは変えない。
    # ``geometry=true`` / ``realtime=true`` / ``geometry_params.<name>=v`` は
    # ``inference.*`` の短縮形 (merge は未知の最上位キーを捨てるのでここで写す)。
    if "geometry" in overrides:
        cfg.inference.geometry = bool(overrides["geometry"])
    if "realtime" in overrides:
        cfg.inference.realtime = bool(overrides["realtime"])
    if isinstance(overrides.get("geometry_params"), dict):
        cfg.inference.geometry_params = {
            **(cfg.inference.geometry_params or {}), **overrides["geometry_params"]
        }
    _check_geometry_params(cfg.inference.geometry_params)

    from .inference.realtime import is_live_source

    if is_live_source(source) and not cfg.inference.realtime:
        print(
            f"Error: source={source} is a camera or stream, which only the realtime "
            "pipeline can read. Add realtime=true."
        )
        sys.exit(1)

    model = build_model(cfg.model)
    model.set_stage(4)
    strict_load = bool(config_path or restored_from_checkpoint is not None)
    try:
        missing, unexpected = load_model_weights(weights, model, strict=strict_load)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match the active model configuration. "
            "Pass config=... or use a checkpoint saved with config metadata."
        ) from exc
    if not strict_load and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint metadata is missing, and the default predict configuration only "
            "partially matched the checkpoint. Pass config=... to restore the training setup."
        )

    img_size = (
        (cfg.data.image_size, cfg.data.image_size)
        if isinstance(cfg.data.image_size, int)
        else tuple(cfg.data.image_size)
    )
    action_names = cfg.eval.action_names
    if len(action_names) != cfg.model.action_head.num_actions:
        action_names = [str(i) for i in range(cfg.model.action_head.num_actions)]
    class_names = [str(i) for i in range(cfg.model.detector.head.num_classes)]
    window_size = int(overrides.get("window_size", cfg.data.window_size))
    # A keyframe-trained model saw clips of ``clip_length`` frames ``clip_stride``
    # apart; its checkpoint config says so (written back at training time), and
    # streaming must read track history at that same spacing.
    clip_stride = 1
    if str(getattr(cfg.data, "label_mode", "")) == "keyframe":
        window_size = int(overrides.get("window_size", cfg.data.clip_length or window_size))
        clip_stride = int(overrides.get("clip_stride", cfg.data.clip_stride or 1))
    max_frames = int(overrides["max_frames"]) if "max_frames" in overrides else None

    # detect_only: 箱だけが欲しい呼び出し (アノテーションの候補生成) 用。
    # tracker も時間モデルも行動ヘッドも通さず、注釈動画も書かず、複数フレームを
    # まとめて detector に投げる。検出そのものは streaming と同一 (detector は
    # フレーム単位なので、16 枚を 1 バッチで通しても 1 枚ずつ通しても同じ箱が出る)。
    # ``parse_argv`` already turns "true"/"false" into a real bool, and
    # ``window_size`` / ``max_frames`` above read their overrides the same way.
    if bool(overrides.get("detect_only", False)):
        from .inference import DEFAULT_BATCH, run_detect_only

        run_detect_only(
            model,
            source,
            output_dir,
            image_size=img_size,
            score_threshold=cfg.model.detector.score_threshold,
            batch_size=int(overrides.get("batch_size", DEFAULT_BATCH)),
            max_frames=max_frames,
            use_amp=cfg.inference.use_amp,
            use_channels_last=cfg.inference.use_channels_last,
        )
        return

    if cfg.inference.realtime:
        # streaming と同じ推論を、capture / geometry / writer を別プロセスにして回す。
        if cfg.inference.mode != "streaming":
            print("[INFO] realtime=true runs the streaming inferencer; "
                  f"inference.mode={cfg.inference.mode} is ignored")
        _run_predict_realtime(
            cfg, model, source, output_dir, img_size,
            action_names, class_names, window_size, max_frames,
            clip_stride=clip_stride,
        )
        return

    if cfg.inference.mode == "streaming":
        # 各フレームを detector に 1 回だけ通す (既定)。窓長に依存しない。
        _run_predict_streaming(
            cfg, model, source, output_dir, img_size,
            action_names, class_names, window_size, max_frames,
            clip_stride=clip_stride,
        )
        return

    # mode="batch": 毎フレーム窓全体を通す従来実装 (数値比較・回帰検証用)
    inferencer = Inferencer(
        model=model,
        cfg=cfg.inference,
        action_names=action_names,
        class_names=class_names,
        window_size=window_size,
    )
    inferencer.run(
        input_path=source,
        output_dir=output_dir,
        image_size=img_size,
        max_frames=max_frames,
    )


def _run_predict_streaming(
    cfg, model, source: str, output_dir: str, img_size,
    action_names: list, class_names: list, window_size: int,
    max_frames: Optional[int],
    clip_stride: int = 1,
) -> None:
    """streaming 推論で動画/画像ディレクトリを処理し、注釈動画と JSON を書く。"""
    import json

    from .inference import StreamingInferencer, VideoWriter, draw_detections
    from .inference.inferencer import read_image_dir_frames, read_video_frames

    src = Path(source)
    if src.is_dir():
        frame_gen = read_image_dir_frames(str(src))
    elif src.exists():
        frame_gen = read_video_frames(str(src), max_frames=max_frames)
    else:
        raise FileNotFoundError(f"Input not found: {src}")

    si = StreamingInferencer(
        model=model,
        cfg=cfg.inference,
        action_names=action_names,
        class_names=class_names,
        window_size=window_size,
        image_size=img_size,
        clip_stride=clip_stride,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer: Optional[VideoWriter] = None
    results: Dict[str, Any] = {
        "_meta": _predict_meta(cfg, str(src), action_names, class_names, window_size, clip_stride)
    }
    draw_kw = _draw_options(cfg)

    # シルエット幾何 (docs/SILHOUETTE_GEOMETRY.md): 箱と元解像度のフレームから測る後処理。
    analyzer = None
    table = None
    if cfg.inference.geometry:
        from .analysis.silhouette import (
            GeometryCSV, SilhouetteAnalyzer, SilhouetteParams, draw_geometry, geometry_meta,
        )

        params = SilhouetteParams(**(cfg.inference.geometry_params or {}))
        analyzer = SilhouetteAnalyzer(params)
        table = GeometryCSV(str(out_dir / "geometry.csv"))
        results["_meta"]["geometry"] = geometry_meta(params)

    try:
        for frame_idx, frame_rgb, fps in frame_gen:
            if results["_meta"]["fps"] is None and fps:
                results["_meta"]["fps"] = float(fps)
            res = si.process_frame(frame_rgb, frame_index=frame_idx)
            entry = res.to_json()
            rows = None
            if analyzer is not None and table is not None:
                if analyzer.fps is None and fps:
                    analyzer.fps = float(fps)
                rows = analyzer.update(
                    frame_rgb, frame_idx, res.boxes_xyxy,
                    corners=res.corners, track_ids=res.track_ids,
                )
                entry["geometry"] = rows
                table.write(frame_idx, res.track_ids, rows)
            results[str(frame_idx)] = entry

            if writer is None:
                h, w = frame_rgb.shape[:2]
                writer = VideoWriter(str(out_dir / "output.mp4"), fps, (w, h))
            vis = draw_detections(
                frame_rgb, res.boxes_xyxy, res.scores, res.class_ids,
                res.track_ids, res.action_ids, action_names, class_names, **draw_kw,
            )
            if rows:
                draw_geometry(vis, rows)
            writer.write(vis)
    finally:
        if table is not None:
            table.close()

    if writer is not None:
        writer.release()

    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    stats = si.runtime_stats()
    with open(out_dir / "runtime_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Streaming inference complete. frames={stats.get('frames', 0)} "
        f"FPS={stats.get('fps', 0):.1f} "
        f"(detector {stats.get('detector_ms', 0):.1f}ms / "
        f"tracking {stats.get('tracking_ms', 0):.1f}ms / "
        f"temporal {stats.get('temporal_ms', 0):.1f}ms / "
        f"behavior {stats.get('behavior_ms', 0):.1f}ms per frame)"
    )
    if analyzer is not None:
        print(f"Silhouette geometry: {analyzer.timing()['ms_per_frame']:.2f}ms per frame "
              f"-> {out_dir / 'geometry.csv'}")
    print(f"Results: {out_dir / 'predictions.json'}")


def _run_predict_realtime(
    cfg, model, source: str, output_dir: str, img_size,
    action_names: list, class_names: list, window_size: int,
    max_frames: Optional[int],
    clip_stride: int = 1,
) -> None:
    """streaming 推論を multiprocessing のパイプラインで回す (docs/REALTIME.md)。"""
    from .inference import StreamingInferencer
    from .inference.realtime import RealtimeOptions, is_live_source, run_realtime

    if not is_live_source(source) and not Path(source).exists():
        raise FileNotFoundError(f"Input not found: {source}")
    inf = cfg.inference
    options = RealtimeOptions(
        slots=int(inf.realtime_slots),
        drop_frames=str(inf.realtime_drop_frames),
        display=bool(inf.realtime_display),
        write_video=bool(inf.realtime_write_video),
        max_frames=max_frames,
        max_seconds=float(inf.realtime_max_seconds),
        geometry=bool(inf.geometry),
        geometry_params=dict(inf.geometry_params or {}),
        draw=_draw_options(cfg),
    )
    si = StreamingInferencer(
        model=model,
        cfg=inf,
        action_names=action_names,
        class_names=class_names,
        window_size=window_size,
        image_size=img_size,
        clip_stride=clip_stride,
    )
    print(f"Realtime pipeline: source={source} "
          f"slots={options.slot_count(is_live_source(source))} "
          f"geometry={'on' if options.geometry else 'off'} (Ctrl+C to stop)")
    stats = run_realtime(
        si, source, output_dir,
        image_size=img_size,
        action_names=action_names,
        class_names=class_names,
        meta=_predict_meta(cfg, str(source), action_names, class_names, window_size, clip_stride),
        options=options,
    )
    latency = stats.get("latency_ms", {})
    print(
        f"Realtime inference complete. frames={stats['frames_processed']} "
        f"dropped={stats['frames_dropped']} FPS={stats['fps']:.1f} "
        f"latency p50={latency.get('p50', 0):.0f}ms p95={latency.get('p95', 0):.0f}ms"
        + (" (interrupted)" if stats.get("interrupted") else "")
    )
    for warning in stats.get("warnings", []):
        print(f"[WARN] {warning}")
    print(f"Results: {Path(output_dir) / 'predictions.json'}")


def _predict_meta(
    cfg, source: str, action_names: list, class_names: list, window_size: int,
    clip_stride: int,
) -> Dict[str, Any]:
    """predictions.json の ``_meta`` (逐次・リアルタイム共通)。

    ``_meta`` is not a frame: every reader skips keys that are not integers.
    It is what the bout analysis needs to turn frames into seconds and ids
    into names without being told again.
    """
    meta: Dict[str, Any] = {
        "source": str(source),
        "action_names": list(action_names),
        "class_names": list(class_names),
        "label_mode": str(getattr(cfg.data, "label_mode", "") or "dense"),
        "window": int(window_size),
        "clip_stride": int(clip_stride),
        "fps": None,
    }
    behavior_cfg = getattr(cfg.model, "behavior", None)
    if getattr(behavior_cfg, "classifier_type", "") == "multi_head" and behavior_cfg.heads:
        # Groups decode separately (analysis/bouts.py): each one's slice of
        # ``action_probs`` sums to 1. docs/KEYFRAME_BEHAVIOR.md §15.5.
        meta["action_groups"] = {
            str(h): [str(n) for n in names] for h, names in behavior_cfg.heads.items()
        }
    return meta


def _draw_options(cfg) -> Dict[str, Any]:
    """``draw_detections`` のキーワード引数 (逐次・リアルタイム共通)。"""
    inf = cfg.inference
    return {
        "show_bbox": inf.show_bbox,
        "show_id": inf.show_id,
        "show_action": inf.show_action,
        "show_confidence": inf.show_confidence,
        "thickness": inf.bbox_thickness,
        "font_scale": inf.font_scale,
    }


def _check_geometry_params(params: Optional[Dict[str, Any]]) -> None:
    """``geometry_params.<name>`` の名前と値を、モデルを読む前に確かめる。"""
    if not params:
        return
    from dataclasses import fields

    from .analysis.silhouette import SilhouetteParams

    known = {f.name for f in fields(SilhouetteParams)}
    unknown = sorted(set(params) - known)
    if unknown:
        print(f"Error: unknown geometry_params {unknown}. Known: {sorted(known)}")
        sys.exit(1)
    try:
        SilhouetteParams(**params)
    except (TypeError, ValueError) as exc:
        print(f"Error: geometry_params: {exc}")
        sys.exit(1)


def run_pipeline(argv: list) -> None:
    """レイヤードパイプラインを config に従って動画/画像ディレクトリに適用する。"""
    import json

    config_path, overrides = parse_argv(argv)
    if not config_path:
        print("Error: config=<pipeline.yaml> is required.")
        sys.exit(1)
    source = str(overrides.get("source", ""))
    if not source:
        print("Error: source=<video/dir> is required.")
        sys.exit(1)
    output_dir = str(overrides.get("output_dir", "runs/pipeline"))
    max_frames = int(overrides["max_frames"]) if "max_frames" in overrides else None

    from .pipeline.builder import build_from_yaml
    from .pipeline.io import (
        frames_from_source, interactions_to_json, merge_windows, window_to_predictions,
    )

    runner = build_from_yaml(config_path)
    print(runner.describe())

    frames = list(frames_from_source(source, max_frames))
    try:
        windows = list(runner.run(frames))
    finally:
        runner.teardown()
    merged = merge_windows(windows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = window_to_predictions(merged)
    with open(out / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    inter = interactions_to_json(merged)
    with open(out / "interactions.json", "w", encoding="utf-8") as f:
        json.dump(inter, f, indent=2)
    print(
        f"Pipeline complete. Frames: {len(preds)} | Interactions: {len(inter)} | "
        f"Output: {out / 'predictions.json'}"
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

    if mode == "bouts":
        # Keyframe scheme: per-frame behavior probabilities + tracks -> bouts.
        predictions = str(overrides.get("predictions", ""))
        if not predictions:
            print("Error: mode=bouts requires predictions=")
            sys.exit(1)
        from .analysis.bouts import (
            DEFAULT_MAX_GAP,
            DEFAULT_MIN_BOUT_S,
            DEFAULT_STAY,
            analyze_predictions,
        )

        names = overrides.get("action_names")
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        summary = analyze_predictions(
            predictions,
            output_dir,
            fps=float(overrides["fps"]) if "fps" in overrides else None,
            stay=float(overrides.get("stay", DEFAULT_STAY)),
            min_bout_s=float(overrides.get("min_bout_s", DEFAULT_MIN_BOUT_S)),
            max_gap=int(overrides.get("max_gap", DEFAULT_MAX_GAP)),
            bin_s=float(overrides.get("bin_s", 60.0)),
            action_names=names or None,
        )
        overall = summary["overall"]
        budget = ", ".join(f"{k} {v:.0%}" for k, v in overall["time_budget"].items() if v)
        print(
            f"Bouts: {sum(overall['bouts'].values())} across {overall['tracks']} track(s), "
            f"{overall['tracked_seconds']:.1f} s tracked. Time budget: {budget or '-'}"
        )
        for name, path in summary["files"].items():
            print(f"  {name}: {path}")
        return

    print(
        f"Unknown analyze mode: {mode!r}. "
        "Supported: timeline | distribution | id_switches | confidence | bouts"
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
  analyze  mode=<timeline|distribution|id_switches|confidence|bouts> [key=value ...]
  pipeline config=<pipeline.yaml> source=<video/dir> [output_dir=path] [max_frames=N]
  project  <new|info|collect|relink> [key=value ...]   プロジェクトフォルダ操作
  gui      統一 GUI (Labeling / Training / Video Analysis) を起動する

Examples:
  yoake train stage=1
  yoake train stage=1 config=configs/stage1_small.yaml
  yoake val stage=2 checkpoint=runs/train/stage2/weights/best.pth
  yoake predict source=data/videos/video.mp4 weights=runs/train/stage4/weights/best.pth
  yoake predict source=video.mp4 weights=best.pth geometry=true      体軸・頭尾・形状を追加で測る
  yoake predict source=0 weights=best.pth realtime=true geometry=true inference.realtime_display=true
  yoake analyze mode=timeline predictions=runs/predict/predictions.json
  yoake analyze mode=bouts predictions=runs/predict/predictions.json output_dir=runs/bouts
  yoake pipeline config=configs/pipeline/full_offline.yaml source=data/videos/video.mp4
  yoake project new dir="D:/studies" name="courtship 2026" videos=D:/captures
  yoake project info path="D:/studies/courtship 2026"

predict options (docs/SILHOUETTE_GEOMETRY.md, docs/REALTIME.md):
  geometry=true          検出ごとのシルエット幾何 (学習不要の後処理) → geometry.csv
  realtime=true          capture / GPU / geometry / writer を別プロセスで並走。
                         source=<カメラ番号 | rtsp://...> もこのモードで読める
  inference.realtime_slots=0(auto) / inference.realtime_drop_frames=auto|never|when_full
  inference.realtime_display=true / inference.realtime_max_seconds=600

Layered pipeline (configs/pipeline/*.yaml):
  detect_only | temporal_detect | detect_track | full_offline | yolo_bytetrack
  各段は enabled トグルと backend 差し替えで解析モードを切り替える。

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


#: これらのコマンドは PyTorch を必要とする。base install (``pip install -e .``)
#: は torch-free なので、無い環境では import が
#: ``ModuleNotFoundError: No module named 'torch'`` で落ちる。それは原因も直し方も
#: 伝えないので、ここで先に見て、何を入れれば良いかを言う。
_TORCH_COMMANDS = {"train", "val", "predict", "pipeline"}


def _require_torch(command: str) -> None:
    """torch が必要なコマンドで、無ければ直し方を示して終了する。"""
    from importlib.util import find_spec

    if find_spec("torch") is not None:
        return
    print(
        f"yoake {command}: PyTorch がこの Python にインストールされていません。\n"
        "\n"
        "  YOAKE の base install は torch-free です (ラベリング用マシンに CUDA\n"
        "  ホイールを入れずに済むように、GUI とアノテーション I/O だけで動きます)。\n"
        "  学習・推論には ml extra を入れてください:\n"
        "\n"
        "      pip install -e \".[ml]\"      # torch + torchvision + scipy\n"
        "      pip install -e \".[all]\"     # GUI も含めて全部\n"
        "\n"
        "  CUDA 版を明示したい場合は先に PyTorch を入れてください:\n"
        "      pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu118\n"
        "\n"
        "  torch を必要としないコマンド: yoake gui / yoake project / yoake analyze",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = argv[0]
    rest = argv[1:]

    if command in _TORCH_COMMANDS:
        _require_torch(command)

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

    if command == "pipeline":
        run_pipeline(rest)
        return

    if command == "project":
        from .project_cli import run_project

        run_project(rest)
        return

    if command == "gui":
        from .gui import main as gui_main

        gui_main(rest)
        return

    print(f"yoake: unknown command '{command}'\n")
    _print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
