"""
config.py — YOAKE 全モジュール共通 dataclass 設定群

設計方針:
- argparse を一切使わない
- すべての設定は dataclass で型付け
- yaml ファイルとの相互変換をサポート
- nested config を flatten せず、責務ごとに分割
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# デフォルト出力ルート。実行ディレクトリ直下の runs/ に出力する。
# スクリプト側で root= や train.output_dir= を渡せばここは使われない。
_YOAKE_TRYAL = "runs"

# yaml は optional 依存 (pyyaml)
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Backbone / Detector
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """ResNet 系 backbone の設定"""
    name: str = "resnet18"          # "resnet18" | "resnet34" | "resnet50"
    pretrained: bool = True          # ImageNet pretrained
    freeze_bn: bool = False          # BN を freeze するか
    out_channels: List[int] = field(
        default_factory=lambda: [128, 256, 512]  # C3, C4, C5 channels
    )
    # 出力する feature map の stride (backbone の出力に対応)
    out_strides: List[int] = field(default_factory=lambda: [8, 16, 32])


@dataclass
class FPNConfig:
    """Feature Pyramid Network の設定"""
    in_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    out_channels: int = 256
    num_levels: int = 3


@dataclass
class DetectorHeadConfig:
    """DETR-like decoder head の設定"""
    hidden_dim: int = 256
    num_queries: int = 100           # detection query 数 (最大検出数)
    num_decoder_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_classes: int = 1             # 動物種クラス数 (デフォルトは1種)


@dataclass
class DetectorConfig:
    """Spatial Detector 全体の設定"""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    head: DetectorHeadConfig = field(default_factory=DetectorHeadConfig)
    # HungarianMatcher の cost weights
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    # 推論時の score threshold
    score_threshold: float = 0.3
    # IoU-based NMS threshold (DETR は本来 NMS 不要だが保険として)
    nms_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Hierarchical Temporal Module
# ---------------------------------------------------------------------------

@dataclass
class TemporalBranchConfig:
    """時間方向 branch 単体の設定"""
    kernel_size: int = 3             # 1D conv の kernel size
    dilation: int = 1                # 1D conv の dilation
    num_frames: int = 4              # この branch が使うフレーム数
    use_depthwise: bool = True       # depthwise separable conv を使うか
    use_residual: bool = True        # residual connection
    use_channel_attention: bool = False  # channel attention gate


@dataclass
class HierarchicalTemporalConfig:
    """Hierarchical Temporal Module 全体の設定"""
    feature_dim: int = 256           # 入力 feature 次元 (detector の hidden_dim と一致)
    # 3 branch の設定
    short_branch: TemporalBranchConfig = field(
        default_factory=lambda: TemporalBranchConfig(
            kernel_size=3, dilation=1, num_frames=3,
            use_depthwise=True, use_residual=True
        )
    )
    mid_branch: TemporalBranchConfig = field(
        default_factory=lambda: TemporalBranchConfig(
            kernel_size=3, dilation=2, num_frames=8,
            use_depthwise=True, use_residual=True
        )
    )
    long_branch: TemporalBranchConfig = field(
        default_factory=lambda: TemporalBranchConfig(
            kernel_size=3, dilation=4, num_frames=16,
            use_depthwise=True, use_residual=True
        )
    )
    output_dim: int = 256            # 各 branch 出力次元 (fusion 後の次元)
    fusion_method: str = "concat_proj"  # "concat_proj" | "sum" | "attention"


# ---------------------------------------------------------------------------
# Memory-based ID Head
# ---------------------------------------------------------------------------

@dataclass
class MemoryIDConfig:
    """Memory-based ID Head の設定"""
    feature_dim: int = 256           # 入力 feature 次元
    embedding_dim: int = 128         # ID embedding 次元
    memory_dim: int = 256            # GRU hidden 次元
    max_ids: int = 50                # 同時追跡可能な最大個体数
    new_id_threshold: float = 0.5    # 新規 ID と判断する similarity 閾値
    # memory に含める情報の種類
    use_appearance: bool = True
    use_geometry: bool = True        # bbox geometry history
    use_motion: bool = True          # velocity history
    use_action_summary: bool = False  # action summary (Stage 4 以降)
    # metric learning
    use_metric_loss: bool = False    # triplet / contrastive loss
    metric_loss_margin: float = 0.3
    # memory の寿命 (フレーム数, これを超えると削除)
    memory_ttl: int = 30
    # 種分離IDプール (複数種混在シーン向け)
    use_species_separated_pools: bool = False
    num_species: int = 1             # 種数 (num_classes と合わせる)


# ---------------------------------------------------------------------------
# Action Head
# ---------------------------------------------------------------------------

@dataclass
class ActionHeadConfig:
    """Action classification head の設定"""
    feature_dim: int = 256           # 入力 feature 次元
    temporal_dim: int = 256          # temporal feature 次元
    interaction_dim: int = 64        # interaction feature 次元
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_actions: int = 5             # 行動クラス数 (データセットに合わせる)
    dropout: float = 0.1
    # interaction feature の使用フラグ
    use_interaction: bool = True
    use_nn_distance: bool = True     # nearest neighbor distance
    use_relative_angle: bool = True  # relative angle
    use_relative_velocity: bool = True
    use_overlap: bool = True         # proximity / overlap


# ---------------------------------------------------------------------------
# Interaction Features
# ---------------------------------------------------------------------------

@dataclass
class InteractionConfig:
    """Interaction feature extractor の設定"""
    feature_dim: int = 256
    output_dim: int = 64
    max_neighbors: int = 5           # 考慮する neighbor の最大数
    distance_bins: int = 16          # distance の bin 数
    angle_bins: int = 8


# ---------------------------------------------------------------------------
# Unified Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """YOAKE 統合モデルの設定"""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    temporal: HierarchicalTemporalConfig = field(default_factory=HierarchicalTemporalConfig)
    id_head: MemoryIDConfig = field(default_factory=MemoryIDConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    # 学習段階 (1-4)
    training_stage: int = 1


# ---------------------------------------------------------------------------
# Data / Dataset
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """データセット・ローダーの設定"""
    train_root: str = "data/train"
    val_root: str = "data/val"
    test_root: str = "data/test"
    annotation_format: str = "json"  # "json" | "yolo" | "csv" | "custom"
    # YOLO 形式のときのオプション: classes.txt を明示指定 (未指定なら train_root/classes.txt を自動探索)
    classes_file: str = ""
    image_size: Tuple[int, int] = (640, 640)  # (H, W)
    # Sequence window の設定
    window_size: int = 16            # 1 sample あたりのフレーム数
    window_stride: int = 8           # sliding window のストライド
    # データ拡張
    augment_train: bool = True
    aug_hflip: bool = True
    aug_vflip: bool = False
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.1
    aug_scale_range: Tuple[float, float] = (0.8, 1.2)
    # DataLoader
    batch_size: int = 4              # シーケンス単位のバッチ
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


# ---------------------------------------------------------------------------
# Loss Weights
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """損失関数の重み設定"""
    # Detection losses
    w_class: float = 2.0
    w_bbox_l1: float = 5.0
    w_bbox_giou: float = 2.0
    # Action loss
    w_action: float = 1.0
    # ID loss
    w_id_cls: float = 1.0
    w_id_metric: float = 0.5         # metric loss (use_metric_loss=True 時)
    # Temporal smoothing loss
    w_temporal_smooth: float = 0.1
    # focal loss gamma (0 で cross-entropy に退化)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # Action class imbalance handling
    imbalance_strategy: str = "class_weight"  # none | class_weight | focal
    class_weight_smoothing: float = 1.0       # ラプラス平滑化係数
    class_weight_clip_min: float = 0.1        # 極端な重みを防ぐ下限
    class_weight_clip_max: float = 10.0       # 極端な重みを防ぐ上限


# ---------------------------------------------------------------------------
# Metrics (best model selection & composite score)
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    """stage ごとの best model 判定指標と composite score の重み"""
    stage2_primary: str = "macro_f1"    # macro_f1 | val_loss
    stage3_primary: str = "idf1"        # idf1 | val_loss
    stage4_primary: str = "composite"   # composite | ap50 | val_loss
    # Stage 4 composite score の重み (合計 1.0 推奨)
    composite_ap50: float = 0.4
    composite_macro_f1: float = 0.3
    composite_idf1: float = 0.3


# ---------------------------------------------------------------------------
# Outputs (run directory structure)
# ---------------------------------------------------------------------------

@dataclass
class OutputsConfig:
    """run 出力の設定"""
    save_plots: bool = True
    save_confusion_matrix: bool = True   # Stage 2: confusion matrix PNG
    save_epoch_metrics_json: bool = True  # val/epoch_NNN.json を毎 epoch 保存


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Optimizer の設定"""
    optimizer: str = "adamw"         # "adam" | "adamw" | "sgd"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    momentum: float = 0.9            # SGD のみ使用
    # Backbone の lr は全体より小さくする
    backbone_lr_factor: float = 0.1
    # Gradient clipping
    grad_clip_norm: float = 1.0


@dataclass
class SchedulerConfig:
    """LR Scheduler の設定"""
    scheduler: str = "cosine"        # "cosine" | "step" | "multistep" | "none"
    warmup_epochs: int = 5
    total_epochs: int = 100
    # CosineAnnealingLR
    eta_min: float = 1e-6
    # StepLR
    step_size: int = 30
    gamma: float = 0.1
    # MultiStepLR
    milestones: List[int] = field(default_factory=lambda: [60, 80])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """学習全体の設定"""
    stage: int = 1                   # 1 | 2 | 3 | 4
    output_dir: str = "runs/train/stage1"
    resume: Optional[str] = None     # checkpoint path
    # Epochs
    max_epochs: int = 100
    early_stopping_patience: int = 50
    # AMP (Automatic Mixed Precision)
    use_amp: bool = True
    # EMA (Exponential Moving Average) — val の AP50 安定化に重要
    use_ema: bool = True
    ema_decay: float = 0.9999
    # 再現性
    seed: int = 42
    deterministic: bool = False
    # ログ
    log_interval: int = 10           # step interval
    val_interval: int = 1            # epoch interval
    save_best: bool = True
    save_last: bool = True
    # wandb (optional)
    use_wandb: bool = False
    wandb_project: str = "ht-rtdetr"
    wandb_run_name: str = "run"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """評価の設定"""
    checkpoint: str = "weights/full_model_best.pth"
    output_dir: str = "runs/val"
    split: str = "val"               # "val" | "test"
    # Detection
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.75]
    )
    # Tracking
    min_track_length: int = 5        # 短すぎるトラックを除外
    # Action
    action_names: List[str] = field(
        default_factory=lambda: ["idle", "walk", "groom", "interact", "other"]
    )
    save_predictions: bool = True
    save_visualizations: bool = False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """推論の設定"""
    checkpoint: str = "weights/full_model_best.pth"
    input_path: str = ""             # 動画 or 画像ディレクトリ
    output_path: str = "runs/predict"
    # Video
    fps: Optional[float] = None      # None → 元動画の fps を使用
    # Visualization overlay
    show_bbox: bool = True
    show_id: bool = True
    show_action: bool = True
    show_confidence: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.5
    # Memory reset
    reset_memory_on_scene_change: bool = True
    scene_change_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class HTRTDETRConfig:
    """YOAKE 全体設定 (トップレベル)"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    # ------------------------------------------------------------------ #
    # yaml ↔ dataclass 変換
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HTRTDETRConfig":
        """dict (yaml 読み込み結果) から HTRTDETRConfig を再構築する。
        ネストされた dataclass を再帰的に復元する。
        get_type_hints() を使うことで文字列の型アノテーションも正しく解決する。
        """
        import dataclasses
        import typing

        def _build(dc_cls, data):
            if not dataclasses.is_dataclass(dc_cls) or not isinstance(data, dict):
                return data
            try:
                hints = typing.get_type_hints(dc_cls)
            except Exception:
                hints = {}
            kwargs = {}
            for name in dc_cls.__dataclass_fields__:
                if name not in data:
                    continue
                val = data[name]
                field_type = hints.get(name)
                # ネストされた dataclass を再帰的に復元
                if field_type is not None and dataclasses.is_dataclass(field_type) \
                        and isinstance(val, dict):
                    val = _build(field_type, val)
                kwargs[name] = val
            return dc_cls(**kwargs)

        return _build(cls, d)

    @classmethod
    def from_yaml(cls, path: str) -> "HTRTDETRConfig":
        """yaml ファイルから設定を読み込む"""
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml が必要です: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})

    def save_yaml(self, path: str) -> None:
        """設定を yaml ファイルに保存する"""
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml が必要です: pip install pyyaml")

        def _convert(obj):
            if isinstance(obj, tuple):
                return [_convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_convert(self.to_dict()), f, default_flow_style=False, allow_unicode=True)

    def merge(self, overrides: Dict[str, Any]) -> "HTRTDETRConfig":
        """dict で上書きした新しい config を返す (immutable merge)"""
        base = self.to_dict()
        _deep_update(base, overrides)
        return self.__class__.from_dict(base)


def _deep_update(base: dict, overrides: dict) -> None:
    """base dict を overrides で再帰的に上書きする (in-place)"""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Stage-specific default configs
# ---------------------------------------------------------------------------

def get_stage1_config(overrides: Optional[Dict] = None) -> HTRTDETRConfig:
    """Stage 1: Detector pretraining / finetuning"""
    cfg = HTRTDETRConfig()
    cfg.train.stage = 1
    cfg.train.output_dir = "runs/train/stage1"
    cfg.train.max_epochs = 500
    cfg.train.early_stopping_patience = 200
    cfg.optimizer.lr = 1e-4
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


def get_stage2_config(overrides: Optional[Dict] = None) -> HTRTDETRConfig:
    """Stage 2: Action head pretraining"""
    cfg = HTRTDETRConfig()
    cfg.train.stage = 2
    cfg.train.output_dir = "runs/train/stage2"
    cfg.train.max_epochs = 80
    cfg.optimizer.lr = 5e-5
    # Stage 2 では detector の重みを freeze する
    cfg.optimizer.backbone_lr_factor = 0.0
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


def get_stage3_config(overrides: Optional[Dict] = None) -> HTRTDETRConfig:
    """Stage 3: ID head pretraining"""
    cfg = HTRTDETRConfig()
    cfg.train.stage = 3
    cfg.train.output_dir = "runs/train/stage3"
    cfg.train.max_epochs = 80
    cfg.optimizer.lr = 5e-5
    cfg.model.id_head.use_metric_loss = True
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Config 検証エラー"""
    pass


def validate_config(cfg: "HTRTDETRConfig") -> None:
    """HTRTDETRConfig の内容を検証し、問題があれば ConfigValidationError を raise する。

    呼び出し例:
        from htrtdetr.config.config import validate_config
        cfg = get_stage2_config()
        validate_config(cfg)  # 問題なければ何もしない
    """
    errors: List[str] = []

    # --- Model ---
    m = cfg.model

    if m.detector.head.num_classes < 1:
        errors.append(f"model.detector.head.num_classes must be >= 1, got {m.detector.head.num_classes}")

    if m.detector.head.num_queries < 1:
        errors.append(f"model.detector.head.num_queries must be >= 1, got {m.detector.head.num_queries}")

    if m.detector.head.num_decoder_layers < 1:
        errors.append(f"model.detector.head.num_decoder_layers must be >= 1, got {m.detector.head.num_decoder_layers}")

    if m.detector.head.hidden_dim != m.detector.fpn.out_channels:
        errors.append(
            f"model.detector.head.hidden_dim ({m.detector.head.hidden_dim}) "
            f"must match model.detector.fpn.out_channels ({m.detector.fpn.out_channels})"
        )

    if m.temporal.feature_dim != m.detector.fpn.out_channels:
        errors.append(
            f"model.temporal.feature_dim ({m.temporal.feature_dim}) "
            f"must match model.detector.fpn.out_channels ({m.detector.fpn.out_channels})"
        )

    if m.temporal.output_dim < 1:
        errors.append(f"model.temporal.output_dim must be >= 1, got {m.temporal.output_dim}")

    if m.id_head.embedding_dim < 1:
        errors.append(f"model.id_head.embedding_dim must be >= 1, got {m.id_head.embedding_dim}")

    if m.id_head.max_ids < 2:
        errors.append(f"model.id_head.max_ids must be >= 2, got {m.id_head.max_ids}")

    if m.action_head.num_actions < 2:
        errors.append(f"model.action_head.num_actions must be >= 2, got {m.action_head.num_actions}")

    # --- Data ---
    d = cfg.data
    if d.window_size < 1:
        errors.append(f"data.window_size must be >= 1, got {d.window_size}")
    if d.batch_size < 1:
        errors.append(f"data.batch_size must be >= 1, got {d.batch_size}")
    if d.window_stride < 1:
        errors.append(f"data.window_stride must be >= 1, got {d.window_stride}")
    if len(d.image_size) != 2 or d.image_size[0] < 32 or d.image_size[1] < 32:
        errors.append(f"data.image_size must be (H, W) with H, W >= 32, got {d.image_size}")

    # --- Branch num_frames vs window_size ---
    t = cfg.model.temporal
    for branch_name, branch in [("short", t.short_branch), ("mid", t.mid_branch), ("long", t.long_branch)]:
        if branch.num_frames > d.window_size:
            errors.append(
                f"model.temporal.{branch_name}_branch.num_frames ({branch.num_frames}) "
                f"exceeds data.window_size ({d.window_size})"
            )

    # --- Loss ---
    lc = cfg.loss
    if lc.w_class < 0 or lc.w_bbox_l1 < 0 or lc.w_bbox_giou < 0:
        errors.append("Loss weights (w_class, w_bbox_l1, w_bbox_giou) must be non-negative")

    # --- Optimizer ---
    opt = cfg.optimizer
    if opt.lr <= 0:
        errors.append(f"optimizer.lr must be > 0, got {opt.lr}")
    if opt.weight_decay < 0:
        errors.append(f"optimizer.weight_decay must be >= 0, got {opt.weight_decay}")
    if opt.optimizer not in ("adam", "adamw", "sgd"):
        errors.append(f"optimizer.optimizer must be one of adam/adamw/sgd, got {opt.optimizer!r}")

    # --- Training ---
    tr = cfg.train
    if tr.stage not in (1, 2, 3, 4):
        errors.append(f"train.stage must be 1-4, got {tr.stage}")
    if tr.max_epochs < 1:
        errors.append(f"train.max_epochs must be >= 1, got {tr.max_epochs}")

    if errors:
        msg = f"Config validation failed ({len(errors)} error(s)):\n"
        msg += "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise ConfigValidationError(msg)


def clamp_temporal_branches(cfg: "HTRTDETRConfig") -> None:
    """data.window_size に合わせて temporal branch の num_frames を上限クリップする。

    window_size を CLI や YAML で小さく設定した場合、
    large variant のデフォルト (long_branch.num_frames=16) が
    validate_config() の制約に引っかかるのを自動修正する。

    各学習スクリプトで config 確定後・validate_config() 呼び出し前に実行すること。

    例::
        cfg = get_variant_config("large", stage=2)
        cfg = cfg.merge(overrides)
        clamp_temporal_branches(cfg)   # ← ここで clamp
        validate_config(cfg)           # エラーなし
    """
    ws = cfg.data.window_size
    t = cfg.model.temporal
    for branch in (t.short_branch, t.mid_branch, t.long_branch):
        if branch.num_frames > ws:
            branch.num_frames = ws


def get_stage4_config(overrides: Optional[Dict] = None) -> HTRTDETRConfig:
    """Stage 4: Unified fine-tuning"""
    cfg = HTRTDETRConfig()
    cfg.train.stage = 4
    cfg.train.output_dir = "runs/train/stage4"
    cfg.train.max_epochs = 50
    cfg.optimizer.lr = 1e-5           # 小さい lr で fine-tune
    cfg.model.id_head.use_action_summary = True
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Model size variants: small / medium / large
# ---------------------------------------------------------------------------

# サポートするバリアント名
ModelVariant = str  # Literal["small", "medium", "large"]

# バリアントごとのパラメーター定義
_VARIANT_SPECS: Dict[str, Dict[str, Any]] = {
    "small": {
        # Backbone: ResNet-18 (~11M)
        "backbone_name": "resnet18",
        "backbone_out_channels": [128, 256, 512],
        # FPN / Decoder
        "fpn_out_channels": 128,
        "hidden_dim": 128,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "ffn_dim": 512,
        # Temporal
        "temporal_output_dim": 128,
        "channel_attention": [False, False, False],  # [short, mid, long]
        # ID Head
        "embedding_dim": 64,
        "memory_dim": 128,
        # Action Head
        "interaction_dim": 32,
        "hidden_dims": [256, 128],
        # Interaction module
        "interaction_output_dim": 32,
        # Approximate total: ~15-20M
    },
    "medium": {
        # Backbone: ResNet-34 (~21M)
        "backbone_name": "resnet34",
        "backbone_out_channels": [128, 256, 512],
        # FPN / Decoder
        "fpn_out_channels": 256,
        "hidden_dim": 256,
        "num_decoder_layers": 4,
        "num_heads": 8,
        "ffn_dim": 1024,
        # Temporal
        "temporal_output_dim": 256,
        "channel_attention": [False, False, False],
        # ID Head
        "embedding_dim": 128,
        "memory_dim": 256,
        # Action Head
        "interaction_dim": 64,
        "hidden_dims": [512, 256],
        # Interaction module
        "interaction_output_dim": 64,
        # Approximate total: ~30-40M
    },
    "large": {
        # Backbone: ResNet-50 (~25M)
        "backbone_name": "resnet50",
        "backbone_out_channels": [512, 1024, 2048],
        # FPN / Decoder
        "fpn_out_channels": 512,
        "hidden_dim": 512,
        "num_decoder_layers": 6,
        "num_heads": 16,
        "ffn_dim": 2048,
        # Temporal
        "temporal_output_dim": 512,
        "channel_attention": [True, True, True],
        # ID Head
        "embedding_dim": 256,
        "memory_dim": 512,
        # Action Head
        "interaction_dim": 128,
        "hidden_dims": [1024, 512, 256],
        # Interaction module
        "interaction_output_dim": 128,
        # Approximate total: ~55-70M
    },
}


def build_model_config(variant: ModelVariant) -> ModelConfig:
    """指定バリアント (small / medium / large) に対応する ModelConfig を返す。

    すべての次元依存関係 (fpn.out_channels == temporal.feature_dim など) を
    自動的に整合させる。

    Args:
        variant: "small" | "medium" | "large"

    Returns:
        ModelConfig: バリアントに対応した設定

    Raises:
        ValueError: 不明なバリアント名が渡された場合
    """
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown model variant: {variant!r}. "
            f"Choose from {list(_VARIANT_SPECS.keys())}"
        )

    s = _VARIANT_SPECS[variant]
    d = s["fpn_out_channels"]  # 全モジュール共通の feature 次元

    backbone = BackboneConfig(
        name=s["backbone_name"],
        pretrained=True,
        out_channels=s["backbone_out_channels"],
    )
    fpn = FPNConfig(
        in_channels=s["backbone_out_channels"],
        out_channels=d,
    )
    head = DetectorHeadConfig(
        hidden_dim=s["hidden_dim"],
        num_decoder_layers=s["num_decoder_layers"],
        num_heads=s["num_heads"],
        ffn_dim=s["ffn_dim"],
    )
    detector = DetectorConfig(backbone=backbone, fpn=fpn, head=head)

    ca = s["channel_attention"]
    temporal = HierarchicalTemporalConfig(
        feature_dim=d,
        short_branch=TemporalBranchConfig(
            kernel_size=3, dilation=1, num_frames=3,
            use_depthwise=True, use_residual=True,
            use_channel_attention=ca[0],
        ),
        mid_branch=TemporalBranchConfig(
            kernel_size=3, dilation=2, num_frames=8,
            use_depthwise=True, use_residual=True,
            use_channel_attention=ca[1],
        ),
        long_branch=TemporalBranchConfig(
            kernel_size=3, dilation=4, num_frames=16,
            use_depthwise=True, use_residual=True,
            use_channel_attention=ca[2],
        ),
        output_dim=s["temporal_output_dim"],
    )

    id_head = MemoryIDConfig(
        feature_dim=d,
        embedding_dim=s["embedding_dim"],
        memory_dim=s["memory_dim"],
    )

    action_head = ActionHeadConfig(
        feature_dim=d,
        temporal_dim=s["temporal_output_dim"],
        interaction_dim=s["interaction_dim"],
        hidden_dims=s["hidden_dims"],
    )

    interaction = InteractionConfig(
        feature_dim=d,
        output_dim=s["interaction_output_dim"],
    )

    return ModelConfig(
        detector=detector,
        temporal=temporal,
        id_head=id_head,
        action_head=action_head,
        interaction=interaction,
    )


def get_variant_config(
    variant: ModelVariant,
    stage: int = 1,
    overrides: Optional[Dict] = None,
) -> HTRTDETRConfig:
    """バリアントとステージを組み合わせた HTRTDETRConfig を返す。

    既存の get_stage{N}_config() と build_model_config() を合成し、
    一度の呼び出しで完全な設定を取得できるようにする。

    Args:
        variant: "small" | "medium" | "large"
        stage: 1 | 2 | 3 | 4
        overrides: 追加で上書きしたい設定 (dict)

    Returns:
        HTRTDETRConfig

    Example::

        cfg = get_variant_config("small", stage=1)
        cfg = get_variant_config("large", stage=4, overrides={"data.batch_size": 2})
    """
    _stage_builders = {
        1: get_stage1_config,
        2: get_stage2_config,
        3: get_stage3_config,
        4: get_stage4_config,
    }
    if stage not in _stage_builders:
        raise ValueError(f"stage must be 1-4, got {stage}")

    cfg = _stage_builders[stage]()
    cfg.model = build_model_config(variant)
    cfg.train.output_dir = f"runs/train/{variant}/stage{stage}"

    if overrides:
        cfg = cfg.merge(overrides)
    return cfg
