"""
builder.py — config (YAML/dict) から PipelineRunner を組み立てる

config 形式 (configs/pipeline/*.yaml):

    pipeline:
      mode: lenient            # strict | lenient
      window_size: 16
      stride: 8
      inputs: [image]
      device: auto             # auto | cpu | cuda
      layers:
        L1_detection:
          enabled: true
          backend: rtdetr      # 段名は "{layer}.{backend}" として registry を引く
          params: {score_threshold: 0.3}
        L1b_stabilization:
          enabled: true
          backend: temporal_nms
          params: {iou_link: 0.5}
        ...

「段のトグル = enabled フラグ」「バックエンド差し替え = backend 差し替え」で解析モード
(検出のみ / 時系列検出 / 検出+追跡 / フル) を config だけで切り替えられる。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .runner import PipelineRunner
from .stage import PipelineContext


def _resolve_device(spec: Optional[str]) -> str:
    if spec and spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_pipeline(config: Dict[str, Any]) -> PipelineRunner:
    """pipeline config dict から PipelineRunner を構築する。"""
    # backend を登録するため stages パッケージを読み込む (遅延 import で循環回避)
    from .. import stages as _stages  # noqa: F401
    from .registry import create

    pcfg = config.get("pipeline", config)

    mode = str(pcfg.get("mode", "strict"))
    inputs = set(pcfg.get("inputs", ["image"]))
    ctx = PipelineContext(
        device=_resolve_device(pcfg.get("device", "auto")),
        fps=pcfg.get("fps"),
        window_size=int(pcfg.get("window_size", 16)),
        stride=int(pcfg.get("stride", 8)),
    )

    stages = []
    layers = pcfg.get("layers", {}) or {}
    for layer_name, layer_cfg in layers.items():
        layer_cfg = layer_cfg or {}
        if not layer_cfg.get("enabled", True):
            continue
        backend = layer_cfg.get("backend")
        if not backend:
            raise ValueError(f"layer '{layer_name}' に backend が指定されていません")
        params = layer_cfg.get("params", {}) or {}
        stage_name = f"{layer_name}.{backend}"
        stages.append(create(stage_name, **params))

    return PipelineRunner(stages, inputs=inputs, mode=mode, ctx=ctx)


def build_from_yaml(path: str) -> PipelineRunner:
    """YAML ファイルから PipelineRunner を構築する。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml が必要です: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return build_pipeline(config)
