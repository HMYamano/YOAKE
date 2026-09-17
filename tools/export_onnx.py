"""
export_onnx.py — ONNX Export & Inference Checker

YOAKE の Stage 1 (Detector) または Stage 2/3 の geo-sequence パスを
ONNX にエクスポートし、PyTorch / ONNX Runtime での出力一致を検証する。

使い方:
  python tools/export_onnx.py \
      checkpoint=outputs/stage1/checkpoint_best.pth \
      mode=detector \
      output=exports/ht_rtdetr_detector.onnx \
      image_size=640 \
      opset=17 \
      verify=1

mode:
  detector      — forward_single_frame(image) → pred_logits + pred_boxes (Stage 1)
  obb_detector  — 上記 + pred_obb [cx, cy, w, h, theta] (OBB detector 用)
  geo_stage2    — forward_geo_sequence → action_logits (Stage 2)
  geo_stage3    — forward_geo_sequence → id_embeddings + id_logits (Stage 3)

動的 axes:
  detector / obb_detector : batch (N), height (H), width (W)
  geo_*                   : batch (N), time (T)

注: behavior ブランチは track ごとの可変長履歴と動的な ROI 抽出に依存するため、
    単一の静的グラフとしては export していない (TensorRT のために本体設計を
    歪めないことを優先)。detector を export し、tracking / behavior は
    PyTorch 側で回す構成を想定している。

verify=1 の場合:
  - ONNXRuntime でランダム入力を推論
  - PyTorch の出力と最大絶対誤差を表示
  - 許容誤差 (atol=1e-4) を超えた場合 WARNING を出力

依存:
  pip install onnx onnxruntime   (CPU用)
  pip install onnxruntime-gpu    (GPU用, optional)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


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
# Wrapper modules for clean ONNX export
# ---------------------------------------------------------------------------

class DetectorWrapper(nn.Module):
    """forward_single_frame を ONNX-friendly にラップする。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (N, 3, H, W)
        Returns:
            pred_logits : (N, num_queries, num_classes + 1)
            pred_boxes  : (N, num_queries, 4)  cxcywh normalized
        """
        # HTRTDETROutput は dataclass 風の属性アクセス型 (subscript 不可)
        out = self.model.forward_single_frame(image)
        return out.pred_logits, out.pred_boxes


class OBBDetectorWrapper(nn.Module):
    """
    OBB detector を ONNX-friendly にラップする。

    ``pred_boxes`` (外接 AABB) に加えて ``pred_obb`` を出す。角度は
    ``decode_angle`` (atan2) を通した後の theta なので、後段は追加の復号なしで
    そのまま回転矩形として扱える。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_logits : (N, Q, num_classes + 1)
            pred_boxes  : (N, Q, 4)  cxcywh normalized (OBB の外接矩形)
            pred_obb    : (N, Q, 5)  [cx, cy, w, h, theta]
        """
        out = self.model.forward_single_frame(image)
        if out.pred_obb is None:
            raise ValueError(
                "この checkpoint は OBB detector ではありません "
                "(detector.obb.enabled=false)。mode=detector を使ってください。"
            )
        return out.pred_logits, out.pred_boxes, out.pred_obb


class GeoStage2Wrapper(nn.Module):
    """forward_geo_sequence → action_logits"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geo: (N, T, 10)
        Returns:
            action_logits: (N, num_actions)
        """
        out = self.model.forward_geo_sequence(geo)
        return out["action_logits"]


class GeoStage3Wrapper(nn.Module):
    """forward_geo_sequence → id_embeddings, id_logits"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, geo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            geo: (N, T, 10)
        Returns:
            id_embeddings: (N, embedding_dim)
            id_logits    : (N, max_ids + 1)
        """
        out = self.model.forward_geo_sequence(geo)
        return out["id_embeddings"], out["id_logits"]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build(
    mode: str,
    checkpoint: Optional[str],
    device: torch.device,
) -> Tuple[nn.Module, torch.Tensor, List[str], List[str], Dict]:
    """ラッパーモデル / サンプル入力 / 入出力名 / 動的 axes を返す。"""
    from htrtdetr.config.config import get_stage1_config, get_stage2_config, get_stage3_config
    from htrtdetr.models import build_model
    from htrtdetr.utils.misc import load_checkpoint

    if mode == "detector":
        cfg = get_stage1_config()
        raw = build_model(cfg.model)
        raw.set_stage(1)
        if checkpoint and Path(checkpoint).exists():
            load_checkpoint(checkpoint, raw)
        raw.to(device).eval()
        wrapper = DetectorWrapper(raw)

        dummy = torch.randn(1, 3, 640, 640, device=device)
        input_names = ["image"]
        output_names = ["pred_logits", "pred_boxes"]
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    elif mode == "geo_stage2":
        cfg = get_stage2_config()
        raw = build_model(cfg.model)
        raw.set_stage(2)
        if checkpoint and Path(checkpoint).exists():
            load_checkpoint(checkpoint, raw)
        raw.to(device).eval()
        wrapper = GeoStage2Wrapper(raw)

        dummy = torch.randn(1, 16, 10, device=device)
        input_names = ["geo_features"]
        output_names = ["action_logits"]
        dynamic_axes = {
            "geo_features": {0: "batch", 1: "time"},
            "action_logits": {0: "batch"},
        }

    elif mode == "geo_stage3":
        cfg = get_stage3_config()
        raw = build_model(cfg.model)
        raw.set_stage(3)
        if checkpoint and Path(checkpoint).exists():
            load_checkpoint(checkpoint, raw)
        raw.to(device).eval()
        wrapper = GeoStage3Wrapper(raw)

        dummy = torch.randn(1, 16, 10, device=device)
        input_names = ["geo_features"]
        output_names = ["id_embeddings", "id_logits"]
        dynamic_axes = {
            "geo_features": {0: "batch", 1: "time"},
            "id_embeddings": {0: "batch"},
            "id_logits": {0: "batch"},
        }

    elif mode == "obb_detector":
        # OBB detector は checkpoint の config から復元する必要がある
        # (角度ヘッドの有無で state_dict の形が変わるため)
        from htrtdetr.config.config import enable_obb

        cfg = get_stage1_config()
        enable_obb(cfg)
        raw = build_model(cfg.model)
        raw.set_stage(1)
        if checkpoint and Path(checkpoint).exists():
            load_checkpoint(checkpoint, raw)
        raw.to(device).eval()
        wrapper = OBBDetectorWrapper(raw)

        dummy = torch.randn(1, 3, 640, 640, device=device)
        input_names = ["image"]
        output_names = ["pred_logits", "pred_boxes", "pred_obb"]
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_obb": {0: "batch"},
        }

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Choose: detector | obb_detector | geo_stage2 | geo_stage3"
        )

    return wrapper, dummy, input_names, output_names, dynamic_axes


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export(
    wrapper: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: dict,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Exporting to: {output_path}")
    print(f"  Opset: {opset}")
    print(f"  Input  : {input_names}  shape={list(dummy.shape)}")
    print(f"  Output : {output_names}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ONNX file size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------

def _verify(
    wrapper: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    output_names: List[str],
    atol: float = 1e-4,
) -> Dict[str, float]:
    """PyTorch と ONNX Runtime の出力を比較する。"""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  SKIP: onnx / onnxruntime not installed. pip install onnx onnxruntime")
        return {}

    # Validate ONNX model structure
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: OK")

    # PyTorch inference
    with torch.no_grad():
        pt_out = wrapper(dummy)
    if not isinstance(pt_out, (tuple, list)):
        pt_out = (pt_out,)

    # ONNXRuntime inference
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(output_path), providers=providers)
    input_feed = {sess.get_inputs()[0].name: dummy.cpu().numpy()}
    ort_out = sess.run(None, input_feed)

    # Compare
    results: Dict[str, float] = {}
    for i, name in enumerate(output_names):
        import numpy as np
        pt_np = pt_out[i].cpu().float().numpy()
        ort_np = ort_out[i].astype(np.float32)
        max_diff = float(np.abs(pt_np - ort_np).max())
        mean_diff = float(np.abs(pt_np - ort_np).mean())
        results[f"{name}_max_diff"] = max_diff
        results[f"{name}_mean_diff"] = mean_diff
        status = "OK" if max_diff <= atol else "WARNING (>atol)"
        print(f"  {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}  [{status}]")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse(sys.argv[1:])

    checkpoint = args.get("checkpoint", None)
    mode = args.get("mode", "detector")
    output = args.get("output", f"exports/ht_rtdetr_{mode}.onnx")
    image_size = int(args.get("image_size", "640"))
    opset = int(args.get("opset", "17"))
    verify = args.get("verify", "1").strip().lower() not in ("0", "false", "no")
    atol = float(args.get("atol", "1e-4"))

    output_path = Path(output)
    device = torch.device("cpu")  # Export on CPU for maximum compatibility
    # Note: for GPU export, change to cuda if needed

    print(f"=== ONNX Export ===")
    print(f"  Mode       : {mode}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Output     : {output_path}")
    print(f"  Opset      : {opset}")
    print(f"  Verify     : {verify}")
    print(f"  Device     : {device} (export always on CPU for portability)")

    wrapper, dummy, input_names, output_names, dynamic_axes = _build(
        mode, checkpoint, device
    )

    # Override image size for detector mode
    if mode == "detector":
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        print(f"  Image size : {image_size}")

    wrapper.eval()
    with torch.no_grad():
        _export(wrapper, dummy, output_path, input_names, output_names, dynamic_axes, opset)

    verify_results: Dict = {}
    if verify:
        print("\nVerifying...")
        verify_results = _verify(wrapper, dummy, output_path, output_names, atol=atol)

    # Save metadata
    meta = {
        "mode": mode,
        "opset": opset,
        "checkpoint": checkpoint,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "input_shape": list(dummy.shape),
        "onnx_file": str(output_path),
        "onnx_size_mb": round(output_path.stat().st_size / 1024 / 1024, 3),
        "verification": verify_results,
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved: {meta_path}")
    print("Export complete.")


if __name__ == "__main__":
    main()
