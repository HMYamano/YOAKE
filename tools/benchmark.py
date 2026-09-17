"""
benchmark.py — FPS / Latency / GPU Memory Benchmarking

YOAKE の推論速度・メモリ使用量を計測し、
比較表を markdown / csv / json で outputs/benchmark/ に保存する。

計測項目:
  - FPS (frames per second)
  - Latency mean/std/p50/p95/p99 (ms)
  - GPU memory (MB, peak allocated)
  - CPU memory baseline

比較モード:
  - image_size:   入力解像度 (320/480/640/800)
  - clip_length:  シーケンス長 (4/8/16/32)
  - batch_size:   バッチサイズ (1/2/4/8)
  - decoder_layers: Decoder 層数 (2/3/4/6)

使い方:
  python tools/benchmark.py \
      checkpoint=outputs/stage1/checkpoint_best.pth \
      mode=image_size \
      output_dir=outputs/benchmark \
      warmup=20 \
      repeats=100

mode:
  image_size | clip_length | batch_size | decoder_layers | single | streaming
  single:    1 解像度での全モジュール breakdown 計測
  streaming: 旧 sliding-window 再計算 vs 新 streaming (ring buffer) の速度比較。
             本改修の中心的な高速化を測る:
                 python tools/benchmark.py mode=streaming windows=4,8,16 num_frames=24
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Timing utilities
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_latency(
    fn,
    warmup: int = 20,
    repeats: int = 100,
) -> Dict[str, float]:
    """fn() を繰り返し実行してレイテンシ統計を計算する。"""
    # Warmup
    for _ in range(warmup):
        fn()
    _sync()

    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean = sum(times_ms) / n
    variance = sum((t - mean) ** 2 for t in times_ms) / max(n - 1, 1)
    import math
    std = math.sqrt(variance)

    return {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(times_ms[int(n * 0.50)], 3),
        "p95_ms": round(times_ms[int(n * 0.95)], 3),
        "p99_ms": round(times_ms[int(n * 0.99)], 3),
        "min_ms": round(times_ms[0], 3),
        "max_ms": round(times_ms[-1], 3),
        "fps": round(1000.0 / mean, 2),
        "repeats": repeats,
    }


def _peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    return 0.0


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Optional[str], device: torch.device):
    from htrtdetr.config.config import get_stage1_config
    from htrtdetr.models import build_model
    from htrtdetr.utils.misc import load_checkpoint

    cfg = get_stage1_config()
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print("  No checkpoint - using random weights.")

    model.eval()
    return model


def _load_model_custom(overrides: dict, checkpoint_path: Optional[str], device: torch.device):
    from htrtdetr.config.config import get_stage1_config
    from htrtdetr.models import build_model
    from htrtdetr.utils.misc import load_checkpoint

    cfg = get_stage1_config(overrides=overrides)
    model = build_model(cfg.model)
    model.to(device)
    model.set_stage(1)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

IMAGE_SIZES = [320, 480, 640, 800]
CLIP_LENGTHS = [4, 8, 16, 32]
BATCH_SIZES = [1, 2, 4, 8]
DECODER_LAYERS = [2, 3, 4, 6]


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def _bench_single(
    model,
    device: torch.device,
    image_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    h = w = image_size
    dummy = torch.randn(batch_size, 3, h, w, device=device)

    _reset_peak()

    def _forward():
        _ = model.forward_single_frame(dummy)

    stats = _measure_latency(_forward, warmup=warmup, repeats=repeats)
    stats["peak_gpu_mb"] = _peak_gpu_memory_mb()
    stats["image_size"] = image_size
    stats["batch_size"] = batch_size
    return stats


# ---------------------------------------------------------------------------
# Comparison sweeps
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sweep_image_size(
    model,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows = []
    for sz in IMAGE_SIZES:
        print(f"    image_size={sz} ...", end=" ", flush=True)
        try:
            row = _bench_single(model, device, sz, batch_size=1, warmup=warmup, repeats=repeats)
            print(f"{row['mean_ms']:.1f} ms ({row['fps']:.1f} fps)")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            row = {"image_size": sz, "error": str(e)}
        rows.append(row)
    return rows


@torch.no_grad()
def _sweep_clip_length(
    model,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows = []
    for clip in CLIP_LENGTHS:
        print(f"    clip_length={clip} ...", end=" ", flush=True)
        dummy = torch.randn(1, clip, 10, device=device)  # geo features

        _reset_peak()

        def _forward(geo=dummy):
            _ = model.forward_geo_sequence(geo)

        try:
            stats = _measure_latency(_forward, warmup=warmup, repeats=repeats)
            stats["peak_gpu_mb"] = _peak_gpu_memory_mb()
            stats["clip_length"] = clip
            print(f"{stats['mean_ms']:.1f} ms ({stats['fps']:.1f} fps)")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            stats = {"clip_length": clip, "error": str(e)}
        rows.append(stats)
    return rows


@torch.no_grad()
def _sweep_batch_size(
    model,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows = []
    for bs in BATCH_SIZES:
        print(f"    batch_size={bs} ...", end=" ", flush=True)
        try:
            row = _bench_single(model, device, image_size=640, batch_size=bs, warmup=warmup, repeats=repeats)
            print(f"{row['mean_ms']:.1f} ms ({row['fps']:.1f} fps)")
        except RuntimeError as e:
            print(f"ERROR (OOM?): {e}")
            row = {"batch_size": bs, "error": str(e)}
        rows.append(row)
    return rows


@torch.no_grad()
def _sweep_decoder_layers(
    checkpoint_path: Optional[str],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    rows = []
    for n_layers in DECODER_LAYERS:
        print(f"    decoder_layers={n_layers} ...", end=" ", flush=True)
        overrides = {"model": {"detector": {"head": {"num_decoder_layers": n_layers}}}}
        model = _load_model_custom(overrides, checkpoint_path, device)

        dummy = torch.randn(1, 3, 640, 640, device=device)
        _reset_peak()

        def _forward(x=dummy, m=model):
            _ = m.forward_single_frame(x)

        try:
            stats = _measure_latency(_forward, warmup=warmup, repeats=repeats)
            stats["peak_gpu_mb"] = _peak_gpu_memory_mb()
            stats["decoder_layers"] = n_layers
            print(f"{stats['mean_ms']:.1f} ms ({stats['fps']:.1f} fps)")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            stats = {"decoder_layers": n_layers, "error": str(e)}
        rows.append(stats)
    return rows


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    import csv
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_markdown(rows: List[Dict], title: str, path: Path) -> None:
    if not rows:
        return
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    lines = [f"# {title}", ""]
    header = "| " + " | ".join(all_keys) + " |"
    sep = "| " + " | ".join(["---"] * len(all_keys)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        cells = []
        for k in all_keys:
            v = row.get(k, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_table(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    col_w = {k: max(len(k), 8) for k in all_keys}
    for row in rows:
        for k, v in row.items():
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            col_w[k] = max(col_w.get(k, 8), len(s))

    header = " | ".join(f"{k:<{col_w[k]}}" for k in all_keys)
    sep = "-+-".join("-" * col_w[k] for k in all_keys)
    print(f"\n=== {title} ===")
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for k in all_keys:
            v = row.get(k, "")
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            cells.append(f"{s:<{col_w[k]}}")
        print(" | ".join(cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Streaming vs sliding-window (旧方式) の比較
# ---------------------------------------------------------------------------

def _bench_streaming_vs_batch(
    checkpoint: Optional[str],
    device: torch.device,
    image_size: int,
    num_frames: int,
    windows: List[int],
) -> List[Dict[str, Any]]:
    """
    **本改修の中心的な高速化**を測る:

        旧: 毎フレーム窓全体を detector に通す (sliding-window で再計算)
        新: 各フレームを 1 回だけ通し、過去は track ごとの ring buffer から再利用

    detector の呼び出し回数も併記する。旧方式は window にほぼ比例して増え、
    新方式はフレーム数に等しく一定であることが、そのまま速度差の説明になる。
    """
    import numpy as np

    from htrtdetr.config.config import InferenceConfig, get_stage1_config
    from htrtdetr.inference import StreamingInferencer
    from htrtdetr.models import build_model
    from htrtdetr.utils.misc import load_checkpoint

    cfg = get_stage1_config()
    model = build_model(cfg.model)
    if checkpoint and Path(checkpoint).exists():
        load_checkpoint(checkpoint, model)
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        # ランダム重みでは class head の prior により検出が 1 つも閾値を超えず、
        # tracking / temporal / behavior が丸ごとスキップされてしまう。それでは
        # 段別 breakdown が全て 0 になり比較にならないので、閾値を下げて
        # 下流段も実際に走らせる (検出内容に意味は無いが計算量は代表的になる)。
        cfg.model.detector.score_threshold = 0.0
        model.cfg.detector.score_threshold = 0.0
        print("  No checkpoint - using random weights "
              "(score_threshold=0 so downstream stages still run).")
    model.to(device).eval()
    model.set_stage(4)

    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    rows: List[Dict[str, Any]] = []
    for window in windows:
        si = StreamingInferencer(
            model, InferenceConfig(use_amp=False),
            window_size=window, image_size=(image_size, image_size),
            device=device,
        )

        # --- 旧方式: sliding window で毎回窓全体を通す ---
        calls = {"n": 0}
        orig = model.detector.forward

        def _counting(x, _orig=orig, _c=calls):
            _c["n"] += 1
            return _orig(x)

        model.detector.forward = _counting
        try:
            _reset_peak()
            buf: List = []
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                for f in frames:
                    buf.append(si.preprocess(f))
                    if len(buf) > window:
                        buf.pop(0)
                    _ = model(torch.cat(buf, dim=0).unsqueeze(0))
            _sync()
            t_batch = time.perf_counter() - t0
            batch_calls = calls["n"]
            batch_mem = _peak_gpu_memory_mb()

            # --- 新方式: streaming ---
            calls["n"] = 0
            si.reset()
            _reset_peak()
            _sync()
            t0 = time.perf_counter()
            for i, f in enumerate(frames):
                si.process_frame(f, frame_index=i)
            _sync()
            t_stream = time.perf_counter() - t0
            stream_calls = calls["n"]
            stream_mem = _peak_gpu_memory_mb()
        finally:
            model.detector.forward = orig

        breakdown = si.runtime_stats()
        rows.append({
            "window": window,
            "frames": num_frames,
            "batch_ms_per_frame": round(t_batch / num_frames * 1000, 2),
            "streaming_ms_per_frame": round(t_stream / num_frames * 1000, 2),
            "speedup": round(t_batch / max(t_stream, 1e-9), 2),
            "batch_detector_calls": batch_calls,
            "streaming_detector_calls": stream_calls,
            "call_reduction": round(batch_calls / max(stream_calls, 1), 2),
            "batch_fps": round(num_frames / max(t_batch, 1e-9), 2),
            "streaming_fps": round(num_frames / max(t_stream, 1e-9), 2),
            "batch_peak_gpu_mb": batch_mem,
            "streaming_peak_gpu_mb": stream_mem,
            "detector_ms": breakdown.get("detector_ms", 0.0),
            "tracking_ms": breakdown.get("tracking_ms", 0.0),
            "temporal_ms": breakdown.get("temporal_ms", 0.0),
            "behavior_ms": breakdown.get("behavior_ms", 0.0),
        })
        print(
            f"  window={window:>3}: batch {rows[-1]['batch_ms_per_frame']:>8.2f} ms/f | "
            f"streaming {rows[-1]['streaming_ms_per_frame']:>8.2f} ms/f | "
            f"speedup {rows[-1]['speedup']:>5.2f}x | "
            f"detector calls {batch_calls} -> {stream_calls}"
        )
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse(sys.argv[1:])

    checkpoint = args.get("checkpoint", None)
    mode = args.get("mode", "single")
    output_dir = Path(args.get("output_dir", "outputs/benchmark"))
    warmup = int(args.get("warmup", "20"))
    repeats = int(args.get("repeats", "100"))
    image_size = int(args.get("image_size", "640"))
    batch_size = int(args.get("batch_size", "1"))

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== YOAKE Benchmark ===")
    print(f"  Mode     : {mode}")
    print(f"  Device   : {device}")
    print(f"  Warmup   : {warmup}")
    print(f"  Repeats  : {repeats}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")

    if mode == "decoder_layers":
        print("\nSweeping decoder layers (builds separate models)...")
        rows = _sweep_decoder_layers(checkpoint, device, warmup, repeats)
        title = "Decoder Layers Comparison"
    elif mode == "streaming":
        windows = [int(w) for w in args.get("windows", "4,8,16").split(",")]
        num_frames = int(args.get("num_frames", "24"))
        print(f"\nStreaming vs sliding-window (frames={num_frames}, windows={windows})...")
        rows = _bench_streaming_vs_batch(
            checkpoint, device, image_size, num_frames, windows
        )
        title = "Streaming vs Sliding-Window Inference"
    else:
        model = _load_model(checkpoint, device)

        if mode == "image_size":
            print("\nSweeping image sizes...")
            rows = _sweep_image_size(model, device, warmup, repeats)
            title = "Image Size Comparison"

        elif mode == "clip_length":
            print("\nSweeping clip lengths (geo sequence)...")
            rows = _sweep_clip_length(model, device, warmup, repeats)
            title = "Clip Length Comparison"

        elif mode == "batch_size":
            print("\nSweeping batch sizes...")
            rows = _sweep_batch_size(model, device, warmup, repeats)
            title = "Batch Size Comparison"

        elif mode == "single":
            print(f"\nSingle benchmark (image_size={image_size}, batch_size={batch_size})...")
            row = _bench_single(model, device, image_size, batch_size, warmup, repeats)
            rows = [row]
            title = f"Single Benchmark ({image_size}px, bs={batch_size})"

        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Choose: single | image_size | "
                "clip_length | batch_size | decoder_layers | streaming"
            )

    _print_table(rows, title)

    stem = f"benchmark_{mode}"
    _save_csv(rows, output_dir / f"{stem}.csv")
    _save_json({"mode": mode, "device": str(device), "results": rows},
               output_dir / f"{stem}.json")
    _save_markdown(rows, title, output_dir / f"{stem}.md")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  {stem}.csv")
    print(f"  {stem}.json")
    print(f"  {stem}.md")


if __name__ == "__main__":
    main()
