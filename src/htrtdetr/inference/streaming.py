"""
streaming.py — True streaming inference (過去フレームを detector へ再投入しない)

=== 何を直したか ===

旧 ``Inferencer`` は 16 フレーム窓を 1 フレームずつずらしながら、毎回**窓全体**を
モデルに渡していた:

    frame 16 の推論 → detector(1..16)    16 回の空間推論
    frame 17 の推論 → detector(2..17)    16 回の空間推論 (15 回は再計算)
    frame 18 の推論 → detector(3..18)    ...

同じフレームが最大 window_size 回 detector を通る。空間推論は全体の大半を占めるので、
実質 window_size 倍の無駄になっていた。

本実装は各フレームをちょうど 1 回だけ通す:

    new frame → detector (1回) → detections → tracking → track_id
              → per-track ring buffer → HTM → behavior

過去フレームの情報は ``TrackFeatureCache`` (track ごとのリングバッファ) が持つので、
再計算は不要。計算量は window_size に依存しなくなる。

=== 数値の一致について ===

streaming と batch は「同じ検出に対して同じ時間特徴」を与える設計だが、両者が
ビット単位で一致するとは限らない。batch は窓ごとに tracker を作り直すのに対し、
streaming は動画全体で tracker を持ち続けるため、ID の付き方が異なりうるためである。
検出そのもの (bbox / score) は完全に一致する。
``tests/test_streaming.py`` がこの「検出一致・行動は整合」を固定している。
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.config import InferenceConfig
from ..geometry.obb import cxcywh_to_obb, obb_to_aabb, obb_to_corners
from ..models.ht_rtdetr import HTRTDETR
from ..tracking import OnlineTracker
from ..utils.logging import get_logger
from ..utils.misc import cxcywh_to_xyxy, get_device

__all__ = ["StreamingInferencer", "StreamingResult", "LatencyBreakdown"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LatencyBreakdown:
    """段ごとのレイテンシを積算する (benchmark / GUI 表示用)。"""

    STAGES = ("preprocess", "detector", "tracking", "temporal", "behavior", "total")

    def __init__(self) -> None:
        self._acc: Dict[str, float] = {s: 0.0 for s in self.STAGES}
        self._n = 0

    def add(self, stage: str, seconds: float) -> None:
        if stage in self._acc:
            self._acc[stage] += seconds

    def finish_frame(self) -> None:
        self._n += 1

    def summary(self) -> Dict[str, float]:
        """各段の平均 ms と FPS。"""
        n = max(self._n, 1)
        out = {f"{s}_ms": round(self._acc[s] / n * 1000.0, 3) for s in self.STAGES}
        out["fps"] = round(n / max(self._acc["total"], 1e-9), 2)
        out["frames"] = self._n
        return out

    @contextlib.contextmanager
    def measure(self, stage: str, sync: bool = False):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.add(stage, time.perf_counter() - t0)


class StreamingResult:
    """1 フレーム分の推論結果。"""

    __slots__ = (
        "frame_index", "boxes_xyxy", "obb", "corners", "scores", "class_ids",
        "track_ids", "action_ids", "action_probs", "context_boxes",
        "evidence", "gates", "head_predictions",
    )

    def __init__(
        self,
        frame_index: int,
        boxes_xyxy: np.ndarray,
        obb: Optional[np.ndarray] = None,
        corners: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
        track_ids: Optional[np.ndarray] = None,
        action_ids: Optional[np.ndarray] = None,
        action_probs: Optional[np.ndarray] = None,
        context_boxes: Optional[np.ndarray] = None,
        evidence: Optional[List[Dict[str, float]]] = None,
        gates: Optional[List[Dict[str, float]]] = None,
        head_predictions: Optional[List[Dict[str, str]]] = None,
    ):
        n = len(boxes_xyxy)
        self.frame_index = frame_index
        self.boxes_xyxy = boxes_xyxy
        self.obb = obb
        self.corners = corners
        self.scores = scores if scores is not None else np.ones(n, dtype=np.float32)
        self.class_ids = class_ids if class_ids is not None else np.zeros(n, dtype=int)
        self.track_ids = track_ids if track_ids is not None else np.full(n, -1, dtype=int)
        self.action_ids = action_ids if action_ids is not None else np.full(n, -1, dtype=int)
        self.action_probs = action_probs
        self.context_boxes = context_boxes
        self.evidence = evidence or []
        self.gates = gates or []
        self.head_predictions = head_predictions or []

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    def to_json(self) -> Dict[str, Any]:
        """JSON 出力用の dict (既存 predictions.json と互換なキーを保つ)。"""
        d: Dict[str, Any] = {
            "boxes": self.boxes_xyxy.tolist(),
            "scores": self.scores.tolist(),
            "track_ids": self.track_ids.tolist(),
            "action_ids": self.action_ids.tolist(),
        }
        if self.obb is not None:
            d["obb"] = self.obb.tolist()
        if self.context_boxes is not None:
            d["context_boxes"] = self.context_boxes.tolist()
        if self.evidence:
            d["evidence"] = self.evidence
        if self.gates:
            d["gates"] = self.gates
        if self.head_predictions:
            d["head_predictions"] = self.head_predictions
        return d


class StreamingInferencer:
    """
    フレームを 1 回だけ detector に通す streaming 推論器。

    Args:
        model: 学習済み HTRTDETR
        cfg: InferenceConfig
        action_names / class_names: 表示名
        window_size: track ごとの履歴長 (HTM の時間窓)
        image_size: detector 入力サイズ (H, W)
        device: 明示指定 (None なら自動)
    """

    def __init__(
        self,
        model: HTRTDETR,
        cfg: Optional[InferenceConfig] = None,
        action_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        window_size: int = 16,
        image_size: Tuple[int, int] = (640, 640),
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg or InferenceConfig()
        self.action_names = action_names or ["unknown"]
        self.class_names = class_names or ["animal"]
        self.window_size = int(window_size)
        self.image_size = image_size
        self.logger = get_logger("streaming")

        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.model.set_stage(4)

        # --- 高速化: 利用できない backend では自動 fallback ---
        self._use_amp = bool(self.cfg.use_amp) and self.device.type in ("cuda", "cpu")
        self._amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        if self.cfg.use_channels_last and self.device.type == "cuda":
            # channels_last は conv 主体の backbone で効く。CPU/MPS では逆効果のことが
            # あるので CUDA のみ適用する。
            self.model = self.model.to(memory_format=torch.channels_last)
            self._channels_last = True
        else:
            self._channels_last = False

        if self.cfg.use_compile:
            self._try_compile()

        # --- 持続状態 ---
        feat_dim = self.model.detector.get_hidden_dim()
        self.tracker = OnlineTracker.from_config(
            self.model.cfg.tracking,
            feat_dim=feat_dim,
            window=self.window_size,
            device=self.device,
        )
        self.memory = self.model.id_head.create_memory(self.device)
        self.latency = LatencyBreakdown()
        self._frame_index = -1

    def _try_compile(self) -> None:
        """torch.compile を試す。失敗しても推論自体は継続する。"""
        try:
            self.model.detector = torch.compile(self.model.detector)
            self.logger.info("torch.compile を detector に適用しました")
        except Exception as exc:  # pragma: no cover - 環境依存
            self.logger.warning(f"torch.compile が使えません ({exc}); 通常実行に fallback")

    # ------------------------------------------------------------------ #
    # 状態管理
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """新しい動画の開始時に全ての持続状態を捨てる。"""
        self.tracker.reset()
        self.memory = self.model.id_head.create_memory(self.device)
        self.latency = LatencyBreakdown()
        self._frame_index = -1

    def _autocast(self):
        if not self._use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    # ------------------------------------------------------------------ #
    # 前処理
    # ------------------------------------------------------------------ #

    def preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """np.ndarray (H, W, 3) RGB → 正規化テンソル (1, 3, H, W)。"""
        H, W = self.image_size
        try:
            from PIL import Image
            img = np.asarray(
                Image.fromarray(frame_rgb).resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        except ImportError:  # pragma: no cover
            import cv2
            img = cv2.resize(frame_rgb, (W, H)).astype(np.float32) / 255.0

        img = (img - _MEAN) / _STD
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        # pin_memory + non_blocking で CPU→GPU コピーを計算と重ねる
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        else:
            t = t.to(self.device)
        if self._channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        return t

    # ------------------------------------------------------------------ #
    # 1 フレーム処理 (本体)
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def process_frame(
        self,
        frame_rgb: np.ndarray,
        frame_index: Optional[int] = None,
    ) -> StreamingResult:
        """
        1 フレームを処理する。detector を通すのは**このフレームだけ**。

        Args:
            frame_rgb: (H, W, 3) RGB uint8
            frame_index: フレーム番号 (None なら内部カウンタ)
        Returns:
            StreamingResult
        """
        t_start = time.perf_counter()
        fi = self._frame_index + 1 if frame_index is None else int(frame_index)
        self._frame_index = fi
        H0, W0 = frame_rgb.shape[:2]

        with self.latency.measure("preprocess"):
            tensor = self.preprocess(frame_rgb)

        # --- Detector: 1 フレーム 1 回だけ ---
        with self.latency.measure("detector", sync=True), self._autocast():
            det_out = self.model.detector(tensor)
            dets = self.model.det_adapter.extract_valid_detections(
                det_out.pred_logits, det_out.pred_boxes, det_out.query_features,
                score_threshold=self.model.cfg.detector.score_threshold,
                pred_obb=det_out.pred_obb,
                pred_keypoints=det_out.pred_keypoints,
                pred_polarity=det_out.pred_polarity,
            )[0]

        feats = dets["features"].float()
        boxes = dets["boxes"].float()
        n = int(boxes.shape[0])
        if n == 0:
            self.tracker.cache.update(fi, [], feats, torch.zeros(0, 5, device=self.device))
            self.latency.add("total", time.perf_counter() - t_start)
            self.latency.finish_frame()
            return StreamingResult(fi, np.zeros((0, 4), dtype=np.float32))

        has_obb = "obb" in dets
        obb = dets["obb"].float() if has_obb else cxcywh_to_obb(boxes)

        # --- Tracking (ring buffer への push も含む) ---
        with self.latency.measure("tracking"):
            id_out = self.model.id_head.forward_inference(feats, boxes, self.memory)
            embeddings = id_out["embeddings"].float()
            tracked = self.tracker.update(
                frame_index=fi,
                det_obb=obb,
                det_features=feats,
                det_scores=dets["scores"].float(),
                det_embed=embeddings,
                has_obb=has_obb,
            )

        track_ids = np.full(n, -1, dtype=int)
        for obj in tracked:
            track_ids[obj.detection_index] = obj.track_id

        # --- Temporal: ring buffer の履歴を HTM に通す (再検出なし) ---
        with self.latency.measure("temporal"):
            hist, obb_hist, mask, active_ids = self.tracker.track_history()
            if hist.shape[0] > 0:
                packed = self.model.track_temporal_fusion.pack_from_cache(hist, mask)
                with self._autocast():
                    temporal_seq = self.model.temporal(packed)
                temporal_last = temporal_seq[:, -1].float()
            else:
                temporal_last = torch.zeros(
                    0, self.model.cfg.temporal.output_dim, device=self.device
                )

        # --- Behavior ---
        with self.latency.measure("behavior"):
            action_ids, action_probs, evidence, gates, heads, ctx_boxes = self._run_behavior(
                obb_hist, mask, det_out, temporal_last, hist, active_ids, track_ids, n
            )

        # --- 画素座標へ戻す ---
        scale = np.array([W0, H0, W0, H0], dtype=np.float32)
        boxes_px = cxcywh_to_xyxy(boxes).cpu().numpy() * scale

        obb_px = obb.cpu().numpy().copy()
        obb_px[:, [0, 2]] *= W0
        obb_px[:, [1, 3]] *= H0
        corners_px = obb_to_corners(obb).cpu().numpy()
        corners_px[..., 0] *= W0
        corners_px[..., 1] *= H0

        if ctx_boxes is not None:
            ctx_boxes = ctx_boxes * scale

        self.latency.add("total", time.perf_counter() - t_start)
        self.latency.finish_frame()

        return StreamingResult(
            frame_index=fi,
            boxes_xyxy=boxes_px,
            obb=obb_px,
            corners=corners_px,
            scores=dets["scores"].float().cpu().numpy(),
            class_ids=dets["class_ids"].cpu().numpy(),
            track_ids=track_ids,
            action_ids=action_ids,
            action_probs=action_probs,
            context_boxes=ctx_boxes,
            evidence=evidence,
            gates=gates,
            head_predictions=heads,
        )

    def _run_behavior(
        self, obb_hist, mask, det_out, temporal_last, hist, active_ids,
        track_ids: np.ndarray, n: int,
    ):
        """行動認識を track 単位で走らせ、検出順に並べ直す。"""
        empty = (np.full(n, -1, dtype=int), None, [], [], [], None)
        if not self.model.behavior_enabled or obb_hist.shape[0] == 0:
            return empty

        cfg_b = self.model.cfg.behavior
        spatial = det_out.encoder_features[0] if det_out.encoder_features else None
        if (cfg_b.use_actor or cfg_b.use_context) and spatial is None:
            return empty

        batch_index = torch.zeros(
            obb_hist.shape[0], dtype=torch.long, device=self.device
        )
        with self._autocast():
            pred = self.model.behavior(
                obb_history=obb_hist,
                history_mask=mask,
                spatial_features=spatial.float() if spatial is not None else None,
                batch_index=batch_index,
                detector_features=hist[:, -1] if cfg_b.use_detector_feat else None,
                temporal_features=temporal_last if cfg_b.use_temporal else None,
            )

        row_of = {tid: i for i, tid in enumerate(active_ids)}
        action_ids = np.full(n, -1, dtype=int)
        action_probs = None
        evidence: List[Dict[str, float]] = []
        gates: List[Dict[str, float]] = []
        heads: List[Dict[str, str]] = []

        if pred.logits is not None:
            probs = pred.probs.float().cpu().numpy()
            action_probs = np.zeros((n, probs.shape[1]), dtype=np.float32)
            for i in range(n):
                row = row_of.get(int(track_ids[i]), -1)
                if row < 0:
                    evidence.append({})
                    gates.append({})
                    continue
                action_probs[i] = probs[row]
                action_ids[i] = int(probs[row].argmax())
                evidence.append(pred.explain(row))
                gates.append(pred.gate_weights(row))
        elif pred.head_logits:
            # multi-head: head ごとの最尤クラス名を返す
            head_np = {
                h: v.float().cpu().numpy().argmax(axis=-1)
                for h, v in pred.head_logits.items()
            }
            names = self.model.cfg.behavior.heads
            for i in range(n):
                row = row_of.get(int(track_ids[i]), -1)
                if row < 0:
                    heads.append({})
                    gates.append({})
                    continue
                heads.append({
                    h: (names.get(h, [])[idx[row]]
                        if h in names and idx[row] < len(names[h]) else str(idx[row]))
                    for h, idx in head_np.items()
                })
                gates.append(pred.gate_weights(row))

        # GUI 用: 行動認識が実際に使った context box
        ctx = None
        if cfg_b.use_context:
            ctx_t = self.model.behavior.context_boxes(obb_hist[:, -1])
            if ctx_t is not None:
                ctx_np = ctx_t.float().cpu().numpy()
                ctx = np.zeros((n, 4), dtype=np.float32)
                for i in range(n):
                    row = row_of.get(int(track_ids[i]), -1)
                    if row >= 0:
                        ctx[i] = ctx_np[row]

        return action_ids, action_probs, evidence, gates, heads, ctx

    # ------------------------------------------------------------------ #
    # 動画全体
    # ------------------------------------------------------------------ #

    def run(
        self,
        frames: Iterable[Tuple[int, np.ndarray]],
        output_dir: Optional[str] = None,
    ) -> Dict[int, StreamingResult]:
        """
        フレーム列を順に処理する。

        Args:
            frames: (frame_index, frame_rgb) の iterable
            output_dir: 指定すると predictions.json / runtime_stats.json を書く
        Returns:
            {frame_index: StreamingResult}
        """
        self.reset()
        results: Dict[int, StreamingResult] = {}
        for idx, frame in frames:
            results[idx] = self.process_frame(frame, frame_index=idx)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "predictions.json", "w", encoding="utf-8") as f:
                json.dump({str(k): v.to_json() for k, v in results.items()}, f, indent=2)
            with open(out / "runtime_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.latency.summary(), f, indent=2)

        return results

    def runtime_stats(self) -> Dict[str, float]:
        """段ごとのレイテンシと FPS。"""
        return self.latency.summary()
