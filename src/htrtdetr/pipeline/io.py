"""
io.py — パイプライン入出力 (動画/画像ディレクトリ → FrameResult, 窓 → JSON)

軽量に保つため cv2/PIL を遅延 import する (torch は触らない)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .schema import FrameResult, FrameWindow


def frames_from_source(path: str, max_frames: Optional[int] = None) -> Iterator[FrameResult]:
    """動画ファイル or 画像ディレクトリを FrameResult (image 付き) として yield する。"""
    p = Path(path)
    if p.is_dir():
        yield from _frames_from_dir(p, max_frames)
    elif p.exists():
        yield from _frames_from_video(p, max_frames)
    else:
        raise FileNotFoundError(f"入力が見つかりません: {path}")


def _frames_from_video(path: Path, max_frames: Optional[int]) -> Iterator[FrameResult]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        yield FrameResult(frame_index=idx, image=rgb, width=w, height=h)
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()


def _frames_from_dir(path: Path, max_frames: Optional[int]) -> Iterator[FrameResult]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(q for q in path.iterdir() if q.suffix.lower() in exts)
    for idx, q in enumerate(imgs):
        if max_frames is not None and idx >= max_frames:
            break
        try:
            from PIL import Image
            rgb = np.asarray(Image.open(q).convert("RGB"))
        except Exception:  # pragma: no cover
            import cv2
            rgb = cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        yield FrameResult(frame_index=idx, image=rgb, width=w, height=h)


def window_to_predictions(window: FrameWindow) -> Dict[str, Dict]:
    """窓の per-frame 検出を {frame_index: {...}} の JSON 化しやすい dict へ。"""
    out: Dict[str, Dict] = {}
    for fr in window.frames:
        out[str(fr.frame_index)] = {
            "boxes": [d.bbox.tolist() for d in fr.detections],
            "scores": [float(d.score) for d in fr.detections],
            "class_ids": [int(d.class_id) for d in fr.detections],
            "track_ids": [(-1 if d.track_id is None else int(d.track_id)) for d in fr.detections],
            "interpolated": [bool(d.interpolated) for d in fr.detections],
            "relations": [
                {"src": r.src, "dst": r.dst, "kind": r.kind, "value": round(float(r.value), 4),
                 "directed": r.directed}
                for r in fr.relations
            ],
        }
    return out


def interactions_to_json(window: FrameWindow) -> List[Dict]:
    return [
        {
            "agent": i.agent_track_id, "patient": i.patient_track_id,
            "relation": i.relation, "score": round(float(i.score), 4),
            "t_start": i.t_start, "t_end": i.t_end,
        }
        for i in window.interactions
    ]


def merge_windows(windows: List[FrameWindow]) -> FrameWindow:
    """複数窓を1つに集約する。

    既定 (1動画=1窓) では窓は1つなので単純に返る。明示的な多窓 (非重複) の場合でも
    安全になるよう、frame は frame_index で統合、track は同一 track_id を観測の和集合で
    マージ、interactions は (agent,patient,relation,t_start,t_end) で重複排除する。
    """
    by_frame: Dict[int, FrameResult] = {}
    tracks: Dict = {}
    seen_inter = set()
    interactions: List = []
    available = set()

    for w in windows:
        for fr in w.frames:
            by_frame[fr.frame_index] = fr
        for tid, tr in w.tracks.items():
            tracks[tid] = _merge_track(tracks.get(tid), tr)
        for it in w.interactions:
            key = (it.agent_track_id, it.patient_track_id, it.relation, it.t_start, it.t_end)
            if key not in seen_inter:
                seen_inter.add(key)
                interactions.append(it)
        available |= w.available

    return FrameWindow(
        frames=[by_frame[k] for k in sorted(by_frame)],
        tracks=tracks, interactions=interactions, available=available,
    )


def _merge_track(a, b):
    """同一 track_id の2つの Track を frame 単位の観測和集合でマージする。"""
    if a is None:
        return b
    from .schema import Track

    obs: Dict[int, tuple] = {}

    def _collect(tr):
        for i, f in enumerate(tr.frames):
            c = tr.centers[i] if tr.centers is not None else None
            v = tr.velocities[i] if tr.velocities is not None else None
            h = tr.headings[i] if tr.headings is not None else None
            s = tr.speeds[i] if tr.speeds is not None else None
            obs[f] = (c, v, h, s)

    _collect(a)
    _collect(b)  # 後勝ち (重複 frame は b を採用)
    frames = sorted(obs)
    centers = np.stack([obs[f][0] for f in frames]) if all(obs[f][0] is not None for f in frames) else None
    velocities = np.stack([obs[f][1] for f in frames]) if all(obs[f][1] is not None for f in frames) else None
    headings = np.array([obs[f][2] for f in frames], np.float32) if all(obs[f][2] is not None for f in frames) else None
    speeds = np.array([obs[f][3] for f in frames], np.float32) if all(obs[f][3] is not None for f in frames) else None
    return Track(track_id=a.track_id, frames=frames, centers=centers,
                 velocities=velocities, headings=headings, speeds=speeds)
