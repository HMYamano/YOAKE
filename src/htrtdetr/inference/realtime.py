"""
realtime.py — multiprocessing によるリアルタイム推論 (docs/REALTIME.md)

=== なぜプロセスを分けるのか ===

streaming 推論 (``StreamingInferencer``) は tracker が逐次なので 1 フレームずつしか
進めない。そのうち GPU を使うのは detector / HTM / behavior だけで、残り — 動画の
デコード、640² への引き伸ばしと正規化、シルエット幾何、注釈の描画と mp4 エンコード —
は CPU の仕事であり、逐次ループでは GPU の待ち時間と直列に並ぶ。ここではそれを
別プロセスに出して並走させる:

    capture ──▶ main (GPU) ──▶ geometry ──▶ writer
    decode       detector        silhouette    draw / mp4
    resize       tracker         head-tail     predictions.json(l)
    normalize    HTM / behavior  motion        display

フレーム t の推論中に capture は t+1.. を用意し、writer は t-1.. を書く。GPU (CUDA) を
持つのは main だけで、子プロセスは CUDA に触らない。geometry はシルエット幾何が
有効なときだけ起動する。

=== フレームの受け渡し ===

画素は pickle しない。``FrameRing`` (共有メモリ上の固定数スロット) に capture が生の
フレームとモデル入力 (``prepare_input`` の出力) を書き、キューに流れるのはスロット番号と
小さな結果だけ。スロットは writer が書き終えたら free キューへ戻る。スロット数が
パイプラインの深さ — つまり遅延の上限 — になる。

=== 落とすか待つか ===

ファイル / 画像フォルダは 1 フレームも落とさない (capture が空きスロットを待つ)。
前処理は ``prepare_input`` を capture で走らせるだけなので、結果は逐次の
``yoake predict`` とビット単位で一致する (tests/test_realtime.py が固定している)。

カメラやストリームは待てない。待つとドライバのバッファが溢れ、どのフレームが消えたか
分からないまま時間が詰まる。空きスロットが無ければ capture 側で明示的に捨て、
フレーム番号は欠番として残す。tracker は実フレーム番号で速度を取るので
(``TrackState.velocity``)、欠番は「その間は観測が無い」として扱われる。

=== 止め方 ===

入力が尽きる / ``max_frames`` / ``max_seconds`` / 表示ウィンドウで q か Esc /
Ctrl+C のいずれでも、capture が止まり、終端の印が全段を流れて、writer が
そこまでの結果を書き切ってから終わる。子プロセスは SIGINT を無視する
(Ctrl+C はコンソールの全プロセスに届くため)。どこかの段が例外で落ちれば、その
traceback を付けて main が ``RuntimeError`` を上げる。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import time
import traceback
from dataclasses import asdict, dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.silhouette import (
    GeometryCSV,
    SilhouetteAnalyzer,
    SilhouetteParams,
    draw_geometry,
    geometry_meta,
)
from .inferencer import VideoWriter, draw_detections, read_image_dir_frames, read_video_frames
from .streaming import prepare_input

__all__ = [
    "RealtimeOptions",
    "RingSpec",
    "FrameRing",
    "is_live_source",
    "run_realtime",
]

#: 全キュー共通の終端の印。
_STOP = None
#: 待っている間に停止要求と他プロセスの生死を見る間隔 [秒]。
_POLL = 0.1
#: 異常終了時に子プロセスの終わりを待つ時間 [秒]。writer は書き切るまで少しかかる。
_JOIN_TIMEOUT = 30.0
_WINDOW = "YOAKE realtime"
_LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live_source(source: Any) -> bool:
    """カメラ番号 ("0", "1", ...) か、ネットワークストリーム (rtsp:// など) か。"""
    s = str(source).strip()
    if s.isdigit():
        # 同名のファイルやフォルダがあればそちらを優先する。
        return not Path(s).exists()
    return s.lower().startswith(_LIVE_PREFIXES)


@dataclass
class RealtimeOptions:
    """``run_realtime`` の設定。CLI では ``InferenceConfig.realtime_*`` から作る。"""

    #: 共有メモリのフレームスロット数 (= パイプラインの深さ、遅延の上限)。0 = 自動:
    #: ライブ入力は 4 (main が律速なので、深くしても遅延が増えるだけ)、ファイルは 8。
    slots: int = 0
    #: "auto" (ライブ入力は "when_full"、ファイルは "never") | "never" | "when_full"。
    drop_frames: str = "auto"
    #: 注釈付きの映像をウィンドウに出す (q / Esc で停止)。
    display: bool = False
    #: output.mp4 を書く。
    write_video: bool = True
    #: このフレーム数で止める (動画ファイルとライブ入力)。
    max_frames: Optional[int] = None
    #: この秒数で止める (0 = 制限なし)。
    max_seconds: float = 0.0
    #: シルエット幾何 (analysis/silhouette.py) を geometry プロセスで測る。
    geometry: bool = False
    #: ``SilhouetteParams`` の上書き。
    geometry_params: Dict[str, Any] = field(default_factory=dict)
    #: ``draw_detections`` のキーワード引数 (show_bbox など)。
    draw: Dict[str, Any] = field(default_factory=dict)
    #: 最初のフレームが読めるまで待つ上限 [秒] (子プロセスの起動を含む)。
    startup_timeout: float = 120.0

    def __post_init__(self) -> None:
        if int(self.slots) != 0 and int(self.slots) < 2:
            raise ValueError(f"realtime slots must be 0 (auto) or >= 2, got {self.slots}")
        if self.drop_frames not in ("auto", "never", "when_full"):
            raise ValueError(
                "realtime drop_frames must be 'auto', 'never' or 'when_full', "
                f"got {self.drop_frames!r}"
            )

    def drops(self, live: bool) -> bool:
        """空きスロットが無いときにフレームを捨てるか。"""
        return self.drop_frames == "when_full" or (self.drop_frames == "auto" and live)

    def slot_count(self, live: bool) -> int:
        """実際に使うスロット数 (``slots=0`` を解決する)。"""
        return int(self.slots) or (4 if live else 8)


# --------------------------------------------------------------------------- #
# 共有メモリのフレームリング
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RingSpec:
    """子プロセスへ渡す、リングに接続するための情報 (pickle できる)。"""

    name: str
    slots: int
    frame_shape: Tuple[int, int, int]
    input_shape: Tuple[int, int, int]


def _aligned(nbytes: int, to: int = 64) -> int:
    return (nbytes + to - 1) // to * to


class FrameRing:
    """1 つの共有メモリに、固定数のスロットを並べたもの。

    各スロットは生のフレーム (H0, W0, 3) uint8 と、モデル入力 (H, W, 3) float32 を持つ。
    どのスロットを誰が使っているかはキューで受け渡すスロット番号が決める (ここは
    排他制御をしない)。作成したプロセス (main) だけが ``unlink`` する。
    """

    def __init__(self, shm: shared_memory.SharedMemory, spec: RingSpec, owner: bool):
        self.spec = spec
        self._shm = shm
        self._owner = owner
        frame_bytes = _aligned(int(np.prod(spec.frame_shape)))
        stride = frame_bytes + _aligned(int(np.prod(spec.input_shape)) * 4)
        buf = shm.buf
        self._frames: List[np.ndarray] = [
            np.ndarray(spec.frame_shape, dtype=np.uint8, buffer=buf, offset=k * stride)
            for k in range(spec.slots)
        ]
        self._inputs: List[np.ndarray] = [
            np.ndarray(spec.input_shape, dtype=np.float32, buffer=buf,
                       offset=k * stride + frame_bytes)
            for k in range(spec.slots)
        ]

    @staticmethod
    def nbytes(slots: int, frame_shape: Sequence[int], input_shape: Sequence[int]) -> int:
        """リング全体のバイト数。"""
        frame_bytes = _aligned(int(np.prod(frame_shape)))
        return int(slots) * (frame_bytes + _aligned(int(np.prod(input_shape)) * 4))

    @classmethod
    def create(
        cls, slots: int, frame_shape: Sequence[int], input_shape: Sequence[int]
    ) -> "FrameRing":
        size = cls.nbytes(slots, frame_shape, input_shape)
        shm = shared_memory.SharedMemory(create=True, size=size)
        spec = RingSpec(
            name=shm.name,
            slots=int(slots),
            frame_shape=(int(frame_shape[0]), int(frame_shape[1]), int(frame_shape[2])),
            input_shape=(int(input_shape[0]), int(input_shape[1]), int(input_shape[2])),
        )
        return cls(shm, spec, owner=True)

    @classmethod
    def attach(cls, spec: RingSpec) -> "FrameRing":
        return cls(shared_memory.SharedMemory(name=spec.name), spec, owner=False)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return self.spec.frame_shape

    def frame(self, slot: int) -> np.ndarray:
        """スロットの生フレーム (ビュー。スロットを手放したら使わないこと)。"""
        return self._frames[slot]

    def model_input(self, slot: int) -> np.ndarray:
        """スロットのモデル入力 (ビュー)。"""
        return self._inputs[slot]

    def close(self) -> None:
        self._frames = []
        self._inputs = []
        try:
            self._shm.close()
        except BufferError:
            # どこかがまだビューを持っている。プロセス終了時に OS が解放する。
            pass

    def unlink(self) -> None:
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- #
# main プロセス
# --------------------------------------------------------------------------- #

def run_realtime(
    inferencer: Any,
    source: str,
    output_dir: str,
    *,
    image_size: Tuple[int, int],
    action_names: Sequence[str],
    class_names: Sequence[str],
    meta: Optional[Dict[str, Any]] = None,
    options: Optional[RealtimeOptions] = None,
) -> Dict[str, Any]:
    """動画 / 画像フォルダ / カメラ / ストリームをリアルタイムパイプラインで処理する。

    Args:
        inferencer: ``StreamingInferencer`` (main プロセスで動く。
            ``process_prepared`` と ``runtime_stats`` を使う)
        source: 動画・画像フォルダのパス、カメラ番号 ("0")、ストリーム URL
        output_dir: predictions.json / predictions.jsonl / output.mp4 /
            geometry.csv / runtime_stats.json を書く場所
        image_size: detector 入力 (H, W)。``inferencer.image_size`` と同じにする
        action_names / class_names: 描画用の名前
        meta: predictions.json の ``_meta`` (``yoake predict`` と同じもの)
        options: ``RealtimeOptions``
    Returns:
        runtime_stats.json と同じ dict
    """
    opts = options or RealtimeOptions()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = is_live_source(source)
    if not live and not Path(str(source)).exists():
        raise FileNotFoundError(f"Input not found: {source}")
    # 設定の誤りは子プロセスを起こす前にここで落とす。
    params = SilhouetteParams(**opts.geometry_params) if opts.geometry else None
    drop = opts.drops(live)
    slots = opts.slot_count(live)

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    error_q: Any = ctx.Queue()
    report_q: Any = ctx.Queue()
    meta_q: Any = ctx.Queue()
    frames_q: Any = ctx.Queue()
    free_q: Any = ctx.Queue()
    capture_ctrl: Any = ctx.Queue()
    writer_ctrl: Any = ctx.Queue()
    writer_in: Any = ctx.Queue()
    geometry_ctrl: Any = ctx.Queue() if params is not None else None
    geometry_in: Any = ctx.Queue() if params is not None else None
    post_q = geometry_in if geometry_in is not None else writer_in

    procs: Dict[str, Any] = {
        "capture": ctx.Process(
            target=_capture_main, name="yoake-capture", daemon=True,
            args=(str(source), (int(image_size[0]), int(image_size[1])), opts.max_frames,
                  float(opts.max_seconds), drop, meta_q, capture_ctrl, frames_q, free_q,
                  report_q, error_q, stop),
        ),
    }
    if params is not None:
        procs["geometry"] = ctx.Process(
            target=_geometry_main, name="yoake-geometry", daemon=True,
            args=(geometry_ctrl, geometry_in, writer_in, report_q, error_q, stop),
        )
    procs["writer"] = ctx.Process(
        target=_writer_main, name="yoake-writer", daemon=True,
        args=(writer_ctrl, writer_in, free_q, report_q, error_q, stop, str(out)),
    )
    # 子プロセスの起動 (import) は数秒かかるので、全部を先に起こして重ねる。
    for proc in procs.values():
        proc.start()

    ring: Optional[FrameRing] = None
    stop_forwarded = False
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        info = _receive(meta_q, procs, error_q, timeout=opts.startup_timeout,
                        waiting_for="the first frame")
        height, width = int(info["height"]), int(info["width"])
        ring = FrameRing.create(
            slots, (height, width, 3), (int(image_size[0]), int(image_size[1]), 3)
        )
        for slot in range(slots):
            free_q.put(slot)
        capture_ctrl.put(ring.spec)

        meta_out: Dict[str, Any] = dict(meta or {})
        if not meta_out.get("fps"):
            meta_out["fps"] = float(info["fps"])
        meta_out["realtime"] = {
            "live": live,
            "slots": slots,
            "drop_frames": "when_full" if drop else "never",
            "reported_fps": float(info["fps"]),
            "started_at": started_at,
        }
        if params is not None:
            meta_out["geometry"] = geometry_meta(params)
        writer_ctrl.put((ring.spec, {
            "meta": meta_out,
            "live": live,
            "fps": float(info["fps"]),
            "frame_size": (width, height),
            "action_names": list(action_names),
            "class_names": list(class_names),
            "draw": dict(opts.draw),
            "write_video": bool(opts.write_video),
            "display": bool(opts.display),
        }))
        if geometry_ctrl is not None and params is not None:
            geometry_ctrl.put((ring.spec, float(info["fps"]), asdict(params)))

        processed, interrupted, seconds = _main_loop(
            inferencer, ring, frames_q, post_q, free_q, procs, error_q, stop
        )
        post_q.put(_STOP)
        stop_forwarded = True
        reports = _collect_reports(report_q, procs, error_q)
    finally:
        stop.set()
        if not stop_forwarded:
            # 落ちた段より下流には、手元の結果を書き切らせる。
            post_q.put(_STOP)
        _shutdown(procs.values())
        if ring is not None:
            ring.close()
            ring.unlink()

    capture = reports.get("capture", {})
    writer = reports.get("writer", {})
    stats: Dict[str, Any] = {
        "mode": "realtime",
        "source": str(source),
        "live": live,
        "slots": slots,
        "drop_frames": "when_full" if drop else "never",
        "interrupted": interrupted,
        "frames_read": capture.get("frames_read", 0),
        "frames_dropped": capture.get("frames_dropped", 0),
        "frames_processed": processed,
        "frames_written": writer.get("frames_written", 0),
        "seconds": round(seconds, 3),
        "fps": round(processed / seconds, 2) if seconds > 0 else 0.0,
        "latency_ms": writer.get("latency_ms", {}),
        "inference": inferencer.runtime_stats() if hasattr(inferencer, "runtime_stats") else {},
        "warnings": writer.get("warnings", []),
    }
    if "geometry" in reports:
        stats["geometry"] = reports["geometry"]
    with open(out / "runtime_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def _main_loop(
    inferencer: Any,
    ring: FrameRing,
    frames_q: Any,
    post_q: Any,
    free_q: Any,
    procs: Dict[str, Any],
    error_q: Any,
    stop: Any,
) -> Tuple[int, bool, float]:
    """capture から来たフレームを順に推論して下流へ流す。

    Ctrl+C で止めたときは、それ以降のフレームは推論せずにスロットだけ返し、capture の
    終端の印が来るまで流し切る (writer はそこまでの結果を書く)。

    Returns:
        (推論したフレーム数, 中断したか, 最初のフレームから終わりまでの秒数)
    """
    processed = 0
    interrupted = False
    t_first: Optional[float] = None
    while True:
        try:
            item = frames_q.get(timeout=_POLL)
        except queue.Empty:
            _raise_if_failed(procs, error_q)
            continue
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            continue
        if item is _STOP:
            break
        slot, index, t_capture = item
        if interrupted:
            free_q.put(slot)
            continue
        if t_first is None:
            t_first = time.perf_counter()
        try:
            result = inferencer.process_prepared(
                ring.model_input(slot), ring.frame_shape, frame_index=index
            )
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            free_q.put(slot)
            continue
        post_q.put((slot, index, t_capture, _payload(result)))
        processed += 1
        _raise_if_failed(procs, error_q)
    seconds = time.perf_counter() - t_first if t_first is not None else 0.0
    return processed, interrupted, seconds


def _payload(result: Any) -> Dict[str, Any]:
    """推論結果のうち下流が使う部分 (小さい: 検出数 × 数十バイト)。"""
    return {
        "json": result.to_json(),
        "boxes": np.asarray(result.boxes_xyxy, dtype=np.float32).reshape(-1, 4),
        "scores": np.asarray(result.scores, dtype=np.float32),
        "class_ids": np.asarray(result.class_ids),
        "track_ids": np.asarray(result.track_ids),
        "action_ids": np.asarray(result.action_ids),
        # OBB の隅 (N, 4, 2)。シルエットの ROI と他個体の除外に使う。
        "corners": getattr(result, "corners", None),
    }


def _raise_if_failed(procs: Dict[str, Any], error_q: Any) -> None:
    """どこかの段が例外を報告したか、異常終了していたら RuntimeError。"""
    try:
        name, trace = error_q.get_nowait()
    except queue.Empty:
        pass
    else:
        raise RuntimeError(f"realtime {name} process failed:\n{trace}")
    for name, proc in procs.items():
        if proc.exitcode not in (None, 0):
            raise RuntimeError(f"realtime {name} process exited with code {proc.exitcode}")


def _receive(
    q: Any, procs: Dict[str, Any], error_q: Any, timeout: float, waiting_for: str
) -> Any:
    deadline = time.monotonic() + float(timeout)
    while True:
        try:
            return q.get(timeout=_POLL)
        except queue.Empty:
            _raise_if_failed(procs, error_q)
            if time.monotonic() > deadline:
                raise TimeoutError(f"realtime: gave up waiting for {waiting_for} "
                                   f"after {timeout:.0f} s")


def _collect_reports(report_q: Any, procs: Dict[str, Any], error_q: Any) -> Dict[str, Any]:
    """全段の最終報告を受け取る (writer は結果を書き切ってから報告する)。"""
    reports: Dict[str, Any] = {}
    while set(procs) - set(reports):
        try:
            name, data = report_q.get(timeout=_POLL)
        except queue.Empty:
            _raise_if_failed(procs, error_q)
            missing = set(procs) - set(reports)
            if all(not procs[n].is_alive() for n in missing):
                # 報告は終了直前に書かれる。届くまで少しだけ待つ。
                try:
                    name, data = report_q.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError(
                        f"realtime: {', '.join(sorted(missing))} exited without reporting"
                    ) from None
            else:
                continue
        reports[name] = data
    return reports


def _shutdown(procs: Any) -> None:
    for proc in procs:
        proc.join(timeout=_JOIN_TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# 子プロセス (どれも CUDA に触らない)
# --------------------------------------------------------------------------- #

def _orphaned() -> bool:
    """main プロセスが消えた (強制終了された) か。

    main が正常に終わるなら停止の印と終端の印が必ず届くが、強制終了ではどちらも
    来ない。待つ箇所はどれもこれを見て、親のいないまま残り続けないようにする。
    """
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def _wait_message(q: Any, stop: Any) -> Any:
    """main からの設定を待つ。設定が来る前に停止が要求されたら (か親が消えたら) None。"""
    while not stop.is_set():
        try:
            return q.get(timeout=_POLL)
        except queue.Empty:
            if _orphaned():
                return None
    return None


def _next_item(q: Any) -> Any:
    """上流の次の項目。親プロセスが消えたら待つのをやめる。"""
    while True:
        try:
            return q.get(timeout=0.5)
        except queue.Empty:
            if _orphaned():
                raise SystemExit("the main process is gone")


def _take_slot(free_q: Any, stop: Any, drop: bool) -> Optional[int]:
    """空きスロットを 1 つ取る。drop なら待たずに None、そうでなければ空くまで待つ。"""
    if drop:
        try:
            return int(free_q.get_nowait())
        except queue.Empty:
            return None
    while not stop.is_set():
        try:
            return int(free_q.get(timeout=_POLL))
        except queue.Empty:
            if _orphaned():
                return None
    return None


def _frame_source(
    source: str, max_frames: Optional[int]
) -> Tuple[Iterator[Tuple[int, np.ndarray, float]], bool]:
    """(frame_index, frame_rgb, fps) の列と、ライブ入力かどうか。

    ファイルと画像フォルダは逐次の ``yoake predict`` と同じ読み込み関数を使う
    (同じフレームが出てこそ結果が一致する)。
    """
    if is_live_source(source):
        return _live_frames(source, max_frames), True
    path = Path(source)
    if path.is_dir():
        return read_image_dir_frames(str(path)), False
    return read_video_frames(str(path), max_frames=max_frames), False


def _live_frames(
    source: str, max_frames: Optional[int]
) -> Iterator[Tuple[int, np.ndarray, float]]:
    import cv2

    target: Any = int(source) if str(source).strip().isdigit() else str(source)
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera / stream {source!r}")
    reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = reported if 0.0 < reported < 1000.0 else 30.0
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps
            index += 1
    finally:
        cap.release()


def _capture_main(
    source: str,
    image_size: Tuple[int, int],
    max_frames: Optional[int],
    max_seconds: float,
    drop: bool,
    meta_q: Any,
    ctrl_q: Any,
    frames_q: Any,
    free_q: Any,
    report_q: Any,
    error_q: Any,
    stop: Any,
) -> None:
    """読む → 空きスロットへ生フレームとモデル入力を書く → main へスロット番号を送る。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring: Optional[FrameRing] = None
    read = sent = dropped = 0
    fps = 0.0
    t0 = time.perf_counter()
    try:
        frames, live = _frame_source(source, max_frames)
        first = next(frames, None)
        if first is None:
            raise RuntimeError(f"no frame could be read from {source!r}")
        fps = float(first[2] or 0.0) or 30.0
        height, width = first[1].shape[:2]
        meta_q.put({"height": int(height), "width": int(width), "fps": fps, "live": live})
        spec = _wait_message(ctrl_q, stop)
        if spec is None:
            return
        ring = FrameRing.attach(spec)
        t0 = time.perf_counter()
        item: Optional[Tuple[int, np.ndarray, float]] = first
        while item is not None and not stop.is_set():
            index, rgb, _fps = item
            read += 1
            slot = _take_slot(free_q, stop, drop)
            if slot is None:
                if not drop or _orphaned():
                    break           # 停止要求 (か親の消滅) で待つのをやめた
                dropped += 1        # 下流が詰まっている: このフレームは捨てる
            else:
                if tuple(rgb.shape) != ring.frame_shape:
                    raise ValueError(
                        f"frame {index} is {tuple(rgb.shape)} but the stream started at "
                        f"{ring.frame_shape}; realtime mode needs frames of one size"
                    )
                ring.frame(slot)[...] = rgb
                prepare_input(rgb, image_size, out=ring.model_input(slot))
                frames_q.put((slot, int(index), time.perf_counter()))
                sent += 1
            if max_seconds > 0 and time.perf_counter() - t0 >= max_seconds:
                break
            item = next(frames, None)
    except BaseException:
        error_q.put(("capture", traceback.format_exc()))
        stop.set()
    finally:
        report_q.put(("capture", {
            "frames_read": read,
            "frames_sent": sent,
            "frames_dropped": dropped,
            "source_fps": fps,
            "seconds": round(time.perf_counter() - t0, 3),
        }))
        frames_q.put(_STOP)
        if ring is not None:
            ring.close()


def _geometry_main(
    ctrl_q: Any, in_q: Any, out_q: Any, report_q: Any, error_q: Any, stop: Any
) -> None:
    """シルエット幾何を測って writer へ回す。頭尾の状態を持つので 1 プロセスで順番に。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring: Optional[FrameRing] = None
    analyzer: Optional[SilhouetteAnalyzer] = None
    try:
        message = _wait_message(ctrl_q, stop)
        if message is None:
            return
        spec, fps, params = message
        ring = FrameRing.attach(spec)
        analyzer = SilhouetteAnalyzer(SilhouetteParams(**params), fps=fps)
        while True:
            item = _next_item(in_q)
            if item is _STOP:
                break
            slot, index, t_capture, payload = item
            rows = analyzer.update(
                ring.frame(slot), index, payload["boxes"],
                corners=payload["corners"], track_ids=payload["track_ids"],
            )
            out_q.put((slot, index, t_capture, payload, rows))
    except BaseException:
        error_q.put(("geometry", traceback.format_exc()))
        stop.set()
    finally:
        report_q.put(("geometry", analyzer.timing() if analyzer is not None else {}))
        out_q.put(_STOP)
        if ring is not None:
            ring.close()


def _writer_main(
    ctrl_q: Any, in_q: Any, free_q: Any, report_q: Any, error_q: Any, stop: Any,
    output_dir: str,
) -> None:
    """描画・mp4・predictions.json(l)・geometry.csv を書き、スロットを返す。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring: Optional[FrameRing] = None
    video: Optional[VideoWriter] = None
    table: Optional[GeometryCSV] = None
    lines: Any = None
    display = False
    written = 0
    latencies: List[float] = []
    warnings: List[str] = []
    try:
        message = _wait_message(ctrl_q, stop)
        if message is None:
            return
        spec, cfg = message
        ring = FrameRing.attach(spec)
        out = Path(output_dir)
        meta: Dict[str, Any] = dict(cfg["meta"])
        results: Dict[str, Any] = {"_meta": meta}
        live = bool(cfg["live"])
        display = bool(cfg["display"])
        if "geometry" in meta:
            table = GeometryCSV(str(out / "geometry.csv"))
        # 1 フレーム 1 行。実行中に他のプログラムが追いかけて読め、途中で落ちても残る。
        lines = open(out / "predictions.jsonl", "w", encoding="utf-8")
        first: Optional[Tuple[int, float]] = None
        last: Optional[Tuple[int, float]] = None
        while True:
            item = _next_item(in_q)
            if item is _STOP:
                break
            slot, index, t_capture, payload = item[:4]
            rows = item[4] if len(item) > 4 else None
            try:
                entry = payload["json"]
                if rows is not None:
                    entry["geometry"] = rows
                    if table is not None:
                        table.write(index, payload["track_ids"], rows)
                if first is None:
                    first = (int(index), float(t_capture))
                last = (int(index), float(t_capture))
                if live:
                    # ライブ入力のフレーム番号は「捕まえた順」。実時間はこちら。
                    entry["t"] = round(float(t_capture) - first[1], 4)
                results[str(index)] = entry
                lines.write(json.dumps({"frame": int(index), **entry}) + "\n")
                if cfg["write_video"] or display:
                    vis = draw_detections(
                        ring.frame(slot), payload["boxes"], payload["scores"],
                        payload["class_ids"], payload["track_ids"], payload["action_ids"],
                        cfg["action_names"], cfg["class_names"], **cfg["draw"],
                    )
                    if rows:
                        draw_geometry(vis, rows)
                    if cfg["write_video"]:
                        if video is None:
                            video = VideoWriter(
                                str(out / "output.mp4"), cfg["fps"], tuple(cfg["frame_size"])
                            )
                        video.write(vis)
                    if display:
                        display = _show(vis, stop, warnings)
            finally:
                free_q.put(slot)
            written += 1
            latencies.append(time.perf_counter() - float(t_capture))
            if written % 30 == 0:
                lines.flush()

        realtime_meta = meta.setdefault("realtime", {})
        realtime_meta["frames_written"] = written
        if live and first is not None and last is not None and last[1] > first[1]:
            # ライブ入力はカメラの申告 fps が当てにならない。捕まえた実測で置き換える
            # (bout 解析がフレームを秒に直すのに使う)。
            measured = (last[0] - first[0]) / (last[1] - first[1])
            realtime_meta["measured_fps"] = round(measured, 3)
            meta["fps"] = round(measured, 3)
        with open(out / "predictions.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    except BaseException:
        error_q.put(("writer", traceback.format_exc()))
        stop.set()
    finally:
        report_q.put(("writer", {
            "frames_written": written,
            "latency_ms": _latency_ms(latencies),
            "warnings": warnings,
        }))
        if video is not None:
            video.release()
        if table is not None:
            table.close()
        if lines is not None:
            lines.close()
        if display:
            _close_window()
        if ring is not None:
            ring.close()


def _show(frame_rgb: np.ndarray, stop: Any, warnings: List[str]) -> bool:
    """1 フレーム表示する。表示できない環境なら警告を残して以後は表示しない。"""
    import cv2

    try:
        cv2.imshow(_WINDOW, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        warnings.append(f"display disabled: {exc}")
        return False
    if key in (ord("q"), 27):
        stop.set()
    return True


def _close_window() -> None:
    import cv2

    try:
        cv2.destroyWindow(_WINDOW)
    except cv2.error:
        pass


def _latency_ms(samples: Sequence[float]) -> Dict[str, float]:
    """捕まえてから書き終えるまでの遅延 [ms]。"""
    if not samples:
        return {}
    a = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "max": round(float(a.max()), 2),
    }
