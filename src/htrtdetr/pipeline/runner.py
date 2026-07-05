"""
runner.py — PipelineRunner: 依存解決済みの段列を窓単位で実行する

- 構築時に resolver で実行順を決め、strict/lenient を適用
- 段は窓 (FrameWindow) を順に process する。窓の `available` は入力 capability から
  始まり、各段の provides を足していく (null 判定ではなく集合で「実行済み」を表す)
- オンライン状態 (トラッカのメモリ等) は PipelineContext.state に持ち回る

オフライン解析が主対象なので、既定は「1動画=1窓」で処理し、窓をまたぐ縫合
(track_id 引き継ぎ等) の複雑さを避ける。長尺でメモリが厳しい場合は
window_size/stride を指定して sliding window で流せる。
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, Set

from .resolver import resolve
from .schema import CAP_IMAGE, FrameResult, FrameWindow
from .stage import PipelineContext, Stage


class PipelineRunner:
    def __init__(
        self,
        stages: Sequence[Stage],
        inputs: Optional[Set[str]] = None,
        mode: str = "strict",
        ctx: Optional[PipelineContext] = None,
    ):
        self.inputs: Set[str] = set(inputs) if inputs is not None else {CAP_IMAGE}
        self.ctx: PipelineContext = ctx or PipelineContext()
        self.runnable, self.skipped = resolve(stages, self.inputs, mode)
        self._setup_done = False

    # ------------------------------------------------------------------ #
    # ライフサイクル
    # ------------------------------------------------------------------ #
    def setup(self) -> "PipelineRunner":
        if not self._setup_done:
            for st in self.runnable:
                st.setup(self.ctx)
            self._setup_done = True
        return self

    def reset_state(self) -> None:
        """run() 境界 (=1動画ごと) で各段の状態をリセットする。"""
        for st in self.runnable:
            fn = getattr(st, "reset_state", None)
            if callable(fn):
                fn(self.ctx)

    def teardown(self) -> None:
        for st in self.runnable:
            try:
                st.teardown()
            except Exception:  # pragma: no cover - best effort
                pass
        self._setup_done = False

    def __enter__(self) -> "PipelineRunner":
        return self.setup()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()

    # ------------------------------------------------------------------ #
    # 実行
    # ------------------------------------------------------------------ #
    def process_window(self, window: FrameWindow) -> FrameWindow:
        """1つの窓を全段に通す。"""
        self.setup()
        window.available |= self.inputs
        for st in self.runnable:
            window = st.process(window, self.ctx)
            window.available |= set(st.provides)
        return window

    def run(
        self,
        frames: Iterable[FrameResult],
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> Iterator[FrameWindow]:
        """
        フレーム列を窓に切って順に処理し、処理済み窓を yield する。

        既定は「1動画=1窓」(全フレームを1つの FrameWindow で処理)。これはオフライン解析の
        既定であり、fused など内部で sliding window を持つ段はその内部バッファ
        (ctx.window_size) が Inferencer と同じローリングを再現する。**外側の分割** はここでは
        既定で行わない (ctx.window_size は外側分割には使わない — それは段の内部バッファ用)。

        長尺でメモリが厳しい場合のみ window_size を明示指定する。その場合、状態を持つ段
        (トラッカ/GRU) の二重消費を避けるため、外側の分割は**非重複** (stride>=window_size) に
        強制する (重複窓の縫合は未実装)。
        """
        frame_list: List[FrameResult] = list(frames)
        if not frame_list:
            return

        self.setup()
        self.reset_state()   # per-run (=per-video) の状態リセット

        # 外側分割: 既定は単一窓。ctx.window_size はここでは参照しない。
        if window_size is None or window_size <= 0 or window_size >= len(frame_list):
            yield self.process_window(FrameWindow(frames=frame_list))
            return

        # 明示指定時: 非重複に強制 (状態を持つ段の二重消費を防ぐ)
        st = stride if stride is not None else window_size
        st = max(window_size, max(1, st))
        for start in range(0, len(frame_list), st):
            chunk = frame_list[start:start + window_size]
            if not chunk:
                break
            yield self.process_window(FrameWindow(frames=chunk))
            if start + window_size >= len(frame_list):
                break

    def describe(self) -> str:
        """構成の要約 (実行順・スキップ) を返す。"""
        lines = ["PipelineRunner:"]
        lines.append(f"  inputs: {sorted(self.inputs)}")
        lines.append("  order:")
        for i, s in enumerate(self.runnable):
            lines.append(
                f"    {i+1}. {s.name} (tier={s.tier}) "
                f"req={sorted(s.requires)} -> prov={sorted(s.provides)}"
            )
        if self.skipped:
            lines.append("  skipped (lenient):")
            for name, missing in self.skipped:
                lines.append(f"    - {name}: missing {sorted(missing)}")
        return "\n".join(lines)
