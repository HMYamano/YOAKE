"""
stage.py — Stage プロトコル / StageBase / PipelineContext

per-frame段 (L1,L2) と temporal段 (L3,L4,L5) を1つの契約で扱う。処理単位は窓
(FrameWindow) に統一され、per-frame段は窓のフレームに map するだけ。オンライン状態
(L3 の GRU 等) は `ctx` に持ち回る。

なぜ `tier` / `after` を足すか:
  requires/provides によるデータ依存の整合は「新しい capability を足す段」には効くが、
  L1b の検出安定化のように `boxes` を消費して**同じ `boxes` をきれいにして返す**段では
  効かない (新 capability は temporal_stable のみ)。純粋な providers→requires グラフでは
  「L1 の後・L2/L3 の前」という位置が一意に決まらない。そこで大分類の `tier`
  (検出=1 < 静的=2 < 追跡=3 …) を第2ソートキーに、細かい前後は `after` (段名指定) で
  補う。build系の順序エッジ (systemd の After= 等) と同じ振る舞い。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Protocol, runtime_checkable

from .schema import FrameWindow


# 大分類 tier の名前付き定数 (可読性のため)
TIER_DETECTION = 1
TIER_STATIC = 2
TIER_TRACKING = 3
TIER_MOTION = 4
TIER_INTERACTION = 5


@dataclass
class PipelineContext:
    """段間で共有する状態・設定。オンライン処理の持ち回りに使う。"""
    device: str = "cuda"
    fps: Optional[float] = None
    window_size: int = 16
    stride: int = 8
    # 段が自由に使える永続状態 (例: {"tracker_memory": ..., "_fused_cache": ...})
    state: Dict = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """段の共通インタフェース (構造的型)。"""
    name: str
    tier: int
    requires: FrozenSet[str]
    provides: FrozenSet[str]
    after: FrozenSet[str]

    def setup(self, ctx: PipelineContext) -> None: ...
    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow: ...
    def reset_state(self, ctx: PipelineContext) -> None: ...
    def teardown(self) -> None: ...


class StageBase:
    """
    共通実装。tier/requires/provides/after はクラス属性で宣言する。

    サブクラスは最低限 `process()` を実装する。`setup()`/`teardown()` は任意。
    """
    name: str = "base"
    tier: int = 0
    requires: FrozenSet[str] = frozenset()
    provides: FrozenSet[str] = frozenset()
    after: FrozenSet[str] = frozenset()

    def setup(self, ctx: PipelineContext) -> None:  # noqa: D401 - optional hook
        """モデルロード等の初期化 (任意)。"""
        return None

    def reset_state(self, ctx: PipelineContext) -> None:
        """
        run() 境界 (=1動画ごと) で呼ばれる状態リセット (任意)。

        トラッカのメモリや GRU など「動画をまたいで持ち越してはいけない」状態を持つ段は
        ここでリセットする。窓をまたぐ状態は保持される (per-window ではなく per-run)。
        """
        return None

    def teardown(self) -> None:
        """リソース解放 (任意)。"""
        return None

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        raise NotImplementedError(
            f"Stage '{self.name}' は process() を実装していません"
        )

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return (
            f"<{type(self).__name__} name={self.name!r} tier={self.tier} "
            f"requires={sorted(self.requires)} provides={sorted(self.provides)}>"
        )
