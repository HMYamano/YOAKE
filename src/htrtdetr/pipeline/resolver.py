"""
resolver.py — requires/provides 依存解決 + 実行順序決定

各段が宣言する requires (必要 capability) / provides (付与 capability) / after (明示
順序) と tier (大分類) から、実行順を決めて依存を検証する。

- strict:  requires が満たされない段があれば実行前にエラー (無効な組合せを事前検出)
- lenient: 満たされない段と、それに連鎖して requires を満たせなくなる下流を自動スキップ

順序決定は3種のエッジからトポロジカルソート:
  (a) データ依存: ある段の requires ← 別段の provides → 後者→前者のエッジ
  (b) 明示順序: stage.after に挙がった段名 → その段の後に置くエッジ (ワイルドカード可)
  (c) tier 昇順: 同点解消の第2ソートキー (検出=1 < 静的=2 < 追跡=3 …)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .stage import Stage


class InvalidPipelineError(ValueError):
    """依存が満たせない / 循環がある など、パイプライン構成が無効。"""


# 欠けている capability に対する差し替え示唆 (strict エラーメッセージ用)
_SUGGESTIONS: Dict[str, str] = {
    "dense_features": "L1 を rtdetr(または fused_htrtdetr) にするか、この段を bytetrack など dense 不要のものに差し替える",
    "track_id": "L3_tracking を有効化する (memory_id または bytetrack)",
    "kinematics": "L4_motion を有効化する (kinematic)",
    "boxes": "L1_detection を有効化する",
    "temporal_stable": "L1b_stabilization を有効化する",
    "pose": "pose 段 (keypoint) を有効化する",
    "static_relation": "L2_static を有効化する",
}


def _suggest_backend(missing: Set[str]) -> str:
    hints = [_SUGGESTIONS.get(m, f"'{m}' を供給する段を追加する") for m in sorted(missing)]
    return " / ".join(hints)


def _after_matches(target_name: str, pattern: str) -> bool:
    """stage.after のパターンが target 段名にマッチするか。

    fnmatch による完全な glob (末尾/中間/先頭の `*`, `?` すべて対応)。これにより
    `L1_detection.*` はもちろん `*detection*` のような中間ワイルドカードも機能する
    (旧実装は末尾 `*` のみ対応し、それ以外は黙って no-op になっていた)。
    """
    import fnmatch
    return fnmatch.fnmatchcase(target_name, pattern)


def _build_edges(stages: Sequence[Stage]) -> Dict[int, Set[int]]:
    """依存/明示順のエッジ (producer_idx -> {consumer_idx}) を作る。"""
    edges: Dict[int, Set[int]] = {i: set() for i in range(len(stages))}

    # provides インデックス: capability -> それを提供する段のインデックス集合
    providers: Dict[str, Set[int]] = {}
    for i, st in enumerate(stages):
        for cap in st.provides:
            providers.setdefault(cap, set()).add(i)

    for j, st in enumerate(stages):
        # (a) データ依存エッジ
        for cap in st.requires:
            for i in providers.get(cap, ()):
                if i != j:
                    edges[i].add(j)
        # (b) 明示順序エッジ (after)
        for pattern in st.after:
            for i, other in enumerate(stages):
                if i != j and _after_matches(other.name, pattern):
                    edges[i].add(j)
    return edges


def _topological_sort(stages: Sequence[Stage]) -> List[Stage]:
    """tier を第2ソートキーにした安定トポロジカルソート (Kahn 法)。"""
    n = len(stages)
    edges = _build_edges(stages)

    indeg = {i: 0 for i in range(n)}
    for i in edges:
        for j in edges[i]:
            indeg[j] += 1

    # in-degree 0 の中から (tier, 元の順序) が小さいものを選ぶ
    def _key(i: int) -> Tuple[int, int]:
        return (stages[i].tier, i)

    order: List[int] = []
    ready = sorted((i for i in range(n) if indeg[i] == 0), key=_key)
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(edges[i], key=_key):
            indeg[j] -= 1
            if indeg[j] == 0:
                # 挿入位置を維持するため都度ソート (段数は小さいので許容)
                ready.append(j)
        ready.sort(key=_key)

    if len(order) != n:
        remaining = [stages[i].name for i in range(n) if i not in order]
        raise InvalidPipelineError(
            f"パイプラインに循環依存があります (解決不能: {remaining})"
        )
    return [stages[i] for i in order]


def resolve(
    stages: Sequence[Stage],
    inputs: Set[str],
    mode: str = "strict",
) -> Tuple[List[Stage], List[Tuple[str, Set[str]]]]:
    """
    依存を検証し、実行順に整列する。

    Args:
        stages: 段インスタンス列
        inputs: パイプライン入力として最初から利用可能な capability (例 {"image"})
        mode: "strict" | "lenient"

    Returns:
        (runnable, skipped)
          runnable: 実行順に並んだ段
          skipped:  (段名, 欠けている capability) のリスト (lenient のみ非空)
    """
    if mode not in ("strict", "lenient"):
        raise ValueError(f"mode は 'strict' か 'lenient': {mode!r}")

    order = _topological_sort(stages)

    available: Set[str] = set(inputs)
    runnable: List[Stage] = []
    skipped: List[Tuple[str, Set[str]]] = []

    for st in order:
        missing = set(st.requires) - available
        if missing:
            if mode == "strict":
                raise InvalidPipelineError(
                    f"[{st.name}] は capability {sorted(missing)} を要求しますが、"
                    f"上流が供給していません。対策: {_suggest_backend(missing)}"
                )
            # lenient: 落とす (下流も requires を満たせず連鎖スキップされる)
            skipped.append((st.name, missing))
            continue
        runnable.append(st)
        available |= set(st.provides)

    return runnable, skipped
