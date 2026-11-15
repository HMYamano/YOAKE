"""
evidence_head.py — 説明可能な行動分類ヘッド (evidence decomposition + 階層/多頭)

=== 説明可能性をモデル構造に組み込む ===

後付けの attention 可視化ではなく、**logit 空間での加法分解**を構造として持つ:

    action_logits = Σ_g  gate_g · head_g(feature_g)
                  = appearance 寄与 + motion 寄与 + interaction 寄与 + ...

各 feature group が独立に logit へ寄与するので、「なぜその行動と判断したか」を
寄与値そのもの (単位が logit で、足すと最終スコアになる) として提示できる。
GUI に出る

    Appearance  +0.32
    Motion      +0.54
    Interaction +1.10

はこの ``evidence`` 出力そのものであり、事後的な近似ではない。

gate は各 group の重要度を学習する (``FeatureGating`` の発展)。**gate 重みと
実際の logit 寄与の両方**を返すので、「重視しているが今回は寄与が小さい」といった
区別もできる。

=== 3 つの分類方式 ===

single       単一 softmax (既存 ActionHead 互換)
hierarchical 粗クラス → 細クラスの木構造。親の確率で子を条件付ける。
multi_head   属性ごとに独立 head (locomotion / social / self-directed) を同時出力。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FeatureGating",
    "EvidenceBehaviorHead",
    "HierarchicalClassifier",
    "MultiHeadClassifier",
    "BehaviorPrediction",
    "segmented_softmax",
]


def segmented_softmax(logits: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
    """softmax inside each consecutive block of ``sizes`` columns.

    multi-head の平坦な logit (head を順に連結したもの) を確率にする。各 head の
    区間の和が 1 になる。全体を 1 つの softmax に通すと head 同士が確率を奪い合い、
    「walk かつ courtship」を表せなくなる (docs/KEYFRAME_BEHAVIOR.md §15)。
    """
    sizes = [int(s) for s in sizes]
    if not sizes or sum(sizes) != logits.shape[-1]:
        raise ValueError(
            f"segmented_softmax: head sizes {sizes} do not add up to {logits.shape[-1]} columns"
        )
    return torch.cat([F.softmax(p, dim=-1) for p in logits.split(sizes, dim=-1)], dim=-1)


class FeatureGating(nn.Module):
    """
    feature group ごとの重要度を学習するゲート。

    既存の ``action_head.FeatureGating`` は「同一次元の feature を重み付き和で
    潰す」ものだったため、どの group がどれだけ効いたかを取り出せなかった。
    ここでは group ごとの**スカラー重み**を返し、説明可能性を保つ。

    Args:
        group_dims: group 名 → 特徴次元
        temperature: gate の鋭さ。小さいほど選択的。
        min_gate: gate の下限。学習初期に特定 group が完全に殺されるのを防ぐ。
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        temperature: float = 1.0,
        min_gate: float = 0.05,
    ):
        super().__init__()
        self.group_names: List[str] = list(group_dims.keys())
        self.temperature = float(temperature)
        self.min_gate = float(min_gate)

        total = sum(group_dims.values())
        self.gate_net = nn.Sequential(
            nn.Linear(total, max(total // 4, len(self.group_names))),
            nn.ReLU(inplace=True),
            nn.Linear(max(total // 4, len(self.group_names)), len(self.group_names)),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: group 名 → (N, D_g)
        Returns:
            gates: (N, num_groups) ∈ [min_gate, 1]、順序は ``self.group_names``
        """
        present = [features[g] for g in self.group_names]
        x = torch.cat(present, dim=-1)
        raw = self.gate_net(x) / self.temperature
        gates = torch.sigmoid(raw)
        return self.min_gate + (1.0 - self.min_gate) * gates


class HierarchicalClassifier(nn.Module):
    """
    階層的な行動分類。

    例::

        hierarchy = {
            "_root":  ["Immobile", "Locomotion", "Active"],
            "Active": ["Self-directed", "Social"],
            "Self-directed": ["grooming", "feeding"],
            "Social": ["courtship", "chase", "aggression"],
        }

    各内部ノードに softmax 分類器を置き、葉の確率は根から葉までの**条件付き確率の積**
    にする。粗い区別 (動いているか否か) は学習しやすく、細かい区別 (courtship か
    chase か) は該当サンプルだけで学習されるので、クラス不均衡に強くなる。
    """

    def __init__(
        self,
        in_dim: int,
        hierarchy: Dict[str, List[str]],
        leaf_names: Sequence[str],
        root_key: str = "_root",
    ):
        super().__init__()
        self.hierarchy = {k: list(v) for k, v in hierarchy.items()}
        self.root_key = root_key
        self.leaf_names = list(leaf_names)
        if root_key not in self.hierarchy:
            raise ValueError(
                f"hierarchy に根ノード '{root_key}' がありません。"
                f"存在するキー: {sorted(self.hierarchy)}"
            )

        # 内部ノードごとに分類器を作る
        self.node_names: List[str] = list(self.hierarchy.keys())
        self.classifiers = nn.ModuleDict({
            self._safe(name): nn.Linear(in_dim, len(children))
            for name, children in self.hierarchy.items()
        })
        # 葉 → 根からの経路 [(ノード名, 子index), ...]
        self._paths: Dict[str, List[Tuple[str, int]]] = {}
        for leaf in self.leaf_names:
            path = self._find_path(root_key, leaf, [])
            if path is None:
                raise ValueError(
                    f"葉 '{leaf}' が hierarchy から到達できません "
                    f"(root='{root_key}')"
                )
            self._paths[leaf] = path

    @staticmethod
    def _safe(name: str) -> str:
        """ModuleDict のキーに使える形へ (ドットが使えないため)。"""
        return name.replace(".", "_")

    def _find_path(
        self, node: str, target: str, acc: List[Tuple[str, int]]
    ) -> Optional[List[Tuple[str, int]]]:
        children = self.hierarchy.get(node)
        if children is None:
            return None
        for i, child in enumerate(children):
            step = acc + [(node, i)]
            if child == target:
                return step
            if child in self.hierarchy:
                found = self._find_path(child, target, step)
                if found is not None:
                    return found
        return None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (N, in_dim)
        Returns:
            leaf_logits: (N, num_leaves) — 葉の log 確率 (条件付き確率の積の対数)
            node_logits: ノード名 → (N, num_children)
        """
        node_logits: Dict[str, torch.Tensor] = {}
        node_logprob: Dict[str, torch.Tensor] = {}
        for name in self.node_names:
            logits = self.classifiers[self._safe(name)](x)
            node_logits[name] = logits
            node_logprob[name] = F.log_softmax(logits, dim=-1)

        leaf_cols = []
        for leaf in self.leaf_names:
            # 経路上の条件付き対数確率を足す = 確率の積
            lp = None
            for node, child_idx in self._paths[leaf]:
                term = node_logprob[node][:, child_idx]
                lp = term if lp is None else lp + term
            leaf_cols.append(lp)

        return {
            "leaf_logits": torch.stack(leaf_cols, dim=-1),
            "node_logits": node_logits,
        }


class MultiHeadClassifier(nn.Module):
    """
    複数の行動属性を同時に出力する分類器。

    例::

        heads = {
            "locomotion":    ["stationary", "walking", "running"],
            "self_directed": ["none", "grooming", "feeding"],
            "social":        ["none", "courtship", "chase", "aggression"],
        }

    「歩きながら求愛する」のように属性は同時に成立しうるので、単一 softmax では
    表現できない組合せを扱える。各 head は独立した softmax として学習・評価する。
    """

    def __init__(self, in_dim: int, heads: Dict[str, List[str]]):
        super().__init__()
        if not heads:
            raise ValueError("MultiHeadClassifier には 1 つ以上の head が必要です")
        self.head_names: List[str] = list(heads.keys())
        self.head_classes: Dict[str, List[str]] = {k: list(v) for k, v in heads.items()}
        self.classifiers = nn.ModuleDict({
            name: nn.Linear(in_dim, len(classes))
            for name, classes in self.head_classes.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns: head 名 → (N, num_classes_of_head)"""
        return {name: clf(x) for name, clf in self.classifiers.items()}


class BehaviorPrediction:
    """
    行動認識の出力をまとめる軽量コンテナ。

    Attributes:
        logits: (N, num_actions) — single / hierarchical の最終 logit
        probs: (N, num_actions)
        evidence: group 名 → (N, num_actions) の**logit 寄与**。
            足し合わせると logits になる (説明の加法性が保証される)。
        gates: group 名 → (N,) の gate 重み
        head_logits: multi_head 時の head 名 → (N, C_head)
        node_logits: hierarchical 時の内部ノード logit
        head_sizes: multi_head 時の各 head のクラス数 (head_logits と同じ順)。
            multi_head でも ``logits`` / ``probs`` は埋まる: logits は head の logit を
            順に連結したもの、probs は head ごとの softmax の連結 (各区間の和が 1)。
    """

    __slots__ = (
        "logits", "probs", "evidence", "gates", "head_logits", "node_logits", "head_sizes",
    )

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        gates: Optional[Dict[str, torch.Tensor]] = None,
        head_logits: Optional[Dict[str, torch.Tensor]] = None,
        node_logits: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.logits = logits
        self.probs = probs
        self.evidence = evidence or {}
        self.gates = gates or {}
        self.head_logits = head_logits or {}
        self.node_logits = node_logits or {}
        self.head_sizes: Tuple[int, ...] = tuple(
            int(v.shape[-1]) for v in self.head_logits.values()
        )
        if self.head_logits and self.logits is None:
            # multi_head: 平坦な形も持たせる (検出順への並べ替え・保存・説明が
            # single と同じ経路で済む)。
            self.logits = torch.cat(list(self.head_logits.values()), dim=-1)
        if self.head_logits and self.probs is None and self.logits is not None:
            self.probs = segmented_softmax(self.logits, self.head_sizes)

    def explain(
        self,
        index: int,
        action_names: Optional[Sequence[str]] = None,
        action_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        1 個体の判断根拠を {group 名: logit 寄与} で返す。

        Args:
            index: バッチ内のインデックス
            action_names: 表示用の行動名 (未使用でも API の一貫性のため受ける)
            action_id: 説明対象の行動。None なら予測クラス。
        Returns:
            {"appearance": 0.32, "motion": 0.54, ...} — 値は logit 単位で加算的
        """
        if self.logits is None or not self.evidence:
            return {}
        if action_id is not None:
            aid = action_id
        elif self.head_sizes:
            # multi_head: 既定は最初の head の最尤クラス (平坦な argmax は head を跨ぐ)
            aid = int(self.logits[index, : self.head_sizes[0]].argmax())
        else:
            aid = int(self.logits[index].argmax())
        return {
            group: float(contrib[index, aid].detach())
            for group, contrib in self.evidence.items()
        }

    def gate_weights(self, index: int) -> Dict[str, float]:
        """1 個体の gate 重みを返す (重要度。寄与とは別物)。"""
        return {g: float(w[index].detach()) for g, w in self.gates.items()}


class EvidenceBehaviorHead(nn.Module):
    """
    evidence 分解つき行動分類ヘッド。

    各 feature group を個別に符号化 → group ごとの分類器で logit を出し、
    gate で重み付けして**加算**する。加算なので寄与がそのまま説明になる。

    Args:
        group_dims: group 名 → 入力次元
        num_actions: 行動クラス数 (single / hierarchical の葉の数)
        hidden_dim: group ごとの中間次元
        classifier_type: "single" | "hierarchical" | "multi_head"
        use_evidence: False なら全 group を concat して単一 MLP に通す従来構成
        use_gating: gate を使うか
        hierarchy / heads: それぞれの分類方式の定義
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        num_actions: int,
        hidden_dim: int = 256,
        classifier_type: str = "single",
        use_evidence: bool = True,
        use_gating: bool = True,
        dropout: float = 0.1,
        hierarchy: Optional[Dict[str, List[str]]] = None,
        heads: Optional[Dict[str, List[str]]] = None,
        action_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if not group_dims:
            raise ValueError("EvidenceBehaviorHead には 1 つ以上の feature group が必要です")

        self.group_names: List[str] = list(group_dims.keys())
        self.num_actions = int(num_actions)
        self.classifier_type = classifier_type
        self.use_evidence = bool(use_evidence)
        self.use_gating = bool(use_gating and use_evidence)
        self.action_names = list(action_names) if action_names else [
            str(i) for i in range(num_actions)
        ]

        # group ごとの符号化器 (共通次元 hidden_dim へ揃える)
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for name, dim in group_dims.items()
        })

        if self.use_gating:
            self.gating = FeatureGating({n: hidden_dim for n in self.group_names})

        if self.use_evidence:
            # group ごとに独立した分類器 → logit 空間で加算 (加法的説明の本体)
            self.group_classifiers = nn.ModuleDict({
                name: self._make_classifier(hidden_dim, hierarchy, heads)
                for name in self.group_names
            })
        else:
            # 従来構成: concat → 単一分類器 (説明は得られない)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * len(self.group_names), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = self._make_classifier(hidden_dim, hierarchy, heads)

    def _make_classifier(
        self,
        in_dim: int,
        hierarchy: Optional[Dict[str, List[str]]],
        heads: Optional[Dict[str, List[str]]],
    ) -> nn.Module:
        if self.classifier_type == "single":
            return nn.Linear(in_dim, self.num_actions)
        if self.classifier_type == "hierarchical":
            if not hierarchy:
                raise ValueError("classifier_type='hierarchical' には hierarchy が必要です")
            return HierarchicalClassifier(in_dim, hierarchy, self.action_names)
        if self.classifier_type == "multi_head":
            if not heads:
                raise ValueError("classifier_type='multi_head' には heads が必要です")
            return MultiHeadClassifier(in_dim, heads)
        raise ValueError(f"未知の classifier_type: {self.classifier_type!r}")

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(self, features: Dict[str, torch.Tensor]) -> BehaviorPrediction:
        """
        Args:
            features: group 名 → (N, D_g)。``group_dims`` のキーを全て含むこと。
        Returns:
            BehaviorPrediction
        """
        missing = set(self.group_names) - set(features)
        if missing:
            raise KeyError(
                f"feature group が不足しています: {sorted(missing)} "
                f"(期待: {self.group_names})"
            )

        encoded = {name: self.encoders[name](features[name]) for name in self.group_names}

        if not self.use_evidence:
            x = self.fusion(torch.cat([encoded[n] for n in self.group_names], dim=-1))
            return self._wrap_output(self.classifier(x), evidence=None, gates=None)

        gates = None
        if self.use_gating:
            g = self.gating(encoded)                       # (N, num_groups)
            gates = {name: g[:, i] for i, name in enumerate(self.group_names)}

        # group ごとの寄与を計算し、logit 空間で足す
        contributions: Dict[str, torch.Tensor] = {}
        for name in self.group_names:
            out = self.group_classifiers[name](encoded[name])
            if self.classifier_type == "multi_head":
                # 各 head ごとに寄与を保持する
                contributions[name] = {
                    h: (v * gates[name].unsqueeze(-1) if gates else v)
                    for h, v in out.items()
                }
            else:
                logits = out["leaf_logits"] if isinstance(out, dict) else out
                if gates is not None:
                    logits = logits * gates[name].unsqueeze(-1)
                contributions[name] = logits

        return self._combine(contributions, gates)

    def _combine(self, contributions, gates) -> BehaviorPrediction:
        """group ごとの寄与を合算して最終出力にする。"""
        if self.classifier_type == "multi_head":
            head_names = list(next(iter(contributions.values())).keys())
            head_logits = {
                h: sum(contributions[g][h] for g in self.group_names)
                for h in head_names
            }
            # 説明用: head ごと・group ごとの寄与を平坦化して保持
            evidence = {
                f"{g}": torch.cat([contributions[g][h] for h in head_names], dim=-1)
                for g in self.group_names
            }
            return BehaviorPrediction(
                logits=None, probs=None, evidence=evidence,
                gates=gates, head_logits=head_logits,
            )

        total = sum(contributions[g] for g in self.group_names)
        return BehaviorPrediction(
            logits=total,
            probs=F.softmax(total, dim=-1),
            evidence=contributions,
            gates=gates,
        )

    def _wrap_output(self, out, evidence, gates) -> BehaviorPrediction:
        """evidence 無効時 (従来構成) の出力整形。"""
        if self.classifier_type == "multi_head":
            return BehaviorPrediction(head_logits=out, gates=gates)
        if isinstance(out, dict):     # hierarchical
            return BehaviorPrediction(
                logits=out["leaf_logits"],
                probs=F.softmax(out["leaf_logits"], dim=-1),
                node_logits=out["node_logits"],
                gates=gates,
            )
        return BehaviorPrediction(
            logits=out, probs=F.softmax(out, dim=-1), gates=gates
        )
