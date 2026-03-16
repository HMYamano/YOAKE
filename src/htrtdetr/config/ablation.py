"""
ablation.py — Ablation Study Configuration & Results Manager

ablation 実験を config ベースで定義・実行し、結果を CSV に自動集計する。

設計:
  - AblationVariant: 1つの実験変体（name + overrides dict）
  - AblationSuite:   複数 variant を束ねたグループ
  - ResultsCollector: 実験結果を受け取り CSV / JSON に集計

使い方:
  from htrtdetr.config.ablation import (
      AblationVariant, AblationSuite, ResultsCollector,
      TEMPORAL_BRANCH_SUITE, CLIP_LENGTH_SUITE, get_suite,
  )

  suite = get_suite("temporal_branch")
  for variant in suite.variants:
      cfg = variant.apply(base_cfg)
      # ... train / eval ...
      suite.collector.add(variant.name, results_dict)

  suite.collector.save("outputs/ablation/temporal_branch.csv")
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """1 つの ablation 変体。

    Parameters
    ----------
    name:
        実験識別子 (CSV の行 key になる)
    description:
        人間が読める説明
    overrides:
        HTRTDETRConfig.merge() に渡す dict
    tags:
        検索・フィルタ用タグ
    """
    name: str
    description: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def apply(self, base_cfg):  # type: ignore[return]
        """base_cfg に override を適用して新しい config を返す。"""
        return base_cfg.merge(self.overrides)


@dataclass
class AblationSuite:
    """複数の AblationVariant をまとめたグループ。"""
    name: str
    description: str = ""
    variants: List[AblationVariant] = field(default_factory=list)
    _collector: Optional["ResultsCollector"] = field(default=None, init=False, repr=False)

    @property
    def collector(self) -> "ResultsCollector":
        if self._collector is None:
            self._collector = ResultsCollector(suite_name=self.name)
        return self._collector

    def get(self, name: str) -> Optional[AblationVariant]:
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def filter_by_tag(self, tag: str) -> List[AblationVariant]:
        return [v for v in self.variants if tag in v.tags]


# ---------------------------------------------------------------------------
# Results Collector
# ---------------------------------------------------------------------------

class ResultsCollector:
    """実験結果を収集し CSV / JSON に出力する。

    Usage
    -----
    collector = ResultsCollector("temporal_branch")
    collector.add("short_only",  {"id_accuracy": 0.85, "action_f1": 0.72})
    collector.add("short+mid",   {"id_accuracy": 0.88, "action_f1": 0.75})
    collector.save("outputs/ablation/temporal_branch.csv")
    """

    def __init__(self, suite_name: str = "ablation") -> None:
        self.suite_name = suite_name
        self._rows: List[Dict[str, Any]] = []

    def add(self, variant_name: str, metrics: Dict[str, Any]) -> None:
        """実験結果を1行追加する。"""
        row = {"variant": variant_name}
        row.update(metrics)
        self._rows.append(row)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def save(self, path: str | Path) -> None:
        """CSV と JSON の両方を保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # CSV
        if self._rows:
            all_keys: List[str] = ["variant"]
            for row in self._rows:
                for k in row:
                    if k not in all_keys:
                        all_keys.append(k)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._rows)

        # JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"suite": self.suite_name, "results": self._rows},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Ablation results saved:")
        print(f"  CSV : {path}")
        print(f"  JSON: {json_path}")

    def print_table(self) -> None:
        """結果をターミナルに表形式で表示する。"""
        if not self._rows:
            print("(no results)")
            return
        all_keys: List[str] = ["variant"]
        for row in self._rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)

        col_widths = {k: max(len(k), 8) for k in all_keys}
        for row in self._rows:
            for k, v in row.items():
                col_widths[k] = max(col_widths.get(k, 8), len(f"{v:.4f}" if isinstance(v, float) else str(v)))

        header = " | ".join(f"{k:<{col_widths[k]}}" for k in all_keys)
        sep = "-+-".join("-" * col_widths[k] for k in all_keys)
        print(f"\n=== {self.suite_name} ===")
        print(header)
        print(sep)
        for row in self._rows:
            cells = []
            for k in all_keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    cells.append(f"{v:<{col_widths[k]}.4f}")
                else:
                    cells.append(f"{str(v):<{col_widths[k]}}")
            print(" | ".join(cells))


# ---------------------------------------------------------------------------
# Pre-defined Ablation Suites
# ---------------------------------------------------------------------------

# --- Temporal branch combinations ---
TEMPORAL_BRANCH_SUITE = AblationSuite(
    name="temporal_branch",
    description="Which HTM branches to use (short / mid / long)",
    variants=[
        AblationVariant(
            name="short_only",
            description="Short branch only (dilation=1, frames=3)",
            overrides={},  # runtime: pass use_short=True, use_mid=False, use_long=False
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="mid_only",
            description="Mid branch only (dilation=2, frames=8)",
            overrides={},
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="long_only",
            description="Long branch only (dilation=4, frames=16)",
            overrides={},
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="short+mid",
            description="Short + Mid branches",
            overrides={},
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="short+long",
            description="Short + Long branches",
            overrides={},
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="mid+long",
            description="Mid + Long branches",
            overrides={},
            tags=["branch", "ablation"],
        ),
        AblationVariant(
            name="all_branches",
            description="All three branches (default)",
            overrides={},
            tags=["branch", "ablation", "default"],
        ),
    ],
)

# --- Dilation schedule ---
DILATION_SUITE = AblationSuite(
    name="dilation",
    description="Dilation schedule for HTM branches",
    variants=[
        AblationVariant(
            name="dil_1_2_4",
            description="Dilation [1, 2, 4] (default)",
            overrides={
                "model": {
                    "temporal": {
                        "short_branch": {"dilation": 1},
                        "mid_branch": {"dilation": 2},
                        "long_branch": {"dilation": 4},
                    }
                }
            },
            tags=["dilation", "ablation", "default"],
        ),
        AblationVariant(
            name="dil_1_4_8",
            description="Dilation [1, 4, 8]",
            overrides={
                "model": {
                    "temporal": {
                        "short_branch": {"dilation": 1},
                        "mid_branch": {"dilation": 4},
                        "long_branch": {"dilation": 8},
                    }
                }
            },
            tags=["dilation", "ablation"],
        ),
        AblationVariant(
            name="dil_2_4_8",
            description="Dilation [2, 4, 8]",
            overrides={
                "model": {
                    "temporal": {
                        "short_branch": {"dilation": 2},
                        "mid_branch": {"dilation": 4},
                        "long_branch": {"dilation": 8},
                    }
                }
            },
            tags=["dilation", "ablation"],
        ),
        AblationVariant(
            name="dil_1_1_1",
            description="No dilation (baseline conv)",
            overrides={
                "model": {
                    "temporal": {
                        "short_branch": {"dilation": 1},
                        "mid_branch": {"dilation": 1},
                        "long_branch": {"dilation": 1},
                    }
                }
            },
            tags=["dilation", "ablation", "baseline"],
        ),
    ],
)

# --- Clip length ---
CLIP_LENGTH_SUITE = AblationSuite(
    name="clip_length",
    description="Temporal window size for sequence input",
    variants=[
        AblationVariant(
            name="clip_4",
            description="Window size = 4",
            overrides={"data": {"window_size": 4}},
            tags=["clip", "ablation"],
        ),
        AblationVariant(
            name="clip_8",
            description="Window size = 8",
            overrides={"data": {"window_size": 8}},
            tags=["clip", "ablation"],
        ),
        AblationVariant(
            name="clip_16",
            description="Window size = 16 (default)",
            overrides={"data": {"window_size": 16}},
            tags=["clip", "ablation", "default"],
        ),
        AblationVariant(
            name="clip_32",
            description="Window size = 32",
            overrides={"data": {"window_size": 32}},
            tags=["clip", "ablation"],
        ),
    ],
)

# --- Interaction feature ---
INTERACTION_SUITE = AblationSuite(
    name="interaction",
    description="Interaction feature components",
    variants=[
        AblationVariant(
            name="no_interaction",
            description="No interaction features",
            overrides={
                "model": {
                    "action_head": {
                        "use_interaction": False,
                        "use_nn_distance": False,
                        "use_relative_angle": False,
                        "use_relative_velocity": False,
                        "use_overlap": False,
                    }
                }
            },
            tags=["interaction", "ablation", "baseline"],
        ),
        AblationVariant(
            name="distance_only",
            description="NN distance only",
            overrides={
                "model": {
                    "action_head": {
                        "use_interaction": True,
                        "use_nn_distance": True,
                        "use_relative_angle": False,
                        "use_relative_velocity": False,
                        "use_overlap": False,
                    }
                }
            },
            tags=["interaction", "ablation"],
        ),
        AblationVariant(
            name="distance+angle",
            description="NN distance + relative angle",
            overrides={
                "model": {
                    "action_head": {
                        "use_interaction": True,
                        "use_nn_distance": True,
                        "use_relative_angle": True,
                        "use_relative_velocity": False,
                        "use_overlap": False,
                    }
                }
            },
            tags=["interaction", "ablation"],
        ),
        AblationVariant(
            name="full_interaction",
            description="All interaction features (default)",
            overrides={
                "model": {
                    "action_head": {
                        "use_interaction": True,
                        "use_nn_distance": True,
                        "use_relative_angle": True,
                        "use_relative_velocity": True,
                        "use_overlap": True,
                    }
                }
            },
            tags=["interaction", "ablation", "default"],
        ),
    ],
)

# --- Memory module ---
MEMORY_SUITE = AblationSuite(
    name="memory",
    description="Memory ID head configuration",
    variants=[
        AblationVariant(
            name="no_memory",
            description="No GRU memory (single-frame embedding)",
            overrides={"model": {"id_head": {"memory_dim": 0}}},
            tags=["memory", "ablation", "baseline"],
        ),
        AblationVariant(
            name="memory_64",
            description="GRU hidden dim = 64",
            overrides={"model": {"id_head": {"memory_dim": 64}}},
            tags=["memory", "ablation"],
        ),
        AblationVariant(
            name="memory_128",
            description="GRU hidden dim = 128",
            overrides={"model": {"id_head": {"memory_dim": 128}}},
            tags=["memory", "ablation"],
        ),
        AblationVariant(
            name="memory_256",
            description="GRU hidden dim = 256 (default)",
            overrides={"model": {"id_head": {"memory_dim": 256}}},
            tags=["memory", "ablation", "default"],
        ),
        AblationVariant(
            name="memory_512",
            description="GRU hidden dim = 512",
            overrides={"model": {"id_head": {"memory_dim": 512}}},
            tags=["memory", "ablation"],
        ),
    ],
)

# --- Fusion method ---
FUSION_SUITE = AblationSuite(
    name="fusion",
    description="HTM branch fusion method",
    variants=[
        AblationVariant(
            name="fusion_concat_proj",
            description="Concatenate + linear projection (default)",
            overrides={"model": {"temporal": {"fusion_method": "concat_proj"}}},
            tags=["fusion", "ablation", "default"],
        ),
        AblationVariant(
            name="fusion_sum",
            description="Element-wise sum",
            overrides={"model": {"temporal": {"fusion_method": "sum"}}},
            tags=["fusion", "ablation"],
        ),
        AblationVariant(
            name="fusion_attention",
            description="Learned attention weighting",
            overrides={"model": {"temporal": {"fusion_method": "attention"}}},
            tags=["fusion", "ablation"],
        ),
    ],
)


# Registry of all suites
_SUITE_REGISTRY: Dict[str, AblationSuite] = {
    "temporal_branch": TEMPORAL_BRANCH_SUITE,
    "dilation": DILATION_SUITE,
    "clip_length": CLIP_LENGTH_SUITE,
    "interaction": INTERACTION_SUITE,
    "memory": MEMORY_SUITE,
    "fusion": FUSION_SUITE,
}


def get_suite(name: str) -> AblationSuite:
    """名前で ablation suite を取得する。"""
    if name not in _SUITE_REGISTRY:
        available = ", ".join(_SUITE_REGISTRY.keys())
        raise ValueError(f"Unknown suite: {name!r}. Available: {available}")
    return _SUITE_REGISTRY[name]


def list_suites() -> List[str]:
    """利用可能な suite 名一覧を返す。"""
    return list(_SUITE_REGISTRY.keys())


def register_suite(suite: AblationSuite) -> None:
    """カスタム suite を registry に登録する。"""
    _SUITE_REGISTRY[suite.name] = suite
