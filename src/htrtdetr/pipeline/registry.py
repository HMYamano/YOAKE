"""
registry.py — 段バックエンドの登録・生成

各段の「中身」(rtdetr / yolo / geometric / bytetrack …) を文字列名で登録し、config
から差し替え可能にする。登録時に requires/provides のトークンが既知語彙かを検証し、
打鍵ミス (未知 capability) に早期に気づけるようにする。

使い方:
    @register("L1_detection.rtdetr")
    class RTDETRStage(StageBase):
        ...

    stage = create("L1_detection.rtdetr", score_threshold=0.3)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Type

from .schema import KNOWN_CAPABILITIES
from .stage import Stage, StageBase

# name -> Stage クラス
_REGISTRY: Dict[str, Type[StageBase]] = {}


class UnknownBackendError(KeyError):
    """未登録のバックエンド名を要求したとき。"""


def _validate_capabilities(cls: Type[StageBase]) -> None:
    """requires/provides のトークンが既知語彙かを検証 (未知はタイポ警告)。"""
    for kind in ("requires", "provides"):
        tokens = getattr(cls, kind, frozenset())
        unknown = set(tokens) - KNOWN_CAPABILITIES
        if unknown:
            warnings.warn(
                f"Stage '{getattr(cls, 'name', cls.__name__)}' の {kind} に未知の "
                f"capability {sorted(unknown)} があります (タイポ? "
                f"schema.KNOWN_CAPABILITIES を確認)",
                stacklevel=3,
            )


def register(name: str):
    """段クラスを名前付きで登録するデコレータ。"""
    def _wrap(cls: Type[StageBase]) -> Type[StageBase]:
        # クラスに宣言された name より、デコレータ引数を優先して正規化する
        cls.name = name
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            warnings.warn(
                f"バックエンド '{name}' が再登録されました "
                f"({_REGISTRY[name].__name__} -> {cls.__name__})",
                stacklevel=2,
            )
        _validate_capabilities(cls)
        _REGISTRY[name] = cls
        return cls
    return _wrap


def create(name: str, **params) -> Stage:
    """登録済みバックエンドを生成する。"""
    if name not in _REGISTRY:
        raise UnknownBackendError(
            f"未登録のバックエンド '{name}'。利用可能: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**params)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def get_class(name: str) -> Type[StageBase]:
    if name not in _REGISTRY:
        raise UnknownBackendError(
            f"未登録のバックエンド '{name}'。利用可能: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
