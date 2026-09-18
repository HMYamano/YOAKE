"""
targets.py — 学習用 GT (targets) の形と、その形を揃えるヘルパー

=== なぜ独立したモジュールか ===

YOAKE の DataLoader は stage によって **2 種類の targets** を返す:

* 単フレーム系 (``collate_detection`` / ``collate_single_frame``)::

      targets: BatchFrameTargets = [frame_target, ...]          # 長さ B

* シーケンス系 (``collate_sequence`` / ``collate_scene_sequences``)::

      targets: BatchSequenceTargets = [[frame_target, ...], ...]  # B × T

この 2 つを取り違えると ``'list' object has no attribute 'items'`` で落ちるか、
もっと悪いことに「シーケンスの先頭フレームの GT を全フレームに使う」形で静かに
間違った損失を計算する。どちらの形かを毎回 ``isinstance(targets[0], dict)`` で
その場で判定するコードが複数箇所に散っていたため、判定をここへ集約する。

型エイリアスは実行時の挙動を変えないが、「この List の並びが何を意味するか」を
シグネチャに書けるようにするためのもの。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union, cast

import torch

__all__ = [
    "FrameTargets",
    "SequenceTargets",
    "BatchFrameTargets",
    "BatchSequenceTargets",
    "AnyBatchTargets",
    "is_sequence_targets",
    "last_frame_targets",
    "move_targets_to_device",
]

#: 1 フレーム 1 バッチ要素分の GT。``boxes`` (N, 4) xyxy を必ず持ち、
#: ``class_ids`` / ``track_ids`` / ``action_ids`` / ``keypoints`` などを任意で持つ。
#: 各テンソルの行 i は同じ GT オブジェクト i を指す。
FrameTargets = Dict[str, Any]

#: 1 バッチ要素 (= 1 シーケンス) 分。``[t]`` がフレーム t の GT。
SequenceTargets = List[FrameTargets]

#: 単フレーム学習の 1 バッチ。``[b]`` がバッチ要素 b の GT。
BatchFrameTargets = List[FrameTargets]

#: シーケンス学習の 1 バッチ。``[b][t]`` がバッチ要素 b・フレーム t の GT。
BatchSequenceTargets = List[SequenceTargets]

AnyBatchTargets = Union[BatchFrameTargets, BatchSequenceTargets]


def is_sequence_targets(targets: Sequence[Any]) -> bool:
    """``targets`` が ``BatchSequenceTargets`` (B × T) かどうか。

    空バッチは単フレーム扱い (どちらとしても壊れないので、判定を保留しない)。
    """
    if not targets:
        return False
    return not isinstance(targets[0], dict)


def last_frame_targets(targets: AnyBatchTargets) -> BatchFrameTargets:
    """
    シーケンス targets から**最終フレーム**の GT だけを取り出す。

    YOAKE のシーケンスモデルは「T フレームを入力し、最終フレームについて
    prediction と損失を出す」設計なので、detection / action / ID の各損失が
    見るべき GT は最終フレームのものになる。既に単フレーム形ならそのまま返す。

    Returns:
        長さ B の ``BatchFrameTargets`` — 要素 b がバッチ要素 b の最終フレーム GT
    """
    if not is_sequence_targets(targets):
        return list(cast(BatchFrameTargets, targets))
    out: BatchFrameTargets = []
    for b, seq in enumerate(cast(BatchSequenceTargets, targets)):
        if not seq:
            raise ValueError(
                f"targets[{b}] が空です (シーケンスには最低 1 フレームの GT が必要)"
            )
        out.append(seq[-1])
    return out


def move_targets_to_device(
    targets: AnyBatchTargets, device: torch.device
) -> AnyBatchTargets:
    """
    ``BatchFrameTargets`` / ``BatchSequenceTargets`` のどちらでもデバイスへ移す。

    入れ子の深さは保存する (B × T を渡せば B × T が返る)。
    """
    if is_sequence_targets(targets):
        sequences = cast(BatchSequenceTargets, targets)
        return [[_move_frame(t, device) for t in seq] for seq in sequences]
    return [_move_frame(t, device) for t in cast(BatchFrameTargets, targets)]


def _move_frame(target: FrameTargets, device: torch.device) -> FrameTargets:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in target.items()
    }
