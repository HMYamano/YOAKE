"""
assembly.py — 窓内の検出列を track-aligned な履歴テンソルへ組み立てる

学習と推論で**同じ組み立て方**を使うための共通経路。

    per-frame detections + track_id  →  (M, T, D) の track 単位履歴 + mask

学習時は GT の track_id を使い (teacher forcing)、推論時は tracker が付けた
track_id を使う。どちらも同じ ``TrackFeatureCache`` を通すので、学習時と推論時で
時間軸の意味がズレない。

なぜ teacher forcing にするか: 学習初期の tracker は誤りが多く、そのまま使うと
時間モデルが「別個体の混在した履歴」を正解として学んでしまう。GT が使える学習時は
GT で揃え、tracker の誤りは評価指標 (IDF1/HOTA) 側で測るのが素直な分離になる。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .track_buffer import TrackFeatureCache

__all__ = ["FrameDetections", "build_track_aligned_history", "TrackAlignedBatch"]


class FrameDetections:
    """1 フレーム分の検出 (track 組み立ての入力単位)。"""

    __slots__ = ("features", "obb", "track_ids")

    def __init__(
        self,
        features: torch.Tensor,   # (N, D)
        obb: torch.Tensor,        # (N, 5)
        track_ids: Sequence[int],  # (N,)
    ):
        if features.shape[0] != obb.shape[0] or features.shape[0] != len(track_ids):
            raise ValueError(
                f"件数不一致: features={features.shape[0]}, obb={obb.shape[0]}, "
                f"track_ids={len(track_ids)}"
            )
        self.features = features
        self.obb = obb
        self.track_ids = [int(t) for t in track_ids]


class TrackAlignedBatch:
    """
    ``build_track_aligned_history`` の戻り値。

    Attributes:
        features: (M, T, D) — track ごとの特徴履歴 (時系列順)
        obb: (M, T, 5)
        mask: (M, T) bool — True が実観測
        track_ids: 長さ M の track_id
        batch_index: (M,) 各 track が属するバッチ要素
        current_rows: 最終フレームの検出 i → history の行番号。
            行動予測を元の検出順に戻すために使う。-1 は該当なし。
    """

    __slots__ = ("features", "obb", "mask", "track_ids", "batch_index", "current_rows")

    def __init__(self, features, obb, mask, track_ids, batch_index, current_rows):
        self.features = features
        self.obb = obb
        self.mask = mask
        self.track_ids = track_ids
        self.batch_index = batch_index
        self.current_rows = current_rows

    def __len__(self) -> int:
        return len(self.track_ids)

    @property
    def num_tracks(self) -> int:
        return len(self.track_ids)


def build_track_aligned_history(
    sequences: List[List[FrameDetections]],
    window: int,
    feat_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> TrackAlignedBatch:
    """
    バッチ内の各シーケンスについて track-aligned 履歴を組み立てる。

    Args:
        sequences: バッチ要素ごとのフレーム列。``sequences[b][t]`` が
            バッチ要素 b・フレーム t の検出。
        window: 履歴長 (通常は T と同じ)
        feat_dim: 特徴次元
    Returns:
        TrackAlignedBatch — 全バッチ要素の track を 1 つに積んだもの

    Note:
        track_id はバッチ要素をまたいで衝突しうる (別動画の track 3 同士など)。
        ``batch_index`` で区別されるので、interaction 特徴の近傍探索などは
        必ず batch_index でグループ分けすること。
    """
    device = device or torch.device("cpu")

    all_feats: List[torch.Tensor] = []
    all_obb: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    all_ids: List[int] = []
    all_bidx: List[int] = []
    current_rows: List[List[int]] = []

    for b, frames in enumerate(sequences):
        # 窓が空のバッチ要素はスキップ (下流で行が生えない)
        cache = TrackFeatureCache(
            window=window, feat_dim=feat_dim,
            # 窓内では track を捨てない (履歴を丸ごと使いたい)
            max_age=window, reid_max_age=window,
            device=device, dtype=dtype,
        )
        for t, fd in enumerate(frames):
            cache.update(
                frame_index=t,
                track_ids=fd.track_ids,
                features=fd.features.to(device=device, dtype=dtype),
                obbs=fd.obb.to(device=device, dtype=dtype),
            )

        feats, obb, mask, ids = cache.stacked_history()
        row_of: Dict[int, int] = {tid: len(all_ids) + i for i, tid in enumerate(ids)}

        if ids:
            all_feats.append(feats)
            all_obb.append(obb)
            all_mask.append(mask)
            all_ids.extend(ids)
            all_bidx.extend([b] * len(ids))

        # 最終フレームの検出 → 行番号
        last = frames[-1] if frames else None
        rows = (
            [row_of.get(tid, -1) for tid in last.track_ids] if last is not None else []
        )
        current_rows.append(rows)

    if not all_ids:
        empty_f = torch.zeros(0, window, feat_dim, device=device, dtype=dtype)
        return TrackAlignedBatch(
            features=empty_f,
            obb=torch.zeros(0, window, 5, device=device, dtype=dtype),
            mask=torch.zeros(0, window, device=device, dtype=torch.bool),
            track_ids=[],
            batch_index=torch.zeros(0, dtype=torch.long, device=device),
            current_rows=current_rows,
        )

    return TrackAlignedBatch(
        features=torch.cat(all_feats, dim=0),
        obb=torch.cat(all_obb, dim=0),
        mask=torch.cat(all_mask, dim=0),
        track_ids=all_ids,
        batch_index=torch.tensor(all_bidx, dtype=torch.long, device=device),
        current_rows=current_rows,
    )
