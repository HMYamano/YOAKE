"""
track_buffer.py — track ごとの特徴リングバッファ (streaming 推論の心臓部)

=== なぜこれが必要か ===

旧実装は 16 フレーム窓を 1 フレームずつずらすたびに、窓内の**全フレームを再び
detector へ通していた** (1-16, 2-17, 3-18, ...)。同じフレームが最大 16 回
detector を通るので、空間推論のコストが窓長ぶん丸ごと重複する。

本モジュールは「フレームは一度しか detector に通さない」を成立させる:

    new frame → detector (1回) → detection → tracking → track_id
              → TrackFeatureCache に push → HTM は cache の履歴を読むだけ

履歴は **track_id 単位** で持つ。DETR の query index はフレーム間で同一個体を
保証しないので、query 単位の履歴は「別個体の特徴を時系列として連結する」ことに
なり、時間モデルの前提が壊れる。track 単位なら履歴は常に同一個体のものになる。

=== 欠損・消失の扱い ===

* 検出が無いフレームは ``push_missing`` でマスク付きの穴として記録し、HTM 側は
  マスクを見て補間/ゼロ埋めを選べる (無言で前フレームを複製しない)。
* ``max_age`` を超えて観測が無い track は非アクティブになる。
* ``reid_max_age`` までは特徴を保持し、再出現時に同じ ID を復元できる (re-ID)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

__all__ = ["TrackState", "TrackFeatureCache"]


@dataclass
class TrackState:
    """
    1 つの track の履歴リングバッファ。

    ``window`` 本の固定長テンソルを確保し、``_pos`` を進めながら上書きする。
    毎フレーム新しいテンソルを alloc すると streaming の実時間性が崩れるため、
    確保は track 生成時の 1 回だけにする。

    Attributes:
        track_id: トラック ID
        window: 履歴長 (frame 数)
        feat_dim: 特徴次元
        age: 最後に観測されてから経過したフレーム数 (0 = 今フレームで観測)
        hits: 観測された累計フレーム数
        first_frame / last_frame: 観測フレーム番号の範囲
    """
    track_id: int
    window: int
    feat_dim: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    age: int = 0
    hits: int = 0
    first_frame: int = -1
    last_frame: int = -1

    # リングバッファ本体 (__post_init__ で確保)
    _feats: torch.Tensor = field(init=False, repr=False)
    _obb: torch.Tensor = field(init=False, repr=False)
    _valid: torch.Tensor = field(init=False, repr=False)
    _frames: torch.Tensor = field(init=False, repr=False)
    _pos: int = field(default=0, init=False, repr=False)
    _filled: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._feats = torch.zeros(
            self.window, self.feat_dim, device=self.device, dtype=self.dtype
        )
        # [cx, cy, w, h, theta] 正規化座標
        self._obb = torch.zeros(self.window, 5, device=self.device, dtype=self.dtype)
        self._valid = torch.zeros(self.window, device=self.device, dtype=torch.bool)
        self._frames = torch.full(
            (self.window,), -1, device=self.device, dtype=torch.long
        )

    # ------------------------------------------------------------------ #
    # 書き込み
    # ------------------------------------------------------------------ #

    def push(
        self,
        feature: torch.Tensor,
        obb: torch.Tensor,
        frame_index: int,
    ) -> None:
        """観測を 1 フレーム分追加する。"""
        self._feats[self._pos] = feature.detach().to(self.dtype)
        self._obb[self._pos] = obb.detach().to(self.dtype)
        self._valid[self._pos] = True
        self._frames[self._pos] = frame_index
        self._advance(frame_index)
        self.age = 0
        self.hits += 1
        if self.first_frame < 0:
            self.first_frame = frame_index

    def push_missing(self, frame_index: int) -> None:
        """
        観測が無かったフレームを「穴」として記録する。

        特徴を前フレームで埋めない (無言の複製は動きを止まって見せ、行動認識を
        誤らせる)。valid=False を立てて下流にマスクとして渡す。
        """
        self._feats[self._pos] = 0
        self._obb[self._pos] = 0
        self._valid[self._pos] = False
        self._frames[self._pos] = frame_index
        self._advance(frame_index)
        self.age += 1

    def _advance(self, frame_index: int) -> None:
        self._pos = (self._pos + 1) % self.window
        self._filled = min(self._filled + 1, self.window)
        self.last_frame = frame_index

    # ------------------------------------------------------------------ #
    # 読み出し
    # ------------------------------------------------------------------ #

    def history(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        時系列順 (古い → 新しい) に並べ直した履歴を返す。

        Returns:
            feats: (window, feat_dim) — 未充填の先頭はゼロ padding
            obb:   (window, 5)
            valid: (window,) bool — True の位置だけが実観測
        """
        # _pos は「次に書く位置」= 最古のスロット。そこから巡回させれば時系列順になる。
        idx = (torch.arange(self.window, device=self.device) + self._pos) % self.window
        return self._feats[idx], self._obb[idx], self._valid[idx]

    def last_obb(self) -> Optional[torch.Tensor]:
        """最後に観測された OBB (5,)。未観測なら None。"""
        if self.hits == 0:
            return None
        pos = (self._pos - 1) % self.window
        for _ in range(self.window):
            if bool(self._valid[pos]):
                return self._obb[pos]
            pos = (pos - 1) % self.window
        return None

    def last_feature(self) -> Optional[torch.Tensor]:
        """最後に観測された特徴 (feat_dim,)。未観測なら None。"""
        if self.hits == 0:
            return None
        pos = (self._pos - 1) % self.window
        for _ in range(self.window):
            if bool(self._valid[pos]):
                return self._feats[pos]
            pos = (pos - 1) % self.window
        return None

    def velocity(self) -> torch.Tensor:
        """直近 2 観測から中心速度 [dx, dy] を推定する (観測 1 つ以下ならゼロ)。"""
        _, obb, valid = self.history()
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return torch.zeros(2, device=self.device, dtype=self.dtype)
        last, prev = idx[-1], idx[-2]
        dt = max(int(last - prev), 1)
        return (obb[last, :2] - obb[prev, :2]) / dt


class TrackFeatureCache:
    """
    全 track のリングバッファをまとめて管理する。

    streaming 推論と学習の双方から使える (学習時は窓を一括で push してから
    ``stacked_history`` で取り出せば、offline と同じ track-aligned な履歴になる)。

    Args:
        window: track ごとの履歴長
        feat_dim: 特徴次元
        max_age: これを超えて未観測の track は非アクティブになる
        reid_max_age: 特徴を保持し再同定候補に残す上限 (max_age 以上)
        device / dtype: バッファの確保先

    Example::

        cache = TrackFeatureCache(window=16, feat_dim=256, device=dev)
        cache.update(frame_index=t, track_ids=[3, 7], features=feats, obbs=obbs)
        hist, obb, mask, ids = cache.stacked_history()   # (N, T, D), (N, T, 5), (N, T)
    """

    def __init__(
        self,
        window: int = 16,
        feat_dim: int = 256,
        max_age: int = 30,
        reid_max_age: int = 90,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if reid_max_age < max_age:
            raise ValueError(
                f"reid_max_age ({reid_max_age}) must be >= max_age ({max_age})"
            )
        self.window = int(window)
        self.feat_dim = int(feat_dim)
        self.max_age = int(max_age)
        self.reid_max_age = int(reid_max_age)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._tracks: Dict[int, TrackState] = {}
        self._frame_index: int = -1

    # ------------------------------------------------------------------ #
    # ライフサイクル
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """全 track を破棄する (新しい動画の開始時に呼ぶ)。"""
        self._tracks.clear()
        self._frame_index = -1

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, track_id: int) -> bool:
        return int(track_id) in self._tracks

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(int(track_id))

    def active_ids(self) -> List[int]:
        """max_age 以内に観測された track の ID (昇順)。"""
        return sorted(t.track_id for t in self._tracks.values() if t.age <= self.max_age)

    def reid_ids(self) -> List[int]:
        """再同定候補として残っている (= max_age 超過だが保持中の) track の ID。"""
        return sorted(
            t.track_id for t in self._tracks.values()
            if self.max_age < t.age <= self.reid_max_age
        )

    # ------------------------------------------------------------------ #
    # 更新
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_index: int,
        track_ids: Iterable[int],
        features: torch.Tensor,
        obbs: torch.Tensor,
    ) -> None:
        """
        1 フレーム分の観測を取り込む。

        観測されなかった既存 track には自動で ``push_missing`` を入れるので、
        全 track のバッファが同じフレーム軸で揃う (下流で時間の整合を取る必要がない)。

        Args:
            frame_index: フレーム番号
            track_ids: 各検出の track ID (長さ N)
            features: (N, feat_dim)
            obbs: (N, 5) [cx, cy, w, h, theta] 正規化座標
        """
        ids = [int(t) for t in track_ids]
        if features.shape[0] != len(ids) or obbs.shape[0] != len(ids):
            raise ValueError(
                f"track_ids ({len(ids)}), features ({features.shape[0]}), "
                f"obbs ({obbs.shape[0]}) の件数が一致しません"
            )
        if features.shape[0] and features.shape[1] != self.feat_dim:
            raise ValueError(
                f"feature dim mismatch: cache={self.feat_dim}, got {features.shape[1]}"
            )

        observed = set(ids)
        for i, tid in enumerate(ids):
            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(
                    track_id=tid, window=self.window, feat_dim=self.feat_dim,
                    device=self.device, dtype=self.dtype,
                )
                self._tracks[tid] = state
            state.push(features[i], obbs[i], frame_index)

        # 未観測 track は穴を開けて時間軸を揃える
        for tid, state in self._tracks.items():
            if tid not in observed:
                state.push_missing(frame_index)

        self._frame_index = frame_index
        self._evict()

    def _evict(self) -> None:
        """reid_max_age を超えた track を破棄する。"""
        dead = [tid for tid, s in self._tracks.items() if s.age > self.reid_max_age]
        for tid in dead:
            del self._tracks[tid]

    # ------------------------------------------------------------------ #
    # 読み出し
    # ------------------------------------------------------------------ #

    def stacked_history(
        self,
        track_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        指定 track (既定はアクティブ全件) の履歴を (N, T, ...) に積んで返す。

        これが HTM への入力になる。**時間軸は track に沿って揃っている**ので、
        HTM は「同一個体の時系列」だけを見る。

        Returns:
            feats: (N, T, feat_dim)
            obbs:  (N, T, 5)
            mask:  (N, T) bool — True が実観測
            ids:   長さ N の track_id リスト (feats の行に対応)
        """
        ids = track_ids if track_ids is not None else self.active_ids()
        if not ids:
            empty = torch.zeros(0, self.window, self.feat_dim,
                                device=self.device, dtype=self.dtype)
            return (
                empty,
                torch.zeros(0, self.window, 5, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.window, device=self.device, dtype=torch.bool),
                [],
            )

        feats, obbs, masks = [], [], []
        kept: List[int] = []
        for tid in ids:
            state = self._tracks.get(int(tid))
            if state is None:
                continue
            f, o, v = state.history()
            feats.append(f)
            obbs.append(o)
            masks.append(v)
            kept.append(int(tid))

        return (
            torch.stack(feats, dim=0),
            torch.stack(obbs, dim=0),
            torch.stack(masks, dim=0),
            kept,
        )

    def velocities(self, track_ids: Optional[List[int]] = None) -> torch.Tensor:
        """指定 track の中心速度 (N, 2)。"""
        ids = track_ids if track_ids is not None else self.active_ids()
        if not ids:
            return torch.zeros(0, 2, device=self.device, dtype=self.dtype)
        return torch.stack([
            self._tracks[int(t)].velocity() if int(t) in self._tracks
            else torch.zeros(2, device=self.device, dtype=self.dtype)
            for t in ids
        ], dim=0)

    def stats(self) -> Dict[str, int]:
        """デバッグ・ログ用の要約。"""
        return {
            "num_tracks": len(self._tracks),
            "num_active": len(self.active_ids()),
            "num_reid_candidates": len(self.reid_ids()),
            "frame_index": self._frame_index,
        }
