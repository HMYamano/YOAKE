"""
video_source.py — 動画の読み出し (デコード位置の管理) だけを担う層

``annotator.py`` と ``gui/analysis_panel.py`` は、どちらも「cv2 の capture を持ち、
指定フレームを読み、可能なら seek せず逐次デコードする」という同じ処理を別々に
持っていた。片方だけ直すと挙動が分かれるので、ここへ 1 つにまとめる。

dearpygui にも torch にも依存しない。capture を引数で受け取り状態を返す純関数に
してあるので、GUI を起動せずに単体テストできる (呼び出し側が ``self.cap`` を
monkeypatch する既存テストもそのまま動く)。

=== なぜ逐次デコードなのか ===

1 フレーム進める操作はアノテーション中で最も多い。``cap.set(CAP_PROP_POS_FRAMES)``
は long-GOP コーデックだと直前のキーフレームから復号し直すので、Right キーを
押し続けると同じ GOP を何度も舐めることになる。歩幅ではなく GOP 長に比例して
待たされるため、この tool が対象とするような長い動画ほど重くなる。

さらに fps が非整数の動画では ``cap.set`` が隣のフレームに着地することがある。
frame 0 では一致するので気付きにくいが、逐次デコードならこのズレ自体が起きない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

__all__ = [
    "SEQ_SKIP_MAX",
    "VideoProbe",
    "open_capture",
    "probe_capture",
    "read_frame_sequential",
]

#: これ以下の前方ジャンプなら ``grab()`` で読み飛ばす。これを超えると seek の方が
#: 安い (grab も 1 フレームずつ復号は進めるため)。
SEQ_SKIP_MAX = 12


@dataclass(frozen=True)
class VideoProbe:
    """capture から読み出した動画の基本情報。"""

    fps: float
    width: int
    height: int
    num_frames: int

    @property
    def last_index(self) -> int:
        """最終フレームの index (フレーム数 0 なら 0)。"""
        return max(self.num_frames - 1, 0)


def open_capture(path: Path) -> Optional[Any]:
    """動画を開く。開けなければ ``None`` (例外にはしない)。"""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def probe_capture(cap: Any, *, default_fps: float = 25.0) -> VideoProbe:
    """capture から fps / 解像度 / フレーム数を読む。

    fps が 0 や NaN で返るコンテナがあるので、その場合だけ ``default_fps`` を使う。
    """
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # 0 / None / NaN
        fps = default_fps
    return VideoProbe(
        fps=float(fps),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def read_frame_sequential(
    cap: Optional[Any],
    idx: int,
    decoded_idx: Optional[int],
    *,
    seq_skip_max: int = SEQ_SKIP_MAX,
) -> Tuple[Optional[Any], Optional[int]]:
    """
    フレーム ``idx`` を BGR で読む。可能な限り seek しない。

    分岐は 3 通り:

    * ``idx == decoded_idx + 1`` — seek も grab もせず ``read()`` だけ。
    * 短い前方ジャンプ (``seq_skip_max`` 以内) — 差分ぶん ``grab()`` してから読む。
      grab は捨てるフレームを復号しないので、seek より安い。
    * それ以外 (後方 / 大きなジャンプ / 位置不明) — 素直に seek する。

    Args:
        cap: cv2.VideoCapture 互換オブジェクト。``None`` なら ``(None, None)``。
        idx: 読みたいフレーム番号
        decoded_idx: **直前に実際にデコードしたフレーム番号**。位置が分からない
            ときは ``None`` を渡す (その場合は必ず seek する)。

    Returns:
        ``(frame_bgr, new_decoded_idx)``。読めなければ ``(None, None)`` を返す。
        読み出しに失敗すると capture の位置は当てにできなくなるので、
        ``new_decoded_idx`` は ``None`` にして次回必ず seek させる。
    """
    import cv2

    if cap is None:
        return None, None

    here = decoded_idx
    if here is not None and here < idx <= here + seq_skip_max:
        for _ in range(idx - here - 1):
            if not cap.grab():
                here = None
                break
    else:
        here = None

    if here is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()
    if not ret:
        return None, None
    return frame, idx
