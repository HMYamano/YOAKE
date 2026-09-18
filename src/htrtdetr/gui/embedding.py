"""
embedding.py — a frame's appearance as a vector, from whatever can produce one.

Smart Sampling needs to know how frames differ. It does **not** need to know
what is in them: diversity sampling wants a metric that separates
different-looking frames, and novelty wants a distance from what is already
annotated. Neither is a classification problem, and pretending otherwise is how
a sampling feature acquires a 2 GB dependency it did not need.

So the interface is a vector and a dimension, and nothing else:

.. code-block:: python

    provider = provider_for(client)          # sidecar, or the built-in
    info = provider.info()                   # name, dimension, device
    vectors = provider.encode_frames(read_frame, [0, 15, 30, ...])

Three providers, one protocol
-----------------------------

=====================  ====================================================
:class:`SidecarProvider`   A real encoder in the helper process — DINOv2 by
                           way of ``transformers``, ``torch.hub``, or a
                           torchvision fallback. Never imported here.
:class:`ThumbnailProvider` The built-in 16x16 descriptor from
                           :mod:`htrtdetr.annotation.features`. No
                           dependencies, always available, and good enough to
                           separate layouts on a fixed camera.
:class:`NullProvider`      Encodes nothing. Sampling then runs on motion and
                           scene statistics alone.
=====================  ====================================================

:func:`provider_for` picks the best one that can actually run **right now**,
and every consumer takes whatever it gets: ``features.scan(embeddings=...)``
and ``features.distance()`` work on a 384-dimensional DINOv2 vector and on a
256-dimensional thumbnail identically. Swapping the encoder changes no caller.

Purity: this module imports neither torch nor dearpygui. It talks to the helper
through :mod:`htrtdetr.gui.sidecar`, which owns the process boundary.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Optional,
                    Sequence, Tuple)

__all__ = [
    "EmbeddingInfo", "EmbeddingProvider", "ThumbnailProvider", "NullProvider",
    "SidecarProvider", "provider_for", "DEFAULT_BATCH",
]

#: Frames per request to the helper. Large enough to amortise the round trip,
#: small enough that a Cancel is honoured within a second or two.
DEFAULT_BATCH = 16

#: How long to wait for one batch before giving up on the helper and falling
#: back. A cold DINOv2 on a CPU takes tens of seconds for the *first* batch
#: because the weights load then; later batches are far quicker.
FIRST_BATCH_TIMEOUT = 180.0
BATCH_TIMEOUT = 90.0


@dataclass(frozen=True)
class EmbeddingInfo:
    """What a provider is, for the log and for the cache key.

    :attr:`model_name` and :attr:`embedding_dimension` both go into the cache
    key: vectors from two different encoders are not interchangeable, and a
    cache that returned one for the other would silently produce a nonsense
    selection.
    """

    model_name: str = "none"
    embedding_dimension: int = 0
    device: str = "cpu"
    batch_size: int = DEFAULT_BATCH
    #: Free-text, shown in the UI: ``DINOv2 small (transformers) on cuda``.
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.embedding_dimension > 0

    def describe(self) -> str:
        if not self.available:
            return "No visual encoder"
        return self.detail or (f"{self.model_name} "
                               f"({self.embedding_dimension}-d) on {self.device}")


class EmbeddingProvider:
    """The interface. Model-agnostic by construction.

    A subclass implements :meth:`info` and :meth:`encode_frames`; nothing else
    is part of the contract, and nothing about the model leaks into the return
    value beyond its dimension.
    """

    name = "provider"

    def info(self) -> EmbeddingInfo:
        raise NotImplementedError

    def encode_frames(
        self,
        read_frame: Callable[[int], Any],
        frames: Sequence[int],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[int, Tuple[float, ...]]:
        """``{frame_index: vector}``, L2-normalised, for the frames it managed.

        A frame that could not be read or encoded is **absent** from the result
        rather than present as zeros: a zero vector sits at distance 1 from
        everything and diversity sampling would pick it first.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release anything held. Safe to call on a provider that holds nothing."""


class NullProvider(EmbeddingProvider):
    """Encodes nothing. Sampling falls back to motion and scene statistics."""

    name = "none"

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(model_name="none", embedding_dimension=0,
                            detail="No visual encoder")

    def encode_frames(self, read_frame, frames, *, should_stop=None,
                      on_progress=None) -> Dict[int, Tuple[float, ...]]:
        return {}


class ThumbnailProvider(EmbeddingProvider):
    """The built-in 16x16 grayscale descriptor. Always available.

    Not semantic — it cannot tell a grooming fly from a walking one — but
    diversity sampling does not need semantics. For a fixed camera over a fixed
    arena, coarse spatial layout is exactly the signal that separates
    different-looking frames, and it costs about a millisecond a frame.
    """

    name = "thumbnail"

    def __init__(self, size: int = 0) -> None:
        from ..annotation import features
        self._size = int(size or features.DESCRIPTOR_SIZE)

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model_name="thumbnail",
            embedding_dimension=self._size * self._size,
            device="cpu",
            detail=f"Built-in {self._size}x{self._size} frame descriptor")

    def encode_frames(self, read_frame, frames, *, should_stop=None,
                      on_progress=None) -> Dict[int, Tuple[float, ...]]:
        from ..annotation import features

        out: Dict[int, Tuple[float, ...]] = {}
        total = len(frames)
        for done, index in enumerate(frames, start=1):
            if should_stop is not None and should_stop():
                break
            frame = read_frame(int(index))
            if frame is not None:
                vector = features.descriptor(frame, self._size)
                if vector:
                    out[int(index)] = vector
            if on_progress is not None:
                on_progress(done, total)
        return out


class SidecarProvider(EmbeddingProvider):
    """A real encoder, in the helper process.

    The frames go out through the filesystem, exactly as video propagation's
    window does and for the same reason: a batch of sixteen 1024x1024 frames is
    50 MB and a newline-delimited JSON pipe is the wrong transport for it. Only
    the vectors come back.

    **Falls back rather than failing.** If a batch errors or times out, this
    provider gives up on the helper and returns what it has; the caller then
    sees a partial mapping and :func:`htrtdetr.annotation.features.scan` fills
    the rest with the thumbnail descriptor. A sampling run must not be lost
    because an encoder hiccuped on frame 900.
    """

    name = "sidecar"

    def __init__(self, client: Any, *, batch_size: int = DEFAULT_BATCH,
                 scratch: Optional[Path] = None) -> None:
        self.client = client
        self.batch_size = max(1, int(batch_size))
        self._scratch = Path(scratch) if scratch else None
        hello = getattr(client, "hello", None) or {}
        self._backend = str(hello.get("backend") or "helper")
        self._device = str(hello.get("device") or "cpu")
        self._model = ""
        self._dimension = 0
        self._loader = ""
        self._failed = ""

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model_name=self._model or "visual encoder",
            embedding_dimension=self._dimension,
            device=self._device,
            batch_size=self.batch_size,
            detail=(f"{self._model or 'visual encoder'}"
                    f"{f' ({self._loader})' if self._loader else ''}"
                    f" on {self._device}") if self._dimension else
                   (self._failed or "Encoder not yet contacted"))

    @property
    def failed(self) -> str:
        """Why this provider stopped, or ``""`` while it is still working."""
        return self._failed

    def encode_frames(self, read_frame, frames, *, should_stop=None,
                      on_progress=None) -> Dict[int, Tuple[float, ...]]:
        out: Dict[int, Tuple[float, ...]] = {}
        total = len(frames)
        done = 0
        first_batch = True

        for start in range(0, total, self.batch_size):
            if should_stop is not None and should_stop():
                break
            chunk = [int(f) for f in frames[start:start + self.batch_size]]
            if not chunk:
                continue

            folder, written = _export_list(read_frame, chunk, self._scratch)
            if folder is None or written == 0:
                break

            try:
                request = self.client.embed_frames(
                    frames_dir=str(folder), first_frame=0,
                    batch_size=self.batch_size)
                reply = self.client.wait_for(
                    request,
                    timeout=FIRST_BATCH_TIMEOUT if first_batch else BATCH_TIMEOUT)
            except Exception as exc:                    # noqa: BLE001
                self._failed = f"{type(exc).__name__}: {exc}"
                shutil.rmtree(folder, ignore_errors=True)
                break
            finally:
                first_batch = False

            shutil.rmtree(folder, ignore_errors=True)

            if reply is None:
                self._failed = ("The encoder did not answer in time. Sampling "
                                "continued with the built-in descriptor.")
                break
            if not reply.get("ok"):
                self._failed = str(reply.get("detail")
                                   or "The encoder refused the request.")
                break

            self._model = str(reply.get("model") or self._model)
            self._loader = str(reply.get("loader") or self._loader)
            self._dimension = int(reply.get("dimension") or self._dimension)

            for item in reply.get("items") or ():
                try:
                    offset = int(item["frame"])
                    vector = tuple(float(v) for v in item["vector"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= offset < len(chunk) and vector:
                    out[chunk[offset]] = vector

            done += len(chunk)
            if on_progress is not None:
                on_progress(min(done, total), total)

        return out


def _export_list(read_frame: Callable[[int], Any], frames: Sequence[int],
                 scratch: Optional[Path]) -> Tuple[Optional[Path], int]:
    """Write an arbitrary list of frames as ``00000.jpg`` upward.

    Separate from ``video_segment.export_window`` because that one exports a
    contiguous range and stops at the first hole - correct for propagation,
    where a hole shifts every later offset. Here the frames are a sparse scan
    and a missing one must simply be skipped, with the *position* preserved so
    the reply can be mapped back.
    """
    try:
        import cv2
    except ImportError:
        return None, 0
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="yoake_embed_",
                                   dir=str(scratch) if scratch else None))
    written = 0
    try:
        for offset, index in enumerate(frames):
            frame = read_frame(int(index))
            if frame is None:
                continue
            cv2.imwrite(str(folder / f"{offset:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
    except Exception:                                   # noqa: BLE001
        shutil.rmtree(folder, ignore_errors=True)
        return None, 0
    if written == 0:
        shutil.rmtree(folder, ignore_errors=True)
        return None, 0
    return folder, written


def provider_for(client: Any = None, *, prefer_sidecar: bool = True,
                 batch_size: int = DEFAULT_BATCH,
                 scratch: Optional[Path] = None) -> EmbeddingProvider:
    """The best provider that can actually run right now.

    Gated on the helper's advertised ``embedding`` capability — never on the
    backend's name and never on a guess. A helper that is running but cannot
    encode falls through to the thumbnail descriptor, silently and correctly:
    the caller gets vectors either way and does not branch.
    """
    if prefer_sidecar and client is not None:
        try:
            running = client.is_running()
        except Exception:                               # noqa: BLE001
            running = False
        hello = getattr(client, "hello", None) or {}
        if running and "embedding" in tuple(hello.get("capabilities") or ()):
            return SidecarProvider(client, batch_size=batch_size,
                                   scratch=scratch)
    try:
        return ThumbnailProvider()
    except Exception:                                   # noqa: BLE001
        return NullProvider()
