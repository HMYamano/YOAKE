"""
cache.py — computing a 30-minute video's features once instead of every time.

Scanning a long video for embeddings, running a detector over it, and deriving
tracklets and review issues costs minutes. Doing it again because the user
closed the annotator is the kind of cost that makes people stop using a feature.

So everything derived from *(this video, this model, these settings)* is written
under the project, keyed by a digest of exactly those three things. Change any
of them and the key changes and the stale entry is simply never read (§33).

What invalidation is keyed on, and why each part is there
--------------------------------------------------------

=================  =========================================================
video fingerprint  Size and modification time, **not** a content hash. Hashing
                   a 4 GB video to decide whether to skip a 40-second scan
                   costs more than the scan. Size plus mtime catches every
                   realistic case — re-encoding, trimming, re-recording — and
                   the one case it misses (a byte-identical edit that preserves
                   both) does not occur in a lab capture workflow.
model version      A different checkpoint produces different predictions, so a
                   cached prediction from another model is not a cache hit, it
                   is a wrong answer.
settings           Score threshold, sampling strategy, seed, scan stride. All
                   of them change the output; a cache that ignored them would
                   return the previous run's answer to a different question.
=================  =========================================================

Failure policy: **a cache never raises into its caller.** Every read returns
``None`` on any problem and every write returns ``False``. A full disk, a
read-only project folder or a half-written file from a killed process must cost
a recomputation, never the session. This is the one place in the package where
a broad ``except`` is correct, and each one says so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

__all__ = [
    "CACHE_DIRNAME", "CacheKey", "DiskCache", "video_fingerprint",
]

#: Lives inside the project folder so a project is self-contained: copy the
#: folder to another machine and the cache goes with it. Dot-prefixed so it
#: sorts out of the way and reads as machinery rather than as data.
CACHE_DIRNAME = ".yoake_cache"

#: Bumped when the *meaning* of a cached artefact changes — a new field in
#: FrameFeature, a changed detector threshold's default. Every existing entry
#: then misses instead of being mis-read as the new shape.
SCHEMA_VERSION = 1


def video_fingerprint(path: Any) -> str:
    """``name:size:mtime_ns`` for a video file, or ``"?"`` when unreadable.

    ``"?"`` deliberately does **not** raise and deliberately is not unique: a
    video that cannot be stat'd gets a key that collides with every other such
    video, which is harmless because a missing video means nothing will be
    computed for it either.
    """
    try:
        p = Path(path)
        st = p.stat()
        return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"
    except (OSError, TypeError, ValueError):
        return "?"


@dataclass(frozen=True)
class CacheKey:
    """Everything an artefact depends on, hashed into one directory name."""

    video: str = "?"
    model: str = ""
    settings: Mapping[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    @classmethod
    def for_video(
        cls,
        video_path: Any,
        *,
        model: str = "",
        **settings: Any,
    ) -> "CacheKey":
        """The usual constructor::

            key = CacheKey.for_video(video, model=ckpt_name,
                                     stride=15, strategy="diversity", seed=0)
        """
        return cls(video=video_fingerprint(video_path), model=str(model or ""),
                   settings=dict(settings))

    def with_settings(self, **extra: Any) -> "CacheKey":
        """A derived key for an artefact that depends on one more thing."""
        merged = dict(self.settings)
        merged.update(extra)
        return CacheKey(video=self.video, model=self.model, settings=merged,
                        schema=self.schema)

    def digest(self) -> str:
        """16 hex characters. Collision-free enough for a per-project folder
        and short enough that the paths stay readable during debugging."""
        payload = json.dumps(
            {"v": self.video, "m": self.model, "s": self.settings,
             "schema": self.schema},
            sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """For the log: which video and model an entry belongs to (§40)."""
        model = self.model or "no model"
        return f"{self.video.split(':')[0]} / {model} / {self.digest()}"


class DiskCache:
    """A small JSON store under one project folder.

    Artefacts are named within a key, so one video's scan holds several::

        cache.put(key, "features", [...])
        cache.put(key, "issues", [...])
        cache.get(key, "features")

    Writes are atomic — a temporary file plus :func:`os.replace` — so a process
    killed mid-write leaves the previous entry intact rather than a truncated
    JSON file that every later run fails to parse.
    """

    def __init__(self, root: Any, *, dirname: str = CACHE_DIRNAME) -> None:
        self.root = Path(root) / dirname
        self._enabled = True

    # ── location ─────────────────────────────────────────────────────────
    def dir_for(self, key: CacheKey) -> Path:
        return self.root / key.digest()

    def path_for(self, key: CacheKey, name: str, suffix: str = ".json") -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
        return self.dir_for(key) / f"{safe}{suffix}"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self, reason: str = "") -> None:
        """Stop trying. Called after a write fails so a read-only project folder
        costs one failed write and not one per artefact."""
        if self._enabled:
            self._enabled = False
            _log.info("Cache disabled: %s", reason or "write failed")

    # ── reading ──────────────────────────────────────────────────────────
    def has(self, key: CacheKey, name: str) -> bool:
        return self.path_for(key, name).is_file()

    def get(self, key: CacheKey, name: str, default: Any = None) -> Any:
        """The cached value, or ``default`` for any reason at all."""
        if not self._enabled:
            return default
        path = self.path_for(key, name)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError):
            # Missing is the common case; corrupt is the case that matters, and
            # both mean the same thing to the caller: recompute.
            return default
        if not isinstance(payload, dict) or "data" not in payload:
            return default
        return payload["data"]

    def meta(self, key: CacheKey, name: str) -> Dict[str, Any]:
        """When an entry was written and by what — for the log and for pruning."""
        path = self.path_for(key, name)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return {k: v for k, v in payload.items() if k != "data"}
        except (OSError, ValueError, UnicodeDecodeError):
            return {}

    # ── writing ──────────────────────────────────────────────────────────
    def put(self, key: CacheKey, name: str, data: Any, **extra: Any) -> bool:
        """Store one artefact. ``False`` when it could not be stored.

        The caller is expected to ignore the return value in the normal path:
        a cache that failed to write is slower, not wrong.
        """
        if not self._enabled:
            return False
        path = self.path_for(key, name)
        payload = {
            "schema": key.schema,
            "written": time.time(),
            "key": key.describe(),
            **extra,
            "data": data,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(path.parent),
                prefix=path.name + ".", suffix=".tmp", delete=False)
            try:
                json.dump(payload, handle, ensure_ascii=False,
                          separators=(",", ":"), default=_jsonable)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            _log.debug("Cache write failed for %s: %s", path, exc)
            try:
                os.unlink(handle.name)          # type: ignore[possibly-undefined]
            except (OSError, NameError, UnboundLocalError):
                pass
            if isinstance(exc, OSError):
                self.disable(f"{type(exc).__name__} writing {path.parent}")
            return False

    # ── housekeeping ─────────────────────────────────────────────────────
    def drop(self, key: CacheKey) -> bool:
        """Delete every artefact for one key."""
        try:
            shutil.rmtree(self.dir_for(key))
            return True
        except OSError:
            return False

    def clear(self) -> bool:
        """Delete the whole cache. Safe to call when it does not exist."""
        try:
            shutil.rmtree(self.root)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def entries(self) -> List[Tuple[str, float, int]]:
        """``[(digest, newest_mtime, bytes), ...]``, newest first."""
        out: List[Tuple[str, float, int]] = []
        try:
            children = list(self.root.iterdir())
        except OSError:
            return out
        for child in children:
            if not child.is_dir():
                continue
            newest, total = 0.0, 0
            try:
                for item in child.iterdir():
                    stat = item.stat()
                    newest = max(newest, stat.st_mtime)
                    total += stat.st_size
            except OSError:
                continue
            out.append((child.name, newest, total))
        out.sort(key=lambda row: -row[1])
        return out

    def size_bytes(self) -> int:
        return sum(row[2] for row in self.entries())

    def prune(self, keep: int = 8) -> int:
        """Keep the ``keep`` most recent keys, delete the rest. Returns how many
        were deleted.

        Called on project open. Without it, every threshold a user tries leaves
        a directory behind and the cache grows without bound — which turns a
        performance feature into a disk-space complaint.
        """
        rows = self.entries()
        removed = 0
        for digest, _mtime, _size in rows[max(0, int(keep)):]:
            try:
                shutil.rmtree(self.root / digest)
                removed += 1
            except OSError:
                continue
        return removed

    def describe(self) -> str:
        """``6 cached runs, 42 MB`` for the settings page."""
        rows = self.entries()
        if not rows:
            return "Nothing cached"
        total = sum(r[2] for r in rows) / (1024 * 1024)
        return (f"{len(rows)} cached run{'s' if len(rows) != 1 else ''}, "
                f"{total:.0f} MB")


def _jsonable(obj: Any) -> Any:
    """Last-resort encoder for dataclasses and tuples inside cached payloads."""
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if obj == float("inf"):
        return None
    return str(obj)
