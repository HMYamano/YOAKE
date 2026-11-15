"""
behaviors.py — one behavior per animal, or one per group (multi-head).

``docs/KEYFRAME_BEHAVIOR.md`` §15. The default answer to "what is this animal
doing" is one behavior from one list. A project may instead split its
behaviors into **groups**: exclusive inside a group, simultaneous across
groups, so a fly can be walking *and* courting::

    locomotion: still, walk, run
    social:     none, courtship, copulation

One group is exactly the single-behavior case, which is why one class covers
both and the rest of the code asks it questions instead of branching on mode.

==========================  ==================================================
:class:`BehaviorVocabulary` The groups, their flat numbering, parse and print.
:func:`vocabulary_of`       A dataset's vocabulary (``meta.labeling.behaviors``).
:func:`project_vocabulary`  A project's vocabulary (``settings["behaviors"]``).
:func:`labels_of`           One answer per group for a box.
:func:`set_behavior`        Answer one group and keep the others.
:func:`describe`            ``walk + courtship`` for a box.
:func:`remap`               Keep the answers when the vocabulary is edited.
==========================  ==================================================

Numbering
---------
Behaviors are numbered **through all groups**, in group order: with the groups
above, ``still`` is 0, ``run`` is 2 and ``none`` is 3. That flat number is what
a key press sends, what colours a box, what a CSV column holds and what the
model's flat probability vector is indexed by - the same meaning everywhere.
Names that repeat across groups are told apart in the flat list as
``social:none``.

Storage on a box
----------------
Works for ``AnnObject``, ``ObjectAnno`` and plain dicts alike.
``action_ids`` holds one flat number per group, in group order, ``-1`` for a
group nobody judged. ``action_id`` mirrors the first group's answer, so a
reader that only knows single behaviors still reads something true. In single
mode ``action_ids`` is never written, so those files stay byte-identical.

Purity: no dearpygui, no torch, no cv2, no numpy - like the rest of
:mod:`htrtdetr.annotation`, so it is testable from either interpreter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = [
    "SINGLE", "MULTI_HEAD", "BEHAVIOR_MODES", "SINGLE_GROUP",
    "BehaviorVocabulary", "parse_vocabulary",
    "behavior_mode_of", "vocabulary_of", "project_vocabulary",
    "labels_of", "set_labels", "set_behavior", "clear_group",
    "missing_groups", "is_complete", "has_behavior", "group_targets",
    "describe", "remap",
]

#: One behavior per animal, from one list. The default everywhere.
SINGLE = "single"
#: One behavior per *group*; groups hold at the same time.
MULTI_HEAD = "multi_head"
BEHAVIOR_MODES: Tuple[str, ...] = (SINGLE, MULTI_HEAD)

#: The name of the one group a single-behavior vocabulary has. Never written to
#: a file: single mode keeps using ``action_names`` alone.
SINGLE_GROUP = "behavior"

#: Between the group name and its behaviors when typed (``social: none, ...``).
_GROUP_SEP = ":"
#: Between groups typed on one line (``locomotion: still, walk; social: ...``).
_LINE_SEP = re.compile(r"[\n;]+")


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean(names: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(str(n).strip() for n in names if str(n).strip())


# ══════════════════════════════════════════════════════════════════════════
# The vocabulary
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BehaviorVocabulary:
    """The behavior groups and their flat numbering. Immutable.

    Build it with :meth:`single`, :meth:`from_groups` or
    :func:`parse_vocabulary`; the constructors validate, so an instance is
    always usable.
    """

    groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ((SINGLE_GROUP, ()),)
    mode: str = SINGLE

    # ── constructors ─────────────────────────────────────────────────────
    @classmethod
    def single(cls, names: Sequence[Any]) -> "BehaviorVocabulary":
        """One list, one answer per animal. Accepts an empty list."""
        return cls(groups=((SINGLE_GROUP, _clean(names)),), mode=SINGLE)

    @classmethod
    def from_groups(cls, groups: Any) -> "BehaviorVocabulary":
        """Several groups. ``{name: [behaviors]}`` or ``[(name, [behaviors]), ...]``.

        Raises ``ValueError`` with a sentence a user can act on: no groups, a
        group without a name, two groups with one name, a group with fewer
        than two behaviors (it would always give the same answer), or the
        same behavior twice in one group.
        """
        items = list(groups.items()) if isinstance(groups, Mapping) else list(groups or ())
        if not items:
            raise ValueError("Multi-head behaviors need at least one group.")
        out: List[Tuple[str, Tuple[str, ...]]] = []
        seen = set()
        for raw_name, raw_behaviors in items:
            name = str(raw_name or "").strip()
            if not name:
                raise ValueError("Every behavior group needs a name, as in 'social: none, courtship'.")
            if name in seen:
                raise ValueError(f"Two behavior groups are both called '{name}'.")
            seen.add(name)
            if isinstance(raw_behaviors, str):
                raw_behaviors = raw_behaviors.split(",")
            behaviors = _clean(raw_behaviors or ())
            if len(behaviors) < 2:
                raise ValueError(
                    f"Group '{name}' needs at least two behaviors - with one it would "
                    "always give the same answer. Add 'none' or 'other' if that is what "
                    "the rest of the time is."
                )
            if len(set(behaviors)) != len(behaviors):
                twice = sorted({b for b in behaviors if behaviors.count(b) > 1})
                raise ValueError(f"Group '{name}' lists {', '.join(twice)} twice.")
            out.append((name, behaviors))
        return cls(groups=tuple(out), mode=MULTI_HEAD)

    # ── the shape ────────────────────────────────────────────────────────
    @property
    def multi(self) -> bool:
        return self.mode == MULTI_HEAD

    @property
    def heads(self) -> Tuple[str, ...]:
        """Group names, in order (``("behavior",)`` for a single vocabulary)."""
        return tuple(name for name, _ in self.groups)

    @property
    def sizes(self) -> Tuple[int, ...]:
        return tuple(len(behaviors) for _, behaviors in self.groups)

    @property
    def offsets(self) -> Tuple[int, ...]:
        """The flat number of each group's first behavior."""
        out, total = [], 0
        for size in self.sizes:
            out.append(total)
            total += size
        return tuple(out)

    def __len__(self) -> int:
        return sum(self.sizes)

    @property
    def names(self) -> Tuple[str, ...]:
        """Every behavior in flat order. Repeated names carry their group."""
        counts: Dict[str, int] = {}
        for _, behaviors in self.groups:
            for b in behaviors:
                counts[b] = counts.get(b, 0) + 1
        return tuple(
            f"{group}{_GROUP_SEP}{b}" if self.multi and counts[b] > 1 else b
            for group, behaviors in self.groups
            for b in behaviors
        )

    def behaviors_of(self, group: int) -> Tuple[str, ...]:
        return self.groups[group][1] if 0 <= group < len(self.groups) else ()

    def as_groups(self) -> Dict[str, List[str]]:
        """``{group: [behaviors]}``, the form ``action_groups`` is written in."""
        return {name: list(behaviors) for name, behaviors in self.groups}

    # ── numbering ────────────────────────────────────────────────────────
    def group_of(self, action_id: Any) -> int:
        """Which group flat number ``action_id`` belongs to; -1 if none."""
        a = _int(action_id)
        if a < 0:
            return -1
        for h, (offset, size) in enumerate(zip(self.offsets, self.sizes)):
            if offset <= a < offset + size:
                return h
        return -1

    def group_range(self, group: int) -> range:
        if not 0 <= group < len(self.groups):
            return range(0)
        offset = self.offsets[group]
        return range(offset, offset + self.sizes[group])

    def local_of(self, action_id: Any) -> int:
        """Position of ``action_id`` inside its own group; -1 if out of range."""
        h = self.group_of(action_id)
        return _int(action_id) - self.offsets[h] if h >= 0 else -1

    def flat_id(self, group: int, local: int) -> int:
        if 0 <= group < len(self.groups) and 0 <= local < self.sizes[group]:
            return self.offsets[group] + local
        return -1

    def find(self, group: str, behavior: str) -> int:
        """Flat number of ``behavior`` in the group called ``group``; -1 if absent."""
        for h, (name, behaviors) in enumerate(self.groups):
            if name == group and behavior in behaviors:
                return self.offsets[h] + behaviors.index(behavior)
        return -1

    def name_of(self, action_id: Any, *, short: bool = True) -> str:
        """The behavior's name; ``short`` drops the group even for a repeated name."""
        h = self.group_of(action_id)
        if h < 0:
            return ""
        if short:
            return self.groups[h][1][self.local_of(action_id)]
        return self.names[_int(action_id)]

    # ── text ─────────────────────────────────────────────────────────────
    def to_text(self) -> str:
        """What the settings field shows: a comma list, or one group per line."""
        if not self.multi:
            return ", ".join(self.groups[0][1])
        return "\n".join(f"{name}{_GROUP_SEP} {', '.join(b)}" for name, b in self.groups)


def parse_vocabulary(text: str, multi_head: bool) -> BehaviorVocabulary:
    """Read what a user typed into the Behaviors field.

    Single: a comma-separated list (newlines work as commas too).
    Multi-head: one group per line - or per ``;`` - written ``group: a, b, c``.
    """
    raw = str(text or "")
    if not multi_head:
        return BehaviorVocabulary.single(re.split(r"[,\n;]+", raw))
    groups: List[Tuple[str, List[str]]] = []
    for number, line in enumerate((part.strip() for part in _LINE_SEP.split(raw)), start=1):
        if not line:
            continue
        if _GROUP_SEP not in line:
            raise ValueError(
                f"'{line}' needs a group name in front, as in 'social: none, courtship'."
            )
        name, _, rest = line.partition(_GROUP_SEP)
        groups.append((name.strip(), [b for b in rest.split(",")]))
    return BehaviorVocabulary.from_groups(groups)


# ══════════════════════════════════════════════════════════════════════════
# Where the vocabulary comes from
# ══════════════════════════════════════════════════════════════════════════

def _field_of(container: Any, name: str, default: Any) -> Any:
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def behavior_mode_of(dataset: Any) -> str:
    """:data:`MULTI_HEAD` only if ``meta.labeling.behaviors`` says so.

    ``action_groups`` alone is not enough: the v2.0 format already used it for
    hierarchical definitions, and a file must not change meaning because an
    old optional field happens to be there.
    """
    meta = _field_of(dataset, "meta", None)
    labeling = meta.get("labeling") if isinstance(meta, Mapping) else None
    if isinstance(labeling, Mapping) and labeling.get("behaviors") == MULTI_HEAD:
        return MULTI_HEAD
    return SINGLE


def vocabulary_of(dataset: Any) -> BehaviorVocabulary:
    """A dict's or ``DatasetAnno``'s vocabulary. Raises if multi-head is malformed."""
    if behavior_mode_of(dataset) == MULTI_HEAD:
        groups = _field_of(dataset, "action_groups", None) or {}
        if not groups:
            raise ValueError(
                "The file declares multi-head behaviors (meta.labeling.behaviors) "
                "but has no action_groups to say what the groups are."
            )
        return BehaviorVocabulary.from_groups(groups)
    return BehaviorVocabulary.single(_field_of(dataset, "action_names", None) or ())


def project_vocabulary(
    settings: Mapping[str, Any], action_names: Sequence[Any], action_groups: Any = None
) -> BehaviorVocabulary:
    """A project's vocabulary. Unusable groups fall back to the flat list."""
    if (settings or {}).get("behaviors") == MULTI_HEAD and action_groups:
        try:
            return BehaviorVocabulary.from_groups(action_groups)
        except ValueError:
            pass
    return BehaviorVocabulary.single(action_names or ())


# ══════════════════════════════════════════════════════════════════════════
# One box's answer
# ══════════════════════════════════════════════════════════════════════════

def _put(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[name] = value
    else:
        setattr(obj, name, value)


def labels_of(obj: Any, vocab: BehaviorVocabulary) -> Tuple[int, ...]:
    """One flat number per group (``-1`` = not judged). Single: ``(action_id,)``.

    A box answered before the project went multi-head has only ``action_id``;
    it is read as the answer of whichever group holds that number, so
    switching to groups keeps every label already given.
    """
    primary = _int(_field_of(obj, "action_id", -1))
    if not vocab.multi:
        return (primary,)
    labels = [-1] * len(vocab.groups)
    stored = _field_of(obj, "action_ids", None) or ()
    if stored:
        for h in range(len(labels)):
            if h < len(stored):
                value = _int(stored[h])
                if vocab.group_of(value) == h:
                    labels[h] = value
    elif primary >= 0 and vocab.group_of(primary) >= 0:
        labels[vocab.group_of(primary)] = primary
    return tuple(labels)


def set_labels(obj: Any, vocab: BehaviorVocabulary, labels: Sequence[Any]) -> None:
    """Write a whole answer. Multi-head keeps ``action_id`` as the first group's."""
    values = [_int(v) for v in labels]
    if not vocab.multi:
        _put(obj, "action_id", values[0] if values else -1)
        if _field_of(obj, "action_ids", None):
            _put(obj, "action_ids", [])
        return
    values = (values + [-1] * len(vocab.groups))[: len(vocab.groups)]
    values = [v if vocab.group_of(v) == h else -1 for h, v in enumerate(values)]
    _put(obj, "action_ids", values)
    _put(obj, "action_id", values[0])


def set_behavior(obj: Any, vocab: BehaviorVocabulary, action_id: Any) -> bool:
    """Answer the group ``action_id`` belongs to. Returns whether anything changed.

    The other groups keep their answers: pressing *walk* on a courting fly
    makes it a walking, courting fly. An out-of-range number changes nothing.
    """
    a = _int(action_id)
    if not vocab.multi:
        if _int(_field_of(obj, "action_id", -1)) == a:
            return False
        _put(obj, "action_id", a)
        return True
    h = vocab.group_of(a)
    if h < 0:
        return False
    labels = list(labels_of(obj, vocab))
    stored = list(_field_of(obj, "action_ids", None) or ())
    if labels[h] == a and stored == labels:
        return False
    labels[h] = a
    set_labels(obj, vocab, labels)
    return True


def clear_group(obj: Any, vocab: BehaviorVocabulary, group: int) -> bool:
    """Take back one group's answer (single: the answer). Returns whether it changed."""
    labels = list(labels_of(obj, vocab))
    if not 0 <= group < len(labels) or labels[group] < 0:
        return False
    labels[group] = -1
    set_labels(obj, vocab, labels)
    return True


def missing_groups(obj: Any, vocab: BehaviorVocabulary) -> Tuple[int, ...]:
    """Indices of the groups this box has no answer for."""
    return tuple(h for h, a in enumerate(labels_of(obj, vocab)) if a < 0)


def is_complete(obj: Any, vocab: BehaviorVocabulary) -> bool:
    return not missing_groups(obj, vocab)


def has_behavior(obj: Any, vocab: BehaviorVocabulary) -> bool:
    """Any group answered - what makes a box worth a behavior loss at all."""
    return any(a >= 0 for a in labels_of(obj, vocab))


def group_targets(obj: Any, vocab: BehaviorVocabulary) -> Tuple[int, ...]:
    """Each group's answer as a position inside that group (the model's target)."""
    return tuple(vocab.local_of(a) if a >= 0 else -1 for a in labels_of(obj, vocab))


def describe(obj: Any, vocab: BehaviorVocabulary, unset: str = "?") -> str:
    """``walk + courtship`` (``walk + ?`` while social is open). Single: the name."""
    parts = [vocab.name_of(a) if a >= 0 else unset for a in labels_of(obj, vocab)]
    return " + ".join(p or unset for p in parts)


def remap(objects: Iterable[Any], old: BehaviorVocabulary, new: BehaviorVocabulary) -> int:
    """Re-number answers after the vocabulary changed. Returns boxes changed.

    Answers follow their **names**, not their numbers: adding a behavior in the
    middle of a group must not turn every later answer into its neighbour.
    Looked up by (group, behavior); failing that, a behavior of the same name
    in the group at the same position (the group was renamed); failing that,
    in a single new list, by name. Anything not found becomes "not judged".
    Two old answers landing in one new group keep the earlier group's.
    """
    changed = 0
    for obj in objects:
        before = labels_of(obj, old)
        after = [-1] * len(new.groups)
        for h, a in enumerate(before):
            if a < 0:
                continue
            group = old.groups[h][0] if old.multi else (new.groups[0][0] if new.groups else "")
            behavior = old.name_of(a)
            target = new.find(group, behavior)
            if target < 0 and h < len(new.groups) and behavior in new.groups[h][1]:
                target = new.flat_id(h, new.groups[h][1].index(behavior))
            if target < 0 and not new.multi and behavior in new.groups[0][1]:
                target = new.groups[0][1].index(behavior)
            if target < 0 and new.multi and not old.multi:
                # A single list split into groups: find the name in any group.
                for g, (_name, behaviors) in enumerate(new.groups):
                    if behavior in behaviors:
                        target = new.flat_id(g, behaviors.index(behavior))
                        break
            if target < 0:
                continue
            g = new.group_of(target)
            if after[g] < 0:
                after[g] = target
        stored_before = (
            list(_field_of(obj, "action_ids", None) or ()),
            _int(_field_of(obj, "action_id", -1)),
        )
        set_labels(obj, new, after)
        stored_after = (
            list(_field_of(obj, "action_ids", None) or ()),
            _int(_field_of(obj, "action_id", -1)),
        )
        if stored_after != stored_before:
            changed += 1
    return changed
