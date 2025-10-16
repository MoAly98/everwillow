from __future__ import annotations

import typing as tp

import jax.tree_util as jtu

KeyPath = tuple[tp.Any, ...]
KeyTuple = tp.TypeVar("KeyTuple", bound=tuple[tp.Any, ...])


def canonical_key(path: KeyPath) -> KeyPath:
    """Convert a JAX key path to a tuple of plain Python entries."""
    result: list[tp.Any] = []
    for entry in path:
        if isinstance(entry, jtu.DictKey):
            result.append(entry.key)
        elif isinstance(entry, jtu.GetAttrKey):
            result.append(entry.name)
        elif isinstance(entry, (jtu.SequenceKey, jtu.FlattenedIndexKey)):
            result.append(entry.idx)
        else:
            raise ValueError(f"Unrecognised key path entry: {entry}")
    return tuple(result)


def ensure_public_key(key: KeyTuple) -> KeyTuple:
    """Validate and normalise an external key into tuple form."""
    if not isinstance(key, tuple):
        raise KeyError("FlatState keys must be tuples")
    return key


def _make_key_entry(value: tp.Any, template: tp.Any | None) -> tp.Any:
    if isinstance(template, jtu.DictKey):
        return jtu.DictKey(value)
    if isinstance(template, jtu.SequenceKey):
        return jtu.SequenceKey(value)
    if isinstance(template, jtu.GetAttrKey):
        return jtu.GetAttrKey(value)
    if isinstance(template, jtu.FlattenedIndexKey):
        return jtu.FlattenedIndexKey(value)
    if isinstance(value, int):
        return jtu.SequenceKey(value)
    return jtu.DictKey(value)


def derive_key_path(
    key: tuple[tp.Any, ...],
    *,
    template: KeyPath | None = None,
) -> KeyPath:
    """Construct a JAX key path matching the provided tuple key."""
    if template is None or len(template) != len(key):
        template = None
    entries = [
        _make_key_entry(value, None if template is None else template[index])
        for index, value in enumerate(key)
    ]
    return tuple(entries)


__all__ = ["canonical_key", "derive_key_path", "ensure_public_key", "KeyPath"]
