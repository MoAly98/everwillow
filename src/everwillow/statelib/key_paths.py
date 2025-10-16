from __future__ import annotations

import typing as tp

import jax.tree_util as jtu

KeyPath = tuple[tp.Any, ...]
KeyTuple = tp.TypeVar("KeyTuple", bound=tuple[tp.Any, ...])


def canonical_key(path: KeyPath) -> KeyPath:
    """Convert a JAX key path to plain Python entries.

    Args:
        path: Key path emitted by ``jax.tree_util`` while walking a pytree.

    Returns:
        Tuple containing the concrete dictionary keys, indices, or attribute
        names that identify the same leaf within a pytree.

    Examples:
        >>> import jax.tree_util as jtu
        >>> canonical_key((jtu.DictKey("a"), jtu.SequenceKey(0)))
        ('a', 0)
    """
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
    """Validate and normalise an external key path.

    Args:
        key: Candidate key path provided by a caller.

    Returns:
        Tuple representation of the key path that can safely index a
        ``FlatState``.

    Raises:
        KeyError: If ``key`` is not already a tuple.
    """
    if not isinstance(key, tuple):
        raise KeyError("FlatState keys must be tuples")
    return key


def _make_key_entry(value: tp.Any, template: tp.Any | None) -> tp.Any:
    """Build a ``jax.tree_util`` key object for a single path component.

    Args:
        value: Public representation of the key component.
        template: Optional existing key entry used to preserve type fidelity.

    Returns:
        ``jax.tree_util`` key object matching ``value``.
    """
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
    """Construct a JAX key path matching a public key tuple.

    Args:
        key: Tuple describing the public representation of the key.
        template: Optional existing JAX key path whose entry types should be
            preserved. When omitted the entry type is inferred.

    Returns:
        Tuple of ``jax.tree_util`` key objects that reference the same pytree
        leaf as ``key``.

    Examples:
        >>> derive_key_path(("a", 0))
        (DictKey(key='a'), SequenceKey(idx=0))
    """
    if template is None or len(template) != len(key):
        template = None
    entries = [
        _make_key_entry(value, None if template is None else template[index])
        for index, value in enumerate(key)
    ]
    return tuple(entries)


__all__ = ["canonical_key", "derive_key_path", "ensure_public_key", "KeyPath"]
