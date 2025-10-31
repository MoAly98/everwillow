"""Immutable mapping helpers for working with JAX pytrees.

The utilities in this module provide read-only dictionaries that keep track of
the canonical tuple keys JAX emits when flattening nested structures.
"""

from __future__ import annotations

import dataclasses
import typing as tp
from functools import partial
from types import MappingProxyType

import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree, PyTreeDef

KeyPath = tp.Hashable
LeafT = tp.TypeVar("LeafT", bound=ArrayLike)


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str) -> KeyPath: ...


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: None) -> tuple: ...


def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str | None = None) -> KeyPath:
    """Convert a JAX key path to plain Python entries.

    Args:
        path: Key path emitted by :func:`jax.tree_util.tree_flatten_with_path`.
        sep: Optional separator used to join the entries into a string. When
            ``None`` (default) the key is returned as a tuple.

    Returns:
        Canonical key representation that can be used to index a :class:`State`.

    Raises:
        ValueError: If ``path`` contains an unsupported key type.

    Examples:
        Build tuple keys that match the structure of the original pytree:

        >>> import jax.tree_util as jtu
        >>> canonicalize_key((jtu.DictKey("a"), jtu.SequenceKey(0)))
        ('a', 0)

        Produce joined string keys when ``sep`` is provided:

        >>> canonicalize_key((jtu.DictKey("a"), jtu.SequenceKey(0)), sep="/")
        'a/0'
    """
    result: list[tp.Any] = []
    for entry in path:
        if isinstance(entry, jtu.DictKey):
            result.append(entry.key)
        elif isinstance(entry, jtu.GetAttrKey):
            result.append(entry.name)
        elif isinstance(entry, jtu.SequenceKey):
            result.append(entry.idx)
        elif isinstance(entry, jtu.FlattenedIndexKey):
            result.append(entry.key)
        else:
            msg = f"Unrecognised key path entry: {entry}"
            raise ValueError(msg)
    if sep is not None:
        return sep.join(map(str, result))
    return tuple(result)


class BaseMapping(tp.Mapping[KeyPath, LeafT], tp.Generic[LeafT]):
    """Read-only mapping facade used by the state containers.

    This class wraps an immutable mapping and exposes the standard mapping
    protocol while providing helpers that are convenient for tests.

    Examples:
        Create a :class:`State` and access its mapping interface:

        >>> state = State.from_pytree({"a": 1.0})
        >>> isinstance(state, BaseMapping)
        True
        >>> state[("a",)]
        1.0
    """

    _mapping: MappingProxyType

    def __getitem__(self, key: KeyPath) -> LeafT:
        return tp.cast(LeafT, self._mapping[key])

    def __iter__(self) -> tp.Iterator[KeyPath]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def to_dict(self) -> dict[KeyPath, LeafT]:
        return dict(self._mapping)

    @property
    def mapping(self) -> tp.Mapping[KeyPath, LeafT]:
        """Read-only view of the underlying mapping.

        Returns:
            :class:`types.MappingProxyType` exposing the raw key/value pairs.

        Examples:
            >>> state = State.from_pytree({"a": 1.0})
            >>> ("a",) in state.mapping
            True
        """

        return tp.cast(tp.Mapping[KeyPath, LeafT], self._mapping)


@jtu.register_pytree_with_keys_class
class State(BaseMapping[LeafT]):
    """Container that stores flattened pytrees keyed by canonical tuples.

    The state keeps track of the pytree definition so it can be converted back
    to the original nested structure.

    Examples:
        >>> state = State.from_pytree({"a": {"b": 2.0}})
        >>> state[("a", "b")]
        2.0
        >>> state.to_pytree()
        {'a': {'b': 2.0}}
    """

    def __init__(
        self,
        mapping: tp.Mapping[KeyPath, LeafT],
        *,
        treedef: PyTreeDef | None = None,
    ) -> None:
        """Initialise a state from an existing mapping.

        Args:
            mapping: Mapping whose keys are canonical tuples and values are
                pytree leaves.
            treedef: Optional pytree definition used for reconstruction.

        Examples:
            >>> State(mapping={("a",): 1}, treedef=None).to_dict()
            {('a',): 1}
        """
        self._mapping = MappingProxyType(dict(mapping))
        self._treedef = treedef

    @classmethod
    def from_pytree(cls, pytree: PyTree, *, sep: str | None = None) -> State[tp.Any]:
        """Build a :class:`State` instance from an arbitrary pytree.

        Args:
            pytree: Nested structure supported by :mod:`jax.tree_util`.
            sep: Optional separator used to join key entries when constructing
                public keys. The default forward slash mirrors filesystem paths.

        Returns:
            New :class:`State` representing ``pytree``.

        Examples:
            >>> State.from_pytree({"a": [1, 2]}).mapping
            mappingproxy({('a', 0): 1.0, ('a', 1): 2.0})
        """

        # noop if it is already as state
        if isinstance(pytree, State):
            return pytree

        path_leaves, treedef = jtu.tree_flatten_with_path(pytree)
        data = {canonicalize_key(path, sep=sep): leaf for path, leaf in path_leaves}
        return cls(data, treedef=treedef)

    def to_pytree(self) -> PyTree:
        """Reconstruct the stored pytree using the cached tree definition.

        Returns:
            Pytree with the same structure used to create the state.

        Raises:
            ValueError: If the state was created without a tree definition.

        Examples:
            >>> state = State.from_pytree({"x": 1})
            >>> state.to_pytree()
            {'x': 1}
        """

        if self.treedef is None:
            msg = "can't convert to PyTree if there's no tree definition"
            raise ValueError(msg)
        return jtu.tree_unflatten(self.treedef, list(self.values()))

    @property
    def treedef(self) -> PyTreeDef | None:
        return self._treedef

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    def tree_flatten_with_keys(self):
        children_with_keys = ((jtu.GetAttrKey("_mapping"), dict(self._mapping)),)
        aux_data = (self._treedef,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[PyTreeDef | None],
        children: tuple[tp.Mapping[KeyPath, LeafT], ...],
    ) -> State[LeafT]:
        (treedef,) = aux_data
        (mapping,) = children
        return cls(mapping, treedef=treedef)


@partial(
    jtu.register_dataclass,
    data_fields=[],
    meta_fields=["treedefs", "keys"],
)
@dataclasses.dataclass(frozen=True)
class MergeMetadata:
    """Metadata retained when multiple :class:`State` objects are merged.

    The metadata stores the treedef for each original state alongside the group
    of keys that belong to that state so the merged mapping can be split later.
    """

    treedefs: tuple[PyTreeDef | None, ...]
    keys: tuple[tuple[KeyPath, ...], ...]

    def split(
        self,
        mapping: tp.Mapping[KeyPath, LeafT],
    ) -> tuple[State[LeafT], ...]:
        """Partition a merged mapping back into individual states.

        Args:
            mapping: Mapping produced by :func:`merge`.

        Returns:
            Tuple of :class:`State` objects restoring the original order.

        Examples:
            >>> state_a = State.from_pytree({"a": 1})
            >>> merged, metadata = merge(state_a)
            >>> metadata.split(merged)[0].to_pytree()
            {'a': 1}
        """
        states: list[State[LeafT]] = []
        for key_group, treedef in zip(self.keys, self.treedefs, strict=True):
            data = {key: mapping[key] for key in key_group}
            states.append(State(data, treedef=treedef))
        return tuple(states)


def merge(*states: State[LeafT]) -> tuple[dict[KeyPath, LeafT], MergeMetadata]:
    """Combine several :class:`State` objects into a single mapping.

    Args:
        *states: Ordered sequence of states to merge.

    Returns:
        Tuple containing the merged mapping and the metadata required to split it.

    Examples:
        >>> first = State.from_pytree({"a": 1})
        >>> second = State.from_pytree({"b": 2})
        >>> merged, metadata = merge(first, second)
        >>> merged["b",]
        2
    """

    merged: dict[KeyPath, LeafT] = {}
    key_groups: list[tuple[KeyPath, ...]] = []
    treedefs: list[PyTreeDef | None] = []
    for state in states:
        merged.update(state.mapping)
        key_groups.append(tuple(state.mapping.keys()))
        treedefs.append(state.treedef)
    metadata = MergeMetadata(treedefs=tuple(treedefs), keys=tuple(key_groups))
    return merged, metadata


def split(
    mapping: tp.Mapping[KeyPath, LeafT],
    metadata: MergeMetadata,
) -> tuple[State[LeafT], ...]:
    """Split a merged mapping back into its original states.

    Args:
        mapping: Mapping produced by :func:`merge`.
        metadata: Metadata returned by :func:`merge` for the same merge call.

    Returns:
        Tuple containing one :class:`State` per merged input.

    Examples:
        >>> first = State.from_pytree({"a": 1})
        >>> merged, metadata = merge(first)
        >>> split(merged, metadata)[0].to_pytree()
        {'a': 1}
    """

    return metadata.split(mapping)


@jtu.register_pytree_with_keys_class
class PartitionedMapping(BaseMapping[LeafT]):
    """Read-only mapping that remembers which object it was partitioned from.

    Each partition stores the ``id`` of the original mapping so that only
    compatible partitions can be combined again.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, lambda key, _: key == ("a",))
        >>> dict(left.mapping)
        {('a',): 1}
    """

    def __init__(
        self,
        mapping: tp.Mapping[KeyPath, LeafT],
        *,
        origin: int,
    ) -> None:
        self._mapping = MappingProxyType(dict(mapping))
        self._origin = origin

    @property
    def origin(self) -> int:
        """Identifier of the mapping the partition originated from.

        Returns:
            Integer identifier equal to :func:`id` of the mapping passed during
            construction.
        """
        return self._origin

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r}, origin={self.origin})"

    def tree_flatten_with_keys(self):
        children_with_keys = ((jtu.GetAttrKey("_mapping"), dict(self._mapping)),)
        aux_data = (self._origin,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[int],
        children: tuple[tp.Mapping[KeyPath, LeafT], ...],
    ) -> PartitionedMapping[LeafT]:
        (origin,) = aux_data
        (mapping,) = children
        return cls(mapping, origin=origin)

    def __or__(self, other: tp.Any) -> dict[KeyPath, LeafT]:
        """Combine two partitions that came from the same origin.

        Args:
            other: Another :class:`PartitionedMapping`.

        Returns:
            Dictionary containing the union of both partitions.

        Raises:
            ValueError: If ``other`` is not a partition or was created from a
                different origin.

        Examples:
            >>> state = State.from_pytree({"a": 1, "b": 2})
            >>> left, right = partition(state, lambda key, _: key == ("a",))
            >>> (left | right)["b",]
            2
        """
        if not isinstance(other, PartitionedMapping):
            msg = "can only merge with another 'PartitionedMapping'"
            raise ValueError(msg)
        if self.origin != other.origin:
            msg = "partitions must originate from the same original mapping"
            raise ValueError(msg)
        merged = dict(self.mapping)
        merged.update(other.mapping)
        return merged


def partition(
    mapping: tp.Mapping[KeyPath, LeafT],
    predicate: tp.Callable[[KeyPath, LeafT], bool],
) -> tuple[PartitionedMapping[LeafT], PartitionedMapping[LeafT]]:
    """Split a mapping into two partitions based on a predicate.

    Args:
        mapping: Mapping obtained from a :class:`State`. The identity of this
            mapping is stored to ensure only compatible partitions are merged.
        predicate: Callable returning ``True`` for items that should go into
            the first partition.

    Returns:
        Tuple ``(left, right)`` containing two :class:`PartitionedMapping`
        objects with the same ``origin``.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, lambda key, _: key == ("a",))
        >>> dict(right.mapping)
        {('b',): 2}
    """

    left_data: dict[KeyPath, LeafT] = {}
    right_data: dict[KeyPath, LeafT] = {}
    for key, value in mapping.items():
        if predicate(key, value):
            left_data[key] = value
        else:
            right_data[key] = value

    origin = id(mapping)
    return (
        PartitionedMapping(left_data, origin=origin),
        PartitionedMapping(right_data, origin=origin),
    )


def combine_partitions(
    left: PartitionedMapping[LeafT],
    right: PartitionedMapping[LeafT],
) -> dict[KeyPath, LeafT]:
    """Merge two partitions that originated from the same mapping.

    Args:
        left: First partition returned by :func:`partition`.
        right: Second partition returned by :func:`partition`.

    Returns:
        Dictionary containing the union of both partitions' mappings.

    Raises:
        ValueError: If the partitions do not share the same ``origin``.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, lambda key, _: key == ("a",))
        >>> combine_partitions(left, right)["b",]
        2
    """

    return left | right


def update(
    state: State[LeafT],
    updates: tp.Mapping[KeyPath, LeafT],
) -> State[LeafT]:
    """Return a new state with specific entries replaced.

    Args:
        state: Original :class:`State` to copy.
        updates: Mapping of existing keys to replacement values. Entries whose
            value is ``Ellipsis`` are ignored, which makes it easy to reuse the
            same dictionary across multiple updates.

    Returns:
        New :class:`State` with the replacements applied.

    Raises:
        KeyError: If ``updates`` includes a key that is not present in ``state``.

    Examples:
        >>> base = State.from_pytree({"a": 1, "b": 2})
        >>> update(base, {("b",): 99}).to_dict()
        {('a',): 1, ('b',): 99}
    """
    if not isinstance(state, State):
        msg = "Can only update State types"  # type: ignore[unreachable]
        raise ValueError(msg)

    data = dict(state.mapping)
    for key, value in updates.items():
        if value is ...:
            continue
        if key not in data:
            msg = f"cannot update missing key {key}"
            raise KeyError(msg)
        data[key] = value
    return State(data, treedef=state.treedef)


__all__ = [
    "KeyPath",
    "LeafT",
    "MergeMetadata",
    "PartitionedMapping",
    "State",
    "canonicalize_key",
    "combine_partitions",
    "merge",
    "partition",
    "split",
    "update",
]
