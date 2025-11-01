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

K: tp.TypeAlias = str | tuple[str, ...]
T = tp.TypeVar("T", bound=ArrayLike)


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str) -> K: ...


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: None) -> tuple: ...


def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str | None = None) -> K:
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


def FrozenChainMap(
    *mappings: tp.Mapping[K, T],
) -> tp.ChainMap[K, T]:
    """Create a read-only ChainMap from multiple mappings.

    Args:
        *mappings: Ordered sequence of mappings to combine.

    Returns:
        A read-only ChainMap containing the combined mappings.
    """
    return tp.ChainMap(*map(MappingProxyType, mappings))  # type: ignore[arg-type]


class BaseMapping(tp.Mapping[K, T], tp.Generic[T]):
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

    def __getitem__(self, key: K) -> T:
        return self._mapping[key]

    def __iter__(self) -> tp.Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def to_dict(self) -> dict[K, T]:
        return dict(self._mapping)

    @property
    def mapping(self) -> MappingProxyType[K, T]:
        """Read-only view of the underlying mapping.

        Returns:
            :class:`types.MappingProxyType` exposing the raw key/value pairs.

        Examples:
            >>> state = State.from_pytree({"a": 1.0})
            >>> ("a",) in state.mapping
            True
        """

        return self._mapping


@jtu.register_pytree_with_keys_class
class State(BaseMapping[T]):
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
        mapping: tp.Mapping[K, T],
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
        self._mapping = MappingProxyType(mapping)
        self._treedef = treedef

    @classmethod
    def from_pytree(cls, pytree: PyTree[T], *, sep: str | None = None) -> State[T]:
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

        # noop if it is already a state
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
        return jtu.tree_unflatten(self.treedef, self.values())

    @property
    def treedef(self) -> PyTreeDef | None:
        return self._treedef

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    def tree_flatten_with_keys(self):
        children_with_keys = ((jtu.GetAttrKey("_mapping"), self._mapping),)
        aux_data = (self._treedef,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[PyTreeDef | None],
        children: tuple[tp.Mapping[K, T], ...],
    ) -> State[T]:
        (treedef,) = aux_data
        (mapping,) = children
        return cls(mapping, treedef=treedef)


@partial(
    jtu.register_dataclass,
    data_fields=[],
    meta_fields=["treedefs"],
)
@dataclasses.dataclass(frozen=True)
class MergeMetadata:
    """Metadata retained when multiple :class:`State` objects are merged.

    The metadata stores the treedef for each original state alongside the group
    of keys that belong to that state so the merged mapping can be split later.
    """

    treedefs: tuple[PyTreeDef | None, ...]

    def split(
        self,
        chain_map: tp.ChainMap[K, T],
    ) -> tuple[State[T], ...]:
        """Partition a merged ChainMap back into individual states.

        Args:
            chain_map: ChainMap produced by :func:`merge`.

        Returns:
            Tuple of :class:`State` objects restoring the original order.

        Examples:
            >>> state_a = State.from_pytree({"a": 1})
            >>> merged, metadata = merge(state_a)
            >>> metadata.split(merged)[0].to_pytree()
            {'a': 1}
        """
        states: list[State[T]] = []
        for mapping, treedef in zip(chain_map.maps, self.treedefs, strict=True):
            states.append(State(mapping, treedef=treedef))
        return tuple(states)


def merge(*states: State[T]) -> tuple[tp.ChainMap[K, T], MergeMetadata]:
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

    merged, treedefs = [], []
    for state in states:
        merged.append(state.mapping)
        treedefs.append(state.treedef)
    metadata = MergeMetadata(treedefs=tuple(treedefs))
    return FrozenChainMap(*merged), metadata


def split(
    mapping: tp.ChainMap[K, T],
    metadata: MergeMetadata,
) -> tuple[State[T], ...]:
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
class PartitionedMapping(BaseMapping[T]):
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
        mapping: tp.Mapping[K, T],
        *,
        origin: int,
    ) -> None:
        self._mapping = MappingProxyType(mapping)
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
        children_with_keys = ((jtu.GetAttrKey("_mapping"), self.to_dict()),)
        aux_data = (self._origin,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[int],
        children: tuple[tp.Mapping[K, T], ...],
    ) -> PartitionedMapping[T]:
        (origin,) = aux_data
        (mapping,) = children
        return cls(mapping, origin=origin)


def partition(
    mapping: tp.Mapping[K, T],
    predicate: tp.Callable[[K, T], bool],
) -> tuple[PartitionedMapping[T], PartitionedMapping[T]]:
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

    left_data, right_data = {}, {}
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
    left: PartitionedMapping[T],
    right: PartitionedMapping[T],
) -> tp.ChainMap[K, T]:
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
    if left.origin != right.origin:
        msg = "partitions must originate from the same original mapping"
        raise ValueError(msg)
    # this order is important here to preserve original mapping order
    return FrozenChainMap(right.mapping, left.mapping)


def update(
    state: State[T],
    updates: tp.Mapping[K, T],
) -> State[T]:
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
    "K",
    "MergeMetadata",
    "PartitionedMapping",
    "State",
    "T",
    "canonicalize_key",
    "combine_partitions",
    "merge",
    "partition",
    "split",
    "update",
]
