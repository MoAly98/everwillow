"""Immutable mapping helpers for working with JAX pytrees.

The utilities in this module provide read-only dictionaries that keep track of
the canonical tuple keys JAX emits when flattening nested structures.
"""

from __future__ import annotations

import typing as tp
from types import EllipsisType, MappingProxyType

import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree

from everwillow.statelib.meta import MergeMeta, TreeDefMeta

__all__ = [
    "K",
    "PartitionedMapping",
    "State",
    "V",
    "canonicalize_key",
    "combine_partitions",
    "merge",
    "partition",
    "split",
    "update",
]


K: tp.TypeAlias = str | tuple[str, ...]
V = tp.TypeVar("V", bound=ArrayLike)


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str) -> K: ...


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: None) -> K: ...


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
            raise TypeError(msg)
    if sep is not None:
        return sep.join(map(str, result))

    return tuple(result)


class BaseMapping(tp.Mapping[K, V], tp.Generic[V]):
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

    __slots__ = ("_mapping",)

    _mapping: tp.Mapping[K, V]

    def __getitem__(self, key: K) -> V:
        return self._mapping[key]

    def __iter__(self) -> tp.Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def to_dict(self) -> dict[K, V]:
        return dict(self._mapping)

    @property
    def mapping(self) -> tp.Mapping[K, V]:
        """Read-only view of the underlying mapping.

        Returns:
            :class:`types.MappingProxyType` exposing the raw key/value pairs.

        Examples:
            >>> state = State.from_pytree({"a": 1.0})
            >>> ("a",) in state.mapping
            True
        """

        return self._mapping


class FrozenChainMap(BaseMapping[V]):
    """Create a read-only ChainMap from multiple mappings.

    Args:
        *mappings: Ordered sequence of mappings to combine.

    Returns:
        A read-only ChainMap containing the combined mappings.
        In contrast to ``collections.ChainMap``, this class is immutable.
    """

    def __init__(self, *mappings: tp.Mapping[K, V]) -> None:
        # Ensure all internal maps are immutable
        self._mapping = tp.ChainMap(*map(MappingProxyType, mappings))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    @property
    def maps(self) -> tuple[tp.Mapping[K, V], ...]:
        # forward from ChainMap
        return self._mapping.maps  # type: ignore[attr-defined]


@jtu.register_pytree_with_keys_class
class State(BaseMapping[V]):
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

    __slots__ = ("_treedefmeta",)

    def __init__(
        self,
        mapping: tp.Mapping[K, V],
        *,
        treedefmeta: TreeDefMeta,
    ) -> None:
        """Initialise a state from an existing mapping.

        Args:
            mapping: Mapping whose keys are canonical tuples and values are
                pytree leaves.
            treedefmeta: TreeDefMeta instance containing the pytree definition
                and (ordered) keys for reconstruction.
        """
        # Ensure the mapping is immutable
        self._mapping = MappingProxyType(mapping)

        if not isinstance(treedefmeta, TreeDefMeta):
            msg = "'treedefmeta' must be a TreeDefMeta instance. Use State.from_pytree() instead."  # type: ignore[unreachable]
            raise TypeError(msg)
        self._treedefmeta = treedefmeta

    @classmethod
    def from_pytree(
        cls,
        pytree: PyTree[V],
        *,
        is_leaf: tp.Callable[[V], bool] | None = None,
        sep: str | None = None,
    ) -> State[V]:
        """Build a :class:`State` instance from an arbitrary pytree.

        Args:
            pytree: Nested structure supported by :mod:`jax.tree_util`.
            is_leaf: Optional callable passed to :func:`jax.tree_util.tree_flatten_with_path`
                to customize which nodes are treated as leaves.
            sep: Optional separator used to join key entries when constructing
                public keys. When ``None`` (default), keys are returned as tuples.

        Returns:
            New :class:`State` representing ``pytree``.

        Examples:
            >>> State.from_pytree({"a": [1, 2]}).mapping
            mappingproxy({('a', 0): 1.0, ('a', 1): 2.0})
        """

        if isinstance(pytree, State):
            msg = f"{pytree=} is already a State instance"
            raise TypeError(msg)

        # flatten the pytree with paths to build canonical keys
        path_leaves, treedef = jtu.tree_flatten_with_path(pytree, is_leaf=is_leaf)
        data, keys = {}, []
        for path, leaf in path_leaves:
            key = canonicalize_key(path, sep=sep)
            data[key] = leaf
            keys.append(key)

        treedefmeta = TreeDefMeta(treedef=treedef, keys=tuple(keys))
        return cls(data, treedefmeta=treedefmeta)

    def to_pytree(self) -> PyTree[V]:
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

        return self.treedefmeta.to_pytree(self.mapping)

    @property
    def treedefmeta(self) -> TreeDefMeta:
        return self._treedefmeta

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    # jax.tree_util.register_pytree_with_keys_class methods
    def tree_flatten_with_keys(self):
        # .to_dict() because jax.tree_util already knows how to flatten dicts
        children_with_keys = ((jtu.GetAttrKey("_mapping"), self.to_dict()),)
        aux_data = (self._treedefmeta,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[TreeDefMeta, ...],
        children: tuple[tp.Mapping[K, V], ...],
    ) -> State[V]:
        (treedefmeta,) = aux_data
        (mapping,) = children
        return cls(mapping, treedefmeta=treedefmeta)


@jtu.register_pytree_with_keys_class
class PartitionedMapping(BaseMapping[V]):
    """Read-only mapping that remembers which object it was partitioned from.

    Each partition stores the ``id`` of the original mapping so that only
    compatible partitions can be combined again.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, lambda key, _: key == ("a",))
        >>> dict(left.mapping)
        {('a',): 1}
    """

    __slots__ = ("_origin",)

    def __init__(
        self,
        mapping: tp.Mapping[K, V],
        *,
        origin: int,
    ) -> None:
        # Ensure the mapping is immutable
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

    # jax.tree_util.register_pytree_with_keys_class methods
    def tree_flatten_with_keys(self):
        # .to_dict() because jax.tree_util already knows how to flatten dicts
        children_with_keys = ((jtu.GetAttrKey("_mapping"), self.to_dict()),)
        aux_data = (self._origin,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[int],
        children: tuple[tp.Mapping[K, V], ...],
    ) -> PartitionedMapping[V]:
        (mapping,) = children
        (origin,) = aux_data
        return cls(mapping, origin=origin)


def merge(*states: State[V]) -> tuple[FrozenChainMap[V], MergeMeta]:
    """Combine several :class:`State` objects into a single mapping.

    Args:
        *states: Ordered sequence of states to merge.

    Returns:
        Tuple containing the merged mapping and the mergemeta required to split it.

    Note:
        When multiple states contain the same key, the value from the first state
        in ``*states`` takes precedence due to :class:`~collections.ChainMap` semantics.

    Examples:
        >>> first = State.from_pytree({"a": 1})
        >>> second = State.from_pytree({"b": 2})
        >>> merged, mergemeta = merge(first, second)
        >>> merged["b",]
        2
    """

    merged, treedefmetas = [], []
    for state in states:
        merged.append(state.mapping)
        treedefmetas.append(state.treedefmeta)
    mergemeta = MergeMeta(treedefmetas=tuple(treedefmetas))
    return FrozenChainMap(*merged), mergemeta


def split(
    mapping: FrozenChainMap[V],
    mergemeta: MergeMeta,
) -> tuple[State[V], ...]:
    """Split a merged mapping back into its original states.

    Args:
        mapping: Mapping produced by :func:`merge`.
        mergemeta: MergeMeta returned by :func:`merge` for the same merge call.

    Returns:
        Tuple containing one :class:`State` per merged input.

    Examples:
        >>> first = State.from_pytree({"a": 1})
        >>> merged, mergemeta = merge(first)
        >>> split(merged, mergemeta)[0].to_pytree()
        {'a': 1}
    """

    return mergemeta.split(mapping)


def partition(
    mapping: tp.Mapping[K, V],
    *,
    predicate: tp.Callable[[K, V], bool],
) -> tuple[PartitionedMapping[V], PartitionedMapping[V]]:
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
    left: PartitionedMapping[V],
    right: PartitionedMapping[V],
) -> FrozenChainMap[V]:
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
    return FrozenChainMap(left.mapping, right.mapping)


def update(
    state: State[V],
    *,
    updates: tp.Mapping[K, V | EllipsisType],
) -> State[V]:
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
        raise TypeError(msg)

    if missing := set(updates.keys()) - set(state.keys()):
        msg = f"cannot update missing keys: {missing}"
        raise KeyError(msg)

    data = dict(state.mapping)
    for key, value in updates.items():
        if value is ...:
            continue
        if key not in data:
            msg = f"cannot update missing key {key}"
            raise KeyError(msg)
        data[key] = value
    return State(data, treedefmeta=state.treedefmeta)
