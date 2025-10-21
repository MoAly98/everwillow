"""Core state container used by :mod:`everwillow` fitting utilities."""

from __future__ import annotations

__all__ = [
    "FlatState",
    "combine_partitions",
    "map_state",
    "merge_states",
    "partition_state",
    "split_state",
    "update_state",
]

import dataclasses
import types
import typing as tp
from collections import ChainMap as TypingChainMap
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING

import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree, PyTreeDef

from .key_paths import canonical_key, derive_key_path, ensure_public_key

KeyPath: tp.TypeAlias = tuple[tp.Any, ...]
V = tp.TypeVar("V", bound=ArrayLike)

SegmentKeyPaths: tp.TypeAlias = tp.Mapping[KeyPath, KeyPath]
TreeFlattenMetadata: tp.TypeAlias = tuple[
    tuple[KeyPath, ...],
    object,
    tp.Mapping[object, PyTreeDef | None],
    tp.Mapping[object, frozenset[KeyPath]],
    tp.Mapping[object, SegmentKeyPaths],
]
TreeFlattenResult: tp.TypeAlias = tuple[
    list[V],
    TreeFlattenMetadata,
    tuple[KeyPath, ...],
]


@dataclasses.dataclass(slots=True, frozen=True)
class _SegmentRecord(tp.Generic[V]):
    """Lightweight container describing a single flattened segment."""

    treedef: PyTreeDef | None
    keys: frozenset[KeyPath]
    values: dict[KeyPath, V]
    key_paths: dict[KeyPath, KeyPath]
    order: tuple[KeyPath, ...]

    def copy(self) -> _SegmentRecord[V]:
        return _SegmentRecord(
            self.treedef,
            frozenset(self.keys),
            dict(self.values),
            dict(self.key_paths),
            tuple(self.order),
        )


class _SegmentKeyPathsView(Mapping[KeyPath, KeyPath]):
    """Immutable view over stored key-path metadata with ordering information."""

    __slots__ = ("_data", "order")

    def __init__(
        self,
        key_paths: tp.Mapping[KeyPath, KeyPath],
        order: tuple[KeyPath, ...] | None,
    ) -> None:
        normalized_paths = {key: tuple(path) for key, path in key_paths.items()}
        self._data = types.MappingProxyType(normalized_paths)
        self.order = tuple(order) if order is not None else None

    def __getitem__(self, key: KeyPath) -> KeyPath:
        return self._data[key]

    def __iter__(self) -> tp.Iterator[KeyPath]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> tp.ItemsView[KeyPath, KeyPath]:
        return self._data.items()

    def keys(self) -> tp.KeysView[KeyPath]:
        return self._data.keys()

    def values(self) -> tp.ValuesView[KeyPath]:
        return self._data.values()


class FlatState(Mapping[KeyPath, V], tp.Generic[V]):
    """Immutable mapping of flattened pytree values with rich metadata.

    The state tracks ordered segment records that store the original treedef,
    the owned keys, and the associated values. A read-only ``ChainMap`` provides
    the public mapping interface while preserving per-segment isolation.

    Examples:
        >>> state = FlatState.from_pytree({"a": 1, "b": 2})
        >>> state[("a",)]
        1
        >>> state.to_pytree()
        {'a': 1, 'b': 2}
    """

    __slots__ = (
        "__dict__",
        "_mapping",
        "_primary_segment",
        "_segment_order",
        "_segments",
    )
    __hash__ = None  # type: ignore[assignment]

    if TYPE_CHECKING:
        _mapping: TypingChainMap[KeyPath, V]
        _segments: dict[object, _SegmentRecord[V]]
        _segment_order: list[object]
        _primary_segment: object

    def __new__(cls, *args, **kwargs):
        del args, kwargs  # unused
        message = (
            "'FlatState' should never be directly instantiated, use "
            "'FlatState.from_pytree' instead"
        )
        raise TypeError(message)

    @classmethod
    def _new(
        cls: type[FlatState[V]],
        mapping: tp.Mapping[KeyPath, V] | tp.Any,
        /,
        *,
        treedef: PyTreeDef | None = None,
        key_paths: tp.Mapping[KeyPath, KeyPath] | None = None,
    ) -> FlatState[V]:
        """Construct a ``FlatState`` from a mapping of canonical key tuples.

        Args:
            mapping: Dictionary-like object mapping canonical key tuples (the
                same shape produced by ``ensure_public_key`` /
                ``canonical_key``) to leaf values.
            treedef: Optional ``jax.tree_util.PyTreeDef`` describing the
                original structure of the state slice. Stored so
                ``FlatState.to_pytree`` can reconstruct the source pytree.
            key_paths: Optional mapping from canonical key tuples to the
                original JAX key-path objects emitted by
                ``tree_flatten_with_path``. If omitted, key paths are
                regenerated with ``derive_key_path``.

        Returns:
            ``FlatState`` instance that owns a single internal slice populated
            with ``mapping``.
        """

        if not isinstance(mapping, Mapping):
            message = (
                f"{mapping!r} is not a mapping. Convert your pytree using "
                "FlatState.from_pytree or pass a mapping directly."
            )
            raise ValueError(message)

        typed_mapping = tp.cast(Mapping[KeyPath, V], mapping)

        if not all(isinstance(k, tuple) for k in typed_mapping):
            message = (
                "FlatState keys must be tuples. Use FlatState.from_pytree or "
                "ensure your mapping uses tuple paths."
            )
            raise ValueError(message)

        self = super().__new__(cls)
        segment_id = object()
        slice_map = dict(typed_mapping)
        if key_paths is None:
            path_map = {key: derive_key_path(key) for key in slice_map}
        else:
            path_map = {key: tuple(path) for key, path in key_paths.items()}
        key_order = tuple(slice_map.keys())
        segment = _SegmentRecord(
            treedef,
            frozenset(slice_map),
            slice_map,
            path_map,
            key_order,
        )

        self._primary_segment = segment_id
        self._segments = {segment_id: segment}
        self._segment_order = [segment_id]
        self._rebuild_mapping()
        _validate_state(self)
        return self

    def _rebuild_mapping(self) -> None:
        """
        Reconstruct the internal ChainMap from the segment records.
        """
        sources = [
            self._segments[segment_id].values
            for segment_id in reversed(self._segment_order)
        ]
        self._mapping = TypingChainMap(*sources)
        if "is_partitioned" in self.__dict__:
            self.__dict__.pop("is_partitioned", None)

    @property
    def n_internal_states(self) -> int:
        return len(self._segment_order)

    @property
    def raw_mapping(self) -> tp.Mapping[KeyPath, V]:
        return types.MappingProxyType(self._mapping)

    @property
    def treedefs(self) -> tp.Mapping[object, PyTreeDef | None]:
        """Mapping of segment identifiers to JAX treedefs."""
        data = {
            segment_id: self._segments[segment_id].treedef
            for segment_id in self._segment_order
        }
        return types.MappingProxyType(data)

    @property
    def own_keys(self) -> tp.Mapping[object, frozenset[KeyPath]]:
        data = {
            segment_id: self._segments[segment_id].keys
            for segment_id in self._segment_order
        }
        return types.MappingProxyType(data)

    @cached_property
    def is_partitioned(self) -> bool:
        """Return True when any segment is missing keys from its original order."""
        return any(
            len(record.values) != len(record.order)
            for record in self._segments.values()
        )

    def get_state(self, segment_id: object) -> FlatState[V]:
        """Return the sub-state registered under a segment identifier.

        Args:
            segment_id: Identifier of the desired sub-state.

        Returns:
            FlatState: New instance containing only entries owned by the
            requested segment.

        Raises:
            KeyError: If ``segment_id`` is not present in the state.
        """
        if segment_id not in self._segments:
            message = f"Tag {segment_id!r} not found in FlatState"
            raise KeyError(message)
        record = self._segments[segment_id]
        flat_state = object.__new__(type(self))
        flat_state._primary_segment = segment_id
        copied_record = record.copy()
        flat_state._segments = {segment_id: copied_record}
        flat_state._segment_order = [segment_id]
        flat_state._rebuild_mapping()
        _validate_state(flat_state)
        return flat_state

    def copy(self) -> FlatState[V]:
        """Return a shallow copy of the state preserving metadata."""
        new = object.__new__(type(self))
        new._primary_segment = self._primary_segment
        new._segment_order = list(self._segment_order)
        new._segments = {
            segment_id: record.copy() for segment_id, record in self._segments.items()
        }
        new._rebuild_mapping()
        _validate_state(new)
        return new

    def __getitem__(self, key: KeyPath) -> V:
        normalized_key: KeyPath = ensure_public_key(key)
        return self._mapping[normalized_key]

    def __setitem__(self, key: KeyPath, value: V) -> None:
        del key, value  # unused
        message = "FlatState is immutable"
        raise NotImplementedError(message)

    def __delitem__(self, key: KeyPath) -> None:
        del key  # unused
        message = "FlatState is immutable"
        raise NotImplementedError(message)

    def __iter__(self) -> tp.Iterator[KeyPath]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"FlatState({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatState):
            return False
        return self._mapping == other._mapping

    def to_dict(self, sep: object | None = None) -> dict[tp.Any, V]:
        """Convert the flattened mapping to a dictionary.

        Args:
            sep: Optional separator used to join tuple keys into strings. When
                ``None`` the tuple keys are preserved.

        Returns:
            Dictionary representation of the flattened mapping.

        Raises:
            ValueError: If ``sep`` is not ``None`` or a string.
        """
        if isinstance(sep, str):
            return {sep.join(map(str, k)): v for k, v in self._mapping.items()}
        if sep is None:
            return dict(self._mapping.items())
        message = "sep must be a string or None (use '.' or '/' etc)."
        raise ValueError(message)

    @classmethod
    def from_pytree(
        cls: type[FlatState[V]],
        pytree: PyTree | FlatState[V],
    ) -> FlatState[V]:
        """Create a ``FlatState`` from a pytree or another ``FlatState``.

        Args:
            pytree: Source pytree to flatten or an existing ``FlatState``.

        Returns:
            New ``FlatState`` instance containing the flattened values.
        """
        if isinstance(pytree, FlatState):
            return pytree.copy()
        path_leaves, treedef = jtu.tree_flatten_with_path(pytree)
        mapping_dict: dict[KeyPath, V] = {}
        path_map: dict[KeyPath, KeyPath] = {}
        for path, value in path_leaves:
            public_key = canonical_key(path)
            mapping_dict[public_key] = value
            path_map[public_key] = tuple(path)
        return cls._new(mapping_dict, treedef=treedef, key_paths=path_map)

    def to_pytree(self, treedef: PyTreeDef | None = None) -> PyTree:
        """Rebuild the original pytree representation of the state.

        Args:
            treedef: Optional ``jax.tree_util.PyTreeDef`` that determines the
                structure of the reconstructed tree. When omitted the primary
                segment's treedef is used.

        Returns:
            Pytree containing the state values arranged according to
            ``treedef``.

        Raises:
            ValueError: If a treedef cannot be determined for the conversion.
        """
        if treedef is not None:
            return jtu.tree_unflatten(treedef, self._mapping.values())
        if self.n_internal_states != 1:
            message = (
                f"Cannot convert to pytree with {self.n_internal_states} internal "
                "states. Use 'split_state' first."
            )
            raise ValueError(message)
        # get the only treedef
        treedef = self._segments[self._primary_segment].treedef
        if treedef is None:
            message = (
                "Cannot convert to pytree without treedef. Ensure the state was "
                "created via FlatState.from_pytree or carries a treedef."
            )
            raise ValueError(message)
        if self.is_partitioned:
            message = (
                "Cannot convert a partitioned FlatState to a pytree. Combine "
                "the partitions first using 'combine_partitions'."
            )
            raise ValueError(message)
        return jtu.tree_unflatten(treedef, self._mapping.values())

    def tree_flatten(self) -> TreeFlattenResult:
        """Compatibility wrapper delegating to :func:`tree_flatten`."""
        return tree_flatten(self)

    @classmethod
    def tree_unflatten(
        cls: type[FlatState[V]],
        metadata: TreeFlattenMetadata,
        children: Iterable[V],
    ) -> FlatState[V]:
        """Compatibility wrapper delegating to :func:`tree_unflatten`."""
        return tree_unflatten(metadata, children, cls=cls)


def map_state(
    fn: tp.Callable[[KeyPath, V], V],
    state: FlatState[V],
) -> FlatState[V]:
    """Apply a callable to every value in a ``FlatState``.

    Args:
        fn: Callable receiving ``(key, value)`` pairs and returning the updated
            value.
        state: Source state to transform.

    Returns:
        New ``FlatState`` containing the transformed values.
    """
    flat_state = state.copy()
    transformed_values = {p: fn(p, v) for p, v in state.items()}
    for segment_id in flat_state._segment_order:
        record = flat_state._segments[segment_id]
        for key in list(record.values.keys()):
            record.values[key] = transformed_values[key]
    flat_state._rebuild_mapping()
    _validate_state(flat_state)
    return flat_state


def merge_states(*states: FlatState[V]) -> FlatState[V]:
    """Merge one or more ``FlatState`` objects into a combined state.

    Args:
        *states: Ordered sequence of states to merge.

    Returns:
        ``FlatState`` whose segments equal the concatenation of the inputs.

    Warning:
        Later segments overwrite earlier ones for overlapping keys. Normalize
        values first (e.g. via ``apply_transformations``) when shadowing is not
        desired.

    Raises:
        ValueError: If no states are provided or the merge would duplicate a
            segment identifier.
        ValueError: If a segment is merged more than once.

    Examples:
        >>> s1 = FlatState.from_pytree({"a": 1})
        >>> s2 = FlatState.from_pytree({"b": 2})
        >>> merged = merge_states(s1, s2)
        >>> merged.n_internal_states
        2
        >>> list(merged.raw_mapping.keys())
        [('a',), ('b',)]

    Note:
        If multiple states contain the same key, the value from the last state
        in ``*states`` takes precedence. Ensure duplicates carry the same value
        unless you explicitly want later states to override earlier ones.
    """

    def _imerge(this: FlatState[V], other: tp.Any) -> FlatState[V]:
        if not isinstance(other, FlatState):
            message = (
                "Can only merge FlatState instances. Convert inputs using "
                "FlatState.from_pytree first."
            )
            raise TypeError(message)
        other_state = tp.cast(FlatState[V], other)
        for segment_id in other_state._segment_order:
            if segment_id in this._segments:
                message = (
                    "One of the segments has already been merged into this FlatState. "
                    "Did you merge the same state twice?"
                )
                raise ValueError(message)
            this._segments[segment_id] = other_state._segments[segment_id].copy()
        this._segment_order.extend(other_state._segment_order)
        this._rebuild_mapping()
        return this

    if len(states) == 0:
        message = "merge_states() requires at least one FlatState."
        raise ValueError(message)
    state, *rest = states
    out = state.copy()
    for state in rest:
        out = _imerge(out, state)
    _validate_state(out)
    return out


def split_state(state: FlatState[V]) -> tuple[FlatState[V], ...]:
    """Split a merged ``FlatState`` back into its constituent states.

    Args:
        state: Combined state to split.

    Returns:
        Tuple whose elements correspond to the original segments.

    Examples:
        >>> merged = merge_states(
        ...     FlatState.from_pytree({"a": 1}),
        ...     FlatState.from_pytree({"b": 2}),
        ... )
        >>> first, second = split_state(merged)
        >>> first.to_pytree(), second.to_pytree()
        ({'a': 1}, {'b': 2})
    """
    return tuple(state.get_state(segment_id) for segment_id in state._segment_order)


def update_state(
    state: FlatState[V],
    updates: Mapping[KeyPath, V],
) -> FlatState[V]:
    """Return a new state with updates applied to existing keys.

    Args:
        state: Original state to update.
        updates: Mapping of key tuples to replacement values.

    Returns:
        ``FlatState`` copy reflecting the requested updates.

    Raises:
        KeyError: If any key in ``updates`` does not exist in ``state``.

    Examples:
        >>> state = FlatState.from_pytree({"a": 1, "b": 2})
        >>> updated = update_state(state, {("a",): 42})
        >>> updated.to_pytree()
        {'a': 42, 'b': 2}
    """
    new_state = state.copy()

    if len(updates) == 0:
        return new_state

    for raw_key, value in updates.items():
        key: KeyPath = ensure_public_key(raw_key)
        found = False
        for segment_id in new_state._segment_order:
            segment = new_state._segments[segment_id]
            if key in segment.values:
                segment.values[key] = value
                found = True
        if not found:
            message = (
                f"Key {key!r} not present in FlatState. Check the flattened "
                "path before attempting to update."
            )
            raise KeyError(message)

    new_state._rebuild_mapping()
    _validate_state(new_state)
    return new_state


def _subset_segment(
    record: _SegmentRecord[V],
    keys: tp.AbstractSet[KeyPath],
) -> _SegmentRecord[V]:
    order = record.order
    values = {
        key: record.values[key] for key in order if key in keys and key in record.values
    }
    key_paths = {key: record.key_paths[key] for key in values}
    return _SegmentRecord(
        record.treedef,
        frozenset(values),
        values,
        key_paths,
        order,
    )


def _merge_partition_records(
    first: _SegmentRecord[V],
    second: _SegmentRecord[V],
) -> _SegmentRecord[V]:
    if first.treedef != second.treedef:
        message = (
            "Partitions carry different treedef metadata. Combine matching pairs "
            "produced from the same source state."
        )
        raise ValueError(message)

    order_first = first.order
    order_second = second.order
    if order_first != order_second:
        message = (
            "Partitions carry mismatched key-order metadata. Recreate them from "
            "the same FlatState instance."
        )
        raise ValueError(message)

    overlap = first.keys & second.keys
    if overlap:
        overlap_str = ", ".join(sorted(map(str, overlap)))
        message = f"Partitions share duplicate keys: {overlap_str}"
        raise ValueError(message)

    union_keys = first.keys | second.keys
    extra = union_keys - set(order_first)
    if extra:
        extra_str = ", ".join(sorted(map(str, extra)))
        message = (
            "Partitions reference keys absent from the original order: "
            f"{extra_str}. Re-create the partitions from the same FlatState."
        )
        raise ValueError(message)

    values: dict[KeyPath, V] = {}
    key_paths: dict[KeyPath, KeyPath] = {}
    for key in order_first:
        if key in first.values:
            values[key] = first.values[key]
            key_paths[key] = first.key_paths[key]
        elif key in second.values:
            values[key] = second.values[key]
            key_paths[key] = second.key_paths[key]
        else:
            continue

    return _SegmentRecord(
        first.treedef, frozenset(values), values, key_paths, order_first
    )


def partition_state(
    state: FlatState[V],
    /,
    *,
    keys: Iterable[KeyPath] | None = None,
    predicate: tp.Callable[[KeyPath, V], bool] | None = None,
) -> tuple[FlatState[V], FlatState[V]]:
    """Split a state into two orthogonal ``FlatState`` instances.

    Exactly one of ``keys`` or ``predicate`` must be provided. When ``keys`` is
    supplied the matching keys will populate the first partition. When
    ``predicate`` is supplied it is evaluated for every ``(key, value)`` pair and
    the matching entries will populate the first partition.

    The resulting partitions retain the original segment metadata (including
    treedefs and key ordering) but do not contain all values. Operations such as
    ``to_pytree`` will raise guidance errors until the partitions are combined
    again.

    Examples:
        >>> state = FlatState.from_pytree({"a": 1, "b": 2, "c": 3})
        >>> evens, odds = partition_state(state, predicate=lambda key, _v: key[0] in {"a", "c"})
        >>> dict(evens.raw_mapping)
        {('a',): 1, ('c',): 3}
        >>> dict(odds.raw_mapping)
        {('b',): 2}
    """
    if (keys is None) == (predicate is None):
        message = "Provide exactly one of 'keys' or 'predicate'."
        raise ValueError(message)

    raw = state.raw_mapping
    all_keys = set(raw.keys())

    if keys is not None:
        selected_set = {ensure_public_key(key) for key in keys}
        missing = selected_set - all_keys
        if missing:
            missing_keys = ", ".join(sorted(map(str, missing)))
            message = f"Keys {missing_keys} are not present in the FlatState."
            raise KeyError(message)
    else:
        assert predicate is not None
        selected_set = {key for key in all_keys if predicate(key, raw[key])}

    remaining_set = all_keys - selected_set

    def _build_partition(target_keys: set[KeyPath]) -> FlatState[V]:
        partition = object.__new__(type(state))
        partition._primary_segment = state._primary_segment
        partition._segment_order = list(state._segment_order)
        partition._segments = {
            segment_id: _subset_segment(state._segments[segment_id], target_keys)
            for segment_id in state._segment_order
        }
        partition._rebuild_mapping()
        return partition

    selected_state = _build_partition(selected_set)
    remainder_state = _build_partition(remaining_set)
    return selected_state, remainder_state


def combine_partitions(
    first: FlatState[V],
    second: FlatState[V],
    /,
) -> FlatState[V]:
    """Combine two orthogonal ``FlatState`` partitions into a single state.

    Partitions must originate from the same source state. Their stored segment
    metadata is used to restore the original structure and ordering.

    Examples:
        >>> state = FlatState.from_pytree({"a": 1, "b": 2})
        >>> favs, rest = partition_state(state, keys=[("a",)])
        >>> restored = combine_partitions(favs, rest)
        >>> restored.to_pytree()
        {'a': 1, 'b': 2}
    """
    if first._segment_order != second._segment_order:
        message = (
            "Cannot combine partitions with different segment ordering. Ensure "
            "both partitions were produced from the same FlatState."
        )
        raise ValueError(message)

    combined = object.__new__(type(first))
    combined._primary_segment = first._primary_segment
    combined._segment_order = list(first._segment_order)
    combined._segments = {
        segment_id: _merge_partition_records(
            first._segments[segment_id],
            second._segments[segment_id],
        )
        for segment_id in combined._segment_order
    }

    combined._rebuild_mapping()
    _validate_state(combined)
    return combined


def _validate_state(state: FlatState[V]) -> None:
    """Check that the ``FlatState`` metadata remains consistent.

    Args:
        state: State instance to validate.

    Raises:
        ValueError: If any structural invariants are violated.
    """
    segment_ids = set(state._segments)
    if set(state._segment_order) != segment_ids:
        message = (
            "Segment order metadata is inconsistent. This indicates the state "
            "was modified outside the supported APIs."
        )
        raise ValueError(message)
    if len(state._segment_order) != len(segment_ids):
        message = (
            "Duplicate segment identifiers detected. This often means the same "
            "state was merged multiple times."
        )
        raise ValueError(message)
    mapping_keys = set(state._mapping)
    union_keys: set[KeyPath] = set()
    for segment_id in state._segment_order:
        record = state._segments[segment_id]
        tag_keys = set(record.keys)
        missing_in_mapping = tag_keys - mapping_keys
        if missing_in_mapping:
            message = (
                f"Segment {segment_id!r} references keys missing from the mapping: "
                f"{sorted(missing_in_mapping)!r}. Ensure updates only target "
                "existing keys."
            )
            raise ValueError(message)
        union_keys.update(tag_keys)
        slice_keys = set(record.values.keys())
        if slice_keys != tag_keys:
            message = (
                f"Segment {segment_id!r} has mismatched slice keys: "
                f"{sorted(slice_keys ^ tag_keys)!r}. Avoid manual mutation of "
                "FlatState internals."
            )
            raise ValueError(message)
        path_keys = set(record.key_paths.keys())
        if path_keys != tag_keys:
            message = (
                f"Segment {segment_id!r} has mismatched key path metadata. "
                "Ensure key renames also update associated key paths."
            )
            raise ValueError(message)
    missing_from_tags = mapping_keys - union_keys
    if missing_from_tags:
        message = (
            "State mapping contains keys not owned by any segment. This usually "
            "means values were added without updating segment metadata."
        )
        raise ValueError(message)


def _key_path_for(state: FlatState[V], key: KeyPath) -> KeyPath:
    """Look up the stored JAX key path for a public key."""
    for segment_id in reversed(state._segment_order):
        record = state._segments[segment_id]
        if key in record.key_paths:
            return record.key_paths[key]
    return derive_key_path(key)


def _gather_leaf_key_paths(state: FlatState[V]) -> dict[KeyPath, KeyPath]:
    """Collect the internal key-path mapping for all leaves."""
    return {key: _key_path_for(state, key) for key in state._mapping}


def tree_flatten(state: FlatState[V]) -> TreeFlattenResult:
    """Return flattened values and metadata for a ``FlatState`` instance."""
    keys = tuple(state._mapping)
    treedefs_map: dict[object, PyTreeDef | None] = {}
    own_keys_map: dict[object, frozenset[KeyPath]] = {}
    key_paths_map: dict[object, _SegmentKeyPathsView] = {}
    for segment_id in state._segment_order:
        record = state._segments[segment_id]
        treedefs_map[segment_id] = record.treedef
        own_keys_map[segment_id] = record.keys
        key_paths_map[segment_id] = _SegmentKeyPathsView(record.key_paths, record.order)

    metadata: TreeFlattenMetadata = (
        keys,
        state._primary_segment,
        types.MappingProxyType(treedefs_map),
        types.MappingProxyType(own_keys_map),
        types.MappingProxyType(key_paths_map),
    )
    return ([state[key] for key in keys], metadata, keys)


def tree_unflatten(
    metadata: TreeFlattenMetadata,
    children: Iterable[V],
    *,
    cls: type[FlatState[V]] = FlatState,
) -> FlatState[V]:
    """Reconstruct a ``FlatState`` from flatten metadata."""
    keys, primary_segment, treedefs, own_keys, key_paths = metadata

    values_list = list(children)
    if len(values_list) != len(keys):
        message = "Tree flatten metadata has mismatched lengths"
        raise ValueError(message)

    flat_mapping: dict[KeyPath, V] = dict(zip(keys, values_list, strict=True))
    flat_state = tp.cast(FlatState[V], object.__new__(cls))
    flat_state._primary_segment = primary_segment
    flat_state._segment_order = list(own_keys.keys())
    flat_state._segments = {}

    for segment_id in flat_state._segment_order:
        owned_keys = own_keys[segment_id]
        record_treedef = treedefs[segment_id]

        stored_key_paths = key_paths.get(segment_id)
        if stored_key_paths is None:
            segment_key_paths = {key: derive_key_path(key) for key in owned_keys}
            stored_order: tuple[KeyPath, ...] | None = None
        else:
            segment_key_paths = {
                key: tuple(path) for key, path in stored_key_paths.items()
            }
            for key in owned_keys:
                segment_key_paths.setdefault(key, derive_key_path(key))
            stored_order = getattr(stored_key_paths, "order", None)

        if stored_order is not None:
            order = tuple(stored_order)
        else:
            order = tuple(key for key in keys if key in owned_keys)

        segment_values = {key: flat_mapping[key] for key in order if key in owned_keys}
        flat_state._segments[segment_id] = _SegmentRecord(
            record_treedef,
            frozenset(owned_keys),
            segment_values,
            segment_key_paths,
            order,
        )

    flat_state._rebuild_mapping()
    _validate_state(flat_state)
    return flat_state


def _flatstate_flatten(state: FlatState[V]) -> tuple[list[V], TreeFlattenMetadata]:
    """Wrapper used by JAX pytree registration."""

    values, metadata, _ = tree_flatten(state)
    return values, metadata


def _flatstate_flatten_with_keys(
    state: FlatState[V],
) -> tuple[list[tuple[KeyPath, V]], TreeFlattenMetadata]:
    """Return flattened pairs of key paths and values for JAX."""
    values, metadata, keys = tree_flatten(state)
    key_lookup = _gather_leaf_key_paths(state)
    key_children = [
        (key_lookup[key], value) for key, value in zip(keys, values, strict=True)
    ]
    return key_children, metadata


def _flatstate_unflatten(
    metadata: TreeFlattenMetadata,
    children: Iterable[V],
) -> FlatState[V]:
    """Inverse of :func:`_flatstate_flatten` for JAX pytree support."""
    return tree_unflatten(metadata, children, cls=FlatState)


# Register FlatState as a JAX pytree node
jtu.register_pytree_node(  # type: ignore[arg-type]
    FlatState,
    _flatstate_flatten,
    _flatstate_unflatten,
    _flatstate_flatten_with_keys,
)
