from __future__ import annotations

import dataclasses
import types
import typing as tp
from collections import ChainMap as TypingChainMap
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import jax.tree_util as jtu

from .key_paths import canonical_key, derive_key_path, ensure_public_key

KeyPath: tp.TypeAlias = tuple[tp.Any, ...]
V = tp.TypeVar("V")

PyTree: tp.TypeAlias = tp.Any
SegmentKeyPaths: tp.TypeAlias = tp.Mapping[KeyPath, KeyPath]
TreeFlattenMetadata: tp.TypeAlias = tuple[
    tuple[KeyPath, ...],
    object,
    tp.Mapping[object, jtu.PyTreeDef | None],
    dict[object, frozenset[KeyPath]],
    tp.Mapping[object, SegmentKeyPaths],
]
TreeFlattenResult: tp.TypeAlias = tuple[
    list[V],
    TreeFlattenMetadata,
    tuple[KeyPath, ...],
]


@dataclasses.dataclass(slots=True, frozen=True)
class SegmentRecord(tp.Generic[V]):
    """Internal bookkeeping for a single segment of the flat state."""

    treedef: jtu.PyTreeDef | None
    keys: frozenset[KeyPath]
    values: dict[KeyPath, V]
    key_paths: dict[KeyPath, KeyPath]

    def copy(self) -> SegmentRecord[V]:
        return SegmentRecord(
            self.treedef,
            self.keys,
            self.values.copy(),
            self.key_paths.copy(),
        )


class FlatState(Mapping[KeyPath, V], tp.Generic[V]):
    """Immutable mapping of flattened pytree values with rich metadata.

    The state tracks ordered segment records that store the original treedef,
    the owned keys, and the associated values. A read-only ``ChainMap`` provides
    the public mapping interface while preserving per-segment isolation.

    Attributes:
        n_internal_states: Number of registered segments.
        raw_mapping: Read-only view over the flattened mapping.
        treedefs: Mapping from segment identifiers to ``jax.tree_util.PyTreeDef``.
        own_keys: Mapping from segment identifiers to the keys they own.

    Examples:
        >>> state = FlatState.from_pytree({"a": 1, "b": 2})
        >>> state[("a",)]
        1
        >>> state.to_pytree()
        {'a': 1, 'b': 2}
    """

    __slots__ = ("_mapping", "_primary_segment", "_segment_order", "_segments")
    __hash__ = None  # type: ignore[assignment]

    if TYPE_CHECKING:
        _mapping: TypingChainMap[KeyPath, V]
        _segments: dict[object, SegmentRecord[V]]
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
        treedef: jtu.PyTreeDef | None = None,
        key_paths: tp.Mapping[KeyPath, KeyPath] | None = None,
    ) -> FlatState[V]:
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
        segment = SegmentRecord(
            treedef,
            frozenset(slice_map.keys()),
            slice_map,
            path_map,
        )

        self._primary_segment = segment_id
        self._segments = {segment_id: segment}
        self._segment_order = [segment_id]
        self._rebuild_mapping()
        _validate_state(self)
        return self

    def _rebuild_mapping(self) -> None:
        sources = [
            self._segments[segment_id].values
            for segment_id in reversed(self._segment_order)
        ]
        self._mapping = TypingChainMap(*sources)

    @property
    def n_internal_states(self) -> int:
        return len(self._segment_order)

    @property
    def raw_mapping(self) -> tp.Mapping[KeyPath, V]:
        return types.MappingProxyType(self._mapping)

    @property
    def treedefs(self) -> tp.Mapping[object, jtu.PyTreeDef | None]:
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

    def to_pytree(self, treedef: jtu.PyTreeDef | None = None) -> PyTree:
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
        return jtu.tree_unflatten(treedef, self._mapping.values())

    def tree_flatten(self) -> TreeFlattenResult:
        """Return the flattened values and accompanying metadata.

        Returns:
            Tuple containing the flattened values, metadata required for
            ``tree_unflatten``, and the ordered keys.
        """
        keys = tuple(self._mapping)
        treedefs_map = {
            segment_id: self._segments[segment_id].treedef
            for segment_id in self._segment_order
        }
        own_keys_map = {
            segment_id: self._segments[segment_id].keys
            for segment_id in self._segment_order
        }
        key_paths_map = {
            segment_id: types.MappingProxyType(
                dict(self._segments[segment_id].key_paths),
            )
            for segment_id in self._segment_order
        }
        metadata: TreeFlattenMetadata = (
            keys,
            self._primary_segment,
            types.MappingProxyType(treedefs_map),
            {segment_id: frozenset(own) for segment_id, own in own_keys_map.items()},
            types.MappingProxyType(key_paths_map),
        )
        return ([self[key] for key in keys], metadata, keys)

    @classmethod
    def tree_unflatten(
        cls: type[FlatState[V]],
        metadata: TreeFlattenMetadata,
        children: Iterable[V],
    ) -> FlatState[V]:
        """Reconstruct a ``FlatState`` from flatten metadata.

        Args:
            metadata: Metadata tuple produced by ``tree_flatten``.
            children: Iterable of values matching the flattened order.

        Returns:
            Reconstructed ``FlatState``.

        Raises:
            ValueError: If the number of provided children does not match the
                metadata.
        """
        keys, primary_segment, treedefs, own_keys, key_paths = metadata
        value_list = list(children)
        if len(value_list) != len(keys):
            message = "Tree flatten metadata has mismatched lengths"
            raise ValueError(message)
        flat_mapping: dict[KeyPath, V] = dict(zip(keys, value_list, strict=True))
        flat_state = object.__new__(cls)
        flat_state._primary_segment = primary_segment
        flat_state._segment_order = list(own_keys.keys())
        flat_state._segments = {}
        for segment_id, keys_set in own_keys.items():
            record_treedef = treedefs[segment_id]
            key_set = frozenset(keys_set)
            values = {key: flat_mapping[key] for key in key_set}
            stored_key_paths = key_paths.get(segment_id)
            if stored_key_paths is None:
                segment_key_paths = {key: derive_key_path(key) for key in key_set}
            else:
                segment_key_paths = {
                    key: tuple(path) for key, path in stored_key_paths.items()
                }
                for key in key_set:
                    segment_key_paths.setdefault(key, derive_key_path(key))
            flat_state._segments[segment_id] = SegmentRecord(
                record_treedef,
                key_set,
                values,
                segment_key_paths,
            )
        flat_state._rebuild_mapping()
        _validate_state(flat_state)
        return flat_state


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

    Raises:
        ValueError: If no states are provided or the merge would duplicate a
            segment identifier.

    Examples:
        >>> s1 = FlatState.from_pytree({"a": 1})
        >>> s2 = FlatState.from_pytree({"b": 2})
        >>> merged = merge_states(s1, s2)
        >>> merged.n_internal_states
        2
        >>> list(merged.raw_mapping.keys())
        [('a',), ('b',)]
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


def _flatstate_flatten(state: FlatState[V]) -> tuple[list[V], TreeFlattenMetadata]:
    """Wrapper used by JAX pytree registration."""
    values, metadata, _ = state.tree_flatten()
    return values, metadata


def _flatstate_flatten_with_keys(
    state: FlatState[V],
) -> tuple[list[tuple[KeyPath, V]], TreeFlattenMetadata]:
    """Return flattened pairs of key paths and values for JAX."""
    values, metadata, keys = state.tree_flatten()
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
    return FlatState.tree_unflatten(metadata, children)


jtu.register_pytree_node(  # type: ignore[arg-type]
    FlatState,
    _flatstate_flatten,
    _flatstate_unflatten,
    _flatstate_flatten_with_keys,
)
