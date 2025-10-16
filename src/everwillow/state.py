"""State management for parameter pytrees using optree."""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

import optree

K = tp.TypeVar("K", bound=tuple)
V = tp.TypeVar("V")

PyTree: tp.TypeAlias = tp.Any
PyTreeDef: tp.TypeAlias = tp.Any


namespace = "everwillow"


@optree.register_pytree_node_class(namespace=namespace)
class ParamState(Mapping[K, V]):
    """
    Flattened parameter state for inference operations.

    Wraps a parameter pytree and provides utilities for partitioning (free/fixed),
    merging, and other inference operations.
    """

    __slots__ = ("_mapping", "_treedef")

    TREE_PATH_ENTRY_TYPE = optree.MappingEntry

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        raise TypeError(
            "'ParamState' should never be directly instantiated, use 'ParamState.from_pytree' instead"
        )

    @classmethod
    def _new(cls, mapping: tp.Mapping[K, V], /, *, treedef: PyTreeDef | None = None):
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{mapping!r} must be a Mapping")

        if not all(isinstance(k, tuple) for k in mapping.keys()):
            raise ValueError("All keys in mapping must be tuples")

        self = super().__new__(cls)
        self._mapping = dict(mapping)
        self._treedef = treedef
        return self

    @property
    def raw_mapping(self) -> tp.Mapping[K, V]:
        return self._mapping

    def copy(self):
        return type(self)._new(dict(self._mapping), treedef=self._treedef)

    def __getitem__(self, key: K) -> V:
        return self._mapping[key]

    def __setitem__(self, key: K, value: V) -> None:
        raise NotImplementedError("ParamState is immutable")

    def __delitem__(self, key: K) -> None:
        raise NotImplementedError("ParamState is immutable")

    def __iter__(self) -> tp.Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"ParamState({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamState):
            return False
        return self._mapping == other._mapping

    def to_dict(self, sep: str | None = None) -> dict:
        """Convert to dict, optionally with joined keys."""
        if isinstance(sep, str):
            return {sep.join(map(str, k)): v for k, v in self._mapping.items()}
        if sep is None:
            return {k: v for k, v in self._mapping.items()}
        raise ValueError("sep must be a string or None")

    @classmethod
    def from_pytree(cls, pytree: PyTree):
        """Create ParamState from a pytree."""
        keys, flat, treedef = optree.tree_flatten_with_path(pytree)
        return cls._new(dict(zip(keys, flat, strict=True)), treedef=treedef)

    def to_pytree(self) -> PyTree:
        """Convert back to original pytree structure."""
        if self._treedef is None:
            raise ValueError("Cannot convert to pytree with 'None' treedef")
        # Extract values in the same order as original flattening
        # The treedef knows the correct key order via paths()
        keys_order = self._treedef.paths()
        values = [self._mapping[key] for key in keys_order]
        return optree.tree_unflatten(self._treedef, values)

    def tree_flatten(self):
        """For optree registration."""
        keys = sorted(self.keys())
        return (
            [self[key] for key in keys],  # children
            (keys, self._treedef),  # metadata
            keys,  # entries
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """For optree registration."""
        keys, treedef = metadata
        return cls._new(dict(zip(keys, children, strict=True)), treedef=treedef)


def partition_state(
    state: ParamState, fixed_keys: set[K]
) -> tuple[ParamState, ParamState, PyTreeDef]:
    """
    Partition state into free and fixed parameters.

    Args:
        state: Full parameter state
        fixed_keys: Set of keys (tuples) to treat as fixed

    Returns:
        (free_state, fixed_state, original_treedef)
    """
    free_mapping = {k: v for k, v in state.items() if k not in fixed_keys}
    fixed_mapping = {k: v for k, v in state.items() if k in fixed_keys}

    free_state = ParamState._new(free_mapping)
    fixed_state = ParamState._new(fixed_mapping)

    return free_state, fixed_state, state._treedef


def merge_states(free_state: ParamState, fixed_state: ParamState) -> ParamState:
    """
    Merge free and fixed states back together.

    Args:
        free_state: Free parameter state
        fixed_state: Fixed parameter state

    Returns:
        Merged state
    """
    merged_mapping = {**fixed_state._mapping, **free_state._mapping}
    return ParamState._new(merged_mapping, treedef=None)


def update_state(state: ParamState, updates: tp.Mapping[K, V]) -> ParamState:
    """
    Create new state with updated values.

    Args:
        state: Original state
        updates: Dict of {key: new_value} to update

    Returns:
        New state with updates applied
    """
    new_mapping = {**state._mapping, **updates}
    return ParamState._new(new_mapping, treedef=state._treedef)
