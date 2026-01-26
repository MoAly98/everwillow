"""Tests for the :mod:`everwillow.statelib.state` helpers."""

from __future__ import annotations

import typing as tp

import pytest

import everwillow.statelib as sl

FState: tp.TypeAlias = sl.State[float]


def test_state_roundtrip_preserves_structure() -> None:
    """Flattening and rehydrating a pytree keeps the same layout."""
    tree = {"a": 1.0, "b": {"c": 2.0}}
    state: FState = sl.State.from_pytree(tree, sep=None)

    assert dict(state.mapping) == {("a",): 1.0, ("b", "c"): 2.0}
    assert state.to_pytree() == tree


def test_state_behaves_like_mapping() -> None:
    """``State`` exposes mapping operations for convenience."""
    state: FState = sl.State.from_pytree({"x": {"y": 3.0}}, sep=None)

    assert ("x", "y") in state
    assert state["x", "y"] == 3.0
    assert list(state.keys()) == [("x", "y")]


def test_merge_and_split_restore_inputs() -> None:
    """Merged mappings can be split back into the original states."""
    state_a: FState = sl.State.from_pytree({"a": 1.0}, sep=None)
    state_b: FState = sl.State.from_pytree({"b": {"c": 2.0}}, sep=None)

    merged = sl.merge(state_a, state_b)
    assert merged["a",] == 1.0
    assert merged["b", "c"] == 2.0

    restored_a, restored_b = sl.split(merged)
    assert restored_a.to_pytree() == {"a": 1.0}
    assert restored_b.to_pytree() == {"b": {"c": 2.0}}


def test_update_replaces_only_existing_keys() -> None:
    """Updating a state yields a new instance with selected keys replaced."""
    state: FState = sl.State.from_pytree({"a": 1.0, "b": 2.0}, sep=None)

    updated = sl.update(state, updates={("b",): 99.0})
    assert updated["b",] == 99.0
    assert state["b",] == 2.0

    with pytest.raises(KeyError):
        sl.update(state, updates={("missing",): 0.0})


def test_partition_and_combine_roundtrip() -> None:
    """Partitioning a mapping and recombining yields the same data."""
    state: FState = sl.State.from_pytree({"a": {"x": 1.0}, "b": 2.0}, sep=None)

    left, right = sl.partition(
        state,
        predicate=lambda key, _value: "a" in key,
    )

    # Partitions now contain None for excluded keys
    assert {k: v for k, v in left.items() if v is not None} == {("a", "x"): 1.0}  # type: ignore[redundant-expr]
    assert {k: v for k, v in right.items() if v is not None} == {("b",): 2.0}  # type: ignore[redundant-expr]

    combined = sl.combine_partitions(left, right)
    # combined is now a State directly
    assert combined.to_pytree() == state.to_pytree()


def test_partition_origin_mismatch_raises() -> None:
    """Merging partitions originating from different mappings is rejected."""
    state_one: FState = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    state_two: FState = sl.State.from_pytree({"a": 5.0, "c": 6.0})

    left_one, _right_one = sl.partition(
        state_one,
        predicate=lambda key, _value: key == ("a",),
    )
    _, right_two = sl.partition(
        state_two,
        predicate=lambda key, _value: key == ("a",),
    )

    with pytest.raises(ValueError, match="same original state"):
        sl.combine_partitions(left_one, right_two)
