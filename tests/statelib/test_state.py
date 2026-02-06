"""Tests for the :mod:`everwillow.statelib.state` helpers."""

from __future__ import annotations

import pytest

import everwillow.statelib as sl

# -- State creation / inspection --


def test_state_roundtrip_preserves_structure() -> None:
    """Flattening and rehydrating a pytree keeps the same layout."""
    tree = {"a": 1.0, "b": {"c": 2.0}}
    state = sl.State.from_pytree(tree, sep=None)

    assert state.to_dict() == {("a",): 1.0, ("b", "c"): 2.0}
    assert state.to_pytree() == tree


def test_state_behaves_like_mapping() -> None:
    """``State`` exposes mapping operations for convenience."""
    state = sl.State.from_pytree({"x": {"y": 3.0}}, sep=None)

    assert ("x", "y") in state
    assert state["x", "y"] == 3.0
    assert list(state.keys()) == [("x", "y")]


def test_from_pytree_rejects_state_input() -> None:
    """Passing an existing State to ``from_pytree`` raises TypeError."""
    state = sl.State.from_pytree({"a": 1.0})

    with pytest.raises(TypeError, match="already a State"):
        sl.State.from_pytree(state)


def test_from_pytree_with_sep_returns_string_keys() -> None:
    """Using ``sep`` joins key path entries into a single string."""
    tree = {"a": 1.0, "b": {"c": 2.0}}
    state = sl.State.from_pytree(tree, sep=".")

    assert state.to_dict() == {"a": 1.0, "b.c": 2.0}


def test_from_pytree_with_list_input() -> None:
    """Lists produce integer-indexed keys."""
    state = sl.State.from_pytree({"a": [10, 20]})

    assert state.to_dict() == {("a", 0): 10, ("a", 1): 20}


def test_from_pytree_with_tuple_input() -> None:
    """Tuples produce integer-indexed keys like lists."""
    state = sl.State.from_pytree({"a": (10, 20)})

    assert state.to_dict() == {("a", 0): 10, ("a", 1): 20}


def test_from_pytree_with_namedtuple_input() -> None:
    """NamedTuples produce attribute-named keys."""
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    tree = {"p": Point(x=1.0, y=2.0)}
    state = sl.State.from_pytree(tree)

    assert state.to_dict() == {("p", "x"): 1.0, ("p", "y"): 2.0}


def test_from_pytree_with_nested_sequences() -> None:
    """Nested lists/tuples produce compound integer-indexed keys."""
    state = sl.State.from_pytree([[1, 2], [3, 4]])

    assert state.to_dict() == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}


def test_from_pytree_with_registered_dataclass() -> None:
    """Registered dataclasses produce FlattenedIndexKey entries."""
    import dataclasses
    from functools import partial

    import jax.tree_util as jtu

    @partial(jtu.register_dataclass, data_fields=["x", "y"], meta_fields=[])
    @dataclasses.dataclass
    class Vec:
        x: float
        y: float

    tree = {"v": Vec(x=1.0, y=2.0)}
    state = sl.State.from_pytree(tree)

    assert state.to_dict() == {("v", "x"): 1.0, ("v", "y"): 2.0}


def test_from_pytree_with_tuple_dict_keys() -> None:
    """Dict keys that are tuples get flattened into the canonical key."""
    tree = {("a", "b"): 1.0, ("c",): 2.0}
    state = sl.State.from_pytree(tree)

    assert state.to_dict() == {("a", "b"): 1.0, ("c",): 2.0}


def test_from_pytree_with_nested_tuple_dict_keys() -> None:
    """Nested tuple dict keys get fully flattened."""
    tree = {(("a", "b"), "c"): 1.0}
    state = sl.State.from_pytree(tree)

    assert state.to_dict() == {("a", "b", "c"): 1.0}


def test_from_pytree_with_string_dict_keys() -> None:
    """String dict keys are not character-iterated."""
    tree = {"abc": 1.0}
    state = sl.State.from_pytree(tree)

    assert state.to_dict() == {("abc",): 1.0}


def test_notnone_filters_none_values() -> None:
    """The ``.notnone`` property excludes keys with None values."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    left, _right = sl.partition(state, predicate=lambda k, _v: k == ("a",))

    assert dict(left.notnone) == {("a",): 1.0}


def test_state_repr() -> None:
    """``repr(state)`` uses the ``State({...})`` format."""
    state = sl.State.from_pytree({"a": 1.0})

    assert repr(state) == f"State({state.to_dict()!r})"


# -- Update --


def test_update_replaces_values() -> None:
    """Updating a state returns a new instance with replaced values."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0, "c": 3.0})

    updated = sl.update(state, updates={("a",): 10.0, ("c",): 30.0})

    assert updated.to_dict() == {("a",): 10.0, ("b",): 2.0, ("c",): 30.0}
    # Original unchanged
    assert state.to_dict() == {("a",): 1.0, ("b",): 2.0, ("c",): 3.0}


def test_update_ellipsis_skips_key() -> None:
    """Entries with Ellipsis as value are left unchanged."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})

    updated = sl.update(state, updates={("a",): ..., ("b",): 99.0})

    assert updated["a",] == 1.0
    assert updated["b",] == 99.0


def test_update_rejects_missing_key() -> None:
    """Updating with a key not in the state raises KeyError."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})

    with pytest.raises(KeyError):
        sl.update(state, updates={("missing",): 0.0})


def test_update_rejects_non_state() -> None:
    """Calling ``update`` on a plain dict raises TypeError."""
    with pytest.raises(TypeError, match="State types"):
        sl.update({"a": 1.0}, updates={})


# -- Merge / Split --


def test_merge_and_split_roundtrip() -> None:
    """Merged states can be split back into the originals."""
    state_a = sl.State.from_pytree({"a": 1.0}, sep=None)
    state_b = sl.State.from_pytree({"b": {"c": 2.0}}, sep=None)

    merged = sl.merge(state_a, state_b)
    assert merged["a",] == 1.0
    assert merged["b", "c"] == 2.0

    restored_a, restored_b = sl.split(merged)
    assert restored_a.to_pytree() == {"a": 1.0}
    assert restored_b.to_pytree() == {"b": {"c": 2.0}}


def test_merge_three_states() -> None:
    """Merging three states and splitting recovers all three."""
    sa = sl.State.from_pytree({"a": 1.0})
    sb = sl.State.from_pytree({"b": 2.0})
    sc = sl.State.from_pytree({"c": 3.0})

    merged = sl.merge(sa, sb, sc)
    assert merged["a",] == 1.0
    assert merged["b",] == 2.0
    assert merged["c",] == 3.0

    ra, rb, rc = sl.split(merged)
    assert ra.to_pytree() == {"a": 1.0}
    assert rb.to_pytree() == {"b": 2.0}
    assert rc.to_pytree() == {"c": 3.0}


def test_merge_overlapping_keys_last_writer_wins() -> None:
    """When states share a key, the merged mapping keeps the last value."""
    sa = sl.State.from_pytree({"x": 1.0})
    sb = sl.State.from_pytree({"x": 2.0})

    merged = sl.merge(sa, sb)
    assert merged["x",] == 2.0


def test_split_overlapping_keys_uses_merged_value() -> None:
    """After merge, split returns the merged value for overlapping keys in both segments."""
    sa = sl.State.from_pytree({"x": 1.0})
    sb = sl.State.from_pytree({"x": 2.0})

    merged = sl.merge(sa, sb)
    ra, rb = sl.split(merged)

    # Both segments get the merged value (last writer wins)
    assert ra["x",] == 2.0
    assert rb["x",] == 2.0


def test_merge_rejects_single_state() -> None:
    """Merging a single state raises ValueError."""
    sa = sl.State.from_pytree({"a": 1.0})

    with pytest.raises(ValueError, match="at least two"):
        sl.merge(sa)


def test_merge_rejects_empty_args() -> None:
    """Merging with no arguments raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        sl.merge()


def test_split_rejects_non_merged_state() -> None:
    """Splitting a state not produced by merge raises ValueError."""
    state = sl.State.from_pytree({"a": 1.0})

    with pytest.raises(ValueError, match="merge"):
        sl.split(state)


# -- Partition / Combine --


def test_partition_and_combine_roundtrip() -> None:
    """Partitioning a state and recombining yields the same data."""
    state = sl.State.from_pytree({"a": {"x": 1.0}, "b": 2.0}, sep=None)

    left, right = sl.partition(
        state,
        predicate=lambda key, _value: "a" in key,
    )

    # Partitions contain None for excluded keys
    assert dict(left.notnone) == {("a", "x"): 1.0}
    assert dict(right.notnone) == {("b",): 2.0}

    combined = sl.combine_partitions(left, right)
    assert combined.to_pytree() == state.to_pytree()


def test_partition_all_keys_match() -> None:
    """When every key matches the predicate, right has only None values."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    left, right = sl.partition(state, predicate=lambda _k, _v: True)

    assert dict(left.notnone) == {("a",): 1.0, ("b",): 2.0}
    assert dict(right.notnone) == {}


def test_partition_no_keys_match() -> None:
    """When no key matches the predicate, left has only None values."""
    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    left, right = sl.partition(state, predicate=lambda _k, _v: False)

    assert dict(left.notnone) == {}
    assert dict(right.notnone) == {("a",): 1.0, ("b",): 2.0}


def test_combine_partitions_rejects_mismatched_origins() -> None:
    """Combining partitions from different states raises ValueError."""
    state_one = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    state_two = sl.State.from_pytree({"a": 5.0, "c": 6.0})

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
