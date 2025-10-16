from __future__ import annotations

import typing as tp

import pytest

import statelib as sl


def test_flat_state_roundtrip_to_pytree() -> None:
    tree = {"a": 1, "b": {"c": 2, "d": (3, 4)}}
    state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(tree)

    assert state.n_internal_states == 1
    assert state.to_pytree() == tree
    assert all(isinstance(k, tuple) for k in state.raw_mapping)


def test_flat_state_from_flat_state_returns_copy() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
    clone: sl.FlatState[int] = sl.FlatState.from_pytree(state)

    assert clone == state and clone is not state


def test_raw_mapping_is_read_only() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

    with pytest.raises(TypeError):
        state.raw_mapping[("a",)] = 2  # type: ignore[index]


def test_merge_and_split_roundtrip_allows_overlapping_keys() -> None:
    state1: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})
    state2: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})

    merged = sl.merge_states(state1, state2)
    first, second = sl.split_state(merged)

    assert merged.n_internal_states == 2
    assert first == state1
    assert second == state2


def test_map_state_applies_function_without_mutating_original() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})

    mapped: sl.FlatState[int] = sl.map_state(lambda _k, v: v * 10, state)

    assert state[("a",)] == 1
    assert mapped[("a",)] == 10
    assert mapped.to_pytree() == {"a": 10, "b": 20}


def test_tree_flatten_metadata_is_immutable() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

    _, metadata, _ = state.tree_flatten()
    keys, tag, treedefs, own_keys, key_paths = metadata

    assert isinstance(keys, tuple)
    with pytest.raises(TypeError):
        treedefs[tag] = None  # type: ignore[index]

    owner_keys = own_keys[tag]
    assert isinstance(owner_keys, frozenset)
    with pytest.raises(AttributeError):
        owner_keys.add(("extra",))  # type: ignore[attr-defined]

    path_map = key_paths[tag]
    first_key = next(iter(path_map))
    assert isinstance(path_map[first_key], tuple)
    with pytest.raises(TypeError):
        key_paths[tag] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        path_map[first_key] = ()  # type: ignore[index]


def test_update_state_replaces_values() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
    updated: sl.FlatState[int] = sl.update_state(state, {("a",): 99})

    assert dict(state.raw_mapping) == {("a",): 1, ("b",): 2}
    assert dict(updated.raw_mapping) == {("a",): 99, ("b",): 2}
    assert updated.to_pytree() == {"a": 99, "b": 2}


def test_update_state_missing_key_raises() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
    with pytest.raises(KeyError):
        sl.update_state(state, {("missing",): 0})


def test_update_state_on_merged_state() -> None:
    s1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
    s2: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 3, "c": 4})
    merged: sl.FlatState[int] = sl.merge_states(s1, s2)

    updated: sl.FlatState[int] = sl.update_state(merged, {("a",): 10})
    seg1, seg2 = sl.split_state(updated)

    assert dict(seg1.raw_mapping) == {("a",): 10, ("b",): 2}
    assert dict(seg2.raw_mapping) == {("a",): 10, ("c",): 4}
