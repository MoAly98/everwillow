from __future__ import annotations

import typing as tp

import pytest

import everwillow.statelib as sl


def test_flat_state_direct_init_raises() -> None:
    with pytest.raises(
        TypeError,
        match="'FlatState' should never be directly instantiated, use 'FlatState.from_pytree' instead",
    ):
        sl.FlatState(raw_mapping={}, own_keys={}, n_internal_states=1)  # type: ignore[arg-type]


def test_flat_state_roundtrip_to_pytree() -> None:
    tree = {"a": 1, "b": {"c": 2, "d": (3, 4)}}
    state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(tree)

    assert state.n_internal_states == 1
    assert state.to_pytree() == tree
    assert all(isinstance(k, tuple) for k in state.raw_mapping)


def test_flat_state_from_flat_state_returns_copy() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
    clone: sl.FlatState[int] = sl.FlatState.from_pytree(state)

    assert clone == state
    assert clone is not state


def test_raw_mapping_is_read_only() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

    with pytest.raises(TypeError):
        state.raw_mapping["a",] = 2  # type: ignore[index]


def test_merge_and_split_roundtrip_allows_overlapping_keys() -> None:
    state1: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})
    state2: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})

    merged = sl.merge_states(state1, state2)
    first, second = sl.split_state(merged)

    assert merged.n_internal_states == 2
    assert first == state1
    assert second == state2


def test_merge_states_overlapping_keys_last_segment_wins() -> None:
    state1: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 1})
    state2: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 2})

    merged = sl.merge_states(state1, state2)
    assert merged["shared",] == 2
    first, second = sl.split_state(merged)
    assert first["shared",] == 1
    assert second["shared",] == 2


def test_map_state_applies_function_without_mutating_original() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})

    mapped: sl.FlatState[int] = sl.map_state(lambda _k, v: v * 10, state)

    assert state["a",] == 1
    assert mapped["a",] == 10
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
    s2: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "c": 4})
    merged: sl.FlatState[int] = sl.merge_states(s1, s2)

    updated: sl.FlatState[int] = sl.update_state(merged, {("a",): 10})
    seg1, seg2 = sl.split_state(updated)

    assert dict(seg1.raw_mapping) == {("a",): 10, ("b",): 2}
    assert dict(seg2.raw_mapping) == {("a",): 10, ("c",): 4}


def test_partition_state_by_keys_roundtrip() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree(
        {"user": {"name": 1, "email": 2}, "prefs": {"theme": 3, "lang": 4}}
    )
    assert state.is_partitioned is False
    selected, remainder = sl.partition_state(
        state,
        keys={("user", "email"), ("prefs", "theme")},
    )

    assert dict(selected.raw_mapping) == {
        ("user", "email"): 2,
        ("prefs", "theme"): 3,
    }
    assert dict(remainder.raw_mapping) == {
        ("user", "name"): 1,
        ("prefs", "lang"): 4,
    }

    with pytest.raises(ValueError, match="Combine the partitions first"):
        selected.to_pytree()
    with pytest.raises(ValueError, match="Combine the partitions first"):
        remainder.to_pytree()
    assert selected.is_partitioned is True
    assert remainder.is_partitioned is True

    restored = sl.combine_partitions(selected, remainder)
    assert restored == state
    assert restored.to_pytree() == state.to_pytree()
    assert restored.is_partitioned is False


def test_partition_state_by_predicate_roundtrip_without_template() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree(
        {
            "user": {"name": 1, "email": 2},
            "flags": {"beta": 3, "notifications": 4},
        }
    )

    def is_flag(key: tuple[object, ...], _value: int) -> bool:
        return key[0] == "flags"

    flags, rest = sl.partition_state(state, predicate=is_flag)
    merged = sl.combine_partitions(flags, rest)

    assert dict(flags.raw_mapping) == {
        ("flags", "beta"): 3,
        ("flags", "notifications"): 4,
    }
    assert dict(rest.raw_mapping) == {
        ("user", "name"): 1,
        ("user", "email"): 2,
    }
    assert dict(merged.raw_mapping) == dict(state.raw_mapping)
    assert merged.to_pytree() == state.to_pytree()
    with pytest.raises(ValueError, match="Combine the partitions first"):
        flags.to_pytree()
    assert flags.is_partitioned is True
    assert rest.is_partitioned is True
    assert merged.is_partitioned is False


def test_partitioning_partition_roundtrip() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2, "c": 3})

    first, remainder = sl.partition_state(state, keys={("a",)})
    sub_first, sub_rest = sl.partition_state(first, keys={("a",)})

    restored_first = sl.combine_partitions(sub_first, sub_rest)
    assert dict(restored_first.raw_mapping) == {("a",): 1}
    assert restored_first.is_partitioned is True

    recombined = sl.combine_partitions(restored_first, remainder)
    assert recombined == state
    assert recombined.to_pytree() == state.to_pytree()


def test_partition_state_unknown_key_raises() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

    with pytest.raises(KeyError):
        sl.partition_state(state, keys={("missing",)})


def test_combine_partitions_overlap_raises() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
    first, _rest = sl.partition_state(state, keys={("a",)})

    with pytest.raises(ValueError, match="duplicate keys"):
        sl.combine_partitions(first, first)


def test_combine_partitions_mismatched_sources_raises() -> None:
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
    first, _rest = sl.partition_state(state, keys={("a",)})

    other_state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2, "c": 3})
    _, other_remainder = sl.partition_state(other_state, keys={("a",)})

    with pytest.raises(ValueError, match="same FlatState"):
        sl.combine_partitions(first, other_remainder)
