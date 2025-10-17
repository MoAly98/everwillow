from __future__ import annotations

import re
import typing as tp

import pytest

import everwillow.statelib as sl
from everwillow.statelib.state import _validate_state


class TestFlatStateConstruction:
    def test_direct_init_raises(self) -> None:
        with pytest.raises(
            TypeError,
            match="'FlatState' should never be directly instantiated, use 'FlatState.from_pytree' instead",
        ):
            sl.FlatState(raw_mapping={}, own_keys={}, n_internal_states=1)  # type: ignore[arg-type]

    def test_roundtrip_to_pytree(self) -> None:
        tree = {"a": 1, "b": {"c": 2, "d": (3, 4)}}
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(tree)

        assert state.n_internal_states == 1
        assert state.to_pytree() == tree
        assert all(isinstance(k, tuple) for k in state.raw_mapping)

    def test_to_pytree_multiple_internal_states_raises(self) -> None:
        state1 = sl.FlatState.from_pytree({"a": 1})
        state2 = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        with pytest.raises(
            ValueError,
            match=(
                "Cannot convert to pytree with 2 internal "
                "states. Use 'split_state' first."
            ),
        ):
            merged.to_pytree()

    def test_from_flat_state_returns_copy(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        clone = sl.FlatState.from_pytree(state)

        assert clone == state
        assert clone is not state

    def test_raw_mapping_is_read_only(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})

        with pytest.raises(TypeError):
            state.raw_mapping["a",] = 2  # type: ignore[index]

    def test_tree_flatten_metadata_is_immutable(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})

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


class TestMergeStates:
    def test_segments_order_is_preserved(self) -> None:
        state1 = sl.FlatState.from_pytree({"a": 1})
        state2 = sl.FlatState.from_pytree({"b": 2})
        state3 = sl.FlatState.from_pytree({"c": 3})

        merged = sl.merge_states(state1, state2, state3)

        assert merged.n_internal_states == 3
        split = sl.split_state(merged)
        assert [s.to_pytree() for s in split] == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_overlapping_keys_last_segment_wins(self) -> None:
        state1 = sl.FlatState.from_pytree({"shared": 1})
        state2 = sl.FlatState.from_pytree({"shared": 2})

        merged = sl.merge_states(state1, state2)

        assert merged["shared",] == 2
        first, second = sl.split_state(merged)
        assert first["shared",] == 1
        assert second["shared",] == 2

    def test_with_non_flat_state_raises(self) -> None:
        state1 = sl.FlatState.from_pytree({"a": 1})
        non_flat_state = {"b": 2}

        with pytest.raises(
            TypeError,
            match="Can only merge FlatState instances",
        ):
            sl.merge_states(state1, non_flat_state)  # type: ignore[arg-type]

    def test_with_same_state_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})

        with pytest.raises(
            ValueError,
            match="One of the segments has already been merged into this FlatState",
        ):
            sl.merge_states(state, state)

    def test_with_no_states_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("merge_states() requires at least one FlatState."),
        ):
            sl.merge_states()

    def test_merge_and_split_roundtrip_allows_overlapping_keys(self) -> None:
        state1 = sl.FlatState.from_pytree({"shared": 5})
        state2 = sl.FlatState.from_pytree({"shared": 5})

        merged = sl.merge_states(state1, state2)
        first, second = sl.split_state(merged)

        assert merged.n_internal_states == 2
        assert first == state1
        assert second == state2

        ordered_shared = [seg.to_pytree() for seg in sl.split_state(merged)]
        assert ordered_shared == [{"shared": 5}, {"shared": 5}]


class TestValidation:
    def test_detects_segment_order_mismatch(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        state._segment_order.append(object())

        with pytest.raises(ValueError, match="Segment order metadata is inconsistent"):
            _validate_state(state)

    def test_detects_duplicate_segment_ids(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        state._segment_order.append(state._segment_order[0])

        with pytest.raises(ValueError, match="Duplicate segment identifiers detected"):
            _validate_state(state)

    def test_detects_missing_mapping_entries(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.values.pop(("a",))

        with pytest.raises(
            ValueError,
            match="references keys missing from the mapping",
        ):
            _validate_state(state)

    def test_detects_mismatched_slice_keys(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.values[("ghost",)] = 99

        with pytest.raises(ValueError, match="has mismatched slice keys"):
            _validate_state(state)

    def test_detects_mismatched_key_paths(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.key_paths.pop(("a",))

        with pytest.raises(ValueError, match="has mismatched key path metadata"):
            _validate_state(state)

    def test_detects_orphan_mapping_keys(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        state._mapping.maps.insert(0, {("orphan",): 5})

        with pytest.raises(
            ValueError,
            match="State mapping contains keys not owned by any segment",
        ):
            _validate_state(state)


class TestMapState:
    def test_applies_function_without_mutating_original(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1, "b": 2})

        mapped = sl.map_state(lambda _k, v: v * 10, state)

        assert state["a",] == 1
        assert mapped["a",] == 10
        assert mapped.to_pytree() == {"a": 10, "b": 20}


class TestUpdateState:
    def test_replaces_values(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1, "b": 2})
        updated = sl.update_state(state, {("a",): 99})

        assert dict(state.raw_mapping) == {("a",): 1, ("b",): 2}
        assert dict(updated.raw_mapping) == {("a",): 99, ("b",): 2}
        assert updated.to_pytree() == {"a": 99, "b": 2}

    def test_missing_key_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        with pytest.raises(KeyError, match="not present in FlatState"):
            sl.update_state(state, {("missing",): 0})

    def test_on_merged_state_updates_all_segments(self) -> None:
        s1 = sl.FlatState.from_pytree({"a": 1, "b": 2})
        s2 = sl.FlatState.from_pytree({"a": 3, "c": 4})
        merged = sl.merge_states(s1, s2)

        updated = sl.update_state(merged, {("a",): 10})
        seg1, seg2 = sl.split_state(updated)

        assert dict(seg1.raw_mapping) == {("a",): 10, ("b",): 2}
        assert dict(seg2.raw_mapping) == {("a",): 10, ("c",): 4}


class TestPartitionState:
    def test_partition_state_by_keys_roundtrip(self) -> None:
        state = sl.FlatState.from_pytree(
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

    def test_partition_state_by_predicate_roundtrip(self) -> None:
        state = sl.FlatState.from_pytree(
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

    def test_partitioning_partition_roundtrip(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1, "b": 2, "c": 3})

        first, remainder = sl.partition_state(state, keys={("a",)})
        sub_first, sub_rest = sl.partition_state(first, keys={("a",)})

        restored_first = sl.combine_partitions(sub_first, sub_rest)
        assert dict(restored_first.raw_mapping) == {("a",): 1}
        assert restored_first.is_partitioned is True

        recombined = sl.combine_partitions(restored_first, remainder)
        assert recombined == state
        assert recombined.to_pytree() == state.to_pytree()

    def test_partition_state_unknown_key_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        with pytest.raises(KeyError):
            sl.partition_state(state, keys={("missing",)})

    def test_combine_partitions_overlap_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1, "b": 2})
        first, _rest = sl.partition_state(state, keys={("a",)})

        with pytest.raises(ValueError, match="duplicate keys"):
            sl.combine_partitions(first, first)

    def test_combine_partitions_mismatched_sources_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1, "b": 2})
        first, _rest = sl.partition_state(state, keys={("a",)})

        other_state = sl.FlatState.from_pytree({"a": 1, "b": 2, "c": 3})
        _, other_remainder = sl.partition_state(other_state, keys={("a",)})

        with pytest.raises(ValueError, match="same FlatState"):
            sl.combine_partitions(first, other_remainder)


class TestAccessors:
    def test_get_state_returns_correct_segment(self) -> None:
        state1 = sl.FlatState.from_pytree({"a": 1})
        state2 = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        retrieved1 = merged.get_state(merged._segment_order[0])
        retrieved2 = merged.get_state(merged._segment_order[1])

        assert retrieved1 == state1
        assert retrieved2 == state2

    def test_get_state_segment_record_is_copy(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]

        segment = state.get_state(segment_id)._segments[segment_id]
        assert segment == state._segments[segment_id]
        assert segment is not state._segments[segment_id]

    def test_get_state_wrong_segment_id_raises(self) -> None:
        state = sl.FlatState.from_pytree({"a": 1})
        wrong_id = object()

        with pytest.raises(KeyError, match=f"Tag {wrong_id!r} not found in FlatState"):
            state.get_state(wrong_id)

    def test_split_state_returns_correct_segments(self) -> None:
        state1 = sl.FlatState.from_pytree({"a": 1})
        state2 = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        first, second = sl.split_state(merged)

        assert merged.n_internal_states == 2
        assert first == state1
        assert second == state2
        assert first is not state1
        assert second is not state2
