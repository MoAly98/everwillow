"""Extensive tests for :mod:`everwillow.statelib.state`."""

from __future__ import annotations

import re
import typing as tp

import jax.tree_util as jtu
import pytest

import everwillow.statelib as sl
import everwillow.statelib.state as sl_state


class TestFlatStateConstruction:
    """Tests covering construction and round-tripping of ``FlatState``."""

    def test_direct_init_raises(self) -> None:
        """Direct instantiation without ``from_pytree`` is forbidden."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "'FlatState' should never be directly instantiated, use 'FlatState.from_pytree' instead"
            ),
        ):
            sl.FlatState(raw_mapping={}, own_keys={}, n_internal_states=1)  # type: ignore[arg-type]

    def test_roundtrip_to_pytree(self) -> None:
        """Flattening and rehydrating a pytree preserve structure."""
        tree = {"a": 1, "b": {"c": 2, "d": (3, 4)}}
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(tree)

        assert state.n_internal_states == 1
        assert state.to_pytree() == tree
        assert all(isinstance(k, tuple) for k in state.raw_mapping)

    def test_to_pytree_multiple_internal_states_raises(self) -> None:
        """Attempting to rebuild with multiple segments raises ``ValueError``."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot convert to pytree with 2 internal "
                "states. Use 'split_state' first."
            ),
        ):
            merged.to_pytree()

    def test_from_flat_state_returns_copy(self) -> None:
        """Constructing from another ``FlatState`` returns a copy."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        clone: sl.FlatState[int] = sl.FlatState.from_pytree(state)

        assert clone == state
        assert clone is not state

    def test_raw_mapping_is_read_only(self) -> None:
        """The raw mapping proxy cannot be mutated."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

        with pytest.raises(TypeError):
            state.raw_mapping["a",] = 2  # type: ignore[index]

    def test_tree_flatten_metadata_is_immutable(self) -> None:
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

    def test_jax_flatten_roundtrip_preserves_state(self) -> None:
        """JAX flatten/unflatten returns an equal but new ``FlatState``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree(
            {"mu": 0.0, "sigma": 0.5, "nested": {"value": (1, 2)}}
        )

        leaves, treedef = jtu.tree_flatten(state)
        rebuilt = jtu.tree_unflatten(treedef, leaves)

        assert rebuilt == state
        assert rebuilt is not state
        assert state.to_pytree() == rebuilt.to_pytree()


class TestMergeStates:
    """Tests for merging and splitting ``FlatState`` segments."""

    def test_segments_order_is_preserved(self) -> None:
        """Merging preserves input ordering and allows reconstruction."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"b": 2})
        state3: sl.FlatState[int] = sl.FlatState.from_pytree({"c": 3})

        merged = sl.merge_states(state1, state2, state3)

        assert merged.n_internal_states == 3
        split = sl.split_state(merged)
        assert [s.to_pytree() for s in split] == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_overlapping_keys_last_segment_wins(self) -> None:
        """Later segments shadow earlier ones for duplicate keys."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 2})

        merged = sl.merge_states(state1, state2)

        assert merged["shared",] == 2
        first, second = sl.split_state(merged)
        assert first["shared",] == 1
        assert second["shared",] == 2

    def test_with_non_flat_state_raises(self) -> None:
        """Only ``FlatState`` instances can be merged."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        non_flat_state = {"b": 2}

        with pytest.raises(
            TypeError,
            match=re.escape("Can only merge FlatState instances"),
        ):
            sl.merge_states(state1, non_flat_state)  # type: ignore[arg-type]

    def test_with_same_state_raises(self) -> None:
        """Merging the same state twice raises ``ValueError``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})

        with pytest.raises(
            ValueError,
            match=re.escape(
                "One of the segments has already been merged into this FlatState"
            ),
        ):
            sl.merge_states(state, state)

    def test_with_no_states_raises(self) -> None:
        """Calling ``merge_states`` with no inputs raises ``ValueError``."""
        with pytest.raises(
            ValueError,
            match=re.escape("merge_states() requires at least one FlatState."),
        ):
            sl.merge_states()

    def test_merge_and_split_roundtrip_allows_overlapping_keys(self) -> None:
        """Splitting after merge returns the original per-segment states."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"shared": 5})

        merged = sl.merge_states(state1, state2)
        first, second = sl.split_state(merged)

        assert merged.n_internal_states == 2
        assert first == state1
        assert second == state2

        ordered_shared = [seg.to_pytree() for seg in sl.split_state(merged)]
        assert ordered_shared == [{"shared": 5}, {"shared": 5}]


class TestValidation:
    """Validation helpers enforce internal ``FlatState`` invariants."""

    def test_detects_segment_order_mismatch(self) -> None:
        """Mismatch between segment order metadata and segments raises."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state._segment_order.append(object())

        with pytest.raises(
            ValueError, match=re.escape("Segment order metadata is inconsistent")
        ):
            sl_state._validate_state(state)

    def test_detects_duplicate_segment_ids(self) -> None:
        """Duplicate identifiers in the segment order are rejected."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state._segment_order.append(state._segment_order[0])

        with pytest.raises(
            ValueError, match=re.escape("Duplicate segment identifiers detected")
        ):
            sl_state._validate_state(state)

    def test_detects_missing_mapping_entries(self) -> None:
        """Keys referenced by a segment must exist in the mapping."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.values.pop(("a",))

        with pytest.raises(
            ValueError,
            match=re.escape("references keys missing from the mapping"),
        ):
            sl_state._validate_state(state)

    def test_detects_mismatched_slice_keys(self) -> None:
        """Segment slices must match their declared key set."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.values["ghost",] = 99

        with pytest.raises(ValueError, match=re.escape("has mismatched slice keys")):
            sl_state._validate_state(state)

    def test_detects_mismatched_key_paths(self) -> None:
        """Stored key-path metadata must align with segment keys."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]
        segment = state._segments[segment_id]
        segment.key_paths.pop(("a",))

        with pytest.raises(
            ValueError, match=re.escape("has mismatched key path metadata")
        ):
            sl_state._validate_state(state)

    def test_detects_orphan_mapping_keys(self) -> None:
        """All mapping entries must belong to exactly one segment."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state._mapping.maps.insert(0, {("orphan",): 5})

        with pytest.raises(
            ValueError,
            match=re.escape("State mapping contains keys not owned by any segment"),
        ):
            sl_state._validate_state(state)


class TestMapState:
    """Mapping operations should leave inputs untouched."""

    def test_applies_function_without_mutating_original(self) -> None:
        """Mapping returns a new state while preserving the original."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})

        mapped = sl.map_state(lambda _k, v: v * 10, state)

        assert state["a",] == 1
        assert mapped["a",] == 10
        assert mapped.to_pytree() == {"a": 10, "b": 20}


class TestUpdateState:
    """Updating state values should be contained and safe."""

    def test_replaces_values(self) -> None:
        """Updating a key produces a new state with the change applied."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        updated = sl.update_state(state, {("a",): 99})

        assert dict(state.raw_mapping) == {("a",): 1, ("b",): 2}
        assert dict(updated.raw_mapping) == {("a",): 99, ("b",): 2}
        assert updated.to_pytree() == {"a": 99, "b": 2}

    def test_missing_key_raises(self) -> None:
        """Updating an unknown key raises a ``KeyError``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        with pytest.raises(KeyError, match=re.escape("not present in FlatState")):
            sl.update_state(state, {("missing",): 0})

    def test_on_merged_state_updates_all_segments(self) -> None:
        """Updates propagate to every segment that owns the key."""
        s1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        s2: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 3, "c": 4})
        merged = sl.merge_states(s1, s2)

        updated = sl.update_state(merged, {("a",): 10})
        seg1, seg2 = sl.split_state(updated)

        assert dict(seg1.raw_mapping) == {("a",): 10, ("b",): 2}
        assert dict(seg2.raw_mapping) == {("a",): 10, ("c",): 4}


class TestPartitionState:
    """Partition API splits and recombines state slices."""

    def test_partition_state_by_keys_roundtrip(self) -> None:
        """Partitioning by explicit keys round-trips via combine."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree(
            {"user": {"name": 1, "email": 2}, "prefs": {"theme": 3, "lang": 4}}
        )
        assert state.is_partitioned is False

        selected: sl.FlatState[int]
        remainder: sl.FlatState[int]
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
        with pytest.raises(ValueError, match=re.escape("Combine the partitions first")):
            selected.to_pytree()
        with pytest.raises(ValueError, match=re.escape("Combine the partitions first")):
            remainder.to_pytree()
        assert selected.is_partitioned is True
        assert remainder.is_partitioned is True

        restored: sl.FlatState[int] = sl.combine_partitions(selected, remainder)
        assert restored == state
        assert restored.to_pytree() == state.to_pytree()
        assert restored.is_partitioned is False

    def test_partition_state_by_predicate_roundtrip(self) -> None:
        """Predicate-based partitioning round-trips via combine."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree(
            {
                "user": {"name": 1, "email": 2},
                "flags": {"beta": 3, "notifications": 4},
            }
        )

        def is_flag(key: tuple[object, ...], _value: int) -> bool:
            return key[0] == "flags"

        flags: sl.FlatState[int]
        rest: sl.FlatState[int]
        flags, rest = sl.partition_state(state, predicate=is_flag)
        merged: sl.FlatState[int] = sl.combine_partitions(flags, rest)

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
        with pytest.raises(ValueError, match=re.escape("Combine the partitions first")):
            flags.to_pytree()
        assert flags.is_partitioned is True
        assert rest.is_partitioned is True
        assert merged.is_partitioned is False

    def test_partitioning_partition_roundtrip(self) -> None:
        """Nested partitioning can be combined in stages."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2, "c": 3})

        first: sl.FlatState[int]
        remainder: sl.FlatState[int]
        first, remainder = sl.partition_state(state, keys={("a",)})
        sub_first: sl.FlatState[int]
        sub_rest: sl.FlatState[int]
        sub_first, sub_rest = sl.partition_state(first, keys={("a",)})

        restored_first: sl.FlatState[int] = sl.combine_partitions(sub_first, sub_rest)
        assert dict(restored_first.raw_mapping) == {("a",): 1}
        assert restored_first.is_partitioned is True

        recombined: sl.FlatState[int] = sl.combine_partitions(restored_first, remainder)
        assert recombined == state
        assert recombined.to_pytree() == state.to_pytree()

    def test_partition_state_unknown_key_raises(self) -> None:
        """Partitioning a missing key raises ``KeyError``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        with pytest.raises(KeyError):
            sl.partition_state(state, keys={("missing",)})

    def test_combine_partitions_overlap_raises(self) -> None:
        """Combining overlapping partitions raises ``ValueError``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        first, _rest = sl.partition_state(state, keys={("a",)})

        with pytest.raises(ValueError, match=re.escape("duplicate keys")):
            sl.combine_partitions(first, first)

    def test_combine_partitions_mismatched_sources_raises(self) -> None:
        """Partitions originating from different states cannot be combined."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        first, _rest = sl.partition_state(state, keys={("a",)})

        other_state: sl.FlatState[int] = sl.FlatState.from_pytree(
            {"a": 1, "b": 2, "c": 3}
        )
        _, other_remainder = sl.partition_state(other_state, keys={("a",)})

        with pytest.raises(ValueError, match=re.escape("same FlatState")):
            sl.combine_partitions(first, other_remainder)


class TestAccessors:
    """Accessor helpers surface the expected segments."""

    def test_get_state_returns_correct_segment(self) -> None:
        """Retrieving by segment identifier yields the original slice."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        retrieved1 = merged.get_state(merged._segment_order[0])
        retrieved2 = merged.get_state(merged._segment_order[1])

        assert retrieved1 == state1
        assert retrieved2 == state2

    def test_get_state_segment_record_is_copy(self) -> None:
        """Returned segment records are defensive copies."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        segment_id = state._segment_order[0]

        segment = state.get_state(segment_id)._segments[segment_id]
        assert segment == state._segments[segment_id]
        assert segment is not state._segments[segment_id]

    def test_get_state_wrong_segment_id_raises(self) -> None:
        """Asking for an unknown segment identifier raises ``KeyError``."""
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        wrong_id = object()

        with pytest.raises(KeyError, match=f"Tag {wrong_id!r} not found in FlatState"):
            state.get_state(wrong_id)

    def test_split_state_returns_correct_segments(self) -> None:
        """Splitting a merged state recreates the original slices."""
        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"b": 2})
        merged = sl.merge_states(state1, state2)

        first, second = sl.split_state(merged)

        assert merged.n_internal_states == 2
        assert first == state1
        assert second == state2
        assert first is not state1
        assert second is not state2
