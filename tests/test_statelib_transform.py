"""Tests covering :mod:`everwillow.statelib.transform`."""

from __future__ import annotations

import pytest

import everwillow.statelib as sl


class TestApplyTransformations:
    """Behavioural tests for :func:`everwillow.statelib.apply_transformations`."""

    def test_renames_keys_and_updates_values(self) -> None:
        """Renaming a key updates both the public path and stored value."""

        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        transforms: dict[tuple[str, ...], sl.Transform[int]] = {
            ("a",): sl.Transform(new_key=("alpha",), value_fn=lambda _k, v: v + 1),
            ("b",): sl.Transform(new_key=("beta",), value_fn=lambda _k, v: v * 2),
        }

        transformed = sl.apply_transformations(state, transforms)

        assert set(transformed.raw_mapping) == {("alpha",), ("beta",)}
        assert transformed["alpha",] == 2
        assert transformed["beta",] == 4

        tag = next(iter(transformed.own_keys))
        assert transformed.own_keys[tag] == frozenset({("alpha",), ("beta",)})

    def test_rejects_duplicate_targets(self) -> None:
        """Conflicting ``new_key`` entries raise a ``ValueError``."""

        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": 2})
        transforms: dict[tuple[str, ...], sl.Transform[int]] = {
            ("a",): sl.Transform(new_key=("shared",)),
            ("b",): sl.Transform(new_key=("shared",)),
        }

        with pytest.raises(ValueError, match="duplicate target key"):
            sl.apply_transformations(state, transforms)

    def test_respects_segment_boundaries(self) -> None:
        """Transformations operate independently on each merged segment."""

        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1, "y": 2})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 3, "z": 4})
        merged = sl.merge_states(state1, state2)

        state1_tree = state1.to_pytree()
        state2_tree = state2.to_pytree()

        transforms: dict[tuple[str, ...], sl.Transform[int]] = {
            ("x",): sl.Transform(new_key=("x_shared",)),
            ("y",): sl.Transform(new_key=("y_only",)),
        }

        transformed = sl.apply_transformations(merged, transforms)

        assert set(transformed.raw_mapping) == {("x_shared",), ("y_only",), ("z",)}

        seg1, seg2 = sl.split_state(transformed)
        assert dict(seg1.raw_mapping) == {("x_shared",): 1, ("y_only",): 2}
        assert dict(seg2.raw_mapping) == {("x_shared",): 3, ("z",): 4}

        # renamed keys should not affect pytree reconstruction per segment
        assert seg1.to_pytree() == state1_tree
        assert seg2.to_pytree() == state2_tree
