"""Tests for everwillow.parameters.bounds."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import pytest

import everwillow.parameters.bounds as bounds
import everwillow.parameters.transforms as transforms
import everwillow.statelib as sl

jax.config.update("jax_enable_x64", True)


class TestMatchBoundsToState:
    def test_matches_by_name(self):
        """matching by string key finds canonical tuple."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(
            {"mu": 0.5, "sigma": 1.2}
        )
        transform = transforms.MinuitTransform(lower=0.0, upper=2.0)

        result = bounds.match_bounds_to_state(state, {"mu": transform})

        assert list(result) == [("mu",)]
        assert result["mu",] is transform

    def test_matches_by_keypath(self):
        """matching by tuple key uses canonical path."""
        params = {"left": {"gamma": 1.0}, "right": {"gamma": 2.0}}
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(params)
        key = ("left", "gamma")
        transform = transforms.OneSidedLogTransform(bound=0.0, direction="lower")

        result = bounds.match_bounds_to_state(state, {key: transform})

        assert list(result) == [key]
        assert result[key] is transform

    def test_ignores_none_entries(self):
        """None entries are treated as pass-through."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"value": 3.0})
        result = bounds.match_bounds_to_state(state, {"value": None})
        assert result == {}

    def test_raises_for_unknown_name(self):
        """unknown string keys raise KeyError."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"alpha": 1.0})
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)

        with pytest.raises(KeyError):
            bounds.match_bounds_to_state(state, {"beta": transform})

    def test_raises_for_invalid_transform_type(self):
        """non-transform objects trigger TypeError."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"x": 1.0})
        bogus = tp.cast(bounds.TransformSpec, object())
        invalid_mapping = tp.cast(
            dict[str | tuple[tp.Any, ...], bounds.TransformSpec],
            {"x": bogus},
        )
        with pytest.raises(TypeError):
            bounds.match_bounds_to_state(state, invalid_mapping)

    def test_raises_for_invalid_spec_type(self):
        """specifiers must be str or tuple."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"x": 1.0})
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
        invalid_mapping = tp.cast(
            dict[str | tuple[tp.Any, ...], bounds.TransformSpec],
            {42: transform},
        )
        with pytest.raises(TypeError):
            bounds.match_bounds_to_state(state, invalid_mapping)


class TestApplyBoundsTransform:
    def test_returns_unwrapped_state_and_maps(self):
        """apply returns unwrapped state plus forward/inverse maps."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"mu": 0.5})
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)

        unwrapped, unwrap_map, wrap_map = bounds.apply_bounds_transform(
            state,
            {"mu": transform},
        )

        assert jnp.isclose(unwrapped["mu",], transform.unwrap(0.5))
        assert set(unwrap_map) == {("mu",)}
        assert set(wrap_map) == {("mu",)}

        # round-trip using statelib helper
        restored = sl.transform.apply_transformations(unwrapped, wrap_map)
        assert jnp.isclose(restored["mu",], 0.5)

    def test_handles_multiple_matches(self):
        """duplicate names create entries for each matching key."""
        params = {"a": {"x": 0.2, "y": 0.3}, "b": {"y": 0.4}}
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(params)
        transform = transforms.OneSidedLogTransform(bound=0.0, direction="lower")

        unwrapped, unwrap_map, wrap_map = bounds.apply_bounds_transform(
            state,
            {"y": transform},
        )

        expected_keys = {("a", "y"), ("b", "y")}
        assert set(unwrap_map) == expected_keys
        assert set(wrap_map) == expected_keys

        restored = sl.transform.apply_transformations(unwrapped, wrap_map)
        for key in expected_keys:
            assert jnp.isclose(restored[key], state[key])

    def test_no_transforms_returns_identity(self):
        """empty bounds mapping yields identity state and maps."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"value": 3.14})
        unwrapped, unwrap_map, wrap_map = bounds.apply_bounds_transform(state, {})
        assert unwrapped is state
        assert unwrap_map == {}
        assert wrap_map == {}


class TestResolveKeysHelper:
    """Directly exercise the internal _resolve_keys helper."""

    def test_resolve_by_name(self):
        """string spec resolves to matching canonical key."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"foo": 1.0})
        keys = bounds._resolve_keys(state, "foo")  # type: ignore[attr-defined]
        assert keys == [("foo",)]

    def test_resolve_by_tuple(self):
        """tuple spec resolves to the provided key."""
        params = {"left": {"bar": 2.0}}
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree(params)
        keys = bounds._resolve_keys(state, ("left", "bar"))  # type: ignore[attr-defined]
        assert keys == [("left", "bar")]

    def test_invalid_tuple_raises(self):
        """missing tuple raises KeyError."""
        state: sl.FlatState[tp.Any] = sl.FlatState.from_pytree({"foo": 1.0})
        with pytest.raises(KeyError):
            bounds._resolve_keys(state, ("missing",))
