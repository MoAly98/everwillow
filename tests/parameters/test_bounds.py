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


FState: tp.TypeAlias = sl.State[tp.Any]
TMapping: tp.TypeAlias = tp.Mapping[sl.K, transforms.TransformBase]


class TestApplyBoundsTransform:
    def test_returns_unwrapped_state_and_maps(self):
        """apply returns unwrapped state plus forward/inverse maps."""
        state: FState = sl.State.from_pytree({"mu": 0.5})
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)

        transform_map: TMapping = {("mu",): transform}
        unwrapped = bounds.unwrap(state, transform_map)

        assert jnp.isclose(unwrapped["mu",], transform.unwrap(0.5))

        # round-trip using statelib helper
        restored = bounds.wrap(unwrapped, transform_map)
        assert jnp.isclose(restored["mu",], 0.5)

    def test_handles_multiple_matches(self):
        """duplicate names create entries for each matching key."""
        params = {"a": {"x": 0.2, "y": 0.3}, "b": {"y": 0.4}}
        state: FState = sl.State.from_pytree(params)
        transform = transforms.OneSidedLogTransform(bound=0.0, direction="lower")

        transform_map: TMapping = {("a", "y"): transform, ("b", "y"): transform}
        unwrapped = bounds.unwrap(state, transform_map)

        restored = bounds.wrap(unwrapped, transform_map)
        for key in transform_map:
            assert jnp.isclose(restored[key], state[key])

    def test_no_transforms_returns_identity(self):
        """empty bounds mapping yields identity state and maps."""
        state: FState = sl.State.from_pytree({"value": 3.14})
        unwrapped = bounds.unwrap(state, {})
        assert unwrapped is state

    def test_wrap_no_transforms_returns_identity(self):
        """empty bounds mapping for wrap yields identity state."""
        state: FState = sl.State.from_pytree({"value": 3.14})
        wrapped = bounds.wrap(state, {})
        assert wrapped is state

    def test_unwrap_missing_key_raises(self):
        """unwrap raises KeyError if transform key not in state."""
        state = sl.State.from_pytree({"mu": 0.5})
        transform_map = {("missing",): transforms.MinuitTransform(lower=0.0, upper=1.0)}

        with pytest.raises(KeyError, match="not in state"):
            bounds.unwrap(state, transform_map)

    def test_wrap_missing_key_raises(self):
        """wrap raises KeyError if transform key not in state."""
        state = sl.State.from_pytree({"mu": 0.5})
        transform_map = {("missing",): transforms.MinuitTransform(lower=0.0, upper=1.0)}

        with pytest.raises(KeyError, match="not in state"):
            bounds.wrap(state, transform_map)
