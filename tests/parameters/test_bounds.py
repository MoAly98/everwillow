"""Tests for parameter bounds transformations."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import pytest

import everwillow.parameters.bounds as bounds
import everwillow.statelib as sl

jax.config.update("jax_enable_x64", True)


def _bounds(
    mapping: dict[str, tuple[float | None, float | None] | None],
) -> dict[str | tuple[object, ...], bounds.BoundSpec]:
    """Helper to express bound specifications with precise typing."""
    return cast(dict[str | tuple[object, ...], bounds.BoundSpec], mapping)


class TestTransformations:
    """Test individual transformation functions."""

    def test_transform_unbounded_both_bounds(self):
        """Test transformation with both lower and upper bounds."""
        # Midpoint should map to 0
        result = bounds.transform_to_unbounded(0.5, 0.0, 1.0)
        assert jnp.isclose(result, 0.0, atol=1e-6)

        # Value closer to lower bound should be negative
        result = bounds.transform_to_unbounded(0.1, 0.0, 1.0)
        assert result < 0

        # Value closer to upper bound should be positive
        result = bounds.transform_to_unbounded(0.9, 0.0, 1.0)
        assert result > 0

    def test_transform_unbounded_lower_only(self):
        """Test transformation with lower bound only."""
        # Value at lower + 1 should map to 0
        result = bounds.transform_to_unbounded(1.0, 0.0, None)
        assert jnp.isclose(result, 0.0, atol=1e-6)

        # Larger values should be positive
        result = bounds.transform_to_unbounded(10.0, 0.0, None)
        assert result > 0

    def test_transform_unbounded_upper_only(self):
        """Test transformation with upper bound only."""
        # Value at upper - 1 should map to 0
        result = bounds.transform_to_unbounded(0.0, None, 1.0)
        assert jnp.isclose(result, 0.0, atol=1e-6)

        # Smaller values should be positive
        result = bounds.transform_to_unbounded(-10.0, None, 1.0)
        assert result > 0

    def test_transform_unbounded_no_bounds(self):
        """Test transformation with no bounds (identity)."""
        value = 42.0
        result = bounds.transform_to_unbounded(value, None, None)
        assert result == value

    def test_transform_bounded_both_bounds(self):
        """Test inverse transformation with both bounds."""
        # 0 should map to midpoint
        result = bounds.transform_to_bounded(0.0, 0.0, 1.0)
        assert jnp.isclose(result, 0.5, atol=1e-6)

        # Large positive values should approach upper bound
        result = bounds.transform_to_bounded(10.0, 0.0, 1.0)
        assert 0.99 < result < 1.0

        # Large negative values should approach lower bound
        result = bounds.transform_to_bounded(-10.0, 0.0, 1.0)
        assert 0.0 < result < 0.01

    def test_transform_bounded_lower_only(self):
        """Test inverse transformation with lower bound only."""
        # 0 should map to lower + 1
        result = bounds.transform_to_bounded(0.0, 0.0, None)
        assert jnp.isclose(result, 1.0, atol=1e-6)

        # Positive values should be > lower bound
        result = bounds.transform_to_bounded(5.0, 0.0, None)
        assert result > 0.0

    def test_transform_bounded_upper_only(self):
        """Test inverse transformation with upper bound only."""
        # 0 should map to upper - 1
        result = bounds.transform_to_bounded(0.0, None, 1.0)
        assert jnp.isclose(result, 0.0, atol=1e-6)

        # Positive values should be < upper bound
        result = bounds.transform_to_bounded(5.0, None, 1.0)
        assert result < 1.0

    def test_transform_bounded_no_bounds(self):
        """Test inverse transformation with no bounds (identity)."""
        value = 42.0
        result = bounds.transform_to_bounded(value, None, None)
        assert result == value

    def test_roundtrip_both_bounds(self):
        """Test forward and inverse transformations roundtrip."""
        original = 0.7
        unbounded = bounds.transform_to_unbounded(original, 0.0, 1.0)
        recovered = bounds.transform_to_bounded(unbounded, 0.0, 1.0)
        assert jnp.isclose(recovered, original, atol=1e-6)

    def test_roundtrip_lower_only(self):
        """Test roundtrip with lower bound only."""
        original = 5.0
        unbounded = bounds.transform_to_unbounded(original, 0.0, None)
        recovered = bounds.transform_to_bounded(unbounded, 0.0, None)
        assert jnp.isclose(recovered, original, atol=1e-6)

    def test_roundtrip_upper_only(self):
        """Test roundtrip with upper bound only."""
        original = -5.0
        unbounded = bounds.transform_to_unbounded(original, None, 1.0)
        recovered = bounds.transform_to_bounded(unbounded, None, 1.0)
        assert jnp.isclose(recovered, original, atol=1e-6)

    def test_jit_compatibility(self):
        """Test that transformations are JIT-compatible."""

        @jax.jit
        def forward(x):
            return bounds.transform_to_unbounded(x, 0.0, 1.0)

        @jax.jit
        def inverse(x):
            return bounds.transform_to_bounded(x, 0.0, 1.0)

        # Should not raise
        result = forward(0.5)
        assert jnp.isclose(result, 0.0, atol=1e-6)

        recovered = inverse(result)
        assert jnp.isclose(recovered, 0.5, atol=1e-6)


class TestValidateBounds:
    """Test the validate_bounds function."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        params = {"mu": 1.0, "sigma": 0.5}
        bounds_spec = _bounds({"mu": (0.0, 5.0), "sigma": (0.0, None)})
        # Should not raise
        bounds.validate_bounds(params, bounds_spec)

    def test_validate_violates_lower_bound(self):
        """Test validation catches lower bound violation."""
        params = {"mu": -1.0, "sigma": 0.5}
        bounds_spec = _bounds({"mu": (0.0, 5.0), "sigma": (0.0, None)})
        with pytest.raises(ValueError, match="violates lower bound"):
            bounds.validate_bounds(params, bounds_spec)

    def test_validate_violates_upper_bound(self):
        """Test validation catches upper bound violation."""
        params = {"mu": 6.0, "sigma": 0.5}
        bounds_spec = _bounds({"mu": (0.0, 5.0), "sigma": (0.0, None)})
        with pytest.raises(ValueError, match="violates upper bound"):
            bounds.validate_bounds(params, bounds_spec)

    def test_validate_invalid_bounds_spec(self):
        """Test validation catches malformed bounds."""
        params = {"mu": 1.0}
        bounds_spec = _bounds({"mu": (5.0, 0.0)})  # lower > upper
        with pytest.raises(ValueError, match="must be < upper bound"):
            bounds.validate_bounds(params, bounds_spec)

    def test_validate_missing_parameter(self):
        """Test validation catches missing parameters."""
        params = {"mu": 1.0}
        bounds_spec = _bounds({"sigma": (0.0, None)})
        with pytest.raises(KeyError, match="not found"):
            bounds.validate_bounds(params, bounds_spec)

    def test_validate_none_bounds(self):
        """Test validation ignores None bounds."""
        params = {"mu": 1.0, "sigma": -10.0}  # sigma is out of typical range
        bounds_spec = _bounds({"mu": (0.0, 5.0), "sigma": None})
        # Should not raise
        bounds.validate_bounds(params, bounds_spec)


class TestCreateBoundsTransforms:
    """Test the create_bounds_transforms function."""

    def test_create_transforms_both_bounds(self):
        """Test creating transforms for parameters with both bounds."""
        state: sl.FlatState[float] = sl.FlatState.from_pytree({"mu": 1.0, "sigma": 0.5})
        bounds_spec = _bounds({"mu": (0.0, 5.0)})

        fwd, inv = bounds.create_bounds_transforms(state, bounds_spec)

        # Should have transforms for mu
        assert len(fwd) == 1
        assert len(inv) == 1
        assert ("mu",) in fwd
        assert ("mu",) in inv

    def test_create_transforms_mixed_bounds(self):
        """Test creating transforms with mixed bound types."""
        state: sl.FlatState[float] = sl.FlatState.from_pytree(
            {"mu": 1.0, "sigma": 0.5, "offset": 0.0}
        )
        bounds_spec = _bounds(
            {
                "mu": (0.0, 5.0),  # Both bounds
                "sigma": (0.0, None),  # Lower only
                "offset": (None, 10.0),  # Upper only
            }
        )

        fwd, inv = bounds.create_bounds_transforms(
            cast(sl.FlatState[float], state), bounds_spec
        )

        # Should have transforms for all three
        assert len(fwd) == 3
        assert len(inv) == 3

    def test_create_transforms_skips_none(self):
        """Test that None bounds are skipped."""
        state: sl.FlatState[float] = sl.FlatState.from_pytree({"mu": 1.0, "sigma": 0.5})
        bounds_spec = _bounds({"mu": (0.0, 5.0), "sigma": None})

        fwd, _inv = bounds.create_bounds_transforms(
            cast(sl.FlatState[float], state), bounds_spec
        )

        # Should only have transforms for mu
        assert len(fwd) == 1
        assert ("mu",) in fwd

    def test_transforms_work_with_apply_transformations(self):
        """Test that created transforms work with apply_transformations."""
        state: sl.FlatState[float] = sl.FlatState.from_pytree({"mu": 2.5})
        bounds_spec = _bounds({"mu": (0.0, 5.0)})

        fwd, _inv = bounds.create_bounds_transforms(
            cast(sl.FlatState[float], state), bounds_spec
        )

        # Transform to unbounded
        unbounded_state = sl.apply_transformations(state, fwd)
        # Midpoint should map to 0
        assert jnp.isclose(unbounded_state["mu",], 0.0, atol=1e-6)

        # Transform back to bounded
        bounded_state = sl.apply_transformations(unbounded_state, _inv)
        # Should recover original value
        assert jnp.isclose(bounded_state["mu",], 2.5, atol=1e-6)
