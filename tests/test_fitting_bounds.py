"""Integration tests for fitting with parameter bounds."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import everwillow as ew

jax.config.update("jax_enable_x64", True)


class TestFitWithBounds:
    """Test the fit() function with parameter bounds."""

    def test_fit_hits_upper_bound(self):
        """Test that upper bound is enforced when minimum is above it."""

        def nll(params):
            # Unconstrained minimum at mu=10
            return (params["mu"] - 10.0) ** 2

        result = ew.fit(
            nll,
            {"mu": 0.5},
            bounds={"mu": (0.0, 5.0)},  # Constrain to [0, 5]
        )

        # Should hit the upper bound since true minimum is at 10
        assert 0.0 <= result.params["mu"] <= 5.0
        assert jnp.isclose(result.params["mu"], 5.0, atol=1e-2)
        assert result.nll < 100.0  # Should be at (5-10)^2 = 25, not (0.5-10)^2

    def test_fit_hits_lower_bound(self):
        """Test that lower bound is enforced when minimum is below it."""

        def nll(params):
            # Unconstrained minimum at mu=-5
            return (params["mu"] + 5.0) ** 2

        result = ew.fit(
            nll,
            {"mu": 2.0},
            bounds={"mu": (0.0, 10.0)},  # Constrain to [0, 10]
        )

        # Should hit the lower bound since true minimum is at -5
        assert 0.0 <= result.params["mu"] <= 10.0
        assert jnp.isclose(result.params["mu"], 0.0, atol=1e-2)
        assert result.nll < 100.0  # Should be at (0+5)^2 = 25

    def test_fit_lower_bound_only_prevents_negative(self):
        """Test lower bound prevents going negative when minimum would be negative."""

        def nll(params):
            # Unconstrained minimum at sigma=-2
            return (params["sigma"] + 2.0) ** 2

        result = ew.fit(
            nll,
            {"sigma": 1.0},
            bounds={"sigma": (0.0, None)},  # sigma >= 0
        )

        # Should hit the lower bound
        assert result.params["sigma"] >= 0.0
        assert jnp.isclose(result.params["sigma"], 0.0, atol=1e-2)

    def test_fit_upper_bound_only_prevents_large_values(self):
        """Test upper bound prevents large values when minimum is above bound."""

        def nll(params):
            # Unconstrained minimum at x=100
            return (params["x"] - 100.0) ** 2

        result = ew.fit(
            nll,
            {"x": 3.0},
            bounds={"x": (None, 10.0)},  # x <= 10
        )

        # Should hit the upper bound
        assert result.params["x"] <= 10.0
        assert jnp.isclose(result.params["x"], 10.0, atol=1e-2)

    def test_fit_within_bounds_unconstrained(self):
        """Test that bounds don't affect fit when minimum is within bounds."""

        def nll(params):
            # Unconstrained minimum at mu=2.5 (inside [0, 5])
            return (params["mu"] - 2.5) ** 2

        result = ew.fit(
            nll,
            {"mu": 1.0},
            bounds={"mu": (0.0, 5.0)},
        )

        # Should find true minimum
        assert 0.0 <= result.params["mu"] <= 5.0
        assert jnp.isclose(result.params["mu"], 2.5, atol=1e-2)

    def test_fit_multiple_params_different_constraints(self):
        """Test multiple parameters where some hit bounds and others don't."""

        def nll(params):
            # Unconstrained minima: mu=-10 (outside), sigma=0.5 (inside)
            return (params["mu"] + 10.0) ** 2 + (params["sigma"] - 0.5) ** 2

        result = ew.fit(
            nll,
            {"mu": 1.0, "sigma": 0.1},
            bounds={
                "mu": (0.0, 5.0),  # Will hit lower bound
                "sigma": (0.01, 2.0),  # Won't hit bound
            },
        )

        # mu should hit lower bound
        assert jnp.isclose(result.params["mu"], 0.0, atol=1e-2)
        # sigma should find true minimum
        assert jnp.isclose(result.params["sigma"], 0.5, atol=1e-2)

    def test_fit_with_bounds_and_fixed_params(self):
        """Test combining bounds with fixed parameters."""

        def nll(params):
            # Unconstrained minimum: mu=10, sigma=-5
            return (params["mu"] - 10.0) ** 2 + (params["sigma"] + 5.0) ** 2

        result = ew.fit(
            nll,
            {"mu": 1.0, "sigma": 0.5, "background": 100.0},
            fixed=["background"],
            bounds={
                "mu": (0.0, 5.0),  # Will hit upper bound
                "sigma": (0.0, None),  # Will hit lower bound
            },
        )

        # Fixed param should remain fixed
        assert result.params["background"] == 100.0

        # Bounded params should hit their bounds
        assert jnp.isclose(result.params["mu"], 5.0, atol=1e-2)
        assert jnp.isclose(result.params["sigma"], 0.0, atol=1e-2)

    def test_nll_receives_bounded_values(self):
        """Test that the NLL function always receives bounded values."""
        call_count = [0]

        def nll(params):
            call_count[0] += 1
            # The key test: if params["x"] is out of bounds, the quadratic will have
            # a different shape. Since the minimum is at x=10 (way outside [0,1]),
            # the optimizer should push towards x=1 (the upper bound).
            return (params["x"] - 10.0) ** 2  # Minimum way outside bounds

        result = ew.fit(
            nll,
            {"x": 0.5},
            bounds={"x": (0.0, 1.0)},
        )

        # Should have been called multiple times
        assert call_count[0] > 1
        # Final result should be at upper bound (closest point to true minimum)
        assert 0.0 <= result.params["x"] <= 1.0
        assert jnp.isclose(result.params["x"], 1.0, atol=1e-2)


class TestFixedParamFitWithBounds:
    """Test the fixed_param_fit() function with bounds."""

    def test_fixed_param_fit_with_bounds_constrains_free_param(self):
        """Test profile likelihood fit where free parameter would violate bound."""

        def nll(params):
            # Unconstrained minimum: mu=any, sigma=-10
            return (params["sigma"] + 10.0) ** 2

        result = ew.fixed_param_fit(
            {"mu": 1.5},  # Fix mu
            nll,
            {"mu": 1.0, "sigma": 0.5},
            bounds={"sigma": (0.0, None)},  # sigma >= 0
        )

        # mu should be fixed at 1.5
        assert jnp.isclose(result.params["mu"], 1.5, atol=1e-6)

        # sigma should hit lower bound
        assert result.params["sigma"] >= 0.0
        assert jnp.isclose(result.params["sigma"], 0.0, atol=1e-2)


class TestBoundsEdgeCases:
    """Test edge cases and error conditions."""

    def test_fit_with_both_params_hitting_bounds(self):
        """Test when both parameters would violate their bounds."""

        def nll(params):
            # Unconstrained minima: x=-100, y=100
            return (params["x"] + 100.0) ** 2 + (params["y"] - 100.0) ** 2

        result = ew.fit(
            nll,
            {"x": 0.5, "y": 0.5},
            bounds={
                "x": (0.0, 10.0),  # Will hit lower bound
                "y": (0.0, 10.0),  # Will hit upper bound
            },
        )

        assert jnp.isclose(result.params["x"], 0.0, atol=1e-2)
        assert jnp.isclose(result.params["y"], 10.0, atol=1e-2)

    def test_fit_with_asymmetric_bounds(self):
        """Test asymmetric bounds where minimum is way outside."""

        def nll(params):
            # Unconstrained minimum at x=1000
            return (params["x"] - 1000.0) ** 2

        result = ew.fit(
            nll,
            {"x": 0.1},
            bounds={"x": (0.01, 1.0)},  # Very far from minimum
        )

        # Should hit upper bound
        assert 0.01 <= result.params["x"] <= 1.0
        assert jnp.isclose(result.params["x"], 1.0, atol=1e-2)

    def test_fit_with_none_bounds_no_constraint(self):
        """Test that None bounds allow finding true minimum."""

        def nll(params):
            # Unconstrained minima: x=-50, y=50
            return (params["x"] + 50.0) ** 2 + (params["y"] - 50.0) ** 2

        result = ew.fit(
            nll,
            {"x": 1.0, "y": 1.0},
            bounds={"x": None, "y": None},  # No bounds
        )

        # Should find true minima
        assert jnp.isclose(result.params["x"], -50.0, atol=1e-1)
        assert jnp.isclose(result.params["y"], 50.0, atol=1e-1)
