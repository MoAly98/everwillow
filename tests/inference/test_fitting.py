"""Unit tests for everwillow.inference module.

Tests cover:
- FitResult dataclass
- fit() function with various parameter structures and options
- fit() with parameter bounds
- fixed_param_fit() function for profile likelihood
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import everwillow as ew
from everwillow.inference import FitResult
from everwillow.parameters.transforms import (
    MinuitTransform,
    OneSidedLogTransform,
    SigmoidTransform,
    SoftPlusTransform,
)

jax.config.update("jax_enable_x64", True)


def _expect_close(expected: float, *, atol: float = 1e-2):
    def _check(value: float) -> None:
        assert jnp.isclose(value, expected, atol=atol)

    return _check


def _expect_interval(*, lower: float | None = None, upper: float | None = None):
    def _check(value: float) -> None:
        if lower is not None:
            assert value >= lower
        if upper is not None:
            assert value <= upper

    return _check


# ============================================================================
# FitResult dataclass tests
# ============================================================================


class TestFitResult:
    """Tests for FitResult dataclass."""

    def test_fitresult_creation(self):
        """Test creating a FitResult with all fields."""
        params = {"mu": 1.0, "sigma": 0.5}
        result = FitResult(params=params, nll=5.5, success=True, solver_result=None)

        assert result.params == params
        assert result.nll == 5.5
        assert result.success is True
        assert result.solver_result is None

    def test_fitresult_frozen(self):
        """Test that FitResult is immutable (frozen dataclass)."""
        result = FitResult(params={}, nll=0.0, success=True)

        with pytest.raises(AttributeError):
            result.nll = 10.0  # type: ignore[misc]

    def test_fitresult_optional_solver_result(self):
        """Test that solver_result is optional."""
        result = FitResult(params={}, nll=0.0, success=True)
        assert result.solver_result is None


# ============================================================================
# fit() function tests - basic functionality
# ============================================================================


class TestFitBasic:
    """Tests for basic fit() functionality."""

    def test_simple_quadratic(self):
        """Test fitting a simple quadratic NLL."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2

        result = ew.fit(nll, {"mu": 0.0, "sigma": 0.5})

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-4
        assert result.nll < 1e-8
        assert result.success is True
        assert result.solver_result is not None

    def test_single_parameter(self):
        """Test fitting with a single parameter."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        result = ew.fit(nll, {"x": 0.0})

        assert abs(result.params["x"] - 5.0) < 1e-4

    def test_multiple_parameters(self):
        """Test fitting with multiple parameters."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = ew.fit(nll, {"a": 0.0, "b": 0.0, "c": 0.0})

        assert abs(result.params["a"] - 1.0) < 1e-4
        assert abs(result.params["b"] - 2.0) < 1e-4
        assert abs(result.params["c"] - 3.0) < 1e-4


# ============================================================================
# fit() function tests - pytree structures
# ============================================================================


class TestFitPytreeStructures:
    """Tests for fit() with various pytree structures."""

    def test_nested_dict(self):
        """Test fitting with nested dict structure."""

        def nll(params):
            return (params["level1"]["mu"] - 2.0) ** 2 + (
                params["level1"]["sigma"] - 1.0
            ) ** 2

        initial = {"level1": {"mu": 0.0, "sigma": 0.5}}
        result = ew.fit(nll, initial)

        assert abs(result.params["level1"]["mu"] - 2.0) < 1e-4
        assert abs(result.params["level1"]["sigma"] - 1.0) < 1e-4

    def test_deeply_nested_dict(self):
        """Test fitting with deeply nested structure."""

        def nll(params):
            return (params["a"]["b"]["c"] - 5.0) ** 2

        initial = {"a": {"b": {"c": 0.0}}}
        result = ew.fit(nll, initial)

        assert abs(result.params["a"]["b"]["c"] - 5.0) < 1e-4

    def test_mixed_structure(self):
        """Test fitting with mixed flat and nested structure."""

        def nll(params):
            return (params["flat"] - 1.0) ** 2 + (params["nested"]["value"] - 2.0) ** 2

        initial = {"flat": 0.0, "nested": {"value": 0.0}}
        result = ew.fit(nll, initial)

        assert abs(result.params["flat"] - 1.0) < 1e-4
        assert abs(result.params["nested"]["value"] - 2.0) < 1e-4


# ============================================================================
# fit() function tests - fixed parameters
# ============================================================================


class TestFitFixedParameters:
    """Tests for fit() with fixed parameters."""

    def test_single_fixed_parameter(self):
        """Test fixing a single parameter."""

        def nll(params):
            return (
                (params["mu"] - 2.0) ** 2
                + (params["sigma"] - 1.0) ** 2
                + (params["background"] - 100.0) ** 2
            )

        result = ew.fit(
            nll,
            {"mu": 0.0, "sigma": 0.5, "background": 50.0},
            fixed=["background"],
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-4
        assert (
            abs(result.params["background"] - 50.0) < 1e-10
        )  # Should be exactly fixed

    def test_multiple_fixed_parameters(self):
        """Test fixing multiple parameters."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = ew.fit(
            nll,
            {"a": 0.0, "b": 10.0, "c": 20.0},
            fixed=["b", "c"],
        )

        assert abs(result.params["a"] - 1.0) < 1e-4
        assert abs(result.params["b"] - 10.0) < 1e-10
        assert abs(result.params["c"] - 20.0) < 1e-10

    def test_all_parameters_fixed(self):
        """Test when all parameters are fixed (no optimization needed)."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        result = ew.fit(nll, {"x": 3.0}, fixed=["x"])

        assert abs(result.params["x"] - 3.0) < 1e-10

    def test_fixed_none(self):
        """Test that fixed=None works (no fixed parameters)."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fixed=None)

        assert abs(result.params["mu"] - 2.0) < 1e-4

    def test_fixed_empty_mapping(self):
        """Test that fixed=[] works (no fixed parameters)."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fixed=[])

        assert abs(result.params["mu"] - 2.0) < 1e-4

    def test_fixed_nested_parameter(self):
        """Test fixing a parameter in nested structure."""

        def nll(params):
            return (params["level1"]["mu"] - 2.0) ** 2 + (
                params["level1"]["sigma"] - 1.0
            ) ** 2

        initial = {"level1": {"mu": 0.0, "sigma": 5.0}}
        result = ew.fit(
            nll,
            initial,
            fixed=[("level1", "sigma")],
        )

        assert abs(result.params["level1"]["mu"] - 2.0) < 1e-4
        assert abs(result.params["level1"]["sigma"] - 5.0) < 1e-10

    def test_fixed_predicate(self) -> None:
        """Fix parameters using the ``fixed_predicate`` hook."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = ew.fit(
            nll,
            {"a": 0.0, "b": 10.0, "c": 20.0},
            fixed_predicate=lambda key, _value: key[-1] in {"b", "c"},
        )

        assert abs(result.params["a"] - 1.0) < 1e-4
        assert abs(result.params["b"] - 10.0) < 1e-10
        assert abs(result.params["c"] - 20.0) < 1e-10


# ============================================================================
# fit() function tests - additional arguments
# ============================================================================


class TestFitAdditionalArguments:
    """Tests for fit() with additional positional and keyword arguments."""

    def test_positional_args(self):
        """Test fit() with additional positional arguments."""

        def nll(params, target_mu, target_sigma):
            return (params["mu"] - target_mu) ** 2 + (
                params["sigma"] - target_sigma
            ) ** 2

        result = ew.fit(nll, {"mu": 0.0, "sigma": 0.5}, fn_args=(3.0, 1.5))

        assert abs(result.params["mu"] - 3.0) < 1e-4
        assert abs(result.params["sigma"] - 1.5) < 1e-4

    def test_keyword_args(self):
        """Test fit() with keyword arguments."""

        def nll(params, *, target_mu, target_sigma):
            return (params["mu"] - target_mu) ** 2 + (
                params["sigma"] - target_sigma
            ) ** 2

        result = ew.fit(
            nll,
            {"mu": 0.0, "sigma": 0.5},
            fn_kwargs={"target_mu": 4.0, "target_sigma": 0.8},
        )

        assert abs(result.params["mu"] - 4.0) < 1e-4
        assert abs(result.params["sigma"] - 0.8) < 1e-4

    def test_both_args_and_kwargs(self):
        """Test fit() with both positional and keyword arguments."""

        def nll(params, target_mu, *, offset):
            return (params["mu"] - target_mu - offset) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fn_args=(2.0,), fn_kwargs={"offset": 0.5})

        assert abs(result.params["mu"] - 2.5) < 1e-4

    def test_args_with_fixed_params(self):
        """Test additional args combined with fixed parameters."""

        def nll(params, scale):
            return (params["a"] - scale) ** 2 + (params["b"] - 10.0) ** 2

        result = ew.fit(
            nll,
            {"a": 0.0, "b": 5.0},
            fixed=["b"],
            fn_args=(7.0,),
        )

        assert abs(result.params["a"] - 7.0) < 1e-4
        assert abs(result.params["b"] - 5.0) < 1e-10

    def test_empty_args(self):
        """Test that empty args tuple works."""

        def nll(params):
            return (params["mu"] - 1.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fn_args=())
        assert abs(result.params["mu"] - 1.0) < 1e-4

    def test_empty_kwargs(self):
        """Test that empty kwargs dict works."""

        def nll(params):
            return (params["mu"] - 1.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fn_kwargs={})
        assert abs(result.params["mu"] - 1.0) < 1e-4

    def test_none_kwargs(self):
        """Test that fn_kwargs=None works (default)."""

        def nll(params):
            return (params["mu"] - 1.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, fn_kwargs=None)
        assert abs(result.params["mu"] - 1.0) < 1e-4


# ============================================================================
# fit() function tests - solver options
# ============================================================================


class TestFitSolverOptions:
    """Tests for fit() with custom solvers and solver options."""

    def test_custom_solver(self):
        """Test fit() with custom solver."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        custom_solver = optx.BFGS(rtol=1e-6, atol=1e-6)
        result = ew.fit(nll, {"mu": 0.0}, solver=custom_solver)

        assert abs(result.params["mu"] - 2.0) < 1e-5

    def test_solver_kwargs(self):
        """Test that solver_kwargs are passed through."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = ew.fit(nll, {"mu": 0.0}, max_steps=50)

        assert abs(result.params["mu"] - 2.0) < 1e-4


# ============================================================================
# fit() function tests - realistic examples
# ============================================================================


class TestFitRealisticExamples:
    """Tests with realistic statistical models."""

    def test_poisson_likelihood(self):
        """Test Poisson negative log-likelihood fit."""

        def poisson_nll(params, observed):
            signal = 10.0
            expected = params["mu"] * signal + params["background"]
            # Poisson NLL (ignoring constant term)
            return expected - observed * jnp.log(expected)

        observed = 25.0
        result = ew.fit(poisson_nll, {"mu": 1.0, "background": 10.0}, fn_args=(observed,))

        # MLE for Poisson: expected ≈ observed
        expected_total = result.params["mu"] * 10.0 + result.params["background"]
        assert (
            abs(expected_total - observed) < 0.02
        )  # Relaxed tolerance for optimizer convergence

    def test_gaussian_with_constraint(self):
        """Test Gaussian likelihood with constraint term."""

        def nll_with_constraint(params):
            # Main term
            main = (params["mu"] - 2.0) ** 2

            # Constraint on sigma (Gaussian prior)
            constraint = ((params["sigma"] - 1.0) / 0.2) ** 2

            return main + constraint

        result = ew.fit(nll_with_constraint, {"mu": 0.0, "sigma": 0.5})

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-3


# ============================================================================
# fit() function tests - parameter bounds
# ============================================================================


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
            bounds={"mu": MinuitTransform(lower=0.0, upper=5.0)},  # Constrain to [0, 5]
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
            bounds={
                "mu": MinuitTransform(lower=0.0, upper=10.0)
            },  # Constrain to [0, 10]
        )

        # Should hit the lower bound since true minimum is at -5
        assert 0.0 <= result.params["mu"] <= 10.0
        assert jnp.isclose(result.params["mu"], 0.0, atol=1e-2)
        assert result.nll < 100.0  # Should be at (0+5)^2 = 25

    def test_fit_within_bounds_unconstrained(self):
        """Test that bounds don't affect fit when minimum is within bounds."""

        def nll(params):
            # Unconstrained minimum at mu=2.5 (inside [0, 5])
            return (params["mu"] - 2.5) ** 2

        result = ew.fit(
            nll,
            {"mu": 1.0},
            bounds={"mu": MinuitTransform(lower=0.0, upper=5.0)},
        )

        # Should find true minimum
        assert 0.0 <= result.params["mu"] <= 5.0
        assert jnp.isclose(result.params["mu"], 2.5, atol=1e-2)

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

    @pytest.mark.parametrize(
        ("transform_factory", "target", "initial", "expected"),
        [
            (partial(MinuitTransform, lower=0.0, upper=5.0), 2.5, 0.1, 2.5),
            (partial(SigmoidTransform, lower=-2.0, upper=3.0), 2.0, -1.0, 2.0),
            (
                partial(OneSidedLogTransform, bound=0.0, direction="lower"),
                -3.0,
                1.0,
                0.0,
            ),
            (
                partial(OneSidedLogTransform, bound=5.0, direction="upper"),
                8.0,
                1.0,
                5.0,
            ),
            (partial(SoftPlusTransform), 1.5, 0.2, 1.5),
        ],
    )
    def test_fit_supports_all_transform_variants(
        self,
        transform_factory,
        target,
        initial,
        expected,
    ):
        """Ensure fit integrates each transform class."""

        def nll(params):
            return (params["x"] - target) ** 2

        result = ew.fit(
            nll,
            {"x": initial},
            bounds={"x": transform_factory()},
        )

        assert jnp.isclose(result.params["x"], expected, atol=1e-2)

    @pytest.mark.parametrize(
        ("transform_factory", "target", "initial", "check"),
        [
            (
                partial(MinuitTransform, lower=0.0, upper=1.0),
                -2.0,
                0.3,
                _expect_close(0.0),
            ),
            (
                partial(MinuitTransform, lower=0.0, upper=1.0),
                5.0,
                0.7,
                _expect_close(1.0),
            ),
            (
                partial(SigmoidTransform, lower=0.0, upper=1.0),
                -5.0,
                0.4,
                _expect_close(0.0),
            ),
            (
                partial(SigmoidTransform, lower=0.0, upper=1.0),
                5.0,
                0.6,
                _expect_close(1.0),
            ),
            (
                partial(OneSidedLogTransform, bound=0.0, direction="lower"),
                -5.0,
                0.8,
                _expect_close(0.0),
            ),
            (
                partial(OneSidedLogTransform, bound=2.0, direction="upper"),
                10.0,
                1.0,
                _expect_close(2.0),
            ),
            (
                partial(SoftPlusTransform),
                -1.0,
                0.5,
                _expect_interval(lower=0.0, upper=5e-2),
            ),
        ],
    )
    def test_fit_transforms_enforce_bounds(
        self,
        transform_factory,
        target,
        initial,
        check,
    ):
        """Verify each transform clamps solutions at its boundary."""

        def nll(params):
            return (params["x"] - target) ** 2

        result = ew.fit(
            nll,
            {"x": initial},
            bounds={"x": transform_factory()},
        )

        check(float(result.params["x"]))


# ============================================================================
# fixed_param_fit() function tests
# ============================================================================


class TestFixedParamFit:
    """Tests for fixed_param_fit() function."""

    def test_fix_single_parameter(self):
        """Test fixing a single parameter to a specific value."""

        def nll(params):
            return (params["mu"] - 3.0) ** 2 + (params["sigma"] - 1.0) ** 2

        result = ew.fixed_param_fit({"mu": 2.0}, nll, {"mu": 0.0, "sigma": 0.5})

        assert abs(result.params["mu"] - 2.0) < 1e-10  # Should be exactly 2.0
        assert abs(result.params["sigma"] - 1.0) < 1e-4

    def test_fix_multiple_parameters(self):
        """Test fixing multiple parameters."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = ew.fixed_param_fit(
            {"a": 5.0, "c": 7.0}, nll, {"a": 0.0, "b": 0.0, "c": 0.0}
        )

        assert abs(result.params["a"] - 5.0) < 1e-10
        assert abs(result.params["b"] - 2.0) < 1e-4
        assert abs(result.params["c"] - 7.0) < 1e-10

    def test_with_additional_fixed_list(self):
        """Test combining param_values with additional fixed list."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = ew.fixed_param_fit(
            {"a": 5.0},
            nll,
            {"a": 0.0, "b": 10.0, "c": 0.0},
            fixed=["b"],
        )

        assert abs(result.params["a"] - 5.0) < 1e-10
        assert abs(result.params["b"] - 10.0) < 1e-10
        assert abs(result.params["c"] - 3.0) < 1e-4

    def test_with_args(self):
        """Test fixed_param_fit() with additional positional arguments."""

        def nll(params, target):
            return (params["mu"] - target) ** 2 + (params["sigma"] - 1.0) ** 2

        result = ew.fixed_param_fit(
            {"mu": 3.0}, nll, {"mu": 0.0, "sigma": 0.5}, fn_args=(5.0,)
        )

        assert abs(result.params["mu"] - 3.0) < 1e-10
        assert abs(result.params["sigma"] - 1.0) < 1e-4

    def test_with_kwargs(self):
        """Test fixed_param_fit() with keyword arguments."""

        def nll(params, *, target_sigma):
            return (params["mu"] - 2.0) ** 2 + (params["sigma"] - target_sigma) ** 2

        result = ew.fixed_param_fit(
            {"mu": 1.5}, nll, {"mu": 0.0, "sigma": 0.5}, fn_kwargs={"target_sigma": 1.8}
        )

        assert abs(result.params["mu"] - 1.5) < 1e-10
        assert abs(result.params["sigma"] - 1.8) < 1e-4

    def test_with_both_args_and_kwargs(self):
        """Test fixed_param_fit() with both args and kwargs."""

        def nll(params, scale, *, offset):
            return (params["mu"] - scale - offset) ** 2

        result = ew.fixed_param_fit(
            {"mu": 5.0}, nll, {"mu": 0.0}, fn_args=(2.0,), fn_kwargs={"offset": 3.0}
        )

        assert abs(result.params["mu"] - 5.0) < 1e-10

    def test_profile_likelihood_scan_point(self):
        """Test using fixed_param_fit for a single profile likelihood point."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2

        # Unconditional fit
        result_uncond = ew.fit(nll, {"mu": 0.0, "sigma": 0.5})

        # Profile likelihood at mu=1.5
        result_profile = ew.fixed_param_fit({"mu": 1.5}, nll, {"mu": 0.0, "sigma": 0.5})

        # Profile likelihood should have higher NLL than unconditional
        assert result_profile.nll > result_uncond.nll
        assert abs(result_profile.params["mu"] - 1.5) < 1e-10
        assert abs(result_profile.params["sigma"] - 1.0) < 1e-4

    def test_fixed_param_fit_with_bounds_constrains_free_param(self):
        """Test profile likelihood fit where free parameter would violate bound."""

        def nll(params):
            # Unconstrained minimum: mu=any, sigma=-10
            return (params["sigma"] + 10.0) ** 2

        result = ew.fixed_param_fit(
            {"mu": 1.5},  # Fix mu
            nll,
            {"mu": 1.0, "sigma": 0.5},
            bounds={
                "sigma": OneSidedLogTransform(bound=0.0, direction="lower")
            },  # sigma >= 0
        )

        # mu should be fixed at 1.5
        assert jnp.isclose(result.params["mu"], 1.5, atol=1e-6)

        # sigma should hit lower bound
        assert result.params["sigma"] >= 0.0
        assert jnp.isclose(result.params["sigma"], 0.0, atol=1e-2)
