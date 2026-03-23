"""Tests for parameter uncertainty estimation.

Mathematical Background
-----------------------
These tests use a simple quadratic NLL with known analytical properties:

    NLL(x, y) = (x - a)² / (2σ_x²) + (y - b)² / (2σ_y²)

This corresponds to a Gaussian likelihood L ∝ exp(-NLL), i.e., two independent
Gaussians with standard deviations σ_x and σ_y.

The Hessian (matrix of second derivatives) is:

    H = | 1/σ_x²    0     |
        |   0     1/σ_y²  |

The covariance matrix (inverse Hessian) is:

    Cov = H⁻¹ = | σ_x²   0    |
                |  0    σ_y²  |

The parameter uncertainties (Cramér-Rao bound) are:

    σ(x) = √(Cov_xx) = σ_x
    σ(y) = √(Cov_yy) = σ_y

So when we construct the NLL with sigma_x=0.5, sigma_y=0.25, the expected
uncertainties are exactly 0.5 and 0.25.
"""

from __future__ import annotations

import typing as tp
from types import EllipsisType
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.uncertainty import (
    correlation_matrix,
    covariance_matrix,
    hessian_matrix,
    uncertainties,
)

jax.config.update("jax_enable_x64", True)

# Type aliases for cleaner annotations
FState: tp.TypeAlias = sl.State[float]
EState: tp.TypeAlias = sl.State[float | EllipsisType]


# ============================================================================
# Test helpers
# ============================================================================


def simple_quadratic_nll(sigma_x: float = 1.0, sigma_y: float = 1.0):
    """Create a simple quadratic NLL with known analytical solution.

    See module docstring for derivation of expected Hessian/covariance/uncertainties.
    """

    def nll(params, observation):
        return (params["x"] - observation["x"]) ** 2 / (2 * sigma_x**2) + (
            params["y"] - observation["y"]
        ) ** 2 / (2 * sigma_y**2)

    return nll


# Standard observation for tests - the "true" values
OBSERVED: dict[str, float] = {"x": 2.0, "y": 3.0}


# ============================================================================
# hessian_matrix tests
# ============================================================================


class TestHessianMatrix:
    """Tests for hessian_matrix function."""

    def test_simple_quadratic(self):
        """Test Hessian of simple quadratic has correct diagonal entries."""
        sigma_x, sigma_y = 0.5, 0.25
        nll = simple_quadratic_nll(sigma_x, sigma_y)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        hess = hessian_matrix(nll, params, OBSERVED)

        # H_ii = 1/sigma_i^2 (see module docstring)
        expected_hess = jnp.diag(jnp.array([1 / sigma_x**2, 1 / sigma_y**2]))
        assert jnp.allclose(hess, expected_hess, atol=1e-10)

    def test_shape_matches_free_params(self):
        """Hessian shape should be (n_free, n_free)."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        hess = hessian_matrix(nll, params, OBSERVED)

        assert hess.shape == (2, 2)

    def test_fixed_params_excluded(self):
        """Fixed parameters should be excluded from Hessian."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        hess = hessian_matrix(nll, params, OBSERVED, fixed=fixed)

        # Only x is free, so Hessian is 1x1
        assert hess.shape == (1, 1)
        assert jnp.isclose(hess[0, 0], 1.0, atol=1e-10)  # 1/sigma_x^2 = 1/1^2 = 1

    def test_all_fixed_returns_empty(self):
        """All params fixed should return empty (0x0) Hessian."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"x": ..., "y": ...})

        hess = hessian_matrix(nll, params, OBSERVED, fixed=fixed)

        assert hess.shape == (0, 0)

    def test_symmetric(self):
        """Hessian should be symmetric."""

        def correlated_nll(params, observation):
            # NLL with off-diagonal Hessian terms
            del observation
            x, y = params["x"], params["y"]
            return x**2 + y**2 + x * y

        params: FState = sl.State.from_pytree({"x": 1.0, "y": 1.0})
        hess = hessian_matrix(correlated_nll, params, {})

        assert jnp.allclose(hess, hess.T, atol=1e-10)

    def test_validates_params_type(self):
        """Should raise TypeError if params is not a State."""
        nll = simple_quadratic_nll()

        with pytest.raises(TypeError, match="params must be a State"):
            hessian_matrix(nll, {"x": 2.0, "y": 3.0}, OBSERVED)  # type: ignore[arg-type]

    def test_validates_fixed_type(self):
        """Should raise TypeError if fixed is not State or None."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        with pytest.raises(TypeError, match="fixed must be a State or None"):
            hessian_matrix(nll, params, OBSERVED, fixed={"y": ...})  # type: ignore[arg-type]


# ============================================================================
# covariance_matrix tests
# ============================================================================


class TestCovarianceMatrix:
    """Tests for covariance_matrix function."""

    def test_simple_quadratic(self):
        """Covariance should be inverse of Hessian."""
        sigma_x, sigma_y = 0.5, 0.25
        nll = simple_quadratic_nll(sigma_x, sigma_y)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        cov = covariance_matrix(nll, params, OBSERVED)

        # Cov_ii = sigma_i^2 (see module docstring)
        expected_cov = jnp.diag(jnp.array([sigma_x**2, sigma_y**2]))
        assert jnp.allclose(cov, expected_cov, atol=1e-10)

    def test_positive_definite(self):
        """Covariance matrix should be positive definite."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        cov = covariance_matrix(nll, params, OBSERVED)

        # All eigenvalues should be positive
        eigenvalues = jnp.linalg.eigvalsh(cov)
        assert jnp.all(eigenvalues > 0)

    def test_fixed_params_excluded(self):
        """Fixed parameters should be excluded from covariance."""
        sigma_x = 0.5
        nll = simple_quadratic_nll(sigma_x, 1.0)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        cov = covariance_matrix(nll, params, OBSERVED, fixed=fixed)

        assert cov.shape == (1, 1)
        assert jnp.isclose(cov[0, 0], sigma_x**2, atol=1e-10)


# ============================================================================
# correlation_matrix tests
# ============================================================================


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    def test_diagonal_is_one(self):
        """Diagonal elements should be exactly 1.0."""
        nll = simple_quadratic_nll(0.5, 0.25)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        corr = correlation_matrix(nll, params, OBSERVED)

        assert jnp.allclose(jnp.diag(corr), 1.0, atol=1e-10)

    def test_uncorrelated_params(self):
        """Uncorrelated params should have off-diagonal = 0."""
        # Simple quadratic NLL has diagonal Hessian -> uncorrelated
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        corr = correlation_matrix(nll, params, OBSERVED)

        # Off-diagonal should be 0
        assert jnp.isclose(corr[0, 1], 0.0, atol=1e-10)
        assert jnp.isclose(corr[1, 0], 0.0, atol=1e-10)

    def test_correlated_params(self):
        """Correlated params should have off-diagonal in [-1, 1]."""

        def correlated_nll(params, observation):
            del observation
            x, y = params["x"], params["y"]
            # Introduce correlation via cross term
            return x**2 + y**2 + 0.5 * x * y

        params: FState = sl.State.from_pytree({"x": 0.0, "y": 0.0})
        corr = correlation_matrix(correlated_nll, params, {})

        # Off-diagonal should be non-zero and in valid range
        assert -1.0 <= corr[0, 1] <= 1.0
        assert corr[0, 1] != 0.0
        # Should be symmetric
        assert jnp.isclose(corr[0, 1], corr[1, 0], atol=1e-10)

    def test_values_in_valid_range(self):
        """All correlation values should be in [-1, 1]."""

        def correlated_nll(params, observation):
            del observation
            x, y = params["x"], params["y"]
            return x**2 + y**2 + 0.8 * x * y

        params: FState = sl.State.from_pytree({"x": 0.0, "y": 0.0})
        corr = correlation_matrix(correlated_nll, params, {})

        assert jnp.all(corr >= -1.0)
        assert jnp.all(corr <= 1.0)


# ============================================================================
# uncertainties tests
# ============================================================================


class TestUncertainties:
    """Tests for uncertainties function."""

    def test_simple_quadratic(self):
        """Uncertainties should equal sigma values from NLL construction."""
        sigma_x, sigma_y = 0.5, 0.25
        nll = simple_quadratic_nll(sigma_x, sigma_y)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        errs = uncertainties(nll, params, OBSERVED)

        # sigma_i = sqrt(Cov_ii) = sigma_i (see module docstring)
        assert jnp.isclose(errs["x"], sigma_x, atol=1e-10)
        assert jnp.isclose(errs["y"], sigma_y, atol=1e-10)

    def test_returns_state(self):
        """Result should be a State object."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        errs = uncertainties(nll, params, OBSERVED)

        assert isinstance(errs, sl.State)

    def test_fixed_params_none_uncertainty(self):
        """Fixed parameters should appear with None uncertainty."""
        sigma_x = 0.5
        nll = simple_quadratic_nll(sigma_x, 1.0)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        errs = uncertainties(nll, params, OBSERVED, fixed=fixed)

        assert "x" in errs
        assert jnp.isclose(errs["x"], sigma_x, atol=1e-10)
        # 'y' should be in errs with value None
        assert "y" in errs
        assert errs["y"] is None

    def test_single_parameter(self):
        """Should work with single parameter."""

        def nll(params, observation):
            return (params["x"] - observation["x"]) ** 2 / (2 * 0.3**2)

        params: FState = sl.State.from_pytree({"x": 5.0})
        errs = uncertainties(nll, params, {"x": 5.0})

        assert jnp.isclose(errs["x"], 0.3, atol=1e-10)

    def test_nested_params(self):
        """Should work with nested parameter structure."""
        sigma = 0.4

        def nll(params, observation):
            return (params["level1"]["mu"] - observation["mu"]) ** 2 / (2 * sigma**2)

        params: FState = sl.State.from_pytree({"level1": {"mu": 2.0}})
        errs = uncertainties(nll, params, {"mu": 2.0})

        assert jnp.isclose(errs["level1.mu"], sigma, atol=1e-10)


# ============================================================================
# Integration tests
# ============================================================================


class TestUncertaintyIntegration:
    """Integration tests for uncertainty functions."""

    def test_workflow_after_fit(self):
        """Test typical workflow: fit -> uncertainties."""
        import everwillow as ew

        sigma_x, sigma_y = 0.5, 0.25
        nll = simple_quadratic_nll(sigma_x, sigma_y)
        initial_params: FState = sl.State.from_pytree({"x": 0.0, "y": 0.0})

        # Fit
        result = ew.fit(nll, initial_params, OBSERVED)

        # Extract uncertainties at fitted params
        errs = uncertainties(nll, result.params, OBSERVED)

        # Check fitted values
        assert jnp.isclose(result.params["x"], 2.0, atol=1e-4)
        assert jnp.isclose(result.params["y"], 3.0, atol=1e-4)

        # Check uncertainties
        assert jnp.isclose(errs["x"], sigma_x, atol=1e-4)
        assert jnp.isclose(errs["y"], sigma_y, atol=1e-4)

    def test_covariance_and_uncertainties_consistent(self):
        """uncertainties should equal sqrt(diag(covariance))."""
        nll = simple_quadratic_nll(0.5, 0.25)
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        cov = covariance_matrix(nll, params, OBSERVED)
        errs = uncertainties(nll, params, OBSERVED)

        # Manual extraction from covariance diagonal
        expected_errs = jnp.sqrt(jnp.diag(cov))
        assert jnp.isclose(errs["x"], expected_errs[0], atol=1e-10)
        assert jnp.isclose(errs["y"], expected_errs[1], atol=1e-10)

    def test_hessian_times_covariance_is_identity(self):
        """H @ Cov = I (by definition of covariance as inverse Hessian)."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})

        hess = hessian_matrix(nll, params, OBSERVED)
        cov = covariance_matrix(nll, params, OBSERVED)

        product = hess @ cov
        identity = jnp.eye(2)
        assert jnp.allclose(product, identity, atol=1e-10)


# ============================================================================
# Argument forwarding tests (call_args verification)
# ============================================================================


class TestArgumentForwarding:
    """Tests verifying kwargs are properly forwarded through function chains."""

    def test_covariance_matrix_forwards_fixed_to_hessian(self):
        """covariance_matrix() should forward fixed to hessian_matrix."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        with patch(
            "everwillow._src.inference.uncertainty.hessian_matrix"
        ) as mock_hessian:
            # Return a valid 1x1 hessian (since y is fixed)
            mock_hessian.return_value = jnp.array([[1.0]])

            covariance_matrix(nll, params, OBSERVED, fixed=fixed)

            assert mock_hessian.call_count == 1
            call_kwargs = mock_hessian.call_args[1]
            assert call_kwargs["fixed"] is fixed

    def test_correlation_matrix_forwards_fixed_to_covariance(self):
        """correlation_matrix() should forward fixed to covariance_matrix."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        with patch(
            "everwillow._src.inference.uncertainty.covariance_matrix"
        ) as mock_cov:
            # Return a valid 1x1 covariance (since y is fixed)
            mock_cov.return_value = jnp.array([[0.25]])

            correlation_matrix(nll, params, OBSERVED, fixed=fixed)

            assert mock_cov.call_count == 1
            call_kwargs = mock_cov.call_args[1]
            assert call_kwargs["fixed"] is fixed

    def test_uncertainties_forwards_fixed_to_covariance(self):
        """uncertainties() should forward fixed to covariance_matrix."""
        nll = simple_quadratic_nll()
        params: FState = sl.State.from_pytree({"x": 2.0, "y": 3.0})
        fixed: EState = sl.State.from_pytree({"y": ...})

        with patch(
            "everwillow._src.inference.uncertainty.covariance_matrix"
        ) as mock_cov:
            # Return a valid 1x1 covariance (since y is fixed)
            mock_cov.return_value = jnp.array([[0.25]])

            uncertainties(nll, params, OBSERVED, fixed=fixed)

            assert mock_cov.call_count == 1
            call_kwargs = mock_cov.call_args[1]
            assert call_kwargs["fixed"] is fixed
