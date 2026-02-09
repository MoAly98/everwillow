"""Tests for hypothesis test distributions.

Tests asymptotic and empirical distributions with concrete expected p-values.
Uses the formulas from Cowan et al., arXiv:1007.1727.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from everwillow.inference.hypotest import (
    EmpiricalDistribution,
    Q0Asymptotic,
    QMuAsymptotic,
    QTildeAsymptotic,
    TMuAsymptotic,
)
from everwillow.inference.hypotest import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)

# =============================================================================
# Helper functions for computing expected p-values
# =============================================================================


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_sf(x: float) -> float:
    """Standard normal survival function: 1 - CDF."""
    return 1.0 - normal_cdf(x)


# =============================================================================
# QTildeAsymptotic Tests
# =============================================================================


class TestQTildeAsymptotic:
    """Tests for QTildeAsymptotic distribution."""

    def test_pvalues_at_asimov(self):
        """At q=q_asimov, pnull=0.5 and palt depends on q_asimov.

        For q=q_asimov=4.0:
        - pnull = 1 - Φ(√q - √q_A) = 1 - Φ(0) = 0.5
        - palt = 1 - Φ(√q) = 1 - Φ(2) = 0.0228
        """
        q = 4.0
        q_asimov = 4.0
        expected_pnull = 0.5
        expected_palt = normal_sf(2.0)  # 0.02275

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QTildeAsymptotic()
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-4)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)

    def test_pvalues_q_less_than_asimov(self):
        """Test p-values when q < q_asimov (downward fluctuation).

        q=1.0, q_asimov=4.0:
        - pnull = 1 - Φ(1 - 2) = 1 - Φ(-1) = Φ(1) = 0.8413
        - palt = 1 - Φ(1) = 0.1587
        """
        q = 1.0
        q_asimov = 4.0
        sqrt_q = math.sqrt(q)
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf(sqrt_q - sqrt_q_asimov)  # Φ(1) = 0.8413
        expected_palt = normal_sf(sqrt_q)  # 0.1587

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QTildeAsymptotic()
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)

    def test_pvalues_q_greater_than_asimov(self):
        """Test boundary case when q > q_asimov (Eq. 66).

        q=9.0, q_asimov=4.0:
        - pnull = 1 - Φ((q - q_A)/(2√q_A)) = 1 - Φ(5/4) = 1 - Φ(1.25) = 0.1056
        - palt = 1 - Φ((q + q_A)/(2√q_A)) = 1 - Φ(13/4) = 1 - Φ(3.25) = 0.00058
        """
        q = 9.0
        q_asimov = 4.0
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf((q - q_asimov) / (2 * sqrt_q_asimov))  # 0.1056
        expected_palt = normal_sf((q + q_asimov) / (2 * sqrt_q_asimov))  # 0.00058

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QTildeAsymptotic()
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-2)

    def test_expected_bands_structure(self):
        """Test expected bands returns correct structure."""
        result = TSResult(q=jnp.array(4.0), extras={"q_asimov": jnp.array(4.0)})
        dist = QTildeAsymptotic()
        bands = dist.expected_pvalues(result)

        assert hasattr(bands, "minus_2sigma")
        assert hasattr(bands, "minus_1sigma")
        assert hasattr(bands, "median")
        assert hasattr(bands, "plus_1sigma")
        assert hasattr(bands, "plus_2sigma")

    def test_expected_bands_median(self):
        """At median (nsigma=0), expected CLs should be moderate.

        For q_asimov=4.0:
        - pnull_median = 0.5
        - palt_median = 1 - Φ(2) = 0.0228
        - CLs = 0.0228 / 0.5 = 0.0456
        """
        q_asimov = 4.0
        expected_pnull = 0.5
        expected_palt = normal_sf(math.sqrt(q_asimov))  # 0.0228

        result = TSResult(q=jnp.array(4.0), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QTildeAsymptotic()
        bands = dist.expected_pvalues(result)

        pnull_med, palt_med = bands.median
        assert float(pnull_med) == pytest.approx(expected_pnull, rel=1e-4)
        assert float(palt_med) == pytest.approx(expected_palt, rel=1e-3)


# =============================================================================
# QMuAsymptotic Tests
# =============================================================================


class TestQMuAsymptotic:
    """Tests for QMuAsymptotic distribution."""

    def test_expected_bands_median(self):
        """Test expected bands at median (nsigma=0).

        For q=q_asimov=4.0:
        - pnull = 1 - Φ(√q - √q_asimov - 0) = 1 - Φ(0) = 0.5
        - palt = 1 - Φ(√q - 0) = 1 - Φ(2) = 0.0228
        """
        q_asimov = 4.0
        expected_pnull = 0.5
        expected_palt = normal_sf(2.0)  # 0.0228

        result = TSResult(
            q=jnp.array(q_asimov), extras={"q_asimov": jnp.array(q_asimov)}
        )
        dist = QMuAsymptotic()
        bands = dist.expected_pvalues(result)

        pnull_med, palt_med = bands.median
        assert float(pnull_med) == pytest.approx(expected_pnull, rel=1e-4)
        assert float(palt_med) == pytest.approx(expected_palt, rel=1e-3)

    def test_pvalues_at_asimov(self):
        """At q=q_asimov, pnull=0.5.

        For q=q_asimov=4.0:
        - pnull = 1 - Φ(√q - √q_A) = 1 - Φ(0) = 0.5
        - palt = 1 - Φ(√q) = 1 - Φ(2) = 0.0228
        """
        q = 4.0
        q_asimov = 4.0
        expected_pnull = 0.5
        expected_palt = normal_sf(2.0)

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QMuAsymptotic()
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-4)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)

    def test_pvalues_no_boundary(self):
        """QMuAsymptotic has no boundary handling (unlike QTildeAsymptotic).

        For q=9.0, q_asimov=4.0:
        - pnull = 1 - Φ(3 - 2) = 1 - Φ(1) = 0.1587
        - palt = 1 - Φ(3) = 0.00135
        """
        q = 9.0
        q_asimov = 4.0
        sqrt_q = math.sqrt(q)
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf(sqrt_q - sqrt_q_asimov)  # 0.1587
        expected_palt = normal_sf(sqrt_q)  # 0.00135

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q_asimov)})
        dist = QMuAsymptotic()
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-2)


# =============================================================================
# Q0Asymptotic Tests
# =============================================================================


class TestQ0Asymptotic:
    """Tests for Q0Asymptotic distribution (discovery)."""

    def test_expected_bands_median(self):
        """Test expected bands at median (nsigma=0).

        For q=q_asimov=4.0:
        - pnull = 1 - Φ(√q - 0) = 1 - Φ(2) = 0.0228
        - palt = 1 - Φ(√q - √q_asimov - 0) = 1 - Φ(0) = 0.5
        """
        q_asimov = 4.0
        expected_pnull = normal_sf(2.0)  # 0.0228
        expected_palt = 0.5

        result = TSResult(
            q=jnp.array(q_asimov), extras={"q_asimov": jnp.array(q_asimov)}
        )
        dist = Q0Asymptotic()
        bands = dist.expected_pvalues(result)

        pnull_med, palt_med = bands.median
        assert float(pnull_med) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt_med) == pytest.approx(expected_palt, rel=1e-4)

    def test_pvalues_discovery(self):
        """Test discovery p-value.

        For q0=9.0:
        - pnull = 1 - Φ(3) = 0.00135 (discovery significance)
        """
        q = 9.0
        expected_pnull = normal_sf(3.0)  # 0.00135

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q)})
        dist = Q0Asymptotic()
        pnull, _palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-2)

    def test_pvalues_no_excess(self):
        """Test p-value when no excess (q0=0).

        For q0=0:
        - pnull = 1 - Φ(0) = 0.5
        """
        q = 0.0
        expected_pnull = 0.5

        result = TSResult(q=jnp.array(q), extras={"q_asimov": jnp.array(q)})
        dist = Q0Asymptotic()
        pnull, _palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-4)


# =============================================================================
# TMuAsymptotic Tests
# =============================================================================


class TestTMuAsymptotic:
    """Tests for TMuAsymptotic distribution (two-sided)."""

    def test_expected_bands_median(self):
        """Test expected bands at median (nsigma=0).

        For t=t_asimov=4.0:
        - sigma = √|t_asimov| = 2
        - pnull = 2 * (1 - Φ(|t|/sigma)) = 2 * (1 - Φ(2)) = 0.0456
        - palt = 2 * (1 - Φ((|t| + √|t_asimov|)/sigma)) = 2 * (1 - Φ(2)) = 0.0456
        """
        t_asimov = 4.0
        sigma = math.sqrt(t_asimov)
        expected_pnull = 2 * normal_sf(t_asimov / sigma)  # 0.0456
        expected_palt = 2 * normal_sf((t_asimov + sigma) / sigma)  # 0.00134

        result = TSResult(
            q=jnp.array(t_asimov), extras={"q_asimov": jnp.array(t_asimov)}
        )
        dist = TMuAsymptotic()
        bands = dist.expected_pvalues(result)

        pnull_med, palt_med = bands.median
        assert float(pnull_med) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt_med) == pytest.approx(expected_palt, rel=1e-2)

    def test_pvalues_at_zero(self):
        """At t=0, pnull should be 1.0 (no exclusion).

        For t=0, t_asimov=4.0:
        sigma = √4 = 2
        pnull = 2 * (1 - Φ(0/2)) = 2 * 0.5 = 1.0
        """
        t = 0.0
        t_asimov = 4.0

        result = TSResult(q=jnp.array(t), extras={"q_asimov": jnp.array(t_asimov)})
        dist = TMuAsymptotic()
        pnull, _palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(1.0, rel=1e-4)

    def test_pvalues_positive_t(self):
        """Test two-sided p-value for positive t.

        For t=4.0, t_asimov=4.0:
        sigma = 2
        pnull = 2 * (1 - Φ(4/2)) = 2 * (1 - Φ(2)) = 2 * 0.0228 = 0.0456
        """
        t = 4.0
        t_asimov = 4.0
        sigma = math.sqrt(t_asimov)
        expected_pnull = 2 * normal_sf(abs(t) / sigma)  # 0.0456

        result = TSResult(q=jnp.array(t), extras={"q_asimov": jnp.array(t_asimov)})
        dist = TMuAsymptotic()
        pnull, _palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)


# =============================================================================
# EmpiricalDistribution Tests
# =============================================================================


class TestEmpiricalDistribution:
    """Tests for EmpiricalDistribution from toys."""

    def test_pvalues_simple(self):
        """Test empirical p-values with known toy arrays.

        q_null = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        q_alt = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

        At q_obs=5:
        - pnull = fraction of q_null >= 5 = 6/10 = 0.6
        - palt = fraction of q_alt >= 5 = 1/10 = 0.1
        """
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_obs = 5.0
        expected_pnull = 0.6
        expected_palt = 0.1

        dist = EmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(q=jnp.array(q_obs), extras={})
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-5)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-5)

    def test_pvalues_edge_case_all_above(self):
        """Test when q_obs is below all toys."""
        q_null = jnp.array([5.0, 6.0, 7.0, 8.0, 9.0])
        q_alt = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_obs = 0.5

        dist = EmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(q=jnp.array(q_obs), extras={})
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(1.0, rel=1e-5)
        assert float(palt) == pytest.approx(1.0, rel=1e-5)

    def test_pvalues_edge_case_all_below(self):
        """Test when q_obs is above all toys."""
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        q_obs = 10.0

        dist = EmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(q=jnp.array(q_obs), extras={})
        pnull, palt = dist.pvalues(result)

        assert float(pnull) == pytest.approx(0.0, abs=1e-5)
        assert float(palt) == pytest.approx(0.0, abs=1e-5)

    def test_expected_bands_structure(self):
        """Test expected bands returns correct structure."""
        q_null = jnp.linspace(0, 10, 100)
        q_alt = jnp.linspace(0, 5, 100)

        dist = EmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(q=jnp.array(5.0), extras={})
        bands = dist.expected_pvalues(result)

        assert hasattr(bands, "minus_2sigma")
        assert hasattr(bands, "median")
        assert hasattr(bands, "plus_2sigma")

    def test_cls_from_pvalues(self):
        """Test CLs = palt / pnull."""
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_obs = 5.0

        dist = EmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(q=jnp.array(q_obs), extras={})
        pnull, palt = dist.pvalues(result)

        expected_cls = 0.1 / 0.6  # 0.1667
        actual_cls = float(palt) / float(pnull)
        assert actual_cls == pytest.approx(expected_cls, rel=1e-5)
