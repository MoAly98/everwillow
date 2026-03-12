"""Tests for hypothesis test distributions.

Tests asymptotic and empirical distributions with concrete expected p-values.
Uses the formulas from Cowan et al., arXiv:1007.1727.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from everwillow.inference.hypotest import (
    Q0Asymptotic,
    QMuAsymptotic,
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
    TMuAsymptotic,
    ToyResult,
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
        """Test p-values when q equals q_asimov.

        For q=q_asimov=4.0:
        - null_pval (μ'=μ): 1 - Φ(√q) = 1 - Φ(2) = 0.0228
        - alt_pval (μ'=0):  1 - Φ(√q - √q_A) = 1 - Φ(0) = 0.5
        """
        q = 4.0
        q_asimov = 4.0
        expected_pnull = normal_sf(2.0)  # 0.02275
        expected_palt = 0.5

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = QTildeAsymptotic()
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-4)

    def test_pvalues_q_less_than_asimov(self):
        """Test p-values when q < q_asimov.

        q=1.0, q_asimov=4.0:
        - null_pval (μ'=μ): 1 - Φ(√1) = 1 - Φ(1) = 0.1587
        - alt_pval (μ'=0):  1 - Φ(1 - 2) = 1 - Φ(-1) = Φ(1) = 0.8413
        """
        q = 1.0
        q_asimov = 4.0
        sqrt_q = math.sqrt(q)
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf(sqrt_q)  # 0.1587
        expected_palt = normal_sf(sqrt_q - sqrt_q_asimov)  # Φ(1) = 0.8413

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = QTildeAsymptotic()
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)

    def test_pvalues_q_greater_than_asimov(self):
        """Test p-values when q > q_asimov (Eq. 66).

        q=9.0, q_asimov=4.0:
        - null_pval (μ'=μ): 1 - Φ((q+q_A)/(2√q_A)) = 1 - Φ(3.25) = 0.00058
        - alt_pval (μ'=0):  1 - Φ((q-q_A)/(2√q_A)) = 1 - Φ(1.25) = 0.1056
        """
        q = 9.0
        q_asimov = 4.0
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf((q + q_asimov) / (2 * sqrt_q_asimov))  # 0.00058
        expected_palt = normal_sf((q - q_asimov) / (2 * sqrt_q_asimov))  # 0.1056

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = QTildeAsymptotic()
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-2)


# =============================================================================
# QMuAsymptotic Tests
# =============================================================================


class TestQMuAsymptotic:
    """Tests for QMuAsymptotic distribution."""

    def test_pvalues_at_asimov(self):
        """Test p-values when q equals q_asimov.

        For q=q_asimov=4.0:
        - null_pval (μ'=μ): 1 - Φ(√q) = 1 - Φ(2) = 0.0228
        - alt_pval (μ'=0):  1 - Φ(√q - √q_A) = 1 - Φ(0) = 0.5
        """
        q = 4.0
        q_asimov = 4.0
        expected_pnull = normal_sf(2.0)  # 0.02275
        expected_palt = 0.5

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = QMuAsymptotic()
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-4)

    def test_pvalues_large_q(self):
        """Test p-values for large q (no piecewise boundary unlike QTilde).

        For q=9.0, q_asimov=4.0:
        - null_pval (μ'=μ): 1 - Φ(√9) = 1 - Φ(3) = 0.00135
        - alt_pval (μ'=0):  1 - Φ(3 - 2) = 1 - Φ(1) = 0.1587
        """
        q = 9.0
        q_asimov = 4.0
        sqrt_q = math.sqrt(q)
        sqrt_q_asimov = math.sqrt(q_asimov)
        expected_pnull = normal_sf(sqrt_q)  # 0.00135
        expected_palt = normal_sf(sqrt_q - sqrt_q_asimov)  # 0.1587

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = QMuAsymptotic()
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-2)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)


# =============================================================================
# Q0Asymptotic Tests
# =============================================================================


class TestQ0Asymptotic:
    """Tests for Q0Asymptotic distribution (discovery)."""

    def test_pvalues_discovery(self):
        """Test discovery p-value.

        For q0=9.0:
        - pnull = 1 - Φ(3) = 0.00135 (discovery significance)
        """
        q = 9.0
        expected_pnull = normal_sf(3.0)  # 0.00135

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(0.0),
            q_asimov=jnp.array(q),
        )
        dist = Q0Asymptotic()
        pnull = dist.null_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-2)

    def test_pvalues_no_excess(self):
        """Test p-value when no excess (q0=0).

        For q0=0:
        - pnull = 1 - Φ(0) = 0.5
        """
        q = 0.0
        expected_pnull = 0.5

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(0.0),
            q_asimov=jnp.array(q),
        )
        dist = Q0Asymptotic()
        pnull = dist.null_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-4)

    def test_alt_pval(self):
        """Test alt p-value with q_asimov.

        For q0=9.0, q_asimov=4.0:
        - palt = 1 - Φ(√9 - √4) = 1 - Φ(1) = 0.1587
        """
        q = 9.0
        q_asimov = 4.0
        expected_palt = normal_sf(math.sqrt(q) - math.sqrt(q_asimov))

        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(0.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = Q0Asymptotic()
        palt = dist.alt_pval(result)

        assert float(palt) == pytest.approx(expected_palt, rel=1e-3)


# =============================================================================
# TMuAsymptotic Tests
# =============================================================================


class TestTMuAsymptotic:
    """Tests for TMuAsymptotic distribution (two-sided)."""

    def test_pvalues_at_zero(self):
        """At t=0, pnull should be 1.0 (no exclusion).

        For t=0:
        pnull = 2 * (1 - Φ(0)) = 2 * 0.5 = 1.0
        """
        t = 0.0

        result = TSResult(
            value=jnp.array(t),
            test=jnp.array(1.0),
            q_asimov=jnp.array(4.0),
        )
        dist = TMuAsymptotic()
        pnull = dist.null_pval(result)

        assert float(pnull) == pytest.approx(1.0, rel=1e-4)

    def test_pvalues_positive_t(self):
        """Test two-sided p-value for positive t.

        For t=4.0:
        pnull = 2 * (1 - Φ(2)) = 2 * 0.0228 = 0.0456
        """
        t = 4.0
        expected_pnull = 2 * normal_sf(math.sqrt(t))  # 0.0456

        result = TSResult(
            value=jnp.array(t),
            test=jnp.array(1.0),
            q_asimov=jnp.array(4.0),
        )
        dist = TMuAsymptotic()
        pnull = dist.null_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-3)

    def test_alt_pval(self):
        """Test alt p-value.

        For t=4.0, q_asimov=4.0:
        palt = 2 - Φ(√4 + √4) - Φ(√4 - √4) = 2 - Φ(4) - Φ(0) = 2 - 0.99997 - 0.5
        """
        t = 4.0
        q_asimov = 4.0
        sqrt_t = math.sqrt(t)
        sqrt_qa = math.sqrt(q_asimov)
        expected_palt = (
            2.0 - normal_cdf(sqrt_t + sqrt_qa) - normal_cdf(sqrt_t - sqrt_qa)
        )

        result = TSResult(
            value=jnp.array(t),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = TMuAsymptotic()
        palt = dist.alt_pval(result)

        assert float(palt) == pytest.approx(expected_palt, rel=1e-2)


# =============================================================================
# SimpleEmpiricalDistribution Tests
# =============================================================================


class TestSimpleEmpiricalDistribution:
    """Tests for SimpleEmpiricalDistribution (tail counting)."""

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

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(expected_pnull, rel=1e-5)
        assert float(palt) == pytest.approx(expected_palt, rel=1e-5)

    def test_pvalues_edge_case_all_above(self):
        """Test when q_obs is below all toys."""
        q_null = jnp.array([5.0, 6.0, 7.0, 8.0, 9.0])
        q_alt = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_obs = 0.5

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(1.0, rel=1e-5)
        assert float(palt) == pytest.approx(1.0, rel=1e-5)

    def test_pvalues_edge_case_all_below(self):
        """Test when q_obs is above all toys."""
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        q_obs = 10.0

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(0.0, abs=1e-5)
        assert float(palt) == pytest.approx(0.0, abs=1e-5)

    def test_cls_from_pvalues(self):
        """Test CLs = palt / pnull."""
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_obs = 5.0

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        expected_cls = 0.1 / 0.6  # 0.1667
        actual_cls = float(palt) / float(pnull)
        assert actual_cls == pytest.approx(expected_cls, rel=1e-5)

    def test_from_toys(self):
        """Test constructing SimpleEmpiricalDistribution from ToyResult."""
        q_alt = jnp.array([1.0, 2.0, 3.0])
        q_null = jnp.array([4.0, 5.0, 6.0])
        toys = ToyResult(q_alt=q_alt, q_null=q_null)

        dist = SimpleEmpiricalDistribution.from_toys(toys)

        assert isinstance(dist, SimpleEmpiricalDistribution)
        assert jnp.array_equal(dist.q_alt, q_alt)
        assert jnp.array_equal(dist.q_null, q_null)


# =============================================================================
# Significance Tests
# =============================================================================


class TestSignificance:
    """Tests for null_significance and alt_significance."""

    def test_null_significance_q0(self):
        """For q_0=4: p = 1 - Φ(2) → Z = 2.0."""
        result = TSResult(value=jnp.array(4.0), test=jnp.array(0.0))
        dist = Q0Asymptotic()
        z = dist.null_significance(result)
        assert float(z) == pytest.approx(2.0, abs=1e-5)

    def test_null_significance_qmu(self):
        """For q_μ=9: p = 1 - Φ(3) → Z = 3.0."""
        result = TSResult(value=jnp.array(9.0), test=jnp.array(1.0))
        dist = QMuAsymptotic()
        z = dist.null_significance(result)
        assert float(z) == pytest.approx(3.0, abs=1e-5)

    def test_alt_significance_qmu(self):
        """For q_μ=9, q_asimov=4: p = 1 - Φ(3-2) = 1 - Φ(1) → Z = 1.0."""
        result = TSResult(
            value=jnp.array(9.0), test=jnp.array(1.0), q_asimov=jnp.array(4.0)
        )
        dist = QMuAsymptotic()
        z = dist.alt_significance(result)
        assert float(z) == pytest.approx(1.0, abs=1e-5)

    def test_significance_none_without_asimov(self):
        """Significance returns None when q_asimov is missing."""
        result = TSResult(value=jnp.array(4.0), test=jnp.array(1.0))
        dist = QMuAsymptotic()
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            assert dist.alt_significance(result) is None

    def test_null_significance_qtilde_standard_region(self):
        """For q̃=4, q_asimov=4 (standard): p = 1 - Φ(2) → Z = 2.0."""
        result = TSResult(
            value=jnp.array(4.0), test=jnp.array(1.0), q_asimov=jnp.array(4.0)
        )
        dist = QTildeAsymptotic()
        z = dist.null_significance(result)
        assert float(z) == pytest.approx(2.0, abs=1e-5)

    def test_discovery_significance(self):
        """5σ discovery: q_0=25 → Z=5.0."""
        result = TSResult(value=jnp.array(25.0), test=jnp.array(0.0))
        dist = Q0Asymptotic()
        z = dist.null_significance(result)
        assert float(z) == pytest.approx(5.0, rel=1e-2)
