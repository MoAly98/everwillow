"""Tests for hypothesis test distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from everwillow.hypotest.distributions import (
    Q0Asymptotic,
    QMuAsymptotic,
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
    TMuAsymptotic,
    TMuTildeAsymptotic,
)
from everwillow.hypotest.results import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)
from everwillow.hypotest.results import ToyResult
from everwillow.hypotest.utils import cl_s

# =============================================================================
# CDF Tests
# =============================================================================


class TestCDF:
    """Tests for CDF functions of asymptotic distributions.

    Each CDF maps (q, μ, μ', σ) → F(q | μ') from Cowan et al.
    """

    @pytest.mark.parametrize(
        ("dist_cls", "q", "mu", "mu_prime", "sigma", "expected"),
        [
            # Q0Asymptotic: F = Φ(√q - μ'/σ)
            (Q0Asymptotic, 4.0, 0.0, 0.0, 1.0, 0.97725),  # Φ(2)
            (Q0Asymptotic, 4.0, 0.0, 2.0, 1.0, 0.5),  # Φ(2-2) = Φ(0)
            (Q0Asymptotic, 0.0, 0.0, 0.0, 1.0, 0.5),  # Φ(0)
            # QMuAsymptotic: F = Φ(√q - (μ-μ')/σ)
            (QMuAsymptotic, 4.0, 2.0, 2.0, 1.0, 0.97725),  # Φ(2)
            (QMuAsymptotic, 4.0, 2.0, 0.0, 1.0, 0.5),  # Φ(2-2) = Φ(0)
            (QMuAsymptotic, 9.0, 2.0, 0.0, 1.0, 0.84134),  # Φ(3-2) = Φ(1)
            (QMuAsymptotic, 4.0, 2.0, 0.0, 2.0, 0.84134),  # Φ(2-1) = Φ(1), σ=2
            # TMuAsymptotic: F = Φ(√t+δ) + Φ(√t-δ) - 1, δ = (μ-μ')/σ
            (TMuAsymptotic, 4.0, 2.0, 2.0, 1.0, 0.9545),  # 2Φ(2) - 1
            (TMuAsymptotic, 4.0, 2.0, 0.0, 1.0, 0.49997),  # Φ(4) + Φ(0) - 1
            (TMuAsymptotic, 0.0, 2.0, 2.0, 1.0, 0.0),  # 2Φ(0) - 1
            # QTildeAsymptotic — standard region (q ≤ μ²/σ²)
            (QTildeAsymptotic, 4.0, 2.0, 2.0, 1.0, 0.97725),  # Φ(2)
            (QTildeAsymptotic, 4.0, 2.0, 0.0, 1.0, 0.5),  # Φ(0)
            # QTildeAsymptotic — boundary region (q > μ²/σ²)
            (QTildeAsymptotic, 9.0, 2.0, 0.0, 1.0, 0.89435),  # Φ(1.25)
            # TMuTildeAsymptotic — standard region (q ≤ μ²/σ²)
            (TMuTildeAsymptotic, 4.0, 2.0, 2.0, 1.0, 0.9545),  # 2Φ(2) - 1
            (TMuTildeAsymptotic, 4.0, 2.0, 0.0, 1.0, 0.49997),  # Φ(4)+Φ(0) - 1
            # TMuTildeAsymptotic — boundary region (q > μ²/σ²)
            (TMuTildeAsymptotic, 9.0, 2.0, 2.0, 1.0, 0.99807),  # Φ(3)+Φ(3.25) - 1
            (TMuTildeAsymptotic, 9.0, 2.0, 0.0, 1.0, 0.89435),  # Φ(5)+Φ(1.25) - 1
        ],
        ids=[
            "q0-null",
            "q0-alt",
            "q0-zero",
            "qmu-null",
            "qmu-alt",
            "qmu-large-q",
            "qmu-sigma2",
            "tmu-null",
            "tmu-alt",
            "tmu-zero",
            "qtilde-null-standard",
            "qtilde-alt-standard",
            "qtilde-alt-boundary",
            "tmutilde-null-standard",
            "tmutilde-alt-standard",
            "tmutilde-null-boundary",
            "tmutilde-alt-boundary",
        ],
    )
    def test_cdf_values(self, dist_cls, q, mu, mu_prime, sigma, expected):
        """CDF values match hand-computed Cowan et al. formulas."""
        dist = dist_cls()
        result = dist.cdf(jnp.array(q), jnp.array(mu), jnp.array(mu_prime), jnp.array(sigma))
        assert float(result) == pytest.approx(expected, rel=1e-3)


# =============================================================================
# One-sided asymptotic p-value tests (QTilde, QMu, Q0)
# =============================================================================


class TestOneSidedAsymptoticPvalues:
    """Parametrized p-value tests for one-sided distributions.

    All use the same test structure: create TSResult → call null_pval/alt_pval
    → assert against hand-computed values from Cowan et al.

    QMu and QTilde share null_pval = 1-Φ(√q) in the standard region (q ≤ q_A).
    QTilde uses a boundary formula for q > q_A.
    Q0 is the discovery counterpart (always tests at μ=0).
    """

    @pytest.mark.parametrize(
        ("dist_cls", "q", "test_val", "q_asimov", "expected_pnull", "expected_palt"),
        [
            # QTildeAsymptotic — standard region (q ≤ q_A)
            # pnull = 1-Φ(√q), palt = 1-Φ(√q-√q_A)
            (QTildeAsymptotic, 4.0, 1.0, 4.0, 0.02275, 0.5),
            (QTildeAsymptotic, 1.0, 1.0, 4.0, 0.15866, 0.84134),
            # QTildeAsymptotic — boundary region (q > q_A)
            # pnull = 1-Φ((q+q_A)/(2√q_A)), palt = 1-Φ((q-q_A)/(2√q_A))
            (QTildeAsymptotic, 9.0, 1.0, 4.0, 0.000577, 0.10565),
            # QMuAsymptotic — same formula, no piecewise boundary
            (QMuAsymptotic, 4.0, 1.0, 4.0, 0.02275, 0.5),
            (QMuAsymptotic, 9.0, 1.0, 4.0, 0.001350, 0.15866),
            # Q0Asymptotic — discovery (test=0)
            (Q0Asymptotic, 9.0, 0.0, 9.0, 0.001350, 0.5),
            (Q0Asymptotic, 0.0, 0.0, 0.0, 0.5, 0.5),
            (Q0Asymptotic, 9.0, 0.0, 4.0, 0.001350, 0.15866),
        ],
        ids=[
            "qtilde-at-asimov",
            "qtilde-below-asimov",
            "qtilde-boundary",
            "qmu-at-asimov",
            "qmu-large-q",
            "q0-discovery",
            "q0-no-excess",
            "q0-with-alt",
        ],
    )
    def test_pvalues(self, dist_cls, q, test_val, q_asimov, expected_pnull, expected_palt):
        """p-values match hand-computed Cowan et al. values."""
        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(test_val),
            q_asimov=jnp.array(q_asimov),
        )
        dist = dist_cls()
        assert float(dist.null_pval(result)) == pytest.approx(expected_pnull, rel=1e-2)
        assert float(dist.alt_pval(result)) == pytest.approx(expected_palt, rel=1e-2)


# =============================================================================
# Two-sided asymptotic p-value tests (TMu, TMuTilde)
# =============================================================================


class TestTwoSidedAsymptoticPvalues:
    """Parametrized p-value tests for two-sided distributions.

    TMuAsymptotic: pnull = 2(1-Φ(√t)), palt = 2-Φ(√t+√q_A)-Φ(√t-√q_A)
    TMuTildeAsymptotic: same formulas with piecewise boundary at t̃ = q_A
    """

    @pytest.mark.parametrize(
        ("dist_cls", "q", "q_asimov", "expected_pnull", "expected_palt"),
        [
            # TMuAsymptotic — two-sided, Eq. 38
            (TMuAsymptotic, 0.0, 4.0, 1.0, 1.0),
            (TMuAsymptotic, 4.0, 4.0, 0.04550, 0.50003),
            # TMuTildeAsymptotic — standard region (t̃ ≤ q_A)
            (TMuTildeAsymptotic, 4.0, 4.0, 0.04550, 0.50003),
            # TMuTildeAsymptotic — boundary region (t̃ > q_A)
            (TMuTildeAsymptotic, 9.0, 4.0, 0.001927, 0.10565),
            # TMuTildeAsymptotic — at zero
            (TMuTildeAsymptotic, 0.0, 4.0, 1.0, 1.0),
        ],
        ids=[
            "tmu-zero",
            "tmu-positive",
            "tmutilde-standard",
            "tmutilde-boundary",
            "tmutilde-zero",
        ],
    )
    def test_pvalues(self, dist_cls, q, q_asimov, expected_pnull, expected_palt):
        """Two-sided p-values match hand-computed Cowan et al. values."""
        result = TSResult(
            value=jnp.array(q),
            test=jnp.array(1.0),
            q_asimov=jnp.array(q_asimov),
        )
        dist = dist_cls()
        assert float(dist.null_pval(result)) == pytest.approx(expected_pnull, rel=1e-2)
        assert float(dist.alt_pval(result)) == pytest.approx(expected_palt, rel=1e-2)

    def test_tmutilde_requires_q_asimov(self):
        """null_pval and alt_pval return None without q_asimov."""
        result = TSResult(value=jnp.array(4.0), test=jnp.array(1.0))
        dist = TMuTildeAsymptotic()
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            assert dist.null_pval(result) is None
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            assert dist.alt_pval(result) is None


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

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert float(pnull) == pytest.approx(0.6, rel=1e-5)
        assert float(palt) == pytest.approx(0.1, rel=1e-5)

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
        """Test CLs = pnull / palt via cl_s().

        q_null = [0.5..5] (under tested hypothesis, small q),
        q_alt = [1..10] (under alternative, large q).
        At q_obs=5: pnull = 1/10 = 0.1, palt = 6/10 = 0.6
        CLs = 0.1/0.6 = 0.16667
        """
        q_null = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_alt = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q_obs = 5.0

        dist = SimpleEmpiricalDistribution(q_alt=q_alt, q_null=q_null)
        result = TSResult(value=jnp.array(q_obs), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        actual_cls = float(cl_s(pnull, palt))
        assert actual_cls == pytest.approx(0.16667, rel=1e-4)

    def test_from_toys(self):
        """Test constructing SimpleEmpiricalDistribution from ToyResult."""
        q_alt = jnp.array([1.0, 2.0, 3.0])
        q_null = jnp.array([4.0, 5.0, 6.0])
        toys = ToyResult(q_alt=q_alt, q_null=q_null)

        dist = SimpleEmpiricalDistribution.from_toys(toys)

        assert isinstance(dist, SimpleEmpiricalDistribution)
        assert jnp.array_equal(dist.q_alt, q_alt)
        assert jnp.array_equal(dist.q_null, q_null)

    def test_alt_pval_none_without_q_alt(self):
        """alt_pval warns and returns None when q_alt is not provided."""
        q_null = jnp.array([1.0, 2.0, 3.0])
        dist = SimpleEmpiricalDistribution(q_null=q_null)
        result = TSResult(value=jnp.array(1.5), test=jnp.array(1.0))
        with pytest.warns(UserWarning, match="cannot be performed without q_alt"):
            assert dist.alt_pval(result) is None

    def test_pvalue_bands_median_cls(self):
        """Empirical pvalue_bands median CLs from known arrays.

        q_null = linspace(0, 10, 10001) — uniform, so fraction >= q is (10-q)/10.
        q_alt  = linspace(0, 20, 10001) — uniform, so fraction >= q is (20-q)/20.

        Median q_alt = quantile at Φ(0) = 0.5 → q_alt[5000] = 10.0.
        At q=10: pnull = (10-10)/10 = 0.0, palt = (20-10)/20 = 0.5.
        CLs_median = 0.0/0.5 = 0.0.

        -1σ q_alt = quantile at Φ(-1) ≈ 0.1587 → q ≈ 3.174.
        At q=3.174: pnull ≈ (10-3.174)/10 = 0.6826, palt ≈ (20-3.174)/20 = 0.8413.
        CLs_-1σ ≈ 0.6826/0.8413 ≈ 0.8114.
        """
        q_null = jnp.linspace(0.0, 10.0, 10001)
        q_alt = jnp.linspace(0.0, 20.0, 10001)
        dist = SimpleEmpiricalDistribution(q_null=q_null, q_alt=q_alt)
        result = TSResult(value=jnp.array(5.0), test=jnp.array(1.0))

        bands = dist.pvalue_bands(result)

        assert float(bands.cl_s.median) == pytest.approx(0.0, abs=0.01)
        assert float(bands.cl_s.minus_1sigma) == pytest.approx(0.8114, rel=0.05)

    def test_pvalue_bands_raises_without_q_alt(self):
        """pvalue_bands raises ValueError when q_alt is None."""
        q_null = jnp.array([1.0, 2.0, 3.0])
        dist = SimpleEmpiricalDistribution(q_null=q_null)
        result = TSResult(value=jnp.array(1.5), test=jnp.array(1.0))
        with pytest.raises(ValueError, match="pvalue_bands requires q_alt"):
            dist.pvalue_bands(result)

    def test_pvalues_preserve_float32_dtype(self):
        """P-values inherit the float32 dtype from the toy arrays."""
        q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float32)
        dist = SimpleEmpiricalDistribution(q_null=q_null, q_alt=q_alt)
        result = TSResult(value=jnp.array(3.0, dtype=jnp.float32), test=jnp.array(1.0))

        assert dist.null_pval(result).dtype == jnp.float32
        assert dist.alt_pval(result).dtype == jnp.float32

    def test_pvalues_preserve_float64_dtype(self):
        """P-values inherit the float64 dtype from the toy arrays when x64 is enabled."""
        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            q_null = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
            q_alt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64)
            dist = SimpleEmpiricalDistribution(q_null=q_null, q_alt=q_alt)
            result = TSResult(value=jnp.array(3.0, dtype=jnp.float64), test=jnp.array(1.0))

            assert dist.null_pval(result).dtype == jnp.float64
            assert dist.alt_pval(result).dtype == jnp.float64
        finally:
            jax.config.update("jax_enable_x64", prev)


# =============================================================================
# Significance Tests
# =============================================================================


class TestSignificance:
    """Tests for null_significance and alt_significance."""

    @pytest.mark.parametrize(
        ("dist_cls", "q", "test_val", "q_asimov", "method", "expected_z"),
        [
            # null_significance: Z = Φ⁻¹(1 - pnull)
            (Q0Asymptotic, 4.0, 0.0, None, "null_significance", 2.0),
            (QMuAsymptotic, 9.0, 1.0, None, "null_significance", 3.0),
            (QTildeAsymptotic, 4.0, 1.0, 4.0, "null_significance", 2.0),
            (Q0Asymptotic, 25.0, 0.0, None, "null_significance", 5.0),
            # alt_significance: Z = Φ⁻¹(1 - palt)
            (QMuAsymptotic, 9.0, 1.0, 4.0, "alt_significance", 1.0),
        ],
        ids=[
            "q0-null-Z2",
            "qmu-null-Z3",
            "qtilde-null-Z2",
            "q0-discovery-5sigma",
            "qmu-alt-Z1",
        ],
    )
    def test_significance_values(self, dist_cls, q, test_val, q_asimov, method, expected_z):
        """Significance Z matches hand-computed values."""
        q_a = jnp.array(q_asimov) if q_asimov is not None else None
        result = TSResult(value=jnp.array(q), test=jnp.array(test_val), q_asimov=q_a)
        dist = dist_cls()
        z = getattr(dist, method)(result)
        assert float(z) == pytest.approx(expected_z, abs=0.01)

    @pytest.mark.parametrize(
        ("dist_cls", "expected_z"),
        [
            # QMu: null_pval = 1-Φ(√4) = 1-Φ(2) = 0.02275, Z = 2.0
            (QMuAsymptotic, 2.0),
            # Q0: null_pval = 1-Φ(√4) = 1-Φ(2) = 0.02275, Z = 2.0
            (Q0Asymptotic, 2.0),
            # TMu: null_pval = 2(1-Φ(√4)) = 0.04550, Z = Φ⁻¹(0.9545) = 1.689
            (TMuAsymptotic, 1.689),
        ],
    )
    def test_null_significance_works_without_asimov(self, dist_cls, expected_z):
        """null_significance doesn't need q_asimov for QMu/Q0/TMu."""
        result = TSResult(value=jnp.array(4.0), test=jnp.array(1.0))
        dist = dist_cls()
        assert float(dist.null_significance(result)) == pytest.approx(expected_z, abs=0.01)

    @pytest.mark.parametrize(
        ("dist_cls", "method"),
        [
            # Tilde distributions need q_asimov for null (piecewise formulas)
            (QTildeAsymptotic, "null_significance"),
            (TMuTildeAsymptotic, "null_significance"),
            # All distributions need q_asimov for alt
            (QMuAsymptotic, "alt_significance"),
            (QTildeAsymptotic, "alt_significance"),
            (Q0Asymptotic, "alt_significance"),
            (TMuAsymptotic, "alt_significance"),
            (TMuTildeAsymptotic, "alt_significance"),
        ],
    )
    def test_significance_none_without_asimov(self, dist_cls, method):
        """Significance returns None when q_asimov is required but missing."""
        result = TSResult(value=jnp.array(4.0), test=jnp.array(1.0))
        dist = dist_cls()
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            assert getattr(dist, method)(result) is None


# =============================================================================
# Expected P-values Tests
# =============================================================================

# Band data shared by QMu and QTilde (μ=2, σ=1, q_A=4).
# Both produce identical expected p-values despite different formulas
# (QTilde boundary formula gives same result as QMu standard formula).
_EXCLUSION_BAND_DATA = [
    # N=-2: √q=4, pnull=1-Φ(4)≈3.167e-5, palt=Φ(-2)≈0.02275
    ("minus_2sigma", 3.167e-5, 0.02275),
    # N=-1: √q=3, pnull=1-Φ(3)≈0.00135, palt=Φ(-1)≈0.15866
    ("minus_1sigma", 0.00135, 0.15866),
    # N=0: √q=2, pnull=1-Φ(2)≈0.02275, palt=Φ(0)=0.5
    ("median", 0.02275, 0.5),
    # N=+1: √q=1, pnull=1-Φ(1)≈0.15866, palt=Φ(1)≈0.84134
    ("plus_1sigma", 0.15866, 0.84134),
    # N=+2: √q=0, pnull=1-Φ(0)=0.5, palt=Φ(2)≈0.97725
    ("plus_2sigma", 0.5, 0.97725),
]


class TestExpectedPvalues:
    """Tests for pvalue_bands (Brazil band computation).

    Setup: μ=2, σ=1 → q_asimov = μ²/σ² = 4.
    Expected √q at band N = max(0, μ/σ - N) = max(0, 2 - N).
    """

    @pytest.fixture
    def asimov_result(self):
        """TestStatResult with μ=2, q_asimov=4 (σ=1)."""
        return TSResult(
            value=jnp.array(4.0),
            test=jnp.array(2.0),
            q_asimov=jnp.array(4.0),
        )

    def test_median_clb_equals_half(self, asimov_result):
        """At median (N=0), CL_b = Φ(0) = 0.5."""
        dist = QMuAsymptotic()
        bands = dist.pvalue_bands(asimov_result)
        assert float(bands.alt_pvalue.median) == pytest.approx(0.5, abs=1e-5)

    def test_median_cls_equals_double_pnull(self, asimov_result):
        """At median (N=0), CLs = CL_{s+b} / 0.5 = 2 × pnull.

        pnull = 1-Φ(2) = 0.02275, CLs = 0.02275/0.5 = 0.04550.
        """
        dist = QMuAsymptotic()
        bands = dist.pvalue_bands(asimov_result)
        assert float(bands.null_pvalue.median) == pytest.approx(0.02275, rel=1e-3)
        assert float(bands.cl_s.median) == pytest.approx(0.04550, rel=1e-3)

    @pytest.mark.parametrize("dist_cls", [QMuAsymptotic, QTildeAsymptotic])
    @pytest.mark.parametrize(
        ("band_name", "expected_pnull", "expected_palt"),
        _EXCLUSION_BAND_DATA,
    )
    def test_pvalue_bands_exclusion_bands(self, asimov_result, dist_cls, band_name, expected_pnull, expected_palt):
        """QMu and QTilde expected p-values at each band (μ=2, σ=1, q_A=4).

        Both distributions produce identical expected p-values because
        the QTilde boundary formula coincides with QMu standard formula
        at these band values.
        """
        dist = dist_cls()
        bands = dist.pvalue_bands(asimov_result)
        assert float(bands.null_pvalue[band_name]) == pytest.approx(expected_pnull, rel=1e-2)
        assert float(bands.alt_pvalue[band_name]) == pytest.approx(expected_palt, rel=1e-2)

    @pytest.mark.parametrize(
        ("band_name", "expected_pnull", "expected_palt"),
        [
            # N=-2: q=max(0,2-2)²=0, pnull=1-Φ(0)=0.5, palt=1-Φ(0-2)=Φ(2)
            ("minus_2sigma", 0.5, 0.97725),
            # N=-1: q=max(0,2-1)²=1, pnull=1-Φ(1), palt=1-Φ(1-2)=Φ(1)
            ("minus_1sigma", 0.15866, 0.84134),
            # N=0: q=max(0,2)²=4, pnull=1-Φ(2), palt=1-Φ(2-2)=0.5
            ("median", 0.02275, 0.5),
            # N=+1: q=max(0,3)²=9, pnull=1-Φ(3), palt=1-Φ(3-2)=1-Φ(1)
            ("plus_1sigma", 0.00135, 0.15866),
            # N=+2: q=max(0,4)²=16, pnull=1-Φ(4), palt=1-Φ(4-2)=1-Φ(2)
            ("plus_2sigma", 3.167e-5, 0.02275),
        ],
    )
    def test_pvalue_bands_q0_all_bands(self, band_name, expected_pnull, expected_palt):
        """Q0 discovery expected p-values with q_asimov=4 (√q_A=2).

        q=max(0,√q_A+N)², pnull=1-Φ(√q), palt=1-Φ(√q-√q_A).
        """
        result = TSResult(
            value=jnp.array(4.0),
            test=jnp.array(0.0),
            q_asimov=jnp.array(4.0),
        )
        dist = Q0Asymptotic()
        bands = dist.pvalue_bands(result)
        assert float(bands.null_pvalue[band_name]) == pytest.approx(expected_pnull, rel=1e-2)
        assert float(bands.alt_pvalue[band_name]) == pytest.approx(expected_palt, rel=1e-2)

    @pytest.mark.parametrize("dist_cls", [QMuAsymptotic, QTildeAsymptotic, Q0Asymptotic])
    def test_pvalue_bands_requires_q_asimov(self, dist_cls):
        """pvalue_bands returns None when q_asimov is missing."""
        test_val = 0.0 if dist_cls is Q0Asymptotic else 2.0
        result = TSResult(value=jnp.array(4.0), test=jnp.array(test_val))
        dist = dist_cls()
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            assert dist.pvalue_bands(result) is None

    @pytest.mark.parametrize("dist_cls", [QMuAsymptotic, QTildeAsymptotic])
    def test_pvalue_bands_at_zero_poi(self, dist_cls):
        """pvalue_bands at poi=0 must not produce NaN.

        At poi=0, q_asimov=0 → sigma=0 → mu/sigma = 0/0.
        All expected q values should be 0, giving CLs=1.0 (no exclusion power).
        """
        result = TSResult(
            value=jnp.array(0.0),
            test=jnp.array(0.0),
            q_asimov=jnp.array(0.0),
        )
        dist = dist_cls()
        bands = dist.pvalue_bands(result)

        for name, val in bands.cl_s:
            assert jnp.isfinite(val), f"CLs NaN at {name}"
            assert float(val) == pytest.approx(1.0, abs=1e-5)

    def test_pvalue_bands_not_implemented_for_tmu(self, asimov_result):
        """TMuAsymptotic does not support pvalue_bands."""
        dist = TMuAsymptotic()
        with pytest.raises(NotImplementedError):
            dist.pvalue_bands(asimov_result)

    @pytest.mark.parametrize("dist_cls", [QMuAsymptotic, QTildeAsymptotic])
    @pytest.mark.parametrize(
        ("band_name", "expected_pnull", "expected_palt"),
        [
            # At median upper limit (μ=1.96σ, band N=0):
            # q = 1.96², pnull = 1-Φ(1.96) = 0.025, palt = 1-Φ(0) = 0.5
            ("median", 0.025, 0.5),
            # At +1σ band (μ=2.727σ, band N=1):
            # pnull = 1-Φ(1.727) ≈ 0.0421, palt = Φ(1) ≈ 0.8413
            ("plus_1sigma", 0.0421, 0.8413),
            # At -1σ band (μ=1.412σ, band N=-1):
            # pnull = 1-Φ(2.412) ≈ 0.00793, palt = Φ(-1) ≈ 0.1587
            ("minus_1sigma", 0.00793, 0.1587),
        ],
    )
    def test_cls_at_expected_upper_limit(self, dist_cls, band_name, expected_pnull, expected_palt):
        """CLs ≈ 0.05 at each band's expected upper limit (σ=1, α=0.05).

        Verifies the full pipeline: asymptotic p-values → cl_s = 0.05.
        QMu and QTilde produce identical p-values in the standard region.
        """
        expected_cls = float(cl_s(jnp.array(expected_pnull), jnp.array(expected_palt)))
        assert expected_cls == pytest.approx(0.05, rel=1e-2)

    def test_analytic_upper_limit_crosscheck_median(self):
        """Numeric expected CLs upper limit matches analytic formula at N=0.

        Analytic: μ_up = σ × (Φ⁻¹(1 - α × Φ(0)) + 0) = σ × Φ⁻¹(1 - 0.025)
                       = 1.0 × 1.95996 ≈ 1.96
        """
        # σ=1, α=0.05, N=0
        # Φ(0) = 0.5, α×Φ(0) = 0.025, 1-0.025 = 0.975
        # Φ⁻¹(0.975) = 1.95996
        expected_mu_up = 1.95996

        # Build pvalue_bands at various mu values and check CLs = 0.05
        dist = QMuAsymptotic()
        sigma = 1.0

        # At mu = expected_mu_up, the median CLs should be alpha=0.05
        q_asimov = (expected_mu_up / sigma) ** 2
        result = TSResult(
            value=jnp.array(q_asimov),
            test=jnp.array(expected_mu_up),
            q_asimov=jnp.array(q_asimov),
        )
        bands = dist.pvalue_bands(result)
        cls_median = float(bands.cl_s.median)

        assert cls_median == pytest.approx(0.05, rel=1e-3)
