"""Distributions for test statistics.

This module provides distribution classes that convert test statistic values
to p-values. Each distribution encapsulates the statistical formulas for a
specific test statistic type, separating these assumptions from the calculators.

References:
    Cowan et al., "Asymptotic formulae for likelihood-based tests of new physics"
    Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from everwillow.inference.hypotest._results import ExpectedBands, TestStatResult

__all__ = [
    "Distribution",
    "EmpiricalDistribution",
    "Q0Asymptotic",
    "QMuAsymptotic",
    "QTildeAsymptotic",
    "TMuAsymptotic",
]


class Distribution(eqx.Module):
    """Abstract base for test statistic distributions.

    Distributions receive the full TestStatResult, allowing them to extract
    whatever data they need from the extras dict.

    Subclasses must implement:
        - pvalues: Compute (pnull, palt) from test statistic result
        - expected_pvalues: Compute expected p-values at sigma bands
    """

    @abc.abstractmethod
    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Compute (pnull, palt) from test statistic result.

        Args:
            result: Full TestStatResult with q value and extras dict.

        Returns:
            Tuple of (pnull, palt) arrays.
        """
        ...

    @abc.abstractmethod
    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Compute expected p-values at standard sigma bands.

        Args:
            result: Full TestStatResult (used to extract q_asimov, etc.)

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level.
        """
        ...


# =============================================================================
# Asymptotic Distributions (Cowan et al. formulas)
# =============================================================================


class QTildeAsymptotic(Distribution):
    """Asymptotic distribution for QTilde (upper limits).

    Used with the q̃_μ test statistic for hypothesis testing.
    Expects result.extras to contain 'q_asimov' for expected band computation.

    References:
        Cowan et al., Eq. 59-64 (standard) and Eq. 66-67 (boundary), arXiv:1007.1727
    """

    def _pnull(self, q: Array, q_asimov: Array, nsigma: float = 0.0) -> Array:
        """p-value under null hypothesis (background-only).

        For expected bands (q=q_asimov), this reduces to Φ(nsigma).
        """
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_q_asimov = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        # Standard case: 1 - Φ(√q - √q_A - nsigma)
        pnull = 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - sqrt_q_asimov - nsigma)

        # Boundary case: q > q_asimov (Eq. 66)
        cond = (q > q_asimov) & (q_asimov > 0)
        pnull_boundary = 1.0 - jax.scipy.stats.norm.cdf(
            (q - q_asimov) / (2.0 * sqrt_q_asimov) - nsigma
        )
        return jnp.where(cond, pnull_boundary, pnull)

    def _palt(self, q: Array, q_asimov: Array, nsigma: float = 0.0) -> Array:
        """p-value under alternative hypothesis (signal+background).

        Implements Eq. 66-67 boundary handling from Cowan et al.
        """
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_q_asimov = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        # Standard case (Eq. 64)
        palt = 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - nsigma)

        # Boundary case: q > q_asimov (Eq. 66 second branch)
        cond = (q > q_asimov) & (q_asimov > 0)
        palt_boundary = 1.0 - jax.scipy.stats.norm.cdf(
            (q + q_asimov) / (2.0 * sqrt_q_asimov) - nsigma
        )
        return jnp.where(cond, palt_boundary, palt)

    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Return (pnull, palt)."""
        q = result.q
        q_asimov = result.extras.get("q_asimov", q)
        return self._pnull(q, q_asimov, 0.0), self._palt(q, q_asimov, 0.0)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Expected p-values at sigma bands.

        At q=q_asimov:
        - pnull = Φ(nsigma)
        - palt = 1 - Φ(√q_asimov - nsigma)

        At poi=0 (q_asimov=0), CLs = palt/pnull = 1.0 for all bands.
        """
        q_asimov = result.extras.get("q_asimov", result.q)

        def compute_band(nsigma: float) -> tuple[Array, Array]:
            return (
                self._pnull(q_asimov, q_asimov, nsigma),
                self._palt(q_asimov, q_asimov, nsigma),
            )

        return ExpectedBands(
            minus_2sigma=compute_band(-2.0),
            minus_1sigma=compute_band(-1.0),
            median=compute_band(0.0),
            plus_1sigma=compute_band(1.0),
            plus_2sigma=compute_band(2.0),
        )


class QMuAsymptotic(Distribution):
    """Asymptotic distribution for QMu.

    Used with the q_μ test statistic (no boundary handling).
    Expects result.extras to contain 'q_asimov'.

    References:
        Cowan et al., Eq. 57 arXiv:1007.1727
    """

    def _pnull(self, q: Array, q_asimov: Array, nsigma: float = 0.0) -> Array:
        """p-value under null hypothesis (background-only).

        For expected bands (q=q_asimov), this reduces to Φ(nsigma).
        """
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_q_asimov = jnp.sqrt(jnp.maximum(q_asimov, 0.0))
        return 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - sqrt_q_asimov - nsigma)

    def _palt(self, q: Array, q_asimov: Array, nsigma: float = 0.0) -> Array:
        """p-value under alternative hypothesis (signal+background)."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - nsigma)

    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Return (pnull, palt)."""
        q = result.q
        q_asimov = result.extras.get("q_asimov", q)
        return self._pnull(q, q_asimov, 0.0), self._palt(q, q_asimov, 0.0)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Expected p-values at sigma bands."""
        q_asimov = result.extras.get("q_asimov", result.q)

        def compute_band(nsigma: float) -> tuple[Array, Array]:
            return (
                self._pnull(q_asimov, q_asimov, nsigma),
                self._palt(q_asimov, q_asimov, nsigma),
            )

        return ExpectedBands(
            minus_2sigma=compute_band(-2.0),
            minus_1sigma=compute_band(-1.0),
            median=compute_band(0.0),
            plus_1sigma=compute_band(1.0),
            plus_2sigma=compute_band(2.0),
        )


class Q0Asymptotic(Distribution):
    """Asymptotic distribution for Q0 (discovery).

    Used with the q_0 test statistic for discovery significance.
    Expects result.extras to contain 'q_asimov'.

    References:
        Cowan et al., Eq. 49-52, arXiv:1007.1727
    """

    def _pnull(self, q: Array, nsigma: float = 0.0) -> Array:
        """p-value under null hypothesis (background-only).

        For discovery, this is the main quantity of interest.
        """
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - nsigma)

    def _palt(self, q: Array, q_asimov: Array, nsigma: float = 0.0) -> Array:
        """p-value under alternative hypothesis (signal+background).

        For expected bands (q=q_asimov), this reduces to Φ(nsigma).
        """
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_q_asimov = jnp.sqrt(jnp.maximum(q_asimov, 0.0))
        return 1.0 - jax.scipy.stats.norm.cdf(sqrt_q - sqrt_q_asimov - nsigma)

    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Return (pnull, palt) - raw hypothesis p-values."""
        q = result.q
        q_asimov = result.extras.get("q_asimov", q)
        return self._pnull(q, 0.0), self._palt(q, q_asimov, 0.0)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Expected p-values at sigma bands using unified formulas."""
        q_asimov = result.extras.get("q_asimov", result.q)

        def compute_band(nsigma: float) -> tuple[Array, Array]:
            return (
                self._pnull(q_asimov, nsigma),
                self._palt(q_asimov, q_asimov, nsigma),
            )

        return ExpectedBands(
            minus_2sigma=compute_band(-2.0),
            minus_1sigma=compute_band(-1.0),
            median=compute_band(0.0),
            plus_1sigma=compute_band(1.0),
            plus_2sigma=compute_band(2.0),
        )


class TMuAsymptotic(Distribution):
    """Asymptotic distribution for TMu (two-sided).

    Used with the t_μ signed test statistic for two-sided confidence intervals.
    Expects result.extras to contain 'q_asimov' (which is t_asimov for TMu).

    References:
        Cowan et al., Eq. 33-37, arXiv:1007.1727
    """

    def _pnull(self, t: Array, t_asimov: Array, nsigma: float = 0.0) -> Array:
        """Two-sided p-value under null hypothesis (background-only)."""
        sigma = jnp.sqrt(jnp.maximum(jnp.abs(t_asimov), 1e-10))
        return 2.0 * (1.0 - jax.scipy.stats.norm.cdf(jnp.abs(t) / sigma - nsigma))

    def _palt(self, t: Array, t_asimov: Array, nsigma: float = 0.0) -> Array:
        """Two-sided p-value under alternative hypothesis (signal+background)."""
        sigma = jnp.sqrt(jnp.maximum(jnp.abs(t_asimov), 1e-10))
        shift = jnp.sqrt(jnp.abs(t_asimov))
        return 2.0 * (
            1.0 - jax.scipy.stats.norm.cdf((jnp.abs(t) + shift) / sigma - nsigma)
        )

    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Return (pnull, palt) - raw hypothesis p-values."""
        t = result.q
        t_asimov = result.extras.get("q_asimov", t)
        return self._pnull(t, t_asimov, 0.0), self._palt(t, t_asimov, 0.0)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Expected p-values at sigma bands using unified formulas."""
        t_asimov = result.extras.get("q_asimov", result.q)

        def compute_band(nsigma: float) -> tuple[Array, Array]:
            return (
                self._pnull(t_asimov, t_asimov, nsigma),
                self._palt(t_asimov, t_asimov, nsigma),
            )

        return ExpectedBands(
            minus_2sigma=compute_band(-2.0),
            minus_1sigma=compute_band(-1.0),
            median=compute_band(0.0),
            plus_1sigma=compute_band(1.0),
            plus_2sigma=compute_band(2.0),
        )


# =============================================================================
# Empirical Distribution (from toys)
# =============================================================================


class EmpiricalDistribution(Distribution):
    """Empirical distribution from toy samples.

    Unlike asymptotic distributions, this stores the toy test statistic arrays
    directly and computes p-values empirically.

    Attributes:
        q_alt: Test statistics under alternative (signal+background) hypothesis.
        q_null: Test statistics under null (background-only) hypothesis.
    """

    q_alt: Array
    q_null: Array

    def pvalues(self, result: TestStatResult) -> tuple[Array, Array]:
        """Compute empirical (pnull, palt) from toy distributions."""
        q = result.q
        pnull = jnp.mean(self.q_null >= q)
        palt = jnp.mean(self.q_alt >= q)
        return pnull, palt

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Compute expected p-values at standard sigma bands from toy distributions.

        For each test statistic value in q_null, computes what the p-values would be,
        then takes percentiles at the standard normal quantiles.
        """
        # For each q in q_null, compute p-values
        pnulls = jax.vmap(lambda q: jnp.mean(self.q_null >= q))(self.q_null)
        palts = jax.vmap(lambda q: jnp.mean(self.q_alt >= q))(self.q_null)

        # Normal distribution percentiles at -2σ, -1σ, 0, +1σ, +2σ
        # These are CDF values: Φ(-2), Φ(-1), Φ(0), Φ(1), Φ(2)
        percentiles = jnp.array(
            [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681]
        )

        pnull_bands = jnp.percentile(pnulls, percentiles)
        palt_bands = jnp.percentile(palts, percentiles)

        return ExpectedBands(
            minus_2sigma=(pnull_bands[0], palt_bands[0]),
            minus_1sigma=(pnull_bands[1], palt_bands[1]),
            median=(pnull_bands[2], palt_bands[2]),
            plus_1sigma=(pnull_bands[3], palt_bands[3]),
            plus_2sigma=(pnull_bands[4], palt_bands[4]),
        )
