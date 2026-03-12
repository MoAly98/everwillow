"""Distributions for converting test statistics to p-values.

Provides asymptotic distribution classes (Cowan et al., arXiv:1007.1727)
and empirical distributions from toy Monte Carlo. Each class exposes
``cdf``, ``null_pval``, and ``alt_pval``.
"""

from __future__ import annotations

import abc
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from everwillow.inference.hypotest._results import (
    ExpectedBands,
    TestStatResult,
    ToyResult,
)

__all__ = [
    "Distribution",
    "EmpiricalDistribution",
    "Q0Asymptotic",
    "QMuAsymptotic",
    "QTildeAsymptotic",
    "SimpleEmpiricalDistribution",
    "TMuAsymptotic",
    "TMuTildeAsymptotic",
]

_PHI = jax.scipy.stats.norm.cdf
_PPF = jax.scipy.stats.norm.ppf


def _require_q_asimov(result: TestStatResult, cls_name: str, pval_type: str) -> bool:
    """Check that q_asimov is available, warn if not.

    Returns:
        True if q_asimov is present, False otherwise.
    """
    if result.q_asimov is None:
        warnings.warn(
            f"{pval_type} p-value computation in {cls_name} "
            "cannot be performed without an Asimov test statistic.",
            stacklevel=3,
        )
        return False
    return True


# =============================================================================
# Base Distribution
# =============================================================================


class Distribution(eqx.Module):
    """Abstract base for test statistic distributions.

    Subclasses must implement:
        - ``cdf``: CDF F(q | μ') with explicit σ.
        - ``null_pval``: p-value under null hypothesis (:math:`\\mu'= \\mu` where :math:`\\mu` is the hypothesis being tested).
        - ``alt_pval``: p-value under an alternative hypothesis (:math:`\\mu'=0` for exclusion, :math:`\\mu'=1` for discovery).

    """

    @abc.abstractmethod
    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """CDF F(q | μ') with known σ.

        Args:
            q: Test statistic value.
            mu: POI value being tested.
            mu_prime: Assumed true POI value.
            sigma: Standard deviation of μ̂.

        Returns:
            Cumulative distribution function value.
        """
        ...

    @abc.abstractmethod
    def null_pval(self, result: TestStatResult) -> Array | None:
        """p-value under the null hypothesis (μ' = μ).

        Args:
            result: Test statistic result.

        Returns:
            Null p-value, or None if required data (e.g. q_asimov) is missing.
        """
        ...

    @abc.abstractmethod
    def alt_pval(self, result: TestStatResult) -> Array | None:
        """p-value under an alternative hypothesis.

        Args:
            result: Test statistic result.

        Returns:
            Alternative p-value, or None if required data (e.g. q_asimov) is missing.
        """
        ...

    def null_significance(self, result: TestStatResult) -> Array | None:
        """Significance under the null hypothesis: Z = Φ⁻¹(1 - pnull).

        Args:
            result: Test statistic result.

        Returns:
            Significance Z, or None if pnull is None.
        """
        pnull = self.null_pval(result)
        if pnull is None:
            return None
        return -_PPF(pnull)

    def alt_significance(self, result: TestStatResult) -> Array | None:
        """Significance under the alternative hypothesis: Z = Φ⁻¹(1 - palt).

        Args:
            result: Test statistic result.

        Returns:
            Significance Z, or None if palt is None.
        """
        palt = self.alt_pval(result)
        if palt is None:
            return None
        return -_PPF(palt)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Compute expected p-values at standard sigma bands.

        Args:
            result: Test statistic result.

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level.
        """
        raise NotImplementedError


# =============================================================================
# Asymptotic Distributions (Cowan et al. formulas)
# =============================================================================


class TMuAsymptotic(Distribution):
    """Asymptotic distribution for t_μ (two-sided, Eq. 38).

    Used with the t_μ test statistic for two-sided confidence intervals.

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """F(t_μ | μ') = Φ(√t + (μ-μ')/σ) + Φ(√t - (μ-μ')/σ) - 1."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        delta = (mu - mu_prime) / sigma
        return _PHI(sqrt_q + delta) + _PHI(sqrt_q - delta) - 1.0

    def null_pval(self, result: TestStatResult) -> Array:
        """p = 2(1 - Φ(√t_μ)). No σ needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 2.0 * (1.0 - _PHI(sqrt_q))

    def alt_pval(self, result: TestStatResult) -> Array | None:
        """p = 2 - Φ(√t + √q_A) - Φ(√t - √q_A)."""
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 2.0 - _PHI(sqrt_q + sqrt_qa) - _PHI(sqrt_q - sqrt_qa)


class TMuTildeAsymptotic(Distribution):
    """Asymptotic distribution for t̃_μ (two-sided with physical bound, Eq. 40/44).

    Used with the t̃_μ test statistic for two-sided tests with the physical
    constraint μ ≥ 0. The CDF has a piecewise structure with the Φ+Φ-1
    form in both regions (Eq. 44).

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """F(t̃_μ | μ') — piecewise at threshold μ²/σ²."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        delta = (mu - mu_prime) / sigma
        threshold = (mu / sigma) ** 2

        # Standard region: Φ(√t̃ + δ) + Φ(√t̃ - δ) - 1
        f_standard = _PHI(sqrt_q + delta) + _PHI(sqrt_q - delta) - 1.0

        # Boundary region: Φ(√t̃ + δ) + Φ((t̃ + μ²/σ²)/(2μ/σ) - δ) - 1
        f_boundary = (
            _PHI(sqrt_q + delta)
            + _PHI((q + threshold) / (2.0 * mu / sigma) - delta)
            - 1.0
        )

        return jnp.where(q <= threshold, f_standard, f_boundary)

    def null_pval(self, result: TestStatResult) -> Array | None:
        """Null p-value (μ' = μ).

        Standard: p = 2(1 - Φ(√t̃))
        Boundary: p = 2 - Φ(√t̃) - Φ((t̃ + q_A)/(2√q_A))
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Null"):
            return None
        q = result.value
        q_asimov = result.q_asimov
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        p_standard = 2.0 * (1.0 - _PHI(sqrt_q))
        p_boundary = 2.0 - _PHI(sqrt_q) - _PHI((q + q_asimov) / (2.0 * sqrt_qa))

        return jnp.where(q <= q_asimov, p_standard, p_boundary)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        """Alt p-value (μ' = 0, so (μ-μ')/σ = √q_A).

        Standard: p = 2 - Φ(√t̃ + √q_A) - Φ(√t̃ - √q_A)
        Boundary: p = 2 - Φ(√t̃ + √q_A) - Φ((t̃ - q_A)/(2√q_A))
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        q = result.value
        q_asimov = result.q_asimov
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        p_standard = 2.0 - _PHI(sqrt_q + sqrt_qa) - _PHI(sqrt_q - sqrt_qa)
        p_boundary = (
            2.0 - _PHI(sqrt_q + sqrt_qa) - _PHI((q - q_asimov) / (2.0 * sqrt_qa))
        )

        return jnp.where(q <= q_asimov, p_standard, p_boundary)


class Q0Asymptotic(Distribution):
    """Asymptotic distribution for q_0 (discovery, Eq. 49).

    Used with the q_0 test statistic for discovery significance.
    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """F(q_0 | μ') = Φ(√q_0 - μ'/σ)."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return _PHI(sqrt_q - mu_prime / sigma)

    def null_pval(self, result: TestStatResult) -> Array:
        """p = 1 - Φ(√q_0). No σ needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 1.0 - _PHI(sqrt_q)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        """p = 1 - Φ(√q_0 - √q_A)."""
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 1.0 - _PHI(sqrt_q - sqrt_qa)


class QMuAsymptotic(Distribution):
    """Asymptotic distribution for q_μ (upper limit, Eq. 57).

    Used with the q_μ test statistic (no boundary handling).

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """F(q_μ | μ') = Φ(√q_μ - (μ - μ')/σ)."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return _PHI(sqrt_q - (mu - mu_prime) / sigma)

    def null_pval(self, result: TestStatResult) -> Array:
        """p = 1 - Φ(√q_μ). No σ needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 1.0 - _PHI(sqrt_q)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        """p = 1 - Φ(√q_μ - √q_A)."""
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 1.0 - _PHI(sqrt_q - sqrt_qa)


class QTildeAsymptotic(Distribution):
    """Asymptotic distribution for q̃_μ (upper limit with physical bound, Eq. 64).

    Used with the q̃_μ test statistic for hypothesis testing with the
    physical constraint μ ≥ 0. The CDF is piecewise at q̃ = μ²/σ² = q_asimov.

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """F(q̃_μ | μ') — piecewise at threshold μ²/σ²."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        threshold = (mu / sigma) ** 2

        # Standard region: Φ(√q̃ - (μ-μ')/σ)
        f_standard = _PHI(sqrt_q - (mu - mu_prime) / sigma)

        # Boundary region: Φ((q̃ - (μ²-2μμ')/σ²) / (2μ/σ))
        f_boundary = _PHI(
            (q - (mu**2 - 2 * mu * mu_prime) / sigma**2) / (2.0 * mu / sigma)
        )

        return jnp.where(q <= threshold, f_standard, f_boundary)

    def null_pval(self, result: TestStatResult) -> Array | None:
        """Null p-value (μ' = μ).

        q̃ = 0: p = 1
        Standard (0 < q̃ ≤ q_A): p = 1 - Φ(√q̃)
        Boundary (q̃ > q_A): p = 1 - Φ((q̃ + q_A)/(2√q_A))
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Null"):
            return None
        q = result.value
        q_asimov = result.q_asimov
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        p_standard = 1.0 - _PHI(sqrt_q)
        p_boundary = 1.0 - _PHI((q + q_asimov) / (2.0 * sqrt_qa))

        return jnp.where(q <= q_asimov, p_standard, p_boundary)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        """Alt p-value (μ' = 0, so (μ-μ')/σ = √q_A).

        q̃ = 0: p = 1
        Standard (0 < q̃ ≤ q_A): p = 1 - Φ(√q̃ - √q_A)
        Boundary (q̃ > q_A): p = 1 - Φ((q̃ - q_A)/(2√q_A))
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        q = result.value
        q_asimov = result.q_asimov
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(q_asimov, 0.0))

        p_standard = 1.0 - _PHI(sqrt_q - sqrt_qa)
        p_boundary = 1.0 - _PHI((q - q_asimov) / (2.0 * sqrt_qa))

        return jnp.where(q <= q_asimov, p_standard, p_boundary)


# =============================================================================
# Empirical Distribution (from toys)
# =============================================================================


class EmpiricalDistribution(Distribution):
    """Base class for distributions built from toy test statistics.

    Stores the raw test statistic arrays from toy generation and provides
    the ``from_toys`` factory method. Subclass this and override
    ``null_pval`` / ``alt_pval`` to implement custom p-value computation
    methods (e.g. KDE smoothing, tail extrapolation).

    Attributes:
        q_alt: Test statistics under alternative (signal+background) hypothesis.
        q_null: Test statistics under null (background-only) hypothesis.
    """

    q_alt: Array
    q_null: Array

    @classmethod
    def from_toys(cls, toys: ToyResult) -> EmpiricalDistribution:
        """Construct from a ToyResult.

        Args:
            toys: Raw toy generation output containing q_alt and q_null arrays.

        Returns:
            An instance of this distribution class.
        """
        return cls(q_alt=toys.q_alt, q_null=toys.q_null)

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        """Not applicable for empirical distributions."""
        raise NotImplementedError("Empirical distributions do not have an analytic CDF")


class SimpleEmpiricalDistribution(EmpiricalDistribution):
    """Empirical p-values via simple tail counting.

    ``pnull = fraction of q_null >= q_obs``
    ``palt  = fraction of q_alt  >= q_obs``
    """

    def null_pval(self, result: TestStatResult) -> Array:
        """Empirical p-value under null: fraction of q_null >= q_obs."""
        return jnp.mean(self.q_null >= result.value)

    def alt_pval(self, result: TestStatResult) -> Array:
        """Empirical p-value under alternative: fraction of q_alt >= q_obs."""
        return jnp.mean(self.q_alt >= result.value)
