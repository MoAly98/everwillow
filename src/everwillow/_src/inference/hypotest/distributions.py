"""Distributions for converting test statistics to p-values.

Provides asymptotic distribution classes (Cowan et al., arXiv:1007.1727)
and empirical distributions from toy Monte Carlo. Each class exposes
``cdf``, ``null_pval``, and ``alt_pval``.
"""

from __future__ import annotations

import abc
import typing as tp
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from everwillow._src.inference.hypotest.results import (
    BandValues,
    ExpectedBands,
    TestStatResult,
    ToyResult,
)
from everwillow._src.inference.hypotest.utils import (
    cl_s,
    sigma_from_asimov,
    significance,
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

_BAND_SIGMAS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _build_expected_bands(
    dist: Distribution,
    result: TestStatResult,
    expected_q_fn: tp.Callable[[float], Array],
) -> ExpectedBands:
    """Build ExpectedBands by evaluating p-values at each sigma fluctuation.

    Eagerly computes all derived quantities (CLs, significance) so that
    the returned ExpectedBands contains fully populated BandValues.

    Args:
        dist: Distribution whose null_pval/alt_pval will be called.
        result: Original result (used as template for test and q_asimov).
        expected_q_fn: Maps band index N to the expected test statistic value.

    Returns:
        ExpectedBands with BandValues for null_pvalue, alt_pvalue, cl_s,
        null_sig, and alt_sig.
    """
    pnulls = []
    palts = []
    for n in _BAND_SIGMAS:
        synthetic = TestStatResult(
            value=expected_q_fn(n), test=result.test, q_asimov=result.q_asimov
        )
        pnulls.append(dist.null_pval(synthetic))
        palts.append(dist.alt_pval(synthetic))

    null_pvalue = BandValues(*pnulls)
    alt_pvalue = BandValues(*palts)
    cls_values = BandValues(
        **{
            n: cl_s(pn, pa)
            for (n, pn), (_, pa) in zip(null_pvalue, alt_pvalue, strict=False)
        }
    )
    null_sig = BandValues(**{n: significance(pn) for n, pn in null_pvalue})
    alt_sig = BandValues(**{n: significance(pa) for n, pa in alt_pvalue})

    return ExpectedBands(
        null_pvalue=null_pvalue,
        alt_pvalue=alt_pvalue,
        cl_s=cls_values,
        null_sig=null_sig,
        alt_sig=alt_sig,
    )


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
        - ``null_pval``: p-value under null hypothesis (:math:`\\mu'= \\mu` where :math:`\\mu` is the hypothesis being tested).
        - ``alt_pval``: p-value under an alternative hypothesis (:math:`\\mu'=0` for exclusion, :math:`\\mu'=1` for discovery).

    """

    @abc.abstractmethod
    def null_pval(self, result: TestStatResult) -> Array | None:
        r"""p-value under the null hypothesis (:math:`\mu' = \mu`).

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
        r"""Significance under the null hypothesis: :math:`Z = \Phi^{-1}(1 - p_\text{null})`.

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
        r"""Significance under the alternative hypothesis: :math:`Z = \Phi^{-1}(1 - p_\text{alt})`.

        Args:
            result: Test statistic result.

        Returns:
            Significance Z, or None if palt is None.
        """
        palt = self.alt_pval(result)
        if palt is None:
            return None
        return -_PPF(palt)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands | None:
        """Compute expected p-values at standard sigma bands.

        Args:
            result: Test statistic result.

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level.

        Raises:
            NotImplementedError: If the distribution does not support
                expected p-value computation.
        """
        raise NotImplementedError


# =============================================================================
# Asymptotic Distributions (Cowan et al. formulas)
# =============================================================================


class TMuAsymptotic(Distribution):
    r"""Asymptotic distribution for :math:`t_\mu` (two-sided, Eq. 38).

    Used with the :math:`t_\mu` test statistic for two-sided confidence intervals.

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        r"""CDF: :math:`F(t_\mu \mid \mu') = \Phi(\sqrt{t} + \frac{\mu-\mu'}{\sigma}) + \Phi(\sqrt{t} - \frac{\mu-\mu'}{\sigma}) - 1`."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        delta = (mu - mu_prime) / sigma
        return _PHI(sqrt_q + delta) + _PHI(sqrt_q - delta) - 1.0

    def null_pval(self, result: TestStatResult) -> Array:
        r"""Null p-value: :math:`p = 2(1 - \Phi(\sqrt{t_\mu}))`. No :math:`\sigma` needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 2.0 * (1.0 - _PHI(sqrt_q))

    def alt_pval(self, result: TestStatResult) -> Array | None:
        r"""Alt p-value: :math:`p = 2 - \Phi(\sqrt{t} + \sqrt{q_A}) - \Phi(\sqrt{t} - \sqrt{q_A})`.

        :math:`q_A = \mu^2/\sigma^2` (Asimov under :math:`\mu'=0`),
        so :math:`\sqrt{q_A} = \mu/\sigma = (\mu-\mu')/\sigma`.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 2.0 - _PHI(sqrt_q + sqrt_qa) - _PHI(sqrt_q - sqrt_qa)


class TMuTildeAsymptotic(Distribution):
    r"""Asymptotic distribution for :math:`\tilde{t}_\mu` (two-sided with physical bound, Eq. 40/44).

    Used with the :math:`\tilde{t}_\mu` test statistic for two-sided tests with the
    physical constraint :math:`\mu \geq 0`. The CDF has a piecewise structure with the
    :math:`\Phi + \Phi - 1` form in both regions (Eq. 44).

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        r"""CDF: :math:`F(\tilde{t}_\mu \mid \mu')` — piecewise at threshold :math:`\mu^2/\sigma^2`."""
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
        r"""Null p-value (:math:`\mu' = \mu`), where :math:`q_A = \mu^2/\sigma^2`.

        .. math::

            p_{\mu'=\mu} = \begin{cases}
                2\bigl(1 - \Phi(\sqrt{\tilde{t}})\bigr)
                    & \text{if } \tilde{t} \leq q_A \\
                2 - \Phi(\sqrt{\tilde{t}})
                  - \Phi\!\left(\frac{\tilde{t} + q_A}{2\sqrt{q_A}}\right)
                    & \text{if } \tilde{t} > q_A
            \end{cases}
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
        r"""Alt p-value (:math:`\mu' = 0`), where :math:`q_A = \mu^2/\sigma^2`.

        .. math::

            p_{\mu'=0} = \begin{cases}
                2 - \Phi(\sqrt{\tilde{t}} + \sqrt{q_A})
                  - \Phi(\sqrt{\tilde{t}} - \sqrt{q_A})
                    & \text{if } \tilde{t} \leq q_A \\
                2 - \Phi(\sqrt{\tilde{t}} + \sqrt{q_A})
                  - \Phi\!\left(\frac{\tilde{t} - q_A}{2\sqrt{q_A}}\right)
                    & \text{if } \tilde{t} > q_A
            \end{cases}
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
    r"""Asymptotic distribution for :math:`q_0` (discovery, Eq. 49).

    Used with the :math:`q_0` test statistic for discovery significance.
    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        r"""CDF: :math:`F(q_0 \mid \mu') = \Phi(\sqrt{q_0} - \mu'/\sigma)`."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return _PHI(sqrt_q - mu_prime / sigma)

    def null_pval(self, result: TestStatResult) -> Array:
        r"""Null p-value: :math:`p = 1 - \Phi(\sqrt{q_0})`. No :math:`\sigma` needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 1.0 - _PHI(sqrt_q)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        r"""Alt p-value: :math:`p = 1 - \Phi(\sqrt{q_0} - \sqrt{q_A})`.

        :math:`q_A = \mu_\text{asimov}^2/\sigma^2` (Asimov under signal),
        so :math:`\sqrt{q_A} = \mu_\text{asimov}/\sigma`.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 1.0 - _PHI(sqrt_q - sqrt_qa)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands | None:
        r"""Expected p-values at :math:`\pm N\sigma` fluctuations under signal hypothesis.

        :math:`q_A = \mu_\text{asimov}^2/\sigma^2` (Asimov under signal),
        so :math:`\sqrt{q_A} = \mu_\text{asimov}/\sigma`.
        :math:`q = \max(0, \sqrt{q_A} + N)^2`. Upward fluctuations (:math:`+N`) increase
        discovery significance, opposite to exclusion tests.

        Args:
            result: Must contain ``q_asimov`` for :math:`\sqrt{q_A}`.

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level,
            or None if q_asimov is missing.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Expected"):
            return None

        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))

        def expected_q_fn(n: float) -> Array:
            return jnp.maximum(sqrt_qa + n, 0.0) ** 2

        return _build_expected_bands(self, result, expected_q_fn)


class QMuAsymptotic(Distribution):
    r"""Asymptotic distribution for :math:`q_\mu` (upper limit, Eq. 57).

    Used with the :math:`q_\mu` test statistic for upper limit calculations.

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        r"""CDF: :math:`F(q_\mu \mid \mu') = \Phi(\sqrt{q_\mu} - (\mu - \mu')/\sigma)`."""
        sqrt_q = jnp.sqrt(jnp.maximum(q, 0.0))
        return _PHI(sqrt_q - (mu - mu_prime) / sigma)

    def null_pval(self, result: TestStatResult) -> Array:
        r"""Null p-value: :math:`p = 1 - \Phi(\sqrt{q_\mu})`. No :math:`\sigma` needed."""
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        return 1.0 - _PHI(sqrt_q)

    def alt_pval(self, result: TestStatResult) -> Array | None:
        r"""Alt p-value: :math:`p = 1 - \Phi(\sqrt{q_\mu} - \sqrt{q_A})`.

        :math:`q_A = \mu^2/\sigma^2` (Asimov under :math:`\mu'=0`),
        so :math:`\sqrt{q_A} = \mu/\sigma`.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Alternative"):
            return None
        sqrt_q = jnp.sqrt(jnp.maximum(result.value, 0.0))
        sqrt_qa = jnp.sqrt(jnp.maximum(result.q_asimov, 0.0))
        return 1.0 - _PHI(sqrt_q - sqrt_qa)

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands | None:
        r"""Expected p-values at :math:`\pm N\sigma` fluctuations under background-only.

        :math:`q_A = \mu^2/\sigma^2` (Asimov under :math:`\mu'=0`),
        so :math:`\sqrt{q_A} = \mu/\sigma`.
        At band :math:`N`, the expected :math:`\hat{\mu} = N\sigma`, giving
        :math:`\sqrt{q} = \max(0, \mu/\sigma - N)`.
        Synthetic TestStatResult objects are passed through the existing
        null_pval/alt_pval methods to reuse the CDF logic.

        Args:
            result: Must contain ``q_asimov`` for :math:`\sigma` extraction.

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level,
            or None if q_asimov is missing.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Expected"):
            return None

        sigma = sigma_from_asimov(result.test, result.q_asimov)
        # Guard: at poi=0, sigma=0 → mu/sigma = 0/0 = NaN.
        # Use 0 instead: all expected q become 0, giving CLs=1.0.
        mu_over_sigma = jnp.where(sigma > 0, result.test / sigma, 0.0)

        def expected_q_fn(n: float) -> Array:
            return jnp.maximum(mu_over_sigma - n, 0.0) ** 2

        return _build_expected_bands(self, result, expected_q_fn)


class QTildeAsymptotic(Distribution):
    r"""Asymptotic distribution for :math:`\tilde{q}_\mu` (upper limit with physical bound, Eq. 64).

    Used with the :math:`\tilde{q}_\mu` test statistic for hypothesis testing with the
    physical constraint :math:`\mu \geq 0`. The CDF is piecewise at
    :math:`\tilde{q} = \mu^2/\sigma^2 = q_\text{asimov}`.

    """

    def cdf(self, q: Array, mu: Array, mu_prime: Array, sigma: Array) -> Array:
        r"""CDF: :math:`F(\tilde{q}_\mu \mid \mu')` — piecewise at threshold :math:`\mu^2/\sigma^2`."""
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
        r"""Null p-value (:math:`\mu' = \mu`), where :math:`q_A = \mu^2/\sigma^2`.

        .. math::

            p_{\mu'=\mu} = \begin{cases}
                1 - \Phi(\sqrt{\tilde{q}})
                    & \text{if } \tilde{q} \leq q_A \\
                1 - \Phi\!\left(\frac{\tilde{q} + q_A}{2\sqrt{q_A}}\right)
                    & \text{if } \tilde{q} > q_A
            \end{cases}
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
        r"""Alt p-value (:math:`\mu' = 0`), where :math:`q_A = \mu^2/\sigma^2`.

        .. math::

            p_{\mu'=0} = \begin{cases}
                1 - \Phi(\sqrt{\tilde{q}} - \sqrt{q_A})
                    & \text{if } \tilde{q} \leq q_A \\
                1 - \Phi\!\left(\frac{\tilde{q} - q_A}{2\sqrt{q_A}}\right)
                    & \text{if } \tilde{q} > q_A
            \end{cases}
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

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands | None:
        r"""Expected p-values at :math:`\pm N\sigma` fluctuations under background-only.

        At band :math:`N`, :math:`\hat{\mu} = N\sigma`, so the expected test
        statistic is (with :math:`q_A = \mu^2/\sigma^2`):

        .. math::

            \tilde{q}_\text{exp} = \begin{cases}
                \max(0,\; \mu/\sigma - N)^2
                    & \text{if } N \geq 0 \\
                (\mu/\sigma)^2 - 2(\mu/\sigma)\,N
                    & \text{if } N < 0
            \end{cases}

        Args:
            result: Must contain ``q_asimov`` for :math:`\sigma` extraction.

        Returns:
            ExpectedBands with (pnull, palt) at each sigma level,
            or None if q_asimov is missing.
        """
        if not _require_q_asimov(result, self.__class__.__name__, "Expected"):
            return None

        sigma = sigma_from_asimov(result.test, result.q_asimov)
        # Guard: at poi=0, sigma=0 → mu/sigma = 0/0 = NaN.
        # Use 0 instead: all expected q become 0, giving CLs=1.0.
        mu_over_sigma = jnp.where(sigma > 0, result.test / sigma, 0.0)

        def expected_q_fn(n: float) -> Array:
            standard = jnp.maximum(mu_over_sigma - n, 0.0) ** 2
            boundary = mu_over_sigma**2 - 2.0 * mu_over_sigma * n
            # q̃ is piecewise in μ̂: standard for μ̂ ≥ 0, boundary for μ̂ < 0.
            # At band N, μ̂ = Nσ, so μ̂ ≥ 0 ⟺ N ≥ 0.
            return jnp.where(n >= 0, standard, boundary)

        return _build_expected_bands(self, result, expected_q_fn)


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
        q_null: Test statistics under the tested hypothesis (poi_test).
        q_alt: Test statistics under the alternative hypothesis (poi_alt).
            None if poi_alt was not provided to the ToyGenerator.
    """

    q_null: Array
    q_alt: Array | None = None

    @classmethod
    def from_toys(cls, toys: ToyResult) -> EmpiricalDistribution:
        """Construct from a ToyResult.

        Args:
            toys: Raw toy generation output containing q_null and optionally q_alt.

        Returns:
            An instance of this distribution class.
        """
        return cls(q_null=toys.q_null, q_alt=toys.q_alt)


class SimpleEmpiricalDistribution(EmpiricalDistribution):
    r"""Empirical p-values via simple tail counting.

    :math:`p_\text{null} = \text{fraction of } q_\text{null} \geq q_\text{obs}`,
    :math:`p_\text{alt} = \text{fraction of } q_\text{alt} \geq q_\text{obs}`.
    """

    def null_pval(self, result: TestStatResult) -> Array:
        r"""Empirical p-value under tested hypothesis: fraction of :math:`q_\text{null} \geq q_\text{obs}`."""
        return jnp.mean((self.q_null >= result.value).astype(self.q_null.dtype))

    def alt_pval(self, result: TestStatResult) -> Array | None:
        r"""Empirical p-value under alternative: fraction of :math:`q_\text{alt} \geq q_\text{obs}`.

        Returns None if q_alt was not provided (no alternative toys generated).
        """
        if self.q_alt is None:
            warnings.warn(
                "Alternative p-value computation in SimpleEmpiricalDistribution "
                "cannot be performed without q_alt toys. "
                "Generate toys with poi_alt to compute alternative p-values.",
                stacklevel=2,
            )
            return None
        return jnp.mean((self.q_alt >= result.value).astype(self.q_alt.dtype))

    def expected_pvalues(self, result: TestStatResult) -> ExpectedBands:
        """Compute expected p-values at standard sigma bands using toy quantiles.

        Uses quantiles of q_alt at standard sigma percentiles as synthetic
        test statistic values, then evaluates empirical p-values at each
        via ``_build_expected_bands``.

        Args:
            result: Test statistic result (used as template for synthetic results).

        Returns:
            ExpectedBands with empirical p-values at each sigma level.

        Raises:
            ValueError: If q_alt is None (no alternative toys generated).
        """
        if self.q_alt is None:
            raise ValueError(
                "expected_pvalues requires q_alt toys. "
                "Generate toys with poi_alt to use this method."
            )

        # Standard sigma percentiles: Φ(N) for N in (-2, -1, 0, 1, 2)
        percentiles = jnp.array([_PHI(n) for n in _BAND_SIGMAS])
        q_quantiles = jnp.quantile(self.q_alt, percentiles)

        # Map band index (position in _BAND_SIGMAS) to the quantile value
        def expected_q_fn(n: float) -> Array:
            idx = _BAND_SIGMAS.index(n)
            return q_quantiles[idx]

        return _build_expected_bands(self, result, expected_q_fn)
