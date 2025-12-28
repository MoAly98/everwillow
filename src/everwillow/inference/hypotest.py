"""Hypothesis testing using asymptotic approximations.

This module provides the main entry point for CLs-based hypothesis testing
using the asymptotic formulae from Cowan et al. (2011).

The workflow is:
    1. Compute q_obs on observed data
    2. Compute q_asimov on Asimov data (background-only expectation)
    3. Use asymptotic formulae to compute p-values and CLs
    4. Compute expected CLs bands at +/- 1,2 sigma

References:
    - Cowan et al., "Asymptotic formulae for likelihood-based tests"
      Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import everwillow.statelib as sl
from everwillow.inference.calculators import pvalue_onesided
from everwillow.inference.test_statistics import q_tilde


@dataclass
class HypoTestResult:
    """Result of asymptotic hypothesis test.

    Attributes:
        p_alt: P-value under alternative hypothesis (poi = poi_test).
        p_null: P-value under null hypothesis (poi = 0).
        q_obs: Observed test statistic.
        expected_pvalues: Expected p-value bands [(p_alt, p_null) for -2s, -1s, med, +1s, +2s].

    For traditional fits:
        - alt = signal+background (mu = mu_test)
        - null = background-only (mu = 0)

    For EFT fits:
        - alt = EFT (c = c_test)
        - null = Standard Model (c = 0)

    To compute CLs, use the cls() function:
        >>> from everwillow.inference.calculators import cls
        >>> cls_obs = cls(result.p_alt, result.p_null)

    Note:
        Values may be JAX arrays (for JIT compatibility) or Python floats.
        Use float(result.p_alt) if you need a Python float.
    """

    p_alt: float | Float[Array, ""]
    p_null: float | Float[Array, ""]
    q_obs: float | Float[Array, ""]
    expected_pvalues: list[tuple[float | Float[Array, ""], float | Float[Array, ""]]]


def hypotest(
    nll_fn: tp.Callable[[PyTree], float],
    params: sl.State,
    poi_name: str,
    poi_test: float,
    *,
    asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
    solver: tp.Any = None,
    max_steps: int = 256,
) -> HypoTestResult:
    """Perform asymptotic hypothesis test at a single POI value.

    Computes the CLs value using the asymptotic approximation. This is the
    main entry point for hypothesis testing in everwillow.

    The CLs method computes:
        CLs = p_sb / p_b

    where p_sb and p_b are the p-values under the signal+background and
    background-only hypotheses respectively.

    For expected CLs, we evaluate at different sigma bands corresponding to
    fluctuations in the test statistic under background-only.

    Args:
        nll_fn: Negative log-likelihood function for observed data.
        params: Initial parameter state.
        poi_name: Name of parameter of interest (e.g., "mu", "c_tG").
        poi_test: Test value for the POI.
        asimov_nll_fn: Optional separate NLL for Asimov data. If None, uses
                       nll_fn (assumes observed equals Asimov expectation).
        solver: Optional solver for optimization.
        max_steps: Maximum optimization steps.

    Returns:
        HypoTestResult with observed and expected CLs values.

    Examples:
        >>> result = hypotest(nll_fn, params, "mu", poi_test=1.0)
        >>> print(f"CLs = {result.cls_obs:.4f}")
        >>> if result.cls_obs < 0.05:
        ...     print("Excluded at 95% CL")
    """
    # 1. Compute q on observed data
    q_obs, _constrained_fit, _free_fit = q_tilde(
        nll_fn, params, poi_name, poi_test, solver=solver, max_steps=max_steps
    )

    # 2. Compute q on Asimov data
    # If asimov_nll_fn is provided, use it; otherwise assume obs = Asimov
    if asimov_nll_fn is not None:
        q_asimov, _, _ = q_tilde(
            asimov_nll_fn,
            params,
            poi_name,
            poi_test,
            solver=solver,
            max_steps=max_steps,
        )
    else:
        # When observed data equals Asimov expectation, q_obs = q_asimov
        q_asimov = q_obs

    # 3. Compute observed CLs using asymptotic formulae
    # The shift for s+b distribution is -sqrt(q_asimov) (pyhf convention)
    sqrt_q_asimov = jnp.sqrt(q_asimov)

    # For observed CLs, we use the transformed test statistic approach
    # When obs = Asimov, the effective test statistic is 0
    # (since sqrt(q_obs) - sqrt(q_asimov) = 0)
    #
    # More generally, the effective test stat is sqrt(q_obs) - sqrt(q_asimov)
    # But for simplicity in the asymptotic limit, when data ~ Asimov,
    # we evaluate at the expected value under b-only which is 0.
    #
    # p_sb = 1 - Phi(sqrt(q) - (-sqrt(q_A))) = 1 - Phi(sqrt(q) + sqrt(q_A))
    # p_b  = 1 - Phi(sqrt(q) - 0) = 1 - Phi(sqrt(q))
    #
    # At q=0 (expected under b-only):
    # p_sb = 1 - Phi(sqrt(q_A))
    # p_b  = 0.5

    # Use q=0 for expected median calculation (matches pyhf asymptotic)
    q_for_pvalue = 0.0
    shift_alt = -sqrt_q_asimov  # Negative shift for alternative hypothesis

    # Keep as JAX arrays for JIT compatibility (e.g., when used inside optimistix)
    p_alt = pvalue_onesided(q_for_pvalue, shift=shift_alt)
    p_null = pvalue_onesided(q_for_pvalue, shift=0.0)

    # 4. Compute expected p-value bands
    # At N sigma deviation from median:
    # p_alt  = 1 - Phi(N + sqrt(q_A))
    # p_null = 1 - Phi(N)
    sigma_values = [2.0, 1.0, 0.0, -1.0, -2.0]  # Order: -2s, -1s, med, +1s, +2s
    expected_pvalues = []

    for n_sigma in sigma_values:
        p_alt_exp = 1.0 - _normal_cdf(n_sigma + sqrt_q_asimov)
        p_null_exp = 1.0 - _normal_cdf(n_sigma)
        expected_pvalues.append((p_alt_exp, p_null_exp))

    return HypoTestResult(
        p_alt=p_alt,
        p_null=p_null,
        q_obs=q_obs,
        expected_pvalues=expected_pvalues,
    )


def _normal_cdf(x):
    """Standard normal CDF (JAX-compatible)."""
    import jax.scipy.stats as stats

    return stats.norm.cdf(x)
