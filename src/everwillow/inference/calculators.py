"""Asymptotic p-value and CLs calculations for hypothesis testing.

Mathematical Background
-----------------------
Under asymptotic approximation (Wilks' theorem), the test statistic sqrt(q)
follows a normal distribution:

    - Under null hypothesis (mu=0):        sqrt(q) ~ N(0, 1)
    - Under alternative hypothesis (mu'):  sqrt(q) ~ N(sqrt(q_A), 1)

where q_A is the Asimov test statistic (q evaluated on Asimov data).

The p-value is the probability of observing a test statistic at least as
extreme as the observed value:

    p = P(q >= q_obs | hypothesis)
      = P(sqrt(q) >= sqrt(q_obs) | hypothesis)
      = 1 - Phi((sqrt(q_obs) - mu_distribution) / 1)

For the standard CLs calculation at the **expected median** (Asimov approximation):
    - The expected test statistic under b-only is 0 (since sqrt(q) ~ N(0,1))
    - p_sb = 1 - Phi(0 - (-sqrt(q_A))) = 1 - Phi(sqrt(q_A))
    - p_b  = 1 - Phi(0 - 0) = 0.5
    - CLs  = p_sb / p_b = 2 * (1 - Phi(sqrt(q_A)))

References:
    - Cowan et al., "Asymptotic formulae for likelihood-based tests"
      Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
    - pyhf: https://github.com/scikit-hep/pyhf
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.stats as stats


def pvalue_onesided(q: float, shift: float = 0.0) -> jnp.ndarray:
    """Compute one-sided p-value using asymptotic approximation.

    Formula:
        p = 1 - Phi(sqrt(q) - shift)

    where Phi is the standard normal CDF.

    Under the asymptotic approximation, sqrt(q) is normally distributed:
        - Under null (b-only):  sqrt(q) ~ N(0, 1)           -> shift = 0
        - Under alt (s+b):      sqrt(q) ~ N(sqrt(q_A), 1)   -> shift = sqrt(q_A)

    For expected (median) CLs calculation:
        - Use q = 0 (expected value under b-only)
        - p_sb: shift = -sqrt(q_A)  (note: NEGATIVE for pyhf compatibility)
        - p_b:  shift = 0

    Note on sign convention:
        The shift can be positive or negative. For expected CLs matching pyhf:
        - Use shift = -sqrt(q_asimov) for the s+b hypothesis
        - This gives p_sb = 1 - Phi(sqrt(q_A)) when q=0

    Args:
        q: Test statistic value (non-negative).
        shift: Distribution shift parameter. For expected CLs with pyhf
               convention, use -sqrt(q_asimov) for s+b hypothesis.

    Returns:
        P-value in [0, 1].

    Examples:
        >>> # Expected median CLs calculation
        >>> q_asimov = 1.9  # from Asimov dataset
        >>> p_sb = pvalue_onesided(0.0, shift=-jnp.sqrt(q_asimov))  # ~0.084
        >>> p_b = pvalue_onesided(0.0, shift=0.0)  # 0.5
        >>> cls_val = p_sb / p_b  # ~0.168
    """
    return 1.0 - stats.norm.cdf(jnp.sqrt(q) - shift)


def cls(p_alt: float, p_null: float) -> jnp.ndarray:
    """Compute CLs value from p-values.

    Formula:
        CLs = p_alt / p_null

    The CLs method (modified frequentist) protects against excluding signal
    hypotheses when there is no sensitivity. It's the ratio of the p-value
    under the alternative to the p-value under the null.

    Interpretation:
        - CLs < 0.05: Exclude the signal hypothesis at 95% CL
        - CLs ~ 1: No sensitivity to distinguish hypotheses

    Traditional (signal strength mu):
        - p_alt  = p_s+b = P(q >= q_obs | signal+background)
        - p_null = p_b   = P(q >= q_obs | background-only)

    EFT (Wilson coefficient c):
        - p_alt  = P(q >= q_obs | c = c_test)
        - p_null = P(q >= q_obs | c = 0, Standard Model)

    Args:
        p_alt: P-value under alternative hypothesis (s+b or EFT).
        p_null: P-value under null hypothesis (b-only or SM).

    Returns:
        CLs value in [0, inf). Protected against division by zero.

    References:
        - Read, A.L., "Presentation of search results: the CLs technique"
          J. Phys. G 28 (2002) 2693
    """
    return p_alt / jnp.maximum(p_null, 1e-10)
