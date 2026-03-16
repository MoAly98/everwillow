"""Hypothesis testing framework for statistical inference.

This module provides the required pieces for hypothesis testing:
    - **Test Statistics**: Compute likelihood ratios (QTilde, QMu, Q0, TMu)
    - **Distributions**: Convert test statistics to p-values (asymptotic or empirical)
    - **Calculator**: Orchestrates test statistics and distributions
    - **Toy Generator**: Generates Monte Carlo toys for empirical distributions
    - **Upper Limits**: Root-finding for exclusion limits

Example:
    >>> from everwillow.inference.hypotest import (
    ...     HypoTestCalculator, QTilde, QTildeAsymptotic, upper_limit
    ... )
    >>> calc = HypoTestCalculator(test_statistic=QTilde(), distribution=QTildeAsymptotic())
    >>> result = calc(nll_fn, params, observed, ("mu",), poi_test=1.0)
    >>> print(f"CLs = {result.cl_s:.4f}")
    >>>
    >>> # Find 95% CL upper limit
    >>> limit = upper_limit(
    ...     lambda poi: calc(nll_fn, params, observed, ("mu",), poi).cl_s,
    ...     bounds=(0, 5), level=0.05
    ... )
"""

from __future__ import annotations

from everwillow.inference.hypotest._results import (
    BandValues,
    ExpectedBands,
    ExpectedLimitResult,
    HypoTestResult,
    TestStatResult,
    ToyResult,
)
from everwillow.inference.hypotest._utils import (
    cl_s,
    make_asimov,
    sigma_from_asimov,
    significance,
)
from everwillow.inference.hypotest.calculators import (
    AsymptoticCalculator,
    HypoTestCalculator,
)
from everwillow.inference.hypotest.distributions import (
    Distribution,
    EmpiricalDistribution,
    Q0Asymptotic,
    QMuAsymptotic,
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
    TMuAsymptotic,
    TMuTildeAsymptotic,
)
from everwillow.inference.hypotest.test_statistics import (
    Q0,
    QMu,
    QTilde,
    TestStatistic,
    TMu,
)
from everwillow.inference.hypotest.toys import ToyGenerator
from everwillow.inference.hypotest.upper_limit import (
    expected_upper_limit,
    upper_limit,
    upper_limit_scan,
    upper_limit_toys,
)

__all__ = [
    "Q0",
    "AsymptoticCalculator",
    "BandValues",
    "Distribution",
    "EmpiricalDistribution",
    "ExpectedBands",
    "ExpectedLimitResult",
    "HypoTestCalculator",
    "HypoTestResult",
    "Q0Asymptotic",
    "QMu",
    "QMuAsymptotic",
    "QTilde",
    "QTildeAsymptotic",
    "SimpleEmpiricalDistribution",
    "TMu",
    "TMuAsymptotic",
    "TMuTildeAsymptotic",
    "TestStatResult",
    "TestStatistic",
    "ToyGenerator",
    "ToyResult",
    "cl_s",
    "expected_upper_limit",
    "make_asimov",
    "sigma_from_asimov",
    "significance",
    "upper_limit",
    "upper_limit_scan",
    "upper_limit_toys",
]
