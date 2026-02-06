"""Hypothesis testing framework for statistical inference.

This module provides the required pieces for hypothesis testing:
    - **Test Statistics**: Compute likelihood ratios (QTilde, QMu, Q0, TMu)
    - **Distributions**: Convert test statistics to p-values (asymptotic or empirical)
    - **Calculator**: Orchestrates test statistics and distributions
    - **Toy Generator**: Creates empirical distributions from Monte Carlo toys
    - **Upper Limits**: Root-finding for exclusion limits

Example:
    >>> from everwillow.inference.hypotest import (
    ...     HypoTestCalculator, QTilde, QTildeAsymptotic, upper_limit
    ... )
    >>> calc = HypoTestCalculator(test_statistic=QTilde())
    >>> dist = QTildeAsymptotic()
    >>> result = calc(nll_fn, params, ("mu",), poi_test=1.0, distribution=dist)
    >>> print(f"CLs = {result.cl_s:.4f}")
    >>>
    >>> # Find 95% CL upper limit
    >>> limit = upper_limit(
    ...     lambda poi: calc(nll_fn, params, ("mu",), poi, distribution=dist).cl_s,
    ...     bounds=(0, 5), level=0.05
    ... )
"""

from __future__ import annotations

from everwillow.inference.hypotest._results import (
    ExpectedBands,
    ExpectedLimitResult,
    HypoTestResult,
    HypoTestToysResult,
    TestStatResult,
)
from everwillow.inference.hypotest._utils import cl_s
from everwillow.inference.hypotest.calculators import HypoTestCalculator
from everwillow.inference.hypotest.distributions import (
    Distribution,
    EmpiricalDistribution,
    Q0Asymptotic,
    QMuAsymptotic,
    QTildeAsymptotic,
    TMuAsymptotic,
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
    upper_limit_toys,
)

__all__ = [
    "Q0",
    "Distribution",
    "EmpiricalDistribution",
    "ExpectedBands",
    "ExpectedLimitResult",
    "HypoTestCalculator",
    "HypoTestResult",
    "HypoTestToysResult",
    "Q0Asymptotic",
    "QMu",
    "QMuAsymptotic",
    "QTilde",
    "QTildeAsymptotic",
    "TMu",
    "TMuAsymptotic",
    "TestStatResult",
    "TestStatistic",
    "ToyGenerator",
    "cl_s",
    "expected_upper_limit",
    "upper_limit",
    "upper_limit_toys",
]
