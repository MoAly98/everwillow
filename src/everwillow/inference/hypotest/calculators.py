"""Hypothesis test calculator.

This module provides HypoTestCalculator, a unified calculator that orchestrates
hypothesis testing by delegating p-value computation to Distribution objects.

The calculator is purely orchestration - it does not make assumptions about
what data distributions need. It passes the full TestStatResult and lets each
distribution extract what it needs.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._results import HypoTestResult
from everwillow.inference.hypotest._utils import cl_s
from everwillow.inference.hypotest.distributions import Distribution
from everwillow.inference.hypotest.test_statistics import QTilde, TestStatistic

__all__ = ["HypoTestCalculator"]


class HypoTestCalculator(eqx.Module):
    """Unified hypothesis test calculator.

    The calculator orchestrates hypothesis testing by:
    1. Computing the test statistic
    2. Delegating p-value computation to a Distribution object
    3. Computing CLs and expected bands via the distribution

    Attributes:
        test_statistic: Test statistic to use. Defaults to QTilde.

    Example:
        >>> from everwillow.inference.hypotest import (
        ...     HypoTestCalculator, QTilde, QTildeAsymptotic
        ... )
        >>> calc = HypoTestCalculator(test_statistic=QTilde())
        >>> dist = QTildeAsymptotic()
        >>> result = calc(nll_fn, params, ("mu",), poi_test=1.0, distribution=dist)
        >>> print(f"CLs = {result.cl_s:.4f}")
    """

    test_statistic: TestStatistic = eqx.field(default_factory=QTilde)

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        distribution: Distribution,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run hypothesis test.

        Args:
            nll_fn: Negative log-likelihood function.
            params: Initial parameter state.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            distribution: Distribution object for p-value computation.
            asimov_nll_fn: Optional NLL for Asimov dataset. If provided,
                the test statistic will store q_asimov in extras.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            HypoTestResult with observed and expected p-values.
        """
        # 1. Compute test statistic (includes Asimov if provided)
        ts_result = self.test_statistic(
            nll_fn,
            params,
            poi_key,
            poi_test,
            asimov_nll_fn=asimov_nll_fn,
            **fit_kwargs,
        )

        # 2. Delegate p-value computation to distribution
        pnull, palt = distribution.pvalues(ts_result)

        # 3. Compute CLs = palt / pnull
        cl_s_value = cl_s(palt, pnull)

        # 4. Delegate expected bands to distribution
        expected_bands = distribution.expected_pvalues(ts_result)

        return HypoTestResult(
            q_obs=ts_result.q,
            pnull=pnull,
            palt=palt,
            cl_s=cl_s_value,
            expected_bands=expected_bands,
            test_stat_result=ts_result,
        )
