"""Hypothesis test calculator.

This module provides AsymptoticCalculator, which orchestrates hypothesis
testing by computing the test statistic (with Asimov), then delegating
p-value computation to Distribution objects.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._results import HypoTestResult
from everwillow.inference.hypotest.distributions import Distribution, QTildeAsymptotic
from everwillow.inference.hypotest.test_statistics import QTilde, TestStatistic

__all__ = ["HypoTestCalculator"]


class HypoTestCalculator(eqx.Module):
    """Hypothesis test calculator.

    The calculator orchestrates hypothesis testing by:
    1. Computing the test statistic on observed data (with Asimov if predict_fn provided)
    2. Delegating p-value computation to a Distribution object
    3. Computing CLs

    For asymptotic workflows, provide ``predict_fn`` to generate Asimov
    data at the distribution's ``mu_asimov`` value. For toy-based
    workflows, ``predict_fn`` can be omitted.

    Attributes:
        test_statistic: Test statistic to use. Defaults to QTilde.
        distribution: Distribution to use for p-value computation.
            Defaults to QTildeAsymptotic.
        predict_fn: Function to generate expected observation from parameters.

    Example:
        >>> from everwillow.inference.hypotest import (
        ...     HypoTestCalculator, QTilde, QTildeAsymptotic
        ... )
        >>> calc = HypoTestCalculator(
        ...     test_statistic=QTilde(),
        ...     distribution=QTildeAsymptotic(),
        ...     predict_fn=my_predict_fn,
        ... )
        >>> result = calc(nll_fn, params, observed, ("mu",), poi_test=1.0)
        >>> print(f"CLs = {result.cl_s:.4f}")
    """

    test_statistic: TestStatistic = eqx.field(default_factory=QTilde)
    distribution: Distribution = eqx.field(default_factory=QTildeAsymptotic)
    predict_fn: tp.Callable[[sl.State], PyTree] | None = None

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run hypothesis test.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            HypoTestResult with observed p-values and CLs.
        """
        # Compute test statistic with Asimov handled internally
        ts_result = self.test_statistic(
            nll_fn,
            params,
            observation,
            poi_key,
            poi_test,
            predict_fn=self.predict_fn,
            mu_asimov=self.distribution.mu_asimov,
            **fit_kwargs,
        )

        # Delegate p-value computation to distribution
        pnull = self.distribution.null_pval(ts_result)
        palt = self.distribution.alt_pval(ts_result)

        # TODO: CLs convention (pnull/palt vs palt/pnull) needs resolution
        cl_s_value = jnp.array(float("nan"))

        return HypoTestResult(
            q_obs=ts_result.value,
            pnull=pnull,
            palt=palt,
            cl_s=cl_s_value,
            test_stat_result=ts_result,
            expected_bands=None,
        )
