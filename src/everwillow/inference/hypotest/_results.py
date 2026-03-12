"""Result containers for hypothesis testing.

These modules hold the results of hypothesis tests.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
from jaxtyping import Array

from everwillow.inference.hypotest._utils import cl_s

__all__ = [
    "ExpectedBands",
    "ExpectedLimitResult",
    "HypoTestResult",
    "HypoTestToysResult",
    "TestStatResult",
    "ToyResult",
]


class TestStatResult(eqx.Module):
    """Result of computing a test statistic.

    Attributes:
        value: Test statistic value.
        test: POI value being tested (μ).
        q_asimov: Test statistic evaluated on Asimov data. None if not computed.
        extras: Arbitrary additional data (e.g., fits, mu_hat).
    """

    value: Array
    test: Array
    q_asimov: Array | None = None
    extras: dict[str, tp.Any] = eqx.field(default_factory=dict)


class ToyResult(eqx.Module):
    """Raw output from toy generation.

    Contains the test statistic arrays under both hypotheses,
    decoupled from any particular p-value computation method.

    Attributes:
        q_alt: Test statistic values under alternative (signal+background) hypothesis.
        q_null: Test statistic values under null (background-only) hypothesis.
    """

    q_alt: Array
    q_null: Array


class ExpectedBands(eqx.Module):
    """Expected p-values at standard sigma bands.

    Computed from Asimov dataset under background-only hypothesis.
    Each band contains (pnull, palt) tuple.

    Attributes:
        minus_2sigma: Expected at -2σ fluctuation.
        minus_1sigma: Expected at -1σ fluctuation.
        median: Expected at median (0σ).
        plus_1sigma: Expected at +1σ fluctuation.
        plus_2sigma: Expected at +2σ fluctuation.
    """

    minus_2sigma: tuple[Array, Array]
    minus_1sigma: tuple[Array, Array]
    median: tuple[Array, Array]
    plus_1sigma: tuple[Array, Array]
    plus_2sigma: tuple[Array, Array]

    def cls_bands(self) -> tuple[Array, Array, Array, Array, Array]:
        """Return CLs values at each band.

        ``CLs = palt / pnull``

        Returns:
            Tuple of CLs at (-2σ, -1σ, median, +1σ, +2σ).
        """

        return (
            cl_s(self.minus_2sigma[1], self.minus_2sigma[0]),
            cl_s(self.minus_1sigma[1], self.minus_1sigma[0]),
            cl_s(self.median[1], self.median[0]),
            cl_s(self.plus_1sigma[1], self.plus_1sigma[0]),
            cl_s(self.plus_2sigma[1], self.plus_2sigma[0]),
        )


class HypoTestResult(eqx.Module):
    """Result of an asymptotic hypothesis test.

    Attributes:
        q_obs: Observed test statistic value.
        pnull: p-value under null hypothesis (background-only).
        palt: p-value under alternative hypothesis (signal+background).
        cl_s: CLs value (palt / pnull).
        expected_bands: Expected p-values at sigma bands (from Asimov).
        test_stat_result: Full test statistic result with fit information.
    """

    q_obs: Array
    pnull: Array | None
    palt: Array | None
    cl_s: Array | None
    test_stat_result: TestStatResult
    expected_bands: ExpectedBands | None = None


class HypoTestToysResult(eqx.Module):
    """Result of a toy-based hypothesis test.

    Attributes:
        q_obs: Observed test statistic value.
        pnull: p-value under null hypothesis (background-only).
        palt: p-value under alternative hypothesis (signal+background).
        cl_s: CLs value (palt / pnull).
        expected_bands: Expected p-values at sigma bands (from toy distributions).
        ntoys: Number of toys used in each hypothesis.
        q_alt: Test statistic values from alternative toys.
        q_null: Test statistic values from null toys.
        test_stat_result: Full test statistic result with fit information.
    """

    q_obs: Array
    pnull: Array
    palt: Array
    cl_s: Array
    expected_bands: ExpectedBands
    ntoys: int
    q_alt: Array
    q_null: Array
    test_stat_result: TestStatResult


class ExpectedLimitResult(eqx.Module):
    """Result of expected upper limit computation with Brazil bands.

    Contains observed limit and expected limits at standard sigma levels,
    suitable for producing Brazil band plots.

    Attributes:
        observed: Observed upper limit.
        expected: Expected (median) upper limit.
        minus_2sigma: Expected limit at -2σ fluctuation.
        minus_1sigma: Expected limit at -1σ fluctuation.
        plus_1sigma: Expected limit at +1σ fluctuation.
        plus_2sigma: Expected limit at +2σ fluctuation.
    """

    observed: Array
    expected: Array
    minus_2sigma: Array
    minus_1sigma: Array
    plus_1sigma: Array
    plus_2sigma: Array
