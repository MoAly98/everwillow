"""Result containers for hypothesis testing.

These modules hold the results of hypothesis tests.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
from jaxtyping import Array

__all__ = [
    "BandValues",
    "ExpectedBands",
    "ExpectedLimitResult",
    "HypoTestResult",
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


class BandValues(eqx.Module):
    """Scalar values at standard ±Nσ fluctuation bands.

    Supports iteration via ``for name, value in bv``, indexing via
    ``bv["median"]``, and ``len(bv) == 5``.  ``dict(bv)`` produces a
    ``{name: value}`` mapping, and ``BandValues(**dict(bv))`` roundtrips.

    Attributes:
        minus_2sigma: Value at -2σ fluctuation.
        minus_1sigma: Value at -1σ fluctuation.
        median: Value at median (0σ).
        plus_1sigma: Value at +1σ fluctuation.
        plus_2sigma: Value at +2σ fluctuation.
    """

    _NAMES: tp.ClassVar[tuple[str, ...]] = (
        "minus_2sigma",
        "minus_1sigma",
        "median",
        "plus_1sigma",
        "plus_2sigma",
    )

    minus_2sigma: Array
    minus_1sigma: Array
    median: Array
    plus_1sigma: Array
    plus_2sigma: Array

    def __iter__(self) -> tp.Iterator[tuple[str, Array]]:
        for name in self._NAMES:
            yield name, getattr(self, name)

    def __getitem__(self, key: str) -> Array:
        if key not in self._NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __len__(self) -> int:
        return 5


class ExpectedBands(eqx.Module):
    """Expected quantities at standard sigma bands.

    All derived quantities (CLs, significance) are eagerly computed
    at construction time so access is a simple attribute lookup.

    Attributes:
        null_pvalue: p-value under null hypothesis (p_μ) at each band.
        alt_pvalue: p-value under alternative hypothesis (CL_b) at each band.
        cl_s: CLs = pnull/palt at each band.
        null_sig: Null significance Φ⁻¹(1 - pnull) at each band.
        alt_sig: Alternative significance Φ⁻¹(1 - palt) at each band.
    """

    null_pvalue: BandValues
    alt_pvalue: BandValues
    cl_s: BandValues
    null_sig: BandValues
    alt_sig: BandValues


class HypoTestResult(eqx.Module):
    """Result of an asymptotic hypothesis test.

    Attributes:
        q_obs: Observed test statistic value.
        pnull: p-value under null hypothesis (background-only).
        palt: p-value under alternative hypothesis (signal+background).
        cl_s: CLs value (pnull / palt).
        expected_bands: Expected p-values at sigma bands (from Asimov).
        test_stat_result: Full test statistic result with fit information.
    """

    q_obs: Array
    pnull: Array | None
    palt: Array | None
    cl_s: Array | None
    test_stat_result: TestStatResult
    expected_bands: ExpectedBands | None = None


class ExpectedLimitResult(eqx.Module):
    """Result of expected upper limit computation with Brazil bands.

    Contains observed limit and expected limits at standard sigma levels,
    suitable for producing Brazil band plots.

    Attributes:
        observed: Observed upper limit.
        expected: Expected limits at ±Nσ fluctuation bands.
    """

    observed: Array
    expected: BandValues
