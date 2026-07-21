"""Result containers for hypothesis testing.

These modules hold the results of hypothesis tests.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
from jaxtyping import Array

if tp.TYPE_CHECKING:
    from everwillow._src.inference.hypotest.distributions import Distribution

__all__ = [
    "BandValues",
    "ExpectedBands",
    "HypoTestResult",
    "TestStatResult",
    "ToyResult",
]


class TestStatResult(eqx.Module):
    r"""Result of computing a test statistic.

    Attributes:
        value: Test statistic value.
        test: The tested POI point, a mapping from each POI key to its tested
            value (:math:`\mu`). A single-POI test carries a one-entry mapping;
            a joint test carries one entry per POI.
        q_asimov: Test statistic evaluated on Asimov data. None if not computed.
        extras: Arbitrary additional data (e.g., fits, mu_hat).
    """

    value: Array
    test: tp.Mapping[tp.Any, Array]
    q_asimov: Array | None = None
    extras: dict[str, tp.Any] = eqx.field(default_factory=dict)


class ToyResult(eqx.Module):
    """Raw output from toy generation.

    Contains the test statistic arrays under both hypotheses,
    decoupled from any particular p-value computation method.

    Attributes:
        q_null: Test statistic values under the tested hypothesis (poi_test).
        q_alt: Test statistic values under the alternative hypothesis (poi_alt).
            None if poi_alt was not provided to the ToyGenerator.
    """

    q_null: Array
    q_alt: Array | None = None


class BandValues(eqx.Module):
    r"""Scalar values at standard :math:`\pm N\sigma` fluctuation bands.

    Supports iteration via ``for name, value in bv``, indexing via
    ``bv["median"]``, and ``len(bv) == 5``.  ``dict(bv)`` produces a
    ``{name: value}`` mapping, and ``BandValues(**dict(bv))`` roundtrips.

    Attributes:
        minus_2sigma: Value at :math:`-2\sigma` fluctuation.
        minus_1sigma: Value at :math:`-1\sigma` fluctuation.
        median: Value at median (:math:`0\sigma`).
        plus_1sigma: Value at :math:`+1\sigma` fluctuation.
        plus_2sigma: Value at :math:`+2\sigma` fluctuation.
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
    r"""Expected quantities at standard sigma bands.

    All derived quantities (CLs, significance) are eagerly computed
    at construction time so access is a simple attribute lookup.

    Attributes:
        null_pvalue: p-value under null hypothesis (:math:`p_\mu`) at each band.
        alt_pvalue: p-value under alternative hypothesis (:math:`\text{CL}_b`) at each band.
        cl_s: :math:`\text{CL}_s = p_\text{null}/p_\text{alt}` at each band.
        null_sig: Null significance :math:`\Phi^{-1}(1 - p_\text{null})` at each band.
        alt_sig: Alternative significance :math:`\Phi^{-1}(1 - p_\text{alt})` at each band.
    """

    null_pvalue: BandValues
    alt_pvalue: BandValues
    cl_s: BandValues
    null_sig: BandValues
    alt_sig: BandValues


class HypoTestResult(eqx.Module):
    """Result of a hypothesis test.

    Attributes:
        q_obs: Observed test statistic value.
        pnull: p-value under the tested hypothesis (poi_test).
        palt: p-value under the alternative hypothesis (poi_alt / background-only).
        test_stat_result: Full test statistic result with fit information.
        dist: The distribution of the test statistic which was used to compute p-values.
    """

    q_obs: Array
    pnull: Array | None
    palt: Array | None
    test_stat_result: TestStatResult
    distribution: Distribution | None = None
