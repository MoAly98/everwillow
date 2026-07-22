# Extending Hypothesis Testing

Everwillow's hypothesis testing is built from three independent components that
can be mixed, matched, and subclassed:

| Component | Base class | Role |
|-----------|-----------|------|
| Test statistic | `TestStatistic` | Computes a scalar from the NLL and data |
| Distribution | `Distribution` | Converts the scalar into p-values |
| Calculator | `HypoTestCalculator` | Binds the model and orchestrates the test |

All three are [equinox Modules](https://docs.kidger.site/equinox/), so they are
immutable pytrees that work with `jax.jit`, `jax.vmap`, and `jax.grad` out of
the box.

## Custom test statistic

Subclass `TestStatistic` and implement `_compute`. The base class `compute()`
method calls `_compute()`, packages the result into a `TestStatResult`, and
returns it:

The tested hypothesis arrives as a *POI point*: a mapping from POI key to
value, e.g. `{"mu": 1.0}`. One-sided statistics that order on a scalar
`mu_hat` should reject points naming more than one POI:

```python
import jax.numpy as jnp

import everwillow as ew
from everwillow.hypotest.test_statistics import TestStatistic
from everwillow.hypotest.utils import constrained_fit


class SignedQMu(TestStatistic):
    """Signed q_mu: negative when mu_hat > mu_test."""

    def _compute(self, nll_fn, params, observation, poi_test, **kw):
        (poi_key,) = poi_test  # a signed statistic orders on one scalar POI
        poi_value = poi_test[poi_key]

        fit_free = ew.fit(nll_fn, params, observation, **kw)
        mu_hat = fit_free.params[poi_key]

        fit_cond = constrained_fit(nll_fn, params, observation, poi_test, **kw)

        q = 2.0 * (fit_cond.nll - fit_free.nll)
        q_signed = jnp.sign(poi_value - mu_hat) * q

        return q_signed, {"mu_hat": mu_hat}
```

For Cowan-style test statistics that need an Asimov dataset for asymptotic
p-values, subclass `CowanTestStatistic` instead. This class adds automatic Asimov
generation via `predict_fn` and populates `q_asimov` on the result.

## Custom distribution

Subclass `Distribution` and implement `null_pval` and `alt_pval`. You get
`null_significance`, `alt_significance`, and `pvalue_bands` for free:

```python
import jax
import jax.numpy as jnp

from everwillow.hypotest.distributions import Distribution


class HalfNormalDistribution(Distribution):
    """Toy distribution: q ~ half-normal(sigma=1) under both hypotheses."""

    def null_pval(self, result):
        cdf = jax.scipy.stats.norm.cdf(jnp.sqrt(result.value)) * 2 - 1
        return 1 - cdf

    def alt_pval(self, result):
        cdf = jax.scipy.stats.norm.cdf(jnp.sqrt(result.value)) * 2 - 1
        return 1 - cdf
```

### Custom empirical distribution

For toy-based p-values with non-trivial estimators (KDE smoothing, tail
extrapolation, etc.), subclass `EmpiricalDistribution`:

```python
import jax.numpy as jnp

from everwillow.hypotest.distributions import EmpiricalDistribution


class SmoothedEmpiricalDistribution(EmpiricalDistribution):
    """Gaussian-KDE smoothed empirical p-values."""

    bandwidth: float = 0.1

    def null_pval(self, result):
        z = (self.q_null - result.value) / self.bandwidth
        return jnp.mean(jax.scipy.stats.norm.cdf(z))

    def alt_pval(self, result):
        if self.q_alt is None:
            return None
        z = (self.q_alt - result.value) / self.bandwidth
        return jnp.mean(jax.scipy.stats.norm.cdf(z))
```

`EmpiricalDistribution` provides the `from_toys(toys)` factory and stores
`q_null`/`q_alt` arrays. You only need to define how p-values are computed
from those arrays.

## Custom calculator

`HypoTestCalculator` is the abstract core: it binds the model (NLL,
parameters, data) and provides `cls(result)`, `pvalue_bands(result)`, and the
upper limit methods. Subclasses implement `test(poi)`, which decides where
the p-values come from, and must record the distribution that produced them
on the returned result (`distribution=...`).

For a fixed distribution there is no need to subclass; `AsymptoticCalculator`
accepts any `Distribution`:

```python
from everwillow.hypotest.calculators import AsymptoticCalculator
from everwillow.hypotest.test_statistics import QTilde

calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    test_statistic=QTilde(),  # or your custom TestStatistic
    distribution=HalfNormalDistribution(),  # or any Distribution
)

result = calc.test(1.0)
print(calc.cls(result))
```

The two concrete calculators show the subclassing pattern:
`AsymptoticCalculator` computes p-values from its fixed `distribution` field
and injects the Asimov configuration into each `test()` call;
`ToyCalculator` regenerates toy ensembles per call and builds the
distribution through its `distribution_factory`.

## Custom toy generation

`ToyGenerator` has two extension points: how pseudo-experiments are sampled
(`sample_fn`) and how the single-toy function is mapped over keys (`map_fn`).

### Custom sampling

`ToyGenerator` accepts a `sample_fn(params_state: State, key: PRNGKeyArray) -> PyTree`
for full control over pseudo-experiment generation. The returned pytree must
match the `observation` structure expected by `nll_fn`, since it replaces
`observation` for each toy. The default Poisson sampler is created from
`predict_fn`, but you can replace it with any sampling strategy:

```python
import jax

from everwillow.hypotest.test_statistics import QTilde
from everwillow.hypotest.toys import ToyGenerator


def sample_fn(params_state, key):
    """Gaussian pseudo-experiments instead of Poisson."""
    mu = params_state.to_pytree()["mu"]
    expected = mu * signal + background
    n = expected + jax.random.normal(key) * jnp.sqrt(expected)
    # keep yields positive: a negative count makes the Poisson NLL unbounded
    return {"n": jnp.maximum(n, 0.1)}


toy_gen = ToyGenerator(sample_fn=sample_fn, ntoys=10_000)
toys = toy_gen.generate(
    nll,
    params,
    observed,
    poi_null={"mu": 1.0},
    test_statistic=QTilde(),
    poi_alt={"mu": 0.0},
    key=jax.random.key(0),
)
```

The resulting `ToyResult` feeds into any `EmpiricalDistribution` subclass via
`from_toys(toys)`.

### Custom parallelisation

By default `ToyGenerator` uses `jax.vmap` to map the single-toy function over
keys. The `map_fn` argument lets you swap in any mapping strategy with the same
`map_fn(f) -> batched_f` signature:

```python
from functools import partial

import jax
import jax.numpy as jnp

from everwillow.hypotest.test_statistics import QTilde
from everwillow.hypotest.toys import ToyGenerator

# Batched mapping (processes toys in groups of 8 instead of all at once)
ToyGenerator(
    predict_fn=predict,
    map_fn=lambda fn: partial(jax.lax.map, fn, batch_size=8),
)

# Python loop (no JIT, useful for step-through debugging)
ToyGenerator(
    predict_fn=predict,
    map_fn=lambda fn: lambda keys: jnp.stack([fn(k) for k in keys]),
)
```

## Custom limit solver

Subclass `LimitSolver` and implement `solve(objective, level, *, key=None)`.
The objective is always called as `objective(poi, key)` and may return any
pytree of criterion values (a single CLs, a `BandValues`, a custom
container); `solve` returns the level crossing per leaf. Subclass
`StochasticLimitSolver` instead when the algorithm stays valid for noisy
criteria, where every evaluation throws fresh toys; the toy calculator only
accepts solvers under that base. See `RootFindingLimitSolver`,
`GridScanLimitSolver`, and `BisectionLimitSolver` for the built-in examples
of both kinds.
