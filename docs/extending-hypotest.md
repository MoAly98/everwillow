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

```python
import jax.numpy as jnp

import everwillow as ew
import everwillow.statelib as sl
from everwillow.hypotest.test_statistics import TestStatistic
from everwillow.hypotest.utils import constrained_fit


class SignedQMu(TestStatistic):
    """Signed q_mu: negative when mu_hat > mu_test."""

    def _compute(self, nll_fn, params, observation, poi_key, poi_test, **kw):
        fit_free = ew.fit(nll_fn, params, observation, **kw)
        mu_hat = fit_free.params[poi_key]

        fixed = sl.State.from_pytree({poi_key: poi_test})
        fit_cond = constrained_fit(nll_fn, params, observation, fixed, **kw)

        q = 2.0 * (fit_cond.nll - fit_free.nll)
        q_signed = jnp.sign(poi_test - mu_hat) * q

        return q_signed, {"mu_hat": mu_hat}
```

For Cowan-style test statistics that need an Asimov dataset for asymptotic
p-values, subclass `CowanTestStatistic` instead. This class adds automatic Asimov
generation via `predict_fn` and populates `q_asimov` on the result.

## Custom distribution

Subclass `Distribution` and implement `null_pval` and `alt_pval`. You get
`null_significance`, `alt_significance`, and `expected_pvalues` for free:

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

`HypoTestCalculator` binds the model (NLL, parameters, data) and wires a test
statistic to a distribution. It exposes `test(poi)`, `cls(result)`, and
`expected(result)`. For most use cases it can be used directly — subclassing is
only needed to inject extra state into `test()`:

```python
from everwillow.hypotest.calculators import HypoTestCalculator
from everwillow.hypotest.test_statistics import QTilde

calc = HypoTestCalculator(
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

`AsymptoticCalculator` is one such subclass: it adds `predict_fn` and
`mu_asimov` fields and injects them into each `test()` call so that
`CowanTestStatistic` subclasses can generate the Asimov dataset automatically.
The same pattern works for any calculator that needs to forward extra context
to the test statistic.

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
    return {"n": expected + jax.random.normal(key) * jnp.sqrt(expected)}


toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=10_000)
toys = toy_gen.generate(
    nll,
    params,
    observed,
    "mu",
    poi_null=1.0,
    poi_alt=0.0,
    key=jax.random.key(0),
    sample_fn=sample_fn,
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
    test_statistic=QTilde(),
    map_fn=lambda fn: partial(jax.lax.map, fn, batch_size=8),
)

# Python loop (no JIT, useful for step-through debugging)
ToyGenerator(
    test_statistic=QTilde(),
    map_fn=lambda fn: lambda keys: jnp.stack([fn(k) for k in keys]),
)
```
