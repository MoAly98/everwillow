# Hypothesis Testing

Everwillow provides tools for statistical hypothesis testing using the CLs method. This module computes test statistics, p-values, and upper limits with both asymptotic formulas and toy-based Monte Carlo methods.

## Quick Start

These examples show complete workflows for computing 95% CL upper limits on a parameter of interest.

::::{tab-set}
:::{tab-item} Signal+Background
```python
import jax.numpy as jnp
import everwillow as ew
import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    HypoTestCalculator,
    QTilde,
    QTildeAsymptotic,
    upper_limit,
)


# Poisson counting: n_expected = mu * s + b
def nll(params, obs):
    mu = params["mu"]
    expected = mu * 10.0 + 5.0  # s=10, b=5
    return expected - obs["n"] * jnp.log(expected)


def predict(params_state):
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * 10.0 + 5.0}


params = sl.State.from_pytree({"mu": 1.0})
observed = {"n": 12.0}

# Create calculator
calc = HypoTestCalculator(test_statistic=QTilde())
dist = QTildeAsymptotic()

# Find 95% CL upper limit on mu
limit = upper_limit(
    lambda poi: calc(
        nll, params, observed, ("mu",), poi, distribution=dist, predict_fn=predict
    ).cl_s,
    bounds=(0.0, 5.0),
    level=0.05,
)
print(f"95% CL upper limit: mu < {float(limit):.2f}")
```
:::
:::{tab-item} EFT Parameter
```python
import jax.numpy as jnp
import everwillow as ew
import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    HypoTestCalculator,
    QTilde,
    QTildeAsymptotic,
    upper_limit,
)


# EFT: cross-section scales with Wilson coefficient
def nll(params, obs):
    c = params["c"]
    xsec = 100.0 * (1.0 + 0.1 * c)  # Linear EFT approximation
    return xsec - obs["events"] * jnp.log(xsec)


def predict(params_state):
    c = params_state.to_pytree()["c"]
    return {"events": 100.0 * (1.0 + 0.1 * c)}


params = sl.State.from_pytree({"c": 0.0})
observed = {"events": 105.0}

# Create calculator
calc = HypoTestCalculator(test_statistic=QTilde())
dist = QTildeAsymptotic()

# Find 95% CL upper limit on Wilson coefficient
limit = upper_limit(
    lambda poi: calc(
        nll, params, observed, ("c",), poi, distribution=dist, predict_fn=predict
    ).cl_s,
    bounds=(0.0, 10.0),
    level=0.05,
)
print(f"95% CL upper limit: c < {float(limit):.2f}")
```
:::
::::

## Concepts

::::{grid} 1
:gutter: 3

:::{grid-item-card} Test Statistics
:class-header: bg-primary text-white

Test statistics measure compatibility between data and a hypothesis. Each computes a likelihood ratio with different boundary conditions:

| Statistic | Use Case | Boundary |
|-----------|----------|----------|
| `QTilde` | Upper limits (default) | q=0 when {math}`\hat\mu > \mu` |
| `QMu` | General hypothesis testing | None |
| `Q0` | Discovery significance | q=0 when {math}`\hat\mu < 0` |
| `TMu` | Two-sided confidence intervals | Signed |

All return a `TestStatResult` with the test statistic value and fit information.
:::

:::{grid-item-card} Distributions
:class-header: bg-info text-white

Distributions convert test statistics into p-values. Two approaches:

**Asymptotic** (fast, closed-form):
- `QTildeAsymptotic`, `QMuAsymptotic`, `Q0Asymptotic`, `TMuAsymptotic`
- Uses formulas from Cowan et al. ([arXiv:1007.1727](https://arxiv.org/abs/1007.1727))
- Requires `q_asimov` from Asimov dataset for expected bands

**Empirical** (from toys):
- `EmpiricalDistribution`
- Built from Monte Carlo pseudo-experiments
- More accurate but computationally expensive
:::

:::{grid-item-card} CLs Method
:class-header: bg-success text-white

The CLs method provides conservative exclusion limits:

- **pnull**: P-value under null hypothesis (background only)
- **palt**: P-value under alternative hypothesis (signal + background)
- **CLs = palt / pnull**: Ratio protects against excluding signal when there's no sensitivity

A hypothesis is excluded at 95% CL when CLs < 0.05.

The `HypoTestCalculator` orchestrates test statistics and distributions, returning a `HypoTestResult` with all p-values and expected bands.
:::

::::

## Computing Upper Limits

### Root-Finding Method

Use `upper_limit()` to find where CLs equals your target level:

```python
from everwillow.inference.hypotest import upper_limit

limit = upper_limit(
    lambda poi: calc(
        nll, params, obs, ("mu",), poi, distribution=dist, predict_fn=predict
    ).cl_s,
    bounds=(0.0, 10.0),
    level=0.05,  # 95% CL
)
```

### Expected Limits with Brazil Bands

Use `expected_upper_limit()` to compute observed and expected limits at {math}`\pm 1\sigma` and {math}`\pm 2\sigma`:

```python
from everwillow.inference.hypotest import expected_upper_limit

result = expected_upper_limit(
    lambda poi: calc(
        nll, params, obs, ("mu",), poi, distribution=dist, predict_fn=predict
    ),
    bounds=(0.0, 10.0),
    level=0.05,
)

print(f"Observed limit: {result.observed:.3f}")
print(f"Expected limit: {result.expected:.3f}")
print(f"-2σ to +2σ: [{result.minus_2sigma:.3f}, {result.plus_2sigma:.3f}]")
```

### Grid Scan Method

When root-finding is unstable, use `upper_limit_scan()` with linear interpolation:

```python
from everwillow.inference.hypotest import upper_limit_scan
import jax.numpy as jnp

scan_points = jnp.linspace(0.0, 10.0, 100)
limit = upper_limit_scan(
    lambda poi: calc(..., poi, ...).cl_s,
    scan_points,
    level=0.05,
)
```

## Toy-Based Analysis

For more accurate p-values, generate empirical distributions from pseudo-experiments.

### Generating Toys

```python
import jax
from everwillow.inference.hypotest import ToyGenerator, QTilde

toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=1000)

# Generate toys using predict_fn (uses Poisson sampling internally)
emp_dist = toy_gen.generate(
    nll,
    params,
    observed,
    ("mu",),
    poi_test=1.0,
    key=jax.random.key(42),
    predict_fn=predict,
)


# Or provide custom sampling function
def sample_fn(params_state, key):
    expected = predict(params_state)
    return {"n": jax.random.poisson(key, expected["n"])}


emp_dist = toy_gen.generate(
    nll,
    params,
    observed,
    ("mu",),
    poi_test=1.0,
    key=jax.random.key(42),
    sample_fn=sample_fn,
)
```

### Computing CLs with Toys

```python
# Use empirical distribution in calculator
result = calc(
    nll,
    params,
    observed,
    ("mu",),
    poi_test=1.0,
    distribution=emp_dist,
)
print(f"CLs (from toys): {result.cl_s:.4f}")
```

### Upper Limits with Toys

To find upper limits using toy-based p-values, generate toys at each POI value:

```python
from everwillow.inference.hypotest import upper_limit_toys


# Define objective that generates fresh toys for each POI
def cls_with_toys(poi, key):
    emp_dist = toy_gen.generate(
        nll,
        params,
        observed,
        ("mu",),
        poi_test=poi,
        key=key,
        predict_fn=predict,
    )
    result = calc(
        nll,
        params,
        observed,
        ("mu",),
        poi_test=poi,
        distribution=emp_dist,
    )
    return result.cl_s


# Find limit with stochastic bisection
limit = upper_limit_toys(
    cls_with_toys,
    bounds=(0.0, 5.0),
    key=jax.random.key(42),
    level=0.05,
)
print(f"95% CL upper limit (toys): mu < {float(limit):.2f}")
```

For faster computation, you can also use `upper_limit_scan` with pre-computed toy distributions at fixed POI values.

## Choosing Test Statistics

::::{tab-set}
:::{tab-item} Upper Limits
Use `QTilde` with `QTildeAsymptotic`:

```python
calc = HypoTestCalculator(test_statistic=QTilde())
result = calc(..., distribution=QTildeAsymptotic())
```

The boundary at {math}`\hat\mu > \mu` prevents upward fluctuations from weakening limits.
:::
:::{tab-item} Discovery
Use `Q0` with `Q0Asymptotic`:

```python
calc = HypoTestCalculator(test_statistic=Q0())
result = calc(..., distribution=Q0Asymptotic())

# Convert to significance
import math

significance = math.sqrt(float(result.q_obs))
```

Q0 always tests against {math}`\mu = 0` (null hypothesis = no signal).
:::
:::{tab-item} Confidence Intervals
Use `TMu` with `TMuAsymptotic`:

```python
calc = HypoTestCalculator(test_statistic=TMu())
result = calc(..., distribution=TMuAsymptotic())
```

TMu is signed, enabling two-sided confidence interval construction.
:::
::::

## Implementing Custom Test Statistics

Subclass `TestStatistic` and implement `_compute_q()`:

```python
import equinox as eqx
import jax.numpy as jnp
import everwillow as ew
from everwillow.inference.hypotest import TestStatistic, TestStatResult


class ChiSquare(TestStatistic):
    """Chi-square goodness-of-fit test statistic."""

    def _compute_q(self, nll_fn, params, observation, poi_key, poi_test, **kwargs):
        # Fit the model
        fit_result = ew.fit(nll_fn, params, observation, **kwargs)

        # Compute chi-square from observation vs prediction
        # (implementation depends on your model)
        predicted = predict_fn(params)
        residuals = observation["counts"] - predicted["counts"]
        chi2 = jnp.sum(residuals**2 / predicted["counts"])

        ndof = len(observation["counts"]) - 1

        return chi2, {
            "fit": fit_result,
            "ndof": ndof,
            "poi_test": poi_test,
        }
```

The base class handles Asimov computation automatically if you provide `predict_fn` or `asimov_observation`.

## Implementing Custom Distributions

Subclass `Distribution` and implement `pvalues()` and `expected_pvalues()`:

```python
import equinox as eqx
import jax.numpy as jnp
from jax.scipy.stats import chi2 as chi2_dist
from everwillow.inference.hypotest import Distribution, ExpectedBands


class ChiSquareDistribution(Distribution):
    """Chi-square distribution for goodness-of-fit tests."""

    ndof: int

    def pvalues(self, result):
        """Compute p-values from chi-square statistic."""
        q = result.q
        # Chi-square survival function
        pnull = 1.0 - chi2_dist.cdf(q, self.ndof)
        palt = pnull  # Same distribution for chi-square test
        return pnull, palt

    def expected_pvalues(self, result):
        """Compute expected p-values at sigma levels."""
        # For chi-square, expected q at median is ndof
        q_median = float(self.ndof)
        pnull_med = 1.0 - chi2_dist.cdf(q_median, self.ndof)

        return ExpectedBands(
            minus_2sigma=(jnp.array(pnull_med), jnp.array(pnull_med)),
            minus_1sigma=(jnp.array(pnull_med), jnp.array(pnull_med)),
            median=(jnp.array(pnull_med), jnp.array(pnull_med)),
            plus_1sigma=(jnp.array(pnull_med), jnp.array(pnull_med)),
            plus_2sigma=(jnp.array(pnull_med), jnp.array(pnull_med)),
        )
```

## API Quick Reference

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| **Test Statistics** | `QTilde`, `QMu`, `Q0`, `TMu` | Compute likelihood ratios |
| **Distributions** | `QTildeAsymptotic`, `EmpiricalDistribution`, ... | Convert to p-values |
| **Calculator** | `HypoTestCalculator` | Orchestrate tests |
| **Toys** | `ToyGenerator` | Monte Carlo sampling |
| **Limits** | `upper_limit`, `expected_upper_limit` | Find exclusion limits |
| **Results** | `HypoTestResult`, `TestStatResult`, `ExpectedBands` | Output containers |

See the {doc}`api/inference/hypotest` for complete documentation.

## Next Steps

::::{grid} 1
:gutter: 2

:::{grid-item-card} 📚 API Reference
:link: api/inference/hypotest
:link-type: doc

Complete documentation of all classes and functions
:::

:::{grid-item-card} 🔬 Fitting Guide
:link: introduction
:link-type: doc

Learn about parameter fitting before hypothesis testing
:::

:::{grid-item-card} 🧮 State Management
:link: statelib_overview
:link-type: doc

Understand parameter handling with statelib
:::

::::
