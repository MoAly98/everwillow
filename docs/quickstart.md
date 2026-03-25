# Quickstart

Everwillow is a inference-only library for statistical measurements performed in HEP.
It is built on JAX and focusses on strong interopability with JAX transformations to allow auto-differentiation, JIT-compilation and vectorization for any inference step.

## Fitting

The main entry point to everwillow is having a negative log-likelihood function with signature `nll(params, observation)` compatible with JAX, see the following example of a gaussian fit:

```python
import jax
import jax.numpy as jnp

import optimistix as optx

import everwillow as ew
import everwillow.statelib as sl

jax.config.update("jax_enable_x64", True)

# generate 1M data points for a Gaussian(mean=0.4, sigma=0.4)
key = jax.random.PRNGKey(0)
true_loc, true_scale = 0.4, 0.4
data = jax.random.normal(key, (1_000_000,)) * true_scale + true_loc

# initial set of parameters (will be fitted by everwillow)
init_params = {"loc": 0.0, "scale": 1.0}


def neg_log_likelihood(params, observation):
    logpdf_vals = jax.scipy.stats.norm.logpdf(observation, **params)
    return -jnp.sum(logpdf_vals)


result = ew.fit(
    nll_fn=neg_log_likelihood,
    params=sl.State.from_pytree(init_params),
    observation=data,
    solver=optx.BFGS(rtol=1e-6, atol=1e-6),
    max_steps=1_000,
)

# make sure the solver converged
assert result.success

print(result.params.to_pytree())
# {
#   'loc': Array(0.39995897, dtype=float64),
#   'scale': Array(0.40000754, dtype=float64),
# }
```

### Fixing parameters

To hold parameters constant during a fit, pass a `State` with `...` (Ellipsis) values via the `fixed` argument. For example, fix `scale` at its initial value and only fit `loc`:

```python
result_fixed = ew.fit(
    nll_fn=neg_log_likelihood,
    params=sl.State.from_pytree(init_params),
    observation=data,
    fixed=sl.State.from_pytree({"scale": ...}),
    solver=optx.BFGS(rtol=1e-6, atol=1e-6),
    max_steps=1_000,
)

print(result_fixed.params.to_pytree())
# {
#   'loc': Array(0.39995897, dtype=float64),
#   'scale': 1.0,  # frozen at initial value
# }
```

### Uncertainties

After fitting, extract parameter uncertainties from the inverse Hessian of the NLL:

```python
from everwillow.uncertainty import uncertainties, covariance_matrix, correlation_matrix

# Parameter uncertainties: σ_i = √((H⁻¹)_ii)
unc = uncertainties(neg_log_likelihood, result.params, data)
print(unc.to_pytree())
# {
#   'loc': Array(0.00040001, dtype=float64),
#   'scale': Array(0.00028285, dtype=float64),
# }

# Full covariance matrix
cov = covariance_matrix(neg_log_likelihood, result.params, data)
print(cov)
# [[ 1.6001e-07,  1.2164e-16],
#  [ 1.2164e-16,  8.0003e-08]]

# Correlation matrix (normalized covariance, diagonal = 1)
corr = correlation_matrix(neg_log_likelihood, result.params, data)
print(corr)
# [[ 1.0000e+00,  1.0751e-09],
#  [ 1.0751e-09,  1.0000e+00]]
```

## Hypothesis Testing

The same `nll(params, observation)` interface extends to hypothesis testing. Here is a Poisson counting experiment that computes a 95% CL upper limit on a signal strength parameter:

```python
import jax
import jax.numpy as jnp

import everwillow.statelib as sl
from everwillow.hypotest.calculators import AsymptoticCalculator
from everwillow.hypotest.distributions import QTildeAsymptotic
from everwillow.hypotest.test_statistics import QTilde

jax.config.update("jax_enable_x64", True)

# Poisson counting experiment: n_expected = mu * signal + background
signal, background = 10.0, 5.0


def nll(params, observation):
    """Poisson negative log-likelihood."""
    mu = params["mu"]
    expected = mu * signal + background
    return expected - observation["n"] * jnp.log(expected)


def predict(params_state):
    """Expected observation for a given parameter state (used for Asimov dataset)."""
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * signal + background}


params = sl.State.from_pytree({"mu": 1.0})
observed = {"n": 12.0}

# AsymptoticCalculator binds the model and provides predict_fn for Asimov
# dataset generation. QTilde + QTildeAsymptotic are the defaults.
calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    predict_fn=predict,
    test_statistic=QTilde(),
    distribution=QTildeAsymptotic(),
)

# Run hypothesis test at mu=1
result = calc.test(1.0)

print(f"Test statistic: {result.q_obs:.4f}")
print(f"Null p-value:   {result.pnull:.6f}")
print(f"Alt p-value:    {result.palt:.6f}")
print(f"CLs:            {calc.cls(result):.6f}")
# Test statistic: 0.6446
# Null p-value:   0.211033
# Alt p-value:    0.986078
# CLs:            0.214013

# Expected CLs at standard sigma bands (from Asimov dataset).
# bands.cl_s is a BandValues — iterable as (name, value) pairs.
bands = calc.expected(result)
for name, val in bands.cl_s:
    print(f"  {name}: {float(val):.6f}")
# minus_2sigma: 0.000012
# minus_1sigma: 0.000198
# median:       0.002679
# plus_1sigma:  0.026892
# plus_2sigma:  0.161777
```

### Extra NLL arguments via `functools.partial`

If your NLL function takes extra arguments beyond `(params, observation)`, use
`functools.partial` to bind them before passing to `fit()`:

```python
from functools import partial


def nll_with_config(params, observation, signal, background):
    mu = params["mu"]
    expected = mu * signal + background
    return expected - observation["n"] * jnp.log(expected)


# Bind signal and background, leaving (params, observation) free
nll_fn = partial(nll_with_config, signal=10.0, background=5.0)

result = ew.fit(
    nll_fn=nll_fn,
    params=sl.State.from_pytree({"mu": 1.0}),
    observation={"n": 12.0},
)
```

Hypothesis testing in everwillow is built from three composable pieces:

- **Test statistic**  - computes a scalar from the NLL and data. `QTilde` (default) and `QMu` are one-sided for upper limits, `Q0` is for discovery, `TMu` is two-sided for intervals.
- **Distribution**  - converts the test statistic into p-values. Asymptotic distributions (`QTildeAsymptotic`, `QMuAsymptotic`, etc.) use the Cowan et al. formulas. `SimpleEmpiricalDistribution` uses toys.
- **Calculator**  - binds the model (NLL, parameters, data) and orchestrates the test. `AsymptoticCalculator` extends `HypoTestCalculator` with Asimov dataset generation via `predict_fn`.

### Toy-based p-values

Replace asymptotic distributions with toys by generating pseudo-experiments with `ToyGenerator`:

```python
import jax

from everwillow.hypotest.calculators import HypoTestCalculator
from everwillow.hypotest.distributions import SimpleEmpiricalDistribution
from everwillow.hypotest.test_statistics import QTilde
from everwillow.hypotest.toys import ToyGenerator

# Generate toys under null (poi_null=1.0) and alternative (poi_alt=0.0) hypotheses.
# poi_null is the hypothesis under which null toys are generated.
# poi_alt is the hypothesis under which alternative toys are generated.
# Toys are generated in a vectorised manner using jax.vmap by default.
toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=5000)
toys = toy_gen.generate(
    nll,
    params,
    observed,
    "mu",
    poi_null=1.0,
    poi_alt=0.0,
    key=jax.random.key(42),
    predict_fn=predict,
)

# Build empirical distribution from toys
dist = SimpleEmpiricalDistribution.from_toys(toys)

# Use with calculator for p-values.
# calc.test(poi_value) evaluates the test statistic at poi_value using
# the distribution built from toys — this should match poi_null for
# the distribution to be valid.
toy_calc = HypoTestCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    test_statistic=QTilde(),
    distribution=dist,
)
result = toy_calc.test(1.0)
print(f"CLs (toy): {toy_calc.cls(result):.4f}")
```

### Upper limits

All upper limit functions take a user-composed objective and find where it
crosses a target level. They work with any calculator — asymptotic or toy-based.

**`upper_limit`** — determine limit by using a bisection root-finding algorithm.
This is the standard choice for asymptotic distributions and cases where the objective is a
deterministic function of POI:

```python
from everwillow.hypotest.upper_limit import upper_limit


# Compose a poi -> scalar objective from the calculator:
def cls_objective(poi):
    return calc.cls(calc.test(poi))


# solve for objective = level
limit = upper_limit(cls_objective, bounds=(0.0, 5.0), level=0.05)
print(f"95% CL upper limit: {float(limit):.4f}")
# 95% CL upper limit: 1.3673
```

**`upper_limit_scan`** — evaluates the objective on a grid and interpolates.
Useful when you already know a narrow region around which the limit sits.

```python
from everwillow.hypotest.upper_limit import upper_limit_scan

scan = jnp.linspace(0.0, 3.0, 100)
limit_scan = upper_limit_scan(cls_objective, scan, level=0.05)
print(f"95% CL upper limit (scan): {float(limit_scan):.4f}")
```

**`upper_limit_toys`** — stochastic bisection for objectives that regenerate
toys at each POI evaluation. Takes `(poi, key)` and uses a fresh PRNG key per
iteration:

```python
from everwillow.hypotest.upper_limit import upper_limit_toys


def stochastic_cls(poi, key):
    """Regenerate toys at each POI value during the search."""
    toys = toy_gen.generate(
        nll,
        params,
        observed,
        "mu",
        poi_null=poi,
        poi_alt=0.0,
        key=key,
        predict_fn=predict,
    )
    dist = SimpleEmpiricalDistribution.from_toys(toys)
    toy_calc = HypoTestCalculator(
        nll_fn=nll,
        params=params,
        observation=observed,
        poi_key="mu",
        test_statistic=QTilde(),
        distribution=dist,
    )
    return toy_calc.cls(toy_calc.test(poi))


limit_toys = upper_limit_toys(
    stochastic_cls, bounds=(0.0, 5.0), key=jax.random.key(0), tol=0.01
)
```

**`expected_upper_limit`** — finds expected limits at each sigma band. Takes a
`poi -> BandValues` objective and calls `upper_limit` five times (once per
band):

```python
from everwillow.hypotest.upper_limit import expected_upper_limit


def band_cls_objective(poi):
    result = calc.test(poi)
    bands = calc.expected(result)
    return bands.cl_s


# Returns BandValues — each entry is the upper limit at that sigma band.
brazil = expected_upper_limit(band_cls_objective, bounds=(0.0, 5.0), level=0.05)

for name, val in brazil:
    print(f"  {name}: {float(val):.4f}")
# minus_2sigma: 0.2734
# minus_1sigma: 0.3854
# median:       0.5746
# plus_1sigma:  0.8792
# plus_2sigma:  1.3121
```
