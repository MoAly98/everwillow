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
bands = calc.pvalue_bands(result)
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

Hypothesis testing in everwillow is built from four composable pieces:

- **Test statistic**  - computes a scalar from the NLL and data. `QTilde` (default) and `QMu` are one-sided for upper limits, `Q0` is for discovery, `TMu` is two-sided for intervals.
- **Distribution**  - converts the test statistic into p-values. Asymptotic distributions (`QTildeAsymptotic`, `QMuAsymptotic`, etc.) use the Cowan et al. formulas. `SimpleEmpiricalDistribution` uses toys.
- **Calculator**  - binds the model (NLL, parameters, data) and orchestrates the test. `HypoTestCalculator` is the abstract core; `AsymptoticCalculator` computes p-values from a fixed distribution and can generate the Asimov dataset via `predict_fn`; `ToyCalculator` regenerates toy ensembles at every tested POI.
- **Limit solver**  - locates where a criterion crosses the target level: `RootFindingLimitSolver` (adaptive, deterministic criteria), `GridScanLimitSolver` (fixed grid, all bands in one pass), `BisectionLimitSolver` (stepped bisection with fresh toys per step).

### Toy-based p-values

`ToyCalculator` throws pseudo-experiments (toys) at every tested POI and builds
the p-values from the resulting empirical distribution:

```python
import jax

from everwillow.hypotest.calculators import ToyCalculator
from everwillow.hypotest.toys import ToyGenerator

# The generator holds only the sampling configuration. With predict_fn set,
# each toy Poisson-fluctuates the predicted yields; pass sample_fn instead
# for any other sampling scheme. Toys are vectorised with jax.vmap by default.
toy_gen = ToyGenerator(predict_fn=predict, ntoys=5000)

toy_calc = ToyCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    toy_generator=toy_gen,
    poi_alt=0.0,  # background-only alternative ensembles, needed for CLs
    key=jax.random.key(42),  # one key is enough for a whole analysis
)

# Toys are thrown at poi=1.0 under both hypotheses; p-values come from tail
# counting by default (swap distribution_factory for e.g. smoothed variants).
result = toy_calc.test(1.0)
print(f"CLs (toy): {toy_calc.cls(result):.4f}")
```

The calculator is a pure function of its inputs: rerunning with the same key
gives identical ensembles, and a per-call ``key=`` override draws an
independent replica. If you have toys generated once externally, wrap them as
a fixed distribution on the asymptotic calculator instead:
``AsymptoticCalculator(..., distribution=SimpleEmpiricalDistribution.from_toys(toys))``.

### Upper limits

The calculator computes limits: `upper_limit()` finds the POI value where CLs
crosses the target level, and `upper_limit_bands()` gives the expected
(Brazil-band) limits. A `LimitSolver` decides how the crossing is located;
set it once on the calculator.

```python
from everwillow.hypotest.limit_solvers import RootFindingLimitSolver

calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    predict_fn=predict,
    limit_solver=RootFindingLimitSolver(bounds=(0.0, 5.0)),
)

limit = calc.upper_limit(level=0.05)
print(f"95% CL upper limit: {float(limit):.4f}")
# 95% CL upper limit: 1.3673

brazil = calc.upper_limit_bands(level=0.05)
for name, val in brazil:
    print(f"  {name}: {float(val):.4f}")
# minus_2sigma: 0.2734
# minus_1sigma: 0.3854
# median:       0.5746
# plus_1sigma:  0.8792
# plus_2sigma:  1.3121
```

For a blind analysis, the expected limit is the observed limit computed on
the Asimov dataset:

```python
asimov_data = predict(sl.State.from_pytree({"mu": 0.0}))
expected_calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=asimov_data,
    poi_key="mu",
    predict_fn=predict,
    limit_solver=RootFindingLimitSolver(bounds=(0.0, 5.0)),
)
expected_limit = expected_calc.upper_limit(level=0.05)
```

### Toy-based upper limits

`ToyCalculator` throws fresh toys at every POI the solver evaluates, so the
empirical distribution always matches the tested hypothesis. The `key` seeds
all toy throws: one key covers the whole analysis, rerunning with the same
key reproduces it exactly, and a different key draws an independent replica.

The standard workflow is a grid scan; the observed limit and all bands come
from a single pass over the grid.

```python
from everwillow.hypotest.limit_solvers import BisectionLimitSolver, GridScanLimitSolver

# toy_gen is the ToyGenerator from the toy-based p-values section above.
toy_calc = ToyCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    toy_generator=toy_gen,
    key=jax.random.key(42),
    limit_solver=GridScanLimitSolver(scan=jnp.linspace(0.01, 3.0, 40)),
)

toy_limit = toy_calc.upper_limit(level=0.05)
toy_brazil = toy_calc.upper_limit_bands(level=0.05)

# Solvers can also be swapped per call. A bisection search narrows in on the
# crossing instead of scanning a whole grid; tol stops it once the criterion
# is within the Monte Carlo precision of the toys.
limit_bisect = toy_calc.upper_limit(
    BisectionLimitSolver(bounds=(0.01, 3.0), tol=0.01),
    level=0.05,
)
```
