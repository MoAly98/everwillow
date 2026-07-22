# Multiple Parameters of Interest

A model can carry any number of parameters of interest: several signal
strengths, EFT Wilson coefficients, or a signal strength per truth bin. This
guide covers how to test them jointly and which statistical quantities each
setup produces.

## POI points

The tested hypothesis is a *POI point*: a mapping from parameter key to value.
A single-POI test uses a one-entry mapping, and a joint test names every POI:

```python
calc.test(1.0)  # scalar, resolved via poi_key
calc.test({"mu": 1.0})  # the same point, spelled out
calc.test({"mu_a": 1.0, "mu_b": 0.5})  # joint two-POI test
```

## Working example

Two counting channels, each measuring its own signal strength:

```python
import jax.numpy as jnp

import everwillow.statelib as sl
from everwillow.hypotest.calculators import AsymptoticCalculator
from everwillow.hypotest.distributions import TMuAsymptotic
from everwillow.hypotest.test_statistics import TMu

S, B = 10.0, 5.0


def nll(params, observation):
    exp_a = params["mu_a"] * S + B
    exp_b = params["mu_b"] * S + B
    return (exp_a - observation["n_a"] * jnp.log(exp_a)) + (
        exp_b - observation["n_b"] * jnp.log(exp_b)
    )


def predict(state):
    tree = state.to_pytree()
    return {"n_a": tree["mu_a"] * S + B, "n_b": tree["mu_b"] * S + B}


params = sl.State.from_pytree({"mu_a": 1.0, "mu_b": 1.0})
observed = {"n_a": 10.0, "n_b": 25.0}
```

## Which statistical quantity, which tool

A many-POI model does not change what gets reported. In practice the results
stay at most two-dimensional:

- **Per-POI limit**: an upper limit on one POI with the others profiled. It
  uses the 1-D limit solvers with any criterion. CLs is the default, and
  criteria such as the null p-value or a user-supplied one work the same way.
- **Joint p-value**: the probability of the observed data under a specific
  point in POI space, from `TMu`.
- **Joint confidence region**: the set of POI points not excluded at a given
  confidence level, usually drawn as a 2-D contour.

One caveat on criteria: CLs itself is a single-POI construction. It is built
on the one-sided test statistic, which orders on a scalar, so there is no CLs
region. Joint regions use the profile-likelihood criterion described below.

## Which POI is the limit on?

The limit solvers walk a single POI axis, so they need to know which
parameter that axis belongs to. The rule: a full point mapping never needs a
key, a bare scalar resolves the per-call `poi_key` first and then the field,
and neither set is an error. Three usage patterns follow:

```python
# Single-POI analysis: set the field once.
calc = AsymptoticCalculator(..., poi_key="mu_a")
calc.test(1.0)
calc.upper_limit(solver)  # limit on mu_a, mu_b profiled

# Multi-POI model, per-POI limits: pick the target per call. The search
# geometry usually changes with the target, so pass the solver alongside.
calc.upper_limit(BisectionLimitSolver(bounds=(0.0, 5.0)), poi_key="mu_a")
calc.upper_limit(BisectionLimitSolver(bounds=(0.0, 12.0)), poi_key="mu_b")

# Joint-only work: no poi_key needed at all.
joint = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    test_statistic=TMu(),
    distribution=TMuAsymptotic(dof=2),
    predict_fn=predict,
)
joint.test({"mu_a": 1.0, "mu_b": 0.5})
```

In a per-POI limit the other POIs are free parameters, so every fit profiles
them like nuisance parameters. To pin them instead, fix them per call:

```python
calc.upper_limit(solver, fit_kwargs={"fixed": sl.State.from_pytree({"mu_b": 1.0})})
```

## Joint tests

`TMu` is the joint test statistic. The constrained fit fixes every POI in the
tested point, and `-2 ln λ` follows a chi-square with one degree of freedom
per POI (Cowan et al., arXiv:1007.1727, Eq. 21). Pair it with
`TMuAsymptotic(dof=k)`:

```python
calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    test_statistic=TMu(),
    distribution=TMuAsymptotic(dof=2),
    predict_fn=predict,
)

result = calc.test({"mu_a": 1.0, "mu_b": 1.0})
result.pnull  # chi-square_2 tail of the joint t
```

The one-sided statistics (`QMu`, `QTilde`, `Q0`) raise on multi-POI points.
They are defined through a scalar ordering and have no joint form.

## Confidence regions

`confidence_region` evaluates a criterion over a set of hypothesis points and
returns the field for you to contour. The default criterion is the null
p-value, so with the setup above the region is the standard chi-square
region: a point is inside at 95% CL when `pnull >= 0.05`, equivalently
`t <= 5.99` for two POIs.

```python
grid_a = jnp.linspace(-0.2, 1.8, 41)
grid_b = jnp.linspace(0.6, 3.4, 41)
points = [{"mu_a": a, "mu_b": b} for a in grid_a for b in grid_b]

region = calc.confidence_region(points, level=0.05)
region.values  # pnull per point
region.inside  # membership mask at the given level

field = region.values.reshape(len(grid_a), len(grid_b))
plt.contour(grid_a, grid_b, field.T, levels=[0.05])  # the 95% contour
```

Keep the scanned grid inside the physical boundaries of the model (expected
yields must stay positive). Points outside evaluate to NaN.

The scan is one batched evaluation (`jax.vmap` by default). For large grids
where memory is tight, or for step-through debugging, swap the mapping
strategy exactly as with `ToyGenerator.map_fn`:

```python
from functools import partial
import jax

region = calc.confidence_region(
    points, map_fn=lambda fn: partial(jax.lax.map, fn, batch_size=64)
)
```

## Toy-based regions

`ToyCalculator.confidence_region` regenerates the toy ensembles at every
point, threading an independent subkey into each. The whole scan is
reproducible from the calculator's key:

```python
region = toy_calc.confidence_region(points, criterion=lambda r: r.pnull)
```

The empirical distributions only tail-count the scalar test statistic, so
they work at any number of POIs and make no asymptotic assumption. A scalar
`poi_alt` on the calculator broadcasts over the tested POIs (0.0 means
background-only in every one).

## Array-valued parameters

`State` leaves can be arrays, so a point may hold a vector under a single
key. This fits models whose parameter is naturally a vector: a signal
strength per truth bin in unfolding, or an EFT Wilson-coefficient vector the
model contracts as `c @ M @ c`:

```python
calc.test({"c": jnp.array([0.1, -0.3])})  # a 2-POI joint test
```

The constrained fit fixes the whole array, so each array component counts as
one constrained degree of freedom. The point above needs
`TMuAsymptotic(dof=2)`, exactly as two named scalars would. The leaf is
atomic: to fix one component while floating another, model them as separate
scalar parameters.

## Thresholds

Region membership derives from the chi-square quantile with `dof` equal to
the number of POIs, never from a hardcoded number. For reference (PDG
Statistics review, Table 40.2), thresholds on `t = -2 Δln L`:

| CL | 1 POI | 2 POIs | 3 POIs |
|---|---|---|---|
| 68.27% | 1.00 | 2.30 | 3.53 |
| 95% | 3.84 | 5.99 | 7.81 |

`level` on `confidence_region` states the same cut on the p-value: at 95% CL,
`level = 0.05`.
