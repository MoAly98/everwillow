# Quickstart

Everwillow is a inference-only library for statistical measurements performed in HEP.
It is build on JAX and focusses on strong interopability with JAX transformations to allow auto-differentiation, JIT-compilation, vectorization for any inference step.

The main entry point to everwillow is having a (potentially unnormalized) log-probability density function compatible with JAX, see the following example of two stacked gaussians:

```python
from functools import partial

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


def neg_log_likelihood(params, data):
    logpdf_vals = jax.scipy.stats.norm.logpdf(data, **params)
    return -jnp.sum(logpdf_vals)


result = ew.fit(
    nll_fn=partial(neg_log_likelihood, data=data),
    params=sl.State.from_pytree(init_params),
    solver=optx.BFGS(rtol=1e-6, atol=1e-6),
    max_steps=1_000,
)

# make sure the solver converged
assert result.success

print(result.params)
# {'loc': Array(0.39995897, dtype=float64), 'scale': Array(0.40000754, dtype=float64)}
```
