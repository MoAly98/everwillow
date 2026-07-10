# Tips

## JIT compilation

All everwillow functions are compatible with `jax.jit`. Wrapping your workflow
in JIT compiles it into optimized XLA code, which can significantly speed up
repeated evaluations (e.g. scanning over POI values for limits):

```python
import jax

from everwillow.hypotest.limit_solvers import RootFindingLimitSolver


@jax.jit
def compute_limit(observed_n):
    observed = {"n": observed_n}
    calc = AsymptoticCalculator(
        nll_fn=nll,
        params=params,
        observation=observed,
        poi_key="mu",
        predict_fn=predict,
        limit_solver=RootFindingLimitSolver(bounds=(0.0, 5.0)),
    )
    return calc.upper_limit(level=0.05)


# First call traces and compiles; subsequent calls are fast
limit = compute_limit(12.0)
```

JIT compilation works best when the function structure is fixed and only array
values change. Avoid passing Python objects that change shape or structure
between calls.

## 64-bit precision

JAX defaults to 32-bit floats. For statistical inference, 64-bit precision is
recommended to avoid numerical issues in likelihood evaluations:

```python
jax.config.update("jax_enable_x64", True)
```

Set this at the top of your script, before any JAX operations.
