# Features

## Interactive fitting

`ifit` works like `fit` but adds a live progress bar showing the current NLL at each step. It also accepts callbacks and a checkpoint manager for saving and resuming fits.

```python
import everwillow as ew
import everwillow.statelib as sl

result = ew.ifit(nll_fn, params, observation, max_steps=200)
```

Because `ifit` uses a Python loop with side effects, it is not JIT-compatible. Use `fit` when you need JIT compilation.

## Callbacks

Pass a list of callbacks to `ifit` to run custom logic at each iteration. A callback is any callable with signature `(iteration: int, y: State, state: SolverState) -> None`. The current NLL is available via `state.f_info.f`.

The built-in `HistoryCallback` records step indices and NLL values for plotting convergence:

```python
from everwillow.callback import HistoryCallback

history = HistoryCallback()
result = ew.ifit(nll_fn, params, observation, callbacks=[history])

# Plot convergence
import matplotlib.pyplot as plt

plt.plot(history.steps, history.nlls)
plt.xlabel("Step")
plt.ylabel("NLL")
```

Any callable matching the signature works as a custom callback. For example, to print the NLL every 10 steps:

```python
def print_every_10(iteration, y, state):
    if iteration % 10 == 0:
        print(f"Step {iteration}: NLL = {state.f_info.f:.4f}")


result = ew.ifit(nll_fn, params, observation, callbacks=[history, print_every_10])
```

## Checkpointing

`ifit` integrates with [orbax](https://orbax.readthedocs.io/) for checkpointing. Pass a `CheckpointManager` and the fit state is saved at each step. Reusing the same manager automatically resumes from the last saved step:

```python
import orbax.checkpoint as ocp

mngr = ocp.CheckpointManager("/tmp/my_fit")

# Run for 50 steps
result = ew.ifit(nll_fn, params, observation, max_steps=50, checkpoint_manager=mngr)
mngr.wait_until_finished()  # checkpointing is async

# Resume and run to 100 steps total
result = ew.ifit(nll_fn, params, observation, max_steps=100, checkpoint_manager=mngr)
mngr.wait_until_finished()
```

## State visualization

Call `show()` on any `State` to get a pretty-printed view with rich array formatting. In Jupyter notebooks this produces an interactive, foldable display; in terminals it prints formatted text:

```python
import jax.numpy as jnp
import everwillow.statelib as sl

state = sl.State.from_pytree({"mu": 1.0, "sigma": jnp.array([0.1, 0.2])})
state.show()
# State({
#   'mu': 1.0,
#   'sigma': <jax.Array([0.1, 0.2], dtype=float32)>,
# })
```
