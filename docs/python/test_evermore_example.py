"""Standalone evermore counting experiment example."""

import equinox as eqx
import evermore as evm
import jax.numpy as jnp

import everwillow as ew


# Define parameter structure
class Params(eqx.Module):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter


# Initialize parameters
params = Params(
    mu=evm.Parameter(value=1.0, name="mu"),
    norm1=evm.NormalParameter(value=0.0, name="norm1"),
    norm2=evm.NormalParameter(value=0.0, name="norm2"),
    shape1=evm.NormalParameter(value=0.0, name="shape1"),
)

# Define histogram templates
hists = {
    "nominal": {
        "signal": jnp.array([3.0]),
        "bkg1": jnp.array([10.0]),
        "bkg2": jnp.array([20.0]),
    },
    "shape_up": {
        "bkg1": jnp.array([12.0]),
        "bkg2": jnp.array([23.0]),
    },
    "shape_down": {
        "bkg1": jnp.array([8.0]),
        "bkg2": jnp.array([17.0]),
    },
}

observation = jnp.array([37.0])


# Build the model
def model(params_tree):
    expectations = {}

    sig_mod = params_tree.mu.scale()
    expectations["signal"] = sig_mod(hists["nominal"]["signal"])

    bkg1_lnN = params_tree.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
    bkg1_shape = params_tree.shape1.morphing(
        up_template=hists["shape_up"]["bkg1"],
        down_template=hists["shape_down"]["bkg1"],
    )
    expectations["bkg1"] = (bkg1_lnN @ bkg1_shape)(hists["nominal"]["bkg1"])

    bkg2_lnN = params_tree.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
    bkg2_shape = params_tree.shape1.morphing(
        up_template=hists["shape_up"]["bkg2"],
        down_template=hists["shape_down"]["bkg2"],
    )
    expectations["bkg2"] = (bkg2_lnN @ bkg2_shape)(hists["nominal"]["bkg2"])

    return expectations


# Partition parameters
dynamic, static = evm.tree.partition(params)


# Define loss function
@eqx.filter_jit
def loss(dynamic_params):
    full_params = evm.tree.combine(dynamic_params, static)
    expectations = model(full_params)
    constraints = evm.loss.get_log_probs(full_params)
    log_prob = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    log_prob += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_prob)


# Convert to dict-based NLL for everwillow
def nll(param_dict):
    updated = evm.tree.update_values(
        dynamic,
        values=Params(
            mu=param_dict["mu"],
            norm1=param_dict["norm1"],
            norm2=param_dict["norm2"],
            shape1=param_dict["shape1"],
        ),
    )
    return loss(updated)


# Extract initial values
initial = {
    "mu": float(params.mu.value),
    "norm1": float(params.norm1.value),
    "norm2": float(params.norm2.value),
    "shape1": float(params.shape1.value),
}

# Perform the fit
result = ew.fit(nll, initial)

print(result.params)
