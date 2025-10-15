"""Test everwillow with pyhs3 model matching the evermore example structure."""

from collections.abc import Callable
import rich
import jax
import jax.numpy as jnp

import pyhs3
from pyhs3.typing.aliases import TensorVar
from pyhs3.distributions import PoissonDist, GaussianDist, ProductDist
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.metadata import Metadata
from pyhs3.functions import GenericFunction
from pyhs3.data import PointData
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew
import evermore as evm


def get_jaxified_graph(
    model: pyhs3.Model,
    name: str,
) -> tuple[list[TensorVar], Callable]:
    """Convert PyTensor graph to JAX function."""
    dist = model.distributions[name]
    inputs = [var for var in graph_inputs([dist]) if var.name is not None]
    fgraph = FunctionGraph(inputs=inputs, outputs=[dist], clone=True)
    mode.JAX.optimizer.rewrite(fgraph)
    return inputs, jax_funcify(fgraph)


# ===== Evermore model structure: =====
# - Signal: mu * nominal
# - Bkg1: norm1 * shape1 * nominal (with lnN and shape uncertainties)
# - Bkg2: norm2 * shape1 * nominal (with lnN and shape uncertainties)
# - Total: signal + bkg1 + bkg2
# - Observation: 37
# - Constraints: norm1, norm2, shape1 are constrained nuisance parameters

rich.print("="*60)
rich.print("Building pyhs3 model (evermore-style)")
rich.print("="*60)

# Create metadata
metadata = Metadata(hs3_version="0.2")

# ===== Create distributions =====

# Main Poisson for the observation
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",
    mean="n_expected"
)

# Gaussian constraints for nuisance parameters
norm1_constraint = GaussianDist(
    name="norm1_constraint",
    x="a_norm1",
    mean="norm1",
    sigma=1.0
)

norm2_constraint = GaussianDist(
    name="norm2_constraint",
    x="a_norm2",
    mean="norm2",
    sigma=1.0
)

shape1_constraint = GaussianDist(
    name="shape1_constraint",
    x="a_shape1",
    mean="shape1",
    sigma=1.0
)

# Combined likelihood
combined_config = {
    "type": "product_dist",
    "name": "model",
    "factors": ["main_poisson", "norm1_constraint", "norm2_constraint", "shape1_constraint"]
}
combined = ProductDist(**combined_config)

# ===== Define functions for computed values =====

# Signal expectation: mu * signal_nominal
signal_expected_config = {
    "type": "generic_function",
    "name": "signal_expected",
    "expression": "mu * signal_nominal"
}
signal_expected_func = GenericFunction(**signal_expected_config)

# Bkg1 with log-normal and shape systematic
# lnN: exp(norm1 * log(kappa)) where kappa = 1.1 / 0.9 = 1.222 (symmetric log-normal)
# shape: linear interpolation: nominal + shape1 * (up - nominal)
bkg1_lnN_config = {
    "type": "generic_function",
    "name": "bkg1_lnN_factor",
    "expression": "(1.0 + norm1 * 0.1)"  # Simplified: 1 + norm1 * (up - 1)
}
bkg1_lnN_func = GenericFunction(**bkg1_lnN_config)

bkg1_shape_interp_config = {
    "type": "generic_function",
    "name": "bkg1_shape_interp",
    "expression": "bkg1_nominal + shape1 * (bkg1_shape_up - bkg1_nominal)"
}
bkg1_shape_interp_func = GenericFunction(**bkg1_shape_interp_config)

bkg1_expected_config = {
    "type": "generic_function",
    "name": "bkg1_expected",
    "expression": "bkg1_lnN_factor * bkg1_shape_interp"
}
bkg1_expected_func = GenericFunction(**bkg1_expected_config)

# Bkg2 with log-normal and shape systematic
bkg2_lnN_config = {
    "type": "generic_function",
    "name": "bkg2_lnN_factor",
    "expression": "(1.0 + norm2 * 0.05)"  # Simplified
}
bkg2_lnN_func = GenericFunction(**bkg2_lnN_config)

bkg2_shape_interp_config = {
    "type": "generic_function",
    "name": "bkg2_shape_interp",
    "expression": "bkg2_nominal + shape1 * (bkg2_shape_up - bkg2_nominal)"
}
bkg2_shape_interp_func = GenericFunction(**bkg2_shape_interp_config)

bkg2_expected_config = {
    "type": "generic_function",
    "name": "bkg2_expected",
    "expression": "bkg2_lnN_factor * bkg2_shape_interp"
}
bkg2_expected_func = GenericFunction(**bkg2_expected_config)

# Total expected
n_expected_config = {
    "type": "generic_function",
    "name": "n_expected",
    "expression": "signal_expected + bkg1_expected + bkg2_expected"
}
n_expected_func = GenericFunction(**n_expected_config)

# ===== Create parameter points (only actual parameters, not data or templates) =====
params = ParameterSet(
    name="default_values",
    parameters=[
        # Free parameters
        ParameterPoint(name="mu", value=1.0),
        ParameterPoint(name="norm1", value=0.0),
        ParameterPoint(name="norm2", value=0.0),
        ParameterPoint(name="shape1", value=0.0),
    ]
)

# ===== Create data (observed values and templates) =====
data = [
    # Observed data
    PointData(name="n_obs", value=37.0),
    PointData(name="a_norm1", value=0.0),
    PointData(name="a_norm2", value=0.0),
    PointData(name="a_shape1", value=0.0),

    # Nominal templates
    PointData(name="signal_nominal", value=3.0),
    PointData(name="bkg1_nominal", value=10.0),
    PointData(name="bkg2_nominal", value=20.0),

    # Shape up templates
    PointData(name="bkg1_shape_up", value=12.0),
    PointData(name="bkg2_shape_up", value=23.0),
]

# ===== Create workspace =====
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[main_poisson, norm1_constraint, norm2_constraint, shape1_constraint, combined],
    functions=[
        signal_expected_func,
        bkg1_lnN_func,
        bkg1_shape_interp_func,
        bkg1_expected_func,
        bkg2_lnN_func,
        bkg2_shape_interp_func,
        bkg2_expected_func,
        n_expected_func,
    ],
    parameter_points=[params],
    data=data,
)

# Get the model and transpile to JAX
model = ws.model()
inputs, jaxified = get_jaxified_graph(model, "model")

rich.print(f"Model has {len(inputs)} input parameters: {[inp.name for inp in inputs]}")

# ===== Extract parameters and data as pytree =====
params_set = ws.parameter_points[0]
all_params = {p.name: float(p.value) for p in params_set.parameters}

# Extract data values separately (these are fixed and referenced by the model)
data_values = {data_point.name: float(data_point.value) for data_point in ws.data}

rich.print("\nInitial parameters:")
rich.print(all_params)
rich.print("\nData values:")
rich.print(data_values)

# ===== Define NLL function for everwillow =====
def nll_fn(params):
    """Negative log-likelihood that takes a pytree of parameters."""
    # Combine params with data values for the jaxified function
    all_values = {**params, **data_values}

    param_values = [all_values[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]
    return -jnp.log(prob)

# Test the NLL function
initial_nll = nll_fn(all_params)
rich.print(f"NLL at initial params: {initial_nll}")

# ===== Run unconditional fit with everwillow =====
rich.print("\n" + "="*60)
rich.print("Test 1: Unconditional fit (free: mu, norm1, norm2, shape1)")
rich.print("="*60)

result = ew.fit(
    nll_fn,
    all_params,
    fixed=[],  # No fixed parameters - all four (mu, norm1, norm2, shape1) are free
    max_steps=100
)

rich.print("\nFitted parameters:")
rich.print(f"  mu:     {result.params['mu']:.6f}")
rich.print(f"  norm1:  {result.params['norm1']:.6e}")
rich.print(f"  norm2:  {result.params['norm2']:.6e}")
rich.print(f"  shape1: {result.params['shape1']:.6e}")
rich.print(f"\nNLL at minimum: {result.nll:.6f}")

# Calculate total expected
signal_exp = result.params['mu'] * data_values['signal_nominal']
bkg1_exp = (1.0 + result.params['norm1'] * 0.1) * (
    data_values['bkg1_nominal'] + result.params['shape1'] *
    (data_values['bkg1_shape_up'] - data_values['bkg1_nominal'])
)
bkg2_exp = (1.0 + result.params['norm2'] * 0.05) * (
    data_values['bkg2_nominal'] + result.params['shape1'] *
    (data_values['bkg2_shape_up'] - data_values['bkg2_nominal'])
)
total_exp = signal_exp + bkg1_exp + bkg2_exp

rich.print(f"\nExpected counts:")
rich.print(f"  Signal: {signal_exp:.2f}")
rich.print(f"  Bkg1:   {bkg1_exp:.2f}")
rich.print(f"  Bkg2:   {bkg2_exp:.2f}")
rich.print(f"  Total:  {total_exp:.2f} (observed: 37)")

# ===== Run fixed parameter fit (profile likelihood) =====
rich.print("\n" + "="*60)
rich.print("Test 2: Fixed parameter fit (mu=1.5)")
rich.print("="*60)

result_fixed = ew.fixed_param_fit(
    {"mu": 1.5},
    nll_fn,
    all_params,
    fixed=[]  # No additional fixed parameters beyond mu
)

rich.print("\nFitted parameters:")
rich.print(f"  mu:     {result_fixed.params['mu']:.6f} (fixed)")
rich.print(f"  norm1:  {result_fixed.params['norm1']:.6e}")
rich.print(f"  norm2:  {result_fixed.params['norm2']:.6e}")
rich.print(f"  shape1: {result_fixed.params['shape1']:.6e}")
rich.print(f"\nNLL at mu=1.5: {result_fixed.nll:.6f}")
rich.print(f"Δ(NLL) = {result_fixed.nll - result.nll:.6e}")

rich.print("\n✅ All tests with pyhs3 evermore-style model completed!")

# ===== Compare with evermore implementation =====
rich.print("\n" + "="*60)
rich.print("Running actual evermore model with evermore optx implementation")
rich.print("="*60)

# Import evermore examples
import sys
sys.path.insert(0, "/Users/moaly/Work/iris-hep/evermore/examples")
from model import hists as evm_hists, params as evm_params, observation as evm_observation
from nll_fit_optimistix import fit as evm_fit

# Run evermore fit
evm_bestfit = evm_fit(evm_params, evm_hists, evm_observation)

rich.print("\nEvermore bestfit parameters:")
rich.print(f"  mu: {evm_bestfit.mu.value}")
rich.print(f"  norm1: {evm_bestfit.norm1.value}")
rich.print(f"  norm2: {evm_bestfit.norm2.value}")
rich.print(f"  shape1: {evm_bestfit.shape1.value}")

rich.print("\n" + "="*60)
rich.print("Comparison of results:")
rich.print("="*60)
rich.print(f"{'Parameter':<10} {'everwillow (pyhs3)':<20} {'evermore':<20} {'Difference':<15}")
rich.print("-"*65)
rich.print(f"{'mu':<10} {result.params['mu']:<20.6f} {float(evm_bestfit.mu.value):<20.6f} {abs(result.params['mu'] - float(evm_bestfit.mu.value)):<15.2e}")
rich.print(f"{'norm1':<10} {result.params['norm1']:<20.6f} {float(evm_bestfit.norm1.value):<20.6f} {abs(result.params['norm1'] - float(evm_bestfit.norm1.value)):<15.2e}")
rich.print(f"{'norm2':<10} {result.params['norm2']:<20.6f} {float(evm_bestfit.norm2.value):<20.6f} {abs(result.params['norm2'] - float(evm_bestfit.norm2.value)):<15.2e}")
rich.print(f"{'shape1':<10} {result.params['shape1']:<20.6f} {float(evm_bestfit.shape1.value):<20.6f} {abs(result.params['shape1'] - float(evm_bestfit.shape1.value)):<15.2e}")

rich.print("\n✅ Comparison completed!")
