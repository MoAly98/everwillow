"""Binned Signal + 2 Backgrounds with Systematics - Comprehensive comparison.

This example demonstrates a realistic HEP analysis with:
- Signal process with strength parameter mu
- Two background processes (bkg1, bkg2)
- Log-normal systematic uncertainties (norm1, norm2)
- Shape systematic (shape1) affecting both backgrounds
- Gaussian constraints on all nuisance parameters

Model structure:
    Total = mu * signal + norm1_modifier * shape_interp(bkg1) + norm2_modifier * shape_interp(bkg2)

Where:
    - norm_modifier = kappa^norm, with kappa=up when norm>=0, kappa=1/down when norm<0
    - shape_interp = nominal + shape * (up - nominal)

This showcases:
1. Building complex pyhs3 models with log-normal modifiers
2. Implementing conditional logic using the "abs trick" to avoid PyTensor limitations
3. Fitting the same model with three different approaches:
   - pyhs3 model + everwillow inference
   - evermore model + everwillow inference
   - evermore model + evermore native inference
4. Comparing results across all approaches
"""

from collections.abc import Callable

import evermore as evm
import jax
import jax.numpy as jnp
import pyhs3
import rich
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.typing.aliases import TensorVar
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew

# JAX float64
jax.config.update("jax_enable_x64", True)

# ============================================================================
# UTILITY: Convert PyTensor graph to JAX function
# ============================================================================


def get_jaxified_graph(
    model: pyhs3.Model,
    name: str,
) -> tuple[list[TensorVar], Callable]:
    """Convert PyTensor graph to JAX function.

    Args:
        model: The pyhs3 Model containing distributions
        name: Name of the distribution to convert

    Returns:
        Tuple of (input_variables, jax_function)
    """
    dist = model.distributions[name]
    inputs = [var for var in graph_inputs([dist]) if var.name is not None]
    fgraph = FunctionGraph(inputs=inputs, outputs=[dist], clone=True)
    mode.JAX.optimizer.rewrite(fgraph)
    return inputs, jax_funcify(fgraph)


# ============================================================================
# PART 1: Build pyhs3 model (matching evermore structure)
# ============================================================================
#
# Model structure from evermore:
# - Signal: mu * nominal (3 counts)
# - Bkg1: norm1_modifier * shape_interp * nominal (10 counts nominal, 12 with shape up)
# - Bkg2: norm2_modifier * shape_interp * nominal (20 counts nominal, 23 with shape up)
# - Total observation: 37 counts
# - All nuisance parameters (norm1, norm2, shape1) have Gaussian constraints
#
# ============================================================================

rich.print("=" * 60)
rich.print("Building pyhs3 model (evermore-style)")
rich.print("=" * 60)

metadata = Metadata(hs3_version="0.2")


# ============================================================================
# Distributions
# ============================================================================

# Main Poisson for the observation
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",  # Observed counts
    mean="n_expected",  # Total expected (computed below)
)

# Gaussian constraints for nuisance parameters (auxiliary measurements)
norm1_constraint = GaussianDist(
    name="norm1_constraint",
    x="a_norm1",  # Auxiliary observable
    mean="norm1",  # Nuisance parameter
    sigma=1.0,  # Constraint width
)

norm2_constraint = GaussianDist(
    name="norm2_constraint", x="a_norm2", mean="norm2", sigma=1.0
)

shape1_constraint = GaussianDist(
    name="shape1_constraint", x="a_shape1", mean="shape1", sigma=1.0
)

# Combined likelihood = main * all constraints
combined_config = {
    "type": "product_dist",
    "name": "model",
    "factors": [
        "main_poisson",
        "norm1_constraint",
        "norm2_constraint",
        "shape1_constraint",
    ],
}
combined = ProductDist(**combined_config)


# ============================================================================
# Functions: Signal expectation
# ============================================================================

signal_expected_config = {
    "type": "generic_function",
    "name": "signal_expected",
    "expression": "mu * signal_nominal",
}
signal_expected_func = GenericFunction(**signal_expected_config)


# ============================================================================
# Functions: Bkg1 with log-normal and shape systematics
# ============================================================================

# Log-normal modifier: kappa^norm1 where kappa = 1.1 (up) when norm1 >= 0,
# kappa = 1/0.9 (1/down) when norm1 < 0
#
# Implementation: The "abs trick" avoids PyTensor's conditional limitations
# by using continuous weight functions:
#   weight_up   = (x + abs(x)) / (2*abs(x) + eps)  -> 1 when x>0, 0 when x<0
#   weight_down = (abs(x) - x) / (2*abs(x) + eps)  -> 0 when x>0, 1 when x<0
#
# Final expression: exp(x * log(up)^weight_up * log(down)^weight_down)

bkg1_lnN_config = {
    "type": "generic_function",
    "name": "bkg1_lnN_factor",
    "expression": "exp(norm1 * (log(1.1)**((norm1 + abs(norm1))/(2*abs(norm1) + 1e-10)) * log(1.0/0.9)**((abs(norm1) - norm1)/(2*abs(norm1) + 1e-10))))",
}
bkg1_lnN_func = GenericFunction(**bkg1_lnN_config)

# Shape interpolation: linear interpolation between nominal and up templates
bkg1_shape_interp_config = {
    "type": "generic_function",
    "name": "bkg1_shape_interp",
    "expression": "bkg1_nominal + shape1 * (bkg1_shape_up - bkg1_nominal)",
}
bkg1_shape_interp_func = GenericFunction(**bkg1_shape_interp_config)

# Bkg1 total: log-normal modifier * shape interpolation
bkg1_expected_config = {
    "type": "generic_function",
    "name": "bkg1_expected",
    "expression": "bkg1_lnN_factor * bkg1_shape_interp",
}
bkg1_expected_func = GenericFunction(**bkg1_expected_config)


# ============================================================================
# Functions: Bkg2 with log-normal and shape systematics
# ============================================================================

# Log-normal modifier for bkg2: kappa = 1.05 (up) when norm2 >= 0,
# kappa = 1/0.95 (1/down) when norm2 < 0
bkg2_lnN_config = {
    "type": "generic_function",
    "name": "bkg2_lnN_factor",
    "expression": "exp(norm2 * (log(1.05)**((norm2 + abs(norm2))/(2*abs(norm2) + 1e-10)) * log(1.0/0.95)**((abs(norm2) - norm2)/(2*abs(norm2) + 1e-10))))",
}
bkg2_lnN_func = GenericFunction(**bkg2_lnN_config)

# Shape interpolation for bkg2
bkg2_shape_interp_config = {
    "type": "generic_function",
    "name": "bkg2_shape_interp",
    "expression": "bkg2_nominal + shape1 * (bkg2_shape_up - bkg2_nominal)",
}
bkg2_shape_interp_func = GenericFunction(**bkg2_shape_interp_config)

# Bkg2 total
bkg2_expected_config = {
    "type": "generic_function",
    "name": "bkg2_expected",
    "expression": "bkg2_lnN_factor * bkg2_shape_interp",
}
bkg2_expected_func = GenericFunction(**bkg2_expected_config)


# ============================================================================
# Functions: Total expected
# ============================================================================

n_expected_config = {
    "type": "generic_function",
    "name": "n_expected",
    "expression": "signal_expected + bkg1_expected + bkg2_expected",
}
n_expected_func = GenericFunction(**n_expected_config)


# ============================================================================
# Parameters and Data
# ============================================================================

# Parameters (values to fit)
params = ParameterSet(
    name="default_values",
    parameters=[
        ParameterPoint(name="mu", value=1.0),  # Signal strength (POI)
        ParameterPoint(name="norm1", value=0.0),  # Bkg1 norm uncertainty
        ParameterPoint(name="norm2", value=0.0),  # Bkg2 norm uncertainty
        ParameterPoint(name="shape1", value=0.0),  # Shape uncertainty
    ],
)

# Data (fixed: observations and templates)
data = [
    # Observed data
    PointData(name="n_obs", value=37.0),  # Main observable
    PointData(name="a_norm1", value=0.0),  # Auxiliary measurement
    PointData(name="a_norm2", value=0.0),
    PointData(name="a_shape1", value=0.0),
    # Nominal templates
    PointData(name="signal_nominal", value=3.0),
    PointData(name="bkg1_nominal", value=10.0),
    PointData(name="bkg2_nominal", value=20.0),
    # Shape up templates (for shape systematic)
    PointData(name="bkg1_shape_up", value=12.0),
    PointData(name="bkg2_shape_up", value=23.0),
]


# ============================================================================
# Create workspace and build model
# ============================================================================

ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[
        main_poisson,
        norm1_constraint,
        norm2_constraint,
        shape1_constraint,
        combined,
    ],
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

model = ws.model()
inputs, jaxified = get_jaxified_graph(model, "model")

rich.print(f"Model has {len(inputs)} input parameters: {[inp.name for inp in inputs]}")


# ============================================================================
# Extract parameters and data as pytree
# ============================================================================

params_set = ws.parameter_points[0]
all_params = {p.name: float(p.value) for p in params_set.parameters}

# Extract data values separately (these are fixed and referenced by the model)
data_values = {data_point.name: float(data_point.value) for data_point in ws.data}

rich.print("\nInitial parameters:")
rich.print(all_params)
rich.print("\nData values:")
rich.print(data_values)


# ============================================================================
# Define NLL function for everwillow
# ============================================================================


def nll_fn(params):
    """Negative log-likelihood function for pyhs3 model.

    Args:
        params: Dict of parameters to fit

    Returns:
        Negative log-likelihood value
    """
    # Combine parameters with data values for the jaxified function
    all_values = {**params, **data_values}

    param_values = [all_values[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]
    return -jnp.log(prob)


# Test the NLL function at initial parameters
initial_nll = nll_fn(all_params)
rich.print(f"NLL at initial params: {initial_nll}")


# ============================================================================
# PART 2: Fit pyhs3 model with everwillow
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 1: Unconditional fit (pyhs3 + everwillow)")
rich.print("=" * 60)

result = ew.fit(
    nll_fn,
    all_params,
    fixed=[],  # All parameters free
    max_steps=100,
)

rich.print("\nFitted parameters:")
rich.print(f"  mu:     {result.params['mu']:.6f}")
rich.print(f"  norm1:  {result.params['norm1']:.6e}")
rich.print(f"  norm2:  {result.params['norm2']:.6e}")
rich.print(f"  shape1: {result.params['shape1']:.6e}")
rich.print(f"\nNLL at minimum: {result.nll:.6f}")

# Calculate total expected counts for verification
signal_exp = result.params["mu"] * data_values["signal_nominal"]

# Bkg1: log-normal factor
norm1_val = result.params["norm1"]
if norm1_val >= 0:
    bkg1_lnN = jnp.exp(norm1_val * jnp.log(1.1))
else:
    bkg1_lnN = jnp.exp(norm1_val * jnp.log(1.0 / 0.9))
bkg1_shape_interp = data_values["bkg1_nominal"] + result.params["shape1"] * (
    data_values["bkg1_shape_up"] - data_values["bkg1_nominal"]
)
bkg1_exp = bkg1_lnN * bkg1_shape_interp

# Bkg2: log-normal factor
norm2_val = result.params["norm2"]
if norm2_val >= 0:
    bkg2_lnN = jnp.exp(norm2_val * jnp.log(1.05))
else:
    bkg2_lnN = jnp.exp(norm2_val * jnp.log(1.0 / 0.95))
bkg2_shape_interp = data_values["bkg2_nominal"] + result.params["shape1"] * (
    data_values["bkg2_shape_up"] - data_values["bkg2_nominal"]
)
bkg2_exp = bkg2_lnN * bkg2_shape_interp
total_exp = signal_exp + bkg1_exp + bkg2_exp

rich.print("\nExpected counts:")
rich.print(f"  Signal: {signal_exp:.2f}")
rich.print(f"  Bkg1:   {bkg1_exp:.2f}")
rich.print(f"  Bkg2:   {bkg2_exp:.2f}")
rich.print(f"  Total:  {total_exp:.2f} (observed: 37)")


# ============================================================================
# PART 3: Profile likelihood (fixed parameter fit)
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 2: Fixed parameter fit (mu=1.5)")
rich.print("=" * 60)

result_fixed = ew.fixed_param_fit(
    {"mu": 1.5},  # Fix mu to this value
    nll_fn,
    all_params,
    fixed=[],  # No additional fixed parameters beyond mu
)

rich.print("\nFitted parameters:")
rich.print(f"  mu:     {result_fixed.params['mu']:.6f} (fixed)")
rich.print(f"  norm1:  {result_fixed.params['norm1']:.6e}")
rich.print(f"  norm2:  {result_fixed.params['norm2']:.6e}")
rich.print(f"  shape1: {result_fixed.params['shape1']:.6e}")
rich.print(f"\nNLL at mu=1.5: {result_fixed.nll:.6f}")
rich.print(f"Δ(NLL) = {result_fixed.nll - result.nll:.6e}")

rich.print("\n✅ All tests with pyhs3 evermore-style model completed!")


# ============================================================================
# PART 4: Fit evermore model directly with everwillow
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 3: Fit evermore model directly with everwillow")
rich.print("=" * 60)

# Import evermore example model
import sys  # noqa: E402

sys.path.insert(0, "/Users/moaly/Work/iris-hep/evermore/examples")
from model import hists as evm_hists  # noqa: E402
from model import loss as evm_loss  # noqa: E402
from model import observation as evm_observation  # noqa: E402
from model import params as evm_params  # noqa: E402

# Partition evermore params into dynamic (free) and static (frozen)
evm_dynamic, evm_static = evm.tree.partition(evm_params)

# Demonstrate two approaches: one that fails, one that works

# ===== Approach 1: Try passing evermore Parameters directly (will fail) =====
rich.print("\nAttempt 1: Pass evermore Parameter objects directly")
try:
    result_evm_direct = ew.fit(
        evm_loss,
        evm_dynamic,  # This contains evm.Parameter objects, not just values
        args=(evm_static, evm_hists, evm_observation),
        fixed=[],
        max_steps=100,
    )
    rich.print("✓ Success with direct Parameter objects!")
except Exception as e:
    rich.print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    rich.print(
        "  Reason: evermore Parameters have complex structure that everwillow can't flatten"
    )

# ===== Approach 2: Extract numeric values, use wrapper with args API =====
rich.print("\nAttempt 2: Extract values, use wrapper with args API")

# Extract numeric values from evermore's dynamic parameters
initial_evm_values = {
    "mu": evm_dynamic.mu.value,
    "norm1": evm_dynamic.norm1.value,
    "norm2": evm_dynamic.norm2.value,
    "shape1": evm_dynamic.shape1.value,
}

# Wrapper to reconstruct Params structure for evermore loss
from model import Params  # noqa: E402


def evm_nll_wrapper(params_dict, static, hists, observation):
    """Reconstruct Params and call evermore loss."""
    dynamic = evm.tree.update_values(
        evm_dynamic,
        values=Params(
            mu=params_dict["mu"],
            norm1=params_dict["norm1"],
            norm2=params_dict["norm2"],
            shape1=params_dict["shape1"],
        ),
    )
    return evm_loss(dynamic, static, hists, observation)


# Fit with everwillow
result_evm = ew.fit(
    evm_nll_wrapper,
    initial_evm_values,
    args=(evm_static, evm_hists, evm_observation),
    fixed=[],
    max_steps=100,
)

rich.print("✓ Success")
rich.print("\nFitted parameters (evermore + everwillow):")
rich.print(f"  mu:     {result_evm.params['mu']:.6f}")
rich.print(f"  norm1:  {result_evm.params['norm1']:.6e}")
rich.print(f"  norm2:  {result_evm.params['norm2']:.6e}")
rich.print(f"  shape1: {result_evm.params['shape1']:.6e}")
rich.print(f"(NLL at minimum: {float(result_evm.nll):.6f})")


# ============================================================================
# PART 5: Fit with evermore's native optimizer (for comparison)
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 4: Fit evermore model with evermore's native optimizer")
rich.print("=" * 60)

from nll_fit_optimistix import fit as evm_fit  # noqa: E402

# Run evermore fit
evm_bestfit = evm_fit(evm_params, evm_hists, evm_observation)

rich.print("\nFitted parameters (evermore + native optimistix):")
rich.print(f"  mu:     {evm_bestfit.mu.value:.6f}")
rich.print(f"  norm1:  {evm_bestfit.norm1.value:.6e}")
rich.print(f"  norm2:  {evm_bestfit.norm2.value:.6e}")
rich.print(f"  shape1: {evm_bestfit.shape1.value:.6e}")


# ============================================================================
# PART 6: Comparison table
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Comparison of all approaches:")
rich.print("=" * 60)

# Print header
rich.print(
    f"{'Parameter':<10} {'pyhs3+evw':<15} {'evm+evw':<15} {'evm+native':<15} "
    f"{'Diff(pyhs3,evm+evw)':<20} {'Diff(pyhs3,evm+native)':<20}"
)
rich.print("-" * 95)

# Compare each parameter
for param_name in ["mu", "norm1", "norm2", "shape1"]:
    pyhs3_val = result.params[param_name]
    evm_evw_val = result_evm.params[param_name]  # Now a dict, not Params object
    evm_native_val = float(getattr(evm_bestfit, param_name).value)

    diff1 = abs(pyhs3_val - evm_evw_val)
    diff2 = abs(pyhs3_val - evm_native_val)

    rich.print(
        f"{param_name:<10} {pyhs3_val:<15.6f} {evm_evw_val:<15.6f} {evm_native_val:<15.6f} "
        f"{diff1:<20.2e} {diff2:<20.2e}"
    )

# Note: evermore loss and pyhs3 NLL differ by a constant offset
# This doesn't affect optimization but means NLL values aren't directly comparable
rich.print(
    f"\n{'NLL':<10} {result.nll:<15.6f} {float(result_evm.nll):<15.6f} {'N/A':<15}"
)
rich.print("\nNote: evermore loss differs from pyhs3 NLL by a constant offset.")
rich.print("This doesn't affect optimization - all approaches find the same minimum.")

rich.print("\n✅ All comparisons completed!")
