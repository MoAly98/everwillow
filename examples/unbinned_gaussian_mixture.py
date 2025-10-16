"""Unbinned Gaussian Mixture - Comprehensive comparison.

This example demonstrates fitting an unbinned Gaussian mixture model with:
- Two Gaussian components (gauss1 and gauss2)
- Mixture coefficients (a and b)
- A Gaussian prior constraint on coefficient b
- 1000 synthetic data points

The model: PDF(x) = a * Gauss1(x | mu1, sigma1) + b * Gauss2(x | mu2, sigma2)

This showcases:
1. Unbinned likelihood (sum of log-PDF over individual data points)
2. Using pyhs3 Gaussian distributions for individual PDFs
3. Three different fitting approaches:
   - pyhs3 model + everwillow inference
   - evermore model + evermore native inference (optimistix)
   - evermore model + everwillow inference
4. Comparing results across all approaches
"""

import rich
import jax
import jax.numpy as jnp
from jaxtyping import Array

import everwillow as ew
import evermore as evm

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Generate synthetic data (mixture of two Gaussians)
# ============================================================================

key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)

n_events = 1000
# 40% from gauss1: mean=0.0, sigma=0.5
data1 = jax.random.normal(key1, (int(n_events * 0.4),)) * 0.5 + 0.0
# 60% from gauss2: mean=2.0, sigma=0.3
data2 = jax.random.normal(key2, (int(n_events * 0.6),)) * 0.3 + 2.0

# Combine and filter to reasonable range
data = jnp.concatenate([data1, data2])
data = data[(data >= -2) & (data <= 4)]

rich.print("=" * 60)
rich.print("Generated toy data")
rich.print("=" * 60)
rich.print(f"Total events: {len(data)}")
rich.print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")


# ============================================================================
# PART 1: PyHS3 unbinned model with everwillow
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 1: PyHS3 unbinned model with everwillow")
rich.print("=" * 60)

import pyhs3
from pyhs3.distributions import GaussianDist
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.metadata import Metadata
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify
from pyhs3.typing.aliases import TensorVar
from collections.abc import Callable


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


# Create metadata
metadata = Metadata(hs3_version="0.2")

# Create Gaussian distributions for PDF evaluation
# Note: In unbinned fits, we evaluate the PDF at each data point
gauss1_dist = GaussianDist(
    name="gauss1_pdf",
    x="x_data",             # Data value (will be updated for each event)
    mean="mu1",             # Gauss1 mean parameter
    sigma="sigma1"          # Gauss1 width parameter
)

gauss2_dist = GaussianDist(
    name="gauss2_pdf",
    x="x_data",             # Data value (shared placeholder)
    mean="mu2",             # Gauss2 mean parameter
    sigma="sigma2"          # Gauss2 width parameter
)

# Create parameter set with initial values
pyhs3_params = ParameterSet(
    name="unbinned_params",
    parameters=[
        ParameterPoint(name="a", value=1.0),        # Gauss1 coefficient (fixed)
        ParameterPoint(name="b", value=1.0),        # Gauss2 coefficient (with prior)
        ParameterPoint(name="mu1", value=0.0),      # Gauss1 mean
        ParameterPoint(name="sigma1", value=0.5),   # Gauss1 width
        ParameterPoint(name="mu2", value=2.0),      # Gauss2 mean
        ParameterPoint(name="sigma2", value=0.3),   # Gauss2 width
        ParameterPoint(name="x_data", value=0.0),   # Placeholder for data
    ]
)

# Create workspace
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[gauss1_dist, gauss2_dist],
    functions=[],
    parameter_points=[pyhs3_params],
)

model = ws.model()

# Convert both Gaussian distributions to JAX functions
inputs1, jax_gauss1 = get_jaxified_graph(model, "gauss1_pdf")
inputs2, jax_gauss2 = get_jaxified_graph(model, "gauss2_pdf")

rich.print(f"\nGaussian 1 inputs: {[inp.name for inp in inputs1]}")
rich.print(f"Gaussian 2 inputs: {[inp.name for inp in inputs2]}")

# Extract initial parameters
params_set = ws.parameter_points[0]
pyhs3_initial = {p.name: float(p.value) for p in params_set.parameters}

rich.print("\nInitial parameters:")
for k, v in pyhs3_initial.items():
    if k != "x_data":  # Skip placeholder
        rich.print(f"  {k}: {v}")


def pyhs3_nll(params):
    """Unbinned negative log-likelihood for pyhs3 Gaussian mixture.

    For each data point:
    1. Evaluate both Gaussian PDFs
    2. Compute mixture: a * PDF1 + b * PDF2
    3. Sum -log(mixture) over all data points
    4. Add Gaussian prior on parameter 'b'

    Args:
        params: Dict of parameter values

    Returns:
        Negative log-likelihood value
    """
    def eval_pdf(x_val):
        """Evaluate mixture PDF at a single data point."""
        # Update params with current x value
        params_with_x = params.copy()
        params_with_x["x_data"] = x_val

        # Get parameter values in correct order for each Gaussian
        gauss1_vals = [params_with_x[inp.name] for inp in inputs1]
        gauss2_vals = [params_with_x[inp.name] for inp in inputs2]

        # Evaluate PDFs (returns probability, not log-prob)
        pdf1 = jax_gauss1(*gauss1_vals)[0]
        pdf2 = jax_gauss2(*gauss2_vals)[0]

        # Mixture: a * gauss1 + b * gauss2
        mixture = params["a"] * pdf1 + params["b"] * pdf2
        return mixture

    # Evaluate at all data points using vmap
    pdf_values = jax.vmap(eval_pdf)(data)

    # Add small epsilon to avoid log(0)
    pdf_values = jnp.maximum(pdf_values, 1e-10)

    # Negative log-likelihood: -sum(log(PDF))
    nll = -jnp.sum(jnp.log(pdf_values))

    # Add Gaussian prior on 'b': N(mean=1.0, width=0.3)
    prior_nll = -jnp.log(jnp.exp(-((params["b"] - 1.0) ** 2) / (2 * 0.3**2)) / (0.3 * jnp.sqrt(2 * jnp.pi)))
    nll += prior_nll

    return nll


# Test initial NLL
initial_nll = pyhs3_nll(pyhs3_initial)
rich.print(f"\nInitial NLL: {initial_nll:.2f}")

# Fit with everwillow
result_pyhs3 = ew.fit(
    pyhs3_nll,
    pyhs3_initial,
    fixed=["a", "x_data"],  # Fix 'a' at 1.0, x_data is just a placeholder
    max_steps=100
)

rich.print("\nFitted parameters (pyhs3 + everwillow):")
rich.print(f"  a:      {result_pyhs3.params['a']:.6f} (fixed)")
rich.print(f"  b:      {result_pyhs3.params['b']:.6f}")
rich.print(f"  mu1:    {result_pyhs3.params['mu1']:.6f}")
rich.print(f"  sigma1: {result_pyhs3.params['sigma1']:.6f}")
rich.print(f"  mu2:    {result_pyhs3.params['mu2']:.6f}")
rich.print(f"  sigma2: {result_pyhs3.params['sigma2']:.6f}")
rich.print(f"\nNLL at minimum: {result_pyhs3.nll:.2f}")
rich.print(f"ΔNLL: {result_pyhs3.nll - initial_nll:.2f}")


# ============================================================================
# PART 2: Evermore model with native optimizer
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 2: Evermore model with native optimizer (optimistix)")
rich.print("=" * 60)

# Evermore model setup using Parameter objects
evm_params = {
    "a": evm.Parameter(value=1.0, name="a", frozen=True),  # Fixed coefficient
    "b": evm.Parameter(value=1.0, name="b", prior=evm.pdf.Normal(mean=1.0, width=0.3)),  # With prior
    "gauss1": {
        "mu": evm.Parameter(value=0.0, name="mu1"),
        "sigma": evm.Parameter(value=0.5, name="sigma1")
    },
    "gauss2": {
        "mu": evm.Parameter(value=2.0, name="mu2"),
        "sigma": evm.Parameter(value=0.3, name="sigma2")
    },
}


def evm_model(params, x: Array) -> Array:
    """Model: weighted sum of two Gaussians.

    Args:
        params: Dict of evermore Parameters
        x: Data points to evaluate

    Returns:
        PDF values at each data point
    """
    def gaussian(x: Array, mu: evm.Parameter, sigma: evm.Parameter) -> Array:
        """Gaussian PDF."""
        return jnp.exp(-(((x - mu.value) / sigma.value) ** 2) / 2) / (sigma.value * jnp.sqrt(2 * jnp.pi))

    gauss1 = params["a"].value * gaussian(x, **params["gauss1"])
    gauss2 = params["b"].value * gaussian(x, **params["gauss2"])
    return gauss1 + gauss2


def evm_nll(params, data: Array) -> float:
    """Negative log-likelihood for unbinned data with evermore model.

    Args:
        params: Dict of evermore Parameters
        data: Array of observed data points

    Returns:
        Negative log-likelihood including priors
    """
    # Evaluate model at each data point
    pdf_values = evm_model(params, data)

    # Add small epsilon to avoid log(0)
    pdf_values = jnp.maximum(pdf_values, 1e-10)

    # Negative log-likelihood
    nll = -jnp.sum(jnp.log(pdf_values))

    # Add constraints from priors
    constraints = evm.loss.get_log_probs(params)
    nll += -evm.util.sum_over_leaves(constraints)

    return nll


# Test initial NLL
initial_nll_evm = evm_nll(evm_params, data)
rich.print(f"\nInitial NLL: {initial_nll_evm:.2f}")

# Fit with evermore's native optimizer (optimistix)
import optimistix as optx

# Partition parameters into dynamic (free) and static (frozen)
dynamic, static = evm.tree.partition(evm_params, is_leaf=lambda x: isinstance(x, evm.Parameter))


def optx_loss(dynamic, args):
    """Loss function wrapper for optimistix.

    Args:
        dynamic: Free parameters
        args: Tuple of (static_params, data)

    Returns:
        Loss value
    """
    static, data = args
    params = evm.tree.combine(dynamic, static)
    return evm_nll(params, data)


solver = optx.BFGS(rtol=1e-5, atol=1e-7)
fitresult = optx.minimise(
    optx_loss,
    solver,
    dynamic,
    has_aux=False,
    args=(static, data),
    max_steps=100,
)

evm_bestfit = evm.tree.combine(fitresult.value, static)

rich.print("\nFitted parameters (evermore + native optimistix):")
rich.print(f"  a:      {evm_bestfit['a'].value:.6f} (fixed)")
rich.print(f"  b:      {evm_bestfit['b'].value:.6f}")
rich.print(f"  mu1:    {evm_bestfit['gauss1']['mu'].value:.6f}")
rich.print(f"  sigma1: {evm_bestfit['gauss1']['sigma'].value:.6f}")
rich.print(f"  mu2:    {evm_bestfit['gauss2']['mu'].value:.6f}")
rich.print(f"  sigma2: {evm_bestfit['gauss2']['sigma'].value:.6f}")

evm_final_nll = evm_nll(evm_bestfit, data)
rich.print(f"\nNLL at minimum: {evm_final_nll:.2f}")
rich.print(f"ΔNLL: {evm_final_nll - initial_nll_evm:.2f}")


# ============================================================================
# PART 3: Evermore model with everwillow
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 3: Evermore model fitted with everwillow")
rich.print("=" * 60)


def evm_params_to_pytree(params):
    """Convert evermore Parameters to simple dict for everwillow.

    Args:
        params: Dict of evermore Parameters

    Returns:
        Dict of scalar values
    """
    return {
        "a": params["a"].value,
        "b": params["b"].value,
        "mu1": params["gauss1"]["mu"].value,
        "sigma1": params["gauss1"]["sigma"].value,
        "mu2": params["gauss2"]["mu"].value,
        "sigma2": params["gauss2"]["sigma"].value,
    }


def pytree_to_evm_params(pytree):
    """Convert simple dict back to evermore Parameters.

    Args:
        pytree: Dict of scalar values

    Returns:
        Dict of evermore Parameters
    """
    return {
        "a": evm.Parameter(value=pytree["a"], name="a", frozen=True),
        "b": evm.Parameter(value=pytree["b"], name="b", prior=evm.pdf.Normal(mean=1.0, width=0.3)),
        "gauss1": {
            "mu": evm.Parameter(value=pytree["mu1"], name="mu1"),
            "sigma": evm.Parameter(value=pytree["sigma1"], name="sigma1")
        },
        "gauss2": {
            "mu": evm.Parameter(value=pytree["mu2"], name="mu2"),
            "sigma": evm.Parameter(value=pytree["sigma2"], name="sigma2")
        },
    }


initial_pytree = evm_params_to_pytree(evm_params)
rich.print("\nInitial parameters:")
for k, v in initial_pytree.items():
    rich.print(f"  {k}: {v}")


def nll_fn(params_pytree):
    """NLL function for everwillow.

    Converts pytree to evermore Parameters and evaluates loss.

    Args:
        params_pytree: Dict of parameter values

    Returns:
        Negative log-likelihood
    """
    evm_params_reconstructed = pytree_to_evm_params(params_pytree)
    return evm_nll(evm_params_reconstructed, data)


# Fit with everwillow
result_evm = ew.fit(
    nll_fn,
    initial_pytree,
    fixed=["a"],        # Keep 'a' fixed at 1.0
    max_steps=100
)

rich.print("\nFitted parameters (evermore + everwillow):")
rich.print(f"  a:      {result_evm.params['a']:.6f} (fixed)")
rich.print(f"  b:      {result_evm.params['b']:.6f}")
rich.print(f"  mu1:    {result_evm.params['mu1']:.6f}")
rich.print(f"  sigma1: {result_evm.params['sigma1']:.6f}")
rich.print(f"  mu2:    {result_evm.params['mu2']:.6f}")
rich.print(f"  sigma2: {result_evm.params['sigma2']:.6f}")
rich.print(f"\nNLL at minimum: {result_evm.nll:.2f}")
rich.print(f"ΔNLL: {result_evm.nll - initial_nll_evm:.2f}")


# ============================================================================
# PART 4: Comparison table
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Comparison of all three approaches")
rich.print("=" * 60)

# Print header
rich.print(f"{'Parameter':<10} {'pyhs3+ew':<15} {'evm+native':<15} {'evm+ew':<15} "
          f"{'Diff(pyhs3,native)':<18} {'Diff(pyhs3,ew)':<18}")
rich.print("-" * 91)

# Define comparison mapping
comparisons = [
    ("a", "a", result_pyhs3.params['a'], evm_bestfit['a'].value, result_evm.params['a']),
    ("b", "b", result_pyhs3.params['b'], evm_bestfit['b'].value, result_evm.params['b']),
    ("mu1", "gauss1/mu", result_pyhs3.params['mu1'], evm_bestfit['gauss1']['mu'].value, result_evm.params['mu1']),
    ("sigma1", "gauss1/sigma", result_pyhs3.params['sigma1'], evm_bestfit['gauss1']['sigma'].value, result_evm.params['sigma1']),
    ("mu2", "gauss2/mu", result_pyhs3.params['mu2'], evm_bestfit['gauss2']['mu'].value, result_evm.params['mu2']),
    ("sigma2", "gauss2/sigma", result_pyhs3.params['sigma2'], evm_bestfit['gauss2']['sigma'].value, result_evm.params['sigma2']),
]

for param_name, _, pyhs3_val, evm_native_val, evm_ew_val in comparisons:
    diff1 = abs(pyhs3_val - evm_native_val)
    diff2 = abs(pyhs3_val - evm_ew_val)
    rich.print(f"{param_name:<10} {pyhs3_val:<15.6f} {evm_native_val:<15.6f} {evm_ew_val:<15.6f} "
              f"{diff1:<18.2e} {diff2:<18.2e}")

rich.print(f"\n{'NLL':<10} {result_pyhs3.nll:<15.2f} {evm_final_nll:<15.2f} {result_evm.nll:<15.2f} "
          f"{abs(result_pyhs3.nll - evm_final_nll):<18.2e} {abs(result_pyhs3.nll - result_evm.nll):<18.2e}")

rich.print("\n✅ All three unbinned model comparisons completed!")
