"""Test everwillow with the pyhs3 example."""

import rich
import jax.numpy as jnp
import jax.scipy as jsp

import everwillow as ew

# Create a simple NLL function (using our pyhs3 model concept)
def simple_nll(params):
    """
    Simple negative log-likelihood for testing.

    Model: Poisson(n_obs | mu * signal + bkg_norm * background * (1 + nu))
           * Gaussian(0 | nu, 1)
    """
    # Extract parameters
    mu = params["mu"]
    bkg_norm = params["bkg_norm"]
    nu = params["nu"]
    n_obs = params["n_obs"]
    signal = params["signal"]
    background = params["background"]

    # Compute expected
    background_modified = background
    n_expected = mu * signal + bkg_norm * background_modified

    # Poisson term: -log P(n_obs | n_expected)
    # Using: -log P(n|λ) = λ - n*log(λ) + log(n!)
    poisson_nll = n_expected - n_obs * jnp.log(n_expected + 1e-10) + jsp.special.gammaln(n_obs + 1.0)

    # Gaussian constraint on nu: -log N(0 | nu, 1)
    constraint_nll = 0.5 * nu**2  # add eps for numerical stability

    return poisson_nll + constraint_nll


# Test 1: Unconditional fit
rich.print("\n" + "="*60)
rich.print("Test 1: Unconditional fit")
rich.print("="*60)

params = {
    "mu": 1.0,
    "bkg_norm": 1.0,
    "nu": 0.0,
    "n_obs": 75.0,
    "signal": 5.0,
    "background": 50.0,
}

# Test the NLL function first
nll_initial = simple_nll(params)
rich.print(f"NLL at initial params: {nll_initial}")

result = ew.fit(
    simple_nll,
    params,
    fixed=["n_obs", "signal", "background", "nu"],
    max_steps=100  # Increase max steps
)

rich.print("Fitted parameters:", result.params)
rich.print(f"NLL at minimum: {result.nll}")
rich.print(f"Fitted mu: {result.params['mu']}")
rich.print(f"Fitted bkg_norm: {result.params['bkg_norm']}")
rich.print(f"Fitted nu: {result.params['nu']}")


# Test 2: Fixed parameter fit (profile likelihood point)
rich.print("\n" + "="*60)
rich.print("Test 2: Fixed parameter fit (mu=1.5)")
rich.print("="*60)

result_fixed = ew.fixed_param_fit(
    {"mu": 1.5},
    simple_nll,
    params,
    fixed=["n_obs", "signal", "background"]
)

rich.print("Fitted parameters:", result_fixed.params)
rich.print(f"NLL at mu=1.5: {result_fixed.nll}")
rich.print(f"Δ(NLL) = {result_fixed.nll - result.nll}")

rich.print("\n✅ All tests completed!")
