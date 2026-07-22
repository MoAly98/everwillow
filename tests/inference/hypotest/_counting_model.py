"""Shared Poisson counting experiment for hypotest tests.

Model: n_expected = mu * S + B, with S=10, B=5
Poisson NLL: nll = n_exp - n_obs * log(n_exp)

Analytical solutions:
- MLE: mu_hat = (n_obs - B) / S
- q = 2 * [n_exp_test - n_obs - n_obs * log(n_exp_test / n_obs)]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import everwillow.statelib as sl

S = 10.0  # signal yield
B = 5.0  # background yield


def poisson_nll(params, observation):
    """Poisson NLL for a simple counting experiment."""
    mu = params["mu"]
    n_expected = mu * S + B
    n_observed = observation["n"]
    return n_expected - n_observed * jnp.log(n_expected)


def nll_with_nuisance(params, observation):
    """Poisson NLL with a background nuisance parameter.

    Model: n_expected = mu * s + b * theta
    Includes Gaussian constraint on theta (10% uncertainty).
    """
    mu = params["mu"]
    theta = params["theta"]
    n_expected = mu * S + B * theta
    n_observed = observation["n"]
    poisson_term = n_expected - n_observed * jnp.log(n_expected)
    constraint = 0.5 * (theta - 1.0) ** 2 / 0.1**2  # 10% uncertainty
    return poisson_term + constraint


def nll_two_nuisance(params, observation):
    """Poisson NLL with two nuisance parameters.

    Model: n_expected = mu * S * alpha + B * theta
    - theta: background normalization (10% Gaussian constraint)
    - alpha: signal efficiency (5% Gaussian constraint)
    """
    mu = params["mu"]
    theta = params["theta"]
    alpha = params["alpha"]
    n_expected = mu * S * alpha + B * theta
    n_observed = observation["n"]
    poisson_term = n_expected - n_observed * jnp.log(n_expected)
    constraint_theta = 0.5 * (theta - 1.0) ** 2 / 0.1**2  # 10% uncertainty
    constraint_alpha = 0.5 * (alpha - 1.0) ** 2 / 0.05**2  # 5% uncertainty
    return poisson_term + constraint_theta + constraint_alpha


def poisson_nll_2poi(params, observation):
    """Two independent Poisson channels, each with its own signal strength.

    Channel A: n_a ~ Poisson(mu_a * S + B); channel B likewise with mu_b.
    The NLL is separable, so the joint t statistic is the sum of the two
    single-channel t values, which makes expected values easy to precompute.
    """
    mu_a = params["mu_a"]
    mu_b = params["mu_b"]
    n_exp_a = mu_a * S + B
    n_exp_b = mu_b * S + B
    n_a = observation["n_a"]
    n_b = observation["n_b"]
    return (n_exp_a - n_a * jnp.log(n_exp_a)) + (n_exp_b - n_b * jnp.log(n_exp_b))


def create_params(mu_init: float = 1.0) -> sl.State:
    """Create initial parameter state."""
    return sl.State.from_pytree({"mu": mu_init})


def create_params_2poi(mu_a: float = 1.0, mu_b: float = 1.0) -> sl.State:
    """Create initial parameter state for the two-POI model."""
    return sl.State.from_pytree({"mu_a": mu_a, "mu_b": mu_b})


def create_observation_2poi(n_a: float, n_b: float) -> dict[str, float]:
    """Create a two-channel observation dict."""
    return {"n_a": n_a, "n_b": n_b}


def predict_fn_2poi(params_state: sl.State) -> dict[str, float]:
    """Prediction function for the two-POI model (Asimov data)."""
    tree = params_state.to_pytree()
    return {"n_a": tree["mu_a"] * S + B, "n_b": tree["mu_b"] * S + B}


def create_observation(n: float) -> dict[str, float]:
    """Create observation dict."""
    return {"n": n}


def predict_fn(params_state: sl.State) -> dict[str, float]:
    """Prediction function for Asimov data."""
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * S + B}


def sample_fn(params_state: sl.State, key) -> dict[str, float]:
    """Sample function for toy generation (Poisson sampling)."""
    expected = predict_fn(params_state)
    return {"n": jax.random.poisson(key, expected["n"])}
