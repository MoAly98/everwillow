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


def create_params(mu_init: float = 1.0) -> sl.State:
    """Create initial parameter state."""
    return sl.State.from_pytree({"mu": mu_init})


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
