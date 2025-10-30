"""Minimal wrapper around the evermore reference example."""

import typing as tp
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

import evermore as evm
import iminuit
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from flax import nnx
from jaxtyping import Array, Float, PyTree
from model_config import DEFAULT_DATA, ModelData, expected_components
from scipy.optimize import minimize

import everwillow as ew

# Float64 scalar
F64: tp.TypeAlias = Float[Array, ""]

# histograms / templates
Hist1D: tp.TypeAlias = Float[Array, "nbins"]  # noqa: F821
Hists1D: tp.TypeAlias = PyTree[Hist1D]

# negative log-likelihood implementation
Args: tp.TypeAlias = tuple[
    nnx.GraphDef,  # graphdef from `nnx.split`
    nnx.State,  # static state from `nnx.split`
    Hists1D,  # initial expectations for the histograms / templates
    Hist1D,  # observation: d
]

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision


class Params(NamedTuple):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter


def build_components(
    data: ModelData = DEFAULT_DATA,
) -> tuple[Params, dict[str, dict[str, jnp.ndarray]], jnp.ndarray]:
    """Return the Params, histogram templates, and observed data."""

    params = Params(
        mu=evm.Parameter(name="mu"),
        norm1=evm.NormalParameter(name="norm1"),
        norm2=evm.NormalParameter(name="norm2"),
        shape1=evm.NormalParameter(name="shape1"),
    )

    hists = {
        "nominal": {
            "signal": jnp.array([data.signal_nominal]),
            "bkg1": jnp.array([data.bkg1_nominal]),
            "bkg2": jnp.array([data.bkg2_nominal]),
        },
        "shape_up": {
            "bkg1": jnp.array([data.bkg1_shape_up]),
            "bkg2": jnp.array([data.bkg2_shape_up]),
        },
        "shape_down": {
            "bkg1": jnp.array([data.bkg1_shape_down]),
            "bkg2": jnp.array([data.bkg2_shape_down]),
        },
    }

    observation = jnp.array([data.observed])
    return params, hists, observation


def model(params: Params, hists: dict[str, dict[str, jnp.ndarray]]):
    expectations: dict[str, jnp.ndarray] = {}

    sig_mod = params.mu.scale()
    expectations["signal"] = sig_mod(hists["nominal"]["signal"])

    bkg1_lnN = params.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
    bkg1_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg1"],
        down_template=hists["shape_down"]["bkg1"],
    )
    expectations["bkg1"] = (bkg1_lnN @ bkg1_shape)(hists["nominal"]["bkg1"])

    bkg2_lnN = params.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
    bkg2_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg2"],
        down_template=hists["shape_down"]["bkg2"],
    )
    expectations["bkg2"] = (bkg2_lnN @ bkg2_shape)(hists["nominal"]["bkg2"])

    return expectations


@nnx.jit
def loss(
    dynamic: Params,
    args: tuple,
) -> jnp.ndarray:
    graphdef, static, hists, observation = args
    params = nnx.merge(graphdef, dynamic, static)
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    log_prob = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    log_prob += evm.util.sum_over_leaves(jax.tree.map(jnp.sum, constraints))
    return -jnp.sum(log_prob)


def fit_with_optimistix(
    components,
    max_steps: int = 10_000,
):
    params, hists, observation = components
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        loss,
        solver,
        dynamic,
        has_aux=False,
        args=(graphdef, static, hists, observation),
        max_steps=max_steps,
    )

    best = result.value.to_pure_dict()
    nll = result.state.f_info.f

    return best, nll


def fit_with_iminuit(
    components,
    max_steps: int = 10_000,
):
    params, hists, observation = components
    params.mu.set_metadata(frozen=True)
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)
    args = (graphdef, static, hists, observation)

    # update helper
    def _update(path, param, value):
        del path  # unused
        return param.replace(value=value)

    def update_dynamic(dynamic: nnx.State, new_state: nnx.State) -> nnx.State:
        return jax.tree.map_with_path(
            _update,
            dynamic,
            new_state,
            is_leaf=evm.filter.is_parameter,
            is_leaf_takes_path=True,
        )

    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # Wrapper for iminuit (operates on flat array)
    def iminuit_loss(
        pars: Float[Array, "n_params"],  # noqa: F821
        *,
        dynamic: nnx.State = dynamic,
        args: Args = args,
    ) -> F64:
        flat_values = pars

        # Reconstruct nested parameter state
        updated_dynamic = update_dynamic(dynamic, unravel_fn(flat_values))

        # Compute loss
        return loss(updated_dynamic, args)

    class FcnPartial:
        def __init__(self, fn, dynamic, args):
            self.fn = fn
            self.dynamic = dynamic
            self.args = args

        def __call__(self, flat_values):
            # make a shallow copy of args to modify
            (graphdef, static, hists, observation) = self.args
            graphdef, dynamic, static = nnx.split(
                nnx.merge(graphdef, self.dynamic, static, copy=True),
                evm.filter.is_parameter,
                ...,
            )
            args = (graphdef, static, hists, observation)
            return self.fn(flat_values, dynamic=dynamic, args=args)

    fcn = FcnPartial(iminuit_loss, dynamic, args).__call__

    # Setup Minuit
    minuit = iminuit.Minuit(
        fcn,
        flat_values,
        grad=nnx.grad(fcn),  # analytical gradient
    )
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.strategy = 2
    minuit.tol = 1e-8

    # minimize
    minuit.migrad(ncall=max_steps, use_simplex=False)
    bestfit = update_dynamic(dynamic, unravel_fn(jnp.array(minuit.values)))
    return bestfit.to_pure_dict(), minuit.fval


def fit_with_scipy(
    components,
    max_steps: int = 10_000,
):
    params, hists, observation = components
    params.mu.set_metadata(frozen=True)
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)
    args = (graphdef, static, hists, observation)

    # update helper
    def _update(path, param, value):
        del path  # unused
        return param.replace(value=value)

    def update_dynamic(dynamic: nnx.State, new_state: nnx.State) -> nnx.State:
        return jax.tree.map_with_path(
            _update,
            dynamic,
            new_state,
            is_leaf=evm.filter.is_parameter,
            is_leaf_takes_path=True,
        )

    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # Wrapper for scipy (operates on flat array), similar to iminuit wrapper
    def scipy_loss(
        pars: Float[Array, "n_params"],  # noqa: F821
        *,
        dynamic: nnx.State = dynamic,
        args: Args = args,
    ) -> F64:
        flat_values = pars

        # Reconstruct nested parameter state
        updated_dynamic = update_dynamic(dynamic, unravel_fn(flat_values))

        # Compute loss
        return loss(updated_dynamic, args)

    class FcnPartial:
        def __init__(self, fn, dynamic, args):
            self.fn = fn
            self.dynamic = dynamic
            self.args = args

        def __call__(self, flat_values):
            # make a shallow copy of args to modify
            (graphdef, static, hists, observation) = self.args
            graphdef, dynamic, static = nnx.split(
                nnx.merge(graphdef, self.dynamic, static, copy=True),
                evm.filter.is_parameter,
                ...,
            )
            args = (graphdef, static, hists, observation)
            # Convert from numpy to jax if needed
            if isinstance(flat_values, np.ndarray):
                flat_values = jnp.array(flat_values)
            return self.fn(flat_values, dynamic=dynamic, args=args)

    fcn = FcnPartial(scipy_loss, dynamic, args).__call__

    # Gradient function using nnx.grad (similar to iminuit)
    grad_fcn = nnx.grad(fcn)

    def grad_wrapper(flat_params):
        # Convert gradient to numpy for scipy
        return np.array(grad_fcn(flat_params))

    # Convert initial values to numpy
    init_numpy = np.array(flat_values)

    # Minimize using scipy
    result = minimize(
        lambda x: float(fcn(x)),  # Ensure scalar return
        init_numpy,
        method="SLSQP",
        jac=grad_wrapper,
        options={"maxiter": max_steps, "ftol": 1e-8},
    )

    # Convert result back to parameter dict
    bestfit = update_dynamic(dynamic, unravel_fn(jnp.array(result.x)))
    return bestfit.to_pure_dict(), float(result.fun)


def fit_with_everwillow(
    components,
    max_steps: int = 150,
):
    params, hists, observation = components
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)

    args = (graphdef, static, hists, observation)

    result = ew.fit(
        partial(loss, args=args),
        params=dynamic,
        max_steps=max_steps,
    )

    return result.params.to_pure_dict(), result.nll


def summarise_evermore_fit(params: Mapping[str, float], data: ModelData = DEFAULT_DATA):
    return expected_components(params, data=data)
