"""Minimal wrapper around the evermore reference example."""

import typing as tp
from collections.abc import Mapping
from functools import partial

import evermore as evm
import iminuit
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optimistix as optx
from flax import nnx
from jaxtyping import Array, Float, PyTree
from model_config import DEFAULT_DATA, ModelData, expected_components
from scipy.optimize import minimize

import everwillow as ew
import everwillow.statelib as sl

# Float64 scalar
F64: tp.TypeAlias = Float[Array, ""]

# type defs
Hist1D: tp.TypeAlias = Float[Array, " nbins"]
Args: tp.TypeAlias = tuple[
    nnx.GraphDef,  # graphdef
    nnx.State,  # state
    PyTree[Hist1D],  # hists
    Hist1D,  # observation
]


class Model(nnx.Module):
    def __init__(
        self,
        mu: evm.Parameter,
        norm1: evm.NormalParameter,
        norm2: evm.NormalParameter,
        shape1: evm.NormalParameter,
    ):
        self.mu = mu
        self.norm1 = norm1
        self.norm2 = norm2
        self.shape1 = shape1

    def __call__(self, hists: PyTree[Hist1D]) -> PyTree[Hist1D]:
        expectations = {}

        # signal process
        sig_mod = self.mu.scale()
        expectations["signal"] = sig_mod(hists["nominal"]["signal"])

        # bkg1 process
        bkg1_lnN = self.norm1.scale_log_asymmetric(
            up=jnp.array([1.1]), down=jnp.array([0.9])
        )
        bkg1_shape = self.shape1.morphing(
            up_template=hists["shape_up"]["bkg1"],
            down_template=hists["shape_down"]["bkg1"],
        )
        # combine modifiers
        bkg1_mod = bkg1_lnN @ bkg1_shape
        expectations["bkg1"] = bkg1_mod(hists["nominal"]["bkg1"])

        # bkg2 process
        bkg2_lnN = self.norm2.scale_log_asymmetric(
            up=jnp.array([1.05]), down=jnp.array([0.95])
        )
        bkg2_shape = self.shape1.morphing(
            up_template=hists["shape_up"]["bkg2"],
            down_template=hists["shape_down"]["bkg2"],
        )
        # combine modifiers
        bkg2_mod = bkg2_lnN @ bkg2_shape
        expectations["bkg2"] = bkg2_mod(hists["nominal"]["bkg2"])

        # return the modified expectations
        return expectations


hists = jax.tree.map(
    jnp.atleast_1d,
    {
        "nominal": {
            "signal": DEFAULT_DATA.signal_nominal,
            "bkg1": DEFAULT_DATA.bkg1_nominal,
            "bkg2": DEFAULT_DATA.bkg2_nominal,
        },
        "shape_up": {
            "bkg1": DEFAULT_DATA.bkg1_shape_up,
            "bkg2": DEFAULT_DATA.bkg2_shape_up,
        },
        "shape_down": {
            "bkg1": DEFAULT_DATA.bkg1_shape_down,
            "bkg2": DEFAULT_DATA.bkg2_shape_down,
        },
    },
)


def build_components(
    data: ModelData = DEFAULT_DATA,
) -> tuple[Model, PyTree[Hist1D], Hist1D]:
    model = Model(
        mu=evm.Parameter(name="mu"),
        norm1=evm.NormalParameter(name="norm1"),
        norm2=evm.NormalParameter(name="norm2"),
        shape1=evm.NormalParameter(name="shape1"),
    )
    hists = jax.tree.map(
        jnp.atleast_1d,
        {
            "nominal": {
                "signal": data.signal_nominal,
                "bkg1": data.bkg1_nominal,
                "bkg2": data.bkg2_nominal,
            },
            "shape_up": {
                "bkg1": data.bkg1_shape_up,
                "bkg2": data.bkg2_shape_up,
            },
            "shape_down": {
                "bkg1": data.bkg1_shape_down,
                "bkg2": data.bkg2_shape_down,
            },
        },
    )
    observation = jnp.array([data.observed])
    return model, hists, observation


@nnx.jit
def loss(dynamic: nnx.State, args: Args) -> Float[Array, ""]:
    # unpack
    (graphdef, static, hists, observation) = args
    # reconstruct model
    model = nnx.merge(graphdef, dynamic, static)
    # calculate expectation
    expectations = model(hists)
    # calculate constraints
    constraints = evm.loss.get_log_probs(model)
    loss_val = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    # sum all up
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


def fit_with_optimistix(
    components,
    max_steps: int = 10_000,
):
    model, hists, observation = components
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_parameter, ...)

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
    model, hists, observation = components
    model.mu.set_metadata(frozen=True)
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_parameter, ...)
    args = (graphdef, static, hists, observation)

    # flatten parameter.get_value()(s) for iminuit
    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)  # ty:ignore[possibly-missing-attribute]

    # wrap loss that works on flat array
    @nnx.jit
    def iminuit_loss(flat_values: Float[Array, " nparams"]) -> Float[Array, ""]:
        dynamic.replace_by_pure_dict(unravel_fn(flat_values))
        return loss(dynamic, args)

    minuit = iminuit.Minuit(iminuit_loss, flat_values, grad=nnx.grad(iminuit_loss))  # ty:ignore[invalid-argument-type]
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.tol = 1e-5

    # minimize
    minuit.migrad(ncall=max_steps, use_simplex=False)

    # update dynamic part with bestfit values
    dynamic.replace_by_pure_dict(unravel_fn(jnp.array(minuit.values)))
    return dynamic.to_pure_dict(), minuit.fval


def fit_with_scipy(
    components,
    max_steps: int = 10_000,
):
    model, hists, observation = components
    model.mu.set_metadata(frozen=True)
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_parameter, ...)
    args = (graphdef, static, hists, observation)

    # update helper
    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # Wrapper for scipy (operates on flat array), similar to iminuit wrapper
    def scipy_loss(
        pars: Float[Array, "n_params"],  # noqa: F821
        *,
        dynamic: nnx.State = dynamic,
        args: Args = args,
    ) -> F64:
        dynamic.replace_by_pure_dict(unravel_fn(pars))
        # Compute loss
        return loss(dynamic, args)

    # Minimize using scipy
    result = minimize(
        scipy_loss,
        flat_values,
        method="SLSQP",
        jac=jax.grad(scipy_loss),
        options={"maxiter": max_steps, "ftol": 1e-8},
    )

    # update dynamic part with bestfit values
    dynamic.replace_by_pure_dict(unravel_fn(jnp.array(result.x)))
    return dynamic.to_pure_dict(), float(result.fun)


def fit_with_everwillow(
    components,
    max_steps: int = 150,
    interactive: bool = False,
):
    model, hists, observation = components
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_parameter, ...)

    args = (graphdef, static, hists, observation)
    init_state = sl.State.from_pytree(dynamic)

    if not interactive:
        result = ew.fit(
            partial(loss, args=args),
            params=init_state,
            max_steps=max_steps,
        )
    else:
        result = ew.ifit(
            partial(loss, args=args),
            params=init_state,
            max_steps=max_steps,
        )

    return result.params.to_pure_dict(), result.nll


def summarise_evermore_fit(params: Mapping[str, float], data: ModelData = DEFAULT_DATA):
    print(params)
    return expected_components(params, data=data)
