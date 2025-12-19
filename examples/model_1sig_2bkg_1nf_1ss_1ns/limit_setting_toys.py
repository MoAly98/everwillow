"""Limit setting with toys: JAX/optimistix vs iminuit benchmark."""

import time

import iminuit
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import random, vmap

jax.config.update("jax_enable_x64", True)


def nll_counting(params: dict[str, float], data: dict[str, float]) -> jnp.ndarray:
    """Poisson(n_obs | mu*s + b) + Gaussian(b_aux | b, sigma_b)."""
    mu = params["mu"]
    b = params["b"]

    n_exp = mu * data["signal"] + b
    log_poisson = data["n_obs"] * jnp.log(n_exp) - n_exp
    log_gaussian = -0.5 * ((b - data["b_aux"]) / data["sigma_b"]) ** 2

    return -(log_poisson + log_gaussian)


def compute_qmu_iminuit(nll_fn, data, mu_test, init_params):
    """Compute qμ test statistic using iminuit."""
    param_names = sorted(init_params.keys())
    init = np.array([init_params[name] for name in param_names])

    def nll_array(params):
        params_dict = {name: float(params[i]) for i, name in enumerate(param_names)}
        return float(nll_fn(params_dict, data))

    # Unconditional fit
    m = iminuit.Minuit(nll_array, init)
    m.errordef = iminuit.Minuit.LIKELIHOOD
    m.migrad()
    nll_uncond = m.fval
    mu_hat = m.values[param_names.index("mu")]

    # Conditional fit (fix mu)
    m_cond = iminuit.Minuit(nll_array, init)
    m_cond.errordef = iminuit.Minuit.LIKELIHOOD
    mu_idx = param_names.index("mu")
    m_cond.values[mu_idx] = mu_test
    m_cond.fixed[mu_idx] = True
    m_cond.migrad()

    qmu = max(0.0, 2 * (m_cond.fval - nll_uncond))
    return qmu, mu_hat


def generate_toy(key, mu_true, signal, b_true, sigma_b):
    """Generate single toy dataset."""
    key1, key2 = random.split(key)
    n_exp = mu_true * signal + b_true
    n_obs = random.poisson(key1, n_exp)
    b_aux = random.normal(key2) * sigma_b + b_true
    return {"n_obs": n_obs, "b_aux": b_aux, "signal": signal, "sigma_b": sigma_b}


# Vectorize over toy generation
generate_toys_batch = vmap(generate_toy, in_axes=(0, None, None, None, None))


@jax.jit
def compute_qmu_optimistix_jit(
    data_n_obs, data_b_aux, data_signal, data_sigma_b, mu_test, init_mu, init_b
):
    """JIT-compiled qμ computation for optimistix."""
    data = {
        "n_obs": data_n_obs,
        "b_aux": data_b_aux,
        "signal": data_signal,
        "sigma_b": data_sigma_b,
    }

    def nll_array(theta, args):
        params = {"mu": theta[0], "b": theta[1]}
        return nll_counting(params, data)

    # Unconditional fit
    init = jnp.array([init_mu, init_b])
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        nll_array, solver, init, args=(), max_steps=5000, throw=False
    )
    nll_uncond = result.state.f_info.f

    # Conditional fit (fix mu)
    def nll_cond(theta_b, args):
        theta = jnp.array([mu_test, theta_b[0]])
        return nll_array(theta, args)

    init_nuis = jnp.array([init_b])
    result = optx.minimise(
        nll_cond, solver, init_nuis, args=(), max_steps=5000, throw=False
    )
    nll_cond_val = result.state.f_info.f

    qmu = jnp.maximum(0.0, 2 * (nll_cond_val - nll_uncond))
    return qmu


# Vectorize qmu computation over toys
compute_qmu_batch = vmap(
    compute_qmu_optimistix_jit, in_axes=(0, 0, None, None, None, None, None)
)


def calculate_cls(nll_fn, mu_test, config, n_toys, key, method="optimistix"):
    """Calculate CLs value using toys."""
    # Generate toys
    keys = random.split(key, 2 * n_toys)
    keys_b, keys_sb = keys[:n_toys], keys[n_toys:]

    toys_b = generate_toys_batch(
        keys_b, 0.0, config["signal"], config["b"], config["sigma_b"]
    )
    toys_sb = generate_toys_batch(
        keys_sb, mu_test, config["signal"], config["b"], config["sigma_b"]
    )

    init_params = {"mu": 1.0, "b": config["b"]}

    if method == "optimistix":
        # Vectorized JAX computation
        t0 = time.time()
        qmu_b = compute_qmu_batch(
            toys_b["n_obs"],
            toys_b["b_aux"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )
        qmu_sb = compute_qmu_batch(
            toys_sb["n_obs"],
            toys_sb["b_aux"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )
        fit_time = time.time() - t0

        # Asimov
        asimov_n = mu_test * config["signal"] + config["b"]
        qmu_obs = compute_qmu_optimistix_jit(
            asimov_n,
            config["b"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )

    else:
        # Sequential iminuit
        toys_b_list = [
            {k: float(v[i]) for k, v in toys_b.items()} for i in range(n_toys)
        ]
        toys_sb_list = [
            {k: float(v[i]) for k, v in toys_sb.items()} for i in range(n_toys)
        ]

        t0 = time.time()
        qmu_b = jnp.array(
            [
                compute_qmu_iminuit(nll_fn, toy, mu_test, init_params)[0]
                for toy in toys_b_list
            ]
        )
        qmu_sb = jnp.array(
            [
                compute_qmu_iminuit(nll_fn, toy, mu_test, init_params)[0]
                for toy in toys_sb_list
            ]
        )
        fit_time = time.time() - t0

        asimov = {
            "n_obs": mu_test * config["signal"] + config["b"],
            "b_aux": config["b"],
            "signal": config["signal"],
            "sigma_b": config["sigma_b"],
        }
        qmu_obs = compute_qmu_iminuit(nll_fn, asimov, mu_test, init_params)[0]

    # CLs
    cl_sb = jnp.mean(qmu_sb >= float(qmu_obs))
    cl_b = jnp.mean(qmu_b >= float(qmu_obs))
    cls = float(cl_sb / cl_b) if cl_b > 0 else 0.0

    return cls, fit_time


def calculate_cls_scan_jax(config, n_toys, key, mu_values):
    """Vectorized CLs calculation across mu values using JAX."""
    # Generate all toys once
    keys = random.split(key, 2 * n_toys)
    init_params = {"mu": 1.0, "b": config["b"]}

    def compute_cls_single_mu(mu_test):
        keys_b, keys_sb = keys[:n_toys], keys[n_toys:]
        toys_b = generate_toys_batch(
            keys_b, 0.0, config["signal"], config["b"], config["sigma_b"]
        )
        toys_sb = generate_toys_batch(
            keys_sb, mu_test, config["signal"], config["b"], config["sigma_b"]
        )

        qmu_b = compute_qmu_batch(
            toys_b["n_obs"],
            toys_b["b_aux"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )
        qmu_sb = compute_qmu_batch(
            toys_sb["n_obs"],
            toys_sb["b_aux"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )

        asimov_n = mu_test * config["signal"] + config["b"]
        qmu_obs = compute_qmu_optimistix_jit(
            asimov_n,
            config["b"],
            config["signal"],
            config["sigma_b"],
            mu_test,
            init_params["mu"],
            init_params["b"],
        )

        cl_sb = jnp.mean(qmu_sb >= qmu_obs)
        cl_b = jnp.mean(qmu_b >= qmu_obs)
        return cl_sb / jnp.maximum(cl_b, 1e-10)

    cls_values = vmap(compute_cls_single_mu)(mu_values)
    return cls_values


def find_limit_jax(cls_values, mu_values, target=0.05):
    """Find limit by linear interpolation."""
    idx = jnp.searchsorted(cls_values[::-1], target)
    mu_low = mu_values[::-1][idx - 1]
    mu_high = mu_values[::-1][idx]
    cls_low = cls_values[::-1][idx - 1]
    cls_high = cls_values[::-1][idx]
    limit = mu_low + (target - cls_low) * (mu_high - mu_low) / (cls_high - cls_low)
    return limit


def benchmark_limit_setting(n_toys=1000, seed=42):
    """Benchmark 95% CL limit: optimistix vs iminuit."""
    config = {"signal": 2.0, "b": 100.0, "sigma_b": 7.0}
    key = random.PRNGKey(seed)
    mu_values = jnp.linspace(0.5, 5.0, 10)

    print(f"\n95% CL limit with {n_toys} toys, {len(mu_values)} μ points")
    print(f"Model: s={config['signal']}, b={config['b']}±{config['sigma_b']}\n")

    # optimistix (vectorized)
    t0 = time.time()
    cls_optx = calculate_cls_scan_jax(config, n_toys, key, mu_values)
    limit_optx = find_limit_jax(cls_optx, mu_values)
    time_optx = time.time() - t0

    # iminuit (sequential)
    t0 = time.time()
    cls_iminuit = jnp.array(
        [
            calculate_cls(
                nll_counting, float(mu), config, n_toys, key, method="iminuit"
            )[0]
            for mu in mu_values
        ]
    )
    limit_iminuit = find_limit_jax(cls_iminuit, mu_values)
    time_iminuit = time.time() - t0

    print(f"optimistix: limit={limit_optx:.3f}, time={time_optx:.2f}s")
    print(f"iminuit:    limit={limit_iminuit:.3f}, time={time_iminuit:.2f}s")
    print(f"Speedup:    {time_iminuit / time_optx:.2f}x\n")


if __name__ == "__main__":
    benchmark_limit_setting(n_toys=100, seed=42)
