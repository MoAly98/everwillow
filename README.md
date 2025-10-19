<div align="center">
<tr>
<td width="200px">
<img src="https://raw.githubusercontent.com/MoAly98/everwillow/main/images/logo.svg" alt="everwillow logo" width="180">
</td>

</tr>
</div>

---

everwillow is a statistical inference library for high-energy physics built on JAX pytrees and optimistix optimizers. It provides tools for fitting, profiling, and hypothesis testing with flexible parameter handling and parameter bounds via transformations. It works with any JAX-based statistical model.

## Quick Example

```python
import everwillow as ew

# Define your negative log-likelihood
def nll(params):
    return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2

# Fit with bounds
result = ew.fit(
    nll,
    params={"mu": 0.0, "sigma": 0.5},
    bounds={"mu": (0.0, 5.0), "sigma": (0.0, None)},
)

print(result.params)  # {'mu': 2.0, 'sigma': 1.0}
```

## Features

- **Pytree parameters**: Use dicts, nested structures, or custom classes
- **Parameter bounds**: Automatic transformations keep optimizers in unbounded space
- **Fixed parameters**: Freeze parameters by name or custom predicates
- **JAX-native**: Fast, JIT-compatible, GPU-ready
- **Model agnostic**: Works with any JAX-based statistical model that produces pytrees

## Installation

```bash
pip install everwillow
```

For development:

```bash
git clone https://github.com/MoAly98/everwillow.git
cd everwillow
uv sync --group dev
```

## Documentation

- [Quickstart Guide](https://everwillow.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://everwillow.readthedocs.io/en/latest/api/)
- [Architecture](https://everwillow.readthedocs.io/en/latest/architecture.html)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow.
