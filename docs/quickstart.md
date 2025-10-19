# Quickstart

This tutorial walks through a simple counting experiment implemented with three
JAX-supporting modelling libraries: [`pyhs3`](https://github.com/scipp-atlas/pyhs3),
[`evermore`](https://github.com/pfackeldey/evermore), and
[`pyhf`](https://github.com/scikit-hep/pyhf). To run the examples locally use
`uv` with the `examples` dependency group, for instance
`uv run --group examples python docs/python/test_pyhs3_example.py`. Each script:

1. Builds the likelihood with the library’s native abstractions.
2. Exposes a pytree of parameters that everwillow can optimise.
3. Calls {func}`everwillow.fitting.fit` to obtain the best-fit values.

## The counting model

We use a single-bin signal-plus-background measurement with one shared shape
modifier and two log-normal normalisation nuisances.

- Observed events: 37.
- Signal template: {math}`s = 3` events, scaled by the strength parameter
  {math}`\mu`.
- Background templates: {math}`b_1 = 10` and {math}`b_2 = 20` with upward shape
  variations {math}`b_1^{\mathrm{up}} = 12` and {math}`b_2^{\mathrm{up}} = 23`.
- Log-normal rate modifiers with unit-width Gaussian controls:
  {math}`\theta_{\text{norm1}}` and {math}`\theta_{\text{norm2}}`.
- A single shape nuisance {math}`\theta_{\text{shape}}` shared by both
  backgrounds.

The total expectation in the Poisson term is

```{math}
\lambda(\mu, \theta) = \mu s
 + e^{\log(1.1)\,\theta_{\text{norm1}}}\Bigl(b_1 + \theta_{\text{shape}}(b_1^{\mathrm{up}}-b_1)\Bigr)
 + e^{\log(1.05)\,\theta_{\text{norm2}}}\Bigl(b_2 + \theta_{\text{shape}}(b_2^{\mathrm{up}}-b_2)\Bigr).
```

The negative log-likelihood combines the Poisson term with standard Gaussian
constraints for {math}`\theta_{\text{norm1}}`, {math}`\theta_{\text{norm2}}`,
and {math}`\theta_{\text{shape}}`. The tabs below show the runnable examples
in each toolkit; run them with ``uv run --group examples`` to reproduce the
fits.

::::{tab-set}
:::{tab-item} pyhs3
```{literalinclude} python/test_pyhs3_example.py
:language: python
```
```python
# Expected output:
# {'mu': 2.33, 'norm1': ~-7.2e-09, 'norm2': ~-1.6e-08, 'shape1': ~-2.0e-07}
```
:::
:::{tab-item} evermore
```{literalinclude} python/test_evermore_example.py
:language: python
```
```python
# Expected output:
# {'mu': 2.33, 'norm1': ~-2.5e-06, 'norm2': ~-2.6e-06, 'shape1': ~9.0e-05}
```
:::
:::{tab-item} pyhf
```{literalinclude} python/test_pyhf_example.py
:language: python
```
```python
# Expected output:
# {'shape1': ~-1.3e-07, 'mu': ~2.33, 'norm1': ~-5e-09, 'norm2': ~-1e-08}
```
:::
::::

## Next steps

::::{grid} 1
:gutter: 3

:::{grid-item-card} ✨ Explore statelib
:class-header: bg-info text-white
Review {doc}`statelib_overview` for a deeper look at `FlatState`, partitioning,
and other shared helpers that appear throughout the examples.
:::

:::{grid-item-card} 🧠 Extend the toolkit
:class-header: bg-dark text-white
When you need to integrate new optimisers or larger workflows, head to the
{doc}`architecture` notes for design guidance.
:::

:::{grid-item-card} 📚 API reference
:class-header: bg-light
Bookmark the {doc}`api/index` for detailed function signatures and supporting
utilities referenced across the quickstart.
:::

::::
