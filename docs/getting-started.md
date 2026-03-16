# Getting Started

Whether you are exploring everwillow for the first time or integrating it into
an existing analysis, the steps below will get you up and running quickly.

## Installation

Everwillow is distributed as a Python package that targets Python 3.11+. The
recommended installation method is via [`uv`](https://github.com/astral-sh/uv),
which provides fast, reproducible environments:

```bash
uv pip install everwillow
```

If you prefer `pip`, simply run:

```bash
python -m pip install everwillow
```

## Example Data and Tutorials

The repository ships with a handful of examples in the `examples/` directory,
covering common tasks such as unbinned fits and profiling nuisance parameters.
For a guided walkthrough see the {doc}`quickstart` guide, which demonstrates
everwillow with `pyhs3`, `evermore`, and `pyhf`.

To run the examples from the documentation, install the optional
dependencies via the `examples` group:

```bash
uv pip install --with examples
```
