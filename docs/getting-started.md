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

Many of the examples in this documentation make use of optional dependencies,
notably:

- [`pyhs3`](https://github.com/pyhf/pyhs3) for declarative statistical models.
- [`evermore`](https://github.com/atlas-outreach-data-tools/evermore) for ATLAS
  and CMS style fitting utilities.
- [`pyhf`](https://pyhf.github.io) for histogram-based statistical models.

Install them alongside everwillow when needed:

```bash
uv pip install everwillow pyhs3 evermore pyhf
```

## Verifying the Installation

You can confirm that the package is available by importing it and printing the
version:

```python
import everwillow as ew

print(ew.__version__)
```

If everything is configured correctly you should see the installed version
number without any import warnings.

## Example Data and Tutorials

The repository ships with a handful of examples in the `examples/` directory,
covering common tasks such as unbinned fits and profiling nuisance parameters.
For a guided walkthrough see the {doc}`quickstart` guide, which demonstrates
everwillow with `pyhs3`, `evermore`, and `pyhf`.

## Building the Documentation Locally

The documentation is built with Sphinx using the configuration in
`docs/conf.py`. To build the HTML locally (after installing the ``docs`` extra):

```bash
uv pip install '.[docs]'
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser to view the result. The same
layout is used on Read the Docs.
