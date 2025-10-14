# Getting Started

everwillow is a package that allows for evaluation of high-energy physics likelihoods with JAX.

## Installation (for users)

_not yet working:_
```bash
python -m pip install everwillow
```

From source:

```bash
git clone https://github.com/MoAly98/everwillow
cd everwillow
python -m pip install .
```


## Installation (for developers)

First clone the repository (probably a personal fork of it):
```shell
git clone https://github.com/MoAly98/everwillow
cd everwillow
```

Then, initialize the virtualenv once:
```shell
uv venv --python=3.12
source .venv/bin/activate
```

Install from source including dev dependencies:

```shell
uv pip install -e . --group=dev
```

Run tests:
```shell
pytest tests
```

Run pre-commit:
```shell
pre-commit run --all-files
```

Build docs:
```shell
cd docs
make html
open _build/html/index.html
```
