# pydiverse.colspec

[![CI](https://img.shields.io/github/actions/workflow/status/pydiverse/pydiverse.colspec/tests.yml?style=flat-square&branch=main&label=tests)](https://github.com/pydiverse/pydiverse.colspec/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/pydiversecolspec/badge/?version=latest&style=flat-square)](https://pydiversecolspec.readthedocs.io/en/latest)
[![pypi-version](https://img.shields.io/pypi/v/pydiverse-colspec.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/pydiverse-colspec)
[![conda-forge](https://img.shields.io/conda/pn/conda-forge/pydiverse-colspec?logoColor=white&logo=conda-forge&style=flat-square)](https://prefix.dev/channels/conda-forge/packages/pydiverse-colspec)

A data validation library that ensures type conformity of columns in SQL tables and polars data frames.
It can also validate constraints regarding the data as defined in a so-called column specification provided
by the user.

The purpose is to make data pipelines more robust by ensuring that data meets expectations and more readable by adding
type hints when working with tables and data frames.

ColSpec is founded on the ideas of [dataframely](https://github.com/Quantco/dataframely) which does exactly the same but
with focus on polars data frames. ColSpec delegates to dataframely in the back especially for features like sampling random
input data conforming to a given column specification. dataframely uses the term schema as it is also used in the polars
community. Since ColSpec also works with SQL databases where the term schema is used for a collection of tables, the
term is avoided as much as possible. The term column specification means exactly the same but avoids the confusion.

## Merit attribution

ColSpec is the brain child of [dataframely](https://github.com/Quantco/dataframely). Large parts of the codebase is code
duplicated from it. Unfortunately, integrating the SQL native validation into dataframely would have made it a less clean
solution for people who just focus on Polars. Thus the decision was made to replicate the same functionality in the
pydiverse library collection also with the benefit to enable smoother integration with other pydiverse libraries.

## Installation

To install pydiverse colspec try this:

```bash
git clone https://github.com/pydiverse/pydiverse.colspec.git
cd pydiverse.colspec

# Create the environment, activate it and install the pre-commit hooks
pixi install
pixi run pre-commit install
```

## Testing

Tests can be run with:

```bash
pixi run pytest
```

## Packaging and publishing to pypi and conda-forge using github actions

- bump version number in [pyproject.toml](pyproject.toml)
- set correct release date in [changelog.md](docs/source/changelog.md)
- push increased version number to `main` branch
- tag commit with `git tag <version>`, e.g. `git tag 0.7.0`
- `git push --tags`

The package should appear on https://pypi.org/project/pydiverse-colspec/ in a timely manner. It is normal that it takes
a few hours until the new package version is available on https://conda-forge.org/packages/.

### Packaging and publishing to Pypi manually

Packages are first released on test.pypi.org:

- bump version number in [pyproject.toml](pyproject.toml) (check consistency
  with [changelog.md](docs/source/changelog.md))
- push increased version number to `main` branch
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository testpypi dist/*`
- verify with https://test.pypi.org/search/?q=pydiverse.colspec

Finally, they are published via:

- `git tag <version>`
- `git push --tags`
- Attention: Please, only continue here, if automatic publishing fails for some reason!
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository pypi dist/*`

### Publishing package on conda-forge manually

Conda-forge packages are updated via:

- Attention: Please, only continue here, if automatic conda-forge publishing fails for longer than 24h!
- https://github.com/conda-forge/pydiverse-colspec-feedstock#updating-pydiverse-colspec-feedstock
- update `recipe/meta.yaml`
- test meta.yaml in pydiverse colspec repo: `conda-build build ../pydiverse-colspec-feedstock/recipe/meta.yaml`
- commit `recipe/meta.yaml` to branch of fork and submit PR
