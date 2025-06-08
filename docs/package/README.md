# pydiverse.colspec

[![CI](https://github.com/pydiverse/pydiverse.colspec/actions/workflows/tests.yml/badge.svg)](https://github.com/pydiverse/pydiverse.colspec/actions/workflows/tests.yml)

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

## Usage

pydiverse.colspec can either be installed via pypi with `pip install pydiverse-colspec` or via
conda-forge with `conda install pydiverse-colspec -c conda-forge`.
