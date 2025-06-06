# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import pdt


class AliasColSpec(cs.ColSpec):
    a = cs.Int64(alias="hello world: col with space!")


def test_column_names():
    assert AliasColSpec.column_names() == ["hello world: col with space!"]


def test_validation():
    df = pl.DataFrame({"hello world: col with space!": [1, 2]})
    assert AliasColSpec.is_valid_polars(df)
    tbl = pdt.Table(df)
    assert AliasColSpec.is_valid(tbl)


def test_create_empty():
    df = AliasColSpec.create_empty_polars()
    assert AliasColSpec.is_valid_polars(df)
    tbl = pdt.Table(df)
    assert AliasColSpec.is_valid(tbl)
