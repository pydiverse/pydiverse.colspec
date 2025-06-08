# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import C, dy, pdt, pl


class AliasColSpec(cs.ColSpec):
    a = cs.Int64(alias="hello world: col with space!")


def test_column_names():
    assert AliasColSpec.column_names() == ["hello world: col with space!"]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_validation():
    df = pl.DataFrame({"hello world: col with space!": [1, 2]})
    assert AliasColSpec.is_valid_polars(df)
    tbl = pdt.Table(df)
    assert AliasColSpec.is_valid(tbl)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_create_empty():
    df = AliasColSpec.create_empty_polars()
    assert AliasColSpec.is_valid_polars(df)
    tbl = pdt.Table(df)
    assert AliasColSpec.is_valid(tbl)
