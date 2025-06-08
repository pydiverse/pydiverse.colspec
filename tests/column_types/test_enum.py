# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import dy, pl
from pydiverse.colspec.testing.factory import create_colspec


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("dy_enum", "pl_dtype", "valid"),
    [
        (cs.Enum(["x", "y"]), pl.Enum(["x", "y"]), True),
        (cs.Enum(["y", "x"]), pl.Enum(["x", "y"]), False),
        (cs.Enum(["x"]), pl.Enum(["x", "y"]), False),
        (cs.Enum(["x", "y", "z"]), pl.Enum(["x", "y"]), False),
        (cs.Enum(["x", "y"]), pl.String(), False),
    ],
)
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    dy_enum: cs.Enum,
    pl_dtype: pl.Enum,
    valid: bool,
):
    schema = create_colspec("test", {"a": dy_enum})
    df = df_type({"a": ["x", "y", "x", "x"]}).cast(pl_dtype)
    assert schema.is_valid_polars(df) == valid


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("enum", [cs.Enum(["x", "y"]), cs.Enum(["y", "x"])])
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("data", "valid"),
    [
        ({"a": ["x", "y", "x", "x"]}, True),
        ({"a": ["x", "y", "x", "x"]}, True),
        ({"a": ["x", "y", "z"]}, False),
        ({"a": ["x", "y", "z"]}, False),
    ],
)
def test_valid_cast(
    enum: cs.Enum,
    data: Any,
    valid: bool,
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
):
    schema = create_colspec("test", {"a": enum})
    df = df_type(data)
    assert schema.is_valid_polars(df, cast=True) == valid
