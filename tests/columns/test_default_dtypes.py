# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.columns._base import Column
from pydiverse.colspec.optional_dependency import dy, pl
from pydiverse.colspec.testing.factory import create_colspec


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("column", "dtype"),
    [
        (cs.Any(), pl.Null()),
        (cs.Bool(), pl.Boolean()),
        (cs.Date(), pl.Date()),
        (cs.Datetime(), pl.Datetime()),
        (cs.Time(), pl.Time()),
        (cs.Duration(), pl.Duration()),
        (cs.Decimal(), pl.Decimal()),
        (cs.Decimal(12), pl.Decimal(12)),
        (cs.Decimal(None, 8), pl.Decimal(None, 8)),
        (cs.Decimal(6, 2), pl.Decimal(6, 2)),
        (cs.Float(), pl.Float64()),
        (cs.Float32(), pl.Float32()),
        (cs.Float64(), pl.Float64()),
        (cs.Integer(), pl.Int64()),
        (cs.Int8(), pl.Int8()),
        (cs.Int16(), pl.Int16()),
        (cs.Int32(), pl.Int32()),
        (cs.Int64(), pl.Int64()),
        (cs.UInt8(), pl.UInt8()),
        (cs.UInt16(), pl.UInt16()),
        (cs.UInt32(), pl.UInt32()),
        (cs.UInt64(), pl.UInt64()),
        (cs.String(), pl.String()),
        (cs.List(cs.String()), pl.List(pl.String())),
        (cs.Struct({"a": cs.String()}), pl.Struct({"a": pl.String()})),
        (cs.Enum(["a", "b"]), pl.Enum(["a", "b"])),
    ],
)
def test_default_dtype(column: Column, dtype: pl.DataType):
    schema = create_colspec("test", {"a": column})
    df = schema.create_empty_polars()
    assert df.schema["a"] == dtype
    schema.validate_polars(df)
    assert schema.is_valid_polars(df)
