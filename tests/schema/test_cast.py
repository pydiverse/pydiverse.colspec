# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import dy, pl, plexc


class MyColSpec(cs.ColSpec):
    a = cs.Float64()
    b = cs.String()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    "data",
    [
        {"a": [3], "b": [1]},
        {"a": [1], "b": [2], "c": [3]},
    ],
)
def test_cast_valid(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame], data: dict[str, Any]
):
    df = df_type(data)
    out = MyColSpec.cast_polars(df)
    assert isinstance(out, df_type)
    assert out.lazy().collect_schema() == MyColSpec.polars_schema()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_cast_invalid_schema_eager():
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(plexc.ColumnNotFoundError):
        MyColSpec.cast_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_cast_invalid_schema_lazy():
    lf = pl.LazyFrame({"a": [1]})
    lf = MyColSpec.cast_polars(lf)
    with pytest.raises(plexc.ColumnNotFoundError):
        lf.collect()
