# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import polars as pl
import pytest

import dataframely as dy


class AnySchema(cs.ColSpec):
    a = dy.Any()


@pytest.mark.parametrize(
    "data",
    [{"a": [None]}, {"a": [True, None]}, {"a": ["foo"]}, {"a": [3.5]}],
)
def test_any_dtype_passes(data: dict[str, Any]):
    df = pl.DataFrame(data)
    assert AnySchema.is_valid(df)
