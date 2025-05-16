# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import polars as pl

import pydiverse.colspec as cs


class MySchema(cs.ColSpec):
    a = cs.Int64()
    b = cs.String()


def test_create_empty():
    df = MySchema.create_empty()
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Int64, pl.String]
    assert len(df) == 0
