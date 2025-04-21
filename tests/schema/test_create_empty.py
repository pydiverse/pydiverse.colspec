# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


class MySchema(cs.ColSpec):
    a = dy.Int64()
    b = dy.String()


def test_create_empty():
    df = MySchema.create_empty()
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Int64, pl.String]
    assert len(df) == 0
