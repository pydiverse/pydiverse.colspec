# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

import dataframely as dy


class AliasSchema(cs.ColSpec):
    a = dy.Int64(alias="hello world: col with space!")


def test_column_names():
    assert AliasSchema.column_names() == ["hello world: col with space!"]


def test_validation():
    df = pl.DataFrame({"hello world: col with space!": [1, 2]})
    assert AliasSchema.is_valid(df)


def test_create_empty():
    df = AliasSchema.create_empty()
    assert AliasSchema.is_valid(df)
