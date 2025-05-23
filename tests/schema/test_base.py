# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import polars as pl
import pytest
from dataframely._rule import Rule
from dataframely.exc import ImplementationError
from dataframely.testing import create_schema

import pydiverse.colspec as cs


class MySchema(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.String(primary_key=True)
    c = cs.Float64()
    e = cs.Any()


def test_column_names():
    _ = MySchema.e
    with pytest.raises(AttributeError):
        _ = MySchema.d
    assert MySchema.column_names() == ["a", "b", "c", "e"]


def test_columns():
    columns = MySchema.schema
    assert isinstance(columns["a"], cs.Integer)
    assert isinstance(columns["b"], cs.String)
    assert isinstance(columns["c"], cs.Float64)
    assert isinstance(columns["e"], cs.Any)


def test_nullability():
    columns = MySchema.get()
    assert not columns["a"].nullable
    assert not columns["b"].nullable
    assert columns["c"].nullable
    assert columns["e"].nullable


def test_primary_keys():
    assert MySchema.primary_keys() == ["a", "b"]


def test_no_rule_named_primary_key():
    with pytest.raises(ImplementationError):
        create_schema(
            "test",
            {"a": cs.String()},
            {"primary_key": Rule(pl.col("a").str.len_bytes() > 1)},
        )
