# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.columns._base import Column
from pydiverse.colspec.optional_dependency import dy, pl
from pydiverse.colspec.testing.factory import create_colspec


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_simple_struct():
    schema = create_colspec(
        "test", {"s": cs.Struct({"a": cs.Integer(), "b": cs.String()})}
    )
    assert schema.is_valid_polars(
        pl.DataFrame({"s": [{"a": 1, "b": "foo"}, {"a": 2, "b": "foo"}]})
    )


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("column", "dtype", "is_valid"),
    [
        (
            cs.Struct({"a": cs.Int64(), "b": cs.String()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            True,
        ),
        (
            cs.Struct({"b": cs.String(), "a": cs.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            True,
        ),
        (
            cs.Struct({"a": cs.Int64(), "b": cs.String(), "c": cs.String()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            pl.Struct({"a": pl.Int64(), "b": pl.String()}),
            False,
        ),
        (
            cs.Struct({"a": cs.Int64(), "b": cs.String()}),
            pl.Struct({"a": pl.Int64(), "c": pl.String()}),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            cs.String(),
            False,
        ),
        (
            cs.Struct({"a": cs.String(), "b": cs.Int64()}),
            pl.String(),
            False,
        ),
    ],
)
def test_validate_dtype(column: Column, dtype: pl.DataType, is_valid: bool):
    assert column.validate_dtype_polars(dtype) == is_valid


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_invalid_inner_type():
    schema = create_colspec("test", {"a": cs.Struct({"a": cs.Int64()})})
    assert not schema.is_valid_polars(pl.DataFrame({"a": [{"a": "1"}, {"a": "2"}]}))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_nested_structs():
    schema = create_colspec(
        "test",
        {
            "s1": cs.Struct(
                {
                    "s2": cs.Struct({"a": cs.Integer(), "b": cs.String()}),
                    "c": cs.String(),
                }
            )
        },
    )
    assert schema.is_valid_polars(
        pl.DataFrame({"s1": [{"s2": {"a": 1, "b": "foo"}, "c": "bar"}]})
    )


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_struct_with_pk():
    schema = create_colspec(
        "test",
        {"s": cs.Struct({"a": cs.String(), "b": cs.Integer()}, primary_key=True)},
    )
    df = pl.DataFrame(
        {"s": [{"a": "foo", "b": 1}, {"a": "bar", "b": 1}, {"a": "bar", "b": 1}]}
    )
    _, failures = schema.filter_polars(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s": [{"a": "bar", "b": 1}, {"a": "bar", "b": 1}]
    }
    assert failures.counts() == {"primary_key": 2}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_struct_with_rules():
    schema = create_colspec(
        "test", {"s": cs.Struct({"a": cs.String(min_length=2, nullable=False)})}
    )
    df = pl.DataFrame({"s": [{"a": "ab"}, {"a": "a"}, {"a": None}]})
    _, failures = schema.filter_polars(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s": [{"a": "a"}, {"a": None}]
    }
    assert failures.counts() == {"s|inner_a_nullability": 1, "s|inner_a_min_length": 1}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_nested_struct_with_rules():
    schema = create_colspec(
        "test",
        {
            "s1": cs.Struct(
                {"s2": cs.Struct({"a": cs.String(min_length=2, nullable=False)})}
            )
        },
    )
    df = pl.DataFrame(
        {"s1": [{"s2": {"a": "ab"}}, {"s2": {"a": "a"}}, {"s2": {"a": None}}]}
    )
    _, failures = schema.filter_polars(df)
    assert failures.invalid().to_dict(as_series=False) == {
        "s1": [{"s2": {"a": "a"}}, {"s2": {"a": None}}]
    }
    assert failures.counts() == {
        "s1|inner_s2_inner_a_nullability": 1,
        "s1|inner_s2_inner_a_min_length": 1,
    }


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_outer_inner_nullability():
    schema = create_colspec(
        "test",
        {
            "nullable": cs.Struct(
                inner={
                    "not_nullable1": cs.Integer(nullable=False),
                    "not_nullable2": cs.Integer(nullable=False),
                },
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"nullable": [None, None]})

    schema.validate_polars(df, cast=True)
