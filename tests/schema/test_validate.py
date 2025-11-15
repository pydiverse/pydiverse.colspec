# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.exc import (
    RuleValidationError,
    SchemaError,
    ValidationError,
)
from pydiverse.colspec.optional_dependency import (
    C,
    ColExpr,
    assert_frame_equal,
    dy,
    pdt,
    pl,
)


class MyColSpec(cs.ColSpec):
    a = cs.Int64(primary_key=True)
    b = cs.String(nullable=False, max_length=5)
    c = cs.String()


class MyComplexColSpec(cs.ColSpec):
    a = cs.Int64()
    b = cs.Int64()

    @cs.rule_polars()
    @staticmethod
    def b_greater_a() -> pl.Expr:
        return pl.col("b") > pl.col("a")

    @cs.rule_polars(group_by=["a"])
    @staticmethod
    def b_unique_within_a() -> pl.Expr:
        return pl.col("b").n_unique() == 1

    @cs.rule()
    @staticmethod
    def b_greater_a2() -> ColExpr:
        return C.b > C.a

    @staticmethod
    @cs.rule(group_by=["a"])
    def b_unique_within_a2() -> ColExpr:
        return C.b.count() == 1  # TODO: n_unique() is not available in pdt


# -------------------------------------- COLUMNS ------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_missing_columns(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1], "b": [""]})
    with pytest.raises(SchemaError):
        MyColSpec.validate_polars(df)
    assert not MyColSpec.is_valid_polars(df)


# -------------------------------------- DTYPES -------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_dtype(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1], "b": [1], "c": [1]})
    with pytest.raises(
        SchemaError,
        match=r"2 columns with invalid dtype for schema",
    ):
        MyColSpec.validate_polars(df)
    assert not MyColSpec.is_valid_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_dtype_cast(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1], "b": [1], "c": [1]})
    actual = MyColSpec.validate_polars(df, cast=True)
    expected = pl.DataFrame({"a": [1], "b": ["1"], "c": ["1"]})
    assert_frame_equal(actual, expected)
    assert MyColSpec.is_valid_polars(df, cast=True)


# --------------------------------------- RULES -------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_column_contents(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1, 2, 3], "b": ["x", "longtext", None], "c": ["1", None, "3"]})
    with pytest.raises(
        ValidationError,
        match=r"2 rules failed validation",
    ):
        MyColSpec.validate_polars(df)
    assert not MyColSpec.is_valid_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_invalid_primary_key(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1, 1], "b": ["x", "y"], "c": ["1", "2"]})
    with pytest.raises(
        ValidationError,
        match=r"1 rules failed validation",
    ):
        MyColSpec.validate_polars(df)
    assert not MyColSpec.is_valid_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_violated_custom_rule_polars(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    df = df_type({"a": [1, 1, 2, 3, 3], "b": [2, 2, 2, 4, 5]})
    with pytest.raises(
        ValidationError,
        match=r"2 rules failed validation",
    ):
        MyComplexColSpec.validate_polars(df)
    assert not MyComplexColSpec.is_valid_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_violated_custom_rule():
    tbl = pdt.Table({"a": [1, 1, 2, 3, 3], "b": [2, 2, 2, 4, 5]})
    try:
        MyComplexColSpec.validate(tbl)
        raise AssertionError()  # above should raise
    except RuleValidationError as exc:
        assert exc.schema_errors == {"b_greater_a2": 1, "b_unique_within_a2": 4}
        assert len(exc.column_errors) == 0
    assert not MyComplexColSpec.is_valid(tbl)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_success_multi_row_strip_cast(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
):
    df = df_type({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1, None, None], "d": [1, 2, 3]})
    actual = MyColSpec.validate_polars(df, cast=True)
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": ["1", None, None]})
    assert_frame_equal(actual, expected)
    assert MyColSpec.is_valid_polars(df, cast=True)
