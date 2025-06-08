# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: This file does not actually run any tests. Instead, it calls functions for which
# we simply want to ensure that our type checking works as desired. In some instances,
# we add 'type: ignore' markers here but, paired with "warn_unused_ignores = true", this
# allows testing that typing fails where we want it to without failing pre-commit
# checks.

import datetime
import decimal
import functools
from dataclasses import dataclass
from typing import Any

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec import ColSpec
from pydiverse.colspec.columns import ColExpr
from pydiverse.colspec.optional_dependency import C, dy, pl

# Note: To properly test the typing of the library,
# we also need to make sure that imported colspecs are properly processed.
from pydiverse.colspec.testing.typing import MyImportedColSpec

# pytestmark = pytest.mark.skip(reason="typing-only tests")

#                                        FRAMES                                        #
# ------------------------------------------------------------------------------------ #


class ColumnSpecification(ColSpec):
    a = cs.Int64()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_data_frame_lazy():
    df = ColumnSpecification.create_empty_polars()
    df2 = df.lazy()
    ColumnSpecification.validate_polars(df2)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_lazy_frame_lazy():
    df = ColumnSpecification.create_empty_polars().lazy()
    df2 = df.lazy()
    ColumnSpecification.validate_polars(df2)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_lazy_frame_collect():
    df = ColumnSpecification.create_empty_polars().lazy()
    df2 = df.collect()
    ColumnSpecification.validate_polars(df2)


# ------------------------------------------------------------------------------------ #
#                                      COLLECTION                                      #
# ------------------------------------------------------------------------------------ #


class MyFirstColSpec(ColSpec):
    a = cs.Integer(primary_key=True)


class MySecondColSpec(ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer()


@dataclass
class MyCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_collection_filter_return_value():
    _, failure = MyCollection.filter_polars_data(
        {
            "first": MyFirstColSpec.sample_polars(3),
            "second": MySecondColSpec.sample_polars(2),
        },
    )
    assert len(failure["first"]) == 0  # type: ignore[misc]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_collection_filter_return_value2():
    c = MyCollection(
        first=MyFirstColSpec.sample_polars(3), second=MySecondColSpec.sample_polars(2)
    )
    _, failure = c.filter_polars()
    assert len(failure["first"]) == 0  # type: ignore[misc]


# ------------------------------------------------------------------------------------ #
#                                       ITER ROWS                                      #
# ------------------------------------------------------------------------------------ #


Char = functools.partial(cs.String, min_length=1, max_length=1)
Flags = functools.partial(cs.Struct, inner={"x": Char(), "y": Char()})


class MyColSpec(ColSpec):
    a = cs.Int64()
    b = cs.Float32()
    c = cs.Enum(["a", "b", "c"])
    d = cs.Struct({"a": cs.Int64(), "b": cs.Struct({"c": cs.Enum(["a", "b"])})})
    e = cs.List(cs.Struct({"a": cs.Int64()}))
    f = cs.Datetime()
    g = cs.Date()
    h = cs.Any()
    some_decimal = cs.Decimal(12, 8)
    custom_col = Flags()
    custom_col_list = cs.List(Flags())

    @cs.rule()
    @staticmethod
    def b_greater_a() -> ColExpr:
        return C.b > C.a


@pytest.fixture
def my_colspec_df() -> MyColSpec:
    return MyColSpec.validate_polars(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "some_decimal": [decimal.Decimal("1.5")],
                "custom_col": [{"x": "a", "y": "b"}],
                "custom_col_list": [[{"x": "a", "y": "b"}]],
            }
        ),
        cast=True,
    )


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_assignment_correct_type(my_colspec_df: pl.DataFrame):
    entry = next(my_colspec_df.iter_rows(named=True))

    a: int = entry["a"]  # noqa: F841
    b: Any = entry["custom_col"]  # noqa: F841
    c: list[Any] = entry["custom_col_list"]  # noqa: F841


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_colspec_subtypes(my_colspec_df: pl.DataFrame):
    class MySubColSpec(MyColSpec):
        i = cs.Int64()

    class MySubSubColSpec(MySubColSpec):
        j = cs.Int64()

    my_sub_colspec_df = MySubColSpec.validate_polars(
        my_colspec_df.with_columns(i=pl.lit(2, dtype=pl.Int64))
    )
    entry1 = next(my_sub_colspec_df.iter_rows(named=True))

    a1: int = entry1["a"]  # noqa: F841
    i1: int = entry1["i"]  # noqa: F841

    my_sub_sub_colspec_df = MySubSubColSpec.validate_polars(
        my_sub_colspec_df.with_columns(j=pl.lit(2, dtype=pl.Int64))
    )
    entry2 = next(my_sub_sub_colspec_df.iter_rows(named=True))

    a2: int = entry2["a"]  # noqa: F841
    i2: int = entry2["i"]  # noqa: F841
    j2: int = entry2["j"]  # noqa: F841


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_assignment_wrong_type(my_colspec_df: dy.DataFrame[MyColSpec]):
    entry = next(my_colspec_df.iter_rows(named=True))

    a: int = entry["b"]  # type: ignore[assignment] # noqa: F841


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_read_only(my_colspec_df: dy.DataFrame[MyColSpec]):
    entry = next(my_colspec_df.iter_rows(named=True))

    entry["a"] = 1  # type: ignore[typeddict-readonly-mutated]


# @pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
# def test_iter_rows_missing_key(my_colspec_df: dy.DataFrame[MyColSpec]):
#     entry = next(my_colspec_df.iter_rows(named=True))
#
#     _ = entry["i"]  # type: ignore[misc]


# @pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
# def test_iter_rows_without_named(my_colspec_df: dy.DataFrame[MyColSpec]):
#     # Make sure we don't accidentally override the return type of `iter_rows` with
#     # `named=False`.
#     entry = next(my_colspec_df.iter_rows(named=False))
#
#     _ = entry["g"]  # type: ignore[call-overload]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_imported_colspec():
    my_imported_colspec_df = MyImportedColSpec.validate_polars(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "some_decimal": [decimal.Decimal("1.5")],
            }
        ),
        cast=True,
    )
    entry = next(my_imported_colspec_df.iter_rows(named=True))

    a: int = entry["a"]  # noqa: F841
    b: int = entry["b"]  # type: ignore[assignment] # noqa: F841
    entry["a"] = 1  # type: ignore[typeddict-readonly-mutated]
    # _ = entry["i"]  # type: ignore[misc]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_iter_rows_imported_subcolspec():
    class MySubFromImportedColSpec(MyImportedColSpec):
        i = cs.Int64()

    my_sub_from_imported_colspec_df = MySubFromImportedColSpec.validate_polars(
        pl.DataFrame(
            {
                "a": [1],
                "b": [1.0],
                "c": ["a"],
                "d": [{"a": 1, "b": {"c": "a"}}],
                "e": [[{"a": 1}]],
                "f": [datetime.datetime(2022, 1, 1, 0, 0, 0)],
                "g": [datetime.date(2022, 1, 1)],
                "h": [1],
                "some_decimal": [decimal.Decimal("1.5")],
                "i": [1],
            }
        ),
        cast=True,
    )
    entry = next(my_sub_from_imported_colspec_df.iter_rows(named=True))

    _ = entry["i"]  # noqa: F841
