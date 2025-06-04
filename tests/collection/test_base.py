# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.colspec import dy


class MyFirstColSpec(cs.ColSpec):
    a = cs.UInt8(primary_key=True)


class MySecondColSpec(cs.ColSpec):
    a = cs.UInt16(primary_key=True)
    b = cs.Integer()


class MyCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec | None


def test_common_primary_keys():
    assert MyCollection.common_primary_keys() == ["a"]


def test_members():
    members = MyCollection.members()
    assert not members["first"].is_optional
    assert members["second"].is_optional


def test_member_col_specs():
    schemas = MyCollection.member_col_specs()
    assert schemas == {"first": MyFirstColSpec, "second": MySecondColSpec}


def test_required_members():
    required_members = MyCollection.required_members()
    assert required_members == {"first"}


def test_optional_members():
    optional_members = MyCollection.optional_members()
    assert optional_members == {"second"}


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
def test_cast():
    collection = MyCollection.cast_polars_data(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        },
    )
    assert (
        collection.first.collect_schema()
        == MyFirstColSpec.create_empty_polars().collect_schema()
    )
    assert collection.second is not None
    assert (
        collection.second.collect_schema()
        == MySecondColSpec.create_empty_polars().collect_schema()
    )


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
def test_cast2():
    collection = MyCollection._init_polars_data(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        },
    )
    collection = collection.cast_polars()
    assert (
        collection.first.collect_schema()
        == MyFirstColSpec.create_empty_polars().collect_schema()
    )
    assert collection.second is not None
    assert (
        collection.second.collect_schema()
        == MySecondColSpec.create_empty_polars().collect_schema()
    )


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
@pytest.mark.parametrize(
    "expected",
    [
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8}),
            "second": pl.LazyFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": pl.UInt16, "b": pl.Int64}
            ),
        },
        {"first": pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8})},
    ],
)
def test_to_dict(expected: dict[str, pl.LazyFrame]):
    collection = MyCollection.validate_polars_data(expected)

    # Check that export looks as expected
    observed = collection.to_dict()
    assert set(expected.keys()) == set(observed.keys())
    for key in expected.keys():
        pl.testing.assert_frame_equal(expected[key], observed[key])

    # Make sure that "roundtrip" validation works
    assert MyCollection.is_valid_polars_data(observed)


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
def test_collect_all():
    collection = MyCollection.cast_polars_data(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}).filter(pl.col("a") < 3),
            "second": pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(
                pl.col("b") <= 5
            ),
        }
    )
    out = collection.collect_all_polars()

    assert isinstance(out, MyCollection)
    assert out.first.explain() == 'DF ["a"]; PROJECT */1 COLUMNS'
    assert len(out.first.collect()) == 2
    assert out.second is not None
    assert out.second.explain() == 'DF ["a", "b"]; PROJECT */2 COLUMNS'
    assert len(out.second.collect()) == 2


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
def test_collect_all_optional():
    collection = MyCollection.cast_polars_data(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}
    )
    out = collection.collect_all_polars()

    assert isinstance(out, MyCollection)
    assert len(out.first.collect()) == 3
    assert out.second is None


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
@pytest.mark.parametrize(
    "read_fn", [MyCollection.scan_parquet, MyCollection.read_parquet]
)
def test_read_write_parquet(tmp_path: Path, read_fn: Callable[[Path], MyCollection]):
    collection = MyCollection.cast_polars_data(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2], "b": [10, 15]}),
        }
    )
    collection.write_parquet(tmp_path)

    read = read_fn(tmp_path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is not None
    assert read.second is not None
    assert_frame_equal(collection.second, read.second)


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
@pytest.mark.parametrize(
    "read_fn", [MyCollection.scan_parquet, MyCollection.read_parquet]
)
def test_read_write_parquet_optional(
    tmp_path: Path, read_fn: Callable[[Path], MyCollection]
):
    collection = MyCollection.cast_polars_data(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}
    )
    collection.write_parquet(tmp_path)

    read = read_fn(tmp_path)
    assert_frame_equal(collection.first, read.first)
    assert collection.second is None
    assert read.second is None
