# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from pathlib import Path

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.colspec import ColSpecMeta, convert_to_dy_col_spec, dy
from pydiverse.colspec.exc import ImplementationError
from pydiverse.colspec.optional_dependency import SchemaMeta, assert_frame_equal, pl


class MyFirstColSpec(cs.ColSpec):
    a = cs.UInt8(primary_key=True)


class MySecondColSpec(cs.ColSpec):
    a = cs.UInt16(primary_key=True)
    b = cs.Integer


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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_collect_all_optional():
    collection = MyCollection.cast_polars_data(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}
    )
    out = collection.collect_all_polars()

    assert isinstance(out, MyCollection)
    assert len(out.first.collect()) == 3
    assert out.second is None


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
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


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_dataframely_columns_fail():
    class FailColSpec(cs.ColSpec):
        a = cs.Float64()
        b = dy.String()

    class FailColSpec2(cs.ColSpec):
        a = cs.Float64()
        b = dy.String

    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        FailColSpec.column_names()

    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        FailColSpec2.column_names()

    class FailCollection(cs.Collection):
        first: FailColSpec

    class FailCollection2(cs.Collection):
        first: FailColSpec2

    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        cs.collection.convert_collection_to_dy(FailCollection)
    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        cs.collection.convert_collection_to_dy(FailCollection2)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_dataframely_columns_fail_inheritance():
    class GoodColSpec(cs.ColSpec):
        a = cs.Float64()

    class SpecialMeta(ColSpecMeta, SchemaMeta):
        pass

    class GoodSchema(dy.Schema, metaclass=SpecialMeta):
        b = dy.Float64()

    assert GoodColSpec.column_names() == ["a"]
    assert GoodSchema.column_names() == ["b"]

    class FailColSpec(GoodColSpec, GoodSchema):
        pass

    with pytest.raises(
        ImplementationError,
        match="Don't mix Dataframely Schema with ColSpec classes in inheritance",
    ):
        convert_to_dy_col_spec(FailColSpec)

    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        FailColSpec.column_names()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_dataframely_conversion_success():
    class GoodColSpec(cs.ColSpec):
        a = cs.Float64()

    assert GoodColSpec.column_names() == ["a"]

    class GoodCollection(cs.Collection):
        first: GoodColSpec

    coll = cs.collection.convert_collection_to_dy(GoodCollection)
    assert coll.members()["first"].schema.column_names() == ["a"]
