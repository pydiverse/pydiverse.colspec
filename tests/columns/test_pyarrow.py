# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.columns._utils import pydiverse_type_opinions
from pydiverse.colspec.optional_dependency import dy, pa
from pydiverse.colspec.testing import (
    ALL_COLUMN_TYPES,
    COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_colspec,
)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("column_type", set(ALL_COLUMN_TYPES) - {cs.Any})
def test_equal_to_polars_schema(column_type: type[cs.Column]):
    schema = create_colspec("test", {"a": column_type()})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty_polars().to_arrow().schema

    actual_dict = {f.name: f.type for f in actual}

    expected_dict = {f.name: pydiverse_type_opinions(f.type) for f in expected}
    assert actual_dict == expected_dict


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_equal_polars_schema_enum():
    schema = create_colspec("test", {"a": cs.Enum(["a", "b"])})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty_polars().to_arrow().schema
    assert actual == expected


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "inner",
    [c() for c in set(ALL_COLUMN_TYPES) - {cs.Any, cs.Enum}]
    + [cs.List(t()) for t in set(ALL_COLUMN_TYPES) - {cs.Any, cs.Enum}]
    + [cs.Struct({"a": t()}) for t in set(ALL_COLUMN_TYPES) - {cs.Any, cs.Enum}],
)
def test_equal_polars_schema_list(inner: cs.Column):
    schema = create_colspec("test", {"a": cs.List(inner)})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty_polars().to_arrow().schema
    # replace large_list with list
    expected = pa.schema(
        [
            pa.field(
                expected[0].name,
                pydiverse_type_opinions(expected[0].type),
                nullable=expected[0].nullable,
            )
        ]
    )
    assert actual == expected


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [cs.Struct({"a": t()}) for t in ALL_COLUMN_TYPES]
    + [cs.List(t()) for t in ALL_COLUMN_TYPES],
)
def test_equal_polars_schema_struct(inner: cs.Column):
    schema = create_colspec("test", {"a": cs.Struct({"a": inner})})
    actual = schema.pyarrow_schema()
    expected = schema.create_empty_polars().to_arrow().schema
    assert actual == expected


@pytest.mark.skipif(pa.Field is None, reason="pyarrow is required for this test")
@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information(column_type: type[cs.Column], nullable: bool):
    schema = create_colspec("test", {"a": column_type(nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_enum(nullable: bool):
    schema = create_colspec("test", {"a": cs.Enum(["a", "b"], nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [cs.List(t()) for t in ALL_COLUMN_TYPES]
    + [cs.Struct({"a": t()}) for t in ALL_COLUMN_TYPES],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_list(inner: cs.Column, nullable: bool):
    schema = create_colspec("test", {"a": cs.List(inner, nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "inner",
    [c() for c in ALL_COLUMN_TYPES]
    + [cs.Struct({"a": t()}) for t in ALL_COLUMN_TYPES]
    + [cs.List(t()) for t in ALL_COLUMN_TYPES],
)
@pytest.mark.parametrize("nullable", [True, False])
def test_nullability_information_struct(inner: cs.Column, nullable: bool):
    schema = create_colspec("test", {"a": cs.Struct({"a": inner}, nullable=nullable)})
    assert ("not null" in str(schema.pyarrow_schema())) != nullable


@pytest.mark.skipif(pa.Field is None, reason="pyarrow is required for this test")
def test_multiple_columns():
    schema = create_colspec("test", {"a": cs.Int32(nullable=False), "b": cs.Integer()})
    assert str(schema.pyarrow_schema()).split("\n") == ["a: int32 not null", "b: int64"]
