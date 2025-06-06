# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.exc import ValidationError


class FirstColSpec(cs.ColSpec):
    a = cs.Float64()


class SecondColSpec(cs.ColSpec):
    a = cs.String()


class Collection(pydiverse.colspec.collection.Collection):
    first: FirstColSpec
    second: SecondColSpec | None


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_valid(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    first = df_type({"a": [3]})
    second = df_type({"a": [1]})
    out = Collection.cast_polars_data({"first": first, "second": second})  # type: ignore
    assert (
        out.first.collect_schema()
        == FirstColSpec.create_empty_polars().collect_schema()
    )
    assert out.second is not None
    assert (
        out.second.collect_schema()
        == SecondColSpec.create_empty_polars().collect_schema()
    )


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_valid_optional(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    first = df_type({"a": [3]})
    out = Collection.cast_polars_data({"first": first})  # type: ignore
    assert (
        out.first.collect_schema()
        == FirstColSpec.create_empty_polars().collect_schema()
    )
    assert out.second is None


@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_cast_invalid_members(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    first = df_type({"a": [3]})
    with pytest.raises(ValueError):
        Collection.cast_polars_data({"third": first})  # type: ignore


def test_cast_invalid_member_schema_eager():
    first = pl.DataFrame({"b": [3]})
    with pytest.raises(ValidationError):
        Collection.cast_polars_data({"first": first})


def test_cast_invalid_member_schema_lazy():
    first = pl.LazyFrame({"b": [3]})
    collection = Collection.cast_polars_data({"first": first})
    with pytest.raises(ValidationError):
        collection.collect_all_polars()
