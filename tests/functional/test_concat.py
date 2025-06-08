# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.optional_dependency import dy, pl


class MyColSpec(cs.ColSpec):
    a = cs.Int64()


class SimpleCollection(pydiverse.colspec.collection.Collection):
    first: MyColSpec
    second: MyColSpec | None
    third: MyColSpec | None


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_concat():
    col1 = SimpleCollection.cast_polars_data({"first": pl.LazyFrame({"a": [1, 2, 3]})})
    col2 = SimpleCollection.cast_polars_data(
        {
            "first": pl.LazyFrame({"a": [4, 5, 6]}),
            "second": pl.LazyFrame({"a": [4, 5, 6]}),
        }
    )
    col3 = SimpleCollection.cast_polars_data(
        {
            "first": pl.LazyFrame({"a": [7, 8, 9]}),
            "second": pl.LazyFrame({"a": [7, 8, 9]}),
            "third": pl.LazyFrame({"a": [7, 8, 9]}),
        }
    )
    concat = dy.concat_collection_members([col1, col2, col3])
    assert concat["first"].collect().get_column("a").to_list() == list(range(1, 10))
    assert concat["second"].collect().get_column("a").to_list() == list(range(4, 10))
    assert concat["third"].collect().get_column("a").to_list() == list(range(7, 10))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_concat_empty():
    with pytest.raises(ValueError):
        dy.concat_collection_members([])
