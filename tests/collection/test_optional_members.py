# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.optional_dependency import dy, pl

pytestmark = pytest.mark.skipif(
    dy.Column is None, reason="dataframely is required for this test"
)


class TestColSpec(cs.ColSpec):
    a = cs.Integer()


class MyCollection(pydiverse.colspec.collection.Collection):
    first: TestColSpec
    second: TestColSpec | None


def test_collection_optional_member():
    MyCollection.validate_polars_data({"first": pl.LazyFrame({"a": [1, 2, 3]})})


def test_filter_failure_info_keys_only_required():
    out, failure = MyCollection.filter_polars_data(
        {"first": pl.LazyFrame({"a": [1, 2, 3]})}
    )
    assert out.second is None
    assert set(failure.keys()) == {"first"}


def test_filter_failure_info_keys_required_and_optional():
    out, failure = MyCollection.filter_polars_data(
        {
            "first": pl.LazyFrame({"a": [1, 2, 3]}),
            "second": pl.LazyFrame({"a": [1, 2, 3]}),
        },
    )
    assert out.second is not None
    assert set(failure.keys()) == {"first", "second"}
