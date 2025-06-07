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


def test_collection_missing_required_member():
    with pytest.raises(ValueError):
        MyCollection.validate_polars_data({"second": pl.LazyFrame({"a": [1, 2, 3]})})


def test_collection_superfluous_member():
    with pytest.warns(Warning):
        MyCollection.validate_polars_data(
            {
                "first": pl.LazyFrame({"a": [1, 2, 3]}),
                "third": pl.LazyFrame({"a": [1, 2, 3]}),
            },
        )
