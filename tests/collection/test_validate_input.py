# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import pydiverse.colspec as cs


class TestColSpec(cs.ColSpec):
    a = cs.Integer()


class MyCollection(cs.Collection):
    first: TestColSpec
    second: TestColSpec | None


def test_collection_missing_required_member():
    with pytest.raises(ValueError):
        MyCollection.validate({"second": pl.LazyFrame({"a": [1, 2, 3]})})


def test_collection_superfluous_member():
    with pytest.warns(Warning):
        MyCollection.validate(
            {
                "first": pl.LazyFrame({"a": [1, 2, 3]}),
                "third": pl.LazyFrame({"a": [1, 2, 3]}),
            },
        )
