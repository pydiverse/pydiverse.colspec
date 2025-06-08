# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import assert_frame_equal, dy, pl
from pydiverse.colspec.testing import evaluate_rules_polars, rules_from_exprs_polars


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_validate_min_length():
    column = cs.String(min_length=2)
    lf = pl.LazyFrame({"a": ["foo", "x"]})
    actual = evaluate_rules_polars(
        lf, rules_from_exprs_polars(column.validation_rules(pl.col("a")))
    )
    expected = pl.LazyFrame({"min_length": [True, False]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_validate_max_length():
    column = cs.String(max_length=2)
    lf = pl.LazyFrame({"a": ["foo", "x"]})
    actual = evaluate_rules_polars(
        lf, rules_from_exprs_polars(column.validation_rules(pl.col("a")))
    )
    expected = pl.LazyFrame({"max_length": [False, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_validate_regex():
    column = cs.String(regex="[0-9][a-z]$")
    lf = pl.LazyFrame({"a": ["33x", "3x", "44"]})
    actual = evaluate_rules_polars(
        lf, rules_from_exprs_polars(column.validation_rules(pl.col("a")))
    )
    expected = pl.LazyFrame({"regex": [True, True, False]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_validate_all_rules():
    column = cs.String(nullable=False, min_length=2, max_length=4)
    lf = pl.LazyFrame({"a": ["foo", "x", "foobar", None]})
    actual = evaluate_rules_polars(
        lf, rules_from_exprs_polars(column.validation_rules(pl.col("a")))
    )
    expected = pl.LazyFrame(
        {
            "min_length": [True, False, True, True],
            "max_length": [True, True, False, True],
            "nullability": [True, True, True, False],
        }
    )
    assert_frame_equal(actual, expected, check_column_order=False)
