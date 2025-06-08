# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from pydiverse.colspec import GroupRulePolars, RulePolars
from pydiverse.colspec.optional_dependency import assert_frame_equal, dy, pl
from pydiverse.colspec.testing import evaluate_rules_polars


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_single_column_single_rule():
    lf = pl.LazyFrame({"a": [1, 2]})
    rules = {
        "a|min": RulePolars(pl.col("a") >= 2),
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame({"a|min": [False, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_single_column_multi_rule():
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    rules = {
        "a|min": RulePolars(pl.col("a") >= 2),
        "a|max": RulePolars(pl.col("a") <= 2),
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame(
        {"a|min": [False, True, True], "a|max": [True, True, False]}
    )
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_multi_column_multi_rule():
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rules = {
        "a|min": RulePolars(pl.col("a") >= 2),
        "a|max": RulePolars(pl.col("a") <= 2),
        "b|even": RulePolars(pl.col("b") % 2 == 0),
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame(
        {
            "a|min": [False, True, True],
            "a|max": [True, True, False],
            "b|even": [True, False, True],
        }
    )
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_cross_column_rule():
    lf = pl.LazyFrame({"a": [1, 1, 2, 2], "b": [1, 1, 1, 2]})
    rules = {"primary_key": RulePolars(~pl.struct("a", "b").is_duplicated())}
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame({"primary_key": [False, False, True, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_group_rule():
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, RulePolars] = {
        "unique_b": GroupRulePolars(pl.col("b").n_unique() == 1, group_columns=["a"])
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame({"unique_b": [True, True, False, False, True]})
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_simple_rule_and_group_rule():
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, RulePolars] = {
        "b|max": RulePolars(pl.col("b") <= 1),
        "unique_b": GroupRulePolars(pl.col("b").n_unique() == 1, group_columns=["a"]),
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame(
        {
            "b|max": [True, True, True, False, True],
            "unique_b": [True, True, False, False, True],
        }
    )
    assert_frame_equal(actual, expected, check_column_order=False)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_multiple_group_rules():
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 1]})
    rules: dict[str, RulePolars] = {
        "unique_b": GroupRulePolars(pl.col("b").n_unique() == 1, group_columns=["a"]),
        "sum_b": GroupRulePolars(pl.col("b").sum() >= 2, group_columns=["a"]),
        "group_count": GroupRulePolars(pl.len() >= 2, group_columns=["a", "b"]),
    }
    actual = evaluate_rules_polars(lf, rules)

    expected = pl.LazyFrame(
        {
            "unique_b": [True, True, False, False, True],
            "sum_b": [True, True, True, True, False],
            "group_count": [True, True, False, False, False],
        }
    )
    assert_frame_equal(actual, expected)
