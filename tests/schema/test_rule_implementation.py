# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._rule import GroupRule, Rule
from dataframely.exc import ImplementationError, RuleImplementationError
from dataframely.testing import create_schema


def test_group_rule_group_by_error():
    with pytest.raises(
        ImplementationError,
        match=(
            r"Group validation rule 'b_greater_zero' has been implemented "
            r"incorrectly\. It references 1 columns which are not in the schema"
        ),
    ):
        create_schema(
            "test",
            columns={"a": dy.Integer(), "b": dy.Integer()},
            rules={
                "b_greater_zero": GroupRule(
                    (pl.col("b") > 0).all(), group_columns=["c"]
                )
            },
        )


def test_rule_implementation_error():
    with pytest.raises(
        RuleImplementationError, match=r"rule 'integer_rule'.*returns dtype 'Int64'"
    ):
        create_schema(
            "test",
            columns={"a": dy.Integer()},
            rules={"integer_rule": Rule(pl.col("a") + 1)},
        )


def test_group_rule_implementation_error():
    with pytest.raises(
        RuleImplementationError,
        match=(
            r"rule 'b_greater_zero'.*returns dtype 'List\(Boolean\)'.*"
            r"make sure to use an aggregation function"
        ),
    ):
        create_schema(
            "test",
            columns={"a": dy.Integer(), "b": dy.Integer()},
            rules={"b_greater_zero": GroupRule(pl.col("b") > 0, group_columns=["a"])},
        )


def test_rule_column_overlap_error():
    with pytest.raises(
        ImplementationError,
        match=r"Rules and columns must not be named equally but found 1 overlaps",
    ):
        create_schema(
            "test",
            columns={"test": dy.Integer(alias="a")},
            rules={"a": Rule(pl.col("a") > 0)},
        )
