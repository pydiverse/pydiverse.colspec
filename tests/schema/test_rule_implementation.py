# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import polars as pl
import pytest

import pydiverse.colspec as cs
from pydiverse.colspec._rule import GroupRule, Rule
from pydiverse.colspec.exc import ImplementationError, RuleImplementationError
from pydiverse.colspec.testing.factory import create_colspec


def test_group_rule_group_by_error():
    with pytest.raises(
        ImplementationError,
        match=(
            r"Group validation rule 'b_greater_zero' has been implemented "
            r"incorrectly\. It references 1 columns which are not in the schema"
        ),
    ):
        create_colspec(
            "test",
            columns={"a": cs.Integer(), "b": cs.Integer()},
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
        create_colspec(
            "test",
            columns={"a": cs.Integer()},
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
        create_colspec(
            "test",
            columns={"a": cs.Integer(), "b": cs.Integer()},
            rules={"b_greater_zero": GroupRule(pl.col("b") > 0, group_columns=["a"])},
        )


def test_rule_column_overlap_error():
    with pytest.raises(
        ImplementationError,
        match=r"Rules and columns must not be named equally but found 1 overlaps",
    ):
        create_colspec(
            "test",
            columns={"test": cs.Integer(alias="a")},
            rules={"a": Rule(pl.col("a") > 0)},
        )
