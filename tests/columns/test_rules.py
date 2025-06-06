# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

from pydiverse.colspec import Column
from pydiverse.colspec.optional_dependency import pdt
from pydiverse.colspec.testing import COLUMN_TYPES, SUPERTYPE_COLUMN_TYPES
from pydiverse.colspec.testing.rules import evaluate_rules


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_rule_count_nullability(column_type: type[Column], nullable: bool):
    column = column_type(nullable=nullable)
    assert len(column.validation_rules(pl.col("a"))) == int(not nullable)


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_nullability_rule_for_primary_key(column_type: type[Column]):
    column = column_type(primary_key=True)
    assert len(column.validation_rules(pl.col("a"))) == 1


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_nullability_rule(column_type: type[Column]):
    column = column_type(nullable=False)
    dtype = column.dtype()
    polars_type = dtype.to_polars()
    lf = pl.LazyFrame({"a": [None]}, schema={"a": polars_type})
    tbl = pdt.Table(lf)
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    expected = {"nullability": [False]}
    assert actual == expected
