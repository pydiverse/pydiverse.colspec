# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import polars as pl
from dataframely.testing import validation_mask

import pydiverse.colspec as cs


class CheckSchema(cs.ColSpec):
    a = cs.Int64(check=lambda col: (col < 5) | (col > 10))
    b = cs.String(min_length=3, check=lambda col: col.str.contains("x"))


def test_check():
    df = pl.DataFrame({"a": [7, 3, 15], "b": ["abc", "xyz", "x"]})
    _, failures = CheckSchema.filter(df)
    assert validation_mask(df, failures).to_list() == [False, True, False]
    assert failures.counts() == {"a|check": 1, "b|min_length": 1, "b|check": 1}
