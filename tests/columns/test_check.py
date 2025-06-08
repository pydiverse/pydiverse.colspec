# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import C, dy, pdt, pl, validation_mask


class CheckColSpec(cs.ColSpec):
    a = cs.Int64(check=lambda col: (col < 5) | (col > 10))
    b = cs.String(min_length=3, check=lambda col: col.str.contains("x"))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_check():
    df = pl.DataFrame({"a": [7, 3, 15], "b": ["abc", "xyz", "x"]})
    _, failures = CheckColSpec.filter_polars(df)
    assert validation_mask(df, failures).to_list() == [False, True, False]
    assert failures.counts() == {"a|check": 1, "b|min_length": 1, "b|check": 1}
    tbl = pdt.Table(df)
    _, failures = CheckColSpec.filter(tbl)
    # assert validation_mask(df, failures).to_list() == [False, True, False]
    assert failures.counts() == {"a|check": 1, "b|min_length": 1, "b|check": 1}
