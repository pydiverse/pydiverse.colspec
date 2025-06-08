# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import dy, pl


class MyColSpec(cs.ColSpec):
    a = cs.Int64()
    b = cs.String()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_create_empty():
    df = MyColSpec.create_empty_polars()
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Int64, pl.String]
    assert len(df) == 0
