# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import pdt


class AnyColSpec(cs.ColSpec):
    a = cs.Any()


@pytest.mark.parametrize(
    "data",
    [{"a": [None]}, {"a": [True, None]}, {"a": ["foo"]}, {"a": [3.5]}],
)
def test_any_dtype_passes(data: dict[str, Any]):
    df = pl.DataFrame(data)
    assert AnyColSpec.is_valid_polars(df)
    tbl = pdt.Table(df)
    with pytest.raises(NotImplementedError, match="intentionally not implemented"):
        assert AnyColSpec.is_valid(tbl)
