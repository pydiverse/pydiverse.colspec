# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from pydiverse.colspec._validation import validate_columns
from pydiverse.colspec.exc import ValidationError
from pydiverse.colspec.optional_dependency import C, pdt, pl


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_success():
    df = pl.DataFrame(schema={k: pl.Int64() for k in ["a", "b"]})
    tbl = pdt.Table(df)
    out_tbl = validate_columns(tbl, expected=["a"])
    assert set(c.name for c in out_tbl) == {"a"}


@pytest.mark.parametrize(
    ("actual", "expected", "error"),
    [
        (["a"], ["a", "b"], r"1 columns are missing: b;"),
        (["c"], ["a", "b"], r"2 columns are missing: a, b;"),
    ],
)
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_failure(actual: list[str], expected: list[str], error: str):
    df = pl.DataFrame(schema={k: pl.Int64() for k in actual})
    tbl = pdt.Table(df)
    with pytest.raises(ValidationError, match=error):
        validate_columns(tbl, expected=expected)
