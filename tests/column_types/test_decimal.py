# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import decimal
from typing import Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    C,
    DataTypeClass,
    pdt,
    pl,
)
from pydiverse.colspec.testing.rules import evaluate_rules


class DecimalColSpec(cs.ColSpec):
    a = cs.Decimal()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": decimal.Decimal(2), "max": decimal.Decimal(1)},
        {"min_exclusive": decimal.Decimal(2), "max": decimal.Decimal(2)},
        {"min": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
        {"min_exclusive": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
        {"min": decimal.Decimal(2), "min_exclusive": decimal.Decimal(2)},
        {"max": decimal.Decimal(2), "max_exclusive": decimal.Decimal(2)},
    ],
)
def test_args_consistency_min_max(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        cs.Decimal(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale=1, min=decimal.Decimal("3.14")),
        dict(scale=1, min_exclusive=decimal.Decimal("3.14")),
        dict(scale=1, max=decimal.Decimal("3.14")),
        dict(scale=1, max_exclusive=decimal.Decimal("3.14")),
        dict(min=decimal.Decimal(float("inf"))),
        dict(max=decimal.Decimal(float("inf"))),
        dict(scale=0, precision=2, min=decimal.Decimal("100")),
        dict(scale=0, precision=2, max=decimal.Decimal("100")),
    ],
)
def test_invalid_args(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        cs.Decimal(**kwargs)


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    "dtype", [pl.Decimal, pl.Decimal(12), pl.Decimal(None, 8), pl.Decimal(6, 2)]
)
def test_any_decimal_dtype_passes(dtype: DataTypeClass):
    df = pl.DataFrame(schema={"a": dtype})
    tbl = pdt.Table(df)
    assert DecimalColSpec.is_valid(tbl)


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    "dtype", [pl.Boolean, pl.String] + list(INTEGER_DTYPES) + list(FLOAT_DTYPES)
)
def test_non_decimal_dtype_fails(dtype: DataTypeClass):
    if dtype == pl.Int128:
        # this type is not supported by pydiverse libraries, yet
        return
    df = pl.DataFrame(schema={"a": dtype})
    tbl = pdt.Table(df)
    assert not DecimalColSpec.is_valid(tbl)


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"min": [False, False, True, True, True]}),
        (False, {"min_exclusive": [False, False, False, True, True]}),
    ],
)
def test_validate_min(inclusive: bool, valid: dict[str, list[bool]]):
    kwargs = {("min" if inclusive else "min_exclusive"): 3}
    column = cs.Decimal(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"max": [True, True, True, False, False]}),
        (False, {"max_exclusive": [True, True, False, False, False]}),
    ],
)
def test_validate_max(inclusive: bool, valid: dict[str, list[bool]]):
    kwargs = {("max" if inclusive else "max_exclusive"): decimal.Decimal(3)}
    column = cs.Decimal(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("min_inclusive", "max_inclusive", "valid"),
    [
        (
            True,
            True,
            {
                "min": [False, True, True, True, True],
                "max": [True, True, True, True, False],
            },
        ),
        (
            True,
            False,
            {
                "min": [False, True, True, True, True],
                "max_exclusive": [True, True, True, False, False],
            },
        ),
        (
            False,
            True,
            {
                "min_exclusive": [False, False, True, True, True],
                "max": [True, True, True, True, False],
            },
        ),
        (
            False,
            False,
            {
                "min_exclusive": [False, False, True, True, True],
                "max_exclusive": [True, True, True, False, False],
            },
        ),
    ],
)
def test_validate_range(
    min_inclusive: bool,
    max_inclusive: bool,
    valid: dict[str, list[bool]],
):
    kwargs = {
        ("min" if min_inclusive else "min_exclusive"): decimal.Decimal(2),
        ("max" if max_inclusive else "max_exclusive"): decimal.Decimal(4),
    }
    column = cs.Decimal(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid
