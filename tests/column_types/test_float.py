# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import sys
from typing import Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.columns.float import _BaseFloat
from pydiverse.colspec.optional_dependency import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    C,
    DataTypeClass,
    dy,
    pdt,
    pl,
)
from pydiverse.colspec.testing import FLOAT_COLUMN_TYPES
from pydiverse.colspec.testing.rules import evaluate_rules


class FloatColSpec(cs.ColSpec):
    a = cs.Float()


@pytest.mark.parametrize("column_type", FLOAT_COLUMN_TYPES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 2, "max": 1},
        {"min_exclusive": 2, "max": 2},
        {"min": 2, "max_exclusive": 2},
        {"min_exclusive": 2, "max_exclusive": 2},
        {"min": 2, "min_exclusive": 2},
        {"max": 2, "max_exclusive": 2},
    ],
)
def test_args_consistency_min_max(
    column_type: type[_BaseFloat], kwargs: dict[str, Any]
):
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (cs.Float, dict(min=float("-inf"))),
        (cs.Float, dict(max=float("inf"))),
        (cs.Float32, dict(min=-sys.float_info.max)),
        (cs.Float32, dict(max=sys.float_info.max)),
        (cs.Float64, dict(min=float("-inf"))),
        (cs.Float64, dict(max=float("inf"))),
    ],
)
def test_invalid_args(column_type: type[_BaseFloat], kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_any_integer_dtype_passes(dtype: DataTypeClass):
    df = pl.DataFrame(schema={"a": dtype})
    assert FloatColSpec.is_valid_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("dtype", [pl.Boolean, pl.String] + list(INTEGER_DTYPES))
def test_non_integer_dtype_fails(dtype: DataTypeClass):
    df = pl.DataFrame(schema={"a": dtype})
    assert not FloatColSpec.is_valid_polars(df)


@pytest.mark.skipif(C is None, reason="pydiverse-transform is required for this test")
@pytest.mark.parametrize("column_type", FLOAT_COLUMN_TYPES)
@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"min": [False, False, True, True, True]}),
        (False, {"min_exclusive": [False, False, False, True, True]}),
    ],
)
def test_validate_min(
    column_type: type[_BaseFloat], inclusive: bool, valid: dict[str, list[bool]]
):
    kwargs = {("min" if inclusive else "min_exclusive"): 3}
    column = column_type(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(C is None, reason="pydiverse-transform is required for this test")
@pytest.mark.parametrize("column_type", FLOAT_COLUMN_TYPES)
@pytest.mark.parametrize(
    ("inclusive", "valid"),
    [
        (True, {"max": [True, True, True, False, False]}),
        (False, {"max_exclusive": [True, True, False, False, False]}),
    ],
)
def test_validate_max(
    column_type: type[_BaseFloat], inclusive: bool, valid: dict[str, list[bool]]
):
    kwargs = {("max" if inclusive else "max_exclusive"): 3}
    column = column_type(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(C is None, reason="pydiverse-transform is required for this test")
@pytest.mark.parametrize("column_type", FLOAT_COLUMN_TYPES)
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
    column_type: type[_BaseFloat],
    min_inclusive: bool,
    max_inclusive: bool,
    valid: dict[str, list[bool]],
):
    kwargs = {
        ("min" if min_inclusive else "min_exclusive"): 2,
        ("max" if max_inclusive else "max_exclusive"): 4,
    }
    column = column_type(**kwargs)  # type: ignore
    tbl = pdt.Table({"a": [1, 2, 3, 4, 5]})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_unchecked_min_0():
    column = dy.Float(min=0, max=10)
    actual = column._sample_unchecked(dy.random.Generator(), n=10000)
    assert actual.min() >= 0, "There should be no negative values"  # type: ignore
