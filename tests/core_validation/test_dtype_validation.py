# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import re

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec._validation import DtypeCasting, validate_dtypes
from pydiverse.colspec.exc import DtypeValidationError
from pydiverse.colspec.optional_dependency import (
    C,
    assert_frame_equal,
    dy,
    pdt,
    pl,
    plexc,
)


@pytest.mark.skipif(C is None, reason="pydiverse.transform is required for this test")
@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("actual", "expected", "casting"),
    [
        ({"a": pl.Int64()}, {"a": cs.Int64()}, "none"),
        ({"a": pl.Int32()}, {"a": cs.Int64()}, "lenient"),
        ({"a": pl.Int32()}, {"a": cs.Int64()}, "strict"),
        (
            {"a": pl.Int32(), "b": pl.String()},
            {"a": cs.Int64(), "b": cs.UInt8()},
            "strict",
        ),
    ],
)
def test_success(
    actual: dict[str, pl.DataType],
    expected: dict[str, cs.Column],
    casting: DtypeCasting,
):
    df = pl.DataFrame(schema=actual)
    tbl = pdt.Table(df.lazy())
    lf = validate_dtypes(tbl, expected=expected, casting=casting) >> pdt.export(
        pdt.Polars
    )
    schema = lf.collect_schema()
    for key, col in expected.items():
        assert col.validate_dtype_polars(schema[key])


@pytest.mark.skipif(C is None, reason="pydiverse.transform is required for this test")
@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("actual", "expected", "error", "fail_columns"),
    [
        (
            {"a": pl.Int32()},
            {"a": cs.Int64()},
            r"1 columns have an invalid dtype.*\n.*got dtype 'Int32'",
            {"a"},
        ),
        (
            {"a": pl.Int32(), "b": pl.String()},
            {"a": cs.Int64(), "b": cs.UInt8()},
            r"2 columns have an invalid dtype",
            {"a", "b"},
        ),
    ],
)
def test_failure(
    actual: dict[str, pl.DataType],
    expected: dict[str, cs.Column],
    error: str,
    fail_columns: set[str],
):
    df = pl.DataFrame(schema=actual)
    tbl = pdt.Table(df.lazy())
    try:
        validate_dtypes(tbl, expected=expected, casting="none")
        raise AssertionError()  # above should raise
    except DtypeValidationError as exc:
        assert set(exc.errors.keys()) == fail_columns
        assert re.match(error, str(exc))


@pytest.mark.skipif(C is None, reason="pydiverse.transform is required for this test")
@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_lenient_casting():
    lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["foo", "12", "1313"]},
        schema={"a": pl.Int64(), "b": pl.String()},
    )
    tbl = pdt.Table(lf)
    actual = validate_dtypes(
        tbl,
        expected={"a": cs.UInt8(), "b": cs.UInt8()},
        casting="lenient",
    ) >> pdt.export(pdt.Polars(lazy=True))
    expected = pl.LazyFrame(
        {"a": [1, 2, 3], "b": [None, 12, None]},
        schema={"a": pl.UInt8(), "b": pl.UInt8()},
    )
    assert_frame_equal(actual, expected)


@pytest.mark.skipif(C is None, reason="pydiverse.transform is required for this test")
@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_strict_casting():
    lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["foo", "12", "1313"]},
        schema={"a": pl.Int64(), "b": pl.String()},
    )
    tbl = pdt.Table(lf)
    lf_valid = validate_dtypes(
        tbl,
        expected={"a": cs.UInt8(), "b": cs.UInt8()},
        casting="strict",
    ) >> pdt.export(pdt.Polars(lazy=True))
    with pytest.raises(
        plexc.InvalidOperationError, match=r'for 2 out of 3 values: \["foo", "1313"\]'
    ):
        lf_valid.collect()
