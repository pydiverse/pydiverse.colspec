# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
from typing import Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import Column
from pydiverse.colspec.exc import DtypeValidationError
from pydiverse.colspec.optional_dependency import C, Generator, dy, pdt, pl
from pydiverse.colspec.testing.factory import create_colspec
from pydiverse.colspec.testing.rules import evaluate_rules


@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (cs.Date, {"min": dt.date(2020, 1, 15), "max": dt.date(2020, 1, 14)}),
        (cs.Date, {"min_exclusive": dt.date(2020, 1, 15), "max": dt.date(2020, 1, 15)}),
        (cs.Date, {"min": dt.date(2020, 1, 15), "max_exclusive": dt.date(2020, 1, 15)}),
        (
            cs.Date,
            {
                "min_exclusive": dt.date(2020, 1, 15),
                "max_exclusive": dt.date(2020, 1, 15),
            },
        ),
        (
            cs.Date,
            {"min": dt.date(2020, 1, 15), "min_exclusive": dt.date(2020, 1, 15)},
        ),
        (
            cs.Date,
            {"max": dt.date(2020, 1, 15), "max_exclusive": dt.date(2020, 1, 15)},
        ),
        (
            cs.Datetime,
            {"min": dt.datetime(2020, 1, 15), "max": dt.datetime(2020, 1, 14)},
        ),
        (
            cs.Datetime,
            {
                "min_exclusive": dt.datetime(2020, 1, 15),
                "max": dt.datetime(2020, 1, 15),
            },
        ),
        (
            cs.Datetime,
            {
                "min": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            cs.Datetime,
            {
                "min_exclusive": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            cs.Datetime,
            {
                "min": dt.datetime(2020, 1, 15),
                "min_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (
            cs.Datetime,
            {
                "max": dt.datetime(2020, 1, 15),
                "max_exclusive": dt.datetime(2020, 1, 15),
            },
        ),
        (cs.Time, {"min": dt.time(12, 15), "max": dt.time(12, 14)}),
        (cs.Time, {"min_exclusive": dt.time(12, 15), "max": dt.time(12, 15)}),
        (cs.Time, {"min": dt.time(12, 15), "max_exclusive": dt.time(12, 15)}),
        (
            cs.Time,
            {
                "min_exclusive": dt.time(12, 15),
                "max_exclusive": dt.time(12, 15),
            },
        ),
        (
            cs.Time,
            {"min": dt.time(12, 15), "min_exclusive": dt.time(12, 15)},
        ),
        (
            cs.Time,
            {"max": dt.time(12, 15), "max_exclusive": dt.time(12, 15)},
        ),
        (
            cs.Duration,
            {"min": dt.timedelta(seconds=15), "max": dt.timedelta(seconds=14)},
        ),
        (
            cs.Duration,
            {
                "min_exclusive": dt.timedelta(seconds=15),
                "max": dt.timedelta(seconds=15),
            },
        ),
        (
            cs.Duration,
            {
                "min": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            cs.Duration,
            {
                "min_exclusive": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            cs.Duration,
            {
                "min": dt.timedelta(seconds=15),
                "min_exclusive": dt.timedelta(seconds=15),
            },
        ),
        (
            cs.Duration,
            {
                "max": dt.timedelta(seconds=15),
                "max_exclusive": dt.timedelta(seconds=15),
            },
        ),
    ],
)
def test_args_consistency_min_max(column_type: type[Column], kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        column_type(**kwargs)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (cs.Date, {"min": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (cs.Date, {"min_exclusive": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (cs.Date, {"max": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (cs.Date, {"max_exclusive": dt.date(2020, 1, 10), "resolution": "1mo"}),
        (cs.Date, {"resolution": "1h"}),
        (cs.Date, {"resolution": "1d6h"}),
        (cs.Datetime, {"min": dt.datetime(2020, 1, 15, 11), "resolution": "1d"}),
        (
            cs.Datetime,
            {"min_exclusive": dt.datetime(2020, 1, 15, 11), "resolution": "1d"},
        ),
        (cs.Datetime, {"max": dt.datetime(2020, 1, 15, 11), "resolution": "1d"}),
        (
            cs.Datetime,
            {"max_exclusive": dt.datetime(2020, 1, 15, 11), "resolution": "1d"},
        ),
        (cs.Time, {"min": dt.time(12, 15), "resolution": "1h"}),
        (cs.Time, {"min_exclusive": dt.time(12, 15), "resolution": "1h"}),
        (cs.Time, {"max": dt.time(12, 15), "resolution": "1h"}),
        (cs.Time, {"max_exclusive": dt.time(12, 15), "resolution": "1h"}),
        (cs.Time, {"resolution": "1d"}),
        (cs.Time, {"resolution": "1d6h"}),
        (cs.Duration, {"min": dt.timedelta(minutes=30), "resolution": "1h"}),
        (cs.Duration, {"min_exclusive": dt.timedelta(minutes=30), "resolution": "1h"}),
        (cs.Duration, {"max": dt.timedelta(minutes=30), "resolution": "1h"}),
        (cs.Duration, {"max_exclusive": dt.timedelta(minutes=30), "resolution": "1h"}),
    ],
)
def test_args_resolution_invalid(column_type: type[Column], kwargs: dict[str, Any]):
    # Resolution errors are not caught by ColSpec. They only are raised when
    # converting to dataframely Schema in polars functions.
    # In Practice, it is recommended to use microsecond resolution for Datetimes,
    # Time, and Durations and second resolution for Dates. SQL Databases can represent
    # these precisions as well.
    class TestColSpec(cs.ColSpec):
        a = column_type(**kwargs)

    with pytest.raises(ValueError):
        # the ValueError is raised when converting ColSpec to dy.Schema
        TestColSpec.validate_polars(pl.DataFrame(dict(a=[])))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("column_type", "kwargs"),
    [
        (cs.Date, {"min": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (cs.Date, {"min_exclusive": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (cs.Date, {"max": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (cs.Date, {"max_exclusive": dt.date(2020, 1, 1), "resolution": "1mo"}),
        (cs.Date, {"resolution": "1d"}),
        (cs.Date, {"resolution": "1y"}),
        (cs.Datetime, {"min": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (cs.Datetime, {"min_exclusive": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (cs.Datetime, {"max": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (cs.Datetime, {"max_exclusive": dt.datetime(2020, 1, 15), "resolution": "1d"}),
        (cs.Time, {"min": dt.time(12), "resolution": "1h"}),
        (cs.Time, {"min_exclusive": dt.time(12), "resolution": "1h"}),
        (cs.Time, {"max": dt.time(12), "resolution": "1h"}),
        (cs.Time, {"max_exclusive": dt.time(12), "resolution": "1h"}),
        (cs.Time, {"resolution": "6h"}),
        (cs.Time, {"resolution": "15m"}),
        (cs.Duration, {"min": dt.timedelta(hours=3), "resolution": "1h"}),
        (cs.Duration, {"min_exclusive": dt.timedelta(hours=3), "resolution": "1h"}),
        (cs.Duration, {"max": dt.timedelta(hours=3), "resolution": "1h"}),
        (cs.Duration, {"max_exclusive": dt.timedelta(hours=3), "resolution": "1h"}),
    ],
)
def test_args_resolution_valid(column_type: type[Column], kwargs: dict[str, Any]):
    class TestColSpec(cs.ColSpec):
        a = column_type(**kwargs)

    with pytest.raises(DtypeValidationError):
        # this still tests that there is no ValueError when instantiating
        # dy.Column types
        TestColSpec.validate_polars(pl.DataFrame(dict(a=[])))


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("column", "values", "valid"),
    [
        (
            cs.Date(min=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(9999, 12, 31)],
            {"min": [False, True, True]},
        ),
        (
            cs.Date(min_exclusive=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(9999, 12, 31)],
            {"min_exclusive": [False, False, True]},
        ),
        (
            cs.Date(max=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(2020, 4, 2)],
            {"max": [True, True, False]},
        ),
        (
            cs.Date(max_exclusive=dt.date(2020, 4, 1)),
            [dt.date(2020, 3, 31), dt.date(2020, 4, 1), dt.date(2020, 4, 2)],
            {"max_exclusive": [True, False, False]},
        ),
        (
            cs.Time(min=dt.time(3)),
            [dt.time(2, 59), dt.time(3, 0, 0), dt.time(4)],
            {"min": [False, True, True]},
        ),
        (
            cs.Time(min_exclusive=dt.time(3)),
            [dt.time(2, 59), dt.time(3, 0, 0), dt.time(4)],
            {"min_exclusive": [False, False, True]},
        ),
        (
            cs.Time(max=dt.time(11, 59, 59, 999999)),
            [dt.time(11), dt.time(12), dt.time(13)],
            {"max": [True, False, False]},
        ),
        (
            cs.Time(max_exclusive=dt.time(12)),
            [dt.time(11), dt.time(12), dt.time(13)],
            {"max_exclusive": [True, False, False]},
        ),
        (
            cs.Datetime(min=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"min": [False, False, True, True, True]},
        ),
        (
            cs.Datetime(min_exclusive=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"min_exclusive": [False, False, False, True, True]},
        ),
        (
            cs.Datetime(max=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"max": [True, True, True, False, False]},
        ),
        (
            cs.Datetime(max_exclusive=dt.datetime(2020, 3, 1, hour=12)),
            [
                dt.datetime(2020, 2, 29, hour=14),
                dt.datetime(2020, 3, 1, hour=11),
                dt.datetime(2020, 3, 1, hour=12),
                dt.datetime(2020, 3, 1, hour=18),
                dt.datetime(2020, 3, 2, hour=11),
            ],
            {"max_exclusive": [True, True, False, False, False]},
        ),
        (
            cs.Duration(min=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"min": [False, True, True]},
        ),
        (
            cs.Duration(min_exclusive=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"min_exclusive": [False, False, True]},
        ),
        (
            cs.Duration(max=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"max": [True, True, False]},
        ),
        (
            cs.Duration(max_exclusive=dt.timedelta(days=1, seconds=14400)),
            [
                dt.timedelta(seconds=13000),
                dt.timedelta(days=1, seconds=14400),
                dt.timedelta(days=2),
            ],
            {"max_exclusive": [True, False, False]},
        ),
    ],
)
def test_validate_min_max(
    column: Column, values: list[Any], valid: dict[str, list[bool]]
):
    tbl = pdt.Table({"a": values})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.skipif(C is None, reason="pydiverse.transform is required for this test")
@pytest.mark.parametrize(
    ("column", "values", "valid"),
    [
        (
            cs.Date(resolution="1mo"),
            [dt.date(2020, 1, 1), dt.date(2021, 1, 15), dt.date(2022, 12, 1)],
            {},
            # pydiverse.transform does not support resolution validation
            # resolution should be enforced by convention in pydiverse.pipedag
            # {"resolution": [True, False, True]},
        ),
        (
            cs.Time(resolution="1h"),
            [dt.time(12, 0), dt.time(13, 15), dt.time(14, 0, 5)],
            {},
            # pydiverse.transform does not support resolution validation
            # resolution should be enforced by convention in pydiverse.pipedag
            # {"resolution": [True, False, False]},
        ),
        (
            cs.Datetime(resolution="1d"),
            [
                dt.datetime(2020, 4, 5),
                dt.datetime(2021, 1, 1, 12),
                dt.datetime(2022, 7, 10, 0, 0, 1),
            ],
            {},
            # pydiverse.transform does not support resolution validation
            # resolution should be enforced by convention in pydiverse.pipedag
            # {"resolution": [True, False, False]},
        ),
        (
            cs.Duration(resolution="12h"),
            [
                dt.timedelta(hours=12),
                dt.timedelta(days=2),
                dt.timedelta(hours=5),
                dt.timedelta(hours=12, minutes=30),
            ],
            {},
            # pydiverse.transform does not support resolution validation
            # resolution should be enforced by convention in pydiverse.pipedag
            # {"resolution": [True, True, False, False]},
        ),
    ],
)
def test_validate_resolution(
    column: Column, values: list[Any], valid: dict[str, list[bool]]
):
    tbl = pdt.Table({"a": values})
    actual = evaluate_rules(tbl, column.validation_rules(tbl.a))
    assert actual == valid


@pytest.mark.parametrize(
    "column",
    [
        cs.Datetime(
            min=dt.datetime(2020, 1, 1), max=dt.datetime(2021, 1, 1), resolution="1h"
        )
    ],
)
@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_resolution(column: cs.Column):
    generator = Generator(seed=42)
    samples = column.sample_polars(generator, n=10_000)
    schema = create_colspec("test", {"a": column})
    schema.validate_polars(samples.to_frame("a"))
