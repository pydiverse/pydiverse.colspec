# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import decimal
import math
from typing import Any, cast

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import Column
from pydiverse.colspec.optional_dependency import Generator, dy, pl
from pydiverse.colspec.testing import (
    COLUMN_TYPES,
    INTEGER_COLUMN_TYPES,
    SUPERTYPE_COLUMN_TYPES,
    create_colspec,
)


@pytest.fixture()
def generator() -> Generator:
    return Generator(seed=42)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_sample_custom_check(column_type: type[Column], generator: Generator):
    column = column_type(check=lambda expr: expr)
    with pytest.raises(ValueError):
        column.sample_polars(generator)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
def test_sample_valid(column_type: type[Column], nullable: bool, generator: Generator):
    column = column_type(nullable=nullable)
    samples = sample_and_validate(column, generator, n=10_000)
    if nullable:
        assert math.isclose(cast(float, samples.is_null().mean()), 0.1, abs_tol=0.01)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_any(generator: Generator):
    column = cs.Any()
    samples = sample_and_validate(column, generator, n=100)
    assert samples.is_null().all()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
@pytest.mark.parametrize("min_kwargs", [{}, {"min": 10}, {"min_exclusive": 10}])
@pytest.mark.parametrize("max_kwargs", [{}, {"max": 100}, {"max_exclusive": 100}])
def test_sample_integer_min_max(
    column_type: type[Column],
    min_kwargs: dict[str, Any],
    max_kwargs: dict[str, Any],
    generator: Generator,
):
    column = column_type(**min_kwargs, **max_kwargs)
    samples = sample_and_validate(column, generator, n=10_000)
    if min_kwargs and max_kwargs:
        assert samples.min() == (
            min_kwargs["min"]
            if "min" in min_kwargs
            else min_kwargs["min_exclusive"] + 1
        )
        assert samples.max() == (
            max_kwargs["max"]
            if "max" in max_kwargs
            else max_kwargs["max_exclusive"] - 1
        )


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("column_type", INTEGER_COLUMN_TYPES)
def test_sample_integer_is_in(column_type: type[Column], generator: Generator):
    column = column_type(is_in=[4, 5, 6])  # type: ignore
    samples = sample_and_validate(column, generator, n=10_000)
    assert math.isclose(samples.mean(), 5, abs_tol=0.1)  # type: ignore
    assert samples.min() == 4
    assert samples.max() == 6


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "column",
    [
        cs.String(regex=".*", min_length=1),
        cs.String(regex=".*", max_length=2),
        cs.String(regex=".*", min_length=1, max_length=5),
    ],
)
def test_sample_string_invalid(column: Column, generator: Generator):
    with pytest.raises(ValueError):
        column.sample_polars(generator)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    "column",
    [
        cs.String(),
        cs.String(min_length=3),
        cs.String(max_length=5),
        cs.String(min_length=3, max_length=5),
        cs.String(regex="[abc]def(ghi)?"),
    ],
)
def test_sample_string(column: Column, generator: Generator):
    sample_and_validate(column, generator, n=10_000)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_decimal(generator: Generator):
    column = cs.Decimal(precision=3, scale=2, max_exclusive=decimal.Decimal("6.5"))
    samples = sample_and_validate(column, generator, n=100_000)
    assert samples.min() == decimal.Decimal("-9.99")
    assert samples.max() == decimal.Decimal("6.49")


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_date(generator: Generator):
    column = cs.Date(
        min=dt.date(2020, 1, 1), max=dt.date(2021, 12, 1), resolution="1mo"
    )
    samples = sample_and_validate(column, generator, n=100_000)
    assert samples.min() == dt.date(2020, 1, 1)
    assert samples.max() == dt.date(2021, 12, 1)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_date_9999(generator: Generator):
    column = cs.Date(
        min=dt.date(9998, 1, 1), max=dt.date(9999, 12, 1), resolution="1mo"
    )
    samples = sample_and_validate(column, generator, n=100_000)
    assert samples.min() == dt.date(9998, 1, 1)
    assert samples.max() == dt.date(9999, 12, 1)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_datetime(generator: Generator):
    column = cs.Datetime(
        min=dt.datetime(2020, 1, 1),
        max_exclusive=dt.datetime(2022, 1, 1),
        resolution="1d",
    )
    samples = sample_and_validate(column, generator, n=100_000)
    assert samples.min() == dt.datetime(2020, 1, 1)
    assert samples.max() == dt.datetime(2021, 12, 31)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_time(generator: Generator):
    column = cs.Time(min=dt.time(), max=dt.time(23, 59), resolution="1m")
    samples = sample_and_validate(column, generator, n=1_000_000)
    assert samples.min() == dt.time()
    assert samples.max() == dt.time(23, 59)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_duration(generator: Generator):
    column = cs.Duration(
        min=dt.timedelta(hours=24), max=dt.timedelta(hours=120), resolution="12h"
    )
    samples = sample_and_validate(column, generator, n=100_000)
    assert samples.min() == dt.timedelta(hours=24)
    assert samples.max() == dt.timedelta(hours=120)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_enum(generator: Generator):
    column = cs.Enum(["a", "b", "c"], nullable=False)
    samples = sample_and_validate(column, generator, n=10_000)
    assert set(samples) == {"a", "b", "c"}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_list(generator: Generator):
    column = cs.List(cs.String(regex="[abc]"), min_length=5, max_length=10)
    samples = sample_and_validate(column, generator, n=10_000)
    assert set(samples.list.len()) == set(range(5, 11)) | {None}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_struct(generator: Generator):
    column = cs.Struct({"a": cs.String(regex="[abc]"), "b": cs.String(regex="[a-z]xx")})
    samples = sample_and_validate(column, generator, n=10_000)
    assert len(samples) == 10_000


# --------------------------------------- UTILS -------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def sample_and_validate(column: Column, generator: Generator, *, n: int) -> pl.Series:
    samples = column.sample_polars(generator, n=n)
    col_spec = create_colspec("test", {"a": column})
    col_spec.validate_polars(pl.DataFrame({"a": samples}))
    return samples
