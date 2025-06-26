# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import random

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.exc import DtypeValidationError, ValidationError
from pydiverse.colspec.optional_dependency import (
    C,
    DataTypeClass,
    assert_frame_equal,
    dy,
    pdt,
    pl,
    validation_mask,
)


class MyColSpec(cs.ColSpec):
    a = cs.Int64(primary_key=True)
    b = cs.String(max_length=3)
    u = cs.UInt8


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("schema", "expected_columns"),
    [
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "c": pl.String}, None),
        (
            {"a": pl.Int64, "b": pl.String, "c": pl.String, "u": pl.UInt8},
            ["a", "b", "u"],
        ),
    ],
)
def test_filter_extra_columns(
    schema: dict[str, DataTypeClass], expected_columns: list[str] | None
):
    df = pl.DataFrame(schema=schema)
    try:
        filtered, _ = MyColSpec.filter_polars(df)
        assert expected_columns is not None
        assert set(filtered.columns) == set(expected_columns)
    except ValidationError:
        assert expected_columns is None
    except Exception as e:
        raise AssertionError() from e
    tbl = pdt.Table(df)
    try:
        filtered, _ = MyColSpec.filter(tbl)
        assert expected_columns is not None
        assert set(c.name for c in filtered) == set(expected_columns)
    except ValidationError:
        assert expected_columns is None
    except Exception as e:
        raise AssertionError() from e


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("schema", "cast", "success"),
    [
        ({"a": pl.Int64, "b": pl.Int64, "u": pl.UInt8}, False, False),
        ({"a": pl.String, "b": pl.String, "u": pl.UInt8}, True, True),
    ],
)
def test_filter_dtypes(schema: dict[str, DataTypeClass], cast: bool, success: bool):
    df = pl.DataFrame(schema=schema)
    try:
        MyColSpec.filter_polars(df, cast=cast)
        assert success
    except DtypeValidationError:
        assert not success
    except Exception as e:
        raise AssertionError() from e
    tbl = pdt.Table(df)
    try:
        MyColSpec.filter(tbl, cast=cast)
        assert success
    except DtypeValidationError:
        assert not success
    except Exception as e:
        raise AssertionError() from e


def fix(counts: dict[str, int]) -> dict[str, int]:
    """Fix the counts to be compatible with the expected output."""
    ret = {k.replace("_primary_key_", "primary_key"): v for k, v in counts.items()}
    return {k: v for k, v in sorted(ret.items(), key=lambda item: item[0])}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    ("data_a", "data_b", "failure_mask", "counts", "cooccurrence_counts"),
    [
        ([1, 2, 3], ["foo", "bar", None], [True, True, True], {}, {}),
        (
            [1, 2, 3],
            ["foo", "bar", "foobar"],
            [True, True, False],
            {"b|max_length": 1},
            {frozenset({"b|max_length"}): 1},
        ),
        (
            [1, 2, 2],
            ["foo", "bar", "foobar"],
            [True, False, False],
            {"b|max_length": 1, "primary_key": 2},
            {
                frozenset({"b|max_length", "primary_key"}): 1,
                frozenset({"primary_key"}): 1,
            },
        ),
    ],
)
def test_filter_failure(
    df_type: type[pl.DataFrame] | type[pl.LazyFrame],
    data_a: list[int],
    data_b: list[str | None],
    failure_mask: list[bool],
    counts: dict[str, int],
    cooccurrence_counts: dict[frozenset[str], int],
):
    df = df_type({"a": data_a, "b": data_b, "u": 0}).cast(dict(u=pl.UInt8))
    df_valid, failures = MyColSpec.filter_polars(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.filter(pl.Series(failure_mask)).lazy().collect(), df_valid)
    assert validation_mask(df, failures).to_list() == failure_mask
    assert len(failures) == (len(failure_mask) - sum(failure_mask))
    assert failures.counts() == counts
    assert failures.cooccurrence_counts() == cooccurrence_counts
    tbl = pdt.Table(df)
    df_valid, failures = MyColSpec.filter(tbl)
    assert isinstance(df_valid, pdt.Table)
    assert_frame_equal(
        df.filter(pl.Series(failure_mask)).lazy().collect(),
        df_valid >> pdt.export(pdt.Polars()),
    )
    # assert validation_mask(df, failures).to_list() == failure_mask
    # assert len(failures) == (len(failure_mask) - sum(failure_mask))
    assert fix(failures.counts()) == fix(counts)
    # assert failures.cooccurrence_counts() == cooccurrence_counts


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_no_rules(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    class TestColSpec(cs.ColSpec):
        a = cs.Int64(nullable=True)

    df = df_type({"a": [1, 2, 3]})
    df_valid, failures = TestColSpec.filter_polars(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}
    tbl = pdt.Table(df)
    df_valid, failures = TestColSpec.filter(tbl)
    assert isinstance(df_valid, pdt.Table)
    assert_frame_equal(df.lazy().collect(), df_valid >> pdt.export(pdt.Polars()))
    # assert len(failures) == 0
    assert failures.counts() == {}
    # assert failures.cooccurrence_counts() == {}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_with_rule_all_valid(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    class TestColSpec(cs.ColSpec):
        a = cs.String(min_length=3)

    df = df_type({"a": ["foo", "foobar"]})
    df_valid, failures = TestColSpec.filter_polars(df)
    assert isinstance(df_valid, pl.DataFrame)
    assert_frame_equal(df.lazy().collect(), df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}
    assert failures.cooccurrence_counts() == {}
    tbl = pdt.Table(df)
    df_valid, failures = TestColSpec.filter(tbl)
    assert isinstance(df_valid, pdt.Table)
    assert_frame_equal(df.lazy().collect(), df_valid >> pdt.export(pdt.Polars()))
    # assert len(failures) == 0
    assert failures.counts() == {}
    # assert failures.cooccurrence_counts() == {}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("df_type", [pl.DataFrame, pl.LazyFrame])
def test_filter_cast(df_type: type[pl.DataFrame] | type[pl.LazyFrame]):
    data = {
        # validation: [true, true, false, false, false, false]
        "a": ["1", "2", "foo", None, "123x", "9223372036854775808"],
        # validation: [true, false, true, true, false, true]
        "b": [20, 2000, None, 30, 3000, 50],
        "u": 0,
    }
    df = df_type(data).cast(dict(u=pl.UInt8))
    df_valid, failures = MyColSpec.filter_polars(df, cast=True)
    assert isinstance(df_valid, pl.DataFrame)
    assert df_valid.collect_schema().names() == MyColSpec.column_names()
    assert len(failures) == 5
    assert failures.counts() == {
        "a|dtype": 3,
        "a|nullability": 1,
        "b|max_length": 1,
        # NOTE: primary key constraint is violated as failing dtype casts results in
        # multiple null values.
        "primary_key": 1,
    }
    assert failures.cooccurrence_counts() == {
        frozenset({"a|nullability", "primary_key"}): 1,
        frozenset({"b|max_length"}): 1,
        frozenset({"a|dtype"}): 3,
    }
    tbl = pdt.Table(df)
    df_valid, failures = MyColSpec.filter(tbl, cast=True)
    assert isinstance(df_valid, pdt.Table)
    assert [c.name for c in df_valid] == MyColSpec.column_names()
    assert (
        failures.invalid_rows
        >> pdt.summarize(x=pdt.count())
        >> pdt.export(pdt.Scalar())
        == 5
    )
    assert failures.counts() == {
        "a|dtype": 3,
        "a|nullability": 1,
        "b|max_length": 1,
    }
    # assert failures.cooccurrence_counts() == {
    #     frozenset({"a|nullability", "primary_key"}): 1,
    #     frozenset({"b|max_length"}): 1,
    #     frozenset({"a|dtype"}): 3,
    # }


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_nondeterministic_lazyframe():
    n = 10_000
    lf = (
        pl.LazyFrame(
            {
                "a": range(n),
                "b": [random.choice(["foo", "foobar"]) for _ in range(n)],
                "u": 0,
            }
        )
        .cast(dict(u=pl.UInt8))
        .select(pl.all().shuffle())
    )

    filtered, _ = MyColSpec.filter_polars(lf)
    assert filtered.select(pl.col("b").n_unique()).item() == 1
    tbl = pdt.Table(lf)
    filtered, _ = MyColSpec.filter(tbl)
    assert (
        filtered
        >> pdt.group_by(filtered.b)
        >> pdt.summarize()
        >> pdt.summarize(x=pdt.count())
        >> pdt.export(pdt.Scalar)
        == 1
    )
