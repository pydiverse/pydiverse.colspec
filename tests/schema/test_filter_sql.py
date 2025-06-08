# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import random

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.exc import DtypeValidationError, ValidationError
from pydiverse.colspec.optional_dependency import (
    C,
    DataTypeClass,
    assert_frame_equal,
    pdt,
    pl,
    sa,
)
from pydiverse.colspec.testing.assert_equal import assert_table_equal

if sa.Column is not None:
    engine = sa.create_engine("duckdb:///:memory:")
else:
    engine = None


def sql_table(df: pl.DataFrame, *, name: str) -> pdt.Table:
    df.write_database(name, engine, if_table_exists="replace")
    return pdt.Table(name, pdt.SqlAlchemy(engine))


class MyColSpec(cs.ColSpec):
    a = cs.Int64(primary_key=True)
    b = cs.String(max_length=3)


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize(
    ("schema", "expected_columns"),
    [
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "c": pl.String}, None),
        ({"a": pl.Int64, "b": pl.String, "c": pl.String}, ["a", "b"]),
    ],
)
def test_filter_extra_columns(
    schema: dict[str, DataTypeClass], expected_columns: list[str] | None
):
    tbl = sql_table(pl.DataFrame(schema=schema), name="tbl")
    try:
        filtered, _ = MyColSpec.filter(tbl)
        filtered >>= pdt.export(pdt.Polars)
        assert expected_columns is not None
        assert set(filtered.columns) == set(expected_columns)
    except ValidationError:
        assert expected_columns is None


@pytest.mark.parametrize(
    ("schema", "cast", "success"),
    [
        ({"a": pl.Int64, "b": pl.Int64}, False, False),
        ({"a": pl.String, "b": pl.String}, True, True),
    ],
)
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_dtypes(schema: dict[str, DataTypeClass], cast: bool, success: bool):
    tbl = sql_table(pl.DataFrame(schema=schema), name="tbl")
    try:
        MyColSpec.filter(tbl, cast=cast)
        assert success
    except DtypeValidationError:
        assert not success


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
            {"b|max_length": 1, "_primary_key_": 2},
            {
                frozenset({"b|max_length", "_primary_key_"}): 1,
                frozenset({"_primary_key_"}): 1,
            },
        ),
    ],
)
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_failure(
    data_a: list[int],
    data_b: list[str | None],
    failure_mask: list[bool],
    counts: dict[str, int],
    cooccurrence_counts: dict[frozenset[str], int],
):
    tbl = sql_table(pl.DataFrame({"a": data_a, "b": data_b}), name="tbl")
    tbl_valid, failures = MyColSpec.filter(tbl)
    assert isinstance(tbl_valid, pdt.Table)
    assert_frame_equal(
        (tbl >> pdt.export(pdt.Polars)).filter(pl.Series(failure_mask)),
        tbl_valid >> pdt.export(pdt.Polars),
        check_row_order=False,
    )
    assert len(failures) == (len(failure_mask) - sum(failure_mask))
    assert failures.counts() == counts


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_no_rules():
    class TestColSpec(cs.ColSpec):
        a = cs.Int64(nullable=True)

    tbl = sql_table(pl.DataFrame({"a": [1, 2, 3]}), name="tbl")
    df_valid, failures = TestColSpec.filter(tbl)
    assert isinstance(df_valid, pdt.Table)
    assert_table_equal(tbl, df_valid)
    assert len(failures) == 0
    assert failures.counts() == {}


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_with_rule_all_valid():
    class TestColSpec(cs.ColSpec):
        a = cs.String(min_length=3)

    tbl = sql_table(pl.DataFrame({"a": ["foo", "foobar"]}), name="tbl")
    tbl_valid, failures = TestColSpec.filter(tbl)
    assert isinstance(tbl_valid, pdt.Table)
    assert_table_equal(tbl, tbl_valid)
    assert len(failures) == 0
    assert failures.counts() == {}


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_cast():
    data = {
        # validation: [true, true, false, false, false, false]
        "a": ["1", "2", "foo", None, "123x", "9223372036854775808"],
        # validation: [true, false, true, true, false, true]
        "b": [20, 2000, None, 30, 3000, 50],
    }
    tbl = sql_table(pl.DataFrame(data), name="tbl")
    tbl_valid, failures = MyColSpec.filter(tbl, cast=True)
    assert isinstance(tbl_valid, pdt.Table)
    assert [col.name for col in tbl_valid] == MyColSpec.column_names()
    assert len(failures) == 5
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


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_nondeterministic_tbl():
    n = 10_000
    tbl = sql_table(
        pl.DataFrame(
            {
                "a": range(n),
                "b": [random.choice(["foo", "foobar"]) for _ in range(n)],
            }
        ).select(pl.all().shuffle()),
        name="tbl",
    )

    filtered, _ = MyColSpec.filter(tbl)
    assert (
        filtered
        >> pdt.group_by(filtered.b)
        >> pdt.summarize()
        >> pdt.alias()
        >> pdt.summarize(n_unique=pdt.count())
        >> pdt.export(pdt.Scalar)
    ) == 1
