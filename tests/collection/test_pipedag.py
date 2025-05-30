# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import tempfile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pydiverse.colspec as cs
from pydiverse.pipedag import Flow, Stage, materialize
from pydiverse.pipedag.backend import SQLTableStore
from pydiverse.pipedag.backend.table.sql.dialects.duckdb import (
    DuckDBTableStore,
    PolarsTableHook,
)
from pydiverse.pipedag.core.config import create_basic_pipedag_config

# ------------------------------------------------------------------------------------ #
#                                        SCHEMA                                        #
# ------------------------------------------------------------------------------------ #


class MyFirstColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer()


class MySecondColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer(min=1)


class MyCollection(cs.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec

    @cs.filter_polars()
    def equal_primary_keys(self) -> pl.LazyFrame:
        return self.first.join(self.second, on=self.common_primary_keys())

    @cs.filter_polars()
    def first_b_greater_second_b(self) -> pl.LazyFrame:
        return self.first.join(
            self.second, on=self.common_primary_keys(), how="full", coalesce=True
        ).filter((pl.col("b") > pl.col("b_right")).fill_null(True))


@dataclasses.dataclass
class SimpleCollection(cs.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


def data_without_filter_without_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    return first, second


def data_without_filter_with_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 1], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    return first, second


def data_with_filter_without_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 1, 3]})
    second = pl.LazyFrame({"a": [2, 3, 4, 5], "b": [1, 2, 3, 4]})
    return first, second


def data_with_filter_with_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [2, 3, 4, 5], "b": [0, 1, 2, 3]})
    return first, second


@pytest.fixture()
def dag_cfg():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield create_basic_pipedag_config(
            f"duckdb:///{temp_dir}/db.duckdb",
            disable_stage_locking=True,
            auto_table=[
                "pandas.DataFrame",
                "polars.DataFrame",
                "polars.LazyFrame",
                "sqlalchemy.sql.expression.TextClause",
                "sqlalchemy.sql.expression.Selectable",
                "pydiverse.transform.Table",
            ],
        ).get("default")


@DuckDBTableStore.register_table(pl, replace_hooks=[PolarsTableHook])
class CustomPolarsDownloadTableHook(PolarsTableHook):
    @classmethod
    def download_table(cls, query, connection_uri: str) -> pl.DataFrame:
        df = pl.read_database_uri(query, connection_uri, engine="adbc")
        return df

    @classmethod
    def _execute_query(cls, query: str, connection_uri: str, store: SQLTableStore):
        # Connectorx doesn't support duckdb.
        # Instead, we load it like this:  DuckDB -> PyArrow -> Polars
        engine = store.engine
        conn = engine.raw_connection()
        pl_table = conn.sql(query).arrow()

        import polars

        df = polars.from_arrow(pl_table)
        return df


@materialize(nout=2)
def get_data(name: str):
    return globals()[f"data_{name}"]()


@materialize(nout=2, input_type=pl.LazyFrame)
def exec_filter_polars(c: cs.Collection):
    out, failure = c.filter_polars()
    return out, {name: failure[name]._df for name in failure.keys()}


# -------------------------------------- FILTER -------------------------------------- #


def test_filter_without_filter_without_rule_violation(dag_cfg):
    @materialize(input_type=pl.LazyFrame)
    def assertions(out, failure):
        assert isinstance(out, SimpleCollection)
        assert_frame_equal(out.first, data_without_filter_without_rule_violation()[0])
        assert_frame_equal(out.second, data_without_filter_without_rule_violation()[1])
        assert failure["first"].select(pl.len()).collect().item() == 0
        assert failure["second"].select(pl.len()).collect().item() == 0

    with dag_cfg:
        with Flow() as flow:
            with Stage("s01"):
                c = SimpleCollection(*get_data("without_filter_without_rule_violation"))
            with Stage("s02"):
                out, failure = exec_filter_polars(c)
                assertions(out, failure)

        flow.run(config=dag_cfg)


def test_filter_without_filter_with_rule_violation(
    data_without_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = SimpleCollection.filter_polars_data(
        {
            "first": data_without_filter_with_rule_violation[0],
            "second": data_without_filter_with_rule_violation[1],
        }
    )

    assert isinstance(out, SimpleCollection)
    assert len(out.first.collect()) == 1
    assert len(out.second.collect()) == 2
    assert failure["first"].counts() == {"primary_key": 2}
    assert failure["second"].counts() == {"b|min": 1}


def test_filter_with_filter_without_rule_violation(
    data_with_filter_without_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = MyCollection.filter_polars_data(
        {
            "first": data_with_filter_without_rule_violation[0],
            "second": data_with_filter_without_rule_violation[1],
        }
    )

    assert isinstance(out, MyCollection)
    assert_frame_equal(out.first, pl.LazyFrame({"a": [3], "b": [3]}))
    assert_frame_equal(out.second, pl.LazyFrame({"a": [3], "b": [2]}))
    assert failure["first"].counts() == {
        "equal_primary_keys": 1,
        "first_b_greater_second_b": 1,
    }
    assert failure["second"].counts() == {
        "equal_primary_keys": 2,
        "first_b_greater_second_b": 1,
    }


def test_filter_with_filter_with_rule_violation(
    data_with_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = MyCollection.filter_polars_data(
        {
            "first": data_with_filter_with_rule_violation[0],
            "second": data_with_filter_with_rule_violation[1],
        }
    )

    assert isinstance(out, MyCollection)
    assert_frame_equal(out.first, pl.LazyFrame({"a": [3], "b": [3]}))
    assert_frame_equal(out.second, pl.LazyFrame({"a": [3], "b": [1]}))
    assert failure["first"].counts() == {"equal_primary_keys": 2}
    assert failure["second"].counts() == {"b|min": 1, "equal_primary_keys": 2}
