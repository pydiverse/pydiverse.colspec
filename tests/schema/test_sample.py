# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dataframely as dy
from dataframely.random import Generator


class MySimpleSchema(cs.ColSpec):
    a = dy.Int64()
    b = dy.String()


class PrimaryKeySchema(cs.ColSpec):
    a = dy.Int64(primary_key=True)
    b = dy.String()


class CheckSchema(cs.ColSpec):
    a = dy.UInt64()
    b = dy.UInt64()

    @dy.rule()
    def a_ge_b() -> pl.Expr:
        return pl.col("a") >= pl.col("b")


class ComplexSchema(cs.ColSpec):
    a = dy.UInt8(primary_key=True)
    b = dy.UInt8(primary_key=True)

    @dy.rule()
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @dy.rule(group_by=["a"])
    def minimum_two_per_a() -> pl.Expr:
        return pl.len() >= 2


class LimitedComplexSchema(cs.ColSpec):
    a = dy.UInt8(primary_key=True)
    b = dy.UInt8(primary_key=True)

    @dy.rule()
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @dy.rule(group_by=["a"])
    def minimum_two_per_a() -> pl.Expr:
        # We cannot generate more than 768 rows with this rule
        return pl.len() <= 3


# --------------------------------------- TESTS -------------------------------------- #


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_deterministic(n: int):
    with dy.Config(max_sampling_iterations=1):
        df = MySimpleSchema.sample(n)
        MySimpleSchema.validate(df)


@pytest.mark.parametrize("schema", [PrimaryKeySchema, CheckSchema, ComplexSchema])
@pytest.mark.parametrize("n", [0, 1000])
def test_sample_fuzzy(schema: type[cs.ColSpec], n: int):
    df = schema.sample(n, generator=Generator(seed=42))
    assert len(df) == n
    schema.validate_polars(df)


def test_sample_fuzzy_failure():
    with pytest.raises(ValueError):
        with dy.Config(max_sampling_iterations=5):
            ComplexSchema.sample(1000, generator=Generator(seed=42))


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides(n: int):
    df = CheckSchema.sample(n, overrides={"b": range(n)})
    CheckSchema.validate(df)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(range(n))


def test_sample_overrides_with_removing_groups():
    generator = Generator()
    n = 333  # we cannot use something too large here or we'll never return
    overrides = np.random.randint(100, size=n)
    df = LimitedComplexSchema.sample(n, generator, overrides={"b": overrides})
    LimitedComplexSchema.validate(df)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(overrides)


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_allow_no_fuzzy(n: int):
    with dy.Config(max_sampling_iterations=1):
        df = CheckSchema.sample(n, overrides={"b": [0] * n})
        CheckSchema.validate(df)
        assert len(df) == n
        assert df.get_column("b").to_list() == [0] * n


@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_full(n: int):
    df = CheckSchema.sample(n)
    df_override = CheckSchema.sample(n, overrides=df.to_dict())
    assert_frame_equal(df, df_override)


def test_sample_overrides_invalid_column():
    with pytest.raises(ValueError, match=r"not in the schema"):
        MySimpleSchema.sample(overrides={"foo": []})


def test_sample_overrides_invalid_length():
    with pytest.raises(ValueError, match=r"`num_rows` is different"):
        MySimpleSchema.sample(overrides={"a": [1, 2]})
