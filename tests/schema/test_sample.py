# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataframely as dy
import numpy as np
import polars as pl
import pytest
from dataframely.random import Generator
from polars.testing import assert_frame_equal

import pydiverse.colspec as cs


class MySimpleColSpec(cs.ColSpec):
    a = cs.Int64()
    b = cs.String()


class PrimaryKeyColSpec(cs.ColSpec):
    a = cs.Int64(primary_key=True)
    b = cs.String()


class CheckSchema(cs.ColSpec):
    a = cs.UInt64()
    b = cs.UInt64()

    @cs.rule()
    def a_ge_b() -> pl.Expr:
        return pl.col("a") >= pl.col("b")


class ComplexSchema(cs.ColSpec):
    a = cs.UInt8(primary_key=True)
    b = cs.UInt8(primary_key=True)

    @cs.rule()
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @cs.rule(group_by=["a"])
    def minimum_two_per_a() -> pl.Expr:
        return pl.len() >= 2


class LimitedComplexSchema(cs.ColSpec):
    a = cs.UInt8(primary_key=True)
    b = cs.UInt8(primary_key=True)

    @cs.rule()
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @cs.rule(group_by=["a"])
    def minimum_two_per_a() -> pl.Expr:
        # We cannot generate more than 768 rows with this rule
        return pl.len() <= 3


# --------------------------------------- TESTS -------------------------------------- #


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_deterministic(n: int):
    with dy.Config(max_sampling_iterations=1):
        df = MySimpleColSpec.sample(n)
        MySimpleColSpec.validate(df)


@pytest.mark.parametrize("schema", [PrimaryKeyColSpec, CheckSchema, ComplexSchema])
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
        MySimpleColSpec.sample(overrides={"foo": []})


def test_sample_overrides_invalid_length():
    with pytest.raises(ValueError, match=r"`num_rows` is different"):
        MySimpleColSpec.sample(overrides={"a": [1, 2]})
