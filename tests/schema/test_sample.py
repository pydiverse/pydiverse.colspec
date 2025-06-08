# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import (
    C,
    Generator,
    assert_frame_equal,
    dy,
    np,
    pdt,
    pl,
)


class MySimpleColSpec(cs.ColSpec):
    a = cs.Int64()
    b = cs.String()


class PrimaryKeyColSpec(cs.ColSpec):
    a = cs.Int64(primary_key=True)
    b = cs.String()


class CheckColSpec(cs.ColSpec):
    a = cs.UInt64()
    b = cs.UInt64()

    @cs.rule_polars()
    @staticmethod
    def a_ge_b() -> pl.Expr:
        return pl.col("a") >= pl.col("b")


class ComplexColSpec(cs.ColSpec):
    a = cs.UInt8(primary_key=True)
    b = cs.UInt8(primary_key=True)

    @cs.rule_polars()
    @staticmethod
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @cs.rule_polars(group_by=["a"])
    @staticmethod
    def minimum_two_per_a() -> pl.Expr:
        return pl.len() >= 2


class LimitedComplexColSpec(cs.ColSpec):
    a = cs.UInt8(primary_key=True)
    b = cs.UInt8(primary_key=True)

    @cs.rule_polars()
    @staticmethod
    def a_greater_b() -> pl.Expr:
        return pl.col("a") > pl.col("b")

    @cs.rule_polars(group_by=["a"])
    @staticmethod
    def minimum_two_per_a() -> pl.Expr:
        # We cannot generate more than 768 rows with this rule
        return pl.len() <= 3


# --------------------------------------- TESTS -------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("n", [0, 1000])
def test_sample_deterministic(n: int):
    with dy.Config(max_sampling_iterations=1):
        df = MySimpleColSpec.sample_polars(n)
        MySimpleColSpec.validate_polars(df)
        tbl = pdt.Table(df)
        MySimpleColSpec.validate(tbl)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("col_spec", [PrimaryKeyColSpec, CheckColSpec, ComplexColSpec])
@pytest.mark.parametrize("n", [0, 1000])
def test_sample_fuzzy(col_spec, n: int):
    df = col_spec.sample_polars(n, generator=Generator(seed=42))
    assert len(df) == n
    col_spec.validate_polars(df)
    tbl = pdt.Table(df)
    col_spec.validate(tbl)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_fuzzy_failure():
    with pytest.raises(ValueError):
        with dy.Config(max_sampling_iterations=5):
            ComplexColSpec.sample_polars(1000, generator=Generator(seed=42))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides(n: int):
    df = CheckColSpec.sample_polars(n, overrides={"b": range(n)})
    CheckColSpec.validate_polars(df)
    tbl = pdt.Table(df)
    CheckColSpec.validate(tbl)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(range(n))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_sample_overrides_with_removing_groups():
    generator = Generator()
    n = 333  # we cannot use something too large here or we'll never return
    overrides = np.random.randint(100, size=n)
    df = LimitedComplexColSpec.sample_polars(n, generator, overrides={"b": overrides})
    LimitedComplexColSpec.validate_polars(df)
    tbl = pdt.Table(df)
    LimitedComplexColSpec.validate(tbl)
    assert len(df) == n
    assert df.get_column("b").to_list() == list(overrides)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_allow_no_fuzzy(n: int):
    with dy.Config(max_sampling_iterations=1):
        df = CheckColSpec.sample_polars(n, overrides={"b": [0] * n})
        CheckColSpec.validate_polars(df)
        tbl = pdt.Table(df)
        CheckColSpec.validate(tbl)
        assert len(df) == n
        assert df.get_column("b").to_list() == [0] * n


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("n", [1, 1000])
def test_sample_overrides_full(n: int):
    df = CheckColSpec.sample_polars(n)
    df_override = CheckColSpec.sample_polars(n, overrides=df.to_dict())
    assert_frame_equal(df, df_override)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_overrides_invalid_column():
    with pytest.raises(ValueError, match=r"not in the schema"):
        MySimpleColSpec.sample_polars(overrides={"foo": []})


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_sample_overrides_invalid_length():
    with pytest.raises(ValueError, match=r"`num_rows` is different"):
        MySimpleColSpec.sample_polars(overrides={"a": [1, 2]})
