# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import dy, pl, validation_mask
from pydiverse.colspec.testing import create_colspec


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize("inner", [cs.Int64(), cs.Integer()])
def test_integer_list(inner: dy.Column):
    spec = create_colspec("test", {"a": cs.List(inner)})
    assert spec.is_valid_polars(pl.DataFrame({"a": [[1], [2], [3]]}))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_invalid_inner_type():
    spec = create_colspec("test", {"a": cs.List(cs.Int64())})
    assert not spec.is_valid_polars(pl.DataFrame({"a": [["1"], ["2"], ["3"]]}))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("column", "dtype", "is_valid_polars"),
    [
        (
            cs.List(cs.Int64()),
            pl.List(pl.Int64()),
            True,
        ),
        (
            cs.List(cs.String()),
            pl.List(pl.Int64()),
            False,
        ),
        (
            cs.List(cs.String()),
            cs.List(cs.String()),
            False,
        ),
        (
            cs.List(cs.String()),
            cs.String(),
            False,
        ),
        (
            cs.List(cs.String()),
            pl.String(),
            False,
        ),
    ],
)
def test_validate_dtype(column: dy.Column, dtype: pl.DataType, is_valid_polars: bool):
    assert column.validate_dtype_polars(dtype) == is_valid_polars


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_nested_lists():
    spec = create_colspec("test", {"a": cs.List(cs.List(cs.Int64()))})
    assert spec.is_valid_polars(pl.DataFrame({"a": [[[1]], [[2]], [[3]]]}))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_list_with_pk():
    spec = create_colspec(
        "test",
        {"a": cs.List(cs.String(), primary_key=True)},
    )
    df = pl.DataFrame({"a": [["ab"], ["a", "ab"], [None], ["a", "b"], ["a", "b"]]})
    _, failures = spec.filter_polars(df)
    assert validation_mask(df, failures).to_list() == [True, True, True, False, False]
    assert failures.counts() == {"primary_key": 2}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_list_with_rules():
    spec = create_colspec(
        "test", {"a": cs.List(cs.String(min_length=2, nullable=False))}
    )
    df = pl.DataFrame({"a": [["ab"], ["a"], [None]]})
    _, failures = spec.filter_polars(df)
    assert validation_mask(df, failures).to_list() == [True, False, False]
    assert failures.counts() == {"a|inner_nullability": 1, "a|inner_min_length": 1}


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_nested_list_with_rules():
    spec = create_colspec(
        "test", {"a": cs.List(cs.List(cs.String(min_length=2, nullable=False)))}
    )
    df = pl.DataFrame({"a": [[["ab"]], [["a"]], [[None]]]})
    _, failures = spec.filter_polars(df)
    # NOTE: `validation_mask` currently fails for multiply nested lists
    assert failures.invalid().to_dict(as_series=False) == {"a": [[["a"]], [[None]]]}
    assert failures.counts() == {
        "a|inner_inner_nullability": 1,
        "a|inner_inner_min_length": 1,
    }


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_list_length_rules():
    spec = create_colspec(
        "test",
        {
            "a": cs.List(
                cs.Integer(nullable=False),
                min_length=2,
                max_length=5,
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"a": [[31, 12], [-1], [None], None, [1, 2, 3, 4, 23, 1]]})
    _, failures = spec.filter_polars(df)
    assert validation_mask(df, failures).to_list() == [True, False, False, True, False]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_outer_inner_nullability():
    spec = create_colspec(
        "test",
        {
            "nullable": cs.List(
                inner=cs.Integer(nullable=False),
                nullable=True,
            )
        },
    )
    df = pl.DataFrame({"nullable": [None, None]})
    spec.validate_polars(df, cast=True)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_inner_primary_key():
    spec = create_colspec("test", {"a": cs.List(cs.Integer(primary_key=True))})
    df = pl.DataFrame({"a": [[1, 2, 3], [1, 1, 2], [1, 1], [1, 4]]})
    _, failure = spec.filter_polars(df)
    assert failure.counts() == {"a|primary_key": 2}
    assert validation_mask(df, failure).to_list() == [True, False, False, True]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.mark.parametrize(
    ("inner_primary_key", "second_primary_key", "failure_count", "mask"),
    [
        (False, True, 2, [False, False, True, True, True, True, True]),
        (True, True, 1, [True, False, True, True, True, True, True]),
        (False, False, 4, [False, False, True, False, False, True, True]),
        (True, False, 1, [True, False, True, True, True, True, True]),
    ],
)
def test_inner_primary_key_struct(
    inner_primary_key: bool,
    second_primary_key: bool,
    failure_count: int,
    mask: list[bool],
):
    spec = create_colspec(
        "test",
        {
            "a": cs.List(
                cs.Struct(
                    {
                        "pk1": cs.Integer(primary_key=True),
                        "pk2": cs.Integer(primary_key=second_primary_key),
                        "other": cs.Integer(),
                    },
                    primary_key=inner_primary_key,
                )
            )
        },
    )
    df = pl.DataFrame(
        {
            "a": [
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 1, "other": 2}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 1, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 2, "other": 1}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 1, "pk2": 2, "other": 2}],
                [{"pk1": 1, "pk2": 1, "other": 1}, {"pk1": 2, "pk2": 2, "other": 2}],
                [],
            ]
        }
    )
    _, failure = spec.filter_polars(df)
    assert failure.counts() == {"a|primary_key": failure_count}
    assert validation_mask(df, failure).to_list() == mask
