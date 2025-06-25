# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.colspec import convert_to_dy_col_spec
from pydiverse.colspec.exc import ImplementationError
from pydiverse.colspec.optional_dependency import dy


class MyFirstColSpec(cs.ColSpec):
    a = cs.UInt8(primary_key=True)


class MySecondColSpec(cs.ColSpec):
    a = cs.UInt16(primary_key=True)
    b = cs.Integer


class MyThirdColSpec(MyFirstColSpec, MySecondColSpec):
    pass


class MyFourthColSpec(MyFirstColSpec, MySecondColSpec):
    c = cs.Float64()


def test_columns():
    assert sorted(MyFirstColSpec.column_names()) == ["a"]
    assert sorted(MySecondColSpec.column_names()) == ["a", "b"]
    assert sorted(MyThirdColSpec.column_names()) == ["a", "b"]
    assert sorted(MyFourthColSpec.column_names()) == ["a", "b", "c"]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_columns_dataframely():
    first = convert_to_dy_col_spec(MyFirstColSpec)
    second = convert_to_dy_col_spec(MySecondColSpec)
    third = convert_to_dy_col_spec(MyThirdColSpec)
    fourth = convert_to_dy_col_spec(MyFourthColSpec)
    assert sorted(first.column_names()) == ["a"]
    assert sorted(second.column_names()) == ["a", "b"]
    assert sorted(third.column_names()) == ["a", "b"]
    assert sorted(fourth.column_names()) == ["a", "b", "c"]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_dataframely_columns_fail():
    class FailColSpec(cs.ColSpec):
        a = cs.Float64()
        b = dy.String()

    with pytest.raises(
        ImplementationError, match="Dataframely Columns won't work in ColSpec classes."
    ):
        FailColSpec.column_names()
