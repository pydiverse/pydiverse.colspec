# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import Rule
from pydiverse.colspec.exc import ImplementationError
from pydiverse.colspec.optional_dependency import C, pdt
from pydiverse.colspec.testing.factory import create_colspec


class MyColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.String(primary_key=True)
    c = cs.Float64()
    e = cs.Any()


def test_column_names():
    _ = MyColSpec.e
    with pytest.raises(AttributeError):
        _ = MyColSpec.d
    assert MyColSpec.column_names() == ["a", "b", "c", "e"]


def test_columns():
    columns = MyColSpec.columns()
    assert isinstance(columns["a"], cs.Integer)
    assert isinstance(columns["b"], cs.String)
    assert isinstance(columns["c"], cs.Float64)
    assert isinstance(columns["e"], cs.Any)


def test_nullability():
    columns = MyColSpec.columns()
    assert not columns["a"].nullable
    assert not columns["b"].nullable
    assert columns["c"].nullable
    assert columns["e"].nullable


def test_primary_keys():
    assert MyColSpec.primary_keys() == ["a", "b"]


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_no_rule_named_primary_key():
    tbl = pdt.Table(dict(a=["b", "c"]))
    with pytest.raises(
        ImplementationError,
        match="@cs.rule annotated functions must not be called `_primary_key_`",
    ):
        create_colspec(
            "test",
            {"a": cs.String()},
            {"_primary_key_": Rule(tbl.a.str.len() > 1)},
        ).validate(tbl)
