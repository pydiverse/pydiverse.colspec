# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.columns._base import Column
from pydiverse.colspec.testing.const import ALL_COLUMN_TYPES


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_string_representation(column_type: type[Column]):
    column = column_type()
    assert str(column) == column_type.__name__.lower()


def test_string_representation_enum():
    column = cs.Enum(["a", "b"])
    assert str(column) == cs.Enum.__name__.lower()


def test_string_representation_list():
    column = cs.List(cs.String())
    assert str(column) == cs.List.__name__.lower()


def test_string_representation_struct():
    column = cs.Struct({"a": cs.String()})
    assert str(column) == cs.Struct.__name__.lower()
