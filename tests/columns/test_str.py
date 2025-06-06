# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dataframely as dy
from dataframely.columns import Column
from dataframely.testing import ALL_COLUMN_TYPES


@pytest.mark.parametrize("column_type", ALL_COLUMN_TYPES)
def test_string_representation(column_type: type[Column]):
    column = column_type()
    assert str(column) == column_type.__name__.lower()


def test_string_representation_enum():
    column = dy.Enum(["a", "b"])
    assert str(column) == dy.Enum.__name__.lower()


def test_string_representation_list():
    column = dy.List(dy.String())
    assert str(column) == dy.List.__name__.lower()


def test_string_representation_struct():
    column = dy.Struct({"a": dy.String()})
    assert str(column) == dy.Struct.__name__.lower()
