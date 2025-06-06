# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pydiverse.colspec as cs


class ParentColSpec(cs.ColSpec):
    a = cs.Integer()


class ChildColSpec(ParentColSpec):
    b = cs.Integer()


class GrandchildColSpec(ChildColSpec):
    c = cs.Integer()


def test_columns():
    assert ParentColSpec.column_names() == ["a"]
    assert ChildColSpec.column_names() == ["a", "b"]
    assert GrandchildColSpec.column_names() == ["a", "b", "c"]
