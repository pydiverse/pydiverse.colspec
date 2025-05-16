# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pydiverse.colspec as cs


class ParentSchema(cs.ColSpec):
    a = cs.Integer()


class ChildSchema(ParentSchema):
    b = cs.Integer()


class GrandchildSchema(ChildSchema):
    c = cs.Integer()


def test_columns():
    assert ParentSchema.column_names() == ["a"]
    assert ChildSchema.column_names() == ["a", "b"]
    assert GrandchildSchema.column_names() == ["a", "b", "c"]
