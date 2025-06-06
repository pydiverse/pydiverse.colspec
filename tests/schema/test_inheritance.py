# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

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
