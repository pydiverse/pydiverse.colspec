# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import dataframely as dy


class ParentSchema(cs.ColSpec):
    a = dy.Integer()


class ChildSchema(ParentSchema):
    b = dy.Integer()


class GrandchildSchema(ChildSchema):
    c = dy.Integer()


def test_columns():
    assert ParentSchema.column_names() == ["a"]
    assert ChildSchema.column_names() == ["a", "b"]
    assert GrandchildSchema.column_names() == ["a", "b", "c"]
