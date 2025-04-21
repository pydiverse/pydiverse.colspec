# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from .. import (
    Any,
    Date,
    Datetime,
    Decimal,
    Enum,
    Float32,
    Int64,
    List,
    ColSpec,
    Struct,
)


class MyImportedBaseColSpec(ColSpec):
    a = Int64()


class MyImportedColSpec(MyImportedBaseColSpec):
    b = Float32()
    c = Enum(["a", "b", "c"])
    d = Struct({"a": Int64(), "b": Struct({"c": Enum(["a", "b"])})})
    e = List(Struct({"a": Int64()}))
    f = Datetime()
    g = Date()
    h = Any()
    some_decimal = Decimal(12, 8)
