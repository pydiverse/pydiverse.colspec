# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pydiverse.colspec as cs

COLUMN_TYPES: list[type[cs.Column]] = [
    cs.Bool,
    cs.Date,
    cs.Datetime,
    cs.Time,
    cs.Decimal,
    cs.Duration,
    cs.Float32,
    cs.Float64,
    cs.Int8,
    cs.Int16,
    cs.Int32,
    cs.Int64,
    cs.UInt8,
    cs.UInt16,
    cs.UInt32,
    cs.UInt64,
    cs.String,
]
INTEGER_COLUMN_TYPES: list[type[cs.Column]] = [
    cs.Integer,
    cs.Int8,
    cs.Int16,
    cs.Int32,
    cs.Int64,
    cs.UInt8,
    cs.UInt16,
    cs.UInt32,
    cs.UInt64,
]
FLOAT_COLUMN_TYPES: list[type[cs.Column]] = [
    cs.Float,
    cs.Float32,
    cs.Float64,
]

SUPERTYPE_COLUMN_TYPES: list[type[cs.Column]] = [
    cs.Float,
    cs.Integer,
]

ALL_COLUMN_TYPES: list[type[cs.Column]] = (
    [cs.Any] + COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES
)
