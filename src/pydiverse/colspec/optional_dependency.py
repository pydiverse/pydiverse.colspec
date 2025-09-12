# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import types
from typing import Generic, TypeVar

try:
    import numpy as np
except ImportError:
    np = None


try:
    import polars as pl
    from polars.datatypes import DataTypeClass

    PolarsDataType = pl.DataType | DataTypeClass
    import polars.exceptions as plexc
    from polars.datatypes.group import FLOAT_DTYPES, INTEGER_DTYPES
    from polars.testing import assert_frame_equal
except ImportError:

    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass

    DataTypeClass = None
    PolarsDataType = None
    FLOAT_DTYPES, INTEGER_DTYPES = [], []
    assert_frame_equal = None
    plexc = None
    # Create a new module with the given name.
    pl = types.ModuleType("polars")
    pl.DataFrame = DummyClass
    pl.LazyFrame = DummyClass
    pl.DataType = None
    pl.Series = None
    pl.Schema = None
    pl.Expr = object
    for _type in [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Boolean",
        "Utf8",
        "Decimal",
        "Enum",
        "Struct",
        "List",
        "Date",
        "Datetime",
        "Time",
        "Duration",
        "String",
        "Null",
    ]:
        setattr(pl, _type, DummyClass)


try:
    # colspec has optional dependency to dataframely
    import dataframely as dy
    from dataframely._base_schema import SchemaMeta
    from dataframely._polars import FrameType
    from dataframely.random import Generator
    from dataframely.testing import validation_mask
except ImportError:

    class Generator:
        pass

    T = TypeVar("T")

    class DyDataFrame(Generic[T]):
        pass

    class DyDummyClass:
        def __init__(self, *args, **kwargs):
            pass

    FrameType = None
    SchemaMeta = None
    validation_mask = None
    dy = types.ModuleType("dataframely")
    dy.DataFrame = DyDataFrame
    dy.LazyFrame = DyDataFrame
    dy.FailureInfo = None
    dy.Column = None
    dy.Collection = object
    dy.Schema = DyDummyClass
    dy.filter = lambda: lambda fn: fn  # noqa
    dy.rule = lambda: lambda fn: fn  # noqa
    for _type in [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Bool",
        "String",
        "Decimal",
        "Enum",
        "Struct",
        "List",
        "Date",
        "Datetime",
        "Time",
        "Duration",
        "Float",
        "Integer",
    ]:
        setattr(dy, _type, DyDummyClass)


try:
    # colspec has optional dependency to pydiverse.transform
    import pydiverse.transform as pdt
    from pydiverse.transform import C, ColExpr, verb
except ImportError:

    def verb(func):
        """A no-op decorator for functions that are intended to be used as verbs."""
        return func

    class Table:
        pass

    class ColExpr:
        pass

    # Create a new module with the given name.
    pdt = types.ModuleType("pydiverse.transform")
    pdt.Table = Table
    C = None
    # TODO: add members that break if pdt is not there


try:
    # colspec has optional dependency to pydiverse.pipedag
    import pydiverse.pipedag as dag
except ImportError:

    class Table:
        pass

    # Create a new module with the given name.
    dag = types.ModuleType("pydiverse.pipedag")
    dag.Table = Table


try:
    # colspec has optional dependency to pyarrow
    import pyarrow as pa
except ImportError:
    pa = types.ModuleType("pyarrow")
    pa.Schema = None
    pa.Field = None
    pa.DataType = None


try:
    # colspec has optional dependency to sqlalchemy
    import sqlalchemy as sa
except ImportError:
    sa = types.ModuleType("sqlalchemy")
    sa.Dialect = None
    sa.Column = None
    sa.Table = None
    sa.Alias = None
