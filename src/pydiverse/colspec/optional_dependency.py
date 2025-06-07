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
    from polars.datatypes.group import FLOAT_DTYPES, INTEGER_DTYPES
    from polars.testing import assert_frame_equal
except ImportError:

    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass

    DataTypeClass = None
    PolarsDataType = None
    FLOAT_DTYPES, INTEGER_DTYPES = None, None
    assert_frame_equal = None
    # Create a new module with the given name.
    pl = types.ModuleType("polars")
    pl.DataFrame = DummyClass
    pl.LazyFrame = DummyClass
    pl.DataType = None
    pl.Series = None
    pl.Schema = None
    pl.Expr = None
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
    ]:
        setattr(pl, _type, DummyClass)


try:
    # colspec has optional dependency to dataframely
    import dataframely as dy
    from dataframely._polars import FrameType
    from dataframely.random import Generator
except ImportError:

    class Generator:
        pass

    T = TypeVar("T")

    class DyDataFrame(Generic[T]):
        pass

    FrameType = None
    dy = types.ModuleType("dataframely")
    dy.DataFrame = DyDataFrame
    dy.LazyFrame = DyDataFrame
    dy.FailureInfo = None
    dy.Column = None
    dy.Collection = None


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
