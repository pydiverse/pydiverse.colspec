# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types

try:
    import polars as pl
    from polars.datatypes import DataTypeClass

    PolarsDataType = pl.DataType | DataTypeClass
except ImportError:
    PolarsDataType = None
    # Create a new module with the given name.
    pl = types.ModuleType("polars")
    pl.DataFrame = None
    pl.LazyFrame = None
    pl.DataType = None
    pl.Series = None
    pl.Expr = None


try:
    # colspec has optional dependency to dataframely
    import dataframely as dy
    from dataframely._polars import FrameType
    from dataframely.random import Generator
except ImportError:
    Generator = None
    FrameType = None
    dy = types.ModuleType("dataframely")
    dy.DataFrame = None
    dy.LazyFrame = None
    dy.FailureInfo = None


try:
    # colspec has optional dependency to pydiverse.transform
    import pydiverse.transform as pdt
    from pydiverse.transform import verb
except ImportError:

    def verb(func):
        """A no-op decorator for functions that are intended to be used as verbs."""
        return func

    class Table:
        pass

    # Create a new module with the given name.
    pdt = types.ModuleType("pydiverse.transform")
    pdt.Table = Table
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
