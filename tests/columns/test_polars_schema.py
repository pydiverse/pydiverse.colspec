# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import pl
from pydiverse.colspec.testing.factory import create_colspec


@pytest.mark.skipif(pl.Expr is None, reason="polars is required for this test")
def test_polars_schema() -> None:
    schema = create_colspec("test", {"a": cs.Int32(nullable=False), "b": cs.Float32()})
    pl_schema = schema.polars_schema()
    assert pl_schema == {"a": pl.Int32, "b": pl.Float32}
