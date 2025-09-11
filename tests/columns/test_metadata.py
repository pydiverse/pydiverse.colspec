# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import Column
from pydiverse.colspec.testing import COLUMN_TYPES, SUPERTYPE_COLUMN_TYPES


class SchemaWithMetadata(cs.ColSpec):
    a = cs.Int64(metadata={"masked": True, "comment": "foo", "order": 1})
    b = cs.String()


def test_metadata() -> None:
    assert SchemaWithMetadata.a.metadata == {
        "masked": True,
        "comment": "foo",
        "order": 1,
    }
    assert SchemaWithMetadata.b.metadata is None


@pytest.mark.parametrize("column_type", COLUMN_TYPES + SUPERTYPE_COLUMN_TYPES)
def test_constructor_metadata(column_type: type[Column]):
    column = column_type(metadata={"masked": True, "comment": "foo", "order": 1})
    assert column.metadata == {
        "masked": True,
        "comment": "foo",
        "order": 1,
    }
    column2 = column_type()
    assert column2.metadata is None
