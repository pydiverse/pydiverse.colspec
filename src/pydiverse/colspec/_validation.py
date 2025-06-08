# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Literal

from pydiverse.colspec.exc import (
    ColumnValidationError,
    DtypeValidationError,
)
from pydiverse.common import Dtype

from .columns import Column
from .optional_dependency import pdt

DtypeCasting = Literal["none", "lenient", "strict"]


def validate_columns(
    tbl: pdt.Table,
    expected: Iterable[str],
) -> pdt.Table:
    """Validate the existence of expected columns in a table.

    Args:
        tbl: The table whose list of columns to validate.
        expected: The list of columns that _should_ be observed.

    Raises:
        ValidationError: If any expected column is not part of the actual columns.

    Returns:
        The input table, either as-is or with extra columns stripped.
    """
    actual_set = set(col.name for col in tbl)
    expected_set = set(expected)

    missing_columns = expected_set - actual_set
    if len(missing_columns) > 0:
        raise ColumnValidationError(
            list(sorted(missing_columns)),
            extra=list(sorted(actual_set - expected_set)),
            actual=list(sorted(actual_set)),
        )

    return tbl >> pdt.select(*expected)


def validate_dtypes(
    tbl: pdt.Table,
    expected: dict[str, Column],
    *,
    casting: Literal["none", "lenient", "strict"],
) -> pdt.Table:
    """Validate the dtypes of all expected columns in a table.

    Args:
        tbl: The table whose column dtypes to validate.
        expected: The column definitions carrying the expected dtypes.
        casting: The strategy for casting dtypes.

    Raises:
        DtypeValidationError: If the expected column dtypes do not match the input's and
            ``casting`` set to ``none``.

    Returns:
        The input table with all column dtypes ensured to have the expected dtype.
    """
    from pydiverse.transform import C, mutate

    dtype_errors: dict[str, tuple[Dtype, Dtype]] = {}
    for name, col in expected.items():
        if not tbl[name].dtype().is_subtype(col.dtype()):
            dtype_errors[name] = (tbl[name].dtype(), col.dtype())

    if len(dtype_errors) > 0:
        if casting == "none":
            raise DtypeValidationError(dtype_errors)
        else:
            return tbl >> mutate(
                **{
                    name: C[name].cast(
                        expected[name].dtype(), strict=(casting == "strict")
                    )
                    for name in dtype_errors.keys()
                }
            )

    return tbl
