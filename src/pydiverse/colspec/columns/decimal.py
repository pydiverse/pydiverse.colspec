# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
import decimal
import math
from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr, dy, sa
from ._base import Column
from ._mixins import OrdinalMixin


class Decimal(OrdinalMixin[decimal.Decimal], Column):
    """A column of decimal values with given precision and scale."""

    def __init__(
        self,
        precision: int | None = None,
        scale: int | None = None,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: decimal.Decimal | float | int | None = None,  # noqa: A002
        min_exclusive: decimal.Decimal | float | int | None = None,
        max: decimal.Decimal | float | int | None = None,  # noqa: A002
        max_exclusive: decimal.Decimal | float | int | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            precision: Maximum number of digits in each number.
            scale: Number of digits to the right of the decimal point in each number.
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum value for decimals in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for decimals in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
        """
        if min is not None:
            _validate(min, precision, scale, "min")
            if isinstance(min, decimal.Decimal):
                min = float(min)  # noqa: A001, Decimal is more a RDBMS thing
        if min_exclusive is not None:
            _validate(min_exclusive, precision, scale, "min_exclusive")
            if isinstance(min_exclusive, decimal.Decimal):
                min_exclusive = float(min_exclusive)  # Decimal is more a RDBMS thing
        if max is not None:
            _validate(max, precision, scale, "max")
            if isinstance(max, decimal.Decimal):
                max = float(max)  # noqa: A001, Decimal is more a RDBMS thing
        if max_exclusive is not None:
            _validate(max_exclusive, precision, scale, "max_exclusive")
            if isinstance(max_exclusive, decimal.Decimal):
                max_exclusive = float(max_exclusive)  # Decimal is more a RDBMS thing

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.precision = precision
        self.scale = scale

    def dtype(self) -> pdc.Decimal:
        return pdc.Decimal()

    def to_dataframely(self) -> dy.Column:
        if self.scale is None or any(
            isinstance(x, float | int)
            for x in [self.min, self.max, self.min_exclusive, self.max_exclusive]
        ):
            ret = copy.copy(self)
            # in colspec we don't use python decimals for defining boundaries
            ret.min = decimal.Decimal(ret.min) if ret.min is not None else None
            ret.max = decimal.Decimal(ret.max) if ret.max is not None else None
            ret.min_exclusive = (
                decimal.Decimal(ret.min_exclusive)
                if ret.min_exclusive is not None
                else None
            )
            ret.max_exclusive = (
                decimal.Decimal(ret.max_exclusive)
                if ret.max_exclusive is not None
                else None
            )
            # polars cannot deal with scale=None as opposed to SQL
            if ret.precision:
                ret.scale = ret.scale or (ret.precision // 3 + 1)
            else:
                ret.scale = ret.scale or 11
            return ret.to_dataframely()
        else:
            return super().to_dataframely()

    def sqlalchemy_column(self, name: str, dialect: sa.Dialect) -> sa.Column:
        """Obtain the SQL column specification of this column definition.

        Args:
            name: The name of the column.
            dialect: The SQL dialect for which to generate the column specification.

        Returns:
            The column as specified in :mod:`sqlalchemy`.
        """
        _ = dialect  # may be used in the future
        return sa.Column(
            name,
            sa.Numeric(
                self.precision or (38 if self.scale is not None else None), self.scale
            ),
            nullable=self.nullable,
            primary_key=self.primary_key,
            autoincrement=False,
        )


# --------------------------------------- UTILS -------------------------------------- #


def _validate(
    value: decimal.Decimal | int | float,
    precision: int | None,
    scale: int | None,
    name: str,
):
    if not isinstance(value, decimal.Decimal):
        value = decimal.Decimal(value)
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"Encountered 'inf' or 'NaN' for `{name}`.")
    if exponent is not None and scale is not None and -exponent > scale:
        raise ValueError(f"Scale of `{name}` exceeds scale of column.")
    if (
        precision is not None
        and scale is not None
        and _num_digits(int(value)) > precision - scale
    ):
        raise ValueError(f"`{name}` exceeds precision of column.")


def _num_digits(i: int) -> int:
    return int(math.log10(i) + 1)
