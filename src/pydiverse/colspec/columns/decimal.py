# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import decimal
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import pydiverse.common as pdc

from ._base import Column
from ._mixins import OrdinalMixin

if TYPE_CHECKING:
    from pydiverse.colspec.columns import ColExpr


class Decimal(OrdinalMixin[decimal.Decimal], Column):
    """A column of decimal values with given precision and scale."""

    def __init__(
        self,
        precision: int | None = None,
        scale: int = 0,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: decimal.Decimal | None = None,
        min_exclusive: decimal.Decimal | None = None,
        max: decimal.Decimal | None = None,
        max_exclusive: decimal.Decimal | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
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
        if min_exclusive is not None:
            _validate(min_exclusive, precision, scale, "min_exclusive")
        if max is not None:
            _validate(max, precision, scale, "max")
        if max_exclusive is not None:
            _validate(max_exclusive, precision, scale, "max_exclusive")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
        )
        self.precision = precision
        self.scale = scale

    def dtype(self) -> pdc.Decimal:
        return pdc.Decimal()


# --------------------------------------- UTILS -------------------------------------- #


def _validate(value: decimal.Decimal, precision: int | None, scale: int, name: str):
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"Encountered 'inf' or 'NaN' for `{name}`.")
    if -exponent > scale:
        raise ValueError(f"Scale of `{name}` exceeds scale of column.")
    if precision is not None and _num_digits(int(value)) > precision - scale:
        raise ValueError(f"`{name}` exceeds precision of column.")


def _num_digits(i: int) -> int:
    return int(math.log10(i) + 1)
