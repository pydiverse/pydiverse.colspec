# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
import sys
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr
from ._base import Column
from ._mixins import OrdinalMixin
from ._utils import classproperty


class _BaseFloat(OrdinalMixin[float], Column):
    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: float | None = None,  # noqa: A002
        min_exclusive: float | None = None,
        max: float | None = None,  # noqa: A002
        max_exclusive: float | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum value for floats in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for floats in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                happen with metadata. It is just stored.
        """
        if min is not None and min < self.min_value:
            raise ValueError("Minimum value is too small for the data type.")
        if max is not None and max > self.max_value:
            raise ValueError("Maximum value is too big for the data type.")

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

    @classproperty
    @abstractmethod
    def max_value(self) -> float:
        """Maximum value of the column's type."""

    @classproperty
    @abstractmethod
    def min_value(self) -> float:
        """Minimum value of the column's type."""


# ------------------------------------------------------------------------------------ #


class Float(_BaseFloat):
    """A column of floating-point numbers."""

    def dtype(self) -> pdc.Float:
        return pdc.Float()

    @classproperty
    def max_value(self) -> float:
        return sys.float_info.max

    @classproperty
    def min_value(self) -> float:
        return -sys.float_info.max


class Float32(_BaseFloat):
    """A column of 32-bit floating-point numbers."""

    def dtype(self) -> pdc.Float32:
        return pdc.Float32()

    @classproperty
    def max_value(self) -> float:
        return 3.4028234663852886e38  # float(np.finfo(np.float32).max)

    @classproperty
    def min_value(self) -> float:
        return -3.4028234663852886e38  # float(np.finfo(np.float32).min)


class Float64(_BaseFloat):
    """A column of 64-bit floating-point numbers."""

    def dtype(self) -> pdc.Float64:
        return pdc.Float64()

    @classproperty
    def max_value(self) -> float:
        return sys.float_info.max

    @classproperty
    def min_value(self) -> float:
        return -sys.float_info.max
