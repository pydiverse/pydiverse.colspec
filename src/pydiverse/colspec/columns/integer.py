# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr
from ._base import Column
from ._mixins import IsInMixin, OrdinalMixin
from ._utils import classproperty


class _BaseInteger(IsInMixin[int], OrdinalMixin[int], Column):
    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: int | None = None,  # noqa: A002
        min_exclusive: int | None = None,
        max: int | None = None,  # noqa: A002
        max_exclusive: int | None = None,
        is_in: Sequence[int] | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum value for integers in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum value for integers in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            is_in: A (non-contiguous) list of integers indicating valid values in this
                column. If specified, both ``min`` and ``max`` must not bet set.
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
            raise ValueError("`min` is too small for the data type.")
        if max is not None and max > self.max_value:
            raise ValueError("`max` is too big for the data type.")
        if is_in is not None and (min is not None or max is not None):
            raise ValueError(
                "`is_in` may only be specified if `min` and `max` are unspecified."
            )

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            is_in=is_in,
            check=check,
            alias=alias,
            metadata=metadata,
        )

    @classproperty
    @abstractmethod
    def num_bytes(self) -> int:
        """Number of bytes that the column type consumes."""

    @classproperty
    @abstractmethod
    def is_unsigned(self) -> bool:
        """Whether the column type is unsigned."""

    @classproperty
    def max_value(self) -> int:
        """Maximum value of the column's type."""
        return (
            2 ** (self.num_bytes * 8) - 1
            if self.is_unsigned
            else 2 ** (self.num_bytes * 8 - 1) - 1
        )

    @classproperty
    def min_value(self) -> int:
        """Minimum value of the column's type."""
        return 0 if self.is_unsigned else -(2 ** (self.num_bytes * 8 - 1))


# ------------------------------------------------------------------------------------ #


class Integer(_BaseInteger):
    """A column of integers (with any number of bytes)."""

    def dtype(self) -> pdc.Int:
        return pdc.Int()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return False


class Int8(_BaseInteger):
    """A column of int8 values."""

    def dtype(self) -> pdc.Int8:
        return pdc.Int8()

    @classproperty
    def num_bytes(self) -> int:
        return 1

    @classproperty
    def is_unsigned(self) -> bool:
        return False


class Int16(_BaseInteger):
    """A column of int16 values."""

    def dtype(self) -> pdc.Int16:
        return pdc.Int16()

    @classproperty
    def num_bytes(self) -> int:
        return 2

    @classproperty
    def is_unsigned(self) -> bool:
        return False


class Int32(_BaseInteger):
    """A column of int32 values."""

    def dtype(self) -> pdc.Int32:
        return pdc.Int32()

    @classproperty
    def num_bytes(self) -> int:
        return 4

    @classproperty
    def is_unsigned(self) -> bool:
        return False


class Int64(_BaseInteger):
    """A column of int64 values."""

    def dtype(self) -> pdc.Int64:
        return pdc.Int64()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return False


class UInt8(_BaseInteger):
    """A column of uint8 values."""

    def dtype(self) -> pdc.UInt8:
        return pdc.UInt8()

    @classproperty
    def num_bytes(self) -> int:
        return 1

    @classproperty
    def is_unsigned(self) -> bool:
        return True


class UInt16(_BaseInteger):
    """A column of uint16 values."""

    def dtype(self) -> pdc.UInt16:
        return pdc.UInt16()

    @classproperty
    def num_bytes(self) -> int:
        return 2

    @classproperty
    def is_unsigned(self) -> bool:
        return True


class UInt32(_BaseInteger):
    """A column of uint32 values."""

    def dtype(self) -> pdc.UInt32:
        return pdc.UInt32()

    @classproperty
    def num_bytes(self) -> int:
        return 4

    @classproperty
    def is_unsigned(self) -> bool:
        return True


class UInt64(_BaseInteger):
    """A column of uint64 values."""

    def dtype(self) -> pdc.UInt64:
        return pdc.UInt64()

    @classproperty
    def num_bytes(self) -> int:
        return 8

    @classproperty
    def is_unsigned(self) -> bool:
        return True
