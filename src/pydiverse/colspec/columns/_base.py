# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime as dt
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydiverse.colspec.columns import ColExpr
from pydiverse.common import Dtype

if TYPE_CHECKING:
    from ..colspec import Generator, PolarsDataType, pl

EPOCH_DATETIME = dt.datetime(1970, 1, 1)
SECONDS_PER_DAY = 86400

# ------------------------------------------------------------------------------------ #
#                                        COLUMNS                                       #
# ------------------------------------------------------------------------------------ #


class ColumnMeta(ABCMeta):
    def __new__(cls, clsname, bases, attribs):
        # change bases (only ABC is a real base)
        bases = tuple([base for base in bases if not issubclass(base, ColExpr)])
        return super().__new__(cls, clsname, bases, attribs)


class Column(ABC, ColExpr, metaclass=ColumnMeta):
    """Abstract base class for data frame column definitions.

    This class is merely supposed to be used in :class:`~colspec.Schema`
    definitions.
    """

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name. If unset, colspec
                internally sets the alias to the column's name in the parent schema.
        """
        self.nullable = nullable and not primary_key
        self.primary_key = primary_key
        self.check = check
        self.alias = alias

    # ------------------------------------- DTYPE ------------------------------------ #

    @abstractmethod
    def dtype(self) -> Dtype:
        """The common dtype of this column definition.

        Returns a pydiverse.common.Dtype instance that represents the abstract data type
        of this column. This type can then be mapped to specific backend types (SQL,
        Polars, etc.) through the respective conversion methods.

        Returns:
            A pydiverse.common.Dtype instance representing the column's data type.
        """

    # ---------------------------------- VALIDATION ---------------------------------- #

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        """A set of rules evaluating whether a data frame column satisfies the column's
        constraints.

        Args:
            expr: An expression referencing the column of the data frame, i.e. an
                expression created by calling :meth:`polars.col`.

        Returns:
            A mapping from validation rule names to expressions that provide exactly
            one boolean value per column item indicating whether validation with respect
            to the rule is successful. A value of ``False`` indicates invalid data, i.e.
            unsuccessful validation.
        """
        result = {}
        if not self.nullable:
            result["nullability"] = expr.is_not_null()
        if self.check is not None:
            result["check"] = self.check(expr)
        return result

    # -------------------------------- POLARS VALIDATION ----------------------------- #

    def to_dataframely(self):
        """Convert this column to its dataframely equivalent.

        Returns:
            A dataframely.Column instance with the same properties as this column.
        """
        import dataframely as dy

        def convert(value):
            if isinstance(value, Column):
                return value.to_dataframely()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, tuple):
                return tuple(convert(v) for v in value)
            return value

        # Get all non-private attributes
        attrs = {
            k: convert(v) for k, v in self.__dict__.items() if not k.startswith("_")
        }
        return getattr(dy, self.__class__.__name__)(**attrs)

    def validate_polars_dtype(self, dtype: PolarsDataType) -> bool:
        """Validate if the :mod:`polars` data type satisfies the column definition.

        This function requires dataframely to be installed since this is used as colspec
        implementation for polars.
        Args:
            dtype: The dtype to validate.

        Returns:
            Whether the dtype is valid.
        """
        return self.to_dataframely().validate_dtype(dtype)

    # -------------------------------- POLARS SAMPLING ------------------------------- #

    def sample_polars(self, generator: Generator, n: int = 1) -> pl.Series:
        """Sample random elements adhering to the constraints of this column.

        This function requires dataframely to be installed since this is used as colspec
        implementation for polars.
        Args:
            generator: The generator to use for sampling elements.
            n: The number of elements to sample.

        Returns:
            A series with the predefined number of elements. All elements are guaranteed
            to adhere to the column's constraints.

        Raises:
            ValueError: If this column has a custom check. In this case, random values
                cannot be guaranteed to adhere to the column's constraints while
                providing any guarantees on the computational complexity.
        """
        return self.to_dataframely().sample(generator, n)

    # -------------------------------- DUNDER METHODS -------------------------------- #

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
