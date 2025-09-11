# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable, Sequence
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr, sa
from ._base import Column
from .string import String


class Enum(Column):
    """A column of enum (string) values."""

    def __init__(
        self,
        categories: Sequence[str],
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            categories: The list of valid categories for the enum.
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                happen with metadata. It is just stored.
        """
        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.categories = categories

    def dtype(self) -> pdc.Dtype:
        return pdc.Enum(*self.categories)

    def sqlalchemy_column(self, name: str, dialect: sa.Dialect) -> sa.Column:
        """Obtain the SQL column specification of this column definition.

        Args:
            name: The name of the column.
            dialect: The SQL dialect for which to generate the column specification.

        Returns:
            The column as specified in :mod:`sqlalchemy`.
        """
        _ = dialect  # may be used in the future
        str_type = String(
            nullable=self.nullable,
            primary_key=self.primary_key,
            min_length=min(len(cat) for cat in self.categories),
            max_length=max(len(cat) for cat in self.categories),
        )
        return str_type.sqlalchemy_column(name, dialect)

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)
        result["categories"] = expr.is_in(*self.categories)
        return result
