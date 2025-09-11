# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr, sa
from ._base import Column

# ------------------------------------------------------------------------------------ #


class String(Column):
    """A column of strings."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min_length: int | None = None,
        max_length: int | None = None,
        regex: str | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
            min_length: The minimum byte-length of string values in this column.
            max_length: The maximum byte-length of string values in this column.
            regex: A regex that the string values in this column must match. If the
                regex does not use start and end anchors (i.e. ``^`` and ``$``), the
                regex must only be _contained_ in the string.
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
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex

    def dtype(self) -> pdc.String:
        return pdc.String(self.max_length)

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)

        from pydiverse.colspec.optional_dependency import pl

        len_fn = "len_bytes" if isinstance(expr, pl.Expr) else "len"

        if self.min_length is not None:
            result["min_length"] = getattr(expr.str, len_fn)() >= self.min_length
        if self.max_length is not None:
            result["max_length"] = getattr(expr.str, len_fn)() <= self.max_length
        if self.regex is not None:
            result["regex"] = expr.str.contains(self.regex)
        return result

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
            sa.VARCHAR
            if self.max_length is None
            else sa.CHAR(self.min_length)
            if self.min_length == self.max_length
            else sa.VARCHAR(self.max_length),
            nullable=self.nullable,
            primary_key=self.primary_key,
            autoincrement=False,
        )
