# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pydiverse.common as pdc

from ._base import Column

if TYPE_CHECKING:
    from src.pydiverse.colspec.columns import Expr

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
        check: Callable[[Expr], Expr] | None = None,
        alias: str | None = None,
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
        """
        super().__init__(
            nullable=nullable, primary_key=primary_key, check=check, alias=alias
        )
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex

    def dtype(self) -> pdc.String:
        return pdc.String()

    def validation_rules(self, expr: Expr) -> dict[str, Expr]:
        result = super().validation_rules(expr)
        if self.min_length is not None:
            result["min_length"] = expr.str.len_bytes() >= self.min_length
        if self.max_length is not None:
            result["max_length"] = expr.str.len_bytes() <= self.max_length
        if self.regex is not None:
            result["regex"] = expr.str.contains(self.regex)
        return result
