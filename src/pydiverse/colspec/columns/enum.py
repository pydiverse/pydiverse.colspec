# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable, Sequence

import pydiverse.common as pdc

from ..optional_dependency import ColExpr
from ._base import Column


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
        """
        super().__init__(
            nullable=nullable, primary_key=primary_key, check=check, alias=alias
        )
        self.categories = categories

    def dtype(self) -> pdc.Dtype:
        raise NotImplementedError(
            "Enum column type is not yet implemented in pydiverse libraries."
        )
