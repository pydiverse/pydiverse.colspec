# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pydiverse.common as pdc

from ._base import Column

if TYPE_CHECKING:
    from src.pydiverse.colspec.columns import Expr


class Any(Column):
    """A column that can contain any type."""

    def __init__(
        self,
        *,
        check: Callable[[Expr], Expr] | None = None,
        alias: str | None = None,
    ):
        """
        Args:
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
        """
        super().__init__(nullable=True, primary_key=False, check=check, alias=alias)

    def dtype(self) -> pdc.Any:
        return pdc.Any()
