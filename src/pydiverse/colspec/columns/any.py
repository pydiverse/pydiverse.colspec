# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr, dy
from ._base import Column


class Any(Column):
    """A column that can contain any type."""

    def __init__(
        self,
        *,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
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
            nullable=True,
            primary_key=False,
            check=check,
            alias=alias,
            metadata=metadata,
        )

    def dtype(self) -> pdc.Dtype:
        raise NotImplementedError(
            "The Type Any is intentionally not implemented in pydiverse libraries."
        )

    def to_dataframely(self):
        return dy.Any(check=self.check, alias=self.alias)
