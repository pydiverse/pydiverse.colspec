# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr, pl
from ._base import Column


class Struct(Column):
    """A struct column."""

    def __init__(
        self,
        inner: dict[str, Column],
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            inner: The dictionary of struct fields. Struct fields may have
                ``primary_key=True`` set but this setting only takes effect if the
                struct is nested inside a list. In this case, the list items must be
                unique wrt. the struct fields that have ``primary_key=True`` set.
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
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
        self.inner = inner

    def dtype(self) -> pdc.Dtype:
        raise NotImplementedError(
            "Struct column type is not yet implemented in pydiverse libraries."
        )

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        raise NotImplementedError(
            "Struct column type is not yet implemented in pydiverse libraries."
        )
        inner_rules = {
            f"inner_{name}_{rule_name}": (
                pl.when(expr.is_null()).then(pl.lit(True)).otherwise(inner_expr)
            )
            for name, col in self.inner.items()
            for rule_name, inner_expr in col.validation_rules(
                expr.struct.field(name)
            ).items()
        }
        return {
            **super().validation_rules(expr),
            **inner_rules,
        }
