# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr
from ._base import Column
from .struct import Struct


class List(Column):
    """A list column."""

    def __init__(
        self,
        inner: Column,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            inner: The inner column type. If this type has ``primary_key=True`` set, all
                list items are required to be unique. If the inner type is a struct and
                any of the struct fields have ``primary_key=True`` set, these fields
                must be unique across all list items. Note that if the struct itself has
                ``primary_key=True`` set, the fields' settings do not take effect.
            nullable: Whether this column may contain null values.
            primary_key: Whether this column is part of the primary key of the schema.
            check: A custom check to run for this column. Must return a non-aggregated
                boolean expression.
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
        """
        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.inner = inner
        self.min_length = min_length
        self.max_length = max_length

    def dtype(self) -> pdc.List:
        return pdc.List(self.inner.dtype())

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        raise NotImplementedError()

        from pydiverse.colspec.optional_dependency import (
            pl,  # needs to be replaced with pydiverse.transform
        )

        inner_rules = {
            f"inner_{rule_name}": expr.list.eval(inner_expr).list.all()
            for rule_name, inner_expr in self.inner.validation_rules(
                pl.element()
            ).items()
        }

        list_rules: dict[str, ColExpr] = {}
        if self.inner.primary_key:
            list_rules["primary_key"] = ~expr.list.eval(
                pl.element().is_duplicated()
            ).list.any()
        elif isinstance(self.inner, Struct) and any(
            col.primary_key for col in self.inner.inner.values()
        ):
            primary_key_columns = [
                name for name, col in self.inner.inner.items() if col.primary_key
            ]
            # NOTE: We optimize for a single primary key column here as it is much
            #  faster to run duplication checks for non-struct types in polars 1.22.
            if len(primary_key_columns) == 1:
                list_rules["primary_key"] = ~expr.list.eval(
                    pl.element().struct.field(primary_key_columns[0]).is_duplicated()
                ).list.any()
            else:
                list_rules["primary_key"] = ~expr.list.eval(
                    pl.struct(
                        pl.element().struct.field(primary_key_columns)
                    ).is_duplicated()
                ).list.any()

        if self.min_length is not None:
            list_rules["min_length"] = (
                pl.when(expr.is_null())
                .then(pl.lit(None))
                .otherwise(expr.list.len() >= self.min_length)
            )
        if self.max_length is not None:
            list_rules["max_length"] = (
                pl.when(expr.is_null())
                .then(pl.lit(None))
                .otherwise(expr.list.len() <= self.max_length)
            )
        return {
            **super().validation_rules(expr),
            **list_rules,
            **inner_rules,
        }
