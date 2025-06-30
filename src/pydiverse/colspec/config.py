# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import dataclass

from .columns._utils import classproperty
from .optional_dependency import pdt, verb


@dataclass
class Config:
    """
    Configuration for ColSpec operations.

    ColSpec and Collection classes do not handle configuration directly. Instead,
    you must pass a Config instance explicitly to validation or filtering methods.
    Config also enables integration with other libraries in the pydiverse
    ecosystem—for example, by supplying a materialization hook that prevents ColSpec
    from generating subqueries, which can help avoid performance bottlenecks.
    """

    #: If fix_table_names=True, table names are adjusted to match the corresponding
    # ColSpec class name.
    fix_table_names: bool = True
    #: If stop_validation_on_column_error=True, validation stops in case of column
    # type errors before looking at rows.
    stop_validation_on_column_error: bool = True
    #: A function to materialize a pdt.Table expression returning reference to result.
    # It can also be given a second argument with a table name prefix.
    materialize_hook: Callable[[pdt.Table, str | None], pdt.Table] | None = None
    #: SQL dialect
    dialect_name: str = "default"

    @classproperty
    def default(cls) -> "Config":
        return Config()


@verb
def alias_subquery(tbl: pdt.Table, cfg: Config, table_prefix: str | None = None):
    if cfg.materialize_hook is not None:
        return cfg.materialize_hook(tbl, table_prefix)
    return pdt.alias(table_prefix)


@verb
def alias_collection_fail(tbl: pdt.Table, cfg: Config, table_prefix: str | None = None):
    if cfg.materialize_hook is not None:
        return cfg.materialize_hook(tbl, table_prefix)
    return pdt.alias(table_prefix)
