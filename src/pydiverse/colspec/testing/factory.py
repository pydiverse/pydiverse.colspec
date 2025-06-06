# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from pydiverse.colspec import Collection, ColSpec, Column, Filter, Rule, RulePolars


def create_colspec(
    name: str,
    columns: dict[str, Column],
    rules: dict[str, Rule | RulePolars] | None = None,
) -> type[ColSpec]:
    """Dynamically create a new column specification with the provided name.

    Args:
        name: The name of the column specification.
        columns: The columns to set on the column specification. When properly defining
            the column specification, this would be the annotations that define the
            column types.
        rules: The custom non-column-specific validation rules. When properly defining
            the column specification, this would be the functions annotated with
            ``@dy.rule``.

    Returns:
        The dynamically created column specification.
    """
    return type(name, (ColSpec,), {**columns, **(rules or {})})


def create_collection(
    name: str,
    colspecs: dict[str, type[ColSpec]],
    filters: dict[str, Filter] | None = None,
) -> type[Collection]:
    return create_collection_raw(
        name,
        annotations={
            name: colspec  # type: ignore
            for name, colspec in colspecs.items()
        },
        filters=filters,
    )


def create_collection_raw(
    name: str,
    annotations: dict[str, Any],
    filters: dict[str, Filter] | None = None,
) -> type[Collection]:
    return type(
        name,
        (Collection,),
        {
            "__annotations__": annotations,
            **(filters or {}),
        },
    )
