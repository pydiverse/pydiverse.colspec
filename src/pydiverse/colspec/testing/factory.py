# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from typing import Any

from pydiverse.colspec import Filter
from pydiverse.colspec import Rule
from pydiverse.colspec import Collection
from pydiverse.colspec import Column
from pydiverse.colspec import ColSpec


def create_colspec(
    name: str,
    columns: dict[str, Column],
    rules: dict[str, Rule] | None = None,
) -> type[ColSpec]:
    """Dynamically create a new column specification with the provided name.

    Args:
        name: The name of the column specification.
        columns: The columns to set on the column specification. When properly defining the column specification,
            this would be the annotations that define the column types.
        rules: The custom non-column-specific validation rules. When properly defining
            the column specification, this would be the functions annotated with ``@dy.rule``.

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
