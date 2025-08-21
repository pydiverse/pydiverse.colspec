# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar, overload

from pydiverse.colspec.optional_dependency import pa


class classproperty(property):  # noqa: N801
    """Replacement for the deprecated @classmethod @property decorator combination.

    Usage:
        ```
        @classproperty
        def num_bytes(self) -> int:
            ...
        ```
    """

    def __get__(self, instance: Any, owner: type | None = None, /) -> Any:
        return self.fget(owner) if self.fget is not None else None


T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


def map_optional(
    fn: Callable[Concatenate[T, P], R],
    value: T | None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | None:
    if value is None:
        return None
    return fn(value, *args, **kwargs)


@overload
def first_non_null(
    *values: T | None, allow_null_response: Literal[True]
) -> T | None: ...


@overload
def first_non_null(*values: T | None, default: T) -> T: ...


def first_non_null(
    *values: T | None,
    default: T | None = None,
    allow_null_response: Literal[True] | None = None,
) -> T | None:
    """Returns the first element in a sequence that is not None."""
    for value in values:
        if value is not None:
            return value
    if allow_null_response:
        return None
    return default


def pydiverse_type_opinions(_type: pa.DataType):
    return (
        pa.string()
        if _type == pa.large_string()
        else pa.time64("us")
        if _type == pa.time64("ns")
        else pa.decimal128(31, 11)
        if pa.types.is_decimal(_type)
        else pa.list_(pydiverse_type_opinions(_type.value_type))
        if pa.types.is_large_list(_type)
        else _type
    )
