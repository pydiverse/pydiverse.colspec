# Copyright (c) QuantCo and pydiverse contributors 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
import sys
from collections import defaultdict
from collections.abc import Iterable

from pydiverse.common import Dtype


class ValidationError(Exception):
    """Error raised when :mod:`dataframely` validation encounters an issue."""

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return self.message


class ColumnValidationError(ValidationError):
    """Validation error raised when columns mismatch."""

    def __init__(
        self,
        missing: Iterable[str] = tuple(),
        extra: Iterable[str] = tuple(),
        actual: Iterable[str] = tuple(),
    ):
        msg = []
        if missing:
            msg.append(f"Missing columns: {', '.join(missing)}")
        if extra:
            msg.append(f"Additional columns: {', '.join(extra)}")
        super().__init__(
            f"{len(missing)} columns are missing: {', '.join(missing)}; "
            f"found: {', '.join(actual)}"
            if actual
            else "; ".join(msg)
            if msg
            else "Column validation failed"
        )
        self.missing = missing
        self.extra = extra
        self.actual = actual

    def __str__(self) -> str:
        details = [
            f" - Missing columns: {', '.join(self.missing)}",
            f" - Extra columns: {', '.join(self.extra)}",
            f" - Actual columns: {', '.join(self.actual)}",
        ]
        return "\n".join([f"{self.message}:"] + details)


class DtypeValidationError(ValidationError):
    """Validation error raised when column dtypes are wrong."""

    def __init__(self, errors: dict[str, tuple[Dtype, Dtype]]):
        super().__init__(f"{len(errors)} columns have an invalid dtype")
        self.errors = errors

    def __str__(self) -> str:
        details = [
            f" - '{col}': got dtype '{actual}' but expected '{expected}'"
            for col, (actual, expected) in self.errors.items()
        ]
        return "\n".join([f"{self.message}:"] + details)


class RuleValidationError(ValidationError):
    """Complex validation error raised when rule validation fails."""

    def __init__(self, errors: dict[str, int]):
        super().__init__(f"{len(errors)} rules failed validation")

        # Split into schema errors and column errors
        schema_errors: dict[str, int] = {}
        column_errors: dict[str, dict[str, int]] = defaultdict(dict)
        for name, count in sorted(errors.items()):
            if "|" in name:
                column, rule = name.split("|", maxsplit=1)
                column_errors[column][rule] = count
            else:
                schema_errors[name] = count

        self.schema_errors = schema_errors
        self.column_errors = column_errors

    def __str__(self) -> str:
        schema_details = [
            f" - '{name}' failed validation for {count:,} rows"
            for name, count in self.schema_errors.items()
        ]
        column_details = [
            msg
            for column, errors in self.column_errors.items()
            for msg in (
                [f" * Column '{column}' failed validation for {len(errors)} rules:"]
                + [
                    f"   - '{name}' failed for {count:,} rows"
                    for name, count in errors.items()
                ]
            )
        ]
        return "\n".join([f"{self.message}:"] + schema_details + column_details)


class MemberValidationError(ValidationError):
    """Validation error raised when multiple members of a collection fail validation."""

    def __init__(self, errors: dict[str, ValidationError]):
        super().__init__(f"{len(errors)} members failed validation")
        self.errors = errors

    def __str__(self):
        details = [
            f" > Member '{name}' failed validation:\n"
            + "\n".join("   " + line for line in str(error).split("\n"))
            for name, error in self.errors.items()
        ]
        return "\n".join([f"{self.message}:"] + details)


class ImplementationError(Exception):
    """Error raised when a schema is implemented incorrectly."""


class AnnotationImplementationErrorDetail(ImplementationError):
    def __init__(self, message: str, _type: type):
        self._type = _type
        super().__init__(message)


class AnnotationImplementationError(ImplementationError):
    """Error raised when the annotations of a collection are invalid."""

    def __init__(self, attr: str, kls: type):
        message = (
            "Annotations of a 'dy.Collection' may only be an (optional) "
            f"'ColSpec', but \"{attr}\" has type '{kls}'."
        )
        super().__init__(message)


class RuleImplementationError(ImplementationError):
    """Error raised when a rule is implemented incorrectly."""

    def __init__(self, name: str, return_dtype: Dtype, is_group_rule: bool):
        if is_group_rule:
            details = (
                " When implementing a group rule (i.e. when using the `group_by` "
                "parameter), make sure to use an aggregation function such as `.any()`,"
                " `.all()`, and others to reduce an expression evaluated on multiple "
                "rows in the same group to a single boolean value for the group."
            )
        else:
            details = ""

        message = (
            f"Validation rule '{name}' has not been implemented correctly. It "
            f"returns dtype '{return_dtype}' but it must return a boolean value."
            + details
        )
        super().__init__(message)


def colspec_exception(e: Exception) -> Exception:
    exc = sys.modules[__name__]
    err_type = getattr(exc, e.__class__.__name__)
    f = err_type.__new__(err_type)
    for c in dir(e):
        if not c.startswith("_"):
            setattr(f, c, getattr(e, c))
    # f.__dict__.update(e.__dict__)
    return f
