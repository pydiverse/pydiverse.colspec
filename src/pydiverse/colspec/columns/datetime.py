# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
from collections.abc import Callable
from typing import Any

import pydiverse.common as pdc

from ..optional_dependency import ColExpr
from ._base import (
    EPOCH_DATETIME,
    Column,
)
from ._mixins import OrdinalMixin

try:
    import pydiverse.transform as pdt
except ImportError:
    pdt = None

# ------------------------------------------------------------------------------------ #


class Date(OrdinalMixin[dt.date], Column):
    """A column of dates (without time)."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.date | None = None,  # noqa: A002
        min_exclusive: dt.date | None = None,
        max: dt.date | None = None,  # noqa: A002
        max_exclusive: dt.date | None = None,
        resolution: str | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum date for dates in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum date for dates in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that dates in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1mo`` expects all dates to be on the first of the
                month. Note that this setting does *not* affect the storage resolution.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                happen with metadata. It is just stored.
        """
        # TODO: implement date_matches_resolution
        # if resolution is not None:
        #     offset_time = pl.Series([EPOCH_DATETIME]).dt.offset_by(resolution) \
        #       .dt.time()
        #     if offset_time.item() != dt.time():
        #         raise ValueError("`resolution` is too fine for dates.")
        # if resolution is not None and min is not None:
        #     if not date_matches_resolution(min, resolution):
        #         raise ValueError("`min` does not match resolution.")
        # if resolution is not None and min_exclusive is not None:
        #     if not date_matches_resolution(min_exclusive, resolution):
        #         raise ValueError("`min_exclusive` does not match resolution.")
        # if resolution is not None and max is not None:
        #     if not date_matches_resolution(max, resolution):
        #         raise ValueError("`max` does not match resolution.")
        # if resolution is not None and max_exclusive is not None:
        #     if not date_matches_resolution(max_exclusive, resolution):
        #         raise ValueError("`max_exclusive` does not match resolution.")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.resolution = resolution

    def dtype(self) -> pdc.Date:
        return pdc.Date()

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)
        # # pydiverse.transform is currently not able to check resolution problems.
        # # However, it is a problem that can be solved by convention.
        # # pydiverse.pipedag will enforce those.
        # if self.resolution is not None:
        #     result["resolution"] = expr.dt.truncate(self.resolution) == expr
        return result


class Time(OrdinalMixin[dt.time], Column):
    """A column of times (without date)."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.time | None = None,  # noqa: A002
        min_exclusive: dt.time | None = None,
        max: dt.time | None = None,  # noqa: A002
        max_exclusive: dt.time | None = None,
        resolution: str | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum time for times in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum time for times in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that times in the column must have. This uses the
                formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all times to be full hours. Note
                that this setting does *not* affect the storage resolution.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                happen with metadata. It is just stored.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                names, the specified alias is the only valid name.
        """
        # # pydiverse.transform is currently not able to check resolution problems.
        # # However, it is a problem that can be solved by convention.
        # # pydiverse.pipedag will enforce those.
        # if resolution is not None:
        #     offset_date = pl.Series([EPOCH_DATETIME]).dt.offset_by(resolution) \
        #       .dt.date()
        #     if offset_date.item() != EPOCH_DATETIME.date():
        #         raise ValueError("`resolution` is too coarse for times.")
        # if resolution is not None and min is not None:
        #     if not time_matches_resolution(min, resolution):
        #         raise ValueError("`min` does not match resolution.")
        # if resolution is not None and min_exclusive is not None:
        #     if not time_matches_resolution(min_exclusive, resolution):
        #         raise ValueError("`min_exclusive` does not match resolution.")
        # if resolution is not None and max is not None:
        #     if not time_matches_resolution(max, resolution):
        #         raise ValueError("`max` does not match resolution.")
        # if resolution is not None and max_exclusive is not None:
        #     if not time_matches_resolution(max_exclusive, resolution):
        #         raise ValueError("`max_exclusive` does not match resolution.")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            metadata=metadata,
        )
        self.resolution = resolution

    def dtype(self) -> pdc.Time:
        return pdc.Time()

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)
        # # pydiverse.transform is currently not able to check resolution problems.
        # # However, it is a problem that can be solved by convention.
        # # pydiverse.pipedag will enforce those.
        # if self.resolution is not None:
        #     rounded_expr = (
        #         pdt.lit(EPOCH_DATETIME.date())
        #         .dt.combine(expr)
        #         .dt.truncate(self.resolution)
        #         .dt.time()
        #     )
        #     result["resolution"] = rounded_expr == expr
        return result


class Datetime(OrdinalMixin[dt.datetime], Column):
    """A column of datetimes."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.datetime | None = None,  # noqa: A002
        min_exclusive: dt.datetime | None = None,
        max: dt.datetime | None = None,  # noqa: A002
        max_exclusive: dt.datetime | None = None,
        resolution: str | None = None,
        time_zone: str | dt.tzinfo | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum datetime for datetimes in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum datetime for datetimes in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that datetimes in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all datetimes to be full hours. Note
                that this setting does *not* affect the storage resolution.
            time_zone: The time zone that datetimes in the column must have. The time
                zone must use a valid IANA time zone name identifier e.x. ``Etc/UTC`` or
                ``America/New_York``. It does not have any functional implications, but
                it serves as documentation in code.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                happen with metadata. It is just stored.
        """
        # if resolution is not None and min is not None:
        #     if not datetime_matches_resolution(min, resolution):
        #         raise ValueError("`min` does not match resolution.")
        # if resolution is not None and min_exclusive is not None:
        #     if not datetime_matches_resolution(min_exclusive, resolution):
        #         raise ValueError("`min_exclusive` does not match resolution.")
        # if resolution is not None and max is not None:
        #     if not datetime_matches_resolution(max, resolution):
        #         raise ValueError("`max` does not match resolution.")
        # if resolution is not None and max_exclusive is not None:
        #     if not datetime_matches_resolution(max_exclusive, resolution):
        #         raise ValueError("`max_exclusive` does not match resolution.")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.resolution = resolution
        self.time_zone = time_zone

    def dtype(self) -> pdc.Datetime:
        return pdc.Datetime()

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)
        # # pydiverse.transform is currently not able to check resolution problems.
        # # However, it is a problem that can be solved by convention.
        # # pydiverse.pipedag will enforce those.
        # if self.resolution is not None:
        #     result["resolution"] = expr.dt.truncate(self.resolution) == expr
        return result


class Duration(OrdinalMixin[dt.timedelta], Column):
    """A column of durations."""

    def __init__(
        self,
        *,
        nullable: bool = True,
        primary_key: bool = False,
        min: dt.timedelta | None = None,  # noqa: A002
        min_exclusive: dt.timedelta | None = None,
        max: dt.timedelta | None = None,  # noqa: A002
        max_exclusive: dt.timedelta | None = None,
        resolution: str | None = None,
        check: Callable[[ColExpr], ColExpr] | None = None,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            nullable: Whether this column may contain null values.
                Explicitly set `nullable=True` if you want your column to be nullable.
                In a future release, `nullable=False` will be the default if `nullable`
                is not specified.
            primary_key: Whether this column is part of the primary key of the schema.
                If ``True``, ``nullable`` is automatically set to ``False``.
            min: The minimum duration for durations in this column (inclusive).
            min_exclusive: Like ``min`` but exclusive. May not be specified if ``min``
                is specified and vice versa.
            max: The maximum duration for durations in this column (inclusive).
            max_exclusive: Like ``max`` but exclusive. May not be specified if ``max``
                is specified and vice versa.
            resolution: The resolution that durations in the column must have. This uses
                the formatting language used by :mod:`polars` datetime ``round`` method.
                For example, a value ``1h`` expects all durations to be full hours. Note
                that this setting does *not* affect the storage resolution.
            check: A custom rule or multiple rules to run for this column. This can be:
                - A single callable that returns a non-aggregated boolean expression.
                The name of the rule is derived from the callable name, or defaults to
                "check" for lambdas.
                - A list of callables, where each callable returns a non-aggregated
                boolean expression. The name of the rule is derived from the callable
                name, or defaults to "check" for lambdas. Where multiple rules result
                in the same name, the suffix __i is appended to the name.
                - A dictionary mapping rule names to callables, where each callable
                returns a non-aggregated boolean expression.
                All rule names provided here are given the prefix "check_".
            alias: An overwrite for this column's name which allows for using a column
                name that is not a valid Python identifier. Especially note that setting
                this option does _not_ allow to refer to the column with two different
                names, the specified alias is the only valid name.
            metadata: A dictionary of metadata to attach to the column. Nothing will
                happen with metadata. It is just stored.
        """
        # if resolution is not None and min is not None:
        #     if not timedelta_matches_resolution(min, resolution):
        #         raise ValueError("`min` does not match resolution.")
        # if resolution is not None and min_exclusive is not None:
        #     if not timedelta_matches_resolution(min_exclusive, resolution):
        #         raise ValueError("`min_exclusive` does not match resolution.")
        # if resolution is not None and max is not None:
        #     if not timedelta_matches_resolution(max, resolution):
        #         raise ValueError("`max` does not match resolution.")
        # if resolution is not None and max_exclusive is not None:
        #     if not timedelta_matches_resolution(max_exclusive, resolution):
        #         raise ValueError("`max_exclusive` does not match resolution.")

        super().__init__(
            nullable=nullable,
            primary_key=primary_key,
            min=min,
            min_exclusive=min_exclusive,
            max=max,
            max_exclusive=max_exclusive,
            check=check,
            alias=alias,
            metadata=metadata,
        )
        self.resolution = resolution

    def dtype(self) -> pdc.Duration:
        return pdc.Duration()

    def validation_rules(self, expr: ColExpr) -> dict[str, ColExpr]:
        result = super().validation_rules(expr)
        # # pydiverse.transform is currently not able to check resolution problems.
        # # However, it is a problem that can be solved by convention.
        # # pydiverse.pipedag will enforce those.
        # if self.resolution is not None:
        #     datetime = pdt.lit(EPOCH_DATETIME) + expr
        #     result["resolution"] = datetime.dt.truncate(self.resolution) == datetime

        return result


# --------------------------------------- UTILS -------------------------------------- #


def _next_date(t: dt.date, resolution: str | None) -> dt.date | None:
    result = _next_datetime(dt.datetime.combine(t, dt.time()), resolution)
    if result is None:
        return None
    return result.date()


def _next_datetime(t: dt.datetime, resolution: str | None) -> dt.datetime | None:
    from pydiverse.colspec.optional_dependency import pl

    result = pl.Series([t]).dt.offset_by(resolution or "1us")
    if result.dt.year().item() >= 10000:
        # The datetime is out-of-range for a Python datetime object
        return None
    return result.item()


def _next_time(t: dt.time, resolution: str | None) -> dt.time | None:
    from pydiverse.colspec.optional_dependency import pl

    result = pl.cast(
        # `None` can never happen as we can never reach another day by adding time
        dt.datetime,
        _next_datetime(dt.datetime.combine(EPOCH_DATETIME.date(), t), resolution),
    )
    result_time = result.time()
    return None if result_time == dt.time() else result_time


def _next_timedelta(t: dt.timedelta, resolution: str | None) -> dt.timedelta | None:
    from pydiverse.colspec.optional_dependency import pl

    result = pl.cast(
        dt.datetime,  # We run into out-of-date issues before reaching `None`
        _next_datetime(EPOCH_DATETIME + t, resolution),
    )
    return result - EPOCH_DATETIME
