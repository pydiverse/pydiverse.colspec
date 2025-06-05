# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import types
import typing
from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Self, overload

from pydiverse.colspec._validation import validate_columns, validate_dtypes
from pydiverse.colspec.columns._base import Column

from . import GroupRule, GroupRulePolars, Rule, RulePolars, exc
from .config import Config, alias_subquery
from .exc import (
    ImplementationError,
    RuleValidationError,
    ValidationError,
)
from .failure import FailureInfo
from .optional_dependency import Generator, dag, dy, pdt, pl

if TYPE_CHECKING:
    pass

_ORIGINAL_NULL_SUFFIX = "__orig_null__"


def convert_to_dy_col_spec(base_class):
    from pydiverse.colspec import Column

    if base_class == ColSpec:
        # stop base class iteration
        return {}
    elif inspect.isclass(base_class) and issubclass(base_class, ColSpec):
        dy_cols = (
            {
                k: convert_to_dy(v)
                for k, v in base_class.__dict__.items()
                if isinstance(v, Column)
            }
            | {
                k: convert_to_dy(v())
                for k, v in base_class.__dict__.items()
                if inspect.isclass(v) and issubclass(v, Column)
            }
            | {
                k: dy._rule.GroupRule(v.expr, v.group_columns)
                if isinstance(v, GroupRulePolars)
                else dy._rule.Rule(v.expr)
                for k, v in base_class.__dict__.items()
                if isinstance(v, RulePolars)
            }
        )
        for base in base_class.__bases__:
            dy_cols.update(convert_to_dy_col_spec(base))
        return dy_cols
    else:
        return {}


def convert_to_dy_anno(annotation):
    if isinstance(annotation, types.UnionType):
        anno_types = [convert_to_dy_anno(t) for t in typing.get_args(annotation)]
        return reduce(lambda x, y: x | y, anno_types)
    if inspect.isclass(annotation) and issubclass(annotation, ColSpec):
        col_spec = convert_to_dy_col_spec(annotation)
        return dy.LazyFrame[type(annotation.__name__, (dy.Schema,), col_spec.copy())]
    else:
        return annotation


def convert_to_dy_anno_dict(annotations: dict[str, typing.Any]):
    return {k: convert_to_dy_anno(v) for k, v in annotations.items()}


def convert_to_dy(value):
    from pydiverse.colspec import Column

    if isinstance(value, Column) and hasattr(dy, value.__class__.__name__):
        return value.to_dataframely()
    else:
        return value


class ColSpecMeta(type):
    def __new__(cls, clsname, bases, attribs):
        # change bases (only ABC is a real base)
        bases = bases[0:1]
        return super().__new__(cls, clsname, bases, attribs)


class ColSpecBase:
    #  this is just the same as object but ruff does not kill it
    pass


class ColSpec(
    ColSpecBase,
    FailureInfo,
    pdt.Table,
    dag.Table,
    pl.LazyFrame,
    pl.DataFrame,
    metaclass=ColSpecMeta,
):
    """Base class for all column specifications.

    The base classes here are just for code completion support when working with
    Collection objects that store actual data or table references. They are removed
    at runtime by a metaclass.
    """

    def __init__(self):
        pass

    @classmethod
    def get_subquery_name(cls, tbl: pdt.Table, rule_name: str) -> str:
        """Get the name of the subquery for a given rule.

        Args:
            tbl: The table to which the rule applies.
            rule_name: The name of the rule.

        Returns:
            The name of the subquery.
        """
        if tbl._ast.name is not None and tbl._ast.name != "<unnamed>":
            return f"{tbl._ast.name}_{rule_name}"
        else:
            return rule_name

    @classmethod
    def fail_dy_columns_in_colspec(cls):
        if dy is not None:
            if any(
                [
                    isinstance(getattr(cls, c), dy.Column | dy._rule.Rule)
                    for c in dir(cls)
                ]
            ):
                if any([isinstance(getattr(cls, c), dy.Column) for c in dir(cls)]):
                    raise ImplementationError(
                        "Dataframely Columns won't work in ColSpec classes. Most likely"
                        " you find the same column classes in pydiverse.colspec. With "
                        "import pydiverse.colspec as cs, `dy.Integer` becomes "
                        "`cs.Integer` for example."
                    )
                else:
                    raise ImplementationError(
                        "Dataframely Rules won't work in ColSpec classes. You can use "
                        "`@cs.rule` decorator instead of `@dy.rule` decorator in case "
                        "you import pydiverse.colspec as cs."
                    )
        if any(
            [
                isinstance(getattr(cls, c), staticmethod)
                and isinstance(getattr(cls, c).__wrapped__, Rule | RulePolars)
                for c in dir(cls)
            ]
        ):
            # TODO: better message showing member which causes error
            raise ImplementationError(
                "The @staticmethod decorator needs to be after @cs.rule decorator."
            )

    @classmethod
    def primary_keys(cls) -> list[str]:
        """Returns a list of column names that are marked as primary keys.

        Returns:
            list[str]: Names of columns that are primary keys
        """
        from pydiverse.colspec import Column

        cls.fail_dy_columns_in_colspec()
        result = [
            member
            for member in dir(cls)
            if isinstance(getattr(cls, member), Column)
            and getattr(cls, member).primary_key
        ]
        return result

    @classmethod
    def column_names(cls) -> list[str]:
        cls.fail_dy_columns_in_colspec()
        result = [
            getattr(cls, member).alias or member
            for member in dir(cls)
            if isinstance(getattr(cls, member), Column)
        ]
        return result

    @classmethod
    def columns(cls) -> dict[str, Column]:
        cls.fail_dy_columns_in_colspec()
        return {
            col.alias or name: col
            for name, col in cls.__dict__.items()
            if isinstance(col, Column)
        }

    @classmethod
    def alias_map(cls) -> dict[str, str]:
        return {
            col.alias or name: name
            for name, col in cls.__dict__.items()
            if isinstance(col, Column)
        }

    @classmethod
    def validate(cls, tbl: pdt.Table, cast: bool = False) -> pdt.Table:
        valid_rows, failure = cls.filter(tbl, cast=cast)
        if len(failure) > 0:
            raise RuleValidationError(failure.counts())
        return valid_rows

    @classmethod
    def validate_polars(
        cls, data: pl.DataFrame | pl.LazyFrame, cast: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        import dataframely.exc as dy_exc

        dy_schema_cols = convert_to_dy_col_spec(cls)
        try:
            dy_schema = type[dy.Schema](
                cls.__name__, (dy.Schema,), dy_schema_cols.copy()
            )
            return dy_schema.validate(data, cast=cast)
        except (dy_exc.ValidationError, dy_exc.ImplementationError) as e:
            err_type = getattr(exc, e.__class__.__name__)
            f = err_type.__new__(err_type)
            for c in dir(e):
                if not c.startswith("_"):
                    setattr(f, c, getattr(e, c))
            # f.__dict__.update(e.__dict__)
            raise f from e

    @classmethod
    def is_valid(cls, tbl: pdt.Table, *, cast: bool = False) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            tbl: The table to check for validity.
            allow_extra_columns: Whether to allow the data frame to contain columns
                that are not defined in the schema.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type before running validation. If set
                to ``False``, a wrong data type will result in a return value of
                ``False``.

        Returns:
            Whether the provided dataframe can be validated with this schema.
        """
        try:
            from polars.exceptions import (
                InvalidOperationError as PlInvalidOperationError,
            )
        except ImportError:
            PlInvalidOperationError = None

        try:
            cls.validate(tbl, cast=cast)
            return True
        except (ValidationError, PlInvalidOperationError):
            return False
        except Exception as e:  # pragma: no cover
            raise e

    @classmethod
    def is_valid_polars(
        cls, df: pl.DataFrame | pl.LazyFrame, *, cast: bool = False
    ) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            df: The data frame to check for validity.
            allow_extra_columns: Whether to allow the data frame to contain columns
                that are not defined in the schema.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type before running validation. If set
                to ``False``, a wrong data type will result in a return value of
                ``False``.

        Returns:
            Whether the provided dataframe can be validated with this schema.
        """
        import polars.exceptions as plexc

        try:
            cls.validate_polars(df, cast=cast)
            return True
        except (ValidationError, plexc.InvalidOperationError):
            return False
        except Exception as e:  # pragma: no cover
            raise e

    @classmethod
    def sample_polars(
        cls,
        num_rows: int = 1,
        generator: Generator | None = None,
        *,
        overrides: Mapping[str, Iterable[Any]] | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema: type[dy.Schema] = type[dy.Schema](
            cls.__name__, (dy.Schema,), dy_schema_cols.copy()
        )
        return dy_schema.sample(num_rows, generator=generator, overrides=overrides)

    @classmethod
    def create_empty_polars(cls) -> dy.DataFrame[Self]:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.create_empty()

    # ------------------------------------ CASTING ----------------------------------- #

    @overload
    @classmethod
    def cast_polars(
        cls, df: pl.DataFrame
    ) -> dy.DataFrame[Self]: ...  # pragma: no cover

    @overload
    @classmethod
    def cast_polars(
        cls, df: pl.LazyFrame
    ) -> dy.LazyFrame[Self]: ...  # pragma: no cover

    @classmethod
    def cast_polars(
        cls, df: pl.DataFrame | pl.LazyFrame
    ) -> dy.DataFrame[Self] | dy.LazyFrame[Self]:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.cast(df)

    @classmethod
    def polars_schema(cls) -> pl.Schema:
        return pl.Schema(
            {
                name: getattr(cls, name).dtype().to_polars()
                for name in dir(cls)
                if isinstance(getattr(cls, name), Column)
            }
        )

    # ----------------------------------- FILTERING ---------------------------------- #

    @classmethod
    def filter(
        cls,
        tbl: pdt.Table,
        *,
        cast: bool = False,
        cfg: Config = Config.default,
    ) -> tuple[Self, FailureInfo]:
        """Filter the table by the rules of this column specification.

        This method can be thought of as a "soft alternative" to :meth:`validate`.
        While :meth:`validate` raises an exception when a row does not adhere to the
        rules defined in the schema, this method simply filters out these rows and
        succeeds.

        Args:
            tbl: The data frame to filter for valid rows. The data frame is collected
                within this method, regardless of whether a :class:`~polars.DataFrame`
                or :class:`~polars.LazyFrame` is passed.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type if possible. Rows for which the
                cast fails for any column are filtered out.

        Returns:
            A tuple of the validated rows in the input data frame (potentially
            empty) and a simple dataclass carrying information about the rows of the
            data frame which could not be validated successfully.

        Raises:
            ValidationError: If the columns of the input data frame are invalid. This
                happens only if the data frame misses a column defined in the schema or
                a column has an invalid dtype while ``cast`` is set to ``False``.

        Note:
            This method preserves the ordering of the input data frame.
        """

        src_tbl = tbl

        tbl = cls._validate_columns(tbl, casting=("lenient" if cast else "none"))
        rules, group_rules = cls._validation_rules(tbl)

        if cast:
            dtype_rules = {
                f"{col}|dtype": (
                    tbl[col].is_null() == tbl[f"{col}{_ORIGINAL_NULL_SUFFIX}"]
                )
                for col in cls.column_names()
            }
            rules.update(dtype_rules)

        if "_primary_key_" in rules or "_primary_key_" in group_rules:
            raise ImplementationError(
                "@cs.rule annotated functions must not be called `_primary_key_`"
            )
        if len(cls.primary_keys()) > 0:
            group_rules["_primary_key_"] = GroupRule(
                pdt.count() == 1, group_columns=cls.primary_keys()
            )

        for name, group_rule in group_rules.items():
            subquery = (
                tbl
                >> pdt.group_by(*group_rule.group_columns)
                >> pdt.summarize(expr=group_rule.expr)
                >> alias_subquery(cfg, cls.get_subquery_name(tbl, name))
            )
            tbl >>= pdt.left_join(
                subquery,
                on=pdt.all(
                    *[tbl[col] == subquery[col] for col in group_rule.group_columns]
                ),
            ) >> pdt.select(*tbl)
            rules[name] = subquery.expr

        combined = pdt.all(True, *rules.values())

        if cast:
            # Rules other than the "dtype rule" might not be reliable if type casting
            # failed, i.e. if the "dtype rule" evaluated to `False`. For this reason,
            # we set all other rule evaluations to `null` in the case of dtype casting
            # failure.
            all_dtype_casts_valid = pdt.all(
                True, *(col for col in tbl if col.name[-6:] == "|dtype")
            )

            # remove original null information again
            tbl >>= pdt.drop(
                *(tbl[f"{col}{_ORIGINAL_NULL_SUFFIX}"] for col in cls.column_names())
            )

            rules = {
                name: pdt.when(all_dtype_casts_valid)
                .then(expr)
                .otherwise(pdt.lit(None, dtype=pdt.Bool))
                for name, expr in rules.items()
            }

        ok_rows = tbl >> pdt.filter(combined)
        invalid_rows = tbl >> pdt.filter(~combined) >> pdt.mutate(**rules)

        if len(cls.primary_keys()) > 0:
            ok_rows = ok_rows
            invalid_rows = invalid_rows

        return ok_rows, FailureInfo(
            tbl=src_tbl,
            invalid_rows=invalid_rows,
            rule_columns=rules,
            cfg=cfg,
        )

    @classmethod
    def _validate_columns(
        cls, tbl: pdt.Table, *, casting: Literal["none", "lenient", "strict"]
    ):
        cls.fail_dy_columns_in_colspec()

        tbl = validate_columns(tbl, expected=cls.column_names())

        if casting == "lenient":
            tbl >>= pdt.mutate(
                **{f"{col.name}{_ORIGINAL_NULL_SUFFIX}": col.is_null() for col in tbl}
            )

        return validate_dtypes(tbl, expected=cls.columns(), casting=casting)

    @classmethod
    def _validation_rules(
        cls, tbl: pdt.Table
    ) -> tuple[dict[str, pdt.ColExpr], dict[str, GroupRule]]:
        cls.fail_dy_columns_in_colspec()
        alias_map = cls.alias_map()
        return {
            f"{col}|{rule_name}": rule.fill_null(True)
            for col in cls.column_names()
            for rule_name, rule in getattr(cls, alias_map[col])
            .validation_rules(tbl[col])
            .items()
        } | {
            rule: getattr(cls, rule).expr
            for rule in dir(cls)
            if isinstance(getattr(cls, rule), Rule)
        }, {
            rule: getattr(cls, rule)
            for rule in dir(cls)
            if isinstance(getattr(cls, rule), GroupRule)
        }

    @classmethod
    def filter_polars(
        cls, df: pl.DataFrame | pl.LazyFrame, *, cast: bool = False
    ) -> tuple[dy.DataFrame[Self], dy.FailureInfo]:
        """Filter the data frame by the rules of this schema.

        This method can be thought of as a "soft alternative" to :meth:`validate`.
        While :meth:`validate` raises an exception when a row does not adhere to the
        rules defined in the schema, this method simply filters out these rows and
        succeeds.

        Args:
            df: The data frame to filter for valid rows. The data frame is collected
                within this method, regardless of whether a :class:`~polars.DataFrame`
                or :class:`~polars.LazyFrame` is passed.
            cast: Whether columns with a wrong data type in the input data frame are
                cast to the schema's defined data type if possible. Rows for which the
                cast fails for any column are filtered out.

        Returns:
            A tuple of the validated rows in the input data frame (potentially
            empty) and a simple dataclass carrying information about the rows of the
            data frame which could not be validated successfully.

        Raises:
            ValidationError: If the columns of the input data frame are invalid. This
                happens only if the data frame misses a column defined in the schema or
                a column has an invalid dtype while ``cast`` is set to ``False``.

        Note:
            This method preserves the ordering of the input data frame.
        """
        import dataframely.exc as dy_exc

        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())

        try:
            return dy_schema.filter(df, cast=cast)
        except (dy_exc.ValidationError, dy_exc.ImplementationError) as e:
            err_type = getattr(exc, e.__class__.__name__)
            f = err_type.__new__(err_type)
            f.__dict__.update(e.__dict__)
            raise f from e
