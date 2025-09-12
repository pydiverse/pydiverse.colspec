# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import operator
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Self

import structlog

from pydiverse.colspec.config import Config, alias_collection_fail, alias_subquery
from pydiverse.colspec.exc import (
    ColumnValidationError,
    ImplementationError,
    MemberValidationError,
    RuleValidationError,
    ValidationError,
)
from pydiverse.colspec.failure import FailureInfo
from pydiverse.colspec.optional_dependency import (
    ColExpr,
    FrameType,
    Generator,
    dy,
    pdt,
    pl,
)

from . import exc
from ._filter import Filter, FilterPolars
from .colspec import ColSpec, convert_to_dy_anno_dict


@dataclass(kw_only=True)
class CollectionMember:
    """An annotation class that configures different behavior for a collection member.

    Members:
        ignored_in_filters: Indicates that a member should be ignored in the
            ``@dy.filter`` methods of a collection. This also affects the computation
            of the shared primary key in the collection.

    Example:
        .. code:: python

            class MyCollection(dy.Collection):
                a: dy.LazyFrame[MySchema1]
                b: dy.LazyFrame[MySchema2]

                ignored_member: Annotated[
                    dy.LazyFrame[MySchema3],
                    dy.CollectionMember(ignored_in_filters=True)
                ]

                @dy.filter
                def my_filter(self) -> pl.DataFrame:
                    return self.a.join(self.b, on="shared_key")
    """

    #: Whether the member should be ignored in the filter method.
    ignored_in_filters: bool = False
    #: Whether the member's non-primary key columns should be inlined for sampling.
    #: This means that value overrides are supplied on the top-level rather than in
    #: a subkey with the member's name. Only valid if the member's primary key matches
    #: the collection's common primary key. Two members that share common column names
    #: may not both be inlined for sampling.
    inline_for_sampling: bool = False


@dataclass
class MemberInfo(CollectionMember):
    """Information about a member of a collection."""

    #: The schema of the member.
    col_spec: type[ColSpec]
    #: Whether the member is optional.
    is_optional: bool

    @staticmethod
    def is_member(anno: type):
        if dy.Column is not None:
            if inspect.isclass(anno) and issubclass(anno, dy.Schema):
                raise exc.AnnotationImplementationErrorDetail(
                    "Table annotations in Collections must not be a "
                    "dataframely Schema type. Use ColSpec instead.",
                    anno,
                )

        if isinstance(anno, types.UnionType):
            union_types = typing.get_args(anno)
            if dy.Column is not None:
                dy_schemas_in_union = sum(
                    1 if (inspect.isclass(t) and issubclass(t, dy.Schema)) else 0
                    for t in union_types
                )
                if dy_schemas_in_union > 0:
                    raise exc.AnnotationImplementationErrorDetail(
                        "Table annotations in Collections must not be a "
                        "dataframely Schema type. Use ColSpec instead. Found: "
                        f"{union_types}",
                        anno,
                    )
            col_specs_in_union = sum(
                1 if (inspect.isclass(t) and issubclass(t, ColSpec)) else 0
                for t in union_types
            )
            if col_specs_in_union > 1:
                raise exc.AnnotationImplementationErrorDetail(
                    "Table annotations in Collections must contain at most one "
                    f"ColSpec type. Found: {union_types}",
                    anno,
                )
            all_but_one_none = all(
                t is type(None)
                for t in union_types
                if not (inspect.isclass(t) and issubclass(t, ColSpec))
            )
            if not all_but_one_none and col_specs_in_union == 1:
                raise exc.AnnotationImplementationErrorDetail(
                    "Table annotations in Collections only allow None as second option "
                    f"next to ColSpec type. Found: {union_types}",
                    anno,
                )
            return (
                1 <= len(union_types) <= 2
                and col_specs_in_union == 1
                and all_but_one_none
            )
        return inspect.isclass(anno) and issubclass(anno, ColSpec)

    @staticmethod
    def new(anno: type):
        if isinstance(anno, types.UnionType):
            union_types = typing.get_args(anno)
            col_spec = [
                t for t in union_types if inspect.isclass(t) and issubclass(t, ColSpec)
            ][0]
            is_optional = any(t == type(None) for t in union_types)  # noqa: E721
        else:
            col_spec = anno
            is_optional = False
        return MemberInfo(col_spec, is_optional)

    @staticmethod
    def common_primary_keys(col_specs: Iterable[type[ColSpec]]) -> set[str]:
        return set.intersection(
            *[set(col_spec.primary_keys()) for col_spec in col_specs]
        )


class Collection:
    """Base class for all collections of tables with a predefined column specification.

    A collection is comprised of a set of *members* which are collectively "consistent",
    meaning they the collection ensures that invariants are held up *across* members.
    This is different to :mod:`dataframely` schemas which only ensure invariants
    *within* individual members.

    In order to properly ensure that invariants hold up across members, members must
    have a "common primary key", i.e. there must be an overlap of at least one primary
    key column across all members. Consequently, a collection is typically used to
    represent "semantic objects" which cannot be represented in a single table due
    to 1-N relationships that are managed in separate tables.

    A collection must only have type annotations for :class:`~pydiverse.colspec.ColSpec`
    with known column specification:

    .. code:: python
        class MyFirstColSpec:
            a: Integer

        class MyCollection(cs.Collection):
            first_member: MyFirstColSpec
            second_member: MySecondColSpec

    Besides, it may define *filters* (c.f. :meth:`~dataframely.filter`) and arbitrary
    methods.

    A colspec.Collection can also be instantiated and filled with
    pydiverse transform Table, pipedag Table objects, or pipedag task outputs which
    reference a table. This yields quite intuitive syntax:

    .. code:: python

        c = MyCollection.build()
        c.first_member = pipdag_task1()
        c.second_member = pipdag_task2()
        pipdag_task3(c)

    Attention:
        Do NOT use this class in combination with ``from __future__ import annotations``
        as it requires the proper schema definitions to ensure that the collection is
        implemented correctly.
    """

    def get_pdt(self, name: str, cfg: Config = Config.default) -> pdt.Table:
        tbl = getattr(self, name)
        if not isinstance(tbl, pdt.Table):
            raise TypeError(
                f"Collection member '{name}' is not a pydiverse transform Table, but "
                f"{type(tbl)}; collection={self}"
            )
        if cfg.fix_table_names:
            # fix the name of the table according to Collection member name
            tbl._ast.name = name
        return tbl

    def validate(self, *, cast: bool = False):
        out, failure = self.filter(cast=cast)
        if any(len(getattr(failure, tbl_name)) > 0 for tbl_name in failure.members()):
            raise MemberValidationError(
                {
                    tbl_name: RuleValidationError(getattr(failure, tbl_name).counts())
                    for tbl_name in failure.members()
                }
            )
        return out

    def validate_polars(self, *, cast: bool = False, fault_tolerant: bool = False):
        self.finalize()
        return self.validate_polars_data(
            self.__dict__, cast=cast, fault_tolerant=fault_tolerant
        )

    @classmethod
    def validate_polars_data(
        cls,
        data: Mapping[str, FrameType],
        *,
        cast: bool = False,
        fault_tolerant: bool = False,
    ) -> Self:
        """Validate that a set of data frames satisfy the collection's invariants.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
            ValidationError: If any of the input data frames does not satisfy its schema
                definition or the filters on this collection result in the removal of at
                least one row across any of the input data frames.

        Returns:
            An instance of the collection. All members of the collection are guaranteed
            to be valid with respect to their respective schemas and the filters on this
            collection did not remove rows from any member.
        """
        import dataframely.exc as dy_exc
        import polars.exceptions as plexc

        DynCollection = convert_collection_to_dy(cls)
        logger_name = __name__ + "." + cls.__name__ + ".validate_polars"
        try:
            return cls.from_dy_collection(DynCollection.validate(data, cast=True))
        except dy_exc.ImplementationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception(
                "Dataframely raised column specification implementation error"
            )
            if not fault_tolerant:
                raise exc.ImplementationError(str(e))  # noqa: B904
        except plexc.InvalidOperationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception("Dataframely validation failed within polars expression")
            if not fault_tolerant:
                raise ValidationError(str(e))  # noqa: B904
        except dy_exc.ValidationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception("Dataframely validation failed")
            if not fault_tolerant:
                # Try to replicate exact error class. However, the constructor
                # does not always store the arguments to it directly.
                exc_class = getattr(exc, e.__class__.__name__)
                if hasattr(e, "errors"):
                    raise exc_class(e.errors)  # noqa: B904
                elif hasattr(e, "schema_errors") and hasattr(e, "column_errors"):
                    new_e = exc_class({})
                    new_e.schema_errors = e.schema_errors
                    new_e.column_errors = e.column_errors
                    raise new_e  # noqa: B904
                else:
                    raise ValidationError(str(e))  # noqa: B904
        return cls._init_polars_data(data)  # ignore validation if fault_tolerant

    def is_valid(
        self,
        *,
        cast: bool = False,
    ) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Returns:
            Whether the provided members satisfy the invariants of the collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
        """
        try:
            self.validate(cast=cast)
            return True
        except ValidationError:
            return False

    def is_valid_polars(self, *, cast: bool = False, fault_tolerant: bool = False):
        self.finalize()
        return self.is_valid_polars_data(
            self.__dict__, cast=cast, fault_tolerant=fault_tolerant
        )

    @classmethod
    def is_valid_polars_data(
        cls,
        data: Mapping[str, FrameType],
        *,
        cast: bool = False,
        fault_tolerant: bool = False,
    ) -> bool:
        """Utility method to check whether :meth:`validate` raises an exception.

        Args:
            data: The members of the collection which ought to be validated. The
                dictionary must contain exactly one entry per member with the name of
                the member as key. The existence of all keys is checked via the
                :mod:`dataframely` mypy plugin.
            cast: Whether columns with a wrong data type in the member data frame are
                cast to their schemas' defined data types if possible.

        Returns:
            Whether the provided members satisfy the invariants of the collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.
        """
        try:
            cls.validate_polars_data(data, cast=cast, fault_tolerant=fault_tolerant)
            return True
        except ValidationError:
            return False

    def filter(
        self, *, cast: bool = False, cfg: Config = Config.default
    ) -> tuple[Self, Self]:
        """Filter rows which conform to column specifications and collections rules.

        Returns a tuple of two new collections one with the filtered tables as member
        variables and one with FailureInfo objects as member variables.
        """
        from pydiverse.transform.extended import (
            C,
            drop,
            full_join,
            group_by,
            left_join,
            mutate,
            select,
            summarize,
        )

        self.finalize(assert_pdt=True)

        members: dict[str, MemberInfo] = self.members()

        # first filter the tables individually, else invalid rows could cause incorrect
        # results in the multi-table filters
        individually_filtered = self.__class__.build()
        table_level_fail = self.__class__.build()

        for name, member in members.items():
            out, failure = member.col_spec.filter(self.get_pdt(name, cfg), cast=cast)
            setattr(individually_filtered, name, out)
            setattr(table_level_fail, name, failure)

        # join tables needed for executing filter rules
        join_members: dict[str, set[str]] = {name: set() for name in members}
        extra_rules: dict[str, dict[str, ColExpr]] = {name: {} for name in members}

        # for reuse of subqueries across different tables:
        @dataclass
        class GroupSubquery:
            # grouping keys of subquery
            keys: set[str]
            # table names that need subquery for filtering
            tbls: set[str]
            # table names joined in subquery
            join_tbls: set[str]
            # dict[filter name, filter expression]
            cols: dict[str, ColExpr]

            @staticmethod
            def build():
                return GroupSubquery(set(), set(), set(), {})

        group_subqueries: dict[tuple[str], GroupSubquery] = {}

        for pred_name, pred in self.filter_rules().items():
            logic = pred.logic_fn(self)
            expr_tbl_names = [
                tbl_name
                for tbl_name in members.keys()
                if logic.uses_table(self.get_pdt(tbl_name, cfg))
            ]
            expr_col_specs = [members[tbl_name].col_spec for tbl_name in expr_tbl_names]

            expr_pk_union = self._pk_union(*expr_col_specs)

            for name in self.members().keys():
                tbl = members[name].col_spec
                join = self.get_join(name, set(expr_tbl_names) - {name}, cfg=cfg)
                pk_overlap = expr_pk_union.intersection(tbl.primary_keys())
                requires_grouping = (
                    len(expr_pk_union.difference(tbl.primary_keys())) > 0
                )

                if join is not None:
                    if requires_grouping:
                        key = tuple(
                            *pk_overlap, 0, *sorted(tbl.primary_keys() + expr_pk_union)
                        )
                        if key not in group_subqueries:
                            group_subqueries[key] = GroupSubquery.build()
                        group_subqueries[key].keys = pk_overlap
                        group_subqueries[key].tbls |= name
                        group_subqueries[key].join_tbls |= set(expr_tbl_names)
                        group_subqueries[key].cols[pred_name] = logic
                    else:
                        join_members[name] |= set(expr_tbl_names) - {name}
                        extra_rules[name][pred.logic_fn.__name__] = logic

        join_subqueries: dict[str, list[tuple[pdt.Table, set[str]]]] = {
            name: [] for name in members
        }

        for group_subquery in group_subqueries.values():
            for name in group_subquery.tbls:
                subquery = (
                    self._get_join(*group_subquery.join_tbls)
                    >> group_by(group_subquery.keys)
                    >> summarize(**group_subquery.cols)
                )
                join_subqueries[name].append((subquery, group_subquery.keys))
                for col in group_subquery.cols.keys():
                    extra_rules[name][col] = subquery[col]

        new = self.__class__.build()
        fail = self.__class__.build()

        for name in self.members().keys():
            tbl = getattr(individually_filtered, name)
            col_spec = self.member_col_specs()[name]

            if len(join_members[name]) > 0:
                filter_join = individually_filtered.get_join(
                    name, *join_members[name], cfg=cfg
                )
            else:
                filter_join = tbl
            if len(join_subqueries[name]):
                for subquery, keys in join_subqueries[name]:
                    filter_join >>= left_join(subquery >> alias_subquery(cfg), on=keys)

            # NOTE: we currently throw rows out if a rule results in null. We could also
            # keep everything, but then we should do the same if no match in the join
            # is found.
            extra_rules[name] = {
                name: rule.fill_null(False) for name, rule in extra_rules[name].items()
            }

            # The above left_join should not duplicate rows of `tbl`, so we don't need
            # to care about uniqueness here.
            ok_rows = (
                filter_join
                >> pdt.filter(*extra_rules.get(name).values())
                >> select(*tbl)
            )
            setattr(new, name, ok_rows)

            collection_level_invalid_rows = (
                filter_join
                >> pdt.filter(~pdt.all(True, *extra_rules.get(name).values()))
                >> mutate(**extra_rules.get(name))
            )
            table_level_invalid_rows: pdt.Table = getattr(
                table_level_fail, name
            )._invalid_rows

            rule_columns: dict[str, ColExpr] = (
                getattr(table_level_fail, name).rule_columns | extra_rules[name]
            )

            if cfg.dialect_name == "mssql":

                def cast_bool(col):
                    # materialization to collection_level_invalid_rows
                    # converted some boolean columns to int
                    return (
                        (col != 0)
                        if col.name
                        # ruff thinks this is a loop/comprehension
                        in collection_level_invalid_rows  # noqa: B023
                        else col
                    )
            else:

                def cast_bool(col):
                    return col

            failure = (
                table_level_invalid_rows
                >> full_join(
                    (
                        coll_failure := collection_level_invalid_rows
                        >> alias_collection_fail(
                            cfg, self.__class__.__name__.lower() + "_coll_invalid"
                        )
                    )
                    >> drop(*col_spec.column_names()),
                    on=[
                        # collection_level_invalid_rows is based on
                        # individually_filtered and thus has unique primary keys
                        table_level_invalid_rows[pk] == coll_failure[pk]
                        for pk in col_spec.primary_keys()
                    ],
                )
                >> mutate(
                    **{
                        col_name: table_level_invalid_rows[col_name].fill_null(
                            coll_failure[col_name]
                        )
                        for col_name in col_spec.column_names()
                    }
                )
                >> mutate(
                    **{
                        rule: cast_bool(C[rule]).fill_null(True)
                        for rule in rule_columns.keys()
                    }
                )
            )

            original_tbl = getattr(table_level_fail, name).tbl
            setattr(fail, name, FailureInfo(original_tbl, failure, rule_columns, cfg))

        return new, fail

    def filter_rules(self) -> dict[str, Filter]:
        rules = {
            pred: getattr(self, pred)
            for pred in dir(self)
            if isinstance(getattr(self, pred), Filter)
        }
        if "_primary_key_" in rules:
            raise ImplementationError(
                "Collection cannot have a filter named '_primary_key_'"
            )
        return rules

    def _pk_overlap(
        self, tbl: str | type[ColSpec], *more_tbls: str | type[ColSpec]
    ) -> set[str]:
        tbls: list[ColSpec] = [
            self.member_col_specs()[t] if isinstance(t, str) else t
            for t in (tbl, *more_tbls)
        ]
        return set(tbls[0].primary_keys()).intersection(
            *(other.primary_keys() for other in tbls[1:])
        )

    def _pk_union(
        self, tbl: str | type[ColSpec], *more_tbls: Iterable[str | type[ColSpec]]
    ) -> set[str]:
        tbls: list[ColSpec] = [
            self.member_col_specs()[t] if isinstance(t, str) else t
            for t in (tbl, *more_tbls)
        ]
        return set(tbls[0].primary_keys()).union(
            *(other.primary_keys() for other in tbls[1:])
        )

    def _get_join(
        self, *tbls: Iterable[str], cfg: Config = Config.default
    ) -> pdt.Table | None:
        """
        Similar to get_join(), but without given leftmost table.

        It is used for constructing grouped subqueries.
        """
        col_specs = self.member_col_specs()
        primary_keyss = {
            name: spec.primary_keys()
            for name, spec in col_specs.items()
            if name in tbls
        }
        # the ordering should match that of get_join()
        ordered_tbls = sorted(
            primary_keyss.keys(), key=lambda name: (len(primary_keyss[name]), name)
        )
        return self.get_join(*ordered_tbls, cfg=cfg)

    def get_join(
        self, tbl: str, *more_tbls: Iterable[str], cfg: Config = Config.default
    ) -> pdt.Table | None:
        """
        Get a left join expression if tables should be joinable.

        This method is intended to be overridden in case the automatic detection by
        checking primary key overlap is not sufficient. The automatic detection assumes
        that when ordering more_tbls by number of primary keys, that every next table
        can be joined with its primary key columns to the left most table that has them.

        This heuristic fails for some cases where not all tables have at least one
        common primary key column. In such situations it is intended to be overridden
        with a version that solves the problem for the concrete tables in a collection.
        """
        more_tbls = list(more_tbls)
        result = self.get_pdt(tbl, cfg)
        col_specs = self.member_col_specs()
        primary_keyss = {
            name: spec.primary_keys()
            for name, spec in col_specs.items()
            if name in more_tbls
        }
        # it is important to ensure consistent ordering in case additional tables are
        # added to `more_tbls`
        ordered_tbls = sorted(
            primary_keyss.keys(), key=lambda name: (len(primary_keyss[name]), name)
        )
        primary_keyss[tbl] = col_specs[tbl].primary_keys()
        ordered_tbls = [tbl, *ordered_tbls]
        for i, name in enumerate(ordered_tbls):
            if i > 0:
                join_tbls = ordered_tbls[0:i]
                pk_set = set(primary_keyss[name])
                on = None
                for join_name in join_tbls:
                    pk_overlap = pk_set.intersection(primary_keyss[join_name])
                    if len(pk_overlap) > 0:
                        on = reduce(
                            operator.and_,
                            (
                                self.get_pdt(name, cfg)[f]
                                == self.get_pdt(join_name, cfg)[f]
                                for f in pk_overlap
                            ),
                            on or pdt.lit(True),
                        )
                        pk_set -= pk_overlap
                if on is None:
                    # one join table has no matching primary keys to previous tables
                    logger = structlog.getLogger(
                        __name__,
                        collection=self,
                        method="get_join",
                        members=self.members(),
                    )
                    logger.debug(
                        "Heuristic failed to join table {name}. "
                        "Please override get_join method in this collection.",
                        join_tbls=join_tbls,
                        primary_key=pk_set,
                    )
                    return None
                result = result >> pdt.left_join(self.get_pdt(name, cfg), on)
        return result

    def filter_polars(
        self, *, cast: bool = False
    ) -> tuple[Self, dict[str, dy.FailureInfo]]:
        self.finalize()
        return self.filter_polars_data(self.__dict__, cast=cast)

    @classmethod
    def filter_polars_data(
        cls, data: Mapping[str, FrameType], *, cast: bool = False
    ) -> tuple[Self, dict[str, dy.FailureInfo]]:
        DynCollection = convert_collection_to_dy(cls)
        coll, failure = DynCollection.filter(data, cast=cast)
        return cls.from_dy_collection(coll), failure

    def cast_polars(self) -> Self:
        self.finalize()
        return self.cast_polars_data(self.__dict__)

    @classmethod
    def cast_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        DynCollection = convert_collection_to_dy(cls)
        try:
            return cls.from_dy_collection(DynCollection.cast(data))
        except pl.exceptions.ColumnNotFoundError as e:
            # TODO: improve error message by checking missing and extra columns for
            #  all member tables
            raise ColumnValidationError() from e

    # -------------------------------- Member inquiries --------------------------- #

    @classmethod
    def members(cls) -> dict[str, MemberInfo]:
        """Information about the members of the collection."""

        def better_msg(fn, arg, *, name: str):
            try:
                return fn(arg)
            except exc.AnnotationImplementationErrorDetail as e:
                raise exc.AnnotationImplementationError(name, e._type) from e

        return {
            k: MemberInfo.new(v)
            for k, v in typing.get_type_hints(cls).items()
            if better_msg(MemberInfo.is_member, v, name=k)
        }

    @classmethod
    def member_col_specs(cls) -> dict[str, type[ColSpec]]:
        """The column specifications of all members of the collection."""
        return {
            k: MemberInfo.new(v).col_spec
            for k, v in typing.get_type_hints(cls).items()
            if MemberInfo.is_member(v)
        }

    @classmethod
    def required_members(cls) -> set[str]:
        """The names of all required members of the collection."""
        return {
            k
            for k, v in typing.get_type_hints(cls).items()
            if MemberInfo.is_member(v) and not MemberInfo.new(v).is_optional
        }

    @classmethod
    def optional_members(cls) -> set[str]:
        """The names of all optional members of the collection."""
        return {
            k
            for k, v in typing.get_type_hints(cls).items()
            if MemberInfo.is_member(v) and MemberInfo.new(v).is_optional
        }

    @classmethod
    def common_primary_keys(cls) -> list[str]:
        """The primary keys which are shared by all members of the collection."""
        return sorted(MemberInfo.common_primary_keys(cls.member_col_specs().values()))

    def to_dict(self) -> dict[str, ColSpec]:
        """Return a dictionary representation of this collection."""
        return {
            member: getattr(self, member)
            for member in self.member_col_specs()
            if getattr(self, member) is not None
        }

    # ---------------------------------- COLLECTION ---------------------------------- #

    def collect_all_polars(self) -> Self:
        """Collect all members of the collection.

        This method collects all members in parallel for maximum efficiency. It is
        particularly useful when :meth:`filter` is called with lazy frame inputs.

        Returns:
            The same collection with all members collected once.

        Note:
            As all collection members are required to be lazy frames, the returned
            collection's members are still "lazy". However, they are "shallow-lazy",
            meaning they are obtained by calling ``.collect().lazy()``.
        """
        import polars.exceptions as plexc

        try:
            dfs = pl.collect_all([lf for lf in self.to_dict().values()])
        except plexc.PolarsError as e:
            raise ValidationError(str(e)) from e
        return self._init_polars_data(
            {key: dfs[i].lazy() for i, key in enumerate(self.to_dict().keys())}
        )

    # -------------------------- Polars/Parquet PERSISTENCE ----------------------- #

    def write_parquet(self, directory: Path):
        self.finalize()
        DynCollection = convert_collection_to_dy(self.__class__)
        coll = DynCollection._init(
            {k: v for k, v in self.__dict__.items() if v is not None}
        )
        coll.write_parquet(directory)

    @classmethod
    def read_parquet(cls, directory: Path) -> Self:
        DynCollection = convert_collection_to_dy(cls)
        return DynCollection.read_parquet(directory)

    @classmethod
    def scan_parquet(cls, directory: Path) -> Self:
        DynCollection = convert_collection_to_dy(cls)
        return DynCollection.scan_parquet(directory)

    # ---------------------------------- BUILDING --------------------------------- #

    @classmethod
    def build(cls):
        try:
            return cls(**{member: None for member in cls.__annotations__.keys()})
        except TypeError:
            try:
                return cls(**{member: None for member in cls.members().keys()})
            except TypeError:
                try:
                    return cls()
                except TypeError:
                    raise ImplementationError(  # noqa: B904
                        "Failed constructing collection with empty members. Try adding"
                        " @dataclasses.dataclass annotation to a collection class you"
                        " like to build."
                    )

    def finalize(self, assert_pdt=False):
        # finalize builder stage and ensure that all dataclass members have been set
        errors = {
            member
            for member, info in self.members().items()
            if not info.is_optional and getattr(self, member) is None
        }
        assert len(errors) == 0, (
            f"Dataclass building was not finalized before usage. "
            f"Please make sure to assign the following members "
            f"on '{self}': {','.join(errors)}"
        )
        if assert_pdt:
            errors = {
                member: type(getattr(self, member))
                for member, info in self.members().items()
                if getattr(self, member) is not None
                and not isinstance(getattr(self, member), pdt.Table)
            }
            assert len(errors) == 0, (
                f"Collection includes other member type than expected. "
                f"The function you called expects pdt.Table members "
                f"in '{self}': {','.join(errors)}"
            )

    # ---------------------------------- SAMPLING --------------------------------- #

    @classmethod
    def sample(
        cls,
        num_rows: int | None = None,
        *,
        overrides: Sequence[Mapping[str, Any]] | None = None,
        generator: Generator | None = None,
    ) -> Self:
        """Create a random sample from the members of this collection.

        Just like sampling for schemas, **this method should only be used for testing**.
        Contrary to sampling for schemas, the core difficulty when sampling related
        values data frames is that they must share primary keys and individual members
        may have a different number of rows. For this reason, overrides passed to this
        function must be "row-oriented" (or "sample-oriented").

        Args:
            num_rows: The number of rows to sample for each member. If this is set to
                ``None``, the number of rows is inferred from the length of the
                overrides.
            overrides: The overrides to set values in member schemas. The overrides must
                be provided as a list of samples. The structure of the samples must be
                as follows:

                .. code::

                    {
                        "<primary_key_1>": <value>,
                        "<primary_key_2>": <value>,
                        "<member_with_common_primary_key>": {
                            "<column_1>": <value>,
                            ...
                        },
                        "<member_with_superkey_of_primary_key>": [
                            {
                                "<column_1>": <value>,
                                ...
                            }
                        ],
                        ...
                    }

                *Any* member/value can be left out and will be sampled automatically.
                Note that overrides for columns of members that are annotated with
                ``inline_for_sampling=True`` can be supplied on the top-level instead
                of in a nested dictionary.
            generator: The (seeded) generator to use for sampling data. If ``None``, a
                generator with random seed is automatically created.

        Returns:
            A collection where all members (including optional ones) have been sampled
            according to the input parameters.

        Attention:
            In case the collection has members with a common primary key, the
            `_preprocess_sample` method must return distinct primary key values for each
            sample. The default implementation does this on a best-effort basis but may
            cause primary key violations. Hence, it is recommended to override this
            method and ensure that all primary key columns are set.

        Raises:
            ValueError: If the :meth:`_preprocess_sample` method does not return all
                common primary key columns for all samples.
            ValidationError: If the sampled members violate any of the collection
                filters. If the collection does not have filters, this error is never
                raised. To prevent validation errors, overwrite the
                :meth:`_preprocess_sample` method appropriately.
        """
        DynCollection = convert_collection_to_dy(cls)
        return DynCollection.sample(num_rows, overrides=overrides, generator=generator)

    # ----------------------------------- UTILITIES ---------------------------------- #

    @classmethod
    def _init_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        out = cls.build()
        for member_name, member in cls.members().items():
            if member.is_optional and (
                member_name not in data or data[member_name] is None
            ):
                setattr(out, member_name, None)
            else:
                setattr(out, member_name, data[member_name].lazy())
        return out

    @classmethod
    def from_dy_collection(cls, c: dy.Collection) -> Self:
        return cls._init_polars_data(
            {name: getattr(c, name) for name in c.members().keys()}
        )

    @classmethod
    def _validate_polars_input_keys(cls, data: Mapping[str, FrameType]):
        actual = set(data)

        missing = cls.required_members() - actual
        if len(missing) > 0:
            raise ValueError(
                f"Input misses {len(missing)} required members: {', '.join(missing)}."
            )

        superfluous = actual - set(cls.members())
        if len(superfluous) > 0:
            logger = structlog.getLogger(__name__ + "." + cls.__name__ + ".cast")
            logger.warning(
                f"Input provides {len(superfluous)} superfluous members that are "
                f"ignored: {', '.join(superfluous)}."
            )

    def pk_is_null(self, tbl: pdt.Table) -> ColExpr:
        tbl_name = next(attr for attr in dir(self) if getattr(self, attr) == tbl)
        return tbl[self.member_col_specs()[tbl_name].primary_keys()[0]].is_null()


def convert_filter_to_dy(f: FilterPolars):
    return dy._filter.Filter(f.logic)


def convert_collection_to_dy(
    collection: Collection | type[Collection],
) -> type[dy.Collection]:
    from pydiverse.colspec import FilterPolars

    cls = collection.__class__ if isinstance(collection, Collection) else collection
    filters = {
        k: convert_filter_to_dy(v)
        for k, v in collection.__dict__.items()
        if isinstance(v, FilterPolars)
    }
    preprocess = {
        k: v for k, v in collection.__dict__.items() if k == "_preprocess_sample"
    }
    DynCollection = type[dy.Collection](
        cls.__name__,
        (dy.Collection,),
        {
            "__annotations__": convert_to_dy_anno_dict(
                typing.get_type_hints(cls, include_extras=True)
            ),
            **filters,
            **preprocess,
        },
    )  # type:type[dy.Collection]
    return DynCollection
