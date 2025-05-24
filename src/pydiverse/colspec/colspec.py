from __future__ import annotations

import functools
import inspect
import itertools
import operator
import types
import typing
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, overload

import structlog

from pydiverse.colspec.columns._base import Column

from . import exc
from ._filter import Filter
from .exc import ImplementationError, ValidationError
from .failure import FailureInfo
from .optional_dependency import FrameType, Generator, dag, dy, pdt, pl

if TYPE_CHECKING:
    from pydiverse.colspec import FilterPolars


def convert_to_dy_col_spec(base_class):
    from pydiverse.colspec import Column

    if base_class == ColSpec:
        # stop base class iteration
        return {}
    elif inspect.isclass(base_class) and issubclass(base_class, ColSpec):
        dy_cols = {
            k: convert_to_dy(v)
            for k, v in base_class.__dict__.items()
            if isinstance(v, Column)
        } | {
            k: convert_to_dy(v())
            for k, v in base_class.__dict__.items()
            if inspect.isclass(v) and issubclass(v, Column)
        }
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
        DyColClass = getattr(dy, value.__class__.__name__)
        param_keys = set(inspect.signature(DyColClass.__init__).parameters.keys())
        fields = value.__dict__.copy()
        if "inner" in fields:
            if isinstance(value.inner, dict):
                # this case handles struct definitions
                fields["inner"] = {k: convert_to_dy(v) for k, v in value.inner.items()}
            else:
                # this case handles lists with inner types
                fields["inner"] = convert_to_dy(value.inner)
        return DyColClass(**{k: v for k, v in fields.items() if k in param_keys})
    else:
        return value


class ColSpecMeta(type):
    def __new__(cls, clsname, bases, attribs):
        # change bases (only ABC is a real base)
        bases = bases[0:1]
        return super().__new__(cls, clsname, bases, attribs)


class ColSpec(
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
    def primary_keys(cls) -> list[str]:
        """Returns a list of column names that are marked as primary keys.

        Returns:
            list[str]: Names of columns that are primary keys
        """
        from pydiverse.colspec import Column

        result = [
            member
            for member in dir(cls)
            if isinstance(getattr(cls, member), Column)
            and getattr(cls, member).primary_key
        ]
        return result

    @classmethod
    def column_names(cls):
        from pydiverse.colspec import Column

        result = [
            member for member in dir(cls) if isinstance(getattr(cls, member), Column)
        ]
        return result

    @classmethod
    def validate_polars(
        cls, data: pl.DataFrame | pl.LazyFrame, cast: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        try:
            return dy_schema.validate(data, cast=cast)
        except Exception as e:
            err_type = getattr(exc, e.__class__.__name__)
            f = err_type.__new__(err_type)
            f.__dict__.update(e.__dict__)
            raise f from e

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
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.sample(num_rows, generator, overrides=overrides)

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
        extra_rules: dict[str, pdt.ColExpr] | None = None,
        cast: bool = False,
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

        rules = cls._validation_rules(tbl) | (extra_rules or dict())
        combined = functools.reduce(operator.and_, rules.values(), True)
        ok_rows = tbl >> pdt.filter(combined)
        invalid_rows = tbl >> pdt.filter(~combined)
        return ok_rows, FailureInfo(
            tbl=tbl, invalid_rows=invalid_rows, rule_columns=rules
        )

    @classmethod
    def _validation_rules(cls, tbl: pdt.Table) -> dict[str, pdt.ColExpr]:
        return {
            f"{col}_{rule_name}": rule
            for col in cls.column_names()
            for rule_name, rule in getattr(cls, col).validation_rules(tbl[col]).items()
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
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.filter(df, cast=cast)


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
    DynCollection = type[dy.Collection](
        cls.__name__,
        (dy.Collection,),
        {
            "__annotations__": convert_to_dy_anno_dict(typing.get_type_hints(cls)),
            **filters,
        },
    )  # type:type[dy.Collection]
    return DynCollection


@dataclass
class MemberInfo:
    """Information about a member of a collection."""

    #: The schema of the member.
    col_spec: type[ColSpec]
    #: Whether the member is optional.
    is_optional: bool

    @staticmethod
    def is_member(anno: type):
        if isinstance(anno, types.UnionType):
            union_types = typing.get_args(anno)
            return (
                1 <= len(union_types) <= 2
                and sum(
                    1 if (inspect.isclass(t) and issubclass(t, ColSpec)) else 0
                    for t in union_types
                )
                == 1
                and all(
                    t is type(None)
                    for t in union_types
                    if not (inspect.isclass(t) and issubclass(t, ColSpec))
                )
            )
        return inspect.isclass(anno) and issubclass(anno, ColSpec)

    @staticmethod
    def new(anno: type):
        if isinstance(anno, types.UnionType):
            union_types = typing.get_args(anno)
            col_spec = [
                t for t in union_types if inspect.isclass(t) and issubclass(t, ColSpec)
            ][0]
            is_optional = any(t == type(None) for t in union_types)
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

    A collection must only have type annotations for :class:`~pydiverse.colspec.ColSpec`s
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
    reference a table. This yields quite intutive syntax:

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
                raise exc.ImplementationError(e.message)
        except plexc.InvalidOperationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception("Dataframely validation failed within polars expression")
            if not fault_tolerant:
                raise ValidationError(e.message)
        except dy_exc.ValidationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception("Dataframely validation failed")
            if not fault_tolerant:
                # Try to replicate exact error class. However, the constructor
                # does not always store the arguments to it directly.
                exc_class = getattr(exc, e.__class__.__name__)
                if hasattr(e, "errors"):
                    raise exc_class(e.errors)
                elif hasattr(e, "schema_errors") and hasattr(e, "column_errors"):
                    new_e = exc_class({})
                    new_e.schema_errors = e.schema_errors
                    new_e.column_errors = e.column_errors
                    raise new_e
                else:
                    raise ValidationError(e.message)
        return cls._init_polars_data(data)  # ignore validation if fault_tolerant

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

    def filter(self, *, cast: bool = False) -> tuple[Self, Self]:
        """Filter rows which conform to column specifications and collections rules.

        Returns a tuple of two new collections one with the filtered tables as member
        variables and one with FailureInfo objects as member variables.
        """
        from pydiverse.transform.extended import summarize

        self.finalize(assert_pdt=True)

        new_coll = self.__class__.build()
        fail_coll = self.__class__.build()

        members: dict[str, MemberInfo] = self.members()

        for name, member in members.items():
            out, failure = member.col_spec.filter(member.col_spec.tbl, cast=cast)
            setattr(new_coll, name, out)
            setattr(fail_coll, name, failure)
        # TODO: make sure that the filtered members are used everywhere and add the
        # 'how' flag.

        join_members: dict[str, set[str]] = {name: {name} for name in members}
        join_subqueries: dict[str, list[tuple[pdt.Table, set[str]]]] = {
            name: [] for name in members
        }
        extra_rules: dict[str, dict[str, pdt.ColExpr]] = {name: {} for name in members}

        for pred in self.filter_rules().values():
            logic = pred.logic(self)
            expr_tbl_names = [
                tbl_name
                for tbl_name in members.keys()
                if logic.uses_table(getattr(self, tbl_name))
            ]
            expr_col_specs = list(
                itertools.chain(
                    members[tbl_name].col_spec for tbl_name in expr_tbl_names
                )
            )

            expr_pks = self._pk_overlap(*expr_col_specs)
            expr_pk_union = self._pk_union(*expr_col_specs)
            group_subqueries: dict[tuple[str], pdt.Table] = {}

            for tbl_name in self.members().keys():
                tbl = members[tbl_name].col_spec
                pk_overlap = self._pk_overlap(tbl, *expr_col_specs)
                requires_grouping = set(tbl.primary_keys()).issubset(expr_pks) and len(
                    tbl.primary_keys()
                ) < len(expr_pks)

                # TODO: is it correct to only filter on a direct overlap? Or should it
                # rather be transitive? (If tables A and B share a key x and B and C
                # share y, should we filter A if expr_table_names = [C]?)
                if len(pk_overlap) > 0:
                    if requires_grouping:
                        key = tuple(sorted(pk_overlap))
                        if key not in group_subqueries:
                            group_subqueries[key] = self._get_join(
                                expr_tbl_names
                            ) >> summarize(__keep_this__=logic.any())
                        join_subqueries[tbl_name].append(
                            (group_subqueries[key], expr_pk_union)
                        )
                        extra_rules[tbl_name] += [group_subqueries[key].__keep_this__]
                    else:
                        join_members[tbl_name] |= set(expr_tbl_names)
                        extra_rules[tbl_name][pred.logic.__name__] = logic

        join: dict[str, pdt.Table] = dict()
        for name in self.members().keys():
            # It's important that 'name' goes first, so that the columns don't get
            # suffixes.
            join[name] = self._get_join(name, *join_members[name].difference({name}))
            pk_union = self._pk_union(*join_members[name])

            for subquery, subquery_pk_union in join_subqueries[name]:
                join[name] >>= pdt.inner_join(
                    subquery, on=subquery_pk_union.intersection(pk_union)
                )
                pk_union |= subquery_pk_union

        for tbl_name in self.members().keys():
            tbl, fail = self.members()[tbl_name].col_spec.filter(
                join[tbl_name],
                cast=cast,
                extra_rules=extra_rules.get(tbl_name),
            )
            setattr(new_coll, tbl_name, tbl)
            setattr(fail_coll, tbl_name, fail)

        return new_coll, fail_coll

    def filter_rules(self) -> dict[str, Filter]:
        return {
            pred: getattr(self, pred)
            for pred in dir(self)
            if isinstance(getattr(self, pred), Filter)
        }

    def _pk_overlap(
        self, tbl: str | type[ColSpec], *more_tbls: str | types[ColSpec]
    ) -> set[str]:
        tbls: list[ColSpec] = [
            self.member_col_specs()[t] if isinstance(t, str) else t
            for t in (tbl, *more_tbls)
        ]
        return set(tbls[0].primary_keys()).intersection(
            *(other.primary_keys() for other in tbls[1:])
        )

    def _pk_union(
        self, tbl: str | types[ColSpec], *more_tbls: str | type[ColSpec]
    ) -> set[str]:
        tbls: list[ColSpec] = [
            self.member_col_specs()[t] if isinstance(t, str) else t
            for t in (tbl, *more_tbls)
        ]
        return set(tbls[0].primary_keys()).union(
            *(other.primary_keys() for other in tbls[1:])
        )

    def _get_join(self, *tbls: Iterable[str]) -> pdt.Table:
        result = getattr(self, tbls[0])
        pk_union = set(self.member_col_specs()[tbls[0]].primary_keys())
        for name in tbls[1:]:
            col_spec: ColSpec = self.member_col_specs()[name]
            pk_set = set(col_spec.primary_keys())
            result = result >> pdt.inner_join(
                getattr(self, name), on=list(pk_set.intersection(pk_union))
            )
            pk_union |= pk_set
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
        return cls.from_dy_collection(DynCollection.cast(data))

    # -------------------------------- Member inquiries --------------------------- #

    @classmethod
    def members(cls) -> dict[str, MemberInfo]:
        """Information about the members of the collection."""
        return {
            k: MemberInfo.new(v)
            for k, v in typing.get_type_hints(cls).items()
            if MemberInfo.is_member(v)
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

    # ------------------------------------ CASTING ----------------------------------- #

    @classmethod
    def cast_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        """Initialize a collection by casting all members to correct column spec.

        This method calls :meth:`~ColSpec.cast` on every member, thus, removing
        superfluous columns and casting to the correct dtypes for all input data frames.

        You should typically use :meth:`validate` or :meth:`filter` to obtain instances
        of the collection as this method does not guarantee that the returned collection
        upholds any invariants. Nonetheless, it may be useful to use in instances where
        it is known that the provided data adheres to the collection's invariants.

        Args:
            data: The data for all members. The dictionary must contain exactly one
                entry per member with the name of the member as key.

        Returns:
            The initialized collection.

        Raises:
            ValueError: If an insufficient set of input data frames is provided, i.e. if
                any required member of this collection is missing in the input.

        Attention:
            For lazy frames, casting is not performed eagerly. This prevents collecting
            the lazy frames' schemas but also means that a call to :meth:`collect`
            further down the line might fail because of the cast and/or missing columns.
        """
        import polars.exceptions as plexc

        cls._validate_polars_input_keys(data)
        result: dict[str, FrameType] = {}
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                continue
            try:
                result[member_name] = member.col_spec.cast_polars(data[member_name])
            except plexc.PolarsError as e:
                raise ValidationError(str(e)) from e
        return cls._init_polars_data(result)

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
                    raise ImplementationError(
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

    # ----------------------------------- UTILITIES ---------------------------------- #

    @classmethod
    def _init_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        out = cls()
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
