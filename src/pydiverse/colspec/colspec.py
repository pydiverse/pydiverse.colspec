from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Self, overload, Mapping, Iterable

import structlog

from pydiverse.colspec import exc
from pydiverse.colspec.exc import ValidationError

try:
    import polars as pl
    from polars.datatypes import DataTypeClass
    PolarsDataType = pl.DataType | DataTypeClass
except ImportError:
    PolarsDataType = None
    # Create a new module with the given name.
    pl = types.ModuleType("polars")
    pl.DataFrame = None
    pl.LazyFrame = None
    pl.DataType = None
    pl.Series = None
    pl.Expr = None

try:
    # colspec has optional dependency to dataframely
    import dataframely as dy
    from dataframely.random import Generator
    from dataframely._polars import FrameType
except ImportError:
    Generator = None
    FrameType = None
    dy = types.ModuleType("dataframely")
    dy.DataFrame = None
    dy.LazyFrame = None
    dy.FailureInfo = None

try:
    # colspec has optional dependency to pydiverse.transform
    import pydiverse.transform as pdt
except ImportError:
    class Table:
        pass
    # Create a new module with the given name.
    pdt = types.ModuleType("pydiverse.transform")
    pdt.Table = Table


try:
    # colspec has optional dependency to pydiverse.pipedag
    import pydiverse.pipedag as dag
except ImportError:
    class Table:
        pass
    # Create a new module with the given name.
    dag = types.ModuleType("pydiverse.pipedag")
    dag.Table = Table


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
                fields["inner"] = {k: convert_to_dy(v) for k,v in value.inner.items()}
            else:
                # this case handles lists with inner types
                fields["inner"] = convert_to_dy(value.inner)
        return DyColClass(**{k:v for k,v in fields.items() if k in param_keys})
    else:
        return value


class ColSpecMeta(type):
    def __new__(cls, clsname, bases, attribs):
        # change bases (only ABC is a real base)
        bases = bases[0:1]
        return super().__new__(cls, clsname, bases, attribs)


class ColSpec(object, pdt.Table, dag.Table, pl.LazyFrame, pl.DataFrame, metaclass=ColSpecMeta):
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
        cls, data: pl.DataFrame | pl.LazyFrame, cast: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.validate(data, cast=cast)

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

    # ------------------------------------ CASTING ----------------------------------- #

    @overload
    @classmethod
    def cast_polars(cls, df: pl.DataFrame) -> dy.DataFrame[Self]: ...  # pragma: no cover

    @overload
    @classmethod
    def cast_polars(cls, df: pl.LazyFrame) -> dy.LazyFrame[Self]: ...  # pragma: no cover

    @classmethod
    def cast_polars(
        cls, df: pl.DataFrame | pl.LazyFrame
    ) -> dy.DataFrame[Self] | dy.LazyFrame[Self]:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.cast(df)


def convert_collection_to_dy(collection: Collection | type[Collection]) -> type[dy.Collection]:
    cls = collection.__class__ if isinstance(collection, Collection) else collection
    DynCollection = type[dy.Collection](
        cls.__name__,
        (dy.Collection,),
        {"__annotations__": convert_to_dy_anno_dict(cls.__annotations__)},
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
            return 1 <= len(union_types) <= 2 and sum(1 if (inspect.isclass(t) and issubclass(t, ColSpec)) else 0 for t in union_types) == 1 and all(t == type(None) for t in union_types if not (inspect.isclass(t) and issubclass(t, ColSpec)))
        return inspect.isclass(anno) and issubclass(anno, ColSpec)

    @staticmethod
    def new(anno: type):
        if isinstance(anno, types.UnionType):
            union_types = typing.get_args(anno)
            col_spec = [t for t in union_types if inspect.isclass(t) and issubclass(t, ColSpec)][0]
            is_optional = any(t == type(None) for t in union_types)
        else:
            col_spec = anno
            is_optional = False
        return MemberInfo(col_spec, is_optional)

    @staticmethod
    def common_primary_keys(col_specs: Iterable[type[ColSpec]]) -> set[str]:
        return set.intersection(*[set(col_spec.primary_keys()) for col_spec in col_specs])


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
        return self.validate_polars_data(self.__dict__, cast=cast, fault_tolerant=fault_tolerant)

    @classmethod
    def validate_polars_data(cls, data: Mapping[str, FrameType], *, cast: bool = False, fault_tolerant: bool = False) -> Self:
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
            return DynCollection.validate(data, cast=True)
        except dy_exc.ImplementationError as e:
            logger = structlog.getLogger(logger_name)
            logger.exception("Dataframely raised column specification implementation error")
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
        return self.is_valid_polars_data(self.__dict__, cast=cast, fault_tolerant=fault_tolerant)

    @classmethod
    def is_valid_polars_data(cls, data: Mapping[str, FrameType], *, cast: bool = False, fault_tolerant: bool = False) -> bool:
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
        return DynCollection.filter(data, cast=cast)


    def cast_polars(self) -> Self:
        self.finalize()
        return self.cast_polars_data(self.__dict__)


    @classmethod
    def cast_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        DynCollection = convert_collection_to_dy(cls)
        return DynCollection.cast(data)

    # -------------------------------- Member inquiries --------------------------- #

    @classmethod
    def members(cls) -> dict[str, MemberInfo]:
        """Information about the members of the collection."""
        return {k:MemberInfo.new(v) for k,v in cls.__annotations__.items() if MemberInfo.is_member(v)}

    @classmethod
    def member_col_specs(cls) -> dict[str, type[ColSpec]]:
        """The column specifications of all members of the collection."""
        return {k:MemberInfo.new(v).col_spec for k,v in cls.__annotations__.items() if MemberInfo.is_member(v)}

    @classmethod
    def required_members(cls) -> set[str]:
        """The names of all required members of the collection."""
        return {k for k,v in cls.__annotations__.items() if MemberInfo.is_member(v) and not MemberInfo.new(v).is_optional}

    @classmethod
    def optional_members(cls) -> set[str]:
        """The names of all optional members of the collection."""
        return {k for k,v in cls.__annotations__.items() if MemberInfo.is_member(v) and MemberInfo.new(v).is_optional}

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
        cls._validate_polars_input_keys(data)
        result: dict[str, FrameType] = {}
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                continue
            result[member_name] = member.col_spec.cast_polars(data[member_name])
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
        dfs = pl.collect_all([lf for lf in self.to_dict().values()])
        return self._init_polars_data(
            {key: dfs[i].lazy() for i, key in enumerate(self.to_dict().keys())}
        )

    # -------------------------- Polars/Parquet PERSISTENCE ----------------------- #

    def write_parquet(self, directory: Path):
        self.finalize()
        DynCollection = convert_collection_to_dy(self.__class__)
        coll = DynCollection._init({k:v for k,v in self.__dict__.items() if v is not None})
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
        return cls(**{member: None for member in cls.__annotations__.keys()})

    def finalize(self):
        # finalize builder stage and ensure that all dataclass members have been set
        errors = {
            member: getattr(self, member)
            for member, anno in self.__annotations__.items()
            if not isinstance(None, anno) and getattr(self, member) is None
        }
        assert len(errors) == 0, (
            f"Dataclass building was not finalized before usage. "
            f"Please make sure to assign the following members "
            f"on '{self}': {','.join(errors)}"
        )

    # ----------------------------------- UTILITIES ---------------------------------- #

    @classmethod
    def _init_polars_data(cls, data: Mapping[str, FrameType]) -> Self:
        out = cls()
        for member_name, member in cls.members().items():
            if member.is_optional and member_name not in data:
                setattr(out, member_name, None)
            else:
                setattr(out, member_name, data[member_name].lazy())
        return out

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
            logger = structlog.getLogger(
                __name__ + "." + cls.__name__ + ".cast"
            )
            logger.warning(
                f"Input provides {len(superfluous)} superfluous members that are "
                f"ignored: {', '.join(superfluous)}."
            )
