from __future__ import annotations

import inspect
import types
import typing
from abc import ABC
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

try:
    # colspec has optional dependency to dataframely
    import dataframely as dy
    from dataframely.random import Generator
except ImportError:
    dy = None
    Generator = None

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
    from src.pydiverse.colspec import Column
    if base_class == ColSpec:
        # stop base class iteration
        return {}
    elif inspect.isclass(base_class) and issubclass(base_class, ColSpec):
        dy_annos = {
            k: convert_to_dy(v)
            for k, v in base_class.__dict__.items()
            if not k.startswith("__") and isinstance(v, Column)
        }
        for base in base_class.__bases__:
            dy_annos.update(convert_to_dy_col_spec(base))
        return dy_annos
    else:
        return {}


def convert_to_dy_anno(annotation):
    if issubclass(annotation, ColSpec):
        col_spec = convert_to_dy_col_spec(annotation)
        return dy.LazyFrame[type(annotation.__name__, (dy.Schema,), col_spec.copy())]
    else:
        return annotation


def convert_to_dy_anno_dict(annotations: dict[str, typing.Any]):
    return {k: convert_to_dy_anno(v) for k, v in annotations.items()}


def convert_to_dy(annotations):
    from src.pydiverse.colspec import Column
    if isinstance(annotations, Column) and hasattr(dy, annotations.__class__.__name__):
        return getattr(dy, annotations.__class__.__name__)(**annotations.__dict__)
    else:
        return annotations


class ColSpec(ABC, pdt.Table, dag.Table):
    """Base class for all column specifications."""

    @classmethod
    def primary_keys(cls) -> list[str]:
        """Returns a list of column names that are marked as primary keys.

        Returns:
            list[str]: Names of columns that are primary keys
        """
        from src.pydiverse.colspec import Column
        result = [
            member
            for member in dir(cls)
            if isinstance(getattr(cls, member), Column)
            and getattr(cls, member).primary_key
        ]
        return result

    @classmethod
    def column_names(cls):
        from src.pydiverse.colspec import Column
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
    def sample(
        cls,
        num_rows: int = 1,
        generator: Generator | None = None,
        *,
        overrides: typing.Mapping[str, typing.Iterable[Any]] | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        dy_schema_cols = convert_to_dy_col_spec(cls)
        dy_schema = type[dy.Schema](cls.__name__, (dy.Schema,), dy_schema_cols.copy())
        return dy_schema.sample(num_rows, generator, overrides=overrides)


class Collection:
    def validate(self, fault_tolerant: bool = False):
        from dataframely.exc import (
            DtypeValidationError,
        )

        self.finalize()
        DynCollection = type[dy.Collection](
            self.__class__.__name__,
            (dy.Collection,),
            {"__annotations__": convert_to_dy_anno_dict(self.__annotations__)},
        )
        try:
            DynCollection.validate(self.__dict__, cast=True)
        except DtypeValidationError as e:
            # this is just hacky handling with the option to silently show errors
            logger = structlog.getLogger(
                __name__ + "." + self.__class__.__name__ + ".validate"
            )
            logger.exception("Dataframely validation failed")
            if not fault_tolerant:
                raise e

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

    @classmethod
    def build(cls):
        return cls(**{member: None for member in cls.__annotations__.keys()})
