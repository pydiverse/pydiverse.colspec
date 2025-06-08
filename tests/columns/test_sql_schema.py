# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import pyodbc, sa
from pydiverse.colspec.testing import COLUMN_TYPES, create_colspec

try:
    from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
except ImportError:
    MSDialect_pyodbc = lambda: None  # noqa: E731
    PGDialect_psycopg2 = lambda: None  # noqa: E731


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize(
    ("column", "datatype"),
    [
        # (cs.Any(), "SQL_VARIANT"),
        (cs.Bool(), "BIT"),
        (cs.Date(), "DATE"),
        (cs.Datetime(), "DATETIME"),  # consider DATETIME2(6)
        (cs.Time(), "TIME"),  # consider TIME(6)
        (cs.Duration(), "DATETIME"),  # consider DATETIME2(6)
        (cs.Decimal(), "NUMERIC"),
        (cs.Decimal(12), "NUMERIC(12)"),
        (cs.Decimal(None, 8), "NUMERIC(38, 8)"),
        (cs.Decimal(6, 2), "NUMERIC(6, 2)"),
        (cs.Float(), "FLOAT(53)"),
        (cs.Float32(), "FLOAT(24)"),
        (cs.Float64(), "FLOAT(53)"),
        (cs.Integer(), "BIGINT"),
        (cs.Int8(), "SMALLINT"),
        (cs.Int16(), "SMALLINT"),
        (cs.Int32(), "INTEGER"),
        (cs.Int64(), "BIGINT"),
        (cs.UInt8(), "SMALLINT"),  # consider TINYINT
        (cs.UInt16(), "INTEGER"),
        (cs.UInt32(), "BIGINT"),
        (cs.UInt64(), "BIGINT"),
        (cs.String(), "VARCHAR(max)"),
        (cs.String(min_length=3), "VARCHAR(max)"),
        (cs.String(max_length=5), "VARCHAR(5)"),
        (cs.String(min_length=3, max_length=5), "VARCHAR(5)"),
        (cs.String(min_length=5, max_length=5), "CHAR(5)"),
        (cs.String(regex="[abc]de"), "VARCHAR(max)"),
        (cs.String(regex="^[abc]d$"), "VARCHAR(max)"),
        (cs.String(regex="^[abc]{1,3}d$"), "VARCHAR(max)"),
        (cs.Enum(["foo", "bar"]), "CHAR(3)"),
        (cs.Enum(["a", "abc"]), "VARCHAR(3)"),
    ],
)
def test_mssql_datatype(column: cs.Column, datatype: str):
    from sqlalchemy.dialects.mssql.base import MS_2017_VERSION
    from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc

    dialect = MSDialect_pyodbc()
    dialect.server_version_info = MS_2017_VERSION
    schema = create_colspec("test", {"a": column})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].type.compile(dialect) == datatype


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize(
    ("column", "datatype"),
    [
        (cs.Bool(), "BOOLEAN"),
        (cs.Date(), "DATE"),
        (cs.Datetime(), "TIMESTAMP WITHOUT TIME ZONE"),
        (cs.Time(), "TIME WITHOUT TIME ZONE"),
        (cs.Duration(), "INTERVAL"),
        (cs.Decimal(), "NUMERIC"),
        (cs.Decimal(12), "NUMERIC(12)"),
        (cs.Decimal(None, 8), "NUMERIC(38, 8)"),
        (cs.Decimal(6, 2), "NUMERIC(6, 2)"),
        (cs.Float(), "FLOAT(53)"),
        (cs.Float32(), "FLOAT(24)"),
        (cs.Float64(), "FLOAT(53)"),
        (cs.Integer(), "BIGINT"),
        (cs.Int8(), "SMALLINT"),
        (cs.Int16(), "SMALLINT"),
        (cs.Int32(), "INTEGER"),
        (cs.Int64(), "BIGINT"),
        (cs.UInt8(), "SMALLINT"),
        (cs.UInt16(), "INTEGER"),
        (cs.UInt32(), "BIGINT"),
        (cs.UInt64(), "BIGINT"),
        (cs.String(), "VARCHAR"),
        (cs.String(min_length=3), "VARCHAR"),
        (cs.String(max_length=5), "VARCHAR(5)"),
        (cs.String(min_length=3, max_length=5), "VARCHAR(5)"),
        (cs.String(min_length=5, max_length=5), "CHAR(5)"),
        (cs.String(regex="[abc]de"), "VARCHAR"),
        (cs.String(regex="^[abc]d$"), "VARCHAR"),
        (cs.String(regex="^[abc]{1,3}d$"), "VARCHAR"),
        (cs.Enum(["foo", "bar"]), "CHAR(3)"),
        (cs.Enum(["a", "abc"]), "VARCHAR(3)"),
    ],
)
def test_postgres_datatype(column: cs.Column, datatype: str):
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2

    dialect = PGDialect_psycopg2()
    schema = create_colspec("test", {"a": column})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].type.compile(dialect) == datatype


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc()])
def test_sql_nullability(
    column_type: type[cs.Column], nullable: bool, dialect: sa.Dialect
):
    schema = create_colspec("test", {"a": column_type(nullable=nullable)})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].nullable == nullable


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize("column_type", COLUMN_TYPES)
@pytest.mark.parametrize("primary_key", [True, False])
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_sql_primary_key(
    column_type: type[cs.Column], primary_key: bool, dialect: sa.Dialect
):
    schema = create_colspec("test", {"a": column_type(primary_key=primary_key)})
    columns = schema.sql_schema(dialect)
    assert len(columns) == 1
    assert columns[0].primary_key == primary_key
    assert not columns[0].autoincrement


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_sql_multiple_columns(dialect: sa.Dialect):
    schema = create_colspec("test", {"a": cs.Int32(nullable=False), "b": cs.Integer()})
    assert len(schema.sql_schema(dialect)) == 2


@pytest.mark.skipif(
    sa.Column is None or pyodbc is None,
    reason="sqlalchemy and pyodbc are needed for this test",
)
@pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_list_column(dialect: sa.Dialect):
    # TODO: this probably should raise for MSSQL
    # with pytest.raises(
    #     NotImplementedError, match="SQL column cannot have 'List' type."
    # ):
    cs.List(cs.String()).dtype().to_sql()  # dialect


# @pytest.mark.skipif(sa.Column is None or pyodbc is None, reason="sqlalchemy and
#   pyodbc are needed for this test")
# @pytest.mark.parametrize("dialect", [MSDialect_pyodbc(), PGDialect_psycopg2()])
def test_raise_for_struct_column():  # dialect: sa.Dialect):
    with pytest.raises(
        NotImplementedError, match="Struct column type is not yet implemented"
    ):
        cs.Struct({"a": cs.String()}).dtype()
