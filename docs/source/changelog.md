# Changelog

## 0.2.11 (2025-09-11)
- implemented Column.metadata field
- updated sample_polars interface to match dataframely changes (num_rows=None by default)

## 0.2.10 (2025-09-10)
- implemented Column.name and Column.polars fields/properties (dataframely uses Column.col)
- implemented nicer __str__ and __repr__ for Column and ColSpec

## 0.2.9 (2025-09-09)
- support time_zone argument to cs.DateTime(). The argument will not do anything except for serving as documentation.

## 0.2.8 (2025-08-21)
- update dependency to pydiverse.common 0.3.12

## 0.2.7 (2025-08-21)
- support pdc.Enum dtype in cs.Enum

## 0.2.6 (2025-08-14)
- fix optional dependency with pyodbc

## 0.2.5 (2025-07-11)
- fix incompatibility with newer polars versions (e.g. 1.31.0)

## 0.2.4 (2025-07-03)
- dialect specific workaround for mssql

## 0.2.3 (2025-06-30)
- dialect specific workaround for mssql

## 0.2.2 (2025-06-26)
- fixed column order for mixed class and object columns in ColSpec

## 0.2.1 (2025-06-26)
- fixed filter implementation for classes as ColSpec columns

## 0.2.0 (2025-06-25)
- fixed multi-inheritance column specifications
- improved dataframely/colspec messup error messages

## 0.1.1 (2025-06-11)
- fixed pypi package dependencies

## 0.1.0 (2025-06-08)
Initial release.

- Mostly 1:1 copy of dataframely (including testbench)
- Support for SQL validation
- Support for Rules and Filters with pydiverse.transform syntax
