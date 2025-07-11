# Changelog

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
