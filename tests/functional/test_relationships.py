# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import dy, pl

# -------------------------------------- SCHEMA -------------------------------------- #


class DepartmentColSpec(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)


class ManagerColSpec(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)
    name = cs.String(nullable=False)


class EmployeeColSpec(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)
    employee_number = cs.Int64(primary_key=True)
    name = cs.String(nullable=False)


# ------------------------------------- FIXTURES ------------------------------------- #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.fixture()
def departments() -> dy.LazyFrame[DepartmentColSpec]:
    return DepartmentColSpec.cast_polars(pl.LazyFrame({"department_id": [1, 2]}))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.fixture()
def managers() -> dy.LazyFrame[ManagerColSpec]:
    return ManagerColSpec.cast_polars(
        pl.LazyFrame({"department_id": [1], "name": ["Donald Duck"]})
    )


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
@pytest.fixture()
def employees() -> dy.LazyFrame[EmployeeColSpec]:
    return EmployeeColSpec.cast_polars(
        pl.LazyFrame(
            {
                "department_id": [2, 2, 2],
                "employee_number": [101, 102, 103],
                "name": ["Huey", "Dewey", "Louie"],
            }
        )
    )


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_one_to_one(
    departments: dy.LazyFrame[DepartmentColSpec],
    managers: dy.LazyFrame[ManagerColSpec],
):
    actual = dy.filter_relationship_one_to_one(
        departments, managers, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [1]


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_one_to_at_least_one(
    departments: dy.LazyFrame[DepartmentColSpec],
    employees: dy.LazyFrame[EmployeeColSpec],
):
    actual = dy.filter_relationship_one_to_at_least_one(
        departments, employees, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [2]
