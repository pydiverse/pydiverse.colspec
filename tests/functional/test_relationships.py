# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

import pydiverse.colspec as cs

# -------------------------------------- SCHEMA -------------------------------------- #


class DepartmentSchema(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)


class ManagerSchema(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)
    name = cs.String(nullable=False)


class EmployeeSchema(cs.ColSpec):
    department_id = cs.Int64(primary_key=True)
    employee_number = cs.Int64(primary_key=True)
    name = cs.String(nullable=False)


# ------------------------------------- FIXTURES ------------------------------------- #


@pytest.fixture()
def departments() -> dy.LazyFrame[DepartmentSchema]:
    return DepartmentSchema.cast(pl.LazyFrame({"department_id": [1, 2]}))


@pytest.fixture()
def managers() -> dy.LazyFrame[ManagerSchema]:
    return ManagerSchema.cast(
        pl.LazyFrame({"department_id": [1], "name": ["Donald Duck"]})
    )


@pytest.fixture()
def employees() -> dy.LazyFrame[EmployeeSchema]:
    return EmployeeSchema.cast(
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


def test_one_to_one(
    departments: dy.LazyFrame[DepartmentSchema],
    managers: dy.LazyFrame[ManagerSchema],
):
    actual = dy.filter_relationship_one_to_one(
        departments, managers, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [1]


def test_one_to_at_least_one(
    departments: dy.LazyFrame[DepartmentSchema],
    employees: dy.LazyFrame[EmployeeSchema],
):
    actual = dy.filter_relationship_one_to_at_least_one(
        departments, employees, on="department_id"
    )
    assert actual.select("department_id").collect().to_series().to_list() == [2]
