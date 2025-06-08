# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.columns import ColExpr
from pydiverse.colspec.optional_dependency import C, dy, pdt, pl
from pydiverse.colspec.pdt_util import num_rows


class CarColSpec(cs.ColSpec):
    vin = cs.String(primary_key=True)
    manufacturer = cs.String(nullable=False)

    @cs.filter()
    def not_empty(self) -> ColExpr:
        return self.manufacturer != ""


class CarPartColSpec(cs.ColSpec):
    vin = cs.String(primary_key=True)
    part = cs.String(primary_key=True)
    price = cs.Float64(primary_key=True)


class CarFleetPolars(pydiverse.colspec.collection.Collection):
    cars: CarColSpec
    car_parts: CarPartColSpec

    @cs.filter_polars()
    def not_car_with_vin_123(self) -> pl.LazyFrame:
        return self.cars.filter(pl.col("vin") != pl.lit("123"))


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_valid_failure_infos_polars():
    cars = {"vin": ["123", "456"], "manufacturer": ["BMW", "Mercedes"]}
    car_parts: dict[str, list[Any]] = {
        "vin": ["123", "123", "456"],
        "part": ["Motor", "Wheel", "Motor"],
        "price": [1000, 100, 1000],
    }
    car_fleet, failures = CarFleetPolars.filter_polars_data(
        {"cars": pl.DataFrame(cars), "car_parts": pl.DataFrame(car_parts)},
        cast=True,
    )

    assert len(car_fleet.cars.collect()) + len(failures["cars"].invalid()) == len(
        cars["vin"]
    )
    assert len(car_fleet.car_parts.collect()) + len(
        failures["car_parts"].invalid()
    ) == len(car_parts["vin"])
    assert len(failures["cars"].invalid()) == 1
    assert len(failures["car_parts"].invalid()) == 2


@dataclass
class CarFleet(pydiverse.colspec.collection.Collection):
    cars: CarColSpec
    car_parts: CarPartColSpec

    @cs.filter()
    def not_car_with_vin_123(self) -> ColExpr:
        return self.cars.vin != "123"


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_valid_failure_infos():
    cars = {"vin": ["123", "456"], "manufacturer": ["BMW", "Mercedes"]}
    car_parts: dict[str, list[Any]] = {
        "vin": ["123", "123", "456"],
        "part": ["Motor", "Wheel", "Motor"],
        "price": [1000, 100, 1000],
    }
    raw_fleet = CarFleet.build()  # type: CarFleet[pdt.Table]
    raw_fleet.cars = pdt.Table(cars)
    raw_fleet.car_parts = pdt.Table(car_parts)

    car_fleet, failures = raw_fleet.filter(cast=True)  # type: CarFleet[pdt.Table], CarFleet[FailureInfo]

    assert num_rows(car_fleet.cars) + num_rows(failures.cars.invalid_rows) == len(cars)
    assert num_rows(car_fleet.car_parts) + num_rows(
        failures.car_parts.invalid_rows
    ) == len(car_parts)
    assert num_rows(failures.cars.invalid_rows) == 1
    assert num_rows(failures.car_parts.invalid_rows) == 2
