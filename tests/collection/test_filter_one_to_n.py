# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.colspec import dy, pdt


class CarColSpec(cs.ColSpec):
    vin = cs.String(primary_key=True)
    manufacturer = cs.String(nullable=False)


class CarPartColSpec(cs.ColSpec):
    vin = cs.String(primary_key=True)
    part = cs.String(primary_key=True)
    price = cs.Float64(primary_key=True)


class CarFleetPolars(cs.Collection):
    cars: CarColSpec
    car_parts: CarPartColSpec

    @cs.filter_polars()
    def not_car_with_vin_123(self) -> pl.LazyFrame:
        return self.cars.filter(pl.col("vin") != pl.lit("123"))


@pytest.mark.skipif(dy is None, reason="dataframely not installed")
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
class CarFleet(cs.Collection):
    cars: CarColSpec
    car_parts: CarPartColSpec

    @cs.filter()
    def not_car_with_vin_123(self) -> pdt.ColExpr:
        return self.cars.vin != "123"


@pytest.mark.skipif(pdt is None, reason="pydiverse.transform not installed")
def test_valid_failure_infos():
    from pydiverse.transform.extended import collect

    cars = {"vin": ["123", "456"], "manufacturer": ["BMW", "Mercedes"]}
    car_parts: dict[str, list[Any]] = {
        "vin": ["123", "123", "456"],
        "part": ["Motor", "Wheel", "Motor"],
        "price": [1000, 100, 1000],
    }
    raw_fleet = CarFleet.build()
    raw_fleet.cars = pdt.Table(cars)
    raw_fleet.car_parts = pdt.Table(car_parts)

    car_fleet, failures = raw_fleet.filter(cast=True)

    assert len(car_fleet.cars >> collect()) + len(failures.cars >> collect()) == len(
        cars["vin"]
    )
    assert len(car_fleet.car_parts >> collect()) + len(
        failures.car_parts.invalid_rows
    ) == len(car_parts["vin"])
    assert len(failures.cars.invalid_rows) == 1
    assert len(failures.car_parts.invalid_rows) == 2
