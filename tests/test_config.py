# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import ColSpec
from pydiverse.colspec.optional_dependency import dy


def perform_dataframely_operation():
    class C(ColSpec):
        a = cs.Integer(min=0)
        b = cs.String

    df = C.sample_polars(10)
    C.validate_polars(df)


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_config_global():
    try:
        dy.Config.set_max_sampling_iterations(50)
        assert dy.Config.options["max_sampling_iterations"] == 50
        perform_dataframely_operation()
    finally:
        dy.Config.restore_defaults()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_config_local():
    try:
        with dy.Config(max_sampling_iterations=35):
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 10_000
        perform_dataframely_operation()
    finally:
        dy.Config.restore_defaults()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_config_local_nested():
    try:
        with dy.Config(max_sampling_iterations=35):
            assert dy.Config.options["max_sampling_iterations"] == 35
            with dy.Config(max_sampling_iterations=20):
                perform_dataframely_operation()
                assert dy.Config.options["max_sampling_iterations"] == 20
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 10_000
    finally:
        dy.Config.restore_defaults()


@pytest.mark.skipif(dy.Column is None, reason="dataframely is required for this test")
def test_config_global_local():
    try:
        dy.Config.set_max_sampling_iterations(50)
        assert dy.Config.options["max_sampling_iterations"] == 50
        with dy.Config(max_sampling_iterations=35):
            perform_dataframely_operation()
            assert dy.Config.options["max_sampling_iterations"] == 35
        assert dy.Config.options["max_sampling_iterations"] == 50
    finally:
        dy.Config.restore_defaults()
