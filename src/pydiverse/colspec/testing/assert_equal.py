# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.colspec.optional_dependency import assert_frame_equal, pdt


def assert_table_equal(t1: pdt.Table, t2: pdt.Table, check_row_order=True):
    assert_frame_equal(
        t1 >> pdt.export(pdt.Polars()),
        t2 >> pdt.export(pdt.Polars()),
        check_row_order=check_row_order,
    )
