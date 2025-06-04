# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from polars.testing import assert_frame_equal

import pydiverse.transform as pdt
from pydiverse.transform.common import *


def assert_table_equal(t1: pdt.Table, t2: pdt.Table):
    assert_frame_equal(t1 >> export(Polars()), t2 >> export(Polars()))
