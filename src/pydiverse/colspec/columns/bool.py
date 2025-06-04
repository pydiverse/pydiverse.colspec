# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pydiverse.common as pdc

from ._base import Column

# ------------------------------------------------------------------------------------ #


class Bool(Column):
    """A column of booleans."""

    def dtype(self) -> pdc.Bool:
        return pdc.Bool()
