# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from .optional_dependency import pdt


def num_rows(tbl: pdt.Table) -> int:
    return tbl >> pdt.summarize(num_rows=pdt.count()) >> pdt.export(pdt.Scalar)
