# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path
from typing import IO, Self

from pydiverse.colspec.pdt_util import num_rows

from .config import Config, alias_subquery
from .optional_dependency import ColExpr, pdt, pl


class FailureInfo:
    """A container carrying information about rows failing validation in
    :meth:`ColSpec.filter`."""

    def __init__(
        self,
        tbl: pdt.Table,
        invalid_rows: pdt.Table,
        rule_columns: dict[str, ColExpr],
        cfg: Config,
    ):
        #: The subset of the input data frame containing the *invalid* rows along with
        # all boolean columns used for validation. Each of these boolean columns
        # describes a single rule where a value of ``False``` indicates unsuccessful
        # validation. Thus, at least one value per row is ``False``.
        self.tbl = tbl
        self._invalid_rows = invalid_rows >> alias_subquery(cfg=cfg)
        #: The columns in `_tbl` which are used for validation.
        self.rule_columns = rule_columns
        self.cfg = cfg

    @property
    def invalid_rows(self) -> pdt.Table:
        from pydiverse.transform.extended import select

        return self._invalid_rows >> select(*[c.name for c in self.tbl])

    @property
    def debug_invalid_rows(self) -> pdt.Table:
        return self._invalid_rows

    def counts(self) -> dict[str, int]:
        """The number of validation failures for each individual rule.

        Returns:
            A mapping from rule name to counts. If a rule's failure count is 0, it is
            not included here.
        """
        if not self.rule_columns:  # pdt summarize needs at least one column
            return {}
        from pydiverse.transform.extended import C, export, summarize

        # There can be nulls, which should not be counted.
        if self.cfg.dialect_name == "mssql":
            # MSSQL has no boolean columns. So the failure indicator columns
            # have been converted to integers.
            cnts: dict[str, int] = (
                self._invalid_rows
                >> summarize(
                    **{
                        k: (C[k] == 0).sum().fill_null(0)
                        for k in self.rule_columns.keys()
                    }
                )
                >> export(pdt.Dict())
            )
        else:
            cnts: dict[str, int] = (
                self._invalid_rows
                >> summarize(**{k: (~C[k]).sum() for k in self.rule_columns.keys()})
                >> export(pdt.Dict())
            )

        return {k: v for k, v in cnts.items() if v is not None and v > 0}

    def __len__(self) -> int:
        return num_rows(self._invalid_rows)

    # ---------------------------------- PERSISTENCE --------------------------------- #

    def write_parquet(self, file: str | Path | IO[bytes]):
        """Write the failure info to a Parquet file.

        Args:
            file: The file path or writable file-like object to write to.
        """
        # NOTE: We add a dummy column with metadata in the column name to allow writing
        #  the rule columns to the same file.
        rule_columns_json = json.dumps({"rule_columns": self.rule_columns})
        (self._invalid_rows >> pdt.export(pdt.Polars())).with_columns(
            pl.lit(None).alias(rule_columns_json),
        ).write_parquet(file)

    @classmethod
    def scan_parquet(cls, source: str | Path | IO[bytes]) -> Self:
        """Lazily read the parquet file with the failure info.

        Args:
            source: The file path or readable file-like object to read from.

        Returns:
            The failure info object.
        """
        lf = pl.scan_parquet(source)
        # NOTE: In `write_parquet`, the rule columns are added as the name of the last
        #  column.
        last_column = lf.collect_schema().names()[-1]
        rule_columns = json.loads(last_column)["rule_columns"]
        return cls(
            pdt.Table(pl.DataFrame()),
            pdt.Table(lf.drop(last_column)),
            rule_columns,
            Config.default,
        )


# ------------------------------------ COMPUTATION ----------------------------------- #


def _compute_counts(df: pl.DataFrame, rule_columns: list[str]) -> dict[str, int]:
    if len(rule_columns) == 0:
        return {}

    counts = df.select((~pl.col(rule_columns)).sum())
    return {
        name: count for name, count in (counts.row(0, named=True).items()) if count > 0
    }


def _compute_cooccurrence_counts(
    df: pl.DataFrame, rule_columns: list[str]
) -> dict[frozenset[str], int]:
    if len(rule_columns) == 0:
        return {}

    group_lengths = df.group_by(pl.col(rule_columns).fill_null(True)).len()
    if len(group_lengths) == 0:
        return {}

    groups = group_lengths.drop("len")
    counts = group_lengths.get_column("len")
    return {
        frozenset(
            name
            for name, success in zip(rule_columns, row, strict=False)
            if not success
        ): count
        for row, count in zip(groups.iter_rows(), counts, strict=False)
    }
