# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.colspec import RulePolars
from pydiverse.colspec.optional_dependency import ColExpr, pdt, pl


def rules_from_exprs_polars(exprs: dict[str, pl.Expr]) -> dict[str, RulePolars]:
    """Turn a set of expressions into simple rules.

    Args:
        exprs: The expressions, mapping from names to :class:`polars.Expr`.

    Returns:
        The rules corresponding to the expressions.
    """
    return {name: RulePolars(expr) for name, expr in exprs.items()}


def evaluate_rules_polars(
    lf: pl.LazyFrame, rules: dict[str, RulePolars]
) -> pl.LazyFrame:
    """Evaluate the provided rules and return the rules' evaluation.

    Args:
        lf: The data frame on which to evaluate the rules.
        rules: The rules to evaluate where the key of the dictionary provides the name
            of the rule.

    Returns:
        The same return value as :meth:`with_evaluation_rules` only that the columns
        of the input data frame are dropped.
    """
    return RulePolars.append_rules_polars(lf, rules).drop(list(lf.collect_schema()))


def evaluate_rules(tbl: pdt.Table, rules: dict[str, ColExpr]):
    return {
        k: (tbl >> pdt.select() >> pdt.mutate(out=v) >> pdt.export(pdt.DictOfLists()))[
            "out"
        ]
        for k, v in rules.items()
    }
