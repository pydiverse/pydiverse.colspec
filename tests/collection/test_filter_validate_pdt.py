# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import functools
import operator
from dataclasses import dataclass

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.exc import MemberValidationError
from pydiverse.colspec.optional_dependency import C, ColExpr, pdt
from pydiverse.colspec.pdt_util import num_rows
from pydiverse.colspec.testing.assert_equal import assert_table_equal

# ------------------------------------------------------------------------------------ #
#                                        SCHEMA                                        #
# ------------------------------------------------------------------------------------ #


class MyFirstColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer()


class MySecondColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer(min=1)


@dataclass
class MyCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec

    @cs.filter()
    def equal_primary_keys(self) -> ColExpr:
        return functools.reduce(
            operator.and_,
            (self.first[key] == self.second[key] for key in self.common_primary_keys()),
        )

    @cs.filter()
    def first_b_greater_second_b(self) -> ColExpr:
        return (
            (self.first.b > self.second.b)
            | self.pk_is_null(self.first)
            | self.pk_is_null(self.second)
        )


@dataclass
class SimpleCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec


def to_my_collection(c: SimpleCollection) -> MyCollection:
    return MyCollection(c.first, c.second)


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.fixture()
def data_without_filter_without_rule_violation() -> SimpleCollection:
    c = SimpleCollection.build()
    c.first = pdt.Table({"a": [1, 2, 3], "b": [1, 2, 3]})
    c.second = pdt.Table({"a": [1, 2, 3], "b": [1, 2, 3]})
    return c


@pytest.fixture()
def data_without_filter_with_rule_violation() -> SimpleCollection:
    c = SimpleCollection.build()
    c.first = pdt.Table({"a": [1, 2, 1], "b": [1, 2, 3]})
    c.second = pdt.Table({"a": [1, 2, 3], "b": [0, 1, 2]})
    return c


@pytest.fixture()
def data_with_filter_without_rule_violation() -> SimpleCollection:
    c = SimpleCollection.build()
    c.first = pdt.Table({"a": [1, 2, 3], "b": [1, 1, 3]})
    c.second = pdt.Table({"a": [2, 3, 4, 5], "b": [1, 2, 3, 4]})
    return c


@pytest.fixture()
def data_with_filter_with_rule_violation() -> SimpleCollection:
    c = SimpleCollection.build()
    c.first = pdt.Table({"a": [1, 2, 3], "b": [1, 2, 3]})
    c.second = pdt.Table({"a": [2, 3, 4, 5], "b": [0, 1, 2, 3]})
    return c


# -------------------------------------- FILTER -------------------------------------- #


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_without_filter_without_rule_violation(
    data_without_filter_without_rule_violation: SimpleCollection,
):
    out, failure = data_without_filter_without_rule_violation.filter()

    assert isinstance(out, SimpleCollection)
    assert_table_equal(out.first, data_without_filter_without_rule_violation.first)
    assert_table_equal(out.second, data_without_filter_without_rule_violation.second)
    assert len(failure.first) == 0
    assert len(failure.second) == 0


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_without_filter_with_rule_violation(
    data_without_filter_with_rule_violation: SimpleCollection,
):
    out, failure = data_without_filter_with_rule_violation.filter()

    assert isinstance(out, SimpleCollection)
    assert num_rows(out.first) == 1
    assert num_rows(out.second) == 2
    assert failure.first.counts() == {"_primary_key_": 2}
    assert failure.second.counts() == {"b|min": 1}


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_with_filter_without_rule_violation(
    data_with_filter_without_rule_violation: SimpleCollection,
):
    my_collection = to_my_collection(data_with_filter_without_rule_violation)
    out, failure = my_collection.filter()

    assert isinstance(out, MyCollection)
    assert_table_equal(out.first, pdt.Table({"a": [3], "b": [3]}))
    assert_table_equal(out.second, pdt.Table({"a": [3], "b": [2]}))
    assert failure.first.counts() == {
        "equal_primary_keys": 1,
        "first_b_greater_second_b": 1,
    }
    assert failure.second.counts() == {
        "equal_primary_keys": 2,
        "first_b_greater_second_b": 1,
    }


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_filter_with_filter_with_rule_violation(
    data_with_filter_with_rule_violation: SimpleCollection,
):
    my_collection = to_my_collection(data_with_filter_with_rule_violation)
    out, failure = my_collection.filter()

    assert isinstance(out, MyCollection)
    assert_table_equal(out.first, pdt.Table({"a": [3], "b": [3]}))
    assert_table_equal(out.second, pdt.Table({"a": [3], "b": [1]}))
    assert failure.first.counts() == {"equal_primary_keys": 2}
    assert failure.second.counts() == {"b|min": 1, "equal_primary_keys": 2}


# -------------------------------- VALIDATE WITH DATA -------------------------------- #


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_validate_without_filter_without_rule_violation(
    data_without_filter_without_rule_violation: SimpleCollection,
):
    assert data_without_filter_without_rule_violation.is_valid()
    out = data_without_filter_without_rule_violation.validate()

    assert isinstance(out, SimpleCollection)
    assert_table_equal(out.first, data_without_filter_without_rule_violation.first)
    assert_table_equal(out.second, data_without_filter_without_rule_violation.second)


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_validate_without_filter_with_rule_violation(
    data_without_filter_with_rule_violation: SimpleCollection,
):
    assert not data_without_filter_with_rule_violation.is_valid()

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        data_without_filter_with_rule_violation.validate()

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'_primary_key_' failed validation for 2 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"Column 'b' failed validation for 1 rules:")
    exc.match(r"'min' failed for 1 rows")


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_validate_with_filter_without_rule_violation(
    data_with_filter_without_rule_violation: SimpleCollection,
):
    my_collection = to_my_collection(data_with_filter_without_rule_violation)

    assert not my_collection.is_valid()

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        my_collection.validate()

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 1 rows")
    exc.match(r"'first_b_greater_second_b' failed validation for 1 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 2 rows")


@pytest.mark.skipif(C is None, reason="pydiverse.transform not installed")
def test_validate_with_filter_with_rule_violation(
    data_with_filter_with_rule_violation: SimpleCollection,
):
    my_collection = to_my_collection(data_with_filter_with_rule_violation)
    assert not my_collection.is_valid()

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        my_collection.validate()

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 2 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"'min' failed for 1 rows")
