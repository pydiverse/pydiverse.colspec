# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import pytest

import pydiverse.colspec as cs
import pydiverse.colspec.collection
from pydiverse.colspec.exc import MemberValidationError
from pydiverse.colspec.optional_dependency import assert_frame_equal, dy, pl

pytestmark = pytest.mark.skipif(
    dy.Column is None, reason="dataframely is required for this test"
)

# ------------------------------------------------------------------------------------ #
#                                        SCHEMA                                        #
# ------------------------------------------------------------------------------------ #


class MyFirstColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer()


class MySecondColSpec(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer(min=1)


class MyCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec

    @cs.filter_polars()
    def equal_primary_keys(self) -> pl.LazyFrame:
        return self.first.join(self.second, on=self.common_primary_keys())

    @cs.filter_polars()
    def first_b_greater_second_b(self) -> pl.LazyFrame:
        return self.first.join(
            self.second, on=self.common_primary_keys(), how="full", coalesce=True
        ).filter((pl.col("b") > pl.col("b_right")).fill_null(True))


@dataclasses.dataclass
class SimpleCollection(pydiverse.colspec.collection.Collection):
    first: MyFirstColSpec
    second: MySecondColSpec


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.fixture()
def data_without_filter_without_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    return first, second


@pytest.fixture()
def data_without_filter_with_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 1], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    return first, second


@pytest.fixture()
def data_with_filter_without_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 1, 3]})
    second = pl.LazyFrame({"a": [2, 3, 4, 5], "b": [1, 2, 3, 4]})
    return first, second


@pytest.fixture()
def data_with_filter_with_rule_violation() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    first = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    second = pl.LazyFrame({"a": [2, 3, 4, 5], "b": [0, 1, 2, 3]})
    return first, second


# -------------------------------------- FILTER -------------------------------------- #


def test_filter_without_filter_without_rule_violation(
    data_without_filter_without_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = SimpleCollection(
        data_without_filter_without_rule_violation[0],
        data_without_filter_without_rule_violation[1],
    ).filter_polars()

    assert isinstance(out, SimpleCollection)
    assert_frame_equal(out.first, data_without_filter_without_rule_violation[0])
    assert_frame_equal(out.second, data_without_filter_without_rule_violation[1])
    assert len(failure["first"]) == 0
    assert len(failure["second"]) == 0


def test_filter_without_filter_with_rule_violation(
    data_without_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = SimpleCollection.filter_polars_data(
        {
            "first": data_without_filter_with_rule_violation[0],
            "second": data_without_filter_with_rule_violation[1],
        }
    )

    assert isinstance(out, SimpleCollection)
    assert len(out.first.collect()) == 1
    assert len(out.second.collect()) == 2
    assert failure["first"].counts() == {"primary_key": 2}
    assert failure["second"].counts() == {"b|min": 1}


def test_filter_with_filter_without_rule_violation(
    data_with_filter_without_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = MyCollection.filter_polars_data(
        {
            "first": data_with_filter_without_rule_violation[0],
            "second": data_with_filter_without_rule_violation[1],
        }
    )

    assert isinstance(out, MyCollection)
    assert_frame_equal(out.first, pl.LazyFrame({"a": [3], "b": [3]}))
    assert_frame_equal(out.second, pl.LazyFrame({"a": [3], "b": [2]}))
    assert failure["first"].counts() == {
        "equal_primary_keys": 1,
        "first_b_greater_second_b": 1,
    }
    assert failure["second"].counts() == {
        "equal_primary_keys": 2,
        "first_b_greater_second_b": 1,
    }


def test_filter_with_filter_with_rule_violation(
    data_with_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    out, failure = MyCollection.filter_polars_data(
        {
            "first": data_with_filter_with_rule_violation[0],
            "second": data_with_filter_with_rule_violation[1],
        }
    )

    assert isinstance(out, MyCollection)
    assert_frame_equal(out.first, pl.LazyFrame({"a": [3], "b": [3]}))
    assert_frame_equal(out.second, pl.LazyFrame({"a": [3], "b": [1]}))
    assert failure["first"].counts() == {"equal_primary_keys": 2}
    assert failure["second"].counts() == {"b|min": 1, "equal_primary_keys": 2}


# -------------------------------- VALIDATE WITH DATA -------------------------------- #


def test_validate_without_filter_without_rule_violation(
    data_without_filter_without_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    data = {
        "first": data_without_filter_without_rule_violation[0],
        "second": data_without_filter_without_rule_violation[1],
    }
    assert SimpleCollection.is_valid_polars_data(data)
    out = SimpleCollection.validate_polars_data(data)

    assert isinstance(out, SimpleCollection)
    assert_frame_equal(out.first, data_without_filter_without_rule_violation[0])
    assert_frame_equal(out.second, data_without_filter_without_rule_violation[1])


def test_validate_without_filter_with_rule_violation(
    data_without_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    data = {
        "first": data_without_filter_with_rule_violation[0],
        "second": data_without_filter_with_rule_violation[1],
    }
    assert not SimpleCollection.is_valid_polars_data(data)

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        SimpleCollection.validate_polars_data(data)

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'primary_key' failed validation for 2 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"'min' failed for 1 rows")


def test_validate_with_filter_without_rule_violation(
    data_with_filter_without_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    data = {
        "first": data_with_filter_without_rule_violation[0],
        "second": data_with_filter_without_rule_violation[1],
    }
    assert not MyCollection.is_valid_polars_data(data)

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        MyCollection.validate_polars_data(data)

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 1 rows")
    exc.match(r"'first_b_greater_second_b' failed validation for 1 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 2 rows")


def test_validate_with_filter_with_rule_violation(
    data_with_filter_with_rule_violation: tuple[pl.LazyFrame, pl.LazyFrame],
):
    data = {
        "first": data_with_filter_with_rule_violation[0],
        "second": data_with_filter_with_rule_violation[1],
    }
    assert not MyCollection.is_valid_polars_data(data)

    with pytest.raises(
        MemberValidationError, match=r"2 members failed validation"
    ) as exc:
        MyCollection.validate_polars_data(data)

    exc.match(r"Member 'first' failed validation")
    exc.match(r"'equal_primary_keys' failed validation for 2 rows")
    exc.match(r"Member 'second' failed validation")
    exc.match(r"'min' failed for 1 rows")
