# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Any

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec.optional_dependency import Generator, dy
from pydiverse.colspec.testing.factory import create_collection_raw

pytestmark = pytest.mark.skipif(
    dy.Column is None, reason="dataframely is required for this test"
)


class MyFirstSchema(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer()


class MySecondSchema(cs.ColSpec):
    a = cs.Integer(primary_key=True)
    b = cs.Integer(primary_key=True)
    c = cs.Integer()


class NonPkSchema(cs.ColSpec):
    x = cs.Integer()
    y = cs.Integer()


class MyCollection(cs.Collection):
    first: MyFirstSchema
    second: MySecondSchema | None


class MyInlinedCollection(cs.Collection):
    first: Annotated[
        MyFirstSchema,
        cs.CollectionMember(inline_for_sampling=True),
    ]
    second: MySecondSchema


class MyInlinedCollectionWithOptional(cs.Collection):
    first: Annotated[
        MyFirstSchema | None,
        cs.CollectionMember(inline_for_sampling=True),
    ]
    second: MySecondSchema


class SmallCollection(cs.Collection):
    first: MyFirstSchema


class IgnoringCollection(cs.Collection):
    first: MyFirstSchema
    second: Annotated[MySecondSchema, cs.CollectionMember(ignored_in_filters=True)]


class IncorrectOverrideCollection(cs.Collection):
    first: MyFirstSchema
    second: MySecondSchema | None

    @classmethod
    def _preprocess_sample(
        cls, sample: dict[str, Any], index: int, generator: Generator
    ) -> dict[str, Any]:
        return sample


# ------------------------------------------------------------------------------------ #
#                                         TESTS                                        #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_rows(n: int) -> None:
    collection = MyCollection.sample(n)
    assert collection.first.collect().height == n
    assert collection.second is not None
    assert collection.second.collect().is_empty()


def test_sample_with_overrides() -> None:
    collection = MyCollection.sample(
        overrides=[
            {"first": {"b": 4}, "second": [{"c": 3}, {"c": 4}]},
            {"first": {"b": 8}, "second": [{"c": 6}]},
        ]
    )

    first_a = collection.first.collect()["a"].to_list()
    assert len(first_a) == 2
    assert collection.first.collect()["b"].to_list() == [4, 8]

    assert collection.second is not None
    assert collection.second.collect()["a"].to_list() == [first_a[0]] * 2 + [first_a[1]]
    assert collection.second.collect()["c"].to_list() == [3, 4, 6]


def test_sample_with_primary_key_override() -> None:
    collection = MyCollection.sample(
        overrides=[
            {"a": 1, "second": [{"c": 3}, {"c": 4}]},
            {"a": 2, "second": [{"c": 6}]},
        ]
    )

    assert collection.first.collect()["a"].to_list() == [1, 2]

    assert collection.second is not None
    assert collection.second.collect()["a"].to_list() == [1, 1, 2]


@pytest.mark.parametrize(
    "collection_type", [MyInlinedCollection, MyInlinedCollectionWithOptional]
)
def test_sample_inline_with_overrides(
    collection_type: type[MyInlinedCollection] | type[MyInlinedCollectionWithOptional],
) -> None:
    collection = collection_type.sample(
        overrides=[
            {"b": 4, "second": [{"c": 3}, {"c": 4}]},
            {"b": 8, "second": [{"c": 6}]},
        ]
    )

    assert collection.first is not None
    first_a = collection.first.collect()["a"].to_list()
    assert len(first_a) == 2
    assert collection.first.collect()["b"].to_list() == [4, 8]

    assert collection.second is not None
    assert collection.second.collect()["a"].to_list() == [first_a[0]] * 2 + [first_a[1]]
    assert collection.second.collect()["b"].to_list() != [4, 4, 8]
    assert collection.second.collect()["c"].to_list() == [3, 4, 6]


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_without_dependent_members(n: int) -> None:
    collection = SmallCollection.sample(n)
    assert collection.first.collect().height == n


@pytest.mark.parametrize("n", [0, 1000])
def test_sample_with_ignored_members(n: int) -> None:
    collection = IgnoringCollection.sample(n)
    assert collection.first.collect().height == n


def test_sample_num_rows_mismatch() -> None:
    with pytest.raises(ValueError, match=r"`num_rows` mismatches"):
        MyCollection.sample(num_rows=1, overrides=[])


def test_sample_incorrect_override() -> None:
    with pytest.raises(ValueError, match=r"All samples must contain"):
        IncorrectOverrideCollection.sample()


def test_invalid_inline_for_sampling() -> None:
    import dataframely as dy

    with pytest.raises(
        dy.exc.ImplementationError, match=r"its primary key is a superset"
    ):
        create_collection_raw(
            "test",
            {
                "first": MyFirstSchema,
                "second": Annotated[
                    MySecondSchema,
                    cs.CollectionMember(inline_for_sampling=True),
                ],
            },
        ).sample()


def test_duplicate_column_inlined_for_sampling() -> None:
    import dataframely as dy

    with pytest.raises(dy.exc.ImplementationError, match=r"clashes with a column name"):
        create_collection_raw(
            "test",
            {
                "first": Annotated[
                    MyFirstSchema,
                    cs.CollectionMember(inline_for_sampling=True),
                ],
                "second": Annotated[
                    MyFirstSchema,
                    cs.CollectionMember(inline_for_sampling=True),
                ],
            },
        ).sample()
