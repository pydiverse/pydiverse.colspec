# Copyright (c) QuantCo and pydiverse contributors 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.colspec as cs
from pydiverse.colspec import Collection, ColSpec
from pydiverse.colspec._filter import Filter
from pydiverse.colspec.exc import AnnotationImplementationError, ImplementationError
from pydiverse.colspec.optional_dependency import pl
from pydiverse.colspec.testing.factory import (
    create_collection,
    create_collection_raw,
    create_colspec,
)


def test_annotation_union_success():
    """When we use a union annotation, it must contain one typed LazyFrame and None."""
    res = create_collection_raw(
        "test",
        {
            "first": (
                create_colspec("first", {"a": cs.Integer(primary_key=True)}) | None
            ),
        },
    ).members()
    assert len(res) == 1
    assert res["first"].is_optional
    assert res["first"].col_spec.primary_keys() == ["a"]
    assert issubclass(res["first"].col_spec, ColSpec)


def test_annotation_union_too_many_arg_failure():
    """Unions should have a maximum of two types in them."""

    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": create_colspec("first", {"a": cs.Integer(primary_key=True)})
                | create_colspec("second", {"a": cs.Integer(primary_key=True)})
                | None,
            },
        ).members()


def test_annotation_union_conflicting_types_failure():
    """Unions should contain a maximum of one non-None type."""

    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": create_colspec("first", {"a": cs.Integer(primary_key=True)})
                | create_colspec("second", {"a": cs.Integer(primary_key=True)}),
            },
        ).members()


def test_annotation_only_none_failure():
    """Annotations must not just be None."""
    res = create_collection_raw(
        "test",
        {"first": None, "second": int, "third": Collection},
    ).members()
    assert len(res) == 0


def test_annotation_invalid_type_failure():
    """Annotations must not just be None."""
    res = create_collection_raw(
        "test",
        {
            "first": int | None,
        },
    ).members()
    assert len(res) == 0


def test_name_overlap():
    with pytest.raises(
        ImplementationError,
        match=r"Collection cannot have a filter named '_primary_key_'",
    ):
        coll = create_collection(
            "test",
            {
                "first": create_colspec("first", {"a": cs.Integer(primary_key=True)}),
                "second": create_colspec("second", {"a": cs.Integer(primary_key=True)}),
            },
            filters={"_primary_key_": Filter(lambda c: c.first)},
        )()
        coll.filter_rules()


def test_collection_no_primary_key_success():
    """It's ok not to have primary keys if there are no filters."""
    res = create_collection(
        "test",
        {
            "first": create_colspec("first", {"a": cs.Integer()}),
        },
    ).members()
    assert len(res) == 1
    assert not res["first"].is_optional
    assert issubclass(res["first"].col_spec, ColSpec)


def test_collection_no_common_primary_key():
    res = create_collection(
        "test",
        {
            "first": create_colspec("first", {"a": cs.Integer()}),
        },
        filters={"testfilter": Filter(lambda c: c.first.filter(pl.col("a") > 0))},
    ).members()
    assert len(res) == 1
    assert not res["first"].is_optional
    assert res["first"].col_spec.primary_keys() == []


def test_collection_primary_key_but_not_common():
    res = create_collection(
        "test",
        {
            "first": create_colspec("first", {"a": cs.Integer(primary_key=True)}),
            "second": create_colspec("second", {"b": cs.Integer(primary_key=True)}),
        },
        filters={"testfilter": Filter(lambda c: c.first.filter(pl.col("a") > 0))},
    ).members()
    assert len(res) == 2
    assert not res["first"].is_optional
    assert res["first"].col_spec.primary_keys() == ["a"]
    assert not res["second"].is_optional
    assert res["second"].col_spec.primary_keys() == ["b"]
