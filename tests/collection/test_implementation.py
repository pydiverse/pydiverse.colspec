# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import pytest

import dataframely as dy
from dataframely._filter import Filter
from dataframely.exc import AnnotationImplementationError, ImplementationError
from dataframely.testing import create_collection, create_collection_raw, create_schema


def test_annotation_type_failure():
    with pytest.raises(
        AnnotationImplementationError,
    ):
        create_collection(
            "test",
            {
                "first": create_schema("first", {"a": dy.Integer()}),
            },
            annotation_base_class=dy.DataFrame,
        )


def test_annotation_union_success():
    """When we use a union annotation, it must contain one typed LazyFrame and None."""
    create_collection_raw(
        "test",
        {
            "first": dy.LazyFrame[  # type: ignore
                create_schema("first", {"a": dy.Integer(primary_key=True)})
            ]
            | None,
        },
    )


def test_annotation_union_with_data_frame():
    """When we use a union annotation, it must contain one typed LazyFrame and None."""
    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": dy.DataFrame[  # type: ignore
                    create_schema("first", {"a": dy.Integer(primary_key=True)})
                ]
                | None,
            },
        )


def test_annotation_union_too_many_arg_failure():
    """Unions should have a maximum of two types in them."""

    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": dy.LazyFrame[  # type: ignore
                    create_schema("first", {"a": dy.Integer(primary_key=True)})
                ]
                | dy.LazyFrame[  # type: ignore
                    create_schema("second", {"a": dy.Integer(primary_key=True)})
                ]
                | None,
            },
        )


def test_annotation_union_conflicting_types_failure():
    """Unions should contain a maximum of one non-None type."""

    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": dy.LazyFrame[  # type: ignore
                    create_schema("first", {"a": dy.Integer(primary_key=True)})
                ]
                | dy.LazyFrame[  # type: ignore
                    create_schema("second", {"a": dy.Integer(primary_key=True)})
                ],
            },
        )


def test_annotation_only_none_failure():
    """Annotations must not just be None."""
    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": None,
            },
        )


def test_annotation_invalid_type_failure():
    """Annotations must not just be None."""
    with pytest.raises(AnnotationImplementationError):
        create_collection_raw(
            "test",
            {
                "first": int | None,
            },
        )


def test_name_overlap():
    with pytest.raises(
        ImplementationError,
        match=r"Filters defined on the collection must not be named the same",
    ):
        create_collection(
            "test",
            {
                "first": create_schema("first", {"a": dy.Integer(primary_key=True)}),
                "second": create_schema("second", {"a": dy.Integer(primary_key=True)}),
            },
            filters={"primary_key": Filter(lambda c: c.first)},
        )


def test_collection_no_primary_key_success():
    """It's ok not to have primary keys if there are no filters."""
    create_collection(
        "test",
        {
            "first": create_schema("first", {"a": dy.Integer()}),
        },
    )


def test_collection_no_primary_key_failure():
    """If you have a filter, you must also have a primary key."""
    with pytest.raises(
        ImplementationError,
        match=r"Members of a collection must have an overlapping primary key",
    ):
        create_collection(
            "test",
            {
                "first": create_schema("first", {"a": dy.Integer()}),
            },
            filters={"testfilter": Filter(lambda c: c.first.filter(pl.col("a") > 0))},
        )


def test_collection_primary_key_but_not_common():
    """If you have a filter, you must also have a common primary key between members."""
    with pytest.raises(
        ImplementationError,
        match=r"Members of a collection must have an overlapping primary key",
    ):
        create_collection(
            "test",
            {
                "first": create_schema("first", {"a": dy.Integer(primary_key=True)}),
                "second": create_schema("second", {"b": dy.Integer(primary_key=True)}),
            },
            filters={"testfilter": Filter(lambda c: c.first.filter(pl.col("a") > 0))},
        )
