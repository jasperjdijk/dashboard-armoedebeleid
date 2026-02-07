"""Tests for constants.py — sanity checks on configuration values."""

from __future__ import annotations

from constants import (
    HOUSEHOLD_LABELS,
    INCOME_LEVEL_STEP,
    INCOME_LEVELS_COUNT,
    INCOME_MAX_PCT,
    INCOME_MIN_PCT,
    REQUIRED_COLUMNS,
    VALID_REFPER_VALUES,
)


def test_income_range_valid() -> None:
    assert INCOME_MIN_PCT < INCOME_MAX_PCT


def test_income_levels_fit_in_range() -> None:
    span = INCOME_LEVELS_COUNT * INCOME_LEVEL_STEP
    assert span <= (INCOME_MAX_PCT - INCOME_MIN_PCT)


def test_household_labels_has_four_types() -> None:
    assert len(HOUSEHOLD_LABELS) == 4
    assert all(k.startswith('HH') for k in HOUSEHOLD_LABELS)


def test_refper_values_sorted() -> None:
    assert VALID_REFPER_VALUES == sorted(VALID_REFPER_VALUES)


def test_required_columns_not_empty() -> None:
    assert len(REQUIRED_COLUMNS) > 0
