"""Tests for utils.py — formatting and validation helpers."""

from __future__ import annotations

import pytest

from utils import (
    ensure_reg_types_not_empty,
    format_dutch_currency,
    get_default_reg_types,
    validate_fr,
    validate_gemeente,
    validate_household,
    validate_income,
    validate_refper,
)


# =============================================================================
# format_dutch_currency
# =============================================================================

class TestFormatDutchCurrency:
    def test_zero(self) -> None:
        assert format_dutch_currency(0) == "€ 0"

    def test_whole_number(self) -> None:
        assert format_dutch_currency(1234) == "€ 1.234"

    def test_thousands(self) -> None:
        assert format_dutch_currency(1_000_000) == "€ 1.000.000"

    def test_with_decimals(self) -> None:
        assert format_dutch_currency(1234.5, decimals=2) == "€ 1.234,50"

    def test_negative(self) -> None:
        assert format_dutch_currency(-500) == "€ -500"

    def test_small_fraction(self) -> None:
        assert format_dutch_currency(0.99, decimals=2) == "€ 0,99"


# =============================================================================
# validate_income
# =============================================================================

class TestValidateIncome:
    def test_valid(self) -> None:
        assert validate_income(150) == 150

    def test_string(self) -> None:
        assert validate_income("120") == 120

    def test_below_range(self) -> None:
        assert validate_income(50) == 100

    def test_above_range(self) -> None:
        assert validate_income(250) == 100

    def test_none(self) -> None:
        assert validate_income(None) == 100

    def test_garbage(self) -> None:
        assert validate_income("abc") == 100

    def test_boundary_100(self) -> None:
        assert validate_income(100) == 100

    def test_boundary_200(self) -> None:
        assert validate_income(200) == 200


# =============================================================================
# validate_household
# =============================================================================

class TestValidateHousehold:
    def test_valid(self) -> None:
        assert validate_household("HH03") == "HH03"

    def test_invalid(self) -> None:
        assert validate_household("HH99") == "HH01"

    def test_none(self) -> None:
        assert validate_household(None) == "HH01"


# =============================================================================
# validate_refper
# =============================================================================

class TestValidateRefper:
    @pytest.mark.parametrize("val", [0, 1, 3, 5])
    def test_valid_values(self, val: int) -> None:
        assert validate_refper(val) == val

    def test_invalid_value(self) -> None:
        assert validate_refper(2) == 0

    def test_string(self) -> None:
        assert validate_refper("3") == 3

    def test_none(self) -> None:
        assert validate_refper(None) == 0


# =============================================================================
# validate_fr
# =============================================================================

class TestValidateFr:
    @pytest.mark.parametrize("val", [1, 2, 3])
    def test_valid(self, val: int) -> None:
        assert validate_fr(val) == val

    def test_invalid(self) -> None:
        assert validate_fr(4) == 3

    def test_zero(self) -> None:
        assert validate_fr(0) == 3

    def test_none(self) -> None:
        assert validate_fr(None) == 3


# =============================================================================
# validate_gemeente
# =============================================================================

class TestValidateGemeente:
    def test_valid(self) -> None:
        gm = {"GM01": "A", "GM02": "B"}
        assert validate_gemeente("GM02", gm) == "GM02"

    def test_invalid(self) -> None:
        gm = {"GM01": "A", "GM02": "B"}
        assert validate_gemeente("GM99", gm) == "GM01"

    def test_none(self) -> None:
        gm = {"GM01": "A"}
        assert validate_gemeente(None, gm) == "GM01"


# =============================================================================
# get_default_reg_types
# =============================================================================

class TestGetDefaultRegTypes:
    def test_formal(self) -> None:
        assert get_default_reg_types(1) == ["Formeel"]

    def test_informal(self) -> None:
        assert get_default_reg_types(2) == ["Informeel"]

    def test_both(self) -> None:
        assert get_default_reg_types(3) == ["Formeel", "Informeel"]


# =============================================================================
# ensure_reg_types_not_empty
# =============================================================================

class TestEnsureRegTypesNotEmpty:
    def test_non_empty_passthrough(self) -> None:
        assert ensure_reg_types_not_empty(["Formeel"], []) == ["Formeel"]

    def test_empty_toggles_from_formeel(self) -> None:
        assert ensure_reg_types_not_empty([], ["Formeel"]) == ["Informeel"]

    def test_empty_toggles_from_informeel(self) -> None:
        assert ensure_reg_types_not_empty([], ["Informeel"]) == ["Formeel"]

    def test_empty_with_both_previous(self) -> None:
        assert ensure_reg_types_not_empty([], ["Formeel", "Informeel"]) == ["Formeel", "Informeel"]

    def test_empty_with_empty_previous(self) -> None:
        assert ensure_reg_types_not_empty([], []) == ["Formeel", "Informeel"]
