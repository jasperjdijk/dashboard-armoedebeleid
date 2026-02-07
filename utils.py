"""
Utility functions for formatting and validation.

Pure Python module with no Streamlit dependency — safe to use in tests.
"""

from __future__ import annotations

from constants import (
    FR_BOTH,
    HOUSEHOLD_LABELS,
    INCOME_MAX_PCT,
    INCOME_MIN_PCT,
    VALID_REFPER_VALUES,
)


# =============================================================================
# Formatting
# =============================================================================

def format_dutch_currency(value: float, decimals: int = 0) -> str:
    """Format a number as Dutch currency string.

    Uses dots for thousands separators and commas for decimals,
    e.g. ``1234.5`` → ``"€ 1.234,50"`` (with *decimals=2*).
    """
    formatted = (
        f"{value:,.{decimals}f}"
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.')
    )
    return f"€ {formatted}"


# =============================================================================
# Query-parameter / input validation
# =============================================================================

def validate_income(value: str | int | None, default: int = INCOME_MIN_PCT) -> int:
    """Return a valid income percentage in [100, 200], or *default*."""
    try:
        val = int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default
    if val < INCOME_MIN_PCT or val > INCOME_MAX_PCT:
        return default
    return val


def validate_household(value: str | None, default: str = 'HH01') -> str:
    """Return *value* if it is a known household code, else *default*."""
    if value in HOUSEHOLD_LABELS:
        return value  # type: ignore[return-value]
    return default


def validate_refper(value: str | int | None, default: int = 0) -> int:
    """Return *value* if it is a valid reference-period value, else *default*."""
    try:
        val = int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default
    if val not in VALID_REFPER_VALUES:
        return default
    return val


def validate_fr(value: str | int | None, default: int = FR_BOTH) -> int:
    """Return *value* if it is a valid formal/informal filter code (1-3), else *default*."""
    try:
        val = int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default
    if val not in (1, 2, 3):
        return default
    return val


def validate_gemeente(value: str | None, gm_lbl: dict[str, str]) -> str:
    """Return *value* if it exists in *gm_lbl*, else the first available municipality."""
    if value and value in gm_lbl:
        return value
    return next(iter(gm_lbl))


# =============================================================================
# Regulation-type helpers
# =============================================================================

def get_default_reg_types(fr_value: int) -> list[str]:
    """Convert a FR filter integer to a regulation-type label list."""
    if fr_value == 1:
        return ["Formeel"]
    if fr_value == 2:
        return ["Informeel"]
    return ["Formeel", "Informeel"]


def ensure_reg_types_not_empty(
    current: list[str],
    previous: list[str],
) -> list[str]:
    """Guarantee that the regulation-types list is never empty.

    When the user deselects the last option, toggle to the opposite type
    (or fall back to both).
    """
    if current:
        return current
    if len(previous) == 1:
        return ["Informeel"] if "Formeel" in previous else ["Formeel"]
    return ["Formeel", "Informeel"]
