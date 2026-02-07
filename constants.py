"""
Constants for the dashboard application.

Centralizes magic numbers, filter codes, and configuration values
used throughout the dashboard modules.
"""

from __future__ import annotations

# =============================================================================
# Filter codes for formal/informal regulation types
# =============================================================================
FR_FORMAL_ONLY: int = 1
FR_INFORMAL_ONLY: int = 2
FR_BOTH: int = 3

# =============================================================================
# CAV (health insurance discount) filter codes
# =============================================================================
CAV_EXCLUDE: int = 0
CAV_INCLUDE: int = 1

# =============================================================================
# Income range (percentage of social minimum)
# =============================================================================
INCOME_MIN_PCT: int = 100
INCOME_MAX_PCT: int = 200

# =============================================================================
# Reference period valid values (years at low income)
# =============================================================================
VALID_REFPER_VALUES: list[int] = [0, 1, 3, 5]

# =============================================================================
# Household type mapping (code -> Dutch label)
# =============================================================================
HOUSEHOLD_LABELS: dict[str, str] = {
    'HH01': 'Alleenstaande',
    'HH02': 'Alleenstaande ouder met kind',
    'HH03': 'Paar',
    'HH04': 'Paar met twee kinderen',
}

# =============================================================================
# Annual-to-monthly conversion
# =============================================================================
MONTHS_PER_YEAR: int = 12

# =============================================================================
# Graph 2: Income levels display configuration
# =============================================================================
INCOME_LEVELS_COUNT: int = 6       # Number of income levels to show as markers
INCOME_LEVEL_STEP: int = 10        # Step between income levels (percentage points)
INCOME_LEVEL_OFFSET: int = 20      # Offset below selected income to start

# =============================================================================
# Graph 4: Bubble size scaling
# =============================================================================
BUBBLE_SIZE_DIVISOR: int = 200     # Divides population to get marker area

# =============================================================================
# Regulation type labels and values (for sidebar segmented control)
# =============================================================================
REG_TYPE_VALUES: dict[str, int] = {"Formeel": 1, "Informeel": 2}

# =============================================================================
# Required columns in the Parquet data file
# =============================================================================
REQUIRED_COLUMNS: list[str] = [
    'GMcode', 'Gemeentenaam', 'Inwoners', 'N4', 'ID',
    'WB', 'BT', 'CAV', 'FR',
    'WRD_HH01', 'WRD_HH02', 'WRD_HH03', 'WRD_HH04',
    'IG_HH01', 'IG_HH02', 'IG_HH03', 'IG_HH04',
    'Referteperiode_HH01', 'Referteperiode_HH02',
    'Referteperiode_HH03', 'Referteperiode_HH04',
]
