"""
Data loading, filtering, and aggregation functions.

All heavy data operations live here so they can be tested independently
of the Streamlit UI layer.
"""

from __future__ import annotations

import logging
import os
from typing import Union

import pandas as pd
import streamlit as st

from constants import (
    CAV_EXCLUDE,
    FR_BOTH,
    INCOME_LEVEL_OFFSET,
    INCOME_LEVEL_STEP,
    INCOME_LEVELS_COUNT,
    INCOME_MAX_PCT,
    INCOME_MIN_PCT,
    MONTHS_PER_YEAR,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Schema validation
# =============================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Check that *df* contains every column listed in REQUIRED_COLUMNS.

    Raises
    ------
    ValueError
        With a message listing the missing columns.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Databestand mist verplichte kolommen: {', '.join(missing)}"
        )


# =============================================================================
# Data loading
# =============================================================================

@st.cache_data
def load_data(key: str | None) -> pd.DataFrame:
    """Load data from a Parquet file and filter by municipality access key.

    Supports both Streamlit secrets and environment variables for
    configuration.  Validates the schema after loading.
    """
    # Support both Streamlit secrets.toml and environment variables (Cloud Run)
    try:
        data_url = st.secrets["excel_url"]
        key_all = st.secrets["key_all"]
        key_barneveld = st.secrets["key_barneveld"]
        key_delft = st.secrets["key_delft"]
    except (AttributeError, KeyError, FileNotFoundError):
        data_url = os.getenv("EXCEL_URL", "dataset.parquet")
        key_all = os.getenv("KEY_ALL", "")
        key_barneveld = os.getenv("KEY_BARNEVELD", "")
        key_delft = os.getenv("KEY_DELFT", "")

    df = pd.read_parquet(data_url)

    # Validate schema and basic integrity
    validate_schema(df)
    if len(df) == 0:
        raise ValueError("Databestand is leeg — controleer de databron.")

    # Filter by access level
    base_mask = df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '')

    if key == key_all:
        return df[base_mask]

    if key == key_delft:
        excluded = ['Barneveld']
    elif key == key_barneveld:
        excluded = ['Delft']
    else:
        excluded = ['Barneveld', 'Delft']

    return df[base_mask & ~df['Gemeentenaam'].isin(excluded)]


# =============================================================================
# Core filtering
# =============================================================================

def filter_regelingen(
    df: pd.DataFrame,
    gm: Union[str, list[str]],
    hh: str,
    ink: float = 1.0,
    refper: int = 0,
    cav: int = CAV_EXCLUDE,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """Apply standard filters and return matching regulations.

    Parameters
    ----------
    df : DataFrame
        The full regulations dataset.
    gm : str or list of str
        Municipality code(s) (e.g. ``'GM0363'``).
    hh : str
        Household type code (``'HH01'``–``'HH04'``).
    ink : float
        Income level as fraction of social minimum (``1.0`` = 100 %).
    refper : int
        Required years at low income (0, 1, 3, or 5).
    cav : int
        Health insurance discount: 0 = exclude, 1 = include.
    fr : int
        Regulation type: 1 = formal only, 2 = informal only, 3 = both.

    Returns
    -------
    DataFrame
        Filtered rows with renamed columns ``WRD``, ``IG``,
        ``Referteperiode``.
    """
    wrd_col = f'WRD_{hh}'
    ig_col = f'IG_{hh}'
    ref_col = f'Referteperiode_{hh}'

    mask = (df['WB'] == 1) & (df[ig_col] >= ink) & (df[ref_col] <= refper)

    if isinstance(gm, str):
        gm = [gm]
    mask &= df['GMcode'].isin(gm)

    if cav == 1:
        mask &= ((df['BT'] == 1) | (df['CAV'] == 1))
    else:
        mask &= (df['BT'] == 1) & (df['CAV'] == 0)

    if fr == FR_BOTH:
        pass  # no extra filter
    elif fr == 1:
        mask &= (df['FR'] == 'Ja')
    elif fr == 2:
        mask &= (df['FR'] == 'Nee')

    return df[mask].rename(columns={
        ig_col: 'IG',
        ref_col: 'Referteperiode',
        wrd_col: 'WRD',
    })


# =============================================================================
# Graph 1 — Household-type comparison (box plot)
# =============================================================================

def huishoudtypen_data(
    df: pd.DataFrame,
    gm_lbl: dict[str, str],
    hh_lbl: dict[str, str],
    ink: float = 1.0,
    refper: int = 0,
    cav: int = 0,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """Values for all municipalities × all household types.

    Note: Each household type uses different columns (``WRD_HH01``,
    ``IG_HH01``, …), so the four filter calls cannot easily be batched
    into a single pass over the data.
    """
    gm_keys = list(gm_lbl.keys())
    gm_vals = list(gm_lbl.values())
    results: list[pd.DataFrame] = []

    for hh in hh_lbl:
        filtered = (
            filter_regelingen(df, gm_keys, hh, ink, refper, cav, fr)
            .groupby('GMcode')['WRD'].sum().div(MONTHS_PER_YEAR)
            .reindex(gm_keys, fill_value=0.0)
        )

        hh_data = pd.DataFrame({
            'GMcode': gm_keys,
            'Gemeentenaam': gm_vals,
            'Waarde': filtered.values,
            'Huishouden': hh,
        })
        results.append(hh_data[['GMcode', 'Gemeentenaam', 'Waarde', 'Huishouden']])

    return pd.concat(results, ignore_index=True)


# =============================================================================
# Graph 2 — Income progression (markers + line)
# =============================================================================

def inkomensgroepen_data(
    df: pd.DataFrame,
    hh: str,
    gm_lbl: dict[str, str],
    ink_pct: int = INCOME_MIN_PCT,
    refper: int = 0,
    cav: int = 0,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """Values for all municipalities at specific income levels (markers)."""
    # Calculate 6 income levels in steps of 10, centred around selection
    final_digit = ink_pct % INCOME_LEVEL_STEP
    start = ink_pct - INCOME_LEVEL_OFFSET
    max_start = (
        INCOME_MAX_PCT
        - (INCOME_LEVELS_COUNT * INCOME_LEVEL_STEP)
        + final_digit
    )
    start = max(INCOME_MIN_PCT + final_digit, min(max_start, start))
    income_levels = list(range(
        start,
        start + INCOME_LEVELS_COUNT * INCOME_LEVEL_STEP,
        INCOME_LEVEL_STEP,
    ))

    gm_keys = list(gm_lbl.keys())
    gm_vals = list(gm_lbl.values())
    results: list[pd.DataFrame] = []

    for ink_lvl in income_levels:
        filtered = (
            filter_regelingen(
                df, gm_keys, hh, round(ink_lvl / 100, 2), refper, cav, fr,
            )
            .groupby('GMcode')['WRD'].sum().div(MONTHS_PER_YEAR)
            .reindex(gm_keys, fill_value=0.0)
        )

        ink_data = pd.DataFrame({
            'GMcode': gm_keys,
            'Gemeentenaam': gm_vals,
            'Waarde': filtered.values,
            'Inkomen': ink_lvl,
        })
        results.append(ink_data[['GMcode', 'Gemeentenaam', 'Waarde', 'Inkomen']])

    return pd.concat(results, ignore_index=True)


@st.cache_data
def inkomenslijn_data(
    _df: pd.DataFrame,
    gm: str,
    hh: str,
    refper: int = 0,
    cav: int = CAV_EXCLUDE,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """Values for selected municipality at every income level (smooth line).

    Optimised: applies base filters once, then varies only the income
    threshold on the pre-filtered subset.
    """
    wrd_col = f'WRD_{hh}'
    ig_col = f'IG_{hh}'
    ref_col = f'Referteperiode_{hh}'

    # Apply all filters except income threshold once
    mask = (
        (_df['WB'] == 1)
        & (_df[ref_col] <= refper)
        & (_df['GMcode'] == gm)
    )

    if cav == 1:
        mask &= ((_df['BT'] == 1) | (_df['CAV'] == 1))
    else:
        mask &= (_df['BT'] == 1) & (_df['CAV'] == 0)

    if fr == 1:
        mask &= (_df['FR'] == 'Ja')
    elif fr == 2:
        mask &= (_df['FR'] == 'Nee')

    base_df = _df.loc[mask, [ig_col, wrd_col]]

    # Vary only the income threshold on the (much smaller) pre-filtered set
    results: list[dict[str, float | int]] = []
    for income_pct in range(INCOME_MIN_PCT, INCOME_MAX_PCT + 1):
        ink = income_pct / 100
        value = base_df.loc[base_df[ig_col] >= ink, wrd_col].sum() / MONTHS_PER_YEAR
        results.append({'Inkomen': income_pct, 'Waarde': value})

    return pd.DataFrame(results)


# =============================================================================
# Graph 3 — Formal vs. informal (stacked bar)
# =============================================================================

def in_formeel_data(
    df: pd.DataFrame,
    hh: str,
    gm_lbl: dict[str, str],
    ink: float = 1.0,
    refper: int = 0,
    cav: int = 0,
) -> pd.DataFrame:
    """Formal and informal values for all municipalities."""
    gm_keys = list(gm_lbl.keys())

    filtered = (
        filter_regelingen(df, gm_keys, hh, ink, refper, cav)
        .groupby(['GMcode', 'FR'])['WRD'].sum().div(MONTHS_PER_YEAR)
    )

    unstacked = filtered.unstack('FR', fill_value=0)
    unstacked = unstacked.rename(columns={'Ja': 'Formeel', 'Nee': 'Informeel'})

    for col in ('Formeel', 'Informeel'):
        if col not in unstacked.columns:
            unstacked[col] = 0

    unstacked = unstacked.reindex(gm_keys, fill_value=0)
    unstacked['Gemeentenaam'] = list(gm_lbl.values())
    unstacked['Totaal'] = unstacked['Formeel'] + unstacked['Informeel']

    result = unstacked.reset_index()
    return result[['GMcode', 'Gemeentenaam', 'Formeel', 'Informeel', 'Totaal']]


# =============================================================================
# Graph 4 — Value vs. income threshold (scatter)
# =============================================================================

def gem_inkomensgrenzen_data(
    df: pd.DataFrame,
    gm: list[str],
    hh: str,
    refper: int = 0,
    cav: int = 0,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """Weighted income thresholds and values for all municipalities."""
    filtered = filter_regelingen(df, gm, hh, 1, refper, cav, fr)

    filtered = filtered.assign(
        monthly_wrd=filtered['WRD'] / MONTHS_PER_YEAR,
        weighted_component=filtered['WRD'] * (filtered['IG'] - 1) / MONTHS_PER_YEAR,
    )

    result = filtered.groupby('GMcode').agg({
        'monthly_wrd': 'sum',
        'weighted_component': 'sum',
        'Inwoners': 'first',
        'Gemeentenaam': 'first',
    }).reset_index()

    result.columns = ['Gemeente', 'Waarde', 'weighted_sum', 'Inwoners', 'Gemeentenaam']

    # Filter out zero values to prevent division by zero (#2)
    result = result[result['Waarde'] > 0].copy()

    result['Inkomensgrens'] = (
        (1 + (result['weighted_sum'] / result['Waarde'])) * 100
    ).astype(int)

    return result[['Gemeente', 'Gemeentenaam', 'Inkomensgrens', 'Waarde', 'Inwoners']]


# =============================================================================
# Regulations table
# =============================================================================

def regelingen_lijst(
    df: pd.DataFrame,
    gm: str,
    hh: str,
    ink: float = 1.0,
    refper: int = 0,
    cav: int = 0,
    fr: int = FR_BOTH,
) -> pd.DataFrame:
    """All regulations for a municipality, annotated with whether they match."""
    regs = df[df['GMcode'] == gm].copy()
    selected_regs = filter_regelingen(df, gm, hh, ink, refper, cav, fr)

    regs['Regeling'] = regs['N4']
    regs['Waarde'] = regs[f'WRD_{hh}'] / MONTHS_PER_YEAR
    regs['Inkomensgrens'] = regs[f'IG_{hh}']
    regs['Matches'] = regs['ID'].isin(selected_regs['ID'])

    grouped = regs.groupby(['Regeling', 'Matches'], as_index=False).agg({
        'Waarde': 'sum',
        'Inkomensgrens': 'min',
    })

    matching = grouped[grouped['Matches']].sort_values('Waarde', ascending=False)
    non_matching = grouped[~grouped['Matches']].sort_values('Regeling')

    return pd.concat([matching, non_matching])
