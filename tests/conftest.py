"""Shared fixtures for dashboard tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal regulations DataFrame for testing filter_regelingen and friends.

    Contains 4 regulations across 2 municipalities and 2 FR types, with
    realistic column values for HH01.
    """
    return pd.DataFrame({
        'GMcode': ['GM0001', 'GM0001', 'GM0001', 'GM0002', 'GM0002'],
        'Gemeentenaam': ['TestStad', 'TestStad', 'TestStad', 'ProefDorp', 'ProefDorp'],
        'Inwoners': [100_000, 100_000, 100_000, 50_000, 50_000],
        'N4': ['Regeling A', 'Regeling B', 'Regeling C', 'Regeling D', 'Regeling E'],
        'ID': [1, 2, 3, 4, 5],
        'WB': [1, 1, 1, 1, 1],
        'BT': [1, 1, 0, 1, 1],
        'CAV': [0, 0, 1, 0, 0],
        'FR': ['Ja', 'Nee', 'Ja', 'Ja', 'Nee'],
        # HH01 columns
        'WRD_HH01': [1200.0, 600.0, 240.0, 2400.0, 360.0],
        'IG_HH01': [1.3, 1.2, 1.0, 1.5, 1.1],
        'Referteperiode_HH01': [0, 0, 1, 0, 3],
        # HH02 columns (needed by huishoudtypen_data)
        'WRD_HH02': [1400.0, 700.0, 280.0, 2800.0, 420.0],
        'IG_HH02': [1.3, 1.2, 1.0, 1.5, 1.1],
        'Referteperiode_HH02': [0, 0, 1, 0, 3],
        # HH03 columns
        'WRD_HH03': [1600.0, 800.0, 320.0, 3200.0, 480.0],
        'IG_HH03': [1.3, 1.2, 1.0, 1.5, 1.1],
        'Referteperiode_HH03': [0, 0, 1, 0, 3],
        # HH04 columns
        'WRD_HH04': [1800.0, 900.0, 360.0, 3600.0, 540.0],
        'IG_HH04': [1.3, 1.2, 1.0, 1.5, 1.1],
        'Referteperiode_HH04': [0, 0, 1, 0, 3],
    })
