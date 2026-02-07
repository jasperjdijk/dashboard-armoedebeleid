"""Tests for data.py — filtering, aggregation, and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from constants import MONTHS_PER_YEAR, REQUIRED_COLUMNS
from data import (
    filter_regelingen,
    gem_inkomensgrenzen_data,
    huishoudtypen_data,
    in_formeel_data,
    regelingen_lijst,
    validate_schema,
)


# =============================================================================
# validate_schema
# =============================================================================

class TestValidateSchema:
    def test_valid_schema(self, sample_df: pd.DataFrame) -> None:
        # Should not raise
        validate_schema(sample_df)

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({'GMcode': ['GM01'], 'Gemeentenaam': ['Test']})
        with pytest.raises(ValueError, match="mist verplichte kolommen"):
            validate_schema(df)

    def test_empty_but_valid_schema(self, sample_df: pd.DataFrame) -> None:
        empty = sample_df.iloc[:0]
        # Schema check passes (columns exist), even though empty
        validate_schema(empty)


# =============================================================================
# filter_regelingen
# =============================================================================

class TestFilterRegelingen:
    def test_basic_filter_single_gemeente(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, 'GM0001', 'HH01')
        # GM0001 has IDs 1, 2, 3 but ID 3 has BT=0, CAV=1 so excluded with cav=0
        assert set(result['ID']) == {1, 2}

    def test_filter_with_cav(self, sample_df: pd.DataFrame) -> None:
        # ID 3 has BT=0, CAV=1, Referteperiode=1.
        # With cav=1 AND refper=1, ID 3 should be included
        result = filter_regelingen(sample_df, 'GM0001', 'HH01', cav=1, refper=1)
        assert set(result['ID']) == {1, 2, 3}

    def test_filter_formal_only(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, 'GM0001', 'HH01', fr=1)
        assert all(result['ID'].isin([1]))  # Only ID 1 is FR='Ja' & BT=1

    def test_filter_informal_only(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, 'GM0001', 'HH01', fr=2)
        assert set(result['ID']) == {2}  # ID 2 is FR='Nee' & BT=1

    def test_filter_by_income(self, sample_df: pd.DataFrame) -> None:
        # At ink=1.25, only regulations with IG >= 1.25 should match
        result = filter_regelingen(sample_df, 'GM0001', 'HH01', ink=1.25)
        assert set(result['ID']) == {1}  # ID 1 has IG=1.3

    def test_filter_by_refper(self, sample_df: pd.DataFrame) -> None:
        # GM0002 has ID 5 with Referteperiode=3
        result = filter_regelingen(sample_df, 'GM0002', 'HH01', refper=3)
        assert set(result['ID']) == {4, 5}

    def test_filter_multiple_gemeenten(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, ['GM0001', 'GM0002'], 'HH01')
        assert len(result) == 3  # IDs 1, 2, 4

    def test_renamed_columns(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, 'GM0001', 'HH01')
        assert 'WRD' in result.columns
        assert 'IG' in result.columns
        assert 'Referteperiode' in result.columns

    def test_no_matches(self, sample_df: pd.DataFrame) -> None:
        result = filter_regelingen(sample_df, 'GM9999', 'HH01')
        assert len(result) == 0


# =============================================================================
# huishoudtypen_data
# =============================================================================

class TestHuishoudtypenData:
    def test_returns_all_households(self, sample_df: pd.DataFrame) -> None:
        gm_lbl = {'GM0001': 'TestStad', 'GM0002': 'ProefDorp'}
        hh_lbl = {'HH01': 'Alleenstaande', 'HH02': 'Alleenstaande ouder met kind'}
        result = huishoudtypen_data(sample_df, gm_lbl, hh_lbl)

        assert set(result['Huishouden'].unique()) == {'HH01', 'HH02'}
        # 2 municipalities × 2 household types = 4 rows
        assert len(result) == 4

    def test_values_are_monthly(self, sample_df: pd.DataFrame) -> None:
        gm_lbl = {'GM0001': 'TestStad'}
        hh_lbl = {'HH01': 'Alleenstaande'}
        result = huishoudtypen_data(sample_df, gm_lbl, hh_lbl)

        # GM0001, HH01, ink=1.0, refper=0: IDs 1 (WRD=1200) + 2 (WRD=600) = 1800 annual
        expected = (1200 + 600) / MONTHS_PER_YEAR
        actual = result.loc[result['GMcode'] == 'GM0001', 'Waarde'].iloc[0]
        assert actual == pytest.approx(expected)


# =============================================================================
# in_formeel_data
# =============================================================================

class TestInFormeelData:
    def test_columns(self, sample_df: pd.DataFrame) -> None:
        gm_lbl = {'GM0001': 'TestStad'}
        result = in_formeel_data(sample_df, 'HH01', gm_lbl)
        assert 'Formeel' in result.columns
        assert 'Informeel' in result.columns
        assert 'Totaal' in result.columns

    def test_formal_informal_split(self, sample_df: pd.DataFrame) -> None:
        gm_lbl = {'GM0001': 'TestStad'}
        result = in_formeel_data(sample_df, 'HH01', gm_lbl)
        row = result.iloc[0]
        # Formeel: ID 1 (WRD=1200, FR=Ja), Informeel: ID 2 (WRD=600, FR=Nee)
        assert row['Formeel'] == pytest.approx(1200 / MONTHS_PER_YEAR)
        assert row['Informeel'] == pytest.approx(600 / MONTHS_PER_YEAR)
        assert row['Totaal'] == pytest.approx((1200 + 600) / MONTHS_PER_YEAR)


# =============================================================================
# gem_inkomensgrenzen_data
# =============================================================================

class TestGemInkomensgrenzenData:
    def test_no_division_by_zero(self, sample_df: pd.DataFrame) -> None:
        """Municipalities with zero total value should be excluded (#2)."""
        # Create a df where one municipality has WRD=0 for all regulations
        df = sample_df.copy()
        df.loc[df['GMcode'] == 'GM0002', 'WRD_HH01'] = 0.0
        result = gem_inkomensgrenzen_data(df, ['GM0001', 'GM0002'], 'HH01')
        # GM0002 should be excluded (zero value)
        assert 'GM0002' not in result['Gemeente'].values

    def test_includes_expected_columns(self, sample_df: pd.DataFrame) -> None:
        result = gem_inkomensgrenzen_data(
            sample_df, ['GM0001', 'GM0002'], 'HH01',
        )
        assert set(result.columns) == {
            'Gemeente', 'Gemeentenaam', 'Inkomensgrens', 'Waarde', 'Inwoners',
        }

    def test_inkomensgrens_is_percentage(self, sample_df: pd.DataFrame) -> None:
        result = gem_inkomensgrenzen_data(
            sample_df, ['GM0001'], 'HH01',
        )
        # Inkomensgrens should be an integer percentage > 100
        ig = result['Inkomensgrens'].iloc[0]
        assert ig >= 100
        assert int(ig) == ig  # numpy int64 is fine, just check it's integer-valued


# =============================================================================
# regelingen_lijst
# =============================================================================

class TestRegelingenLijst:
    def test_matching_and_non_matching(self, sample_df: pd.DataFrame) -> None:
        result = regelingen_lijst(sample_df, 'GM0001', 'HH01')
        assert 'Matches' in result.columns
        assert any(result['Matches'])
        # With default filters, ID 3 doesn't match (BT=0, CAV=1)
        assert any(~result['Matches'])

    def test_empty_municipality(self, sample_df: pd.DataFrame) -> None:
        result = regelingen_lijst(sample_df, 'GM9999', 'HH01')
        assert len(result) == 0
