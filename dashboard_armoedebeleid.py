"""
Dashboard gemeentelijke inkomensafhankelijke regelingen

A Streamlit dashboard for visualizing municipal income-dependent benefits
in Dutch municipalities. Allows comparison of benefit values across municipalities
for different household types and income levels.

Visualizations:
1. Household comparison - Box plot showing benefit values by household type
2. Income progression - Line chart showing how benefits decrease with income
3. Formal vs Informal - Stacked bar chart comparing regulation types
4. Value vs Threshold - Scatter plot relating benefit value to income thresholds

"""

import logging

import pandas as pd
import streamlit as st

from charts import (
    gem_inkomensgrenzen_grafiek,
    huishoudtypen_grafiek,
    in_formeel_grafiek,
    inkomensgroepen_grafiek,
)
from constants import (
    HOUSEHOLD_LABELS,
    REG_TYPE_VALUES,
    VALID_REFPER_VALUES,
)
from data import (
    gem_inkomensgrenzen_data,
    huishoudtypen_data,
    in_formeel_data,
    inkomensgroepen_data,
    inkomenslijn_data,
    load_data,
    regelingen_lijst,
)
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

logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Dashboard gemeentelijke inkomensafhankelijke regelingen",
    page_icon="Favicon-alt-2.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for layout
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px;
        max-width: 0px;
    }
    [data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
    .stMainBlockContainer {
        padding-top: 0;
        padding-right: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# COLOR CONFIGURATION
# =============================================================================
COLOR_SELECTED = st.get_option("theme.primaryColor")
CHART_COLORS = st.get_option("theme.chartCategoricalColors")

COLOR_OTHER = CHART_COLORS[0]               # #9f9f9f
COLOR_INFORMAL_SELECTED = CHART_COLORS[1]   # #E68C8F
COLOR_INFORMAL_OTHER = CHART_COLORS[2]      # #C5C5C5

# =============================================================================
# HOUSEHOLD TYPE LABELS
# =============================================================================
hh_lbl: dict[str, str] = HOUSEHOLD_LABELS


# =============================================================================
# MAIN APPLICATION
# =============================================================================
try:
    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    data_key = st.query_params.get("key")
    df = load_data(data_key)

    # -------------------------------------------------------------------------
    # Header & Gemeente Labels
    # -------------------------------------------------------------------------
    st.title("Dashboard armoedebeleid", text_alignment="center", anchor=False)

    gm_lbl: dict[str, str] = (
        df[['GMcode', 'Gemeentenaam']]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .set_index('GMcode')['Gemeentenaam']
        .to_dict()
    )

    # -------------------------------------------------------------------------
    # Validated defaults from URL query parameters (#18)
    # -------------------------------------------------------------------------
    params = st.query_params

    default_income     = validate_income(params.get("ink", 100))
    default_huishouden = validate_household(params.get("hh"))
    default_refper     = validate_refper(params.get("ref", 0))
    default_cav        = params.get("cav", "0") == "1"
    default_gemeente   = validate_gemeente(params.get("gm"), gm_lbl)
    default_fr         = validate_fr(params.get("reg", 3))
    default_reg_types  = get_default_reg_types(default_fr)

    # -------------------------------------------------------------------------
    # Sidebar Filters
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters", anchor=False)

        sel_ink_pct = st.slider(
            label="Inkomen",
            min_value=100,
            max_value=200,
            value=default_income,
            step=1,
            format="%d%%",
            key="income",
            help=(
                "Als percentage van het sociale minimum (bijstandsniveau). "
                "100 % = sociaal minimum, 200 % = twee keer het sociaal minimum."
            ),
        )

        sel_gm = st.selectbox(
            label="Gemeente",
            options=gm_lbl,
            format_func=lambda x: gm_lbl[x],
            index=list(gm_lbl).index(default_gemeente),
            key="gemeente",
            help="Welke gemeente moet worden uitgelicht in het dashboard?",
        )

        sel_hh = st.selectbox(
            label="Huishouden:",
            options=hh_lbl,
            format_func=lambda x: hh_lbl[x],
            index=list(hh_lbl).index(default_huishouden),
            key="huishouden",
            help="Welk huishoudtype wilt u meer over weten?",
        )

        sel_refper = st.segmented_control(
            label="Jaren met laag inkomen",
            options=VALID_REFPER_VALUES,
            default=default_refper,
            key="referteperiode",
            help="Jaren dat het voorbeeldhuishouden al het geselecteerde inkomen heeft (referteperiode)",
        )

        # Regulation type — ensure the list is never empty (#11)
        if "reg_types" not in st.session_state:
            st.session_state.reg_types = default_reg_types or ["Formeel", "Informeel"]
        if "prev_reg_types" not in st.session_state:
            st.session_state.prev_reg_types = st.session_state.reg_types

        st.session_state.reg_types = ensure_reg_types_not_empty(
            st.session_state.reg_types,
            st.session_state.prev_reg_types,
        )

        selected_reg_types = st.segmented_control(
            "Type regelingen",
            options=list(REG_TYPE_VALUES.keys()),
            selection_mode="multi",
            key="reg_types",
            help="Selecteer welke regelingen worden meegenomen: formele, informele of beide",
        )

        st.session_state.prev_reg_types = st.session_state.reg_types

        toggle_cav = st.toggle(
            "Korting gemeentepolis",
            value=default_cav,
            key="toggle_cav",
            help="Moet de korting op een eventuele gemeentepolis worden meegenomen op het totaalbedrag? (Deze polis is niet voor ieder huishouden voordelig)",
        )

        # Legend
        st.markdown("**Legenda**")
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {COLOR_SELECTED};"></div>
                <span>{gm_lbl[sel_gm]}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {COLOR_OTHER};"></div>
                <span>Overige gemeenten</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Derived filter values
    # -------------------------------------------------------------------------
    sel_fr = sum(REG_TYPE_VALUES.get(rt, 0) for rt in selected_reg_types) or 3
    sel_cav = 1 if toggle_cav else 0

    # Sync URL with current selections
    new_params: dict[str, str] = {}
    if params.get("key"):
        new_params["key"] = params["key"]
    new_params["ink"] = str(sel_ink_pct)
    new_params["gm"] = sel_gm
    new_params["hh"] = sel_hh
    new_params["ref"] = str(sel_refper)
    new_params["reg"] = str(sel_fr)
    new_params["cav"] = str(sel_cav)
    if dict(st.query_params) != new_params:
        st.query_params.update(new_params)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    graph_col, table_col = st.columns([2, 1])
    show_export = st.query_params.get("export") == "1"

    # Context-dependent text inserts
    if sel_fr == 1:
        formeel = "**formele**"
    elif sel_fr == 2:
        formeel = "**informele**"
    else:
        formeel = ""

    gemeentepolis = (
        "**hierin meegenomen**" if sel_cav == 1
        else "**buiten beschouwing** gelaten"
    )

    reftext = "sinds kort" if sel_refper == 0 else f"al {sel_refper} jaar"

    # -------------------------------------------------------------------------
    # Tabs (inside graph column)
    # -------------------------------------------------------------------------
    with graph_col:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Huishoudtypen",
            "Inkomensgroepen",
            "(In)formeel",
            "Gemiddelde inkomensgrenzen",
        ])

    # -- Graph 1: Box Plot - Household Types --------------------------------
    with tab1:
        hh_data = huishoudtypen_data(
            df, gm_lbl, hh_lbl, sel_ink_pct / 100, sel_refper, sel_cav, sel_fr,
        )
        fig = huishoudtypen_grafiek(
            hh_data, sel_gm, hh_lbl, sel_ink_pct / 100,
            COLOR_SELECTED, COLOR_OTHER,
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})
        st.markdown(
            f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} "
            f"gemeentelijke regelingen waarop een voorbeeldhuishouden recht heeft, "
            f"dat **{reftext}** een inkomen heeft van **{sel_ink_pct}%** van het "
            f"sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}."
        )

    # -- Graph 2: Income Progression ----------------------------------------
    with tab2:
        ink_markers = inkomensgroepen_data(
            df, sel_hh, gm_lbl, sel_ink_pct, sel_refper, sel_cav, sel_fr,
        )
        ink_line = inkomenslijn_data(df, sel_gm, sel_hh, sel_refper, sel_cav, sel_fr)
        fig_income = inkomensgroepen_grafiek(
            ink_markers, ink_line, sel_gm, sel_hh, hh_lbl, sel_ink_pct,
            COLOR_SELECTED, COLOR_OTHER,
        )
        st.plotly_chart(fig_income, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})
        st.markdown(
            f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} "
            f"gemeentelijke regelingen waarop een **{hh_lbl[sel_hh].lower()}** recht "
            f"heeft, met **{reftext}** een bepaald inkomensniveau. De korting op de "
            f"gemeentepolis is {gemeentepolis}."
        )

    # -- Graph 3: Formal vs Informal ----------------------------------------
    with tab3:
        bar_data = in_formeel_data(
            df, sel_hh, gm_lbl, sel_ink_pct / 100, sel_refper, sel_cav,
        )
        fig_bar = in_formeel_grafiek(
            bar_data, sel_gm, sel_hh, hh_lbl, sel_ink_pct,
            COLOR_SELECTED, COLOR_OTHER, COLOR_INFORMAL_SELECTED, COLOR_INFORMAL_OTHER,
        )
        st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})
        st.markdown(
            f"Schatting van de totale waarde (in € per maand) van zowel de formele "
            f"als informele gemeentelijke regelingen waarop een "
            f"**{hh_lbl[sel_hh].lower()}** recht heeft, met **{reftext}** een inkomen "
            f"van **{sel_ink_pct}%** van het sociaal minimum. De korting op de "
            f"gemeentepolis is {gemeentepolis}."
        )

    # -- Graph 4: Value vs Income Threshold ---------------------------------
    with tab4:
        threshold_data = gem_inkomensgrenzen_data(
            df, list(gm_lbl.keys()), sel_hh, sel_refper, sel_cav, sel_fr,
        )
        fig_threshold = gem_inkomensgrenzen_grafiek(
            threshold_data, sel_gm, sel_hh, hh_lbl,
            COLOR_SELECTED, COLOR_OTHER,
        )
        st.plotly_chart(fig_threshold, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})
        st.markdown(
            f"Schatting van de totale waarde (in € per maand) en de gemiddelde "
            f"inkomensgrens van **alle** {formeel} gemeentelijke regelingen waarop "
            f"een **{hh_lbl[sel_hh].lower()}** recht heeft, met **{reftext}** een "
            f"inkomen van 100% van het sociaal minimum. De korting op de "
            f"gemeentepolis is {gemeentepolis}."
        )

    # -------------------------------------------------------------------------
    # Export Buttons (shown below tabs if ?export=1 in URL)
    # -------------------------------------------------------------------------
    if show_export:
        st.markdown("---")
        st.subheader("📥 Exporteer data", anchor=False)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            export_df = hh_data[['Huishouden', 'Gemeentenaam', 'Waarde']].copy()
            export_df['Huishoudtype'] = export_df['Huishouden'].map(hh_lbl)
            export_df = export_df[['Huishoudtype', 'Gemeentenaam', 'Waarde']]
            export_df.columns = ['Huishoudtype', 'Gemeente', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Huishoudtype', 'Gemeente'])
            st.download_button(
                label="Huishoudtypen",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"huishoudtypen_{sel_ink_pct}pct.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            # FIX #1: Inkomen is already an integer percentage — do NOT multiply by 100
            export_ink = ink_markers.copy()
            export_ink['Inkomensniveau'] = export_ink['Inkomen'].astype(str) + '%'
            export_ink = export_ink[['Inkomensniveau', 'Gemeentenaam', 'Waarde']]
            export_ink.columns = ['Inkomensniveau', 'Gemeente', 'Waarde (€ per maand)']
            export_ink = export_ink.sort_values(['Gemeente', 'Inkomensniveau'])
            st.download_button(
                label="Inkomensgroepen",
                data=export_ink.to_csv(index=False).encode('utf-8'),
                file_name=f"inkomensgroepen_{hh_lbl[sel_hh].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            export_bar = bar_data[['Gemeentenaam', 'Formeel', 'Informeel']].melt(
                id_vars=['Gemeentenaam'],
                value_vars=['Formeel', 'Informeel'],
                var_name='Type',
                value_name='Waarde (€ per maand)',
            )
            export_bar.columns = ['Gemeente', 'Type', 'Waarde (€ per maand)']
            export_bar = export_bar.sort_values(['Type', 'Gemeente'])
            st.download_button(
                label="(In)formeel",
                data=export_bar.to_csv(index=False).encode('utf-8'),
                file_name=f"formeel_informeel_{hh_lbl[sel_hh].replace(' ', '_')}_{sel_ink_pct}pct.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col4:
            # FIX #1 related: Inkomensgrens is already in percentage form
            export_thr = threshold_data[['Gemeentenaam', 'Inkomensgrens', 'Waarde', 'Inwoners']].copy()
            export_thr.columns = [
                'Gemeente',
                'Gewogen gemiddelde inkomensgrens (%)',
                'Waarde bij 100% (€ per maand)',
                'Inwoners',
            ]
            export_thr = export_thr.sort_values('Gemeente')
            st.download_button(
                label="Inkomensgrenzen",
                data=export_thr.to_csv(index=False).encode('utf-8'),
                file_name=f"waarden_inkomensgrenzen_{hh_lbl[sel_hh].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # Regulations Table (right column)
    # -------------------------------------------------------------------------
    with table_col:
        display_df = regelingen_lijst(
            df, sel_gm, sel_hh, sel_ink_pct / 100, sel_refper, sel_cav, sel_fr,
        )

        if not display_df.empty:
            matching_df = display_df[display_df['Matches'] == True].copy()
            non_matching_df = display_df[display_df['Matches'] == False].copy()

            # Determine padding width from the maximum value
            max_value = display_df['Waarde'].max() or 0
            pad_width = len(f"{max_value:,.0f}".replace(',', '.')) if max_value > 0 else 3

            def pad_currency(x: float | None) -> str:
                if pd.notna(x) and x is not None:
                    if x == 0:
                        return "€ " + "?".rjust(pad_width)
                    num_str = f"{x:,.0f}".replace(',', '.')
                    return "€ " + num_str.rjust(pad_width)
                return "€ " + "?".rjust(pad_width)

            def format_percentage(x: float | None) -> str:
                if pd.notna(x) and x is not None and x > 0:
                    return f"{int(x * 100)}%"
                return "? %"

            # =================================================================
            # Table 1: Matching regulations
            # =================================================================
            if not matching_df.empty:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                fmt = matching_df.copy()
                fmt['Waarde'] = fmt['Waarde'].apply(pad_currency)
                fmt['Inkomensgrens'] = fmt['Inkomensgrens'].apply(format_percentage)
                fmt = fmt[['Regeling', 'Waarde', 'Inkomensgrens']]

                table_height = 38 + (len(fmt) * 35) + 10
                st.dataframe(
                    fmt,
                    width="stretch",
                    height=table_height,
                    hide_index=True,
                    column_config={
                        "Regeling": st.column_config.TextColumn("Regelingen", width=200),
                        "Waarde": st.column_config.TextColumn("Waarde", width=40),
                        "Inkomensgrens": st.column_config.TextColumn("Grens", width=30),
                    },
                )

                total_waarde = format_dutch_currency(matching_df['Waarde'].sum())
                st.markdown(
                    f"Bovenstaande {formeel} regelingen voor een "
                    f"**{hh_lbl[sel_hh].lower()}** in **{gm_lbl[sel_gm]}** met "
                    f"**{reftext}** een inkomen van **{sel_ink_pct}%** van het "
                    f"sociaal minimum tellen op tot **{total_waarde}** per maand.",
                    unsafe_allow_html=True,
                )
            else:
                # Improved empty state (#13)
                st.info(
                    "Let op! Geen passende regelingen gevonden voor de huidige "
                    "filtercombinatie. Probeer het inkomensniveau te verlagen, "
                    "de referteperiode te verhogen, of een ander type regeling "
                    "te selecteren."
                )

            st.markdown(
                "De gemeente kent ook nog de onderstaande regelingen, die mogelijk "
                "niet van toepassing zijn of waarvan de waarde niet goed te bepalen was."
            )

            # =================================================================
            # Table 2: Non-matching regulations
            # =================================================================
            if not non_matching_df.empty:
                fmt_nm = non_matching_df.copy()
                fmt_nm['Waarde'] = fmt_nm['Waarde'].apply(pad_currency)
                fmt_nm['Inkomensgrens'] = fmt_nm['Inkomensgrens'].apply(format_percentage)
                fmt_nm = fmt_nm[['Regeling', 'Waarde', 'Inkomensgrens']]

                styled = fmt_nm.style.apply(
                    lambda x: [f'color: {COLOR_OTHER}'] * len(x), axis=1,
                )
                st.dataframe(
                    styled,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Regeling": st.column_config.TextColumn("Regelingen", width=210),
                        "Waarde": st.column_config.TextColumn("Waarde", width=40),
                        "Inkomensgrens": st.column_config.TextColumn("Grens", width=30),
                    },
                )
            else:
                st.info("Geen overige regelingen gevonden.")
        else:
            st.info(f"Geen regelingen gevonden voor {gm_lbl[sel_gm]}")

except FileNotFoundError:
    st.error("Parquet databestand niet gevonden. Controleer of de databron beschikbaar is.")
except ValueError as exc:
    # Schema validation or data integrity errors (#20)
    st.error(f"Dataprobleem: {exc}")
    logger.exception("Data validation error")
except Exception:
    # Log full traceback but show a user-friendly message (#20)
    logger.exception("Onverwachte fout in het dashboard")
    st.error(
        "Er is een onverwachte fout opgetreden bij het laden van het dashboard. "
        "Probeer de pagina te vernieuwen. Neem contact op met de beheerder als "
        "het probleem aanhoudt."
    )
