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

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================
st.set_page_config(
    page_title="Dashboard gemeentelijke inkomensafhankelijke regelingen",
    page_icon="Favicon-alt-2.png",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ================================================================================
# COLOR CONFIGURATION
# ================================================================================
# Read colors from Streamlit theme config
COLOR_SELECTED = st.get_option("theme.primaryColor")  # Selected municipality/highlight
CHART_COLORS = st.get_option("theme.chartCategoricalColors")  # All other colors

# Assign semantic names to colors from chartCategoricalColors array
COLOR_OTHER = CHART_COLORS[0]              # #9f9f9f - Other/unselected municipalities
COLOR_INFORMAL_SELECTED = CHART_COLORS[1]  # #E68C8F - Selected informal regulations
COLOR_INFORMAL_OTHER = CHART_COLORS[2]     # #C5C5C5 - Other informal regulations

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

@st.cache_data
def load_data(key):
    """Load all required data from Excel or Parquet file and merge municipality information"""
    # Support both Streamlit secrets.toml and environment variables (for Cloud Run)
    # Use defensive check to avoid issues when secrets.toml doesn't exist
    try:
        has_secrets = hasattr(st, 'secrets') and len(st.secrets) > 0 and "excel_url" in st.secrets
    except Exception:
        has_secrets = False

    if has_secrets:
        # Streamlit Cloud or local development
        data_url = st.secrets["excel_url"]
        key_all = st.secrets["key_all"]
        key_barneveld = st.secrets["key_barneveld"]
        key_delft = st.secrets["key_delft"]
    else:
        # Google Cloud Run (environment variables)
        data_url = os.getenv("EXCEL_URL", "dataoverzicht_dashboard_armoedebeleid.xlsx")
        key_all = os.getenv("KEY_ALL", "")
        key_barneveld = os.getenv("KEY_BARNEVELD", "")
        key_delft = os.getenv("KEY_DELFT", "")

    # Load data - prefer Parquet for speed, fall back to Excel
    # Check if URL contains .parquet (handle query params like ?dl=1)
    is_parquet = '.parquet' in data_url.lower()

    if is_parquet:
        # Load from Parquet (5-10x faster than Excel)
        df = pd.read_parquet(data_url)
    else:
        # Load from Excel with only required columns
        required_columns = [
            'Gemeentenaam', 'GMcode', 'Inwoners', 'ID', 'N4', 'FR', 'WB', 'BT', 'CAV',
            'WRD_HH01', 'IG_HH01', 'Referteperiode_HH01',
            'WRD_HH02', 'IG_HH02', 'Referteperiode_HH02',
            'WRD_HH03', 'IG_HH03', 'Referteperiode_HH03',
            'WRD_HH04', 'IG_HH04', 'Referteperiode_HH04'
        ]
        excel_file = pd.ExcelFile(data_url)
        df = pd.read_excel(excel_file, sheet_name="Totaaloverzicht", usecols=required_columns)

    if key == key_all:
        return df[df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '')]
    elif key == key_delft:
        excluded_municipalities = ['Barneveld']
    elif key == key_barneveld:
        excluded_municipalities = ['Delft']
    else:
        excluded_municipalities = ['Barneveld', 'Delft']

    return df[df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '') & ~df['Gemeentenaam'].isin(excluded_municipalities)]


def get_alle_gemeenten(_df, key):
    """Get all unique municipality codes from the dataframe."""
    return _df['GMcode'].unique()


def format_dutch_currency(value, decimals=0):
    """Format number as Dutch currency (dot for thousands, comma for decimals)."""
    formatted = f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"€ {formatted}"


def filter_regelingen(df, hh, ink=1, refper=0, cav=0, fr=3, gmcode=None):
    """
    Apply standard filters to the dataframe and return a boolean mask.

    Parameters:
    -----------
    df : DataFrame
        The Totaaloverzicht dataframe
    hh : str
        Household type code ('HH01', 'HH02', 'HH03', 'HH04')
    ink : float
        Income level as fraction of social minimum (1.0 = 100%)
    refper : int
        Required years at low income (0-5)
    cav : int
        Include health insurance discount: 0 = no, 1 = yes
    fr : int
        Formal regulation filter: 1 = formal, 2 = informal, 3 = both
    gmcode : str, optional
        Municipality code. If None, applies to all municipalities.
    """

    wrd_col, ig_col, ref_col = f'WRD_{hh}', f'IG_{hh}', f'Referteperiode_{hh}'

    mask = (df['WB'] == 1) & (df[ig_col] >= ink) & (df[ref_col] <= refper)

    if gmcode is not None:
        mask &= (df['GMcode'] == gmcode)

    if cav == 1:
        mask &= ((df['BT'] == 1) | (df['CAV'] == 1))
    else:
        mask &= (df['BT'] == 1) & (df['CAV'] == 0)

    if fr == 1:
        mask &= (df['FR'] == 'Ja')
    elif fr == 2:
        mask &= (df['FR'] == 'Nee')

    return df[mask].rename(columns={
        ig_col: 'IG',
        ref_col: 'Referteperiode',
        wrd_col: 'WRD'
    })


# ================================================================================
# CACHED DATA FUNCTIONS FOR GRAPHS AND TABLE
# ================================================================================
def regelingen_lijst(_df, gmcode, hh, ink, refper, cav, fr, key=None):
    """Get table data for all regulations, with matching status. Sorted: matching by value, non-matching alphabetically."""
    wrd_col, ig_col = f'WRD_{hh}', f'IG_{hh}'
    regs = _df[_df['GMcode'] == gmcode]
    df = filter_regelingen(_df, hh, ink, refper, cav, fr, gmcode)
    matching_ids = df['ID']

    grouped = regs.assign(
        Regeling=regs['N4'],
        Waarde=regs[wrd_col] / 12,
        Inkomensgrens=regs[ig_col],
        Matches=regs['ID'].isin(matching_ids)
    ).groupby(['Regeling', 'Matches'], as_index=False).agg(
        Waarde=('Waarde', 'sum'),
        Inkomensgrens=('Inkomensgrens', 'min')
    )

    return pd.concat([
        grouped[grouped['Matches']].sort_values('Waarde', ascending=False),
        grouped[~grouped['Matches']].sort_values('Regeling')
    ])


def get_household_data(_df, ink, refper, cav, fr, key=None):
    """Calculate values for all municipalities and all household types (Graph 1)."""
    alle_gemeenten = get_alle_gemeenten(_df, key)
    results = []

    for hh in ['HH01', 'HH02', 'HH03', 'HH04']:
        df = filter_regelingen(_df, hh, ink, refper, cav, fr)
        hh_data = (df.groupby('GMcode')['WRD'].sum() / 12)
        hh_data = hh_data.reindex(alle_gemeenten, fill_value=0)
        hh_data = hh_data.reset_index()
        hh_data.columns = ['Gemeente', 'Waarde']
        hh_data['Huishouden'] = hh
        results.append(hh_data)

    return pd.concat(results, ignore_index=True)

def get_income_progression_data(_df, hh, ink, refper, cav, fr, key=None):
    """Calculate values for all municipalities at specific income levels (Graph 2 markers)."""
    # Calculate income levels to show based on selected income
    final_digit = ink % 10
    z = min(max(ink - 20, 100 + final_digit), 140 + final_digit)
    income_levels_to_show = [z/100, (z+10)/100, (z+20)/100, (z+30)/100, (z+40)/100, (z+50)/100]

    alle_gemeenten = get_alle_gemeenten(_df, key)
    results = []
 
    for income_level in income_levels_to_show:
        df = filter_regelingen(_df, hh, income_level, refper, cav, fr)
        ink_data = (df.groupby('GMcode')['WRD'].sum() / 12)
        ink_data = ink_data.reindex(alle_gemeenten, fill_value=0)
        ink_data = ink_data.reset_index()
        ink_data.columns = ['Gemeente', 'Waarde']
        ink_data['Inkomen'] = income_level
        results.append(ink_data)

    return pd.concat(results, ignore_index=True)

@st.cache_data
def get_income_line_data(_df, gmcode, hh, refper, cav, fr, key=None):
    """Calculate values for selected municipality at all income levels (Graph 2 line)."""
    all_income_levels = [i/100 for i in range(100, 201, 1)]
    results = []
    for income_level in all_income_levels:
        df = filter_regelingen(_df, hh, income_level, refper, cav, fr, gmcode)
         
        results.append({
            'Inkomen': income_level,
            'Waarde': df['WRD'].sum() / 12
        })

    return pd.DataFrame(results)

def get_formal_informal_data(_df, hh, ink, refper, cav, key=None):
    """Calculate formal and informal values for all municipalities (Graph 3)."""
    gemeente_codes = get_alle_gemeenten(_df, key)
    df = filter_regelingen(
        df=_df,
        hh=hh,
        ink=ink,
        refper=refper,
        cav=cav,
        fr=3,
        gmcode=None
    )

    grouped = (
        df.groupby(['GMcode', 'FR'])['WRD']
        .sum()
        .div(12)
        .unstack('FR', fill_value=0)
        .rename(columns={'Ja': 'Formeel', 'Nee': 'Informeel'})
        .reindex(gemeente_codes, fill_value=0)
        .reset_index()
        .rename(columns={'GMcode': 'Gemeente'})
    )

    # Ensure both Formeel and Informeel columns exist (they might be missing if no regulations match)
    if 'Formeel' not in grouped.columns:
        grouped['Formeel'] = 0
    if 'Informeel' not in grouped.columns:
        grouped['Informeel'] = 0

    grouped['Totaal'] = grouped['Formeel'] + grouped['Informeel']
    return grouped

def get_threshold_data(_df, selected_huishouden, selected_referteperiode, selected_cav, selected_fr, key=None):
    """Calculate weighted income thresholds and values for all municipalities (Graph 4)."""
    gemeente_codes = get_alle_gemeenten(_df, key)
    threshold_data_values = []
    for gmcode in gemeente_codes:
        gemeente_regs = _df[_df['GMcode'] == gmcode]
        if len(gemeente_regs) == 0:
            continue

        df = filter_regelingen(
            df=_df,
            hh=selected_huishouden,
            ink=1.0,
            refper=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr,
            gmcode=gmcode)

        wrd_at_100 = df['WRD'].sum() / 12
        weighted_sum = (df['WRD'] * (df['IG'] - 1)).sum() /12

        if wrd_at_100 > 0:
            weighted_ig = 1 + (weighted_sum / wrd_at_100)
        else:
            weighted_ig = None

        if 'Inwoners' in _df.columns and len(gemeente_regs) > 0:
            try:
                inwoners = gemeente_regs['Inwoners'].iloc[0]
                if pd.isna(inwoners):
                    inwoners = 50000
            except (IndexError, KeyError):
                inwoners = 50000
        else:
            inwoners = 50000

        if weighted_ig is not None:
            threshold_data_values.append({
                'Gemeente': gmcode,
                'Inkomensgrens': weighted_ig,
                'Waarde': wrd_at_100,
                'Inwoners': inwoners
            })

    return pd.DataFrame(threshold_data_values)

# ================================================================================
# CACHED FIGURE FUNCTIONS
# ================================================================================

def create_household_figure(_df, selected_income, selected_income_pct, selected_referteperiode, selected_cav, selected_fr, selected_gemeente, gemeente_labels, household_labels, key=None):
    """Create box plot figure for household comparison (Graph 1)."""
    # Get data
    plot_df = get_household_data(_df, selected_income, selected_referteperiode, selected_cav, selected_fr, key=key)
    plot_df['Gemeentenaam'] = plot_df['Gemeente'].map(gemeente_labels)
    plot_df['Huishouden_Label'] = plot_df['Huishouden'].map(household_labels)

    fig = go.Figure()
    for household in sorted(plot_df['Huishouden_Label'].unique()):
        household_data = plot_df[plot_df['Huishouden_Label'] == household]

        selected_data = household_data[household_data['Gemeente'] == selected_gemeente]
        other_data = household_data[household_data['Gemeente'] != selected_gemeente]

        if len(other_data) > 0:
            # Vectorized hover text
            hover_text_other = (
                "<b>" + other_data['Gemeentenaam'].astype(str) + "</b><br>" +
                f"{selected_income_pct}% sociaal minimum<br>Waarde: € " +
                other_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
            )

            fig.add_trace(go.Box(
                x=[household] * len(other_data),
                y=other_data['Waarde'],
                name=household,
                boxpoints='all',
                jitter=0.3,
                pointpos=0,
                marker=dict(
                    size=8,
                    color=COLOR_OTHER,
                    opacity=0.6
                ),
                hovertext=hover_text_other,
                hoverinfo='text',
                customdata=other_data['Gemeente'].values,
                showlegend=False,
                fillcolor='rgba(255,255,255,0)',
                line=dict(color='rgba(255,255,255,0)')
            ))

        if len(selected_data) > 0:
            # Vectorized hover text
            hover_text_selected = (
                "<b>" + selected_data['Gemeentenaam'].astype(str) + "</b><br>" +
                f"{selected_income_pct}% sociaal minimum<br>Waarde: € " +
                selected_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
            )

            fig.add_trace(go.Box(
                x=[household] * len(selected_data),
                y=selected_data['Waarde'],
                name=household,
                boxpoints='all',
                jitter=0.3,
                pointpos=0,
                marker=dict(
                    size=10,
                    color=COLOR_SELECTED,
                ),
                hovertext=hover_text_selected,
                hoverinfo='text',
                customdata=selected_data['Gemeente'].values,
                showlegend=False,
                fillcolor='rgba(255,255,255,0)',
                line=dict(color='rgba(255,255,255,0)')
            ))

    selected_municipality_data = plot_df[plot_df['Gemeente'] == selected_gemeente]
    for _, row in selected_municipality_data.iterrows():
        fig.add_annotation(
            x=row['Huishouden_Label'],
            y=row['Waarde'],
            text=format_dutch_currency(row['Waarde']),
            showarrow=False,
            xanchor='left',
            xshift=20,
            yshift=0,
            font=dict(size=14, color='black')
        )

    # Create multi-line x-axis labels with specific breaks
    label_mapping = {
        'Alleenstaande': 'Alleenstaande',
        'Alleenstaande ouder met kind': 'Alleenstaande<br>ouder met kind',
        'Paar': 'Paar',
        'Paar met twee kinderen': 'Paar met<br>twee kinderen'
    }
    x_labels = [label_mapping.get(label, label) for label in sorted(plot_df['Huishouden_Label'].unique())]

    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=450,
        showlegend=False,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=60, l=50, r=20, autoexpand=False),
        yaxis=dict(
            tickprefix="€ ",
            tickfont=dict(size=14),
            fixedrange=True
        ),
        xaxis=dict(
            tickfont=dict(size=14),
            tickangle=0,
            tickmode='array',
            tickvals=sorted(plot_df['Huishouden_Label'].unique()),
            ticktext=x_labels,
            fixedrange=True
        )
    )

    return fig

def create_income_figure(_df, selected_huishouden, selected_income_pct, selected_income, selected_referteperiode, selected_cav, selected_fr, selected_gemeente, gemeente_labels, household_labels, key=None):
    """Create line chart figure for income progression (Graph 2)."""
    fig_income = go.Figure()

    # Get cached data for all municipalities at specific income levels
    income_df = get_income_progression_data(
        _df, selected_huishouden, selected_income_pct, selected_referteperiode,
        selected_cav, selected_fr, key=key
    )

    # Calculate income levels for x-axis ticks
    final_digit = selected_income_pct % 10
    z = min(max(selected_income_pct - 20, 100 + final_digit), 140 + final_digit)
    income_levels_to_show = [z/100, (z+10)/100, (z+20)/100, (z+30)/100, (z+40)/100, (z+50)/100]

    # Split data into selected and other municipalities
    selected_marker_data = income_df[income_df['Gemeente'] == selected_gemeente].copy()
    other_marker_data = income_df[income_df['Gemeente'] != selected_gemeente].copy()

    # Add gemeente names and create hover text vectorized for other municipalities
    other_marker_data['Gemeentenaam'] = other_marker_data['Gemeente'].map(gemeente_labels)
    other_marker_data['hover_text'] = (
        "<b>" + other_marker_data['Gemeentenaam'].astype(str) + "</b><br>" +
        household_labels[selected_huishouden] + "<br>Waarde: € " +
        other_marker_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
    )

    # Single trace for all other municipalities
    if len(other_marker_data) > 0:
        fig_income.add_trace(go.Scatter(
            x=other_marker_data['Inkomen'] * 100,
            y=other_marker_data['Waarde'],
            mode='markers',
            name='Overige gemeenten',
            marker=dict(
                size=8,
                color=COLOR_OTHER,
                opacity=0.6
            ),
            hovertext=other_marker_data['hover_text'],
            hoverinfo='text',
            showlegend=False
        ))

    # Get cached line data for selected municipality
    selected_all_df = get_income_line_data(
        _df, selected_gemeente, selected_huishouden, selected_referteperiode,
        selected_cav, selected_fr, key=key
    )

    selected_gemeente_name = gemeente_labels[selected_gemeente]
    fig_income.add_trace(go.Scatter(
        x=selected_all_df['Inkomen'] * 100,
        y=selected_all_df['Waarde'],
        mode='lines',
        name=selected_gemeente_name,
        line=dict(
            color=COLOR_SELECTED,
            width=2
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Vectorized hover text for selected municipality
    selected_marker_data['hover_text'] = (
        "<b>" + selected_gemeente_name + "</b><br>" +
        household_labels[selected_huishouden] + "<br>Waarde: € " +
        selected_marker_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
    )

    fig_income.add_trace(go.Scatter(
        x=selected_marker_data['Inkomen'] * 100,
        y=selected_marker_data['Waarde'],
        mode='markers',
        name=selected_gemeente_name,
        marker=dict(
            size=10,
            color=COLOR_SELECTED
        ),
        hovertext=selected_marker_data['hover_text'],
        hoverinfo='text',
        showlegend=False
    ))

    # Add label for the selected income level
    selected_income_value = selected_all_df[abs(selected_all_df['Inkomen'] - selected_income) < 0.001]['Waarde'].values
    if len(selected_income_value) > 0:
        fig_income.add_annotation(
            x=selected_income_pct,
            y=selected_income_value[0],
            text=format_dutch_currency(selected_income_value[0]),
            showarrow=False,
            xanchor='center',
            xshift=0,
            yshift=15,
            font=dict(size=14, color='black')
        )

    # Create tick values matching income_levels_to_show
    tick_vals = [round(level * 100) for level in income_levels_to_show]
    tick_text = [f'{val}%' for val in tick_vals]

    fig_income.update_layout(
        xaxis_title="Inkomen (% van sociaal minimum)",
        yaxis_title="",
        height=450,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
        xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=14),
            range=[tick_vals[0] - 5, tick_vals[-1] + 5],
            fixedrange=True
        ),
        yaxis=dict(
            tickprefix="€ ",
            tickfont=dict(size=14),
            fixedrange=True
        )
    )

    return fig_income

def create_formal_informal_figure(_df, selected_huishouden, selected_income, selected_income_pct, selected_referteperiode, selected_cav, selected_gemeente, gemeente_labels, household_labels, key=None):
    """Create stacked bar chart for formal vs informal regulations (Graph 3)."""
    # Get cached formal/informal data
    bar_data = get_formal_informal_data(
        _df, selected_huishouden, selected_income, selected_referteperiode,
        selected_cav, key=key
    )
    bar_data['Gemeentenaam'] = bar_data['Gemeente'].map(gemeente_labels)
    bar_data = bar_data.sort_values('Formeel', ascending=False)

    # Vectorized color and hover text generation
    is_selected = bar_data['Gemeente'] == selected_gemeente
    colors_formal = is_selected.map({True: COLOR_SELECTED, False: COLOR_OTHER}).tolist()
    colors_informal = is_selected.map({True: COLOR_INFORMAL_SELECTED, False: COLOR_INFORMAL_OTHER}).tolist()

    hover_prefix = f"{household_labels[selected_huishouden]}<br>{selected_income_pct}% sociaal minimum<br>"
    hover_formal = (
        hover_prefix + "Waarde formele regelingen: € " +
        bar_data['Formeel'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str) +
        "<extra></extra>"
    ).tolist()
    hover_informal = (
        hover_prefix + "Waarde informele regelingen: € " +
        bar_data['Informeel'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str) +
        "<extra></extra>"
    ).tolist()

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=bar_data['Gemeentenaam'],
        y=bar_data['Formeel'],
        name='Formeel',
        marker=dict(color=colors_formal),
        hovertemplate=hover_formal
    ))

    fig_bar.add_trace(go.Bar(
        x=bar_data['Gemeentenaam'],
        y=bar_data['Informeel'],
        name='Informeel',
        marker=dict(color=colors_informal),
        hovertemplate=hover_informal
    ))

    fig_bar.update_layout(
        barmode='stack',
        xaxis_title="",
        yaxis_title="",
        height=450,
        showlegend=True,
        dragmode=False,
        margin=dict(t=0, b=100, l=50, r=20, autoexpand=False),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.99
        ),
        yaxis=dict(
            tickprefix="€ ",
            tickfont=dict(size=14),
            fixedrange=True
        ),
        xaxis=dict(
            tickfont=dict(size=14),
            tickangle=-45,
            fixedrange=True
        )
    )

    return fig_bar

def create_threshold_figure(_df, selected_huishouden, selected_referteperiode, selected_cav, selected_fr, selected_gemeente, gemeente_labels, household_labels, key=None):
    """Create scatter plot for value vs income threshold (Graph 4)."""
    # Get cached threshold data
    threshold_data = get_threshold_data(
        _df, selected_huishouden, selected_referteperiode, selected_cav,
        selected_fr, key=key
    )
    threshold_data['Gemeentenaam'] = threshold_data['Gemeente'].map(gemeente_labels)
    fig_threshold = go.Figure()

    if len(threshold_data) > 0:
        selected_threshold_data = threshold_data[threshold_data['Gemeente'] == selected_gemeente]
        other_threshold_data = threshold_data[threshold_data['Gemeente'] != selected_gemeente]
    else:
        selected_threshold_data = pd.DataFrame()
        other_threshold_data = pd.DataFrame()

    if len(other_threshold_data) > 0:
        # Vectorized hover text generation
        other_threshold_data = other_threshold_data.copy()
        other_threshold_data['hover_text'] = (
            "<b>" + other_threshold_data['Gemeentenaam'].astype(str) + "</b><br>" +
            household_labels[selected_huishouden] + "<br>Inkomensgrens: " +
            (other_threshold_data['Inkomensgrens'] * 100).astype(int).astype(str) + "%<br>" +
            "Waarde bij 100% sociaal minimum: € " +
            other_threshold_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str) + "<br>" +
            "Inwoners: " + other_threshold_data['Inwoners'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
        )

        fig_threshold.add_trace(go.Scatter(
            x=other_threshold_data['Inkomensgrens'] * 100,
            y=other_threshold_data['Waarde'],
            mode='markers',
            marker=dict(
                size=other_threshold_data['Inwoners'] / 10000,
                color=COLOR_OTHER,
                opacity=0.6,
                sizemode='diameter'
            ),
            hovertext=other_threshold_data['hover_text'],
            hoverinfo='text',
            customdata=other_threshold_data['Gemeente'].values,
            showlegend=False
        ))

    if len(selected_threshold_data) > 0:
        # Vectorized hover text generation
        selected_threshold_data = selected_threshold_data.copy()
        selected_threshold_data['hover_text'] = (
            "<b>" + selected_threshold_data['Gemeentenaam'].astype(str) + "</b><br>" +
            household_labels[selected_huishouden] + "<br>Gemiddelde inkomensgrens: " +
            (selected_threshold_data['Inkomensgrens'] * 100).astype(int).astype(str) + "%<br>" +
            "Waarde bij 100% sociaal minimum: € " +
            selected_threshold_data['Waarde'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str) + "<br>" +
            "Inwoners: " + selected_threshold_data['Inwoners'].apply(lambda x: f"{x:,.0f}".replace(',', '.')).astype(str)
        )

        fig_threshold.add_trace(go.Scatter(
            x=selected_threshold_data['Inkomensgrens'] * 100,
            y=selected_threshold_data['Waarde'],
            mode='markers',
            marker=dict(
                size=selected_threshold_data['Inwoners'] / 10000,
                color=COLOR_SELECTED,
                sizemode='diameter'
            ),
            hovertext=selected_threshold_data['hover_text'],
            hoverinfo='text',
            customdata=selected_threshold_data['Gemeente'].values,
            showlegend=False
        ))

        # Add label for selected municipality
        for _, row in selected_threshold_data.iterrows():
            fig_threshold.add_annotation(
                x=row['Inkomensgrens'] * 100,
                y=row['Waarde'],
                text=row['Gemeentenaam'],
                showarrow=False,
                xanchor='left',
                xshift=10,
                font=dict(size=12, color='black')
            )

    fig_threshold.update_layout(
        xaxis_title="Inkomensgrens (% van sociaal minimum)",
        yaxis_title="",
        height=450,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
        xaxis=dict(
            ticksuffix="%",
            tickfont=dict(size=14),
            fixedrange=True
        ),
        yaxis=dict(
            tickprefix="€ ",
            tickfont=dict(size=14),
            fixedrange=True
        )
    )

    return fig_threshold

# ================================================================================
# MAIN APPLICATION
# ================================================================================

try:
    # ----------------------------------------------------------------------------
    # Data Preparation
    # ----------------------------------------------------------------------------
    data_key = st.query_params.get("key")
    df = load_data(data_key)

    # Household type mapping
    household_labels = {
        'HH01': 'Alleenstaande',
        'HH02': 'Alleenstaande ouder met kind',
        'HH03': 'Paar',
        'HH04': 'Paar met twee kinderen'
    }

    # ----------------------------------------------------------------------------
    # Header and Gemeente Labels Preparation
    # ----------------------------------------------------------------------------
    st.title("Dashboard armoedebeleid", anchor=False)

    # Prepare gemeente labels before tabs (more efficient with set_index)
    gemeente_labels = (df[['GMcode', 'Gemeentenaam']]
                       .dropna()
                       .drop_duplicates()
                       .astype(str)
                       .set_index('GMcode')['Gemeentenaam']
                       .to_dict())

    # ----------------------------------------------------------------------------
    # Selectors (in sidebar) - synced with URL query parameters
    # ----------------------------------------------------------------------------

    # Get defaults from URL query params (fall back to defaults if not present)
    params = st.query_params
    default_income = int(params.get("ink", 100))
    default_gemeente = params.get("gm", "GM0363")
    default_huishouden = params.get("hh", "HH04")
    default_refper = int(params.get("ref", 0))

    with st.sidebar:
        st.header("Filters", anchor=False)

        selected_income_pct = st.slider(
            "Inkomen",
            min_value=100,
            max_value=200,
            value=default_income,
            step=1,
            format="%d%%",
            key="income",
            help="Als percentage van het sociale minimum (bijstandsniveau)"
        )
        selected_income = selected_income_pct / 100

        # Ensure default gemeente exists in the data
        if default_gemeente not in gemeente_labels:
            default_gemeente = list(gemeente_labels.keys())[0]

        selected_gemeente = st.selectbox(
            "Gemeente",
            options=gemeente_labels.keys(),
            format_func=lambda x: gemeente_labels[x],
            index=list(gemeente_labels.keys()).index(default_gemeente),
            key="gemeente",
            help="Welke gemeente moet worden uitgelicht in het dashboard?"
        )

        # Ensure default huishouden is valid
        if default_huishouden not in household_labels:
            default_huishouden = "HH04"

        selected_huishouden = st.selectbox(
            "Huishouden:",
            options=list(household_labels.keys()),
            format_func=lambda x: household_labels[x],
            index=list(household_labels.keys()).index(default_huishouden),
            key="huishouden",
            help="Welk huishoudtype wilt u meer over weten?"
        )

        selected_referteperiode = st.segmented_control(
            "Jaren met laag inkomen",
            options=[0, 1, 3, 5],
            default=default_refper if default_refper in [0, 1, 2, 3, 4, 5] else 0,
            key="referteperiode",
            help="Jaren dat het voorbeeldhuishouden al het geselecteerde inkomen heeft (referteperiode)"
        )

        # Regulation type selector (multi-select)
        # URL format: reg=1 (Formeel), reg=2 (Informeel), reg=3 (both)
        default_fr = int(params.get("reg", 3))

        # Convert fr value to default_reg_types list
        if default_fr == 1:
            default_reg_types = ["Formeel"]
        elif default_fr == 2:
            default_reg_types = ["Informeel"]
        else:
            default_reg_types = ["Formeel", "Informeel"]

        # Initialize session state if it doesn't exist - no rerun needed
        if "reg_types" not in st.session_state:
            st.session_state.reg_types = default_reg_types if default_reg_types else ["Formeel", "Informeel"]

        # Regulation type values: Formeel=1, Informeel=2, both=3
        reg_type_values = {"Formeel": 1, "Informeel": 2}
        selected_reg_types = st.segmented_control(
            "Type regelingen",
            options=list(reg_type_values.keys()),
            selection_mode="multi",
            key="reg_types",
            help="Selecteer welke regelingen worden meegenomen: formele, informele of beide"
        )

        default_cav = params.get("cav", "0") == "1"
        toggle_cav = st.toggle("Korting gemeentepolis", value=default_cav, key="toggle_cav", help="Moet de korting op een eventuele geemeentepolis worden meegenomen op het totaalbedrag? (Deze polis is niet voor ieder huishouden voordelig)")

        # Legend
        st.markdown("**Legenda**")
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {COLOR_SELECTED};"></div>
                <span>{gemeente_labels[selected_gemeente]}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {COLOR_OTHER};"></div>
                <span>Overige gemeenten</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Update URL with current selections (preserving the key parameter)
    # Only update if params actually changed to prevent unnecessary reruns
    new_params = {"key": params.get("key", "")} if params.get("key") else {}
    new_params["ink"] = str(selected_income_pct)
    new_params["gm"] = selected_gemeente
    new_params["hh"] = selected_huishouden
    new_params["ref"] = str(selected_referteperiode)
    # Calculate selected_fr and CAV for URL
    selected_fr = sum(reg_type_values.get(rt, 0) for rt in selected_reg_types) or 3
    selected_cav = 1 if toggle_cav else 0
    new_params["reg"] = str(selected_fr)
    new_params["cav"] = str(selected_cav)
    # Only update query params if they changed
    if dict(st.query_params) != new_params:
        st.query_params.update(new_params)

    # Create two-column layout: graphs on left, table on right
    graph_col, table_col = st.columns([2, 1])

    with graph_col:
        # Create tabs for the graphs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Huishoudtypen",
            "Inkomensgroepen",
            "(In)formeel",
            "Gemiddelde inkomensgrenzen"
        ])

    # ----------------------------------------------------------------------------
    # Graph 1: Box Plot - Value by Household Type
    # ----------------------------------------------------------------------------
    with tab1:
        fig = create_household_figure(
            df,
            selected_income,
            selected_income_pct,
            selected_referteperiode,
            selected_cav,
            selected_fr,
            selected_gemeente,
            gemeente_labels,
            household_labels,
            key=data_key
        )

        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        # Export data to CSV - only show if ?export=1 in URL
        try:
            # Try newer Streamlit API first
            show_export = st.query_params.get("export") == "1"
        except AttributeError:
            # Fallback to older API
            query_params = st.experimental_get_query_params()
            show_export = query_params.get("export", ["0"])[0] == "1"

        if show_export:
            plot_df = get_household_data(df, selected_income, selected_referteperiode, selected_cav, selected_fr, key=data_key)
            plot_df['Gemeentenaam'] = plot_df['Gemeente'].map(gemeente_labels)
            plot_df['Huishouden_Label'] = plot_df['Huishouden'].map(household_labels)

            export_df = plot_df[['Huishouden_Label', 'Gemeentenaam', 'Waarde']].copy()
            export_df.columns = ['Huishoudtype', 'Gemeente', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Huishoudtype', 'Gemeente'])

            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporteer data als CSV",
                data=csv_data,
                file_name=f"huishoudtypen_{selected_income_pct}pct.csv",
                mime="text/csv"
            )

        if selected_fr == 1:
            formeel = "**formele**"
        elif selected_fr == 2:
            formeel = "**informele**"
        elif selected_fr == 3:
            formeel = ""

        if selected_cav == 0:
            gemeentepolis = "**buiten beschouwing** gelaten"
        elif selected_cav == 1:
            gemeentepolis = "**hierin meegenomen**"

        if selected_referteperiode == 0:
            reftext = "sinds kort"
        else:
            reftext = f"al {selected_referteperiode} jaar"
        
        st.markdown(f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} gemeentelijke regelingen waarop een voorbeeldhuishouden recht heeft, dat **{reftext}** een inkomen heeft van **{selected_income_pct}%** van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")

    # ----------------------------------------------------------------------------
    # Graph 2: Income Progression
    # ----------------------------------------------------------------------------
    with tab2:
        fig_income = create_income_figure(
            df,
            selected_huishouden,
            selected_income_pct,
            selected_income,
            selected_referteperiode,
            selected_cav,
            selected_fr,
            selected_gemeente,
            gemeente_labels,
            household_labels,
            key=data_key
        )

        st.plotly_chart(fig_income, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        # Export data to CSV - only show if ?export=1 in URL
        if show_export:
            income_markers_df = get_income_progression_data(
                df, selected_huishouden, selected_income_pct, selected_referteperiode,
                selected_cav, selected_fr, key=data_key
            )
            income_markers_df['Gemeentenaam'] = income_markers_df['Gemeente'].map(gemeente_labels)
            income_markers_df['Inkomensniveau'] = (income_markers_df['Inkomen'] * 100).round(0).astype(int).astype(str) + '%'

            export_df = income_markers_df[['Inkomensniveau', 'Gemeentenaam', 'Waarde']].copy()
            export_df.columns = ['Inkomensniveau', 'Gemeente', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Gemeente', 'Inkomensniveau'])

            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporteer data als CSV",
                data=csv_data,
                file_name=f"inkomensgroepen_{household_labels[selected_huishouden].replace(' ', '_')}.csv",
                mime="text/csv"
            )

        st.markdown(f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} gemeentelijke regelingen waarop een **{household_labels[selected_huishouden].lower()}** recht heeft, met **{reftext}** een bepaald inkomensniveau. De korting op de gemeentepolis is {gemeentepolis}.")
    # ----------------------------------------------------------------------------
    # Graph 3: Formal vs Informal
    # ----------------------------------------------------------------------------
    with tab3:
        fig_bar = create_formal_informal_figure(
            df,
            selected_huishouden,
            selected_income,
            selected_income_pct,
            selected_referteperiode,
            selected_cav,
            selected_gemeente,
            gemeente_labels,
            household_labels,
            key=data_key
        )

        st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        # Export data to CSV - only show if ?export=1 in URL
        if show_export:
            bar_data = get_formal_informal_data(
                df, selected_huishouden, selected_income, selected_referteperiode,
                selected_cav, key=data_key
            )
            bar_data['Gemeentenaam'] = bar_data['Gemeente'].map(gemeente_labels)

            # Melt to long format for export
            export_df = bar_data[['Gemeentenaam', 'Formeel', 'Informeel']].melt(
                id_vars=['Gemeentenaam'],
                value_vars=['Formeel', 'Informeel'],
                var_name='Type',
                value_name='Waarde (€ per maand)'
            )
            export_df.columns = ['Gemeente', 'Type', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Type', 'Gemeente'])

            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporteer data als CSV",
                data=csv_data,
                file_name=f"formeel_informeel_{household_labels[selected_huishouden].replace(' ', '_')}_{selected_income_pct}pct.csv",
                mime="text/csv"
            )

        st.markdown(f"Schatting van de totale waarde (in € per maand) van zowel de formele als informele gemeentelijke regelingen waarop een **{household_labels[selected_huishouden].lower()}** recht heeft, met **{reftext}** een inkomen van **{selected_income_pct}%** van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")
    # ----------------------------------------------------------------------------
    # Graph 4: Population vs Income Threshold
    # ----------------------------------------------------------------------------
    with tab4:
        fig_threshold = create_threshold_figure(
            df,
            selected_huishouden,
            selected_referteperiode,
            selected_cav,
            selected_fr,
            selected_gemeente,
            gemeente_labels,
            household_labels,
            key=data_key
        )

        st.plotly_chart(fig_threshold, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        # Export data to CSV - only show if ?export=1 in URL
        if show_export:
            threshold_data = get_threshold_data(
                df, selected_huishouden, selected_referteperiode, selected_cav,
                selected_fr, key=data_key
            )
            threshold_data['Gemeentenaam'] = threshold_data['Gemeente'].map(gemeente_labels)
            threshold_data['Inkomensgrens_Pct'] = (threshold_data['Inkomensgrens'] * 100).round(0).astype(int)

            export_df = threshold_data[['Gemeentenaam', 'Inkomensgrens_Pct', 'Waarde', 'Inwoners']].copy()
            export_df.columns = ['Gemeente', 'Gewogen gemiddelde inkomensgrens (%)', 'Waarde bij 100% (€ per maand)', 'Inwoners']
            export_df = export_df.sort_values('Gemeente')

            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporteer data als CSV",
                data=csv_data,
                file_name=f"waarden_inkomensgrenzen_{household_labels[selected_huishouden].replace(' ', '_')}.csv",
                mime="text/csv"
            )

        st.markdown(f"Schatting van de totale waarde (in € per maand) en de gemiddelde inkomensgrens van **alle** {formeel} gemeentelijke regelingen waarop een **{household_labels[selected_huishouden].lower()}** recht heeft, met **{reftext}** een inkomen van 100% van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")
    # ----------------------------------------------------------------------------
    # Regulations Table (shown on the right side)
    # ----------------------------------------------------------------------------
    with table_col:
        # Get table data with all regulations (matching and non-matching)
        display_df = regelingen_lijst(
            df, selected_gemeente, selected_huishouden,
            selected_income, selected_referteperiode, selected_cav, selected_fr
        )

        if not display_df.empty:
            # Split into matching and non-matching regulations
            matching_df = display_df[display_df['Matches'] == True].copy()
            non_matching_df = display_df[display_df['Matches'] == False].copy()

            # Find the maximum value to determine padding width
            max_value = display_df['Waarde'].max() or 0
            # Calculate width needed for the largest number (without € sign)
            if max_value > 0:
                max_formatted = f"{max_value:,.0f}".replace(',', '.')
                pad_width = len(max_formatted)
            else:
                pad_width = 3

            # Format currency for Waarde column with fixed-width padding
            def pad_currency(x):
                if pd.notna(x) and x is not None:
                    if x == 0:
                        return "€ " + "?".rjust(pad_width)
                    # Format the number part and pad it
                    num_str = f"{x:,.0f}".replace(',', '.')
                    return "€ " + num_str.rjust(pad_width)
                return "€ " + "?".rjust(pad_width)

            # Format percentage for Inkomensgrens column
            def format_percentage(x):
                if pd.notna(x) and x is not None and x > 0:
                    return f"{int(x * 100)}%"
                return "? %"

            # =================================================================
            # Table 1: Matching Regulations
            # =================================================================
            if not matching_df.empty:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                # Format the dataframe
                formatted_matching = matching_df.copy()
                formatted_matching['Waarde'] = formatted_matching['Waarde'].apply(pad_currency)
                formatted_matching['Inkomensgrens'] = formatted_matching['Inkomensgrens'].apply(format_percentage)

                # Select columns for display
                formatted_matching = formatted_matching[['Regeling', 'Waarde', 'Inkomensgrens']]

                # Calculate height to show all rows (header: 38px + rows: 35px each + padding: 10px)
                table_height = 38 + (len(formatted_matching) * 35) + 10

                # Display table
                st.dataframe(
                    formatted_matching,
                    width="stretch",
                    height=table_height,
                    hide_index=True,
                    column_config={
                        "Regeling": st.column_config.TextColumn("Regelingen", width=200),
                        "Waarde": st.column_config.TextColumn("Waarde", width=40),
                        "Inkomensgrens": st.column_config.TextColumn("Grens", width=30)
                    }
                )
                # Calculate total value of matching regulations
                total_waarde = matching_df['Waarde'].sum() if not matching_df.empty else 0
                total_waarde_formatted = format_dutch_currency(total_waarde, decimals=0)

                st.markdown(f"Bovenstaande {formeel} regelingen voor een **{household_labels[selected_huishouden].lower()}** in **{gemeente_labels[selected_gemeente]}** met **{reftext}** een inkomen van **{selected_income_pct}%** van het sociaal minimum tellen op tot **{total_waarde_formatted}** per maand.", unsafe_allow_html=True)
            
            else:
                st.info("Let op! Geen passende regelingen gevonden.")
            
            st.markdown("De gemeente kent ook nog de onderstaande regelingen, die mogelijk niet van toepassing zijn of waarvan de waarde niet goed te bepalen was.")
            
            # =================================================================
            # Table 2: Non-Matching Regulations
            # =================================================================
            if not non_matching_df.empty:
                # Format the dataframe
                formatted_non_matching = non_matching_df.copy()
                formatted_non_matching['Waarde'] = formatted_non_matching['Waarde'].apply(pad_currency)
                formatted_non_matching['Inkomensgrens'] = formatted_non_matching['Inkomensgrens'].apply(format_percentage)

                # Select columns for display
                formatted_non_matching = formatted_non_matching[['Regeling', 'Waarde', 'Inkomensgrens']]

                # Apply gray color to all text
                styled_non_matching = formatted_non_matching.style\
                    .apply(lambda x: [f'color: {COLOR_OTHER}'] * len(x), axis=1)

                # Display table
                st.dataframe(
                    styled_non_matching,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Regeling": st.column_config.TextColumn("Regelingen", width=210),
                        "Waarde": st.column_config.TextColumn("Waarde", width=40),
                        "Inkomensgrens": st.column_config.TextColumn("Grens", width=30)
                    }
                )
            else:
                st.info("Geen overige regelingen gevonden.")
        else:
            st.info(f"Geen regelingen gevonden voor {gemeente_labels[selected_gemeente]}")

except FileNotFoundError:
    st.error("Databestand 'dataoverzicht_dashboard_armoedebeleid.xlsx' niet gevonden.")
except Exception as e:
    import traceback
    st.error(f"Er is een fout opgetreden: {type(e).__name__}: {str(e)}")
    st.code(traceback.format_exc())
