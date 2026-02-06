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
    """Load all required data from Parquet file and filter by municipality access"""
    # Support both Streamlit secrets.toml and environment variables (for Cloud Run)
    try:
        data_url = st.secrets["excel_url"]
        key_all = st.secrets["key_all"]
        key_barneveld = st.secrets["key_barneveld"]
        key_delft = st.secrets["key_delft"]
    except (AttributeError, KeyError, FileNotFoundError):
        # Fall back to environment variables
        data_url = os.getenv("EXCEL_URL", "dataset.parquet")
        key_all = os.getenv("KEY_ALL", "")
        key_barneveld = os.getenv("KEY_BARNEVELD", "")
        key_delft = os.getenv("KEY_DELFT", "")

    # Load data from Parquet format
    df = pd.read_parquet(data_url)

    if key == key_all:
        return df[df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '')]
    elif key == key_delft:
        excluded_municipalities = ['Barneveld']
    elif key == key_barneveld:
        excluded_municipalities = ['Delft']
    else:
        excluded_municipalities = ['Barneveld', 'Delft']

    return df[df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '') & ~df['Gemeentenaam'].isin(excluded_municipalities)]


def format_dutch_currency(value, decimals=0):
    """Format number as Dutch currency (dot for thousands, comma for decimals)."""
    formatted = f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"€ {formatted}"


def filter_regelingen(df, gm, hh, ink=1, refper=0, cav=0, fr=3):
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
    gmcodes : str or list
        Municipality code(s). Can be a single code or list of codes.
    """

    wrd_col, ig_col, ref_col = f'WRD_{hh}', f'IG_{hh}', f'Referteperiode_{hh}'

    mask = (df['WB'] == 1) & (df[ig_col] >= ink) & (df[ref_col] <= refper)

    # Ensure gmcodes is list-like for .isin()
    if isinstance(gm, str):
        gm = [gm]
    mask &= df['GMcode'].isin(gm)

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
# FUNCTIONS FOR GRAPH 1
# ================================================================================

def huishoudtypen_data(df, gm_lbl, hh_lbl, ink=1, refper=0, cav=0, fr=3):
    """Calculate values for all municipalities and all household types (Graph 1)."""
    results = []
    
    for hh in hh_lbl.keys():
        # Filter and calculate values and reindex to include all municipalities
        filtered_df = (
            filter_regelingen(df, gm_lbl.keys(), hh, ink, refper, cav, fr)
            .groupby('GMcode')['WRD'].sum().div(12)
            .reindex(gm_lbl.keys(), fill_value=0.0)
        )

        # Create dataframe
        hh_data = pd.DataFrame({
            'GMcode': gm_lbl.keys(),
            'Gemeentenaam': gm_lbl.values(),
            'Waarde': filtered_df.values,
            'Huishouden': hh
        })

        results.append(hh_data[['GMcode', 'Gemeentenaam', 'Waarde', 'Huishouden']])

    huishoudtypen_df = pd.concat(results, ignore_index=True)

    # Add hover text for all data
    huishoudtypen_df['hover_text'] = (
        "<b>" + huishoudtypen_df['Gemeentenaam'].astype(str) + "</b><br>" +
        f"{int(ink * 100)}% sociaal minimum<br>Waarde: " +
        huishoudtypen_df['Waarde'].apply(format_dutch_currency).astype(str)
    )

    return huishoudtypen_df

def huishoudtypen_grafiek(df, sel_gm, gm_lbl, hh_lbl, ink=1, refper=0, cav=0, fr=3):
    """Create box plot figure for household comparison (Graph 1)."""

    # Get data
    huishoudtypen_df = huishoudtypen_data(df, gm_lbl, hh_lbl, ink, refper, cav, fr)


    # Create multi-line x-axis labels with specific breaks
    label_mapping = {
        'Alleenstaande': 'Alleenstaande',
        'Alleenstaande ouder met kind': 'Alleenstaande<br>ouder met kind',
        'Paar': 'Paar',
        'Paar met twee kinderen': 'Paar met<br>twee kinderen'
    }
    hh_labels = list(hh_lbl.values())
    x_labels = [label_mapping.get(label, label) for label in hh_labels]

    huishoudtypen_fig = go.Figure()

    for hh_code, hh_label in hh_lbl.items():
        household_data = huishoudtypen_df[huishoudtypen_df['Huishouden'] == hh_code]

        # Create both traces with their specific styling (Box plots need separate traces for different colors)
        trace_configs = [
            {'data': household_data[household_data['GMcode'] != sel_gm], 'size': 8, 'color': COLOR_OTHER, 'opacity': 0.6},
            {'data': household_data[household_data['GMcode'] == sel_gm], 'size': 10, 'color': COLOR_SELECTED, 'opacity': 1.0}
        ]

        for config in trace_configs:
            data = config['data']
            if len(data) > 0:
                huishoudtypen_fig.add_trace(go.Box(
                    x=[hh_label] * len(data),
                    y=data['Waarde'],
                    name=hh_label,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(
                        size=config['size'],
                        color=config['color'],
                        opacity=config['opacity']
                    ),
                    hovertext=data['hover_text'],
                    hoverinfo='text',
                    customdata=data['GMcode'].values,
                    showlegend=False,
                    fillcolor='rgba(255,255,255,0)',
                    line=dict(color='rgba(255,255,255,0)')
                ))

    # Add value annotations for selected municipality
    selected_municipality_data = huishoudtypen_df[huishoudtypen_df['GMcode'] == sel_gm]
    for _, row in selected_municipality_data.iterrows():
        huishoudtypen_fig.add_annotation(
            x=hh_lbl[row['Huishouden']],
            y=row['Waarde'],
            text=format_dutch_currency(row['Waarde']),
            showarrow=False,
            xanchor='left',
            xshift=20,
            yshift=0,
            font=dict(size=14, color='black')
        )

    huishoudtypen_fig.update_layout(
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
            tickvals=hh_labels,
            ticktext=x_labels,
            fixedrange=True
        )
    )

    return huishoudtypen_fig


# ================================================================================
# FUNCTIONS FOR GRAPH 2
# ================================================================================

def inkomensgroepen_data(df, hh, gm_lbl, ink_pct=100, refper=0, cav=0, fr=3):
    """Calculate values for all municipalities at specific income levels (Graph 2 markers)."""
    # Calculate income levels to show based on selected income percentage (100-200)
    # Show 6 levels in steps of 10, centered around selected income (start 20 below)
    final_digit = ink_pct % 10  # Preserve digit alignment (e.g., 105, 115, 125 for ink_pct=115)
    start = ink_pct - 20
    start = max(100 + final_digit, min(140 + final_digit, start))  # Clamp to 100-200 range
    income_levels_to_show = [level for level in range(start, start + 60, 10)]

    results = []

    for ink_lvl in income_levels_to_show:
        # Filter and calculate values and reindex to include all municipalities
        filtered_df = (
            filter_regelingen(df, gm_lbl.keys(), hh, round(ink_lvl / 100, 2), refper, cav, fr)
            .groupby('GMcode')['WRD'].sum().div(12)
            .reindex(gm_lbl.keys(), fill_value=0.0))

        # Create dataframe
        ink_data = pd.DataFrame({
            'GMcode': gm_lbl.keys(),
            'Gemeentenaam': gm_lbl.values(),
            'Waarde': filtered_df.values,
            'Inkomen': ink_lvl
        })

        results.append(ink_data[['GMcode', 'Gemeentenaam', 'Waarde', 'Inkomen']])

    inkomensgroepen_df = pd.concat(results, ignore_index=True)

    # Create hover text for all municipalities
    inkomensgroepen_df['hover_text'] = (
        "<b>" + inkomensgroepen_df['Gemeentenaam'].astype(str) + "</b><br>" +
        hh_lbl[hh] + "<br>Waarde: " +
        inkomensgroepen_df['Waarde'].apply(format_dutch_currency).astype(str)
    )

    return inkomensgroepen_df

@st.cache_data
def inkomenslijn_data(_df, gm, hh, refper=0, cav=0, fr=3):
    """Calculate values for selected municipality at all income levels (Graph 2 line)."""
    all_income_levels_pct = range(100, 201)
    results = []
    for income_pct in all_income_levels_pct:
        filtered_df = filter_regelingen(_df, gm, hh, income_pct / 100, refper, cav, fr)

        results.append({
            'Inkomen': income_pct,  # Store as percentage (integer)
            'Waarde': filtered_df['WRD'].sum() / 12
        })

    return pd.DataFrame(results)


def inkomensgroepen_grafiek(df, sel_gm, hh, hh_lbl, gm_lbl, ink_pct=100, refper=0, cav=0, fr=3):
    """Create line chart figure for income progression (Graph 2)."""
    inkomensgroepen_fig = go.Figure()

    # Get cached data for all municipalities at specific income levels
    inkomensgroepen_df = inkomensgroepen_data(
        df, hh, gm_lbl, ink_pct, refper, cav, fr
    )

    # Extract income levels from data for x-axis ticks
    income_levels_to_show = sorted(inkomensgroepen_df['Inkomen'].unique())

    # Add visual properties based on whether municipality is selected
    is_selected = inkomensgroepen_df['GMcode'] == sel_gm
    inkomensgroepen_df['marker_color'] = is_selected.map({True: COLOR_SELECTED, False: COLOR_OTHER})
    inkomensgroepen_df['marker_size'] = is_selected.map({True: 10, False: 8})
    inkomensgroepen_df['marker_opacity'] = is_selected.map({True: 1.0, False: 0.6})

    inkomensgroepen_fig.add_trace(go.Scatter(
        x=inkomensgroepen_df['Inkomen'],
        y=inkomensgroepen_df['Waarde'],
        mode='markers',
        name='Gemeenten',
        marker=dict(
            size=inkomensgroepen_df['marker_size'],
            color=inkomensgroepen_df['marker_color'],
            opacity=inkomensgroepen_df['marker_opacity']
        ),
        hovertext=inkomensgroepen_df['hover_text'],
        hoverinfo='text',
        showlegend=False
    ))

    # Get cached line data for selected municipality
    selected_all_df = inkomenslijn_data(
        df, sel_gm, hh, refper, cav, fr
    )

    inkomensgroepen_fig.add_trace(go.Scatter(
        x=selected_all_df['Inkomen'],
        y=selected_all_df['Waarde'],
        mode='lines',
        line=dict(
            color=COLOR_SELECTED,
            width=2
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add label for the selected income level
    value = selected_all_df.loc[selected_all_df['Inkomen'] == ink_pct]['Waarde'].squeeze()
    inkomensgroepen_fig.add_annotation(
        x=ink_pct,
        y=value,
        text=format_dutch_currency(value),
        showarrow=False,
        xanchor='center',
        xshift=0,
        yshift=15,
        font=dict(size=14, color='black')
    )

    # Create tick values matching income_levels_to_show
    tick_vals = income_levels_to_show
    tick_text = [f'{val}%' for val in tick_vals]

    inkomensgroepen_fig.update_layout(
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

    return inkomensgroepen_fig


# ================================================================================
# FUNCTIONS FOR GRAPH 3
# ================================================================================

def in_formeel_data(df, hh, gm_lbl, ink=1, refper=0, cav=0):
    """Calculate formal and informal values for all municipalities (Graph 3)."""
    # Group and aggregate WRD only (no need to aggregate Gemeentenaam)
    filtered_df = (
        filter_regelingen(df, gm_lbl.keys(), hh, ink, refper, cav)
        .groupby(['GMcode', 'FR'])['WRD'].sum().div(12)
    )

    # Unstack FR to get Formeel/Informeel columns
    wrd_unstacked = filtered_df.unstack('FR', fill_value=0)

    # Rename columns
    wrd_unstacked = wrd_unstacked.rename(columns={'Ja': 'Formeel', 'Nee': 'Informeel'})

    # Ensure both columns exist
    for col in ['Formeel', 'Informeel']:
        if col not in wrd_unstacked.columns:
            wrd_unstacked[col] = 0

    # Reindex to include all municipalities and add gemeentenaam directly from gm_lbl
    wrd_unstacked = wrd_unstacked.reindex(gm_lbl.keys(), fill_value=0)
    wrd_unstacked['Gemeentenaam'] = gm_lbl.values()
    wrd_unstacked['Totaal'] = wrd_unstacked['Formeel'] + wrd_unstacked['Informeel']

    # Reset index
    result = wrd_unstacked.reset_index()

    return result[['GMcode', 'Gemeentenaam', 'Formeel', 'Informeel', 'Totaal']]

def in_formeel_grafiek(df, sel_gm, hh, hh_lbl, gm_lbl, ink_pct=100, refper=0, cav=0):
    """Create stacked bar chart for formal vs informal regulations (Graph 3)."""
    # Get cached formal/informal data
    bar_data = in_formeel_data(
        df, hh, gm_lbl, ink_pct / 100, refper, cav
    ).sort_values('Formeel', ascending=False)

    # Add visual properties and hover text to dataframe
    is_selected = bar_data['GMcode'] == sel_gm
    bar_data['color_formal'] = is_selected.map({True: COLOR_SELECTED, False: COLOR_OTHER})
    bar_data['color_informal'] = is_selected.map({True: COLOR_INFORMAL_SELECTED, False: COLOR_INFORMAL_OTHER})

    hover_base = f"{hh_lbl[hh]}<br>{ink_pct}% sociaal minimum<br>"
    bar_data['hover_formal'] = bar_data['Formeel'].apply(
        lambda val: f"{hover_base}Waarde formele regelingen: {format_dutch_currency(val)}<extra></extra>"
    )
    bar_data['hover_informal'] = bar_data['Informeel'].apply(
        lambda val: f"{hover_base}Waarde informele regelingen: {format_dutch_currency(val)}<extra></extra>"
    )

    # Create figure with traces
    fig_bar = go.Figure([
        go.Bar(
            x=bar_data['Gemeentenaam'],
            y=bar_data['Formeel'],
            name='Formeel',
            marker_color=bar_data['color_formal'],
            hovertemplate=bar_data['hover_formal']
        ),
        go.Bar(
            x=bar_data['Gemeentenaam'],
            y=bar_data['Informeel'],
            name='Informeel',
            marker_color=bar_data['color_informal'],
            hovertemplate=bar_data['hover_informal']
        )
    ])

    fig_bar.update_layout(
        barmode='stack',
        xaxis_title="",
        yaxis_title="",
        height=450,
        showlegend=True,
        dragmode=False,
        margin=dict(t=0, b=100, l=50, r=20, autoexpand=False),
        legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="right", x=0.99),
        yaxis=dict(tickprefix="€ ", tickfont=dict(size=14), fixedrange=True),
        xaxis=dict(tickfont=dict(size=14), tickangle=-45, fixedrange=True)
    )

    return fig_bar


# ================================================================================
# FUNCTIONS FOR GRAPH 4
# ================================================================================

def gem_inkomensgrenzen_data(df, gm, hh, refper=0, cav=0, fr=3):
    """Calculate weighted income thresholds and values for all municipalities (Graph 4)."""
    # Filter once for all municipalities (instead of once per municipality)
    filtered_df = filter_regelingen(df, gm, hh, 1, refper, cav, fr)

    # Add calculated columns
    filtered_df = filtered_df.assign(
        monthly_wrd=filtered_df['WRD'] / 12,
        weighted_component=filtered_df['WRD'] * (filtered_df['IG'] - 1) / 12
    )

    # Group by municipality and aggregate
    result = filtered_df.groupby('GMcode').agg({
        'monthly_wrd': 'sum',
        'weighted_component': 'sum',
        'Inwoners': 'first',
        'Gemeentenaam': 'first'
    }).reset_index()

    result.columns = ['Gemeente', 'Waarde', 'weighted_sum', 'Inwoners', 'Gemeentenaam']

    # Filter out zero values and calculate income threshold
    result['Inkomensgrens'] = ((1 + (result['weighted_sum'] / result['Waarde']))*100).astype(int)

    return result[['Gemeente', 'Gemeentenaam', 'Inkomensgrens', 'Waarde', 'Inwoners']]

def gem_inkomensgrenzen_grafiek(df, sel_gm, all_gm, hh, hh_lbl, refper=0, cav=0, fr=3):
    """Create scatter plot for value vs income threshold (Graph 4)."""
    # Get cached threshold data
    threshold_data = gem_inkomensgrenzen_data(
        df, all_gm, hh, refper, cav, fr
    )

    fig_threshold = go.Figure()

    if len(threshold_data) > 0:
        # Add visual properties based on whether municipality is selected
        is_selected = threshold_data['Gemeente'] == sel_gm
        threshold_data['marker_color'] = is_selected.map({True: COLOR_SELECTED, False: COLOR_OTHER})
        threshold_data['marker_opacity'] = is_selected.map({True: 1.0, False: 0.6})

        # Format numbers for hover text
        threshold_data['ig_pct'] = threshold_data['Inkomensgrens'].astype(str)
        threshold_data['waarde_fmt'] = threshold_data['Waarde'].apply(format_dutch_currency)
        threshold_data['inwoners_fmt'] = threshold_data['Inwoners'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))

        # Create hover text
        threshold_data['hover_text'] = (
            "<b>" + threshold_data['Gemeentenaam'].astype(str) + "</b><br>" +
            hh_lbl[hh] + "<br>Gemiddelde Inkomensgrens: " +
            threshold_data['ig_pct'] + "%<br>" +
            "Waarde bij 100% sociaal minimum: " + threshold_data['waarde_fmt'] + "<br>" +
            "Inwoners: " + threshold_data['inwoners_fmt']
        )

        # Add single scatter trace with all municipalities
        fig_threshold.add_trace(go.Scatter(
            x=threshold_data['Inkomensgrens'],
            y=threshold_data['Waarde'],
            mode='markers',
            marker=dict(
                size=threshold_data['Inwoners'] / 200,
                color=threshold_data['marker_color'],
                opacity=threshold_data['marker_opacity'],
                sizemode='area'
            ),
            hovertext=threshold_data['hover_text'],
            hoverinfo='text',
            showlegend=False
        ))

        # Add label for selected municipality
        selected_data = threshold_data[is_selected]
        for _, row in selected_data.iterrows():
            fig_threshold.add_annotation(
                x=row['Inkomensgrens'],
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
# FUNCTIONS FOR TABLES
# ================================================================================

def regelingen_lijst(df, gm, hh, ink=1, refper=0, cav=0, fr=3):
    # Get all regulations for this municipality
    regs = df[df['GMcode'] == gm].copy()

    # Get regulation IDs that meet user input criteria
    selected_regs = filter_regelingen(df, gm, hh, ink, refper, cav, fr)

    # Add computed columns
    regs['Regeling'] = regs['N4']
    regs['Waarde'] = regs[f'WRD_{hh}'] / 12
    regs['Inkomensgrens'] = regs[f'IG_{hh}']
    regs['Matches'] = regs['ID'].isin(selected_regs['ID'])

    # Group and aggregate
    grouped = regs.groupby(['Regeling', 'Matches'], as_index=False).agg({
        'Waarde': 'sum',
        'Inkomensgrens': 'min'
    })

    # Sort and combine: matching by value (desc), non-matching alphabetically
    matching = grouped[grouped['Matches']].sort_values('Waarde', ascending=False)
    non_matching = grouped[~grouped['Matches']].sort_values('Regeling')

    return pd.concat([matching, non_matching])


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
    hh_lbl = {
        'HH01': 'Alleenstaande',
        'HH02': 'Alleenstaande ouder met kind',
        'HH03': 'Paar',
        'HH04': 'Paar met twee kinderen'
    }

    # ----------------------------------------------------------------------------
    # Header and Gemeente Labels Preparation
    # ----------------------------------------------------------------------------
    st.title("Dashboard armoedebeleid", text_alignment="center", anchor=False)

    # Prepare gemeente labels before tabs (more efficient with set_index)
    gm_lbl = (df[['GMcode', 'Gemeentenaam']]
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
    default_income      = int(params.get("ink", 100))
    if default_income not in range(100, 200):
        default_income = 100

    default_huishouden  = params.get("hh", list(hh_lbl.keys())[0])
    if default_huishouden not in hh_lbl:
        default_huishouden = list(hh_lbl.keys())[0]

    default_refper      = int(params.get("ref", 0))
    if default_refper not in [0, 1, 3, 5]:
        default_refper = 0

    default_cav         = params.get("cav", "0") == "1"

    # Ensure gemeente exists in the data (handles invalid codes or API key filtering)
    default_gemeente = params.get("gm", list(gm_lbl.keys())[0])
    if default_gemeente not in gm_lbl:
        default_gemeente = list(gm_lbl.keys())[0]
    
    # URL format: reg=1 (Formeel), reg=2 (Informeel), reg=3 (both)
    default_fr = int(params.get("reg", 3))
    if default_fr == 1:
        default_reg_types = ["Formeel"]
    elif default_fr == 2:
        default_reg_types = ["Informeel"]
    else:
        default_reg_types = ["Formeel", "Informeel"]

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
            help="Als percentage van het sociale minimum (bijstandsniveau)"
        )

        sel_gm = st.selectbox(
            label="Gemeente",
            options=gm_lbl,
            format_func=lambda x: gm_lbl[x],
            index=list(gm_lbl).index(default_gemeente),
            key="gemeente",
            help="Welke gemeente moet worden uitgelicht in het dashboard?"
        )

        # Ensure default huishouden is valid
        if default_huishouden not in hh_lbl:
            default_huishouden = "HH04"

        sel_hh = st.selectbox(
            label="Huishouden:",
            options=hh_lbl,
            format_func=lambda x: hh_lbl[x],
            index=list(hh_lbl).index(default_huishouden),
            key="huishouden",
            help="Welk huishoudtype wilt u meer over weten?"
        )

        sel_refper = st.segmented_control(
            label="Jaren met laag inkomen",
            options=[0, 1, 3, 5],
            default=default_refper,
            key="referteperiode",
            help="Jaren dat het voorbeeldhuishouden al het geselecteerde inkomen heeft (referteperiode)"
        )

        # Initialize session state if it doesn't exist - no rerun needed
        if "reg_types" not in st.session_state:
            st.session_state.reg_types = default_reg_types if default_reg_types else ["Formeel", "Informeel"]
        if "prev_reg_types" not in st.session_state:
            st.session_state.prev_reg_types = st.session_state.reg_types

        # Check before rendering: if current value is empty, fix it based on previous selection
        if not st.session_state.reg_types:
            if len(st.session_state.prev_reg_types) == 1:
                # If only one was selected, switch to the other one
                if "Formeel" in st.session_state.prev_reg_types:
                    st.session_state.reg_types = ["Informeel"]
                else:
                    st.session_state.reg_types = ["Formeel"]
            else:
                # Default to both if we can't determine
                st.session_state.reg_types = ["Formeel", "Informeel"]

        # Regulation type values: Formeel=1, Informeel=2, both=3
        reg_type_values = {"Formeel": 1, "Informeel": 2}
        selected_reg_types = st.segmented_control(
            "Type regelingen",
            options=list(reg_type_values.keys()),
            selection_mode="multi",
            key="reg_types",
            help="Selecteer welke regelingen worden meegenomen: formele, informele of beide"
        )

        # Update prev_reg_types after the widget (with the current, possibly corrected value)
        st.session_state.prev_reg_types = st.session_state.reg_types

        toggle_cav = st.toggle("Korting gemeentepolis", value=default_cav, key="toggle_cav", help="Moet de korting op een eventuele geemeentepolis worden meegenomen op het totaalbedrag? (Deze polis is niet voor ieder huishouden voordelig)")

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


    # Update URL with current selections (preserving the key parameter)
    # Only update if params actually changed to prevent unnecessary reruns
    new_params = {"key": params.get("key", "")} if params.get("key") else {}
    new_params["ink"] = str(sel_ink_pct)
    new_params["gm"] = sel_gm
    new_params["hh"] = sel_hh
    new_params["ref"] = str(sel_refper)
    # Calculate selected_fr and CAV for URL
    sel_fr = sum(reg_type_values.get(rt, 0) for rt in selected_reg_types) or 3
    sel_cav = 1 if toggle_cav else 0
    new_params["reg"] = str(sel_fr)
    new_params["cav"] = str(sel_cav)
    # Only update query params if they changed
    if dict(st.query_params) != new_params:
        st.query_params.update(new_params)

    # Create two-column layout: graphs on left, table on right
    graph_col, table_col = st.columns([2, 1])

    # Check if export buttons should be shown (only if ?export=1 in URL)
    show_export = st.query_params.get("export") == "1"

    # make specific text inserts based on current selections
    if sel_fr == 1:
        formeel = "**formele**"
    elif sel_fr == 2:
        formeel = "**informele**"
    elif sel_fr == 3:
        formeel = ""

    if sel_cav == 0:
        gemeentepolis = "**buiten beschouwing** gelaten"
    elif sel_cav == 1:
        gemeentepolis = "**hierin meegenomen**"

    if sel_refper == 0:
        reftext = "sinds kort"
    else:
        reftext = f"al {sel_refper} jaar"

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
        fig = huishoudtypen_grafiek(
            df, sel_gm, gm_lbl, hh_lbl, sel_ink_pct / 100, sel_refper, sel_cav, sel_fr
        )

        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})
        
        st.markdown(f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} gemeentelijke regelingen waarop een voorbeeldhuishouden recht heeft, dat **{reftext}** een inkomen heeft van **{sel_ink_pct}%** van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")

    # ----------------------------------------------------------------------------
    # Graph 2: Income Progression
    # ----------------------------------------------------------------------------
    with tab2:

        fig_income = inkomensgroepen_grafiek(
            df, sel_gm, sel_hh, hh_lbl, gm_lbl, sel_ink_pct, sel_refper, sel_cav, sel_fr
        )

        st.plotly_chart(fig_income, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        st.markdown(f"Schatting van de totale waarde (in € per maand) van **alle** {formeel} gemeentelijke regelingen waarop een **{hh_lbl[sel_hh].lower()}** recht heeft, met **{reftext}** een bepaald inkomensniveau. De korting op de gemeentepolis is {gemeentepolis}.")

    # ----------------------------------------------------------------------------
    # Graph 3: Formal vs Informal
    # ----------------------------------------------------------------------------
    with tab3:
        fig_bar = in_formeel_grafiek(
            df, sel_gm, sel_hh, hh_lbl, gm_lbl, sel_ink_pct, sel_refper, sel_cav
        )

        st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        st.markdown(f"Schatting van de totale waarde (in € per maand) van zowel de formele als informele gemeentelijke regelingen waarop een **{hh_lbl[sel_hh].lower()}** recht heeft, met **{reftext}** een inkomen van **{sel_ink_pct}%** van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")
    # ----------------------------------------------------------------------------
    # Graph 4: Population vs Income Threshold
    # ----------------------------------------------------------------------------
    with tab4:

        fig_threshold = gem_inkomensgrenzen_grafiek(
            df, sel_gm, list(gm_lbl.keys()), sel_hh, hh_lbl, sel_refper, sel_cav, sel_fr
        )

        st.plotly_chart(fig_threshold, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

        st.markdown(f"Schatting van de totale waarde (in € per maand) en de gemiddelde inkomensgrens van **alle** {formeel} gemeentelijke regelingen waarop een **{hh_lbl[sel_hh].lower()}** recht heeft, met **{reftext}** een inkomen van 100% van het sociaal minimum. De korting op de gemeentepolis is {gemeentepolis}.")

    # ----------------------------------------------------------------------------
    # Export Buttons (shown below tabs if ?export=1 in URL)
    # ----------------------------------------------------------------------------
    if show_export:
        st.markdown("---")
        st.subheader("📥 Exporteer data", anchor=False)

        # Create 4 columns for the export buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Export 1: Huishoudtypen
            plot_df = huishoudtypen_data(df, gm_lbl, hh_lbl, sel_ink_pct / 100, sel_refper, sel_cav, sel_fr)
            export_df = plot_df[['Huishouden', 'Gemeentenaam', 'Waarde']].copy()
            export_df['Huishoudtype'] = export_df['Huishouden'].map(hh_lbl)
            export_df = export_df[['Huishoudtype', 'Gemeentenaam', 'Waarde']]
            export_df.columns = ['Huishoudtype', 'Gemeente', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Huishoudtype', 'Gemeente'])
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Huishoudtypen",
                data=csv_data,
                file_name=f"huishoudtypen_{sel_ink_pct}pct.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Export 2: Inkomensgroepen
            income_markers_df = inkomensgroepen_data(
                df, sel_hh, gm_lbl, sel_ink_pct, sel_refper, sel_cav, sel_fr
            )
            income_markers_df['Inkomensniveau'] = (income_markers_df['Inkomen'] * 100).round(0).astype(int).astype(str) + '%'
            export_df = income_markers_df[['Inkomensniveau', 'Gemeentenaam', 'Waarde']].copy()
            export_df.columns = ['Inkomensniveau', 'Gemeente', 'Waarde (€ per maand)']
            export_df = export_df.sort_values(['Gemeente', 'Inkomensniveau'])
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Inkomensgroepen",
                data=csv_data,
                file_name=f"inkomensgroepen_{hh_lbl[sel_hh].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            # Export 3: (In)formeel
            bar_data = in_formeel_data(
                df, sel_hh, gm_lbl, sel_ink_pct / 100, sel_refper, sel_cav
            )
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
                label="(In)formeel",
                data=csv_data,
                file_name=f"formeel_informeel_{hh_lbl[sel_hh].replace(' ', '_')}_{sel_ink_pct}pct.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col4:
            # Export 4: Inkomensgrenzen
            threshold_data = gem_inkomensgrenzen_data(
                df, list(gm_lbl.keys()), sel_hh, sel_refper, sel_cav,
                sel_fr
            )
            threshold_data['Inkomensgrens_Pct'] = (threshold_data['Inkomensgrens'] * 100).round(0).astype(int)
            export_df = threshold_data[['Gemeentenaam', 'Inkomensgrens_Pct', 'Waarde', 'Inwoners']].copy()
            export_df.columns = ['Gemeente', 'Gewogen gemiddelde inkomensgrens (%)', 'Waarde bij 100% (€ per maand)', 'Inwoners']
            export_df = export_df.sort_values('Gemeente')
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Inkomensgrenzen",
                data=csv_data,
                file_name=f"waarden_inkomensgrenzen_{hh_lbl[sel_hh].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ----------------------------------------------------------------------------
    # Regulations Table (shown on the right side)
    # ----------------------------------------------------------------------------
    with table_col:
        # Get table data with all regulations (matching and non-matching)
        display_df = regelingen_lijst(
            df, sel_gm, sel_hh, sel_ink_pct / 100, sel_refper, sel_cav, sel_fr
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

                st.markdown(f"Bovenstaande {formeel} regelingen voor een **{hh_lbl[sel_hh].lower()}** in **{gm_lbl[sel_gm]}** met **{reftext}** een inkomen van **{sel_ink_pct}%** van het sociaal minimum tellen op tot **{total_waarde_formatted}** per maand.", unsafe_allow_html=True)
            
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
            st.info(f"Geen regelingen gevonden voor {gm_lbl[sel_gm]}")

except FileNotFoundError:
    st.error("Parquet databestand niet gevonden.")
except Exception as e:
    #import traceback
    st.error(f"Er is een fout opgetreden: {type(e).__name__}: {str(e)}")
    #st.code(traceback.format_exc())