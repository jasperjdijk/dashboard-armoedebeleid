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
# HELPER FUNCTIONS
# ================================================================================

@st.cache_data
def load_data(key):
    """Load all required data from Excel file and merge municipality information"""
    # Get Excel URL from Streamlit secrets (keeps data private)
    excel_url = st.secrets["excel_url"]
    excel_file = pd.ExcelFile(excel_url)

    df = pd.read_excel(excel_file, sheet_name="Totaaloverzicht")

    if key == st.secrets["key_all"]:
        return df
    elif key == st.secrets["key_delft"]:
        excluded_municipalities = ['Barneveld']
    elif key == st.secrets["key_barneveld"]:
        excluded_municipalities = ['Delft']
    else:
        excluded_municipalities = ['Barneveld', 'Delft']

    df = df[~df['Gemeentenaam'].isin(excluded_municipalities)]
    return df

def filter_benefits(df, gmcode, hh, ink=1, refper=0, cav=0, result="sum", fr="all", mt="all", wb=1, bt=1):
    """
    Filter and aggregate benefits from Totaaloverzicht based on criteria.

    Parameters:
    -----------
    df : DataFrame
        The Totaaloverzicht dataframe containing all regulation data
    gmcode : str
        Municipality code (e.g., 'GM0363' for Amsterdam)
    hh : str
        Household type code: 'HH01' (single), 'HH02' (single parent),
        'HH03' (couple), or 'HH04' (couple with children)
    ink : float
        Income level as fraction of social minimum (1.0 = 100%, 1.5 = 150%)
    refper : int
        Required years at low income (0-5)
    cav : int
        Include health insurance discount (CAV): 0 = no, 1 = yes
    result : str
        Return type: 'sum' for total value, 'ig' for weighted income threshold,
        'list' for detailed regulation list
    fr : str
        Formal regulation filter: 'Ja', 'Nee', or 'all'
    mt : str
        Means test filter: 0, 1, or 'all'
    wb : int
        Include in calculation (WB): 0 or 1
    bt : int
        Standard benefit (BT): 0 or 1

    Returns:
    --------
    float or list
        If result='sum': Monthly total value (€)
        If result='ig': Weighted sum for income threshold calculation
        If result='list': List of dicts with 'name' and 'amount' keys
    """
    # Determine which columns to use based on household type
    ig_column = f'IG_{hh}'
    wrd_column = f'WRD_{hh}'
    ref_column = f'Referteperiode_{hh}'

    # Start with base conditions
    mask = (df['GMcode'] == gmcode) & (df['WB'] == wb)

    if cav == 1:
        mask &= ((df['BT'] == bt) | (df['CAV'] == cav))
    else:
        # All other cases: exact match
        mask &= (df['BT'] == bt) & (df['CAV'] == cav)

    # Add FR filter only if not "all"
    if fr != "all":
        mask &= (df['FR'] == fr)

    # Add MT filter only if not "all"
    if mt != "all":
        mask &= (df['MT'] == mt)

    # Add income threshold filter: INK must be <= IG_HH (income must be at or below the threshold)
    # Convert to numeric, coercing errors to NaN
    ig_numeric = pd.to_numeric(df[ig_column], errors='coerce')
    mask &= ig_numeric.notna() & (ig_numeric >= ink)

    ref_numeric = pd.to_numeric(df[ref_column], errors='coerce')
    mask &= ref_numeric.notna() & (ref_numeric <= refper)

    filtered = df[mask].copy()

    if result=="ig":
        # Calculate weighted sum for average threshold calculation
        # Sum of WRD × (IG - 1.0) for each regulation
        # Calculate contribution: WRD × (IG - 1.0)
        filtered['contribution'] = filtered[wrd_column] * (filtered[ig_column] - 1.0)
        return filtered['contribution'].sum() / 12

    if result=="sum":
        return filtered[wrd_column].sum() / 12

    results = []
    for _, row in filtered.iterrows():
        results.append({
            'id': row['ID'],
            'name': row['N4'],
            'amount': row[wrd_column] / 12
        })

    return results

def format_dutch_currency(value, decimals=0):
    """Format number as Dutch currency (dot for thousands, comma for decimals)."""
    formatted = f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"€ {formatted}"

# ================================================================================
# CACHED DATA FUNCTIONS FOR GRAPHS
# ================================================================================

@st.cache_data
def get_household_data(_df, selected_income, selected_referteperiode, selected_cav, selected_fr, key=None):
    """Calculate values for all municipalities and all household types (Graph 1)."""
    gemeente_codes = _df['GMcode'].dropna().unique()
    household_codes = ['HH01', 'HH02', 'HH03', 'HH04']
    all_values = []
    for gmcode in gemeente_codes:
        for hh_code in household_codes:
            total_value = filter_benefits(
                df=_df,
                gmcode=gmcode,
                hh=hh_code,
                ink=selected_income,
                refper=selected_referteperiode,
                cav=selected_cav,
                fr=selected_fr
            )
            all_values.append({
                'Gemeente': gmcode,
                'Huishouden': hh_code,
                'Waarde': total_value
            })
    return pd.DataFrame(all_values)

@st.cache_data
def get_income_progression_data(_df, selected_huishouden, selected_income_pct, selected_referteperiode, selected_cav, selected_fr, key=None):
    """Calculate values for all municipalities at specific income levels (Graph 2 markers)."""
    # Calculate income levels to show based on selected income
    final_digit = selected_income_pct % 10
    z = min(max(selected_income_pct - 20, 100 + final_digit), 140 + final_digit)
    income_levels_to_show = [z/100, (z+10)/100, (z+20)/100, (z+30)/100, (z+40)/100, (z+50)/100]

    gemeente_codes = _df['GMcode'].dropna().unique()
    all_values = []
    for gmcode in gemeente_codes:
        for income_level in income_levels_to_show:
            total_value = filter_benefits(
                df=_df,
                gmcode=gmcode,
                hh=selected_huishouden,
                ink=income_level,
                refper=selected_referteperiode,
                cav=selected_cav,
                fr=selected_fr
            )
            all_values.append({
                'Gemeente': gmcode,
                'Inkomen': income_level,
                'Waarde': total_value
            })
    return pd.DataFrame(all_values)

@st.cache_data
def get_income_line_data(_df, selected_gemeente, selected_huishouden, selected_referteperiode, selected_cav, selected_fr, key=None):
    """Calculate values for selected municipality at all income levels (Graph 2 line)."""
    all_income_levels = [i/100 for i in range(100, 201, 1)]
    selected_all_data = []
    for income_level in all_income_levels:
        total_value = filter_benefits(
            df=_df,
            gmcode=selected_gemeente,
            hh=selected_huishouden,
            ink=income_level,
            refper=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr
        )
        selected_all_data.append({
            'Inkomen': income_level,
            'Waarde': total_value
        })
    return pd.DataFrame(selected_all_data)

@st.cache_data
def get_formal_informal_data(_df, selected_huishouden, selected_income, selected_referteperiode, selected_cav, key=None):
    """Calculate formal and informal values for all municipalities (Graph 3)."""
    gemeente_codes = _df['GMcode'].dropna().unique()
    bar_data_values = []
    for gmcode in gemeente_codes:
        formal_value = filter_benefits(
            df=_df,
            gmcode=gmcode,
            hh=selected_huishouden,
            ink=selected_income,
            fr='Ja',
            refper=selected_referteperiode,
            cav=selected_cav
        )
        informal_value = filter_benefits(
            df=_df,
            gmcode=gmcode,
            hh=selected_huishouden,
            ink=selected_income,
            fr='Nee',
            refper=selected_referteperiode,
            cav=selected_cav
        )
        bar_data_values.append({
            'Gemeente': gmcode,
            'Formeel': formal_value,
            'Informeel': informal_value,
            'Totaal': formal_value + informal_value
        })
    return pd.DataFrame(bar_data_values)

@st.cache_data
def get_threshold_data(_df, selected_huishouden, selected_referteperiode, selected_cav, selected_fr, key=None):
    """Calculate weighted income thresholds and values for all municipalities (Graph 4)."""
    gemeente_codes = _df['GMcode'].dropna().unique()
    threshold_data_values = []
    for gmcode in gemeente_codes:
        gemeente_regs = _df[_df['GMcode'] == gmcode]
        if len(gemeente_regs) == 0:
            continue

        wrd_at_100 = filter_benefits(
            df=_df,
            gmcode=gmcode,
            hh=selected_huishouden,
            ink=1.0,
            refper=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr
        )

        weighted_sum = filter_benefits(
            df=_df,
            gmcode=gmcode,
            hh=selected_huishouden,
            ink=1.0,
            refper=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr,
            result="ig"
        )

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

@st.cache_data
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
                    color='#9f9f9f',
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
                    color='#d63f44',
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

@st.cache_data
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
                color='#9f9f9f',
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
            color='#d63f44',
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
            color='#d63f44'
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

@st.cache_data
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
    colors_formal = is_selected.map({True: '#d63f44', False: '#9f9f9f'}).tolist()
    colors_informal = is_selected.map({True: '#E68C8F', False: '#C5C5C5'}).tolist()

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

@st.cache_data
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
                color='#9f9f9f',
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
            household_labels[selected_huishouden] + "<br>Inkomensgrens: " +
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
                color='#d63f44',
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
    st.title("Dashboard armoedebeleid", anchor=False, text_alignment="center")

    # Prepare gemeente labels before tabs
    gemeenten_df = df[['GMcode', 'Gemeentenaam']].dropna().drop_duplicates().sort_values('Gemeentenaam')
    gemeente_labels = {str(row['GMcode']): str(row['Gemeentenaam']) for _, row in gemeenten_df.iterrows()}

    # ----------------------------------------------------------------------------
    # Selectors (in sidebar) - synced with URL query parameters
    # ----------------------------------------------------------------------------

    # Get defaults from URL query params (fall back to defaults if not present)
    params = st.query_params
    default_income = int(params.get("ink", 100))
    default_gemeente = params.get("gm", "GM0363")
    default_huishouden = params.get("hh", "HH04")
    default_refper = int(params.get("ref", 0))
    default_regelingen = params.get("reg", "").split(",") if params.get("reg") else []

    with st.sidebar:
        st.header("Filters", anchor=False)

        selected_income_pct = st.slider(
            "Inkomen:",
            min_value=100,
            max_value=200,
            value=default_income,
            step=1,
            format="%d%%",
            key="income"
        )
        selected_income = selected_income_pct / 100

        # Ensure default gemeente exists in the data
        if default_gemeente not in gemeente_labels:
            default_gemeente = list(gemeente_labels.keys())[0]

        selected_gemeente = st.selectbox(
            "Gemeente:",
            options=gemeente_labels.keys(),
            format_func=lambda x: gemeente_labels[x],
            index=list(gemeente_labels.keys()).index(default_gemeente),
            key="gemeente"
        )

        # Ensure default huishouden is valid
        if default_huishouden not in household_labels:
            default_huishouden = "HH04"

        selected_huishouden = st.selectbox(
            "Huishouden:",
            options=list(household_labels.keys()),
            format_func=lambda x: household_labels[x],
            index=list(household_labels.keys()).index(default_huishouden),
            key="huishouden"
        )

        selected_referteperiode = st.segmented_control(
            "Jaren met laag inkomen",
            options=[0, 1, 3, 5],
            default=default_refper if default_refper in [0, 1, 2, 3, 4, 5] else 0,
            key="referteperiode",
            help="Jaren dat het voorbeeldhuishouden al het geselecteerde inkomen heeft (referteperiode)"
        )

        # Regulation type selector (multi-select)
        has_reg_param = params.get("reg") is not None

        # Determine default value for regulation type selector
        if has_reg_param:
            default_reg_types = []
            if "f" in default_regelingen:
                default_reg_types.append("Formeel")
            if "i" in default_regelingen:
                default_reg_types.append("Informeel")
            # Ensure at least one is selected
            if not default_reg_types:
                default_reg_types = ["Formeel", "Informeel"]
        else:
            default_reg_types = ["Formeel", "Informeel"]

        # Handle auto-toggle logic: if session state exists and is empty, determine which option to auto-select
        if "reg_types" in st.session_state:
            current_value = st.session_state.reg_types
            if not current_value or len(current_value) == 0:
                # Get previous value from query params to determine what was selected before
                if "f" in default_regelingen and "i" not in default_regelingen:
                    # Had only Formeel, switch to Informeel
                    st.session_state.reg_types = ["Informeel"]
                    st.rerun()
                elif "i" in default_regelingen and "f" not in default_regelingen:
                    # Had only Informeel, switch to Formeel
                    st.session_state.reg_types = ["Formeel"]
                    st.rerun()
                else:
                    # Default to Formeel
                    st.session_state.reg_types = ["Formeel"]
                    st.rerun()

        selected_reg_types = st.segmented_control(
            "Type regelingen",
            options=["Formeel", "Informeel"],
            default=default_reg_types,
            selection_mode="multi",
            key="reg_types",
            help="Selecteer welke regelingen worden meegenomen: formele, informele of beide"
        )

        toggle_cav = st.toggle("Korting gemeentepolis", value="k" in default_regelingen if has_reg_param else False, key="toggle_cav")

        # Legend
        st.markdown("**Legenda**")
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #EF553B;"></div>
                <span>{gemeente_labels[selected_gemeente]}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #B0B0B0;"></div>
                <span>Overige gemeenten</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Update URL with current selections (preserving the key parameter)
    new_params = {"key": params.get("key", "")} if params.get("key") else {}
    new_params["ink"] = str(selected_income_pct)
    new_params["gm"] = selected_gemeente
    new_params["hh"] = selected_huishouden
    new_params["ref"] = str(selected_referteperiode)
    # Build reg parameter from regulation type selector
    reg_codes = []
    if "Formeel" in selected_reg_types:
        reg_codes.append("f")
    if "Informeel" in selected_reg_types:
        reg_codes.append("i")
    if toggle_cav:
        reg_codes.append("k")
    if reg_codes:
        new_params["reg"] = ",".join(reg_codes)
    st.query_params.update(new_params)

    # Calculate CAV/FR parameters
    selected_cav = 1 if toggle_cav else 0
    if "Formeel" in selected_reg_types and "Informeel" in selected_reg_types:
        selected_fr = "all"
    elif "Formeel" in selected_reg_types:
        selected_fr = "Ja"
    elif "Informeel" in selected_reg_types:
        selected_fr = "Nee"
    else:
        selected_fr = "all"

    # Create two-column layout: graphs on left, table on right
    graph_col, table_col = st.columns([2, 1])

    with graph_col:
        # Create tabs for the graphs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Huishoudtypen",
            "Inkomensgroepen",
            "(In)formeel",
            "Waarden en Inkomensgrenzen"
        ])

    # ----------------------------------------------------------------------------
    # Graph 1: Box Plot - Value by Household Type
    # ----------------------------------------------------------------------------
    with tab1:
        st.header("Waarde regelingen per huishouden", anchor=False)

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

        st.markdown(f"*De gecombineerde waarde (in € per maand) is een schatting o.b.v. alle regelingen waar de vier voorbeeldhuishoudens recht op hebben bij een inkomen van {selected_income_pct}% van het sociaal minimum*")

    # ----------------------------------------------------------------------------
    # Graph 2: Income Progression
    # ----------------------------------------------------------------------------
    with tab2:
        st.header("Waarde regelingen per inkomensgroep", anchor=False)

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

        st.markdown(f"*De gecombineerde waarde (in € per maand) is een schatting o.b.v. alle regelingen voor een {household_labels[selected_huishouden].lower()} bij verschillende inkomensniveaus*")
    # ----------------------------------------------------------------------------
    # Graph 3: Formal vs Informal
    # ----------------------------------------------------------------------------
    with tab3:
        st.header("Formele en informele waarden", anchor=False)

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

        st.markdown(f"*Waarde formele en informele gemeentelijke regelingen (in € per maand) voor een {household_labels[selected_huishouden].lower()} op {selected_income_pct}% van het sociaal minimum*")

    # ----------------------------------------------------------------------------
    # Graph 4: Population vs Income Threshold
    # ----------------------------------------------------------------------------
    with tab4:
        st.header("Waarde en gemiddelde inkomensgrens", anchor=False)

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
        st.markdown(f"*Waarde gemeentelijke regelingen (in € per maand) voor een {household_labels[selected_huishouden].lower()} op 100% van het sociaal minimum en het gewogen gemiddelde van alle inkomensgrenzen die de gemeente hanteert voor dit huishouden*")

    # ----------------------------------------------------------------------------
    # Regulations Table (shown on the right side)
    # ----------------------------------------------------------------------------
    with table_col:
        # Add vertical spacing to align with graphs (account for tabs and header)
        st.markdown("<div style='height: 95px;'></div>", unsafe_allow_html=True)

        # Column names for selected household type
        wrd_column = f'WRD_{selected_huishouden}'
        ig_column = f'IG_{selected_huishouden}'

        # Filter only by gemeente and household (WB=1)
        all_regs_mask = (df['GMcode'] == selected_gemeente) & (df['WB'] == 1)
        all_regs_df = df[all_regs_mask].copy()

        # Get regulations that match ALL current selectors
        regulations_list = filter_benefits(
            df=df,
            gmcode=selected_gemeente,
            hh=selected_huishouden,
            ink=selected_income,
            refper=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr,
            result="list"
        )

        # Create set of regulation IDs that match all filters
        matching_reg_ids = {reg['id'] for reg in regulations_list}

        # Build table data with all regulations
        regelingen_data = []
        for _, row in all_regs_df.iterrows():
            reg_id = row['ID']
            reg_name = row['N4']
            wrd_value = row[wrd_column]
            ig_value = row[ig_column]

            # Check if this regulation matches all selectors
            matches_filters = reg_id in matching_reg_ids

            regelingen_data.append({
                'Regeling': reg_name,
                'Waarde': (wrd_value / 12) if pd.notna(wrd_value) else None,
                'Inkomensgrens': ig_value if pd.notna(ig_value) else None,
                'Matches': matches_filters
            })

        # Split into matching and non-matching
        regs_matching = [r for r in regelingen_data if r['Matches']]
        regs_not_matching = [r for r in regelingen_data if not r['Matches']]

        # Combine rows with the same name and same matching status
        def combine_rows(rows):
            combined = {}
            for r in rows:
                name = r['Regeling']
                if name in combined:
                    # Sum the values
                    if combined[name]['Waarde'] is not None and r['Waarde'] is not None:
                        combined[name]['Waarde'] += r['Waarde']
                    elif r['Waarde'] is not None:
                        combined[name]['Waarde'] = r['Waarde']
                    # Use lower bound for income threshold
                    if combined[name]['Inkomensgrens'] is not None and r['Inkomensgrens'] is not None:
                        combined[name]['Inkomensgrens'] = min(combined[name]['Inkomensgrens'], r['Inkomensgrens'])
                    elif r['Inkomensgrens'] is not None:
                        combined[name]['Inkomensgrens'] = r['Inkomensgrens']
                else:
                    combined[name] = r.copy()
            return list(combined.values())

        regs_matching = combine_rows(regs_matching)
        regs_not_matching = combine_rows(regs_not_matching)

        # Sort matching by WRD value (descending), non-matching alphabetically
        regs_matching.sort(key=lambda x: x['Waarde'] if x['Waarde'] is not None else 0, reverse=True)
        regs_not_matching.sort(key=lambda x: x['Regeling'])

        # Combine
        regelingen_sorted = regs_matching + regs_not_matching

        if regelingen_sorted:
            display_df = pd.DataFrame(regelingen_sorted)

            # Find the maximum value to determine padding width
            max_value = max((r['Waarde'] for r in regelingen_sorted if r['Waarde'] is not None and r['Waarde'] > 0), default=0)
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
                if pd.notna(x) and x is not None:
                    return f"{int(x * 100)}%"
                return ""

            display_df['Waarde'] = display_df['Waarde'].apply(pad_currency)
            display_df['Inkomensgrens'] = display_df['Inkomensgrens'].apply(format_percentage)

            # Style rows: gray out non-matching
            def style_row(row):
                if not row['Matches']:
                    return ['color: #CCCCCC'] * len(row)
                return [''] * len(row)

            # Style Waarde column: monospace font for alignment
            def style_waarde_column(s):
                return ['font-family: monospace'] * len(s)

            num_rows = len(display_df)
            table_height = 38 + (num_rows * 35)  # No max limit, let table fit completely

            # Rename columns for display
            display_df = display_df.rename(columns={
                'Regeling': 'Regelingen',
                'Inkomensgrens': 'Grens'
            })

            styled_df = display_df[['Regelingen', 'Waarde', 'Grens', 'Matches']].style.apply(style_row, axis=1)
            styled_df = styled_df.apply(style_waarde_column, subset=['Waarde'])

            st.dataframe(
                styled_df,
                hide_index=True,
                height=table_height,
                column_config={
                    "Regelingen": st.column_config.TextColumn(
                        "Regelingen",
                        width=200
                    ),
                    "Waarde": st.column_config.TextColumn(
                        "Waarde",
                        width=75
                    ),
                    "Grens": st.column_config.TextColumn(
                        "Grens",
                        width=50
                    ),
                    "Matches": None
                }
            )
        else:
            st.info(f"Geen regelingen gevonden voor {gemeente_labels[selected_gemeente]}")

except FileNotFoundError:
    st.error("Databestand 'dataoverzicht_dashboard_armoedebeleid.xlsx' niet gevonden.")
except Exception as e:
    st.error(f"Er is een fout opgetreden: {str(e)}")