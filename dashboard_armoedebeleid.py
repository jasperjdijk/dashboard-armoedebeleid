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

Data source: dataoverzicht_dashboard_armoedebeleid.xlsx
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64

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
        min-width: 340px;
        max-width: 340px;
    }
    [data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

@st.cache_data
def get_logo_base64():
    """Load logo image and convert to base64 for embedding in Plotly graphs"""
    try:
        with open("IPE Logo 01.png", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return None

# Function to add logo to Plotly figures
def add_logo_to_figure(fig, logo_base64):
    """Add logo as a watermark to a Plotly figure"""
    if logo_base64:
        fig.add_layout_image(
            dict(
                source=logo_base64,
                xref="paper",
                yref="paper",
                x=0.5,  # Center horizontally
                y=0.5,  # Center vertically
                sizex=0.3,  # Width of logo (30% of plot width)
                sizey=0.3,  # Height of logo (30% of plot height)
                xanchor="center",
                yanchor="middle",
                opacity=0.1,  # Very transparent so it doesn't interfere with data
                layer="below"  # Place behind the data
            )
        )
    return fig

@st.cache_data
def load_data():
    """Load all required data from Excel file and merge municipality information"""
    # Get Excel URL from Streamlit secrets (keeps data private)
    excel_url = st.secrets["excel_url"]
    excel_file = pd.ExcelFile(excel_url)
    df = pd.read_excel(excel_file, sheet_name="Totaaloverzicht")
    return df

def filter_benefits(df, gmcode, hh, ink=1, referteperiode=0, cav=0, result="sum", fr="all", mt="all", wb=1, bt=1):
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
    referteperiode : int
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
    mask &= ref_numeric.notna() & (ref_numeric <= referteperiode)

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
            'name': row['N4'],
            'amount': row[wrd_column] / 12
        })

    return results

def format_dutch_currency(value, decimals=2):
    """Format number as Dutch currency (dot for thousands, comma for decimals)."""
    formatted = f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"€ {formatted}"

# ================================================================================
# MAIN APPLICATION
# ================================================================================

try:
    # ----------------------------------------------------------------------------
    # Data Preparation
    # ----------------------------------------------------------------------------
    df = load_data()
    logo_base64 = get_logo_base64()

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
    st.title("Dashboard armoedebeleid")

    # Prepare gemeente labels before tabs
    gemeenten_df = df[['GMcode', 'Gemeentenaam']].dropna().drop_duplicates().sort_values('Gemeentenaam')
    gemeente_labels = {str(row['GMcode']): str(row['Gemeentenaam']) for _, row in gemeenten_df.iterrows()}

    # ----------------------------------------------------------------------------
    # Selectors (in sidebar)
    # ----------------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        selected_income_pct = st.slider(
            "Inkomen:",
            min_value=100,
            max_value=200,
            value=st.session_state.get('selected_income_pct', 100),
            step=1,
            format="%d%%",
            key="income"
        )
        st.session_state['selected_income_pct'] = selected_income_pct
        selected_income = selected_income_pct / 100

        default_gemeente = st.session_state.get('selected_gemeente', 'GM0363')
        selected_gemeente = st.selectbox(
            "Gemeente:",
            options=gemeente_labels.keys(),
            format_func=lambda x: gemeente_labels[x],
            index=list(gemeente_labels.keys()).index(default_gemeente),
            key="gemeente"
        )
        st.session_state['selected_gemeente'] = selected_gemeente

        selected_huishouden = st.selectbox(
            "Huishouden:",
            options=list(household_labels.keys()),
            format_func=lambda x: household_labels[x],
            index=list(household_labels.keys()).index(st.session_state.get('selected_huishouden', 'HH04')),
            key="huishouden"
        )
        st.session_state['selected_huishouden'] = selected_huishouden

        selected_referteperiode = st.pills(
            "Jaren met laag inkomen:",
            options=[0, 1, 2, 3, 4, 5],
            default=st.session_state.get('selected_referteperiode', 0),
            key="referteperiode"
        )
        st.session_state['selected_referteperiode'] = selected_referteperiode

        regelingen_filter = st.multiselect(
            "Type regelingen:",
            options=["Formele regelingen", "Informele regelingen", "Korting gemeentepolis"],
            default=st.session_state.get('regelingen_filter', []),
            key="regelingen"
        )
        st.session_state['regelingen_filter'] = regelingen_filter

        # Legend
        st.markdown("---")
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

    # Calculate CAV/FR parameters
    selected_cav = 1 if "Korting gemeentepolis" in regelingen_filter else 0
    has_formeel = "Formele regelingen" in regelingen_filter
    has_informeel = "Informele regelingen" in regelingen_filter
    if has_formeel and has_informeel:
        selected_fr = "all"
    elif has_formeel:
        selected_fr = "Ja"
    elif has_informeel:
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
        st.header("Waarde regelingen per huishouden")

        # Calculate values for all municipalities and all household types
        all_values = []

        for gmcode, gemeente_name in gemeente_labels.items():
            for hh_code, hh_label in household_labels.items():
                total_value = filter_benefits(
                    df=df,
                    gmcode=gmcode,
                    hh=hh_code,
                    ink=selected_income,
                    referteperiode=selected_referteperiode,
                    cav=selected_cav,
                    fr=selected_fr
                )

                all_values.append({
                    'Gemeente': gmcode,
                    'Gemeentenaam': gemeente_name,
                    'Huishouden': hh_code,
                    'Huishouden_Label': hh_label,
                    'Waarde': total_value
                })

        # Create DataFrame from calculated values
        plot_df = pd.DataFrame(all_values)

        fig = go.Figure()
        for household in sorted(plot_df['Huishouden_Label'].unique()):
            household_data = plot_df[plot_df['Huishouden_Label'] == household]

            selected_data = household_data[household_data['Gemeente'] == selected_gemeente]
            other_data = household_data[household_data['Gemeente'] != selected_gemeente]

            if len(other_data) > 0:
                hover_text_other = [
                    f"<b>{row['Gemeentenaam']}</b><br>{selected_income_pct}% sociaal minimum<br>Waarde: {format_dutch_currency(row['Waarde'], 0)}"
                    for _, row in other_data.iterrows()
                ]
                customdata_other = other_data['Gemeente'].values

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
                    customdata=customdata_other,
                    showlegend=False,
                    fillcolor='rgba(255,255,255,0)',
                    line=dict(color='rgba(255,255,255,0)')
                ))

            if len(selected_data) > 0:
                hover_text_selected = [
                    f"<b>{row['Gemeentenaam']}</b><br>{selected_income_pct}% sociaal minimum<br>Waarde: {format_dutch_currency(row['Waarde'], 0)}"
                    for _, row in selected_data.iterrows()
                ]
                customdata_selected = selected_data['Gemeente'].values

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
                    customdata=customdata_selected,
                    showlegend=False,
                    fillcolor='rgba(255,255,255,0)',
                    line=dict(color='rgba(255,255,255,0)')
                ))

        selected_municipality_data = plot_df[plot_df['Gemeente'] == selected_gemeente]
        for _, row in selected_municipality_data.iterrows():
            fig.add_annotation(
                x=row['Huishouden_Label'],
                y=row['Waarde'],
                text=format_dutch_currency(row['Waarde'], 0),
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
            margin=dict(t=0, b=60, l=50, r=20, autoexpand=False),
            yaxis=dict(
                tickprefix="€ ",
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                tickfont=dict(size=14),
                tickangle=0,
                tickmode='array',
                tickvals=sorted(plot_df['Huishouden_Label'].unique()),
                ticktext=x_labels
            )
        )

        fig = add_logo_to_figure(fig, logo_base64)

        st.plotly_chart(fig, width='stretch')

        st.markdown(f"*De gecombineerde waarde (in € per maand) is een schatting o.b.v. alle regelingen waar de vier voorbeeldhuishoudens recht op hebben bij een inkomen van {selected_income_pct}% van het sociaal minimum*")

    # ----------------------------------------------------------------------------
    # Graph 2: Income Progression
    # ----------------------------------------------------------------------------
    with tab2:
        st.header("Waarde regelingen per inkomensgroep")

        selected_gemeente_name = gemeente_labels[selected_gemeente]

        fig_income = go.Figure()

        # Calculate dynamic income levels based on selected income
        # First column is always 100%, then z, z+10, z+20, z+30, z+40
        # where z = max(selected_income_pct - 20, 110) + final digit of selected_income_pct
        final_digit = selected_income_pct % 10
        z = min(max(selected_income_pct - 20, 100 + final_digit),140 + final_digit)
        income_levels_to_show = [z/100, (z+10)/100, (z+20)/100, (z+30)/100, (z+40)/100, (z+50)/100]

        for gemeente_code, gemeente_name in gemeente_labels.items():
            gemeente_values = []
            for income_level in income_levels_to_show:
                total_value = filter_benefits(
                    df=df,
                    gmcode=gemeente_code,
                    hh=selected_huishouden,
                    ink=income_level,
                    referteperiode=selected_referteperiode,
                    cav=selected_cav,
                    fr=selected_fr
                )
                gemeente_values.append({'Inkomen': income_level, 'Waarde': total_value})

            gemeente_df = pd.DataFrame(gemeente_values)

            if gemeente_code == selected_gemeente:
                selected_marker_data = gemeente_df.copy()
            else:
                hover_text = [
                    f"<b>{gemeente_name}</b><br>{household_labels[selected_huishouden]}<br>Waarde: {format_dutch_currency(row['Waarde'], 0)}"
                    for _, row in gemeente_df.iterrows()
                ]

                fig_income.add_trace(go.Scatter(
                    x=gemeente_df['Inkomen'] * 100,
                    y=gemeente_df['Waarde'],
                    mode='markers',
                    name=gemeente_name,
                    marker=dict(
                        size=8,
                        color='#9f9f9f',
                        opacity=0.6
                    ),
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ))

        all_income_levels = [i/100 for i in range(100, 201, 1)]
        selected_all_data = []

        for income_level in all_income_levels:
            total_value = filter_benefits(
                df=df,
                gmcode=selected_gemeente,
                hh=selected_huishouden,
                ink=income_level,
                referteperiode=selected_referteperiode,
                cav=selected_cav,
                fr=selected_fr
            )
            selected_all_data.append({
                'Inkomen': income_level,
                'Waarde': total_value
            })

        selected_all_df = pd.DataFrame(selected_all_data)

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

        hover_text = [
            f"<b>{selected_gemeente_name}</b><br>{household_labels[selected_huishouden]}<br>Waarde: {format_dutch_currency(row['Waarde'], 0)}"
            for _, row in selected_marker_data.iterrows()
        ]

        fig_income.add_trace(go.Scatter(
            x=selected_marker_data['Inkomen'] * 100,
            y=selected_marker_data['Waarde'],
            mode='markers',
            name=selected_gemeente_name,
            marker=dict(
                size=10,
                color='#d63f44'
            ),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))

        # Add label for the selected income level
        # Find the value at the selected income level from the line data
        selected_income_value = selected_all_df[selected_all_df['Inkomen'] == selected_income]['Waarde'].values
        if len(selected_income_value) > 0:
            fig_income.add_annotation(
                x=selected_income_pct,
                y=selected_income_value[0],
                text=format_dutch_currency(selected_income_value[0], 0),
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
            margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
            xaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=14),
                range=[tick_vals[0] - 5, tick_vals[-1] + 5]
            ),
            yaxis=dict(
                tickprefix="€ ",
                tickfont=dict(size=14)
            )
        )

        fig_income = add_logo_to_figure(fig_income, logo_base64)

        st.plotly_chart(fig_income, width='stretch')

        st.markdown(f"*De gecombineerde waarde (in € per maand) is een schatting o.b.v. alle regelingen voor een {household_labels[selected_huishouden].lower()} bij verschillende inkomensniveaus*")
    # ----------------------------------------------------------------------------
    # Graph 3: Formal vs Informal
    # ----------------------------------------------------------------------------
    with tab3:
        selected_gemeente_name = gemeente_labels[selected_gemeente]
        bar_data_values = []

        for gemeente_code, gemeente_name in gemeente_labels.items():
            formal_value = filter_benefits(
                df=df,
                gmcode=gemeente_code,
                hh=selected_huishouden,
                ink=selected_income,
                fr='Ja',
                referteperiode=selected_referteperiode,
                cav=selected_cav
            )

            informal_value = filter_benefits(
                df=df,
                gmcode=gemeente_code,
                hh=selected_huishouden,
                ink=selected_income,
                fr='Nee',
                referteperiode=selected_referteperiode,
                cav=selected_cav
            )

            bar_data_values.append({
                'Gemeente': gemeente_code,
                'Gemeentenaam': gemeente_name,
                'Formeel': formal_value,
                'Informeel': informal_value,
                'Totaal': formal_value + informal_value
            })

        bar_data = pd.DataFrame(bar_data_values)
        bar_data = bar_data.sort_values('Formeel', ascending=False)

        st.header("Formele en informele waarden")

        colors_formal = ['#d63f44' if code == selected_gemeente else '#9f9f9f' for code in bar_data['Gemeente']]
        colors_informal = ['#E68C8F' if code == selected_gemeente else '#C5C5C5' for code in bar_data['Gemeente']]

        hover_formal = [
            f"{household_labels[selected_huishouden]}<br>{selected_income_pct}% sociaal minimum<br>Waarde formele regelingen: {format_dutch_currency(row['Formeel'], 0)}<extra></extra>"
            for _, row in bar_data.iterrows()
        ]
        hover_informal = [
            f"{household_labels[selected_huishouden]}<br>{selected_income_pct}% sociaal minimum<br>Waarde informele regelingen: {format_dutch_currency(row['Informeel'], 0)}<extra></extra>"
            for _, row in bar_data.iterrows()
        ]

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
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                tickfont=dict(size=14),
                tickangle=-45
            )
        )

        fig_bar = add_logo_to_figure(fig_bar, logo_base64)

        st.plotly_chart(fig_bar, width='stretch')

        st.markdown(f"*Waarde formele en informele gemeentelijke regelingen (in € per maand) voor een {household_labels[selected_huishouden].lower()} op {selected_income_pct}% van het sociaal minimum*")

    # ----------------------------------------------------------------------------
    # Graph 4: Population vs Income Threshold
    # ----------------------------------------------------------------------------
    with tab4:
        st.header("Waarde en gemiddelde inkomensgrens")

        threshold_data_values = []

        for gemeente_code, gemeente_name in gemeente_labels.items():
            gemeente_regs = df[df['GMcode'] == gemeente_code]
            if len(gemeente_regs) == 0:
                continue

            # Get WRD at 100% income
            wrd_at_100 = filter_benefits(
                df=df,
                gmcode=gemeente_code,
                hh=selected_huishouden,
                ink=1.0,
                referteperiode=selected_referteperiode,
                cav=selected_cav,
                fr=selected_fr
            )

            # Get weighted sum: sum of WRD × (min(IG, 2.0) - 1.0)
            weighted_sum = filter_benefits(
                df=df,
                gmcode=gemeente_code,
                hh=selected_huishouden,
                ink=1.0,
                referteperiode=selected_referteperiode,
                cav=selected_cav,
                fr=selected_fr,
                result="ig"
            )

            # Calculate weighted average threshold: 1 + (weighted_sum / wrd_at_100) / 100
            if wrd_at_100 > 0:
                weighted_ig = 1 + (weighted_sum / wrd_at_100)
            else:
                weighted_ig = None

            value_100 = wrd_at_100

            if 'Inwoners' in df.columns and len(gemeente_regs) > 0:
                try:
                    inwoners = gemeente_regs['Inwoners'].iloc[0]
                    if pd.isna(inwoners):
                        inwoners = 50000
                except:
                    inwoners = 50000
            else:
                inwoners = 50000

            if weighted_ig is not None:
                threshold_data_values.append({
                    'Gemeente': gemeente_code,
                    'Gemeentenaam': gemeente_name,
                    'Inkomensgrens': weighted_ig,
                    'Waarde': value_100,
                    'Inwoners': inwoners
                })

        threshold_data = pd.DataFrame(threshold_data_values)
        fig_threshold = go.Figure()

        if len(threshold_data) > 0:
            selected_threshold_data = threshold_data[threshold_data['Gemeente'] == selected_gemeente]
            other_threshold_data = threshold_data[threshold_data['Gemeente'] != selected_gemeente]
        else:
            selected_threshold_data = pd.DataFrame()
            other_threshold_data = pd.DataFrame()

        if len(other_threshold_data) > 0:
            hover_text_other = [
                f"<b>{row['Gemeentenaam']}</b><br>{household_labels[selected_huishouden]}<br>Inkomensgrens: {row['Inkomensgrens']*100:.0f}%<br>Waarde bij 100% sociaal minimum: {format_dutch_currency(row['Waarde'], 0)}<br>Inwoners: {row['Inwoners']:,.0f}".replace(',', '.')
                for _, row in other_threshold_data.iterrows()
            ]
            customdata_other = other_threshold_data['Gemeente'].values

            fig_threshold.add_trace(go.Scatter(
                x=other_threshold_data['Inkomensgrens'] * 100,
                y=other_threshold_data['Waarde'],
                mode='markers',
                marker=dict(
                    size=other_threshold_data['Inwoners'] / 5000,
                    color='#9f9f9f',
                    opacity=0.6,
                    sizemode='diameter'
                ),
                hovertext=hover_text_other,
                hoverinfo='text',
                customdata=customdata_other,
                showlegend=False
            ))

        if len(selected_threshold_data) > 0:
            hover_text_selected = [
                f"<b>{row['Gemeentenaam']}</b><br>{household_labels[selected_huishouden]}<br>Inkomensgrens: {row['Inkomensgrens']*100:.0f}%<br>Waarde bij 100% sociaal minimum: {format_dutch_currency(row['Waarde'], 0)}<br>Inwoners: {row['Inwoners']:,.0f}".replace(',', '.')
                for _, row in selected_threshold_data.iterrows()
            ]
            customdata_selected = selected_threshold_data['Gemeente'].values

            fig_threshold.add_trace(go.Scatter(
                x=selected_threshold_data['Inkomensgrens'] * 100,
                y=selected_threshold_data['Waarde'],
                mode='markers',
                marker=dict(
                    size=selected_threshold_data['Inwoners'] / 5000,
                    color='#d63f44',
                    sizemode='diameter'
                ),
                hovertext=hover_text_selected,
                hoverinfo='text',
                customdata=customdata_selected,
                showlegend=False
            ))

        fig_threshold.update_layout(
            xaxis_title="Inkomensgrens (% van sociaal minimum)",
            yaxis_title="",
            height=450,
            hovermode='closest',
            margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
            xaxis=dict(
                ticksuffix="%",
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                tickprefix="€ ",
                tickfont=dict(size=14)
            )
        )

        fig_threshold = add_logo_to_figure(fig_threshold, logo_base64)

        st.plotly_chart(fig_threshold, width='stretch')
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
            referteperiode=selected_referteperiode,
            cav=selected_cav,
            fr=selected_fr,
            result="list"
        )

        # Create set of regulation names that match all filters
        matching_regs = {reg['name'] for reg in regulations_list}

        # Build table data with all regulations
        regelingen_data = []
        for _, row in all_regs_df.iterrows():
            reg_name = row['N4']
            wrd_value = row[wrd_column]
            ig_value = row[ig_column]

            # Check if this regulation matches all selectors
            matches_filters = reg_name in matching_regs

            regelingen_data.append({
                'Regeling': reg_name,
                'Waarde': (wrd_value / 12) if pd.notna(wrd_value) else None,
                'Inkomensgrens': ig_value if pd.notna(ig_value) else None,
                'Matches': matches_filters
            })

        # Split into matching and non-matching
        regs_matching = [r for r in regelingen_data if r['Matches']]
        regs_not_matching = [r for r in regelingen_data if not r['Matches']]

        # Sort matching by WRD value (descending), non-matching alphabetically
        regs_matching.sort(key=lambda x: x['Waarde'] if x['Waarde'] is not None else 0, reverse=True)
        regs_not_matching.sort(key=lambda x: x['Regeling'])

        # Combine
        regelingen_sorted = regs_matching + regs_not_matching

        if regelingen_sorted:
            display_df = pd.DataFrame(regelingen_sorted)

            # Find the maximum width of the integer part for alignment
            max_int_width = 0
            for val in display_df['Waarde']:
                if pd.notna(val) and val is not None:
                    formatted = format_dutch_currency(val, 2)
                    if ',' in formatted:
                        int_part = formatted.split(',')[0]
                        max_int_width = max(max_int_width, len(int_part))

            # Format currency for Waarde column with alignment on comma
            def pad_currency(x):
                if pd.notna(x) and x is not None:
                    if x == 0:
                        return "Ontbreekt"
                    formatted = format_dutch_currency(x, 2)
                    if ',' in formatted:
                        parts = formatted.split(',')
                        # Pad the integer part to align on the comma
                        padded = parts[0].rjust(max_int_width) + ',' + parts[1]
                        return padded
                    return formatted.rjust(max_int_width + 3)  # +3 for ",XX"
                return "Ontbreekt"

            # Format percentage for Inkomensgrens column
            def format_percentage(x):
                if pd.notna(x) and x is not None:
                    return f"{int(x * 100)}%"
                return ""

            display_df['Waarde'] = display_df['Waarde'].apply(pad_currency)
            display_df['Inkomensgrens'] = display_df['Inkomensgrens'].apply(format_percentage)

            # Style rows: gray out non-matching and make Waarde column monospace
            def style_row(row):
                if not row['Matches']:
                    return ['color: #CCCCCC'] * len(row)
                return [''] * len(row)

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
                        width="100"
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