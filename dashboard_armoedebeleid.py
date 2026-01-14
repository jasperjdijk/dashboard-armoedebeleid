"""
Dashboard for municipal income-dependent regulations in the Netherlands

This dashboard visualizes:
1. Box plot showing value distribution by household type across municipalities
2. Income progression line chart showing how support decreases with income
3. Stacked bar chart comparing formal vs informal regulation values
4. Scatter plot showing relationship between population and income thresholds

Data source: Pythondata.xlsx with CBS 2025 municipality data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================
st.set_page_config(
    page_title="Dashboard gemeentelijke inkomensafhankelijke regelingen 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    excel_url = st.secrets["excel_url"]
    excel_file = pd.ExcelFile(excel_url)
    df = pd.read_excel(excel_file, sheet_name="Totaaloverzicht")
    return df

def filter_benefits(df, gmcode, hh, ink=1, referteperiode=0,cav=0, result="sum", fr="all", mt="all", wb=1, bt=1):
    """
    Filter benefits from Totaaloverzicht based on criteria

    Parameters:
    -----------
    df : DataFrame
        The Totaaloverzicht dataframe
    gmcode : str
        Municipality code (e.g., 'GM0363')
    hh : str
        Household type ('HH01', 'HH02', 'HH03', or 'HH04')
    ink : float
        Income level (e.g., 1.0 for 100%, 1.5 for 150%)
    fr : str or "all"
        FR value ('Ja', 'Nee', or 'all' to skip this filter)
    mt : int or "all"
        MT value (0, 1, or 'all' to skip this filter)
    wb : int
        WB value (0 or 1)
    bt : int
        BT value (0 or 1)
    cav : int
        CAV value (0 or 1)
    referteperiode : int
        Referte periode: 0, 1, 2, 3, 4 of 5 jaar
    sum_only : bool, optional
        If True, return only the sum of all WRD values (default: True)
    return_weighted_sum : bool, optional
        If True, return sum of WRD × (IG - 1.0) for weighted average calculation (default: False)

    Returns:
    --------
    list of dict or float
        If return_weighted_sum=True: Float representing sum of WRD × (IG - 1.0) / 12
        If sum_only=True: Float representing the sum of all WRD amounts / 12
        If sum_only=False: List of dictionaries with 'name' and 'amount' keys for each matching benefit
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
            'name': row['NM'],
            'amount': row[wrd_column] / 12
        })

    return results
    
# ================================================================================
# MAIN APPLICATION
# ================================================================================

try:
    # ----------------------------------------------------------------------------
    # Data Preparation
    # ----------------------------------------------------------------------------
    df = load_data()
    logo_base64 = get_logo_base64()

    def format_dutch_currency(value, decimals=2):
        """Format number with Dutch formatting: dot for thousands, comma for decimals"""
        formatted = f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"€ {formatted}"

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
#    st.title("Dashboard gemeentelijke inkomensafhankelijke regelingen")

    # Prepare gemeente labels before tabs
    gemeenten_df = df[['GMcode', 'Gemeentenaam']].dropna().drop_duplicates().sort_values('Gemeentenaam')
    gemeente_labels = {str(row['GMcode']): str(row['Gemeentenaam']) for _, row in gemeenten_df.iterrows()}

    # ----------------------------------------------------------------------------
    # Selectors (shown above tabs)
    # ----------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
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

    with col2:
        default_gemeente = st.session_state.get('selected_gemeente', 'GM0363')
        selected_gemeente = st.selectbox(
            "Gemeente:",
            options=gemeente_labels.keys(),
            format_func=lambda x: gemeente_labels[x],
            index=list(gemeente_labels.keys()).index(default_gemeente),
            key="gemeente"
        )
        st.session_state['selected_gemeente'] = selected_gemeente

    with col3:
        selected_huishouden = st.selectbox(
            "Huishouden:",
            options=list(household_labels.keys()),
            format_func=lambda x: household_labels[x],
            index=list(household_labels.keys()).index(st.session_state.get('selected_huishouden', 'HH04')),
            key="huishouden"
        )
        st.session_state['selected_huishouden'] = selected_huishouden

    with col4:
        selected_referteperiode = st.pills(
            "Jaren met laag inkomen:",
            options=[0, 1, 2, 3, 4, 5],
            default=st.session_state.get('selected_referteperiode', 0),
            key="referteperiode"
        )
        st.session_state['selected_referteperiode'] = selected_referteperiode

    with col5:
        regelingen_filter = st.multiselect(
            "Type regelingen:",
            options=["Formele regelingen", "Informele regelingen", "Korting gemeentepolis"],
            default=st.session_state.get('regelingen_filter', []),
            key="regelingen"
        )
        st.session_state['regelingen_filter'] = regelingen_filter

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

    # Create tabs for the graphs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Waarde per huishouden",
        "2. Waarde per inkomensgroep",
        "3. Formele en informele regelingen",
        "4. Waarden en inkomensgrenzen"
    ])

    # ----------------------------------------------------------------------------
    # Graph 1: Box Plot - Value by Household Type
    # ----------------------------------------------------------------------------
    with tab1:
        st.header("1. Waarde inkomensafhankelijke regeling verschillen tussen gemeenten")
        st.markdown(f"<p style='margin-bottom: 10px;'><em>De totale waarde (in € per maand) is een schatting o.b.v. alle regelingen waar de vier voorbeeldhuishoudens recht op hebben bij een inkomen van {selected_income_pct}% van het sociaal minimum</em></p>", unsafe_allow_html=True)

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

        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=600,
            showlegend=False,
            hovermode='closest',
            margin=dict(t=20, b=40, l=60, r=40),
            yaxis=dict(
                tickprefix="€ ",
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                tickfont=dict(size=14)
            )
        )

        fig = add_logo_to_figure(fig, logo_base64)

        config = {'displayModeBar': True, 'displaylogo': False}
        event = st.plotly_chart(fig, width='stretch', key="scatter_plot", on_select="rerun", config=config)

        if event and 'selection' in event and event['selection'] and 'points' in event['selection']:
            points = event['selection']['points']
            if len(points) > 0:
                clicked_code = points[0].get('customdata')
                if clicked_code and clicked_code != selected_gemeente:
                    st.session_state['selected_gemeente'] = clicked_code
                    st.rerun()

    # ----------------------------------------------------------------------------
    # Graph 2: Income Progression
    # ----------------------------------------------------------------------------
    with tab2:
        st.header("2. De afbouw van ondersteuning verschilt ook tussen gemeenten")
        st.markdown(f"<p style='margin-bottom: 10px;'><em>Gecombineerde waarde gemeentelijke regelingen voor een {household_labels[selected_huishouden].lower()} bij verschillende inkomensniveaus (uitgedrukt als percentage van het sociaal minimum)</em></p>", unsafe_allow_html=True)

        selected_gemeente_name = gemeente_labels[selected_gemeente]

        fig_income = go.Figure()
        income_levels_to_show = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

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

        fig_income.update_layout(
            xaxis_title="Inkomen (% van sociaal minimum)",
            yaxis_title="",
            height=600,
            hovermode='closest',
            margin=dict(t=20, b=40, l=60, r=40),
            xaxis=dict(
                tickmode='array',
                tickvals=[100, 110, 120, 130, 140, 150],
                ticktext=['100%', '110%', '120%', '130%', '140%', '150%'],
                tickfont=dict(size=14),
                range=[95, 155]
            ),
            yaxis=dict(
                tickprefix="€ ",
                tickfont=dict(size=14)
            )
        )

        fig_income = add_logo_to_figure(fig_income, logo_base64)

        st.plotly_chart(fig_income, width='stretch')

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

        selected_muni_data = bar_data[bar_data['Gemeente'] == selected_gemeente]
        if len(selected_muni_data) > 0:
            formal_value = selected_muni_data.iloc[0]['Formeel']
            total_value = selected_muni_data.iloc[0]['Totaal']
            formal_share = (formal_value / total_value * 100) if total_value > 0 else 0
            formal_share_text = f"{formal_share:.0f}%"
        else:
            formal_share_text = "N/A"

        st.header(f"3. Voor een {household_labels[selected_huishouden].lower()} in {selected_gemeente_name} is {formal_share_text} van de gemeentelijke ondersteuning formeel")
        st.markdown(f"<p style='margin-bottom: 10px;'><em>Waarde formele en informele gemeentelijke regelingen per maand voor een {household_labels[selected_huishouden].lower()} op {selected_income_pct}% van het sociaal minimum</em></p>", unsafe_allow_html=True)

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
            height=600,
            showlegend=True,
            margin=dict(t=20, b=40, l=60, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
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

    # ----------------------------------------------------------------------------
    # Graph 4: Population vs Income Threshold
    # ----------------------------------------------------------------------------
    with tab4:
        st.header("4. Gemeenten met meer inwoners hebben hogere inkomensdrempels")
        st.markdown(f"<p style='margin-bottom: 10px;'><em>Waarde gemeentelijke regelingen (in € per maand) voor iemand op 100% van het sociaal minimum en het gewogen gemiddelde van alle inkomensgrenzen die de gemeente hanteert voor een {household_labels[selected_huishouden].lower()}</em></p>", unsafe_allow_html=True)

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
            xaxis_title="Inkomensgrens",
            yaxis_title="",
            height=600,
            hovermode='closest',
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

        config_threshold = {'displayModeBar': True, 'displaylogo': False}
        event_threshold = st.plotly_chart(fig_threshold, width='stretch', key="threshold_plot", on_select="rerun", config=config_threshold)

        if event_threshold and 'selection' in event_threshold and event_threshold['selection'] and 'points' in event_threshold['selection']:
            points = event_threshold['selection']['points']
            if len(points) > 0:
                clicked_code = points[0].get('customdata')
                if clicked_code and clicked_code != selected_gemeente:
                    st.session_state['selected_gemeente'] = clicked_code
                    st.rerun()

    # ----------------------------------------------------------------------------
    # Regulations Table (shown below all tabs)
    # ----------------------------------------------------------------------------

    # Get values from session state (set by selectors in tabs)
    selected_gemeente = st.session_state.get('selected_gemeente', 'GM0363')
    selected_income_pct = st.session_state.get('selected_income_pct', 100)
    selected_huishouden = st.session_state.get('selected_huishouden', 'HH04')
    selected_referteperiode = st.session_state.get('selected_referteperiode', 0)
    regelingen_filter = st.session_state.get('regelingen_filter', ["Formele regelingen", "Informele regelingen"])
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

    selected_income = selected_income_pct / 100

    # Legend for all graphs
    st.markdown(f"""
    <div style="display: flex; gap: 30px; align-items: center; margin-bottom: 20px; justify-content: flex-end;">
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

    st.markdown("---")

    st.header(f"Regelingen in {gemeente_labels[selected_gemeente]}")
    st.markdown(f"*Overzicht van alle regelingen voor een {household_labels[selected_huishouden].lower()}. Regelingen die voldoen aan de geselecteerde filters staan bovenaan (gesorteerd op waarde).*")

    # Get ALL regulations for selected gemeente and huishouden (no other filters)
    hh = selected_huishouden
    wrd_column = f'WRD_{hh}'
    ig_column = f'IG_{hh}'
    ref_column = f'Referteperiode_{hh}'

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
        reg_name = row['NM']
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
                formatted = format_dutch_currency(x, 2)
                if ',' in formatted:
                    parts = formatted.split(',')
                    # Pad the integer part to align on the comma
                    padded = parts[0].rjust(max_int_width) + ',' + parts[1]
                    return padded
                return formatted.rjust(max_int_width + 3)  # +3 for ",XX"
            return ""

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
        table_height = min(38 + (num_rows * 35), 1400)

        styled_df = display_df[['Regeling', 'Waarde', 'Inkomensgrens', 'Matches']].style.apply(style_row, axis=1)
        styled_df = styled_df.apply(style_waarde_column, subset=['Waarde'])

        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=table_height,
            column_config={
                "Waarde": st.column_config.TextColumn(
                    "Waarde",
                    width="small"
                ),
                "Inkomensgrens": st.column_config.TextColumn(
                    "Inkomensgrens",
                    width="small"
                ),
                "Matches": None
            }
        )
    else:
        st.info(f"Geen regelingen gevonden voor {gemeente_labels[selected_gemeente]}")

except FileNotFoundError:
    st.error("Could not find Pythondata.xlsx. Please make sure the file exists in the project directory.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your data file format and try again.")
