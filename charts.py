"""
Chart generation functions for the dashboard.

Each function receives pre-computed data and returns a Plotly figure.
This keeps visualisation logic separate from data processing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from constants import BUBBLE_SIZE_DIVISOR
from utils import format_dutch_currency


# =============================================================================
# Graph 1 — Household-type comparison (box plot)
# =============================================================================

def huishoudtypen_grafiek(
    huishoudtypen_df: pd.DataFrame,
    sel_gm: str,
    hh_lbl: dict[str, str],
    ink: float,
    color_selected: str,
    color_other: str,
) -> go.Figure:
    """Box plot comparing benefit values across household types."""
    # Add hover text
    huishoudtypen_df = huishoudtypen_df.copy()
    huishoudtypen_df['hover_text'] = (
        "<b>" + huishoudtypen_df['Gemeentenaam'].astype(str) + "</b><br>"
        + f"{int(ink * 100)}% sociaal minimum<br>Waarde: "
        + huishoudtypen_df['Waarde'].apply(format_dutch_currency).astype(str)
    )

    # Multi-line x-axis labels
    label_mapping = {
        'Alleenstaande': 'Alleenstaande',
        'Alleenstaande ouder met kind': 'Alleenstaande<br>ouder met kind',
        'Paar': 'Paar',
        'Paar met twee kinderen': 'Paar met<br>twee kinderen',
    }
    hh_labels = list(hh_lbl.values())
    x_labels = [label_mapping.get(lbl, lbl) for lbl in hh_labels]

    fig = go.Figure()

    for hh_code, hh_label in hh_lbl.items():
        household_data = huishoudtypen_df[huishoudtypen_df['Huishouden'] == hh_code]

        trace_configs = [
            {'data': household_data[household_data['GMcode'] != sel_gm],
             'size': 8, 'color': color_other, 'opacity': 0.6},
            {'data': household_data[household_data['GMcode'] == sel_gm],
             'size': 10, 'color': color_selected, 'opacity': 1.0},
        ]

        for cfg in trace_configs:
            rows = cfg['data']
            if len(rows) > 0:
                fig.add_trace(go.Box(
                    x=[hh_label] * len(rows),
                    y=rows['Waarde'],
                    name=hh_label,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(
                        size=cfg['size'],
                        color=cfg['color'],
                        opacity=cfg['opacity'],
                    ),
                    hovertext=rows['hover_text'],
                    hoverinfo='text',
                    showlegend=False,
                    fillcolor='rgba(255,255,255,0)',
                    line=dict(color='rgba(255,255,255,0)'),
                ))

    # Annotate selected municipality
    selected = huishoudtypen_df[huishoudtypen_df['GMcode'] == sel_gm]
    for _, row in selected.iterrows():
        fig.add_annotation(
            x=hh_lbl[row['Huishouden']],
            y=row['Waarde'],
            text=format_dutch_currency(row['Waarde']),
            showarrow=False,
            xanchor='left',
            xshift=20,
            font=dict(size=14, color='black'),
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=450,
        showlegend=False,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=60, l=50, r=20, autoexpand=False),
        yaxis=dict(tickprefix="€ ", tickfont=dict(size=14), fixedrange=True),
        xaxis=dict(
            tickfont=dict(size=14),
            tickangle=0,
            tickmode='array',
            tickvals=hh_labels,
            ticktext=x_labels,
            fixedrange=True,
        ),
    )
    return fig


# =============================================================================
# Graph 2 — Income progression (line + markers)
# =============================================================================

def inkomensgroepen_grafiek(
    inkomensgroepen_df: pd.DataFrame,
    inkomenslijn_df: pd.DataFrame,
    sel_gm: str,
    hh: str,
    hh_lbl: dict[str, str],
    ink_pct: int,
    color_selected: str,
    color_other: str,
) -> go.Figure:
    """Line chart showing how benefits decrease as income increases."""
    fig = go.Figure()

    inkomensgroepen_df = inkomensgroepen_df.copy()
    income_levels = sorted(inkomensgroepen_df['Inkomen'].unique())

    # Hover text
    inkomensgroepen_df['hover_text'] = (
        "<b>" + inkomensgroepen_df['Gemeentenaam'].astype(str) + "</b><br>"
        + hh_lbl[hh] + "<br>Waarde: "
        + inkomensgroepen_df['Waarde'].apply(format_dutch_currency).astype(str)
    )

    # Visual properties
    is_selected = inkomensgroepen_df['GMcode'] == sel_gm
    inkomensgroepen_df['marker_color'] = is_selected.map({True: color_selected, False: color_other})
    inkomensgroepen_df['marker_size'] = is_selected.map({True: 10, False: 8})
    inkomensgroepen_df['marker_opacity'] = is_selected.map({True: 1.0, False: 0.6})

    # Markers for all municipalities
    fig.add_trace(go.Scatter(
        x=inkomensgroepen_df['Inkomen'],
        y=inkomensgroepen_df['Waarde'],
        mode='markers',
        name='Gemeenten',
        marker=dict(
            size=inkomensgroepen_df['marker_size'],
            color=inkomensgroepen_df['marker_color'],
            opacity=inkomensgroepen_df['marker_opacity'],
        ),
        hovertext=inkomensgroepen_df['hover_text'],
        hoverinfo='text',
        showlegend=False,
    ))

    # Smooth line for selected municipality
    fig.add_trace(go.Scatter(
        x=inkomenslijn_df['Inkomen'],
        y=inkomenslijn_df['Waarde'],
        mode='lines',
        line=dict(color=color_selected, width=2),
        hoverinfo='skip',
        showlegend=False,
    ))

    # Value annotation at selected income
    row = inkomenslijn_df.loc[inkomenslijn_df['Inkomen'] == ink_pct]
    if not row.empty:
        value = row['Waarde'].squeeze()
        fig.add_annotation(
            x=ink_pct,
            y=value,
            text=format_dutch_currency(value),
            showarrow=False,
            xanchor='center',
            yshift=15,
            font=dict(size=14, color='black'),
        )

    tick_text = [f'{v}%' for v in income_levels]
    fig.update_layout(
        xaxis_title="Inkomen (% van sociaal minimum)",
        yaxis_title="",
        height=450,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
        xaxis=dict(
            tickmode='array',
            tickvals=income_levels,
            ticktext=tick_text,
            tickfont=dict(size=14),
            range=[income_levels[0] - 5, income_levels[-1] + 5],
            fixedrange=True,
        ),
        yaxis=dict(tickprefix="€ ", tickfont=dict(size=14), fixedrange=True),
    )
    return fig


# =============================================================================
# Graph 3 — Formal vs. informal (stacked bar)
# =============================================================================

def in_formeel_grafiek(
    bar_data: pd.DataFrame,
    sel_gm: str,
    hh: str,
    hh_lbl: dict[str, str],
    ink_pct: int,
    color_selected: str,
    color_other: str,
    color_informal_selected: str,
    color_informal_other: str,
) -> go.Figure:
    """Stacked bar chart: formal vs. informal regulations."""
    bar_data = bar_data.copy().sort_values('Formeel', ascending=False)

    is_selected = bar_data['GMcode'] == sel_gm
    bar_data['color_formal'] = is_selected.map({True: color_selected, False: color_other})
    bar_data['color_informal'] = is_selected.map({True: color_informal_selected, False: color_informal_other})

    hover_base = f"{hh_lbl[hh]}<br>{ink_pct}% sociaal minimum<br>"
    bar_data['hover_formal'] = bar_data['Formeel'].apply(
        lambda val: f"{hover_base}Waarde formele regelingen: {format_dutch_currency(val)}<extra></extra>"
    )
    bar_data['hover_informal'] = bar_data['Informeel'].apply(
        lambda val: f"{hover_base}Waarde informele regelingen: {format_dutch_currency(val)}<extra></extra>"
    )

    fig = go.Figure([
        go.Bar(
            x=bar_data['Gemeentenaam'],
            y=bar_data['Formeel'],
            name='Formeel',
            marker_color=bar_data['color_formal'],
            hovertemplate=bar_data['hover_formal'],
        ),
        go.Bar(
            x=bar_data['Gemeentenaam'],
            y=bar_data['Informeel'],
            name='Informeel',
            marker_color=bar_data['color_informal'],
            hovertemplate=bar_data['hover_informal'],
        ),
    ])

    fig.update_layout(
        barmode='stack',
        xaxis_title="",
        yaxis_title="",
        height=450,
        showlegend=True,
        dragmode=False,
        margin=dict(t=0, b=100, l=50, r=20, autoexpand=False),
        legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="right", x=0.99),
        yaxis=dict(tickprefix="€ ", tickfont=dict(size=14), fixedrange=True),
        xaxis=dict(tickfont=dict(size=14), tickangle=-45, fixedrange=True),
    )
    return fig


# =============================================================================
# Graph 4 — Value vs. income threshold (scatter)
# =============================================================================

def gem_inkomensgrenzen_grafiek(
    threshold_data: pd.DataFrame,
    sel_gm: str,
    hh: str,
    hh_lbl: dict[str, str],
    color_selected: str,
    color_other: str,
) -> go.Figure:
    """Scatter plot: benefit value vs. weighted income threshold."""
    fig = go.Figure()

    if len(threshold_data) > 0:
        threshold_data = threshold_data.copy()
        is_selected = threshold_data['Gemeente'] == sel_gm
        threshold_data['marker_color'] = is_selected.map({True: color_selected, False: color_other})
        threshold_data['marker_opacity'] = is_selected.map({True: 1.0, False: 0.6})

        threshold_data['ig_pct'] = threshold_data['Inkomensgrens'].astype(str)
        threshold_data['waarde_fmt'] = threshold_data['Waarde'].apply(format_dutch_currency)
        threshold_data['inwoners_fmt'] = threshold_data['Inwoners'].apply(
            lambda x: f"{x:,.0f}".replace(',', '.')
        )

        threshold_data['hover_text'] = (
            "<b>" + threshold_data['Gemeentenaam'].astype(str) + "</b><br>"
            + hh_lbl[hh] + "<br>Gemiddelde Inkomensgrens: "
            + threshold_data['ig_pct'] + "%<br>"
            + "Waarde bij 100% sociaal minimum: " + threshold_data['waarde_fmt'] + "<br>"
            + "Inwoners: " + threshold_data['inwoners_fmt']
        )

        fig.add_trace(go.Scatter(
            x=threshold_data['Inkomensgrens'],
            y=threshold_data['Waarde'],
            mode='markers',
            marker=dict(
                size=threshold_data['Inwoners'] / BUBBLE_SIZE_DIVISOR,
                color=threshold_data['marker_color'],
                opacity=threshold_data['marker_opacity'],
                sizemode='area',
            ),
            hovertext=threshold_data['hover_text'],
            hoverinfo='text',
            showlegend=False,
        ))

        # Label selected municipality
        sel_rows = threshold_data[is_selected]
        for _, row in sel_rows.iterrows():
            fig.add_annotation(
                x=row['Inkomensgrens'],
                y=row['Waarde'],
                text=row['Gemeentenaam'],
                showarrow=False,
                xanchor='left',
                xshift=10,
                font=dict(size=12, color='black'),
            )

    fig.update_layout(
        xaxis_title="Inkomensgrens (% van sociaal minimum)",
        yaxis_title="",
        height=450,
        hovermode='closest',
        dragmode=False,
        margin=dict(t=0, b=50, l=50, r=20, autoexpand=False),
        xaxis=dict(ticksuffix="%", tickfont=dict(size=14), fixedrange=True),
        yaxis=dict(tickprefix="€ ", tickfont=dict(size=14), fixedrange=True),
    )
    return fig
