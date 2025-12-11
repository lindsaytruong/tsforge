# =====================================================================
# tsforge.plots.charts.plot_distribution
# Distribution visualization for time series values
# =====================================================================

import pandas as pd
import numpy as np
from math import ceil
from typing import Union, List, Optional, Literal

from tsforge.plots.core.preprocess import (
    aggregate_by_group,
    select_ids,
)
from tsforge.plots.core.palette import PALETTE, HIGHLIGHT
from tsforge.plots.core.theme import apply_theme, apply_legend


def plot_distribution(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    kind: Literal["histogram", "density", "box", "violin"] = "histogram",
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    bins: int = 30,
    log_scale: bool = False,
    exclude_zeros: bool = False,
    show_stats: bool = True,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Distribution visualization for time series values.

    Supports:
      ✓ histogram, density (KDE), box, violin plots
      ✓ overlay, facet, dropdown layouts
      ✓ optional log scale for skewed distributions
      ✓ optional zero exclusion for intermittent series
      ✓ statistical annotations (mean, median, std)
      ✓ matching theme + style overrides

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing time series data.
    id_col : str
        Column identifying unique series.
    date_col : str
        Column with datetime values.
    value_col : str
        The metric to analyze.
    ids : list, int, str, or None
        Specific IDs to include, number to sample, or None for auto.
    group_col : str, list, or None
        Column(s) to group by before analysis.
    agg : str
        Aggregation function if grouping (default: "sum").
    kind : str
        Plot type: "histogram", "density", "box", or "violin".
    mode : str
        Layout: "overlay", "facet", or "dropdown".
    bins : int
        Number of histogram bins (default: 30).
    log_scale : bool
        Apply log scale to x-axis (useful for skewed data).
    exclude_zeros : bool
        Exclude zero values from distribution.
    show_stats : bool
        Show mean/median annotations on plot.
    wrap : int or None
        Number of columns for facet layout.
    theme : str
        Visual theme (default: "fa").
    style : dict or None
        Custom styling overrides.
    engine : str
        Plotting engine (default: "plotly").

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # -----------------------------
    # Basic prep
    # -----------------------------
    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # -----------------------------
    # Grouping (rollups)
    # -----------------------------
    df, id_col = aggregate_by_group(
        df=df,
        group_col=group_col,
        date_col=date_col,
        value_col=value_col,
        agg=agg,
        id_col=id_col,
    )

    # -----------------------------
    # ID selection
    # -----------------------------
    ids = select_ids(df, id_col=id_col, ids=ids, max_ids=6)
    df = df[df[id_col].isin(ids)].copy()

    # -----------------------------
    # Optional: exclude zeros
    # -----------------------------
    if exclude_zeros:
        df = df[df[value_col] > 0].copy()

    # -----------------------------
    # Optional: log transform
    # -----------------------------
    plot_col = value_col
    if log_scale:
        # Add small constant to handle zeros
        df["_log_value"] = np.log1p(df[value_col].clip(lower=0))
        plot_col = "_log_value"

    # -----------------------------
    # PLOTTING
    # -----------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Compute stats for annotations
    stats = {}
    for uid in ids:
        sub = df[df[id_col] == uid][value_col]
        stats[uid] = {
            "mean": sub.mean(),
            "median": sub.median(),
            "std": sub.std(),
            "min": sub.min(),
            "max": sub.max(),
            "n": len(sub),
        }

    # -----------------------------
    # HISTOGRAM
    # -----------------------------
    if kind == "histogram":

        if mode == "overlay":
            fig = go.Figure()
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Histogram(
                        x=sub[plot_col],
                        name=str(uid),
                        opacity=0.7,
                        nbinsx=bins,
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                    )
                )
            fig.update_layout(barmode="overlay")

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 2
            rows = ceil(n / cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[str(uid) for uid in ids],
            )

            r, c = 1, 1
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Histogram(
                        x=sub[plot_col],
                        name=str(uid),
                        nbinsx=bins,
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                        showlegend=False,
                    ),
                    row=r, col=c
                )

                # Add mean line
                if show_stats:
                    mean_val = sub[plot_col].mean()
                    fig.add_vline(
                        x=mean_val, row=r, col=c,
                        line_dash="dash", line_color="black",
                        annotation_text=f"μ={mean_val:.1f}",
                        annotation_position="top right",
                    )

                c += 1
                if c > cols:
                    r += 1
                    c = 1

            fig.update_layout(height=300 * rows)

        elif mode == "dropdown":
            fig = go.Figure()

            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Histogram(
                        x=sub[plot_col],
                        name=str(uid),
                        nbinsx=bins,
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                        visible=(i == 0),
                        showlegend=False,
                    )
                )

            buttons = []
            for i, uid in enumerate(ids):
                visible = [False] * len(ids)
                visible[i] = True
                s = stats[uid]
                title = f"{uid} | μ={s['mean']:.1f}, σ={s['std']:.1f}, n={s['n']}"
                buttons.append(dict(
                    label=str(uid),
                    method="update",
                    args=[{"visible": visible}, {"title": title}],
                ))

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.0, "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }]
            )

    # -----------------------------
    # DENSITY (KDE)
    # -----------------------------
    elif kind == "density":

        if mode == "overlay":
            fig = go.Figure()
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid][plot_col].dropna()
                if len(sub) < 2:
                    continue

                # Compute KDE using numpy histogram as approximation
                hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=hist,
                        mode="lines",
                        name=str(uid),
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                        fill="tozeroy",
                        opacity=0.6,
                    )
                )

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 2
            rows = ceil(n / cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[str(uid) for uid in ids],
            )

            r, c = 1, 1
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid][plot_col].dropna()
                if len(sub) >= 2:
                    hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    fig.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=hist,
                            mode="lines",
                            name=str(uid),
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                            fill="tozeroy",
                            showlegend=False,
                        ),
                        row=r, col=c
                    )

                c += 1
                if c > cols:
                    r += 1
                    c = 1

            fig.update_layout(height=300 * rows)

        elif mode == "dropdown":
            fig = go.Figure()

            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid][plot_col].dropna()
                if len(sub) >= 2:
                    hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    fig.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=hist,
                            mode="lines",
                            name=str(uid),
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                            fill="tozeroy",
                            visible=(i == 0),
                            showlegend=False,
                        )
                    )

            buttons = []
            for i, uid in enumerate(ids):
                visible = [False] * len(ids)
                visible[i] = True
                buttons.append(dict(
                    label=str(uid),
                    method="update",
                    args=[{"visible": visible}],
                ))

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.0, "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }]
            )

    # -----------------------------
    # BOX PLOT
    # -----------------------------
    elif kind == "box":

        if mode == "overlay":
            fig = go.Figure()
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Box(
                        y=sub[plot_col],
                        name=str(uid),
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                        boxpoints="outliers",
                    )
                )

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 3
            rows = ceil(n / cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[str(uid) for uid in ids],
            )

            r, c = 1, 1
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Box(
                        y=sub[plot_col],
                        name=str(uid),
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                        boxpoints="outliers",
                        showlegend=False,
                    ),
                    row=r, col=c
                )

                c += 1
                if c > cols:
                    r += 1
                    c = 1

            fig.update_layout(height=300 * rows)

        elif mode == "dropdown":
            fig = go.Figure()

            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Box(
                        y=sub[plot_col],
                        name=str(uid),
                        marker=dict(color=PALETTE[i % len(PALETTE)]),
                        boxpoints="outliers",
                        visible=(i == 0),
                        showlegend=False,
                    )
                )

            buttons = []
            for i, uid in enumerate(ids):
                visible = [False] * len(ids)
                visible[i] = True
                buttons.append(dict(
                    label=str(uid),
                    method="update",
                    args=[{"visible": visible}],
                ))

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.0, "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }]
            )

    # -----------------------------
    # VIOLIN PLOT
    # -----------------------------
    elif kind == "violin":

        if mode == "overlay":
            fig = go.Figure()
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Violin(
                        y=sub[plot_col],
                        name=str(uid),
                        line=dict(color=PALETTE[i % len(PALETTE)]),
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 3
            rows = ceil(n / cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[str(uid) for uid in ids],
            )

            r, c = 1, 1
            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Violin(
                        y=sub[plot_col],
                        name=str(uid),
                        line=dict(color=PALETTE[i % len(PALETTE)]),
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False,
                    ),
                    row=r, col=c
                )

                c += 1
                if c > cols:
                    r += 1
                    c = 1

            fig.update_layout(height=300 * rows)

        elif mode == "dropdown":
            fig = go.Figure()

            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Violin(
                        y=sub[plot_col],
                        name=str(uid),
                        line=dict(color=PALETTE[i % len(PALETTE)]),
                        box_visible=True,
                        meanline_visible=True,
                        visible=(i == 0),
                        showlegend=False,
                    )
                )

            buttons = []
            for i, uid in enumerate(ids):
                visible = [False] * len(ids)
                visible[i] = True
                buttons.append(dict(
                    label=str(uid),
                    method="update",
                    args=[{"visible": visible}],
                ))

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.0, "y": 1.15,
                    "xanchor": "right",
                    "yanchor": "top",
                }]
            )

    else:
        raise ValueError("kind must be one of: histogram, density, box, violin")

    # -----------------------------
    # Axis labels
    # -----------------------------
    x_label = f"log({value_col})" if log_scale else value_col
    if kind in ["histogram", "density"]:
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="Frequency" if kind == "histogram" else "Density")
    else:
        fig.update_yaxes(title_text=x_label)

    # -----------------------------
    # THEME + STYLE
    # -----------------------------
    fig = apply_theme(fig, theme)

    if style:
        if "title" in style and "subtitle" not in style:
            fig.update_layout(title=style["title"])
        if "title" in style and "subtitle" in style:
            fig.update_layout(
                title=f"{style['title']}<br><sup>{style['subtitle']}</sup>"
            )
        if "x_title" in style:
            fig.update_xaxes(title_text=style["x_title"])
        if "y_title" in style:
            fig.update_yaxes(title_text=style["y_title"])

    fig = apply_legend(fig, theme)

    # -----------------------------
    # LEGEND POSITIONING (underneath)
    # -----------------------------
    num_legend_items = sum(1 for trace in fig.data if trace.showlegend)
    legend_rows = max(1, (num_legend_items + 5) // 6)
    bottom_margin = 60 + (legend_rows * 25)

    fig.update_layout(
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
        ),
        margin=dict(b=bottom_margin),
    )

    return fig
