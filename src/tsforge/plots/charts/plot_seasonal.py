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


def plot_seasonal(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    seasonal_agg: str = "mean",
    freq: str = "M",
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    kind: Literal["line", "box"] = "line",
    normalize: bool = False,
    show_mean: bool = False,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Seasonal subseries visualization with:
      ✓ monthly, quarterly, weekly, or daily cycles
      ✓ overlay, facet, dropdown layouts
      ✓ line or box seasonal representations
      ✓ optional normalization (removes trend)
      ✓ optional mean seasonal profile overlay
      ✓ matching theme + style overrides used in plot_timeseries()
    """

    # -----------------------------
    # Basic prep
    # -----------------------------
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # -----------------------------
    # Grouping (rollups)
    # -----------------------------
    # NOTE: Correct argument order for your preprocess function:
    # aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    df, id_col = aggregate_by_group(
        df=df,
        group_col=group_col,
        date_col=date_col,
        value_col=value_col,
        agg=agg,
        id_col=id_col,
    )

    # -----------------------------
    # ID selection helper
    # -----------------------------
    ids = select_ids(df, id_col=id_col, ids=ids, max_ids=6)

    df = df[df[id_col].isin(ids)].copy()

    # -----------------------------
    # Extract cycle components
    # -----------------------------
    df["year"] = df[date_col].dt.year

    f = freq.upper()
    if f == "M":
        df["seasonal_x"] = df[date_col].dt.month
        x_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    elif f == "Q":
        df["seasonal_x"] = df[date_col].dt.quarter
        x_labels = ["Q1","Q2","Q3","Q4"]
    elif f == "W":
        df["seasonal_x"] = df[date_col].dt.isocalendar().week.astype(int)
        x_labels = None
    elif f == "D":
        df["seasonal_x"] = df[date_col].dt.dayofyear
        x_labels = None
    else:
        raise ValueError("freq must be one of M, Q, W, D")

    # -----------------------------
    # Normalization BEFORE seasonal collapse
    # -----------------------------
    y_axis_label = value_col
    if normalize:
        yearly_mean = df.groupby([id_col, "year"])[value_col].transform("mean")
        df[value_col] = df[value_col] / yearly_mean
        y_axis_label = f"{value_col} (normalized)"

    # -----------------------------
    # Collapse: one value per (id, year, seasonal_x)
    # -----------------------------
    df = (
        df.groupby([id_col, "year", "seasonal_x"], observed=True)
        [value_col]
        .agg(seasonal_agg)
        .reset_index()
    )

    # -----------------------------
    # Optional Mean Seasonal Profile
    # -----------------------------
    mean_df = None
    if show_mean and kind == "line":
        mean_df = (
            df.groupby([id_col, "seasonal_x"], observed=True)[value_col]
            .mean()
            .reset_index()
        )

    # -----------------------------
    # PLOTTING BACKENDS
    # -----------------------------
    if engine != "plotly":
        raise NotImplementedError("matplotlib not yet implemented in final build")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # -----------------------------
    # Boxplot seasonal views
    # -----------------------------
    if kind == "box":

        if mode == "overlay":
            fig = go.Figure()
            for uid in ids:
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Box(
                        x=sub["seasonal_x"],
                        y=sub[value_col],
                        name=str(uid),
                        boxpoints="all",
                        jitter=0.2,
                        pointpos=0,
                        marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
                    )
                )

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 1
            rows = ceil(n / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(i) for i in ids])

            r = c = 1
            for uid in ids:
                sub = df[df[id_col] == uid]
                fig.add_trace(
                    go.Box(
                        x=sub["seasonal_x"],
                        y=sub[value_col],
                        name=str(uid),
                        boxpoints="all",
                        jitter=0.2,
                        marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
                    ),
                    row=r, col=c
                )
                c += 1
                if c > cols:
                    r += 1; c = 1

        elif mode == "dropdown":
            fig = go.Figure()
            traces_per_series = []

            for i, uid in enumerate(ids):
                sub = df[df[id_col] == uid]

                box = go.Box(
                    x=sub["seasonal_x"],
                    y=sub[value_col],
                    name=str(uid),
                    boxpoints="all",
                    jitter=0.2,
                    marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
                    visible=(i == 0),
                )

                fig.add_trace(box)
                traces_per_series.append(1)

            buttons = []
            total_traces = sum(traces_per_series)

            start_index = 0
            for i, uid in enumerate(ids):
                visible = [False] * total_traces
                visible[start_index] = True
                buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}]))
                start_index += 1

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.05, "y": 1.1
                }]
            )

        # X-axis labels for month/quarter
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(x_labels)+1)) if x_labels else None,
            ticktext=x_labels if x_labels else None
        )

    # -----------------------------
    # Line seasonal views
    # -----------------------------
    if kind == "line":

        opacity = 0.35 if show_mean else 0.9

        if mode == "overlay":
            fig = go.Figure()

            for uid in ids:
                sub = df[df[id_col] == uid]
                for j, (yr, g) in enumerate(sub.groupby("year")):
                    g = g.sort_values("seasonal_x")
                    fig.add_trace(go.Scatter(
                        x=g["seasonal_x"],
                        y=g[value_col],
                        mode="lines+markers",
                        name=f"{uid}-{yr}",
                        line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                        opacity=opacity,
                    ))

                # mean line
                if show_mean:
                    m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                    fig.add_trace(go.Scatter(
                        x=m["seasonal_x"],
                        y=m[value_col],
                        mode="lines",
                        name=f"{uid} mean",
                        line=dict(color="black", width=4, dash="dash"),
                        opacity=1.0,
                    ))

        elif mode == "facet":
            n = len(ids)
            cols = wrap or 1
            rows = ceil(n / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(i) for i in ids], shared_xaxes=True)

            r = c = 1
            for uid in ids:
                sub = df[df[id_col] == uid]
                for j, (yr, g) in enumerate(sub.groupby("year")):
                    g = g.sort_values("seasonal_x")
                    fig.add_trace(
                        go.Scatter(
                            x=g["seasonal_x"],
                            y=g[value_col],
                            mode="lines+markers",
                            name=f"{uid}-{yr}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=opacity,
                            showlegend=(r == 1),
                        ),
                        row=r, col=c
                    )

                if show_mean:
                    m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                    fig.add_trace(
                        go.Scatter(
                            x=m["seasonal_x"],
                            y=m[value_col],
                            mode="lines",
                            name="mean",
                            line=dict(color="black", width=4, dash="dash"),
                            opacity=1.0,
                            showlegend=(r == 1),
                        ),
                        row=r, col=c
                    )

                c += 1
                if c > cols:
                    r += 1; c = 1

        elif mode == "dropdown":
            fig = go.Figure()
            trace_counts = []

            for idx, uid in enumerate(ids):
                sub = df[df[id_col] == uid]
                local_count = 0

                for j, (yr, g) in enumerate(sub.groupby("year")):
                    g = g.sort_values("seasonal_x")
                    fig.add_trace(
                        go.Scatter(
                            x=g["seasonal_x"],
                            y=g[value_col],
                            mode="lines+markers",
                            name=f"{uid}-{yr}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=opacity,
                            visible=(idx == 0),
                        )
                    )
                    local_count += 1

                if show_mean:
                    m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                    fig.add_trace(
                        go.Scatter(
                            x=m["seasonal_x"],
                            y=m[value_col],
                            mode="lines",
                            name="mean",
                            line=dict(color="black", width=4, dash="dash"),
                            visible=(idx == 0),
                        )
                    )
                    local_count += 1

                trace_counts.append(local_count)

            # dropdown assembly
            buttons = []
            total = sum(trace_counts)
            start = 0

            for i, uid in enumerate(ids):
                visible = [False] * total
                for k in range(start, start + trace_counts[i]):
                    visible[k] = True
                buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}]))
                start += trace_counts[i]

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.05, "y": 1.1
                }]
            )

        # shared axis label logic
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(x_labels)+1)) if x_labels else None,
            ticktext=x_labels if x_labels else None,
        )
        fig.update_yaxes(title_text=y_axis_label)

    # -----------------------------
    # THEME + STYLE (matches plot_timeseries)
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

    return fig
