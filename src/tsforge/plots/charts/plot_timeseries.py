# =====================================================================
# tsforge.plots.charts.plot_timeseries
# Unified full-feature time-series plot (overlay, facet, dropdown)
# =====================================================================

from __future__ import annotations
import pandas as pd
from typing import Optional, Union, List, Callable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- core utilities ---------------------------------------------------
from tsforge.plots.core.palette import PALETTE, HIGHLIGHT, hex_to_rgba
from tsforge.plots.core.theme import apply_theme, apply_legend, THEMES
from tsforge.plots.core.preprocess import (
    apply_smoothing,
    aggregate_by_group,
    resample_df,
    select_ids,
)
from tsforge.plots.core.events import merge_all_events
from tsforge.plots.core.anomalies import normalize_anomalies
from tsforge.plots.core.forecast import pi_column_names


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def plot_timeseries(
    df: pd.DataFrame,
    *,
    # required base inputs
    id_col: str,
    date_col: str,
    value_col: Union[str, Callable] = "y",

    # optional grouping / aggregation
    group_col: Optional[Union[str, List[str]]] = None,
    agg: str = "sum",
    freq: Optional[str] = None,

    # selection of series
    ids: Optional[Union[str, int, List[str]]] = None,
    max_ids: int = 6,

    # smoothing
    smooth_window: Optional[int] = None,

    # -------------------------------------------------------------
    # forecast + prediction intervals
    # -------------------------------------------------------------
    forecast: Optional[pd.DataFrame] = None,
    forecast_value_col: str = "yhat",
    level: Optional[List[int]] = None,
    lo_pattern: str = "{col}-lo-{level}",
    hi_pattern: str = "{col}-hi-{level}",

    # -------------------------------------------------------------
    # event system
    # -------------------------------------------------------------
    events: Union[str, pd.DataFrame, None] = None,      # inline or df
    events_global: Optional[pd.DataFrame] = None,        # broadcasted events
    events_local: Optional[pd.DataFrame] = None,         # per-ID events
    event_label_col: str = "event",
    events_config: Optional[dict] = None,                # style overrides

    # -------------------------------------------------------------
    # anomalies
    # -------------------------------------------------------------
    anomalies: Union[pd.DataFrame, str, None] = None,
    anomaly_flag_value: int = 1,
    anomalies_config: Optional[dict] = None,

    # -------------------------------------------------------------
    # layout + modes
    # -------------------------------------------------------------
    mode: str = "overlay",               # "overlay" | "facet" | "dropdown"
    wrap: int = 3,                       # facet layout
    theme: str = "fa",                   # FA default
    style: Optional[dict] = None,        # optional title/subtitle/x/y

    # plot engine
    engine: str = "plotly",
):
    """
    Full-feature Forecast Academy-grade time-series plot.

    Supports:
      • overlay (stacked series)
      • facet (small multiples)
      • dropdown (interactive selector)

    Features:
      • actuals
      • smoothing
      • forecast
      • prediction intervals (80/95/etc)
      • anomalies
      • inline/global/local events + labels
      • grouping & resampling
      • themes (FA default, McKinsey, Minimal, Dark)
    """

    if engine.lower() != "plotly":
        raise NotImplementedError("Only Plotly engine is supported in FA v1.")

    # ------------------------------------------------------------------
    # 0) COPY + normalize
    # ------------------------------------------------------------------
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if callable(value_col):
        df["_value"] = value_col(df)
        value_col = "_value"

    # ------------------------------------------------------------------
    # 1) Group / Resample
    # ------------------------------------------------------------------
    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)

    df = resample_df(df, freq, id_col, date_col, value_col, agg)

    # ------------------------------------------------------------------
    # 2) Select IDs
    # ------------------------------------------------------------------
    ids = select_ids(df, id_col, ids, max_ids)

    df_sub = df[df[id_col].isin(ids)].copy()
    df_sub = df_sub.sort_values([id_col, date_col])

    # ------------------------------------------------------------------
    # 3) Align forecast
    # ------------------------------------------------------------------
    fcst_df = None
    if forecast is not None:
        fcst_df = forecast.copy()
        fcst_df[date_col] = pd.to_datetime(fcst_df[date_col])
        fcst_df = fcst_df[fcst_df[id_col].isin(ids)]
        fcst_df = fcst_df.sort_values([id_col, date_col])

    # ------------------------------------------------------------------
    # 4) Smoothing
    # ------------------------------------------------------------------
    df_sub = apply_smoothing(df_sub, id_col, value_col, smooth_window)

    # ------------------------------------------------------------------
    # 5) Events + anomalies (normalize & merge)
    # ------------------------------------------------------------------
    ev_all = merge_all_events(
        df=df_sub,
        id_col=id_col,
        date_col=date_col,
        event_label_col=event_label_col,
        inline=events if isinstance(events, str) else None,
        global_events=events_global,
        local_events=events_local,
        direct_df=events if isinstance(events, pd.DataFrame) else None,
    )

    an_df = normalize_anomalies(anomalies, df_sub, id_col, date_col, anomaly_flag_value)
    if an_df is not None:
        an_df = (
            an_df.merge(
                df_sub[[id_col, date_col, value_col]],
                on=[id_col, date_col],
                how="left",
            )
            .rename(columns={value_col: "y_anom"})
        )

    # ------------------------------------------------------------------
    # 6) Select mode
    # ------------------------------------------------------------------
    if mode == "overlay":
        fig = _plot_overlay(
            df_sub=df_sub,
            fcst_df=fcst_df,
            ids=ids,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            forecast_value_col=forecast_value_col,
            level=level,
            lo_pattern=lo_pattern,
            hi_pattern=hi_pattern,
            ev_all=ev_all,
            an_df=an_df,
            events_config=events_config,
            anomalies_config=anomalies_config,
            theme=theme,
        )
    elif mode == "facet":
        fig = _plot_facet(
            df_sub=df_sub,
            fcst_df=fcst_df,
            ids=ids,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            forecast_value_col=forecast_value_col,
            level=level,
            lo_pattern=lo_pattern,
            hi_pattern=hi_pattern,
            ev_all=ev_all,
            an_df=an_df,
            events_config=events_config,
            anomalies_config=anomalies_config,
            wrap=wrap,
            theme=theme,
        )

    elif mode == "dropdown":
        fig = _plot_dropdown(
            df_sub=df_sub,
            fcst_df=fcst_df,
            ids=ids,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            forecast_value_col=forecast_value_col,
            level=level,
            lo_pattern=lo_pattern,
            hi_pattern=hi_pattern,
            ev_all=ev_all,
            an_df=an_df,
            events_config=events_config,
            anomalies_config=anomalies_config,
            theme=theme,
        )

    else:
        raise ValueError("mode must be one of: overlay, facet, dropdown")

    # ------------------------------------------------------------------
    # 7) Apply theme + titles
    # ------------------------------------------------------------------
    fig = apply_theme(fig, theme)

    if style:
        if "title" in style:
            fig.update_layout(title={"text": style["title"], "font": {"size": 18}})
        if "subtitle" in style:
            fig.add_annotation(
                x=0, y=1.06,
                xref="paper", yref="paper",
                text=style["subtitle"],
                showarrow=False,
                font=dict(size=13, color="#444"),
            )
        if "x_title" in style:
            fig.update_xaxes(title_text=style["x_title"])
        if "y_title" in style:
            fig.update_yaxes(title_text=style["y_title"])
        # New: allow fixed y-axis range through style["y_range"]
        if "y_range" in style:
            fig.update_yaxes(range=style["y_range"])
    fig = apply_legend(fig, theme)

    # ------------------------------------------------------------------
    # 8) Position legend underneath the plot
    # ------------------------------------------------------------------
    # Count legend items to determine bottom margin
    num_legend_items = sum(1 for trace in fig.data if trace.showlegend)
    
    # Calculate rows needed (assume ~6 items per row)
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


# =====================================================================
# PLOT MODE IMPLEMENTATIONS
# =====================================================================

def _plot_overlay(
    df_sub, fcst_df, ids,
    id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df,
    events_config, anomalies_config,
    theme,
):
    fig = go.Figure()

    # Pull theme-level style (does NOT remove PALETTE functionality)
    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    anomaly_legend_shown = False

    # Config defaults (events)
    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True

    # Anomalies default → theme accent color
    an_color = (
        anomalies_config.get("color", accent_color)
        if anomalies_config
        else accent_color
    )
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    for i, uid in enumerate(ids):
        # Keep PALETTE for multi-series distinction
        color = PALETTE[i % len(PALETTE)]
        # If only one series and theme defines a line_color, use that
        if len(ids) == 1 and "line_color" in t:
            color = t["line_color"]

        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # ---- prediction intervals (background)
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    # lower bound (hidden, used for fill)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col],
                        y=fsub[lo],
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                    # upper bound with fill
                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col],
                        y=fsub[hi],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fillcolor,
                        hoverinfo="skip",
                        showlegend=False,
                    ))

        # ---- actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col],
            y=sub[value_col],
            mode="lines",
            name=str(uid),
            line=dict(color=color, width=line_width),
        ))

        # ---- forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col],
                y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, width=line_width, dash="dash"),
                name=f"{uid} forecast",
                showlegend=False,
            ))

        # ---- anomalies
        if an_sub is not None:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col],
                y=an_sub["y_anom"],
                mode="markers",
                name="Anomalies" if not anomaly_legend_shown else "",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                showlegend=not anomaly_legend_shown,
            ))
            anomaly_legend_shown = True

    # ---- global events
    if ev_all is not None:
        _add_event_lines_and_labels(
            fig, ev_all, date_col=date_col,
            ev_color=ev_color, stagger=ev_stagger
        )

    return fig


def _plot_facet(
    df_sub, fcst_df, ids,
    id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df,
    events_config, anomalies_config,
    wrap,
    theme,
):
    n = len(ids)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[str(uid) for uid in ids],
    )

    # Theme-level style
    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True

    an_color = (
        anomalies_config.get("color", accent_color)
        if anomalies_config
        else accent_color
    )
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    anomaly_legend_shown = False

    for r, uid in enumerate(ids, start=1):
        # Facet mode: one series per panel – use theme line_color if defined
        color = t.get("line_color", PALETTE[(r - 1) % len(PALETTE)])

        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # PIs
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[lo],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False,
                    ), row=r, col=1)

                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[hi],
                        mode="lines", line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fillcolor,
                        hoverinfo="skip", showlegend=False,
                    ), row=r, col=1)

        # actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines",
            name=str(uid),
            line=dict(color=color, width=line_width),
        ), row=r, col=1)

        # forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col], y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, dash="dash", width=line_width),
                showlegend=False,
            ), row=r, col=1)

        # anomalies
        if an_sub is not None:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col], y=an_sub["y_anom"],
                mode="markers",
                name="Anomalies" if not anomaly_legend_shown else "",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                showlegend=not anomaly_legend_shown,
            ), row=r, col=1)
            anomaly_legend_shown = True

    # events in all rows
    if ev_all is not None:
        _add_event_lines_and_labels(
            fig, ev_all, date_col=date_col,
            ev_color=ev_color, stagger=ev_stagger,
            facet=True, nrows=n
        )

    fig.update_layout(height=280 * n)
    return fig


def _plot_dropdown(
    df_sub, fcst_df, ids,
    id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df,
    events_config, anomalies_config,
    theme,
):
    fig = go.Figure()
    trace_map = {uid: [] for uid in ids}
    anomaly_legend_shown = False

    # Theme-level style
    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True

    an_color = (
        anomalies_config.get("color", accent_color)
        if anomalies_config
        else accent_color
    )
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    for i, uid in enumerate(ids):
        color = PALETTE[i % len(PALETTE)]
        # If dropdown only has one ID, prefer theme line_color if set
        if len(ids) == 1 and "line_color" in t:
            color = t["line_color"]

        visible = (i == 0)

        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # PIs
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[lo],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip",
                        visible=visible, showlegend=False,
                    ))
                    trace_map[uid].append(len(fig.data) - 1)

                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[hi],
                        mode="lines", line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fillcolor,
                        hoverinfo="skip",
                        visible=visible, showlegend=False,
                    ))
                    trace_map[uid].append(len(fig.data) - 1)

        # actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines",
            line=dict(color=color, width=line_width),
            visible=visible,
            showlegend=False,
        ))
        trace_map[uid].append(len(fig.data) - 1)

        # forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col], y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, width=line_width, dash="dash"),
                visible=visible,
                showlegend=False,
            ))
            trace_map[uid].append(len(fig.data) - 1)

        # anomalies
        if an_sub is not None:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col], y=an_sub["y_anom"],
                mode="markers",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                visible=visible,
                name="Anomalies" if not anomaly_legend_shown else "",
                showlegend=not anomaly_legend_shown,
            ))
            trace_map[uid].append(len(fig.data) - 1)
            anomaly_legend_shown = True

    # global events
    if ev_all is not None:
        _add_event_lines_and_labels(
            fig, ev_all, date_col=date_col,
            ev_color=ev_color, stagger=ev_stagger
        )

    # dropdown
    buttons = []
    total = len(fig.data)
    for uid in ids:
        mask = [False] * total
        for idx in trace_map[uid]:
            mask[idx] = True

        buttons.append(dict(
            label=str(uid),
            method="update",
            args=[{"visible": mask}],
        ))

    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 1.0, "y": 1.15,
            "xanchor": "right",
            "yanchor": "top"
        }]
    )

    return fig


# =====================================================================
# EVENT LINES + LABELS
# =====================================================================

def _add_event_lines_and_labels(
    fig,
    ev_all,
    date_col,
    event_label_col="event",
    ev_color="#555",
    stagger=True,
    facet=False,
    nrows=None,
):
    """Draw vertical event lines + text labels (works for overlay, facet, dropdown)."""

    # Unique event dates
    unique_dates = sorted(ev_all[date_col].unique())

    # Build mapping: date → event label
    label_map = (
        ev_all.drop_duplicates(subset=[date_col])
              .set_index(date_col)[event_label_col]
              .to_dict()
    )

    for i, d in enumerate(unique_dates):
        label = label_map.get(d, "event")

        # Draw vertical line
        if facet:
            for r in range(1, nrows + 1):
                fig.add_vline(
                    x=d,
                    row=r, col=1,
                    line_width=1,
                    line_dash="dot",
                    line_color=ev_color,
                    opacity=0.7,
                )
        else:
            fig.add_vline(
                x=d,
                line_width=1,
                line_dash="dot",
                line_color=ev_color,
                opacity=0.7,
            )

        # Add label above the plot
        ypos = 0.95
        if stagger:
            ypos += (i % 2) * 0.03

        fig.add_annotation(
            x=d,
            y=ypos,
            xref="x",
            yref="paper",
            showarrow=False,
            text=str(label),
            font=dict(size=11, color=ev_color),
            align="center",
            yanchor="bottom",
        )