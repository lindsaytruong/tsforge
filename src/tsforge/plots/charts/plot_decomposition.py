# =====================================================================
# tsforge.plots.charts.plot_decomposition
# STL decomposition visualization for time series
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


# =====================================================================
# STL DECOMPOSITION HELPER
# =====================================================================

def stl_decompose(
    series: pd.Series,
    period: int = None,
    seasonal: int = 7,
    trend: int = None,
    robust: bool = True,
) -> dict:
    """
    Perform STL decomposition on a time series.

    Parameters
    ----------
    series : pd.Series
        Time series values (should be sorted by time).
    period : int, optional
        Seasonal period. If None, attempts to infer from data length.
    seasonal : int
        Length of seasonal smoother. Must be odd. Default 7.
    trend : int, optional
        Length of trend smoother. Must be odd. If None, auto-calculated.
    robust : bool
        Use robust fitting (reduces impact of outliers). Default True.

    Returns
    -------
    dict
        Dictionary with 'trend', 'seasonal', 'resid', and 'observed' arrays.
    """
    from statsmodels.tsa.seasonal import STL

    # Handle missing values
    series_clean = series.dropna()
    if len(series_clean) < 2 * (period or 7):
        # Not enough data for decomposition
        n = len(series)
        return {
            "observed": series.values,
            "trend": np.full(n, np.nan),
            "seasonal": np.full(n, np.nan),
            "resid": np.full(n, np.nan),
        }

    # Default period based on common frequencies
    if period is None:
        period = 7  # Default to weekly

    # Ensure seasonal is odd
    if seasonal % 2 == 0:
        seasonal += 1

    try:
        stl = STL(
            series_clean,
            period=period,
            seasonal=seasonal,
            trend=trend,
            robust=robust,
        )
        result = stl.fit()

        return {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    except Exception as e:
        # Return NaNs if decomposition fails
        n = len(series)
        return {
            "observed": series.values,
            "trend": np.full(n, np.nan),
            "seasonal": np.full(n, np.nan),
            "resid": np.full(n, np.nan),
        }


def compute_decomposition_stats(decomp: dict) -> dict:
    """
    Compute summary statistics from decomposition results.

    Parameters
    ----------
    decomp : dict
        Decomposition result from stl_decompose().

    Returns
    -------
    dict
        Statistics including trend range, seasonal amplitude, residual std, etc.
    """
    trend = decomp["trend"]
    seasonal = decomp["seasonal"]
    resid = decomp["resid"]
    observed = decomp["observed"]

    # Handle NaN values
    trend_clean = trend[~np.isnan(trend)]
    seasonal_clean = seasonal[~np.isnan(seasonal)]
    resid_clean = resid[~np.isnan(resid)]
    observed_clean = observed[~np.isnan(observed)]

    stats = {
        "trend_min": np.min(trend_clean) if len(trend_clean) > 0 else np.nan,
        "trend_max": np.max(trend_clean) if len(trend_clean) > 0 else np.nan,
        "trend_range": (np.max(trend_clean) - np.min(trend_clean)) if len(trend_clean) > 0 else np.nan,
        "seasonal_amplitude": (np.max(seasonal_clean) - np.min(seasonal_clean)) if len(seasonal_clean) > 0 else np.nan,
        "seasonal_std": np.std(seasonal_clean) if len(seasonal_clean) > 0 else np.nan,
        "resid_std": np.std(resid_clean) if len(resid_clean) > 0 else np.nan,
        "resid_mean": np.mean(resid_clean) if len(resid_clean) > 0 else np.nan,
        "observed_mean": np.mean(observed_clean) if len(observed_clean) > 0 else np.nan,
        "observed_std": np.std(observed_clean) if len(observed_clean) > 0 else np.nan,
    }

    # Signal-to-noise ratio (seasonal amplitude / residual std)
    if stats["resid_std"] > 0:
        stats["snr"] = stats["seasonal_amplitude"] / stats["resid_std"]
    else:
        stats["snr"] = np.nan

    # Strength of trend (1 - Var(resid) / Var(deseasonalized))
    deseasonalized = observed_clean - seasonal_clean[:len(observed_clean)] if len(seasonal_clean) >= len(observed_clean) else observed_clean
    if np.var(deseasonalized) > 0:
        stats["trend_strength"] = max(0, 1 - np.var(resid_clean) / np.var(deseasonalized))
    else:
        stats["trend_strength"] = np.nan

    # Strength of seasonality (1 - Var(resid) / Var(detrended))
    detrended = observed_clean - trend_clean[:len(observed_clean)] if len(trend_clean) >= len(observed_clean) else observed_clean
    if np.var(detrended) > 0:
        stats["seasonal_strength"] = max(0, 1 - np.var(resid_clean) / np.var(detrended))
    else:
        stats["seasonal_strength"] = np.nan

    return stats


# =====================================================================
# MAIN PLOTTING FUNCTION
# =====================================================================

def plot_decomposition(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    period: int = None,
    seasonal: int = 7,
    trend: int = None,
    robust: bool = True,
    mode: Literal["panel", "overlay", "dropdown"] = "panel",
    show_stats: bool = True,
    components: List[str] = None,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    STL decomposition visualization for time series.

    Decomposes series into Trend + Seasonal + Residual components.

    Supports:
      ✓ panel layout (stacked components)
      ✓ overlay mode (all components on one plot)
      ✓ dropdown for multiple series
      ✓ statistical annotations
      ✓ customizable seasonal period

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing time series data.
    id_col : str
        Column identifying unique series.
    date_col : str
        Column with datetime values.
    value_col : str
        The metric to decompose.
    ids : list, int, str, or None
        Specific IDs to include, number to sample, or None for auto.
    group_col : str, list, or None
        Column(s) to group by before analysis.
    agg : str
        Aggregation function if grouping (default: "sum").
    period : int, optional
        Seasonal period for STL. Common values:
        - 7 for daily data with weekly seasonality
        - 12 for monthly data with yearly seasonality
        - 52 for weekly data with yearly seasonality
        If None, defaults to 7.
    seasonal : int
        Length of seasonal smoother. Must be odd. Default 7.
    trend : int, optional
        Length of trend smoother. If None, auto-calculated.
    robust : bool
        Use robust fitting to reduce outlier impact. Default True.
    mode : str
        Layout mode:
        - "panel": Stacked subplots (observed, trend, seasonal, residual)
        - "overlay": All components on one plot
        - "dropdown": Interactive selector for multiple series
    show_stats : bool
        Show decomposition statistics as annotations.
    components : list, optional
        Which components to show. Default: ["observed", "trend", "seasonal", "resid"]
    wrap : int, optional
        Number of columns for multi-series panel layout.
    theme : str
        Visual theme (default: "fa").
    style : dict, optional
        Custom styling overrides.
    engine : str
        Plotting engine (default: "plotly").

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> # Basic decomposition
    >>> plot_decomposition(df, id_col='unique_id', date_col='ds', value_col='y',
    ...                    period=52)  # Weekly data, yearly seasonality

    >>> # Compare multiple series
    >>> plot_decomposition(df, ..., ids=['SKU_A', 'SKU_B'], mode='dropdown')

    >>> # Overlay mode to see component relationships
    >>> plot_decomposition(df, ..., mode='overlay', components=['trend', 'seasonal'])
    """

    # -----------------------------
    # Basic prep
    # -----------------------------
    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Default components
    if components is None:
        components = ["observed", "trend", "seasonal", "resid"]

    # Validate components
    valid_components = {"observed", "trend", "seasonal", "resid"}
    components = [c for c in components if c in valid_components]
    if not components:
        components = ["observed", "trend", "seasonal", "resid"]

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
    ids = select_ids(df, id_col=id_col, ids=ids, max_ids=4)
    df = df[df[id_col].isin(ids)].copy()

    # -----------------------------
    # Perform decomposition for each series
    # -----------------------------
    decomp_results = {}
    stats_results = {}

    for uid in ids:
        sub = df[df[id_col] == uid].sort_values(date_col)
        series = sub[value_col].reset_index(drop=True)
        dates = sub[date_col].values

        decomp = stl_decompose(
            series,
            period=period,
            seasonal=seasonal,
            trend=trend,
            robust=robust,
        )
        decomp["dates"] = dates
        decomp_results[uid] = decomp
        stats_results[uid] = compute_decomposition_stats(decomp)

    # -----------------------------
    # PLOTTING
    # -----------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Component display config
    component_config = {
        "observed": {"name": "Observed", "color": PALETTE[0], "dash": "solid"},
        "trend": {"name": "Trend", "color": PALETTE[1], "dash": "solid"},
        "seasonal": {"name": "Seasonal", "color": PALETTE[2], "dash": "solid"},
        "resid": {"name": "Residual", "color": PALETTE[3], "dash": "solid"},
    }

    # -----------------------------
    # PANEL MODE: Stacked subplots
    # -----------------------------
    if mode == "panel":
        n_components = len(components)

        if len(ids) == 1:
            # Single series: vertical stack of components
            uid = ids[0]
            decomp = decomp_results[uid]
            dates = decomp["dates"]

            fig = make_subplots(
                rows=n_components, cols=1,
                shared_xaxes=True,
                subplot_titles=[component_config[c]["name"] for c in components],
                vertical_spacing=0.08,
            )

            for i, comp in enumerate(components, start=1):
                cfg = component_config[comp]
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=decomp[comp],
                        mode="lines",
                        name=cfg["name"],
                        line=dict(color=cfg["color"], width=2, dash=cfg["dash"]),
                        showlegend=False,
                    ),
                    row=i, col=1
                )

                # Add zero line for seasonal and residual
                if comp in ["seasonal", "resid"]:
                    fig.add_hline(
                        y=0, row=i, col=1,
                        line_dash="dot", line_color="gray", opacity=0.5,
                    )

            # Add stats annotation
            if show_stats:
                stats = stats_results[uid]
                stats_text = (
                    f"Trend: {stats['trend_min']:.1f} → {stats['trend_max']:.1f}<br>"
                    f"Seasonal amp: {stats['seasonal_amplitude']:.1f}<br>"
                    f"Residual σ: {stats['resid_std']:.2f}<br>"
                    f"Trend strength: {stats['trend_strength']:.2f}<br>"
                    f"Seasonal strength: {stats['seasonal_strength']:.2f}"
                )
                fig.add_annotation(
                    x=1.02, y=0.5,
                    xref="paper", yref="paper",
                    text=stats_text,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                )

            fig.update_layout(height=200 * n_components)

        else:
            # Multiple series: grid layout
            n_series = len(ids)
            cols = wrap or min(2, n_series)
            rows = ceil(n_series / cols)

            fig = make_subplots(
                rows=rows * n_components, cols=cols,
                shared_xaxes=True,
                vertical_spacing=0.03,
                horizontal_spacing=0.08,
            )

            for s_idx, uid in enumerate(ids):
                decomp = decomp_results[uid]
                dates = decomp["dates"]

                col = (s_idx % cols) + 1
                base_row = (s_idx // cols) * n_components + 1

                for c_idx, comp in enumerate(components):
                    row = base_row + c_idx
                    cfg = component_config[comp]

                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=decomp[comp],
                            mode="lines",
                            name=f"{uid} - {cfg['name']}",
                            line=dict(color=cfg["color"], width=1.5),
                            showlegend=(s_idx == 0),
                            legendgroup=comp,
                        ),
                        row=row, col=col
                    )

                    # Title for first component row
                    if c_idx == 0:
                        fig.add_annotation(
                            x=0.5, y=1.0,
                            xref=f"x{col if col > 1 else ''} domain",
                            yref=f"y{base_row} domain",
                            text=f"<b>{uid}</b>",
                            showarrow=False,
                            font=dict(size=11),
                            yshift=15,
                        )

            fig.update_layout(height=150 * n_components * rows)

    # -----------------------------
    # OVERLAY MODE: All on one plot
    # -----------------------------
    elif mode == "overlay":
        fig = go.Figure()

        for uid in ids:
            decomp = decomp_results[uid]
            dates = decomp["dates"]

            for comp in components:
                cfg = component_config[comp]
                suffix = f" ({uid})" if len(ids) > 1 else ""

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=decomp[comp],
                        mode="lines",
                        name=f"{cfg['name']}{suffix}",
                        line=dict(
                            color=cfg["color"],
                            width=2 if comp == "observed" else 1.5,
                            dash="solid" if comp in ["observed", "trend"] else "dot",
                        ),
                        opacity=1.0 if comp in ["observed", "trend"] else 0.7,
                    )
                )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)

    # -----------------------------
    # DROPDOWN MODE: Interactive selector
    # -----------------------------
    elif mode == "dropdown":
        n_components = len(components)

        fig = make_subplots(
            rows=n_components, cols=1,
            shared_xaxes=True,
            subplot_titles=[component_config[c]["name"] for c in components],
            vertical_spacing=0.08,
        )

        trace_counts = []

        for s_idx, uid in enumerate(ids):
            decomp = decomp_results[uid]
            dates = decomp["dates"]
            local_count = 0

            for c_idx, comp in enumerate(components, start=1):
                cfg = component_config[comp]
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=decomp[comp],
                        mode="lines",
                        name=cfg["name"],
                        line=dict(color=cfg["color"], width=2),
                        visible=(s_idx == 0),
                        showlegend=(s_idx == 0),
                    ),
                    row=c_idx, col=1
                )
                local_count += 1

            trace_counts.append(local_count)

        # Dropdown buttons
        buttons = []
        total = sum(trace_counts)
        start = 0

        for i, uid in enumerate(ids):
            visible = [False] * total
            for k in range(start, start + trace_counts[i]):
                visible[k] = True

            # Build title with stats
            stats = stats_results[uid]
            title = f"{uid} | Trend: {stats['trend_strength']:.2f}, Seasonal: {stats['seasonal_strength']:.2f}"

            buttons.append(dict(
                label=str(uid),
                method="update",
                args=[{"visible": visible}, {"title": title}],
            ))
            start += trace_counts[i]

        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 1.0, "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            }],
            height=200 * n_components,
        )

        # Add zero lines
        for i, comp in enumerate(components, start=1):
            if comp in ["seasonal", "resid"]:
                fig.add_hline(
                    y=0, row=i, col=1,
                    line_dash="dot", line_color="gray", opacity=0.5,
                )

    else:
        raise ValueError("mode must be one of: panel, overlay, dropdown")

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
            y=-0.08,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
        ),
        margin=dict(b=bottom_margin, r=120 if show_stats and mode == "panel" else 50),
    )

    return fig


# =====================================================================
# CONVENIENCE FUNCTION: Get decomposition as DataFrame
# =====================================================================

def decompose_series(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    period: int = None,
    seasonal: int = 7,
    robust: bool = True,
) -> pd.DataFrame:
    """
    Decompose all series and return results as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column identifying series.
    date_col : str
        Date column.
    value_col : str
        Value column.
    period : int, optional
        Seasonal period.
    seasonal : int
        Seasonal smoother length.
    robust : bool
        Use robust fitting.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id_col, date_col, observed, trend, seasonal, resid
    """
    results = []

    for uid, grp in df.groupby(id_col):
        sub = grp.sort_values(date_col).copy()
        series = sub[value_col].reset_index(drop=True)

        decomp = stl_decompose(
            series,
            period=period,
            seasonal=seasonal,
            robust=robust,
        )

        sub = sub.reset_index(drop=True)
        sub["trend"] = decomp["trend"]
        sub["seasonal"] = decomp["seasonal"]
        sub["resid"] = decomp["resid"]

        results.append(sub[[id_col, date_col, value_col, "trend", "seasonal", "resid"]])

    return pd.concat(results, ignore_index=True)


# =====================================================================
# CONVENIENCE FUNCTION: Get decomposition statistics
# =====================================================================

def get_decomposition_stats(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    period: int = None,
    seasonal: int = 7,
    robust: bool = True,
) -> pd.DataFrame:
    """
    Compute decomposition statistics for all series.

    Returns DataFrame with trend_strength, seasonal_strength, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column identifying series.
    date_col : str
        Date column.
    value_col : str
        Value column.
    period : int, optional
        Seasonal period.
    seasonal : int
        Seasonal smoother length.
    robust : bool
        Use robust fitting.

    Returns
    -------
    pd.DataFrame
        Statistics per series.
    """
    results = []

    for uid, grp in df.groupby(id_col):
        sub = grp.sort_values(date_col)
        series = sub[value_col].reset_index(drop=True)

        decomp = stl_decompose(
            series,
            period=period,
            seasonal=seasonal,
            robust=robust,
        )

        stats = compute_decomposition_stats(decomp)
        stats[id_col] = uid
        results.append(stats)

    stats_df = pd.DataFrame(results)

    # Reorder columns
    cols = [id_col] + [c for c in stats_df.columns if c != id_col]
    return stats_df[cols]
