import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal
# Assuming these exist in your library
from tsforge.plots.style import _apply_tsforge_style, PALETTE, HIGHLIGHT

def plot_seasonal(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",               # aggregation if group_col is used
    seasonal_agg: str = "mean",     # aggregation within seasonal cycle
    freq: str = "M",                # "M","Q","W","D"
    mode: str = "overlay",          # overlay | facet | dropdown
    kind: Literal["line", "box"] = "line",
    normalize: bool = False,        # NEW: Normalize each year by its mean (removes trend)
    show_mean: bool = False,        # NEW: Show the Grand Mean (Average Cycle) on top
    engine: str = "plotly",         # plotly | matplotlib
):
    """
    Seasonal subseries plot: collapse time into a cycle (e.g., months in a year)
    and show either lines (yearly trajectories) or boxplots (distribution across years).

    Parameters
    ----------
    ids : None | int | str | list
        Selection of series to plot.
    group_col : str or list, optional
        Aggregate series before plotting.
    agg : str, default 'sum'
        Aggregation logic for group_col.
    seasonal_agg : str, default 'mean'
        Aggregation for collapsing multiple observations within (id, year, cycle).
    freq : {"M","Q","W","D"}
        Seasonal cycle for x-axis.
    mode : {"overlay","facet","dropdown"}
        Layout mode.
    kind : {"line","box"}, default "line"
        Line = subseries lines per year.
        Box = boxplots by seasonal position.
    normalize : bool, default False
        If True, divides each year's data by that year's mean. 
        Useful for comparing seasonal shapes when there is a strong trend.
    show_mean : bool, default False
        If True, plots a thick dashed line representing the average seasonality across all years.
        Only applies when kind="line".
    engine : {"plotly","matplotlib"}
        Plotting backend.
    """

    # ---- prep
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ---- group_col aggregation
    if group_col:
        if isinstance(group_col, str):
            group_keys = [group_col, date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            id_col = group_col
        else:
            group_keys = list(group_col) + [date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            df["_group_id"] = df[list(group_col)].astype(str).agg("|".join, axis=1)
            id_col = "_group_id"

    # ---- normalize ids
    unique_ids = df[id_col].dropna().unique().tolist()
    if ids is None:
        n_series = min(6, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, int):
        n_series = min(ids, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]; n_series = 1
    else:
        ids = list(ids); n_series = len(ids)

    df_sub = df[df[id_col].isin(ids)].copy()

    # ---- seasonal cycle extraction
    df_sub["year"] = df_sub[date_col].dt.year
    f = freq.upper()
    if f.startswith("M"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.month
        x_labels = ["Jan","Feb","Mar","Apr","May","Jun", "Jul","Aug","Sep","Oct","Nov","Dec"]
    elif f.startswith("Q"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.quarter
        x_labels = ["Q1","Q2","Q3","Q4"]
    elif f.startswith("W"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.isocalendar().week.astype(int)
        x_labels = None
    elif f.startswith("D"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.dayofyear
        x_labels = None
    else:
        raise ValueError(f"Unsupported freq '{freq}'. Use 'M','Q','W','D'.")

    # ---- NEW: Normalization Logic (Trend Removal)
    # We do this BEFORE collapsing to one point per season to ensure the mean is accurate
    y_label = value_col
    if normalize:
        # Calculate yearly mean per ID
        yearly_means = df_sub.groupby([id_col, "year"])[value_col].transform("mean")
        df_sub[value_col] = df_sub[value_col] / yearly_means
        y_label = f"{value_col} (Normalized)"

    # ---- collapse to one point per (id, year, seasonal_x)
    df_sub = (
        df_sub
        .groupby([id_col, "year", "seasonal_x"], observed=True, as_index=False)[value_col]
        .agg(seasonal_agg)
    )

    # ---- NEW: Grand Mean Calculation (Signal)
    mean_seasonality = None
    if show_mean and kind == "line":
        mean_seasonality = (
            df_sub.groupby([id_col, "seasonal_x"], observed=True)[value_col]
            .mean()
            .reset_index()
        )

    # =========================
    # KIND: BOX
    # =========================
    if kind == "box":
        if engine == "plotly":
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            box_trace_style = dict(pointpos=0, jitter=0.2, marker=dict(opacity=0.6, size=5))
            
            # Common layout update helper
            def _update_box_layout(fig, title_suffix=""):
                fig.update_layout(
                    title=dict(text=f"Seasonal Box Plot {title_suffix} (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                    xaxis_title=f"Cycle position ({freq})",
                    yaxis_title=y_label,
                )
                if x_labels:
                    fig.update_xaxes(tickmode="array", tickvals=list(range(1, len(x_labels)+1)), ticktext=x_labels)
                return fig

            if mode == "overlay":
                fig = px.box(df_sub, x="seasonal_x", y=value_col, points="all", color_discrete_sequence=[HIGHLIGHT])
                fig.update_traces(**box_trace_style)
                fig = _update_box_layout(fig)
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "facet":
                fig = make_subplots(rows=n_series, cols=1, subplot_titles=[str(uid) for uid in ids])
                for r, uid in enumerate(ids, 1):
                    sub = df_sub[df_sub[id_col] == uid]
                    box = px.box(sub, x="seasonal_x", y=value_col, points="all", color_discrete_sequence=[HIGHLIGHT])
                    box.update_traces(**box_trace_style)
                    for trace in box.data:
                        fig.add_trace(trace, row=r, col=1)
                fig.update_layout(height=300*n_series, title=f"Seasonal Box Facet (freq={freq})")
                # Fix y-axis label for facets? Often better to just leave generic or add annotation
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "dropdown":
                fig = go.Figure()
                traces_per_series = None
                for i, uid in enumerate(ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    box = px.box(sub, x="seasonal_x", y=value_col, points="all", color_discrete_sequence=[HIGHLIGHT])
                    box.update_traces(**box_trace_style)
                    if traces_per_series is None: traces_per_series = len(box.data)
                    for trace in box.data:
                        trace.visible = (i == 0)
                        trace.name = f"{uid}"
                        fig.add_trace(trace)
                
                # dropdown buttons logic
                buttons = []
                for i, uid in enumerate(ids):
                    visible = [False] * (traces_per_series * len(ids))
                    for j in range(traces_per_series):
                        visible[i*traces_per_series + j] = True
                    buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}]))
                
                fig.update_layout(
                    updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.05, "y": 1.15, "xanchor": "left"}],
                )
                fig = _update_box_layout(fig, title_suffix="Dropdown")
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

        elif engine == "matplotlib":
            import matplotlib.pyplot as plt
            if mode != "overlay":
                raise ValueError("Matplotlib seasonal box only supports overlay mode.")
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped = [df_sub[df_sub["seasonal_x"] == s][value_col].dropna() for s in sorted(df_sub["seasonal_x"].unique())]
            ax.boxplot(grouped, tick_labels=x_labels or sorted(df_sub["seasonal_x"].unique()))
            ax.set_title(f"Seasonal Box Plot (freq={freq})", fontsize=14, pad=20)
            ax.set_xlabel(f"Cycle position ({freq})")
            ax.set_ylabel(y_label)
            plt.tight_layout()
            return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

    # =========================
    # KIND: LINE
    # =========================
    if kind == "line":
        # Opacity logic: if showing mean, fade the spaghetti lines
        line_opacity = 0.3 if show_mean else 0.9

        if engine == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Common layout update helper
            def _update_line_layout(fig, title_suffix=""):
                fig.update_layout(
                    title=dict(text=f"Seasonal Subseries Plot {title_suffix} (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                    xaxis_title=f"Cycle position ({freq})",
                    yaxis_title=y_label,
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(range(1, len(x_labels)+1)) if x_labels else None,
                        ticktext=x_labels if x_labels else None
                    )
                )
                return fig

            if mode == "overlay":
                fig = go.Figure()
                for i, (uid, sub) in enumerate(df_sub.groupby(id_col, observed=True)):
                    # 1. Plot spaghetti years
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers", connectgaps=False,
                            name=f"{uid}-{year}" if n_series > 1 else str(year),
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=line_opacity
                        ))
                    
                    # 2. NEW: Plot Grand Mean
                    if show_mean and mean_seasonality is not None:
                        mean_g = mean_seasonality[mean_seasonality[id_col] == uid].sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=mean_g["seasonal_x"], y=mean_g[value_col],
                            mode="lines",
                            name=f"MEAN ({uid})" if n_series > 1 else "MEAN SEASONALITY",
                            line=dict(color="black", width=4, dash="dash"),
                            opacity=1.0
                        ))

                fig = _update_line_layout(fig)
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "facet":
                fig = make_subplots(rows=n_series, cols=1, shared_xaxes=True, subplot_titles=[str(uid) for uid in ids])
                for r, uid in enumerate(ids, 1):
                    sub = df_sub[df_sub[id_col] == uid]
                    # Spaghetti
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers", connectgaps=False,
                            name=f"{uid}-{year}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=line_opacity,
                            showlegend=(r == 1)
                        ), row=r, col=1)
                    
                    # Grand Mean
                    if show_mean and mean_seasonality is not None:
                        mean_g = mean_seasonality[mean_seasonality[id_col] == uid].sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=mean_g["seasonal_x"], y=mean_g[value_col],
                            mode="lines", name="MEAN",
                            line=dict(color="black", width=4, dash="dash"),
                            opacity=1.0,
                            showlegend=(r == 1)
                        ), row=r, col=1)

                fig.update_layout(height=300*n_series, title=dict(text=f"Seasonal Subseries Facet (freq={freq})", y=0.98))
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "dropdown":
                # Note: Logic gets complex with dropdowns + extra mean traces. 
                # Simplified strategy: Treat Mean as just another trace attached to that ID index.
                fig = go.Figure()
                traces_per_series = [] # store count of traces for each ID
                
                for i, uid in enumerate(ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    current_trace_count = 0
                    
                    # Spaghetti
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        trace = go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers", connectgaps=False,
                            name=f"{uid}-{year}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=line_opacity,
                            visible=(i == 0)
                        )
                        fig.add_trace(trace)
                        current_trace_count += 1
                    
                    # Grand Mean
                    if show_mean and mean_seasonality is not None:
                        mean_g = mean_seasonality[mean_seasonality[id_col] == uid].sort_values("seasonal_x")
                        trace = go.Scatter(
                            x=mean_g["seasonal_x"], y=mean_g[value_col],
                            mode="lines", name="MEAN SEASONALITY",
                            line=dict(color="black", width=4, dash="dash"),
                            opacity=1.0,
                            visible=(i == 0)
                        )
                        fig.add_trace(trace)
                        current_trace_count += 1
                    
                    traces_per_series.append(current_trace_count)

                # Update buttons
                buttons = []
                current_idx = 0
                for i, uid in enumerate(ids):
                    # Construct visibility vector
                    visible = [False] * sum(traces_per_series)
                    # The range for this UID
                    start = sum(traces_per_series[:i])
                    end = start + traces_per_series[i]
                    for k in range(start, end):
                        visible[k] = True
                    
                    buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}]))

                fig.update_layout(
                    updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.05, "y": 1.15, "xanchor": "left"}],
                )
                fig = _update_line_layout(fig, title_suffix="Dropdown")
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

        elif engine == "matplotlib":
            import matplotlib.pyplot as plt

            if mode == "overlay":
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, (uid, sub) in enumerate(df_sub.groupby(id_col, observed=True)):
                    # Spaghetti
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        ax.plot(g["seasonal_x"], g[value_col],
                                label=f"{uid}-{year}" if n_series > 1 else str(year),
                                color=PALETTE[j % len(PALETTE)], lw=2, alpha=line_opacity)
                    
                    # Grand Mean
                    if show_mean and mean_seasonality is not None:
                        mean_g = mean_seasonality[mean_seasonality[id_col] == uid].sort_values("seasonal_x")
                        ax.plot(mean_g["seasonal_x"], mean_g[value_col],
                                label="Mean Seasonality", color="black", lw=3, linestyle="--")

                ax.set_title(f"Seasonal Subseries Plot (freq={freq})", fontsize=14, pad=20)
                ax.set_xlabel(f"Cycle position ({freq})")
                ax.set_ylabel(y_label)
                if x_labels:
                    ax.set_xticks(range(1, len(x_labels)+1))
                    ax.set_xticklabels(x_labels)
                ax.legend()
                plt.tight_layout()
                return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

            elif mode == "facet":
                fig, axes = plt.subplots(n_series, 1, figsize=(10, 4*n_series), sharex=True)
                axes = np.atleast_1d(axes)
                for ax, uid in zip(axes, ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    # Spaghetti
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        ax.plot(g["seasonal_x"], g[value_col],
                                label=f"{uid}-{year}",
                                color=PALETTE[j % len(PALETTE)], lw=2, alpha=line_opacity)
                    
                    # Grand Mean
                    if show_mean and mean_seasonality is not None:
                        mean_g = mean_seasonality[mean_seasonality[id_col] == uid].sort_values("seasonal_x")
                        ax.plot(mean_g["seasonal_x"], mean_g[value_col],
                                label="Mean", color="black", lw=3, linestyle="--")
                    
                    ax.set_title(f"Series {uid}")
                    ax.legend()
                
                axes[-1].set_xlabel(f"Cycle position ({freq})")
                plt.tight_layout()
                return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

    raise ValueError("Invalid combination of kind, mode, and engine.")