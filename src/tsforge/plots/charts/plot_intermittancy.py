# =====================================================================
# tsforge.plots.charts.plot_intermittency
# Intermittency and zero-pattern analysis for time series
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
# INTERMITTENCY CLASSIFICATION
# =====================================================================

def classify_intermittency(
    zero_pct: float,
    cv2: float = None,
    adi: float = None,
) -> str:
    """
    Classify series intermittency based on zero percentage and demand patterns.

    Uses the Syntetos-Boylan classification when CV² and ADI are provided:
      - Smooth: Low variability, regular demand (CV² < 0.49, ADI < 1.32)
      - Erratic: High variability, regular demand (CV² >= 0.49, ADI < 1.32)
      - Intermittent: Low variability, sporadic demand (CV² < 0.49, ADI >= 1.32)
      - Lumpy: High variability, sporadic demand (CV² >= 0.49, ADI >= 1.32)

    Falls back to zero_pct thresholds if CV²/ADI not provided:
      - Continuous: < 10% zeros
      - Low intermittency: 10-30% zeros
      - Moderate intermittency: 30-60% zeros
      - High intermittency: 60-90% zeros
      - Sparse: > 90% zeros

    Parameters
    ----------
    zero_pct : float
        Percentage of zero values (0-100).
    cv2 : float, optional
        Squared coefficient of variation of non-zero demand.
    adi : float, optional
        Average demand interval (mean time between non-zero demands).

    Returns
    -------
    str
        Classification label.
    """
    if cv2 is not None and adi is not None:
        if cv2 < 0.49 and adi < 1.32:
            return "Smooth"
        elif cv2 >= 0.49 and adi < 1.32:
            return "Erratic"
        elif cv2 < 0.49 and adi >= 1.32:
            return "Intermittent"
        else:
            return "Lumpy"
    else:
        if zero_pct < 10:
            return "Continuous"
        elif zero_pct < 30:
            return "Low Intermittency"
        elif zero_pct < 60:
            return "Moderate Intermittency"
        elif zero_pct < 90:
            return "High Intermittency"
        else:
            return "Sparse"


def compute_intermittency_stats(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Compute intermittency statistics for each series.

    Returns DataFrame with:
      - zero_pct: Percentage of zeros
      - nonzero_mean: Mean of non-zero values
      - nonzero_std: Std of non-zero values
      - cv2: Squared coefficient of variation
      - adi: Average demand interval
      - classification: Intermittency class

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column identifying series.
    value_col : str
        Value column to analyze.

    Returns
    -------
    pd.DataFrame
        Statistics per series.
    """
    results = []

    for uid, grp in df.groupby(id_col):
        values = grp[value_col].values
        n = len(values)

        # Zero stats
        n_zeros = (values == 0).sum()
        zero_pct = 100 * n_zeros / n if n > 0 else 0

        # Non-zero stats
        nonzero = values[values > 0]
        if len(nonzero) > 0:
            nonzero_mean = nonzero.mean()
            nonzero_std = nonzero.std()
            cv2 = (nonzero_std / nonzero_mean) ** 2 if nonzero_mean > 0 else 0
        else:
            nonzero_mean = 0
            nonzero_std = 0
            cv2 = 0

        # ADI: Average demand interval
        # Find positions of non-zero values and compute mean interval
        nonzero_idx = np.where(values > 0)[0]
        if len(nonzero_idx) > 1:
            intervals = np.diff(nonzero_idx)
            adi = intervals.mean()
        else:
            adi = n  # All zeros or single demand

        # Classification
        classification = classify_intermittency(zero_pct, cv2, adi)

        results.append({
            id_col: uid,
            "n_obs": n,
            "n_zeros": n_zeros,
            "zero_pct": zero_pct,
            "nonzero_mean": nonzero_mean,
            "nonzero_std": nonzero_std,
            "cv2": cv2,
            "adi": adi,
            "classification": classification,
        })

    return pd.DataFrame(results)


# =====================================================================
# MAIN PLOTTING FUNCTION
# =====================================================================

def plot_intermittency(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    kind: Literal["bar", "scatter", "heatmap", "classification"] = "bar",
    mode: Literal["overlay", "facet"] = "overlay",
    sort_by: Literal["zero_pct", "adi", "cv2", "id"] = "zero_pct",
    top_n: Optional[int] = None,
    show_thresholds: bool = True,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Intermittency and zero-pattern visualization.

    Supports:
      ✓ bar chart of zero percentages
      ✓ scatter plot (CV² vs ADI) with Syntetos-Boylan quadrants
      ✓ heatmap of zeros over time
      ✓ classification summary

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
        Specific IDs to include, number to sample, or None for all.
    group_col : str, list, or None
        Column(s) to group by before analysis.
    agg : str
        Aggregation function if grouping (default: "sum").
    kind : str
        Plot type:
        - "bar": Zero percentage by series
        - "scatter": CV² vs ADI scatter (Syntetos-Boylan)
        - "heatmap": Binary zeros heatmap over time
        - "classification": Summary of classification counts
    mode : str
        Layout: "overlay" or "facet" (for heatmap).
    sort_by : str
        How to sort bar chart: "zero_pct", "adi", "cv2", or "id".
    top_n : int or None
        Limit to top N series (by zero_pct).
    show_thresholds : bool
        Show classification thresholds on scatter plot.
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
    # Compute intermittency stats
    # -----------------------------
    stats_df = compute_intermittency_stats(df, id_col, value_col)

    # -----------------------------
    # ID selection / filtering
    # -----------------------------
    if ids is not None:
        if isinstance(ids, int):
            # Sample N series
            selected = stats_df[id_col].sample(min(ids, len(stats_df))).tolist()
        elif isinstance(ids, str):
            selected = [ids]
        else:
            selected = list(ids)
        stats_df = stats_df[stats_df[id_col].isin(selected)]
        df = df[df[id_col].isin(selected)]

    # Sort
    if sort_by == "zero_pct":
        stats_df = stats_df.sort_values("zero_pct", ascending=False)
    elif sort_by == "adi":
        stats_df = stats_df.sort_values("adi", ascending=False)
    elif sort_by == "cv2":
        stats_df = stats_df.sort_values("cv2", ascending=False)
    else:
        stats_df = stats_df.sort_values(id_col)

    # Top N
    if top_n is not None:
        stats_df = stats_df.head(top_n)
        df = df[df[id_col].isin(stats_df[id_col])]

    # -----------------------------
    # PLOTTING
    # -----------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # -----------------------------
    # BAR CHART: Zero percentage by series
    # -----------------------------
    if kind == "bar":
        # Color by classification
        class_colors = {
            "Continuous": "#2E86AB",
            "Low Intermittency": "#A6D854",
            "Moderate Intermittency": "#F6C85F",
            "High Intermittency": "#E76F51",
            "Sparse": "#9B2335",
            "Smooth": "#2E86AB",
            "Erratic": "#F6C85F",
            "Intermittent": "#E76F51",
            "Lumpy": "#9B2335",
        }

        colors = [class_colors.get(c, PALETTE[0]) for c in stats_df["classification"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=stats_df[id_col].astype(str),
                y=stats_df["zero_pct"],
                marker=dict(color=colors),
                text=stats_df["classification"],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Zero %: %{y:.1f}%<br>"
                    "Class: %{text}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Add threshold lines
        if show_thresholds:
            for pct, label in [(10, "Continuous"), (30, "Low"), (60, "Moderate"), (90, "High")]:
                fig.add_hline(
                    y=pct,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"{pct}%",
                    annotation_position="right",
                )

        fig.update_xaxes(title_text="Series", tickangle=45)
        fig.update_yaxes(title_text="Zero Percentage (%)", range=[0, 105])

    # -----------------------------
    # SCATTER: CV² vs ADI (Syntetos-Boylan)
    # -----------------------------
    elif kind == "scatter":
        class_colors = {
            "Smooth": "#2E86AB",
            "Erratic": "#F6C85F",
            "Intermittent": "#E76F51",
            "Lumpy": "#9B2335",
        }

        fig = go.Figure()

        for classification, grp in stats_df.groupby("classification"):
            fig.add_trace(
                go.Scatter(
                    x=grp["adi"],
                    y=grp["cv2"],
                    mode="markers",
                    name=classification,
                    marker=dict(
                        color=class_colors.get(classification, PALETTE[0]),
                        size=10,
                        opacity=0.7,
                    ),
                    text=grp[id_col],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "ADI: %{x:.2f}<br>"
                        "CV²: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        # Add quadrant lines
        if show_thresholds:
            fig.add_vline(x=1.32, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_hline(y=0.49, line_dash="dash", line_color="gray", opacity=0.7)

            # Quadrant labels
            x_max = max(stats_df["adi"].max() * 1.1, 3)
            y_max = max(stats_df["cv2"].max() * 1.1, 1.5)

            annotations = [
                dict(x=0.66, y=0.25, text="Smooth", showarrow=False, font=dict(size=12, color="gray")),
                dict(x=0.66, y=y_max * 0.8, text="Erratic", showarrow=False, font=dict(size=12, color="gray")),
                dict(x=x_max * 0.7, y=0.25, text="Intermittent", showarrow=False, font=dict(size=12, color="gray")),
                dict(x=x_max * 0.7, y=y_max * 0.8, text="Lumpy", showarrow=False, font=dict(size=12, color="gray")),
            ]
            fig.update_layout(annotations=annotations)

        fig.update_xaxes(title_text="Average Demand Interval (ADI)")
        fig.update_yaxes(title_text="Squared CV (CV²)")

    # -----------------------------
    # HEATMAP: Binary zeros over time
    # -----------------------------
    elif kind == "heatmap":
        # Pivot: rows = series, cols = dates, values = 0/1 for zero/nonzero
        series_list = stats_df[id_col].tolist()
        df_filtered = df[df[id_col].isin(series_list)].copy()

        # Create binary indicator
        df_filtered["is_zero"] = (df_filtered[value_col] == 0).astype(int)

        # Pivot
        pivot = df_filtered.pivot_table(
            index=id_col,
            columns=date_col,
            values="is_zero",
            aggfunc="max",
        ).fillna(0)

        # Reorder rows by zero_pct
        row_order = stats_df[id_col].tolist()
        pivot = pivot.reindex(row_order)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index.astype(str),
                colorscale=[[0, "#2E86AB"], [1, "#E5E5E5"]],
                showscale=False,
                hovertemplate=(
                    "Series: %{y}<br>"
                    "Date: %{x}<br>"
                    "Zero: %{z}<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Series", autorange="reversed")
        fig.update_layout(height=max(400, len(series_list) * 20))

    # -----------------------------
    # CLASSIFICATION: Summary counts
    # -----------------------------
    elif kind == "classification":
        class_counts = stats_df["classification"].value_counts()

        class_colors = {
            "Continuous": "#2E86AB",
            "Low Intermittency": "#A6D854",
            "Moderate Intermittency": "#F6C85F",
            "High Intermittency": "#E76F51",
            "Sparse": "#9B2335",
            "Smooth": "#2E86AB",
            "Erratic": "#F6C85F",
            "Intermittent": "#E76F51",
            "Lumpy": "#9B2335",
        }

        colors = [class_colors.get(c, PALETTE[0]) for c in class_counts.index]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker=dict(color=colors),
                text=class_counts.values,
                textposition="outside",
                showlegend=False,
            )
        )

        total = class_counts.sum()
        # Add percentage annotations
        for i, (cls, count) in enumerate(class_counts.items()):
            pct = 100 * count / total
            fig.add_annotation(
                x=cls,
                y=count,
                text=f"{pct:.1f}%",
                showarrow=False,
                yshift=25,
                font=dict(size=11, color="gray"),
            )

        fig.update_xaxes(title_text="Classification")
        fig.update_yaxes(title_text="Number of Series")

    else:
        raise ValueError("kind must be one of: bar, scatter, heatmap, classification")

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
            y=-0.15,
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
# CONVENIENCE FUNCTION: Quick zero stats
# =====================================================================

def get_zero_stats(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Quick utility to get zero percentage for all series.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column identifying series.
    value_col : str
        Value column.

    Returns
    -------
    pd.DataFrame
        DataFrame with id_col and zero_pct columns.
    """
    return (
        df.groupby(id_col)[value_col]
        .apply(lambda x: 100 * (x == 0).mean())
        .reset_index()
        .rename(columns={value_col: "zero_pct"})
        .sort_values("zero_pct", ascending=False)
    )