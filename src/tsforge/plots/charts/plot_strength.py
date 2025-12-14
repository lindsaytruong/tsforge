# =====================================================================
# tsforge.plots.charts.plot_strength
# Strength distribution visualization for trend and seasonal components
# =====================================================================

import pandas as pd
from typing import Optional, List, Literal

from tsforge.plots.core.palette import PALETTE
from tsforge.plots.core.theme import apply_theme


def plot_strength(
    profiles: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 30,
    thresholds: Optional[List[float]] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Visualize strength distributions for trend and seasonal components.

    Creates side-by-side histograms showing the distribution of strength
    metrics with configurable threshold lines for weak/moderate/strong
    classification boundaries.

    Parameters
    ----------
    profiles : pd.DataFrame
        DataFrame containing strength columns (e.g., from ts feature extraction).
        Expected columns: 'trend_strength', 'seasonal_strength' or as specified.
    columns : list of str, optional
        Column names to plot. Default: ['trend_strength', 'seasonal_strength'].
    bins : int
        Number of histogram bins (default: 30).
    thresholds : list of float, optional
        Threshold values for vertical lines. Default: [0.3, 0.6].
    theme : str
        Visual theme (default: "fa").
    style : dict or None
        Custom styling overrides. Supports:
        - 'title': Main title
        - 'subtitle': Subtitle
        - 'threshold_labels': List of labels for thresholds (e.g., ['Weak/Moderate', 'Moderate/Strong'])
    engine : str
        Plotting engine (default: "plotly").

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> from tsforge.plots import plot_strength
    >>> fig = plot_strength(profiles)
    >>> fig.show()

    >>> # Custom thresholds
    >>> fig = plot_strength(profiles, thresholds=[0.25, 0.5, 0.75])
    """
    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Defaults
    if columns is None:
        columns = ['trend_strength', 'seasonal_strength']

    if thresholds is None:
        thresholds = [0.3, 0.6]

    # Default threshold colors
    threshold_colors = ['#FFA500', '#E63946']  # orange, red

    # Default labels for display
    titles = {
        'trend_strength': 'Trend Strength',
        'seasonal_strength': 'Seasonal Strength',
    }

    # Threshold labels
    threshold_labels = ['Weak/Moderate', 'Moderate/Strong']
    if style and 'threshold_labels' in style:
        threshold_labels = style['threshold_labels']

    # Validate columns exist
    available_cols = [col for col in columns if col in profiles.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns {columns} found in profiles DataFrame.")

    n_cols = len(available_cols)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[titles.get(col, col.replace('_', ' ').title()) for col in available_cols],
        horizontal_spacing=0.1,
    )

    # Add histograms
    for i, col in enumerate(available_cols, start=1):
        data = profiles[col].dropna()

        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker=dict(
                    color=PALETTE[i - 1],
                    line=dict(color='white', width=1),
                ),
                opacity=0.7,
                name=titles.get(col, col),
                showlegend=False,
            ),
            row=1, col=i
        )

        # Add threshold lines
        for j, threshold in enumerate(thresholds):
            label = threshold_labels[j] if j < len(threshold_labels) else f"Threshold {threshold}"
            color = threshold_colors[j % len(threshold_colors)]

            fig.add_vline(
                x=threshold,
                row=1, col=i,
                line_dash="dash",
                line_color=color,
                line_width=2,
                annotation_text=label if i == 1 else None,  # Only label on first plot
                annotation_position="top right" if j == 0 else "top left",
                annotation_font_size=10,
            )

        # Update axis labels
        fig.update_xaxes(title_text=titles.get(col, col), row=1, col=i)
        fig.update_yaxes(title_text="Count", row=1, col=i)

    # Layout
    fig.update_layout(
        height=400,
        width=600 * n_cols,
        showlegend=True,
    )

    # Add legend for thresholds (manual)
    for j, threshold in enumerate(thresholds):
        label = threshold_labels[j] if j < len(threshold_labels) else f"{threshold}"
        color = threshold_colors[j % len(threshold_colors)]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=f"{label} ({threshold})",
                line=dict(color=color, dash='dash', width=2),
                showlegend=True,
            )
        )

    # Apply theme
    fig = apply_theme(fig, theme)

    # Custom styling
    if style:
        if "title" in style and "subtitle" not in style:
            fig.update_layout(title=style["title"])
        if "title" in style and "subtitle" in style:
            fig.update_layout(
                title=f"{style['title']}<br><sup>{style['subtitle']}</sup>"
            )

    # Legend positioning
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
        margin=dict(b=100),
    )

    return fig
