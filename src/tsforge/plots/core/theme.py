# tsforge/plots/core/theme.py
from __future__ import annotations
from typing import Dict, Any

THEMES = {
    # =====================================================
    # FORECAST ACADEMY DEFAULT (BRAND)
    # =====================================================
    "fa": {
        "font": "Source Sans Pro",
        "font_size": 14,
        "background": "white",
        "grid_color": "#E7E9EF",
        "axis_color": "#3A3A3A",

        "title_color": "#2C32D5",
        "subtitle_color": "#444444",

        "line_width": 2.3,
        "pi_opacity": 0.20,

        "legend_bg": "rgba(255,255,255,0.80)",
        "legend_border": "rgba(180,180,180,0.35)",
        "legend_font_color": "#222222",
        "legend_position": "top right",
        "legend_font_size": 12,
    },

    # =====================================================
    # McKINSEY-STYLE
    # =====================================================
    "mckinsey": {
        "font": "Helvetica Neue",
        "font_size": 15,
        "background": "white",
        "grid_color": "#D9D9D9",
        "axis_color": "#2B2B2B",

        "title_color": "#1A1A1A",
        "subtitle_color": "#555555",

        "line_width": 2.8,
        "pi_opacity": 0.15,

        "legend_bg": "rgba(255,255,255,0.40)",
        "legend_border": "rgba(150,150,150,0.30)",
        "legend_font_color": "#222222",
        "legend_position": "top right",
        "legend_font_size": 13,
    },

    # =====================================================
    # MINIMALIST (Nordic / Scandinavian)
    # =====================================================
    "minimal": {
        "font": "Inter",
        "font_size": 13,
        "background": "white",
        "grid_color": "rgba(0,0,0,0.07)",
        "axis_color": "#444444",

        "title_color": "#222222",
        "subtitle_color": "#666666",

        "line_width": 1.6,
        "pi_opacity": 0.10,

        "legend_bg": "rgba(255,255,255,0.55)",
        "legend_border": "rgba(210,210,210,0.20)",
        "legend_font_color": "#333333",
        "legend_position": "top right",
        "legend_font_size": 11,
    },

    # =====================================================
    # DARK THEME
    # =====================================================
    "dark": {
        "font": "Arial",
        "font_size": 13,
        "background": "#111111",
        "grid_color": "#333333",
        "axis_color": "#E5E5E5",

        "title_color": "#F8F8F8",
        "subtitle_color": "#CCCCCC",

        "line_width": 2.2,
        "pi_opacity": 0.18,

        "legend_bg": "rgba(20,20,20,0.75)",
        "legend_border": "rgba(255,255,255,0.25)",
        "legend_font_color": "#F0F0F0",   # <-- readable on dark
        "legend_position": "top right",
        "legend_font_size": 12,
    },
    "seaborn": {
        "font": "Arial",
        "font_size": 13,
        "background": "white",
        "grid_color": "#E5E5E5",
        "axis_color": "#4C4C4C",

        # Slightly heavier lines
        "line_width": 2.0,
        "pi_opacity": 0.18,

        # Legend
        "legend_bg": "rgba(255,255,255,0.85)",
        "legend_border": "rgba(200,200,200,0.3)",
        "legend_position": "top right",

        # Optional seaborn-like color palette accent
        "accent_color": "#4C72B0",
    },
    "ggplot": {
        "font": "Georgia",
        "font_size": 13,
        "background": "#EBEBEB",      # light grey panel
        "grid_color": "white",        # white gridlines like ggplot
        "axis_color": "#333333",

        "line_width": 2.5,            # ggplot has strong lines
        "pi_opacity": 0.20,

        # Legend
        "legend_bg": "rgba(245,245,245,0.9)",
        "legend_border": "rgba(120,120,120,0.3)",
        "legend_position": "bottom right",

        # Optional ggplot-style accent
        "accent_color": "#E24A33",
}
}



def apply_theme(fig, theme: str = "fa"):
    """Apply global tsforge theme to a Plotly figure."""
    t = THEMES.get(theme, THEMES["fa"])

    fig.update_layout(
        font=dict(family=t["font"], size=t["font_size"]),
        plot_bgcolor=t["background"],
        paper_bgcolor=t["background"],
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=t["grid_color"],
        color=t["axis_color"],
        tickcolor=t["axis_color"],
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid_color"],
        color=t["axis_color"],
        tickcolor=t["axis_color"],
    )

    return fig



def apply_legend(fig, theme: str):
    """Apply professional top-right translucent FA/McKinsey-style legend."""
    t = THEMES.get(theme, THEMES["fa"])

    legend_font_color = t.get("legend_font_color", t.get("axis_color", "#222"))
    legend_font_size = t.get("legend_font_size", 12)

    fig.update_layout(
        legend=dict(
            bgcolor=t.get("legend_bg", "rgba(255,255,255,0.65)"),
            bordercolor=t.get("legend_border", "rgba(200,200,200,0.4)"),
            borderwidth=1,
            font=dict(size=legend_font_size, color=legend_font_color),
            x=1.0, y=1.12,
            xanchor="right", yanchor="top",
            orientation="h",
        )
    )

    return fig
