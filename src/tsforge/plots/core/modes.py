"""
Plot mode assembly utilities for tsforge.

Each mode accepts pre-built components (traces, shapes, annotations)
and arranges them into a Plotly figure using a consistent interface.

Supported modes:
- overlay
- facet
- dropdown
"""

from __future__ import annotations
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =====================================================================
# OVERLAY MODE
# =====================================================================
def assemble_overlay_mode(
    traces: List[Any],
    shapes: List[Any],
    annotations: List[Any],
    title: str = "",
) -> go.Figure:
    """Return a standard overlay figure."""
    fig = go.Figure()

    for tr in traces:
        fig.add_trace(tr)

    # Layout-level objects
    if shapes:
        fig.update_layout(shapes=shapes)

    if annotations:
        fig.update_layout(annotations=annotations)

    if title:
        fig.update_layout(title=title)

    return fig



# =====================================================================
# FACET MODE
# =====================================================================
def assemble_facet_mode(
    facet_map: Dict[str, List[Any]],
    shapes: List[Any],
    annotations_top: List[Any],
    title: str = "",
) -> go.Figure:
    """
    facet_map = {
        "series_name": [trace1, trace2, ...],
        ...
    }

    Returns a vertically stacked subplot with shared X-axis.
    """
    n = len(facet_map)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=list(facet_map.keys()),
    )

    # Add traces per row
    for row_idx, (uid, tr_list) in enumerate(facet_map.items(), start=1):
        for tr in tr_list:
            fig.add_trace(tr, row=row_idx, col=1)

    # Add shapes to all rows
    if shapes:
        # must attach row/col to vlines explicitly
        annotated_shapes = []
        for s in shapes:
            for r in range(1, n + 1):
                s2 = s.copy()
                s2["xref"] = f"x{r}"
                s2["yref"] = f"y{r}"
                annotated_shapes.append(s2)
        fig.update_layout(shapes=annotated_shapes)

    # Top annotations (event labels)
    if annotations_top:
        fig.update_layout(annotations=annotations_top)

    if title:
        fig.update_layout(title=title)

    fig.update_layout(height=260 * n)
    return fig



# =====================================================================
# DROPDOWN MODE
# =====================================================================
def assemble_dropdown_mode(
    traces: Dict[str, List[Any]],
    shapes: List[Any],
    annotations: List[Any],
    title: str = "",
) -> go.Figure:
    """
    traces = {
        "series_a": [trace_idx_1, trace_idx_2, ...],
        "series_b": [...],
    }

    This function builds a single figure with all traces invisible except
    the first group. A dropdown menu toggles visibility.
    """
    fig = go.Figure()

    # Add all traces (initially invisible)
    for uid, tr_list in traces.items():
        for tr in tr_list:
            tr.visible = False
            fig.add_trace(tr)

    # First series visible
    first_uid = list(traces.keys())[0]
    for idx in traces[first_uid]:
        fig.data[idx].visible = True

    # Build dropdown menu
    buttons = []
    total = len(fig.data)

    for uid, tr_idxs in traces.items():
        visibility = [False] * total
        for idx in tr_idxs:
            visibility[idx] = True

        buttons.append(
            dict(
                label=str(uid),
                method="update",
                args=[{"visible": visibility}, {"title": f"{title} — {uid}"}],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.0,
                y=1.15,
                xanchor="right",
                yanchor="top",
            )
        ]
    )

    # shapes & annotations
    if shapes:
        fig.update_layout(shapes=shapes)
    if annotations:
        fig.update_layout(annotations=annotations)

    if title:
        fig.update_layout(title=title)

    return fig
