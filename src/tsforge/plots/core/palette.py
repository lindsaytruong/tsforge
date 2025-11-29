# tsforge/plots/core/palette.py
from __future__ import annotations

# Color palette used across tsforge plots
PALETTE = [
    "#2E86AB", "#F6C85F", "#6F4E7C", "#C34A36",
    "#9FD356", "#235789", "#C1292E", "#7EA16B",
    "#436436", "#FFC145", "#8B85C1"
]

HIGHLIGHT = "#ff007f"


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
