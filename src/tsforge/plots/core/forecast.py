# tsforge/plots/core/forecast.py
from __future__ import annotations


def pi_column_names(forecast_value_col, level, lo_pattern, hi_pattern):
    """Return lower & upper PI column names."""
    lo = lo_pattern.format(col=forecast_value_col, level=level)
    hi = hi_pattern.format(col=forecast_value_col, level=level)
    return lo, hi
