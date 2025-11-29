# tsforge/plots/core/base.py
"""
Base utilities that do not belong to a specific category.
Can hold future shared functions.
"""

def ensure_sorted(df, cols):
    return df.sort_values(cols)
