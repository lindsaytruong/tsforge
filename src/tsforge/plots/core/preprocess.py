# tsforge/plots/core/preprocess.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Union, List


def apply_smoothing(df: pd.DataFrame, id_col: str, value_col: str, window: int):
    """Apply rolling mean smoothing while preserving original index alignment."""
    if window is None or window <= 1:
        return df

    df = df.sort_values([id_col, "ds"]).copy()

    # Compute smoothed series per-id AND preserve original index
    smooth_vals = (
        df.groupby(id_col)[value_col]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

    # Replace value_col with smoothed values so plotting uses them
    df[value_col] = smooth_vals
    return df



def aggregate_by_group(df, group_col, date_col, value_col, agg, id_col):
    """Group by category/department/etc., but DO NOT lose original id_col."""
    
    # no grouping → keep original id_col
    if group_col is None:
        return df, id_col

    # single group column
    if isinstance(group_col, str):
        keys = [group_col, date_col]
        df2 = df.groupby(keys, observed=True)[value_col].agg(agg).reset_index()
        return df2, group_col

    # multi-column grouping
    keys = list(group_col) + [date_col]
    df2 = df.groupby(keys, observed=True)[value_col].agg(agg).reset_index()
    df2["_group_id"] = df2[list(group_col)].astype(str).agg("|".join, axis=1)

    return df2, "_group_id"



def resample_df(df, freq, id_col, date_col, value_col, agg):
    if freq is None:
        return df

    return (
        df.set_index(date_col)
          .groupby(id_col)[value_col]
          .resample(freq).agg(agg)
          .reset_index()
    )


def select_ids(df, id_col, ids, max_ids):
    """Standard ID selection logic reused across plots."""
    unique_ids = sorted(df[id_col].unique())

    if ids is None:
        return unique_ids[:max_ids]

    if isinstance(ids, str):
        return [ids]

    if isinstance(ids, int):
        return unique_ids[:ids]

    return [i for i in ids if i in unique_ids]
