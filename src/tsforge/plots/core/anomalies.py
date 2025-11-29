# tsforge/plots/core/anomalies.py
from __future__ import annotations
import pandas as pd


def normalize_anomalies(anoms, df, id_col, date_col, flag_value):
    if anoms is None:
        return None

    if isinstance(anoms, str):
        if anoms not in df.columns:
            raise ValueError(f"Anomaly column '{anoms}' not found.")

        col = df[anoms]
        if col.dtype == bool or pd.api.types.is_numeric_dtype(col):
            mask = (col == flag_value)
        else:
            mask = col.astype(bool)

        return df.loc[mask, [id_col, date_col]].copy()

    an = anoms.copy()
    an[date_col] = pd.to_datetime(an[date_col])
    return an[[id_col, date_col]]
