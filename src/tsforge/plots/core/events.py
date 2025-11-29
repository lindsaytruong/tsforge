# tsforge/plots/core/events.py
from __future__ import annotations
import pandas as pd
from typing import Optional, Union


def extract_inline_events(events, df, id_col, date_col, label_col):
    if events is None:
        return None
    if events not in df.columns:
        raise ValueError(f"Inline events='{events}' not found.")

    col = df[events]

    if col.dtype == bool or pd.api.types.is_numeric_dtype(col):
        mask = col.astype(bool)
        labels = pd.Series(events, index=df.index)
    else:
        mask = col.notna()
        labels = col.astype(str)

    ev = df.loc[mask, [id_col, date_col]].copy()
    ev[label_col] = labels[mask].values
    return ev


def normalize_events_df(ev, df, id_col, date_col, label_col):
    if ev is None:
        return None

    ev = ev.copy()
    if date_col not in ev.columns:
        raise ValueError("Events DF must contain date_col.")

    ev[date_col] = pd.to_datetime(ev[date_col])

    # Broadcast if id_col not present
    if id_col not in ev.columns:
        uids = df[[id_col]].drop_duplicates()
        ev["_k"] = 1
        uids["_k"] = 1
        ev = ev.merge(uids, on="_k").drop(columns="_k")

    if label_col not in ev.columns:
        ev[label_col] = "event"

    return ev[[id_col, date_col, label_col]]


def merge_all_events(
    df,
    id_col,
    date_col,
    event_label_col,
    inline,
    global_events,
    local_events,
    direct_df,
):
    collected = []

    # inline
    inline_df = extract_inline_events(inline, df, id_col, date_col, event_label_col)
    if inline_df is not None:
        collected.append(inline_df)

    # global / local / direct
    for source in (global_events, local_events, direct_df):
        if source is not None:
            normalized = normalize_events_df(
                source,
                df,
                id_col,
                date_col,
                event_label_col,
            )
            collected.append(normalized)

    if not collected:
        return None

    return (
        pd.concat(collected, ignore_index=True)
          .drop_duplicates([id_col, date_col, event_label_col])
          .sort_values([date_col, id_col])
    )

