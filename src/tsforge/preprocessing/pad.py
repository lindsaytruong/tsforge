import pandas as pd
import numpy as np
from utilsforecast.preprocessing import fill_gaps

def pad_by_time(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str | None = None,
    freq: str = "D",
    fill: int | float | str | None = None,
    start_date=None,
    end_date=None,
    fill_static: list[str] | None = None
) -> pd.DataFrame:
    """Pad a panel of time series using Nixtla's fill_gaps, with static column filling."""

    # use Nixtla to pad missing dates
    df_padded = fill_gaps(
        df[[id_col, date_col] + ([value_col] if value_col else []) + (fill_static or [])],
        id_col=id_col,
        time_col=date_col,
        freq=freq
    )

    # fill the value column
    if fill is not None and value_col is not None:
        if fill in ("ffill", "bfill"):
            df_padded[value_col] = df_padded.groupby(id_col)[value_col].fillna(method=fill)
        else:
            df_padded[value_col] = df_padded[value_col].fillna(fill)

    # fill static columns
    if fill_static:
        for col in fill_static:
            df_padded[col] = (
                df_padded.groupby(id_col)[col]
                .transform(lambda x: x.fillna(x.dropna().iloc[0] if x.notna().any() else np.nan))
            )

    return df_padded
