
import pytimetk as tk
import pandas as pd

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
    """
    Pad a panel of time series to complete frequency using pytimetk.pad_by_time,
    with options to fill values and static categorical fields.

    Parameters
    ----------
    df : DataFrame with [id_col, date_col, value_col, ...].
    id_col : str
        Series identifier column.
    date_col : str
        Date column.
    value_col : str, optional
        Numeric value column.
    freq : str
        Pandas offset alias ('D','W','M').
    fill : int, float, str, or None
        Fill method for value_col. (0, 'ffill', 'bfill', None)
    start_date, end_date : str or pd.Timestamp, optional
        Boundaries for padding.
    fill_static : list of str, optional
        List of static columns to fill with first value from each id.

    Returns
    -------
    DataFrame with padded time series.
    """
    # pad the main series
    df_padded = (
        df.groupby(id_col, group_keys=False)
          .pad_by_time(
              date_column=date_col,
              freq=freq,
              start_date=start_date,
              end_date=end_date
          )
    )

    # fill value_col if requested
    if fill is not None and value_col is not None:
        if fill in ("ffill", "bfill"):
            df_padded[value_col] = df_padded[value_col].fillna(method=fill)
        else:
            df_padded[value_col] = df_padded[value_col].fillna(fill)

    # fill static categorical fields if requested
    if fill_static:
        for col in fill_static:
            if col in df.columns:
                df_padded[col] = (
                    df_padded.groupby(id_col)[col]
                    .transform(lambda x: x.fillna(x.dropna().iloc[0] if len(x.dropna()) else np.nan))
                )

    return df_padded

