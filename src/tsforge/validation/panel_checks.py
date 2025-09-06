import pandas as pd
import numpy as np
from scipy import stats

def check_panel(df, id_col: str, date_col: str, freq: str = "D"):
    """
    Quick readiness check for a panel time-series dataframe.

    Parameters
    ----------
    df : DataFrame
        Long-format dataframe with [id_col, date_col, value_col].
    id_col : str
        Column identifying the series (e.g. 'unique_id').
    date_col : str
        Date column.
    freq : str
        Expected frequency ('D','W','M').

    Returns
    -------
    dict with counts of duplicates, continuity issues, and type checks.
    """
    report = {}

    # 1. Duplicate id-date rows
    dups = df.duplicated([id_col, date_col]).sum()
    report["duplicate_rows"] = int(dups)

    # 2. Continuity check
    cont_issues = 0
    for _, g in df.groupby(id_col):
        expected = pd.date_range(g[date_col].min(), g[date_col].max(), freq=freq)
        missing = expected.difference(g[date_col])
        cont_issues += len(missing)
    report["continuity_gaps"] = int(cont_issues)

    # 3. Type checks
    report["date_dtype"] = str(df[date_col].dtype)
    report["n_ids"] = df[id_col].nunique()
    report["n_obs"] = len(df)

    return report
