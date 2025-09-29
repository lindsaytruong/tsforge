import pandas as pd
import numpy as np
from scipy.stats import skew

def summary_diagnostics(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Profile time series data by ID.
    Inspired by R timetk::tk_summary_diagnostics, slimmed down
    for practical forecasting use.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with id, date, and value columns.
    id_col : str
        Column with unique series identifier.
    date_col : str
        Datetime column.
    value_col : str
        Target variable column.

    Returns
    -------
    pd.DataFrame
        One row per series with the following diagnostics:

        - id_col : Identifier of the series.
        - start_date : Earliest timestamp in the series.
        - end_date : Latest timestamp in the series.
        - n_obs : Number of observed (non-NA) values in the series.
        - freq : Inferred time series frequency (e.g., "D", "W", "M").
        - n_expected : Number of observations expected between start_date
          and end_date if the series were perfectly regular.
        - is_regular : Boolean flag, True if the series has exactly n_expected
          unique timestamps, False otherwise.
        - pct_missing : Percentage of expected timestamps that are missing.
        - mean_value : Mean of the observed target values.
        - sd_value : Standard deviation of the observed target values.
        - cv_value : Coefficient of variation (std ÷ mean), a measure of relative volatility.
        - skewness : Skew of the value distribution; positive = right-tailed,
          negative = left-tailed.
        - pct_zeros : Percent of observations equal to zero (measure of intermittency).
        - pct_outliers : Percent of observations flagged as outliers (z-score > 3).
        - trend_strength : R² from a simple linear fit, indicating how much
          of the variation is explained by a trend (0 = no trend, 1 = strong trend).
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    results = []

    for key, g in df.groupby(id_col):
        g = g.sort_values(date_col)
        values = g[value_col].dropna()

        # Coverage
        start_date = g[date_col].min()
        end_date = g[date_col].max()
        n_obs = len(values)

        # Infer frequency (with fallback)
        try:
            freq = pd.infer_freq(g[date_col])
        except Exception:
            freq = None

        if not freq and len(g) > 1:
            diffs = g[date_col].diff().dropna().dt.days
            if len(diffs) > 0:
                mode = diffs.mode().iloc[0]
                freq = f"{mode}D"

        # Regularity / missingness
        if freq:
            expected_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
            n_expected = len(expected_idx)
            is_regular = len(g[date_col].unique()) == n_expected
            pct_missing = (n_expected - n_obs) / n_expected * 100
        else:
            n_expected = np.nan
            is_regular = np.nan
            pct_missing = np.nan

        # Value stats
        if n_obs > 0:
            mean_value = values.mean()
            sd_value = values.std()
            cv_value = sd_value / mean_value if mean_value != 0 else np.nan
            skewness = skew(values) if sd_value > 0 else np.nan
        else:
            mean_value = sd_value = cv_value = skewness = np.nan

        # Intermittency
        n_zeros = (values == 0).sum()
        pct_zeros = n_zeros / n_obs * 100 if n_obs else np.nan

        # Outliers (z > 3)
        if n_obs > 1 and sd_value > 0:
            zscores = (values - mean_value) / sd_value
            pct_outliers = (np.abs(zscores) > 3).mean() * 100
        else:
            pct_outliers = np.nan

        # Trend strength (R² of linear fit)
        if n_obs > 1 and sd_value > 0:
            t = np.arange(n_obs)
            coeffs = np.polyfit(t, values, 1)
            fit = np.polyval(coeffs, t)
            ss_res = np.sum((values - fit) ** 2)
            ss_tot = np.sum((values - mean_value) ** 2)
            trend_strength = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            trend_strength = np.nan

        results.append({
            id_col: key,
            "start_date": start_date,
            "end_date": end_date,
            "n_obs": n_obs,
            "freq": freq,
            "n_expected": n_expected,
            "is_regular": is_regular,
            "pct_missing": pct_missing,
            "mean_value": mean_value,
            "sd_value": sd_value,
            "cv_value": cv_value,
            "skewness": skewness,
            "pct_zeros": pct_zeros,
            "pct_outliers": pct_outliers,
            "trend_strength": trend_strength,
        })

    return pd.DataFrame(results)
