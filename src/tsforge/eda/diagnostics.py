import numpy as np
import pandas as pd
import scipy.stats as st
from tsfeatures import *

from .ts_features_extension import (
    ADI,
    MI_top_k_lags,
    MI_top_k_lags_indices,
    hurst_exp_dfa,
    hyndman_forecastability,
    longest_zero_streak,
    lya_exp,
    monthly_MASE_score,
    number_of_leading_zeros,
    number_of_trailing_zeros,
    overdispersion,
    pct_zeros,
    permutation_entropy,
    quarterly_MASE_score,
    yearly_MASE_score,
)


def pct_missing_dates(x):
    freq = pd.infer_freq(x)
    if not freq:
        return np.nan
    expected_idx = pd.date_range(start=x.min(), end=x.max(), freq=freq)
    return (len(expected_idx) - len(x)) / len(expected_idx) * 100


BASE_SET = [
    # BASE TSFEATURES
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    stl_features,
    stability,
    statistics,
    hw_parameters,
    unitroot_kpss,
    unitroot_pp,
    series_length,
    # NEW FEATURES
    ADI,  # average interval duration
    hurst_exp_dfa,  # hurst exponent of DFA
    lya_exp,  # lyapunov exponent
    longest_zero_streak,  # longest streak of consecutive zeros
    number_of_leading_zeros,  # number of leading zeros
    number_of_trailing_zeros,  # number of trailing zeros
    hyndman_forecastability,  # hyndman forecastability
    monthly_MASE_score,  # monthly MASE score
    yearly_MASE_score,  # yearly MASE score
    quarterly_MASE_score,  # quarterly MASE score
    overdispersion,  # overdispersion
    pct_zeros,  # percentage of zeros in the series, it was included in the original function
    MI_top_k_lags,  # SUM(top 5 MI scores of predictive lags) / SUM(all MI scores)
    MI_top_k_lags_indices,  # top 5 predictive lags sorted by MI
    permutation_entropy,  # normalized permutation entropy
]


def summary_diagnostics(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    freq: int,
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
        - start_candate : Earliest timestamp in the series.
        - end_date : Latest timestamp in the series.
        - n_obs : Number of observed (non-NA) values in the series.
        - is_regular : Boolean flag, True if the series has exactly n_expected
          unique timestamps, False otherwise.
        - pct_missing : Percentage of expected timestamps that are missing.
        - mean_value : Mean of the observed target values.
        - sd_value : Standard deviation of the observed target values.
        - cv_value : Coefficient of variation (std ÷ mean), a measure of relative volatility.
        - skewness : Skew of the value distribution; positive = right-tailed,
          negative = left-tailed.
        - pct_zeros : Percent of observations equal to zero (measure of intermittency).
    """

    col_mapper = {id_col: "unique_id", date_col: "ds", value_col: "y"}
    reverse_col_mapper = {v: k for k, v in col_mapper.items()}
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    base_feats = df.groupby(id_col, as_index=False).agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
        pct_missing=(date_col, pct_missing_dates),
        kurtosis=(value_col, st.kurtosis),
        skewness=(value_col, st.skew),
    )

    renamed_df = df.rename(columns=col_mapper)
    tsfeatures_df = tsfeatures(
        ts=renamed_df,
        freq=freq,
        features=BASE_SET,
    ).rename(columns=reverse_col_mapper)

    return base_feats.merge(
        tsfeatures_df,
        on=id_col,
        how="left",
    )
