import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tsforge.feature_engineering import summarize_by_time  # use your wrapper
from tsforge._utils.call import call_method

from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller


import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

def summary_diagnostics(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    freq: str | None = None,
    full_features: bool = True,
) -> pd.DataFrame:
    """
    Profile time series data by ID. Inspired by R timetk::tk_summary_diagnostics.

    Parameters
    ----------
    ...
    full_features : bool, default True
        If False, skips slow diagnostics (trend_strength, adf_pval).
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    results = []

    for key, g in df.groupby(id_col):
        g = g.sort_values(date_col)

        start_date = g[date_col].min()
        end_date = g[date_col].max()
        n_obs = g[value_col].count()

        # Expected periods if freq provided
        if freq:
            expected_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
            n_expected = len(expected_idx)
            n_missing = n_expected - n_obs
            pct_missing = n_missing / n_expected * 100
        else:
            n_expected = n_missing = pct_missing = np.nan

        # Interval stats
        dates = g[date_col].unique()
        if len(dates) > 1:
            diffs = np.diff(dates).astype("timedelta64[D]").astype(int)
            avg_interval = np.mean(diffs)
            min_interval = np.min(diffs)
            max_interval = np.max(diffs)
            sd_interval = np.std(diffs, ddof=1)
        else:
            avg_interval = min_interval = max_interval = sd_interval = np.nan

        # Value stats
        values = g[value_col].dropna()
        if len(values) > 0:
            min_value = values.min()
            max_value = values.max()
            mean_value = values.mean()
            median_value = values.median()
            sd_value = values.std()
            cv_value = sd_value / mean_value if mean_value != 0 else np.nan
            skewness = skew(values)
            kurt = kurtosis(values)
        else:
            min_value = max_value = mean_value = median_value = sd_value = cv_value = skewness = kurt = np.nan

        # Zeros / intermittency
        n_zeros = (values == 0).sum()
        pct_zeros = n_zeros / len(values) * 100 if len(values) else np.nan
        zero_runs = (values == 0).astype(int).groupby(values.ne(0).cumsum()).cumsum()
        max_zero_run = zero_runs.max() if not zero_runs.empty else np.nan

        # Outliers (z-score > 3)
        if len(values) > 1 and sd_value > 0:
            zscores = (values - mean_value) / sd_value
            n_outliers = (np.abs(zscores) > 3).sum()
            pct_outliers = n_outliers / len(values) * 100
        else:
            n_outliers = pct_outliers = np.nan

        # Trend & stationarity (only if full_features=True)
        if full_features and len(values) > 1:
            # Simple trend strength (R² of linear fit)
            t = np.arange(len(values))
            coeffs = np.polyfit(t, values, 1)
            fit = np.polyval(coeffs, t)
            ss_res = np.sum((values - fit) ** 2)
            ss_tot = np.sum((values - values.mean()) ** 2)
            trend_strength = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # Stationarity (ADF test)
            try:
                adf_pval = adfuller(values.dropna(), autolag="AIC")[1] if len(values) > 10 else np.nan
            except Exception:
                adf_pval = np.nan
        else:
            trend_strength = np.nan
            adf_pval = np.nan

        results.append({
            id_col: key,
            "start_date": start_date,
            "end_date": end_date,
            "n_obs": n_obs,
            "n_expected": n_expected,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "avg_interval": avg_interval,
            "min_interval": min_interval,
            "max_interval": max_interval,
            "sd_interval": sd_interval,
            "min_value": min_value,
            "max_value": max_value,
            "mean_value": mean_value,
            "median_value": median_value,
            "sd_value": sd_value,
            "cv_value": cv_value,
            "skewness": skewness,
            "kurtosis": kurt,
            "n_zeros": n_zeros,
            "pct_zeros": pct_zeros,
            "max_zero_run": max_zero_run,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
            "trend_strength": trend_strength,
            "adf_pval": adf_pval,
        })

    return pd.DataFrame(results)

def profile_card(
    df, id_val, id_col="id", date_col="date", value_col="sales",
    diag=None, freq=None
):
    """Generate a 1-page profile for one series, with optional resampling by freq."""
    g = df[df[id_col] == id_val].sort_values(date_col)

    # --- Resample if freq provided ---
    if freq is not None:
        from tsforge.feature_engineering import summarize_by_time
        g = summarize_by_time(
            g, date_col=date_col, value_col=value_col, freq=freq, agg_func="sum"
        )

    # --- Diagnostics ---
    if diag is None:
        from tsforge.eda import summary_diagnostics
        diag = summary_diagnostics(
            df, id_col=id_col, date_col=date_col, value_col=value_col, freq=freq
        )
    row = diag[diag[id_col] == id_val].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Time series plot
    axes[0,0].plot(g[date_col], g[value_col], label="sales", color="blue")
    axes[0,0].set_title(f"Series {id_val} — Time Plot ({freq or 'raw'})")

    # 2. Histogram
    sns.histplot(g[value_col], bins=30, ax=axes[0,1], color="blue")
    axes[0,1].set_title(f"Distribution of Sales ({freq or 'raw'})")

    # 3. Stats table
    stats_text = (
        f"n_obs: {row['n_obs']}\n"
        f"n_missing: {row['n_missing']}\n"
        f"mean: {row['mean_value']:.2f}\n"
        f"cv: {row['cv_value']:.2f}\n"
        f"pct_zeros: {row['pct_zeros']:.1f}%\n"
        f"max_zero_run: {row['max_zero_run']}\n"
        f"outliers: {row['n_outliers']}\n"
        f"trend_strength: {row['trend_strength']:.2f}\n"
        f"adf_pval: {row['adf_pval']:.3f}"
    )
    axes[1,0].axis("off")
    axes[1,0].text(0, 0.5, stats_text, fontsize=12, va="center", family="monospace")
    axes[1,0].set_title("Summary Stats")

    # 4. Boxplot
    sns.boxplot(x=g[value_col], ax=axes[1,1], color="blue")
    axes[1,1].set_title(f"Boxplot of Sales ({freq or 'raw'})")

    plt.tight_layout()
    plt.show()


def plot_time_series(obj: pd.DataFrame, *, date_col: str, value_col: str, **kwargs):
    """Wrapper for pytimetk.plot_time_series (tsforge naming)."""
    return call_method(obj, "plot_time_series", date_column=date_col, value_column=value_col, **kwargs)

def plot_seasonal_diagnostics(obj: pd.DataFrame, *, date_col: str, value_col: str, **kwargs):
    """Wrapper for pytimetk.plot_seasonal_diagnostics (tsforge naming)."""
    return call_method(obj, "plot_seasonal_diagnostics", date_column=date_col, value_column=value_col, **kwargs)

def plot_acf_diagnostics(obj: pd.DataFrame, *, date_col: str, value_col: str, **kwargs):
    """Wrapper for pytimetk.plot_acf_diagnostics (tsforge naming)."""
    return call_method(obj, "plot_acf_diagnostics", date_column=date_col, value_column=value_col, **kwargs)

def plot_anomaly_diagnostics(obj: pd.DataFrame, *, date_col: str, value_col: str, **kwargs):
    """Wrapper for pytimetk.plot_anomaly_diagnostics (tsforge naming)."""
    return call_method(obj, "plot_anomaly_diagnostics", date_column=date_col, value_column=value_col, **kwargs)
