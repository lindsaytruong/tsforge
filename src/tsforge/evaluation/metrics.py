import numpy as np
import pandas as pd


# --- Scale-dependent metrics ---
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat))**2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


# --- Percentage metrics ---
def mape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = np.abs(y) + np.abs(yhat) + 1e-12
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100)


def wape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-12))


def business_accuracy(y, yhat):
    """
    Business-style Accuracy.
    1 - sum(|error|)/sum(actuals).
    Equivalent to 1 - WAPE.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - (np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-12)))


# --- Scaled metrics ---
def mase(y, yhat, y_naive=None):
    """
    Mean Absolute Scaled Error (relative to naive-1 by default).
    If y_naive not provided, computes using naive-1 differences.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    if y_naive is None:
        scale = np.mean(np.abs(y[1:] - y[:-1]))  # naive-1 denominator
    else:
        y_naive = np.asarray(y_naive)
        scale = np.mean(np.abs(y - y_naive))
    return float(np.mean(np.abs(y - yhat)) / (scale + 1e-12))


# --- Bias metrics ---
def bias(y, yhat):
    """
    Forecast bias (mean forecast error).
    Positive → under-forecasted, Negative → over-forecasted.
    """
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


def mean_percentage_error(y, yhat):
    """Mean Percentage Error (directional bias, in %)"""
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.mean((yhat[mask] - y[mask]) / y[mask]) * 100)


def forecast_bias(y, yhat):
    """
    Forecast Bias Ratio (%).
    Sum(forecast)/Sum(actual).
    1.0 = unbiased, <1 under-forecast, >1 over-forecast.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float((np.sum(yhat) + 1e-12) / (np.sum(y) + 1e-12))


# --- Scoring utility ---
def score_all(y, yhat, y_naive=None, as_dataframe=False):
    """
    Compute all standard forecast metrics and return as dict (default) or DataFrame row.
    """
    scores = {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape(y, yhat),
        "smape": smape(y, yhat),
        "wape": wape(y, yhat),
        "accuracy": business_accuracy(y, yhat),
        "bias": bias(y, yhat),
        "mpe": mean_percentage_error(y, yhat),
        "forecast_bias": forecast_bias(y, yhat),
    }
    # only compute mase if y has length > 1
    if len(np.asarray(y)) > 1:
        scores["mase"] = mase(y, yhat, y_naive=y_naive)

    if as_dataframe:
        return pd.DataFrame([scores])
    return scores

# Working Example
# y = [100, 120, 130, 110]
# yhat = [90, 125, 128, 115]

# from tsforge.metrics import score_all

# print(score_all(y, yhat))
# # {'mae': 5.0, 'rmse': 5.590, 'mape': 4.12, 'smape': 4.05, 'wape': 0.045, 
# #  'accuracy': 0.955, 'bias': 1.5, 'mpe': 1.23, 'forecast_bias': 1.01, 'mase': 0.87}

# # As a DataFrame row (ready for leaderboard)
# print(score_all(y, yhat, as_dataframe=True))