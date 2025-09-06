import numpy as np
# from utilsforecast.losses import mae as _mae, mse as _mse, rmse as _rmse, smape as _smape, mase as _mase, rmsse as _rmsse

# def mae(y, yhat):
#     return _mae(y, yhat)

# def mse(y, yhat):
#     return _mse(y, yhat)

# def rmse(y, yhat):
#     return _rmse(y, yhat)

# def smape(y, yhat):
#     return _smape(y, yhat)

# def mase(y, yhat, y_train=None, seasonality=1):
#     return _mase(y, yhat, y_train=y_train, seasonality=seasonality)

# def rmsse(y, yhat, y_train=None, seasonality=1):
#     return _rmsse(y, yhat, y_train=y_train, seasonality=seasonality)

# # Extra metrics not in utilsforecast
# def wape(y, yhat):
#     """Weighted Absolute Percentage Error (a business favorite)."""
#     y, yhat = np.array(y), np.array(yhat)
#     return np.sum(np.abs(y - yhat)) / (np.sum(y) + 1e-9)

# def bias(y, yhat):
#     """Bias: mean signed error relative to demand size."""
#     y, yhat = np.array(y), np.array(yhat)
#     return (np.sum(yhat) - np.sum(y)) / (np.sum(y) + 1e-9)

# tsforge/evaluation/metrics.py

EPS = 1e-9

def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean(np.abs(yhat - y))

def mse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean((yhat - y) ** 2)

def rmse(y, yhat):
    return np.sqrt(mse(y, yhat))

def smape(y, yhat):
    """Symmetric MAPE in [0,2]. We report mean value (not %)"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y) + np.abs(yhat) + EPS
    return np.mean(2.0 * np.abs(yhat - y) / denom)

def wape(y, yhat):
    """Weighted Absolute Percentage Error (MAD / sum(y))."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.sum(np.abs(yhat - y)) / (np.sum(y) + EPS)

def bias(y, yhat):
    """Signed bias relative to total demand."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return (np.sum(yhat) - np.sum(y)) / (np.sum(y) + EPS)
