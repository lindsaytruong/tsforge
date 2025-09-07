import pandas as pd

def add_rolling_features(
    obj: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    window: int = 7,
    funcs = ("mean","std"),
    **kwargs,
) -> pd.DataFrame:
    """Wrapper for pytimetk.rolling_features (tsforge naming)."""
    return call_method(
        obj,
        "rolling_features",
        date_column=date_col,
        value_column=value_col,
        window=window,
        funcs=list(funcs),
        **kwargs,
    )

def add_expanding_features(
    obj: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    funcs = ("mean","std"),
    **kwargs,
) -> pd.DataFrame:
    """Wrapper for pytimetk.expanding_features (tsforge naming)."""
    return call_method(
        obj,
        "expanding_features",
        date_column=date_col,
        value_column=value_col,
        funcs=list(funcs),
        **kwargs,
    )
