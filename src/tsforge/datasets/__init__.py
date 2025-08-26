import importlib.resources as pkg_resources
import pandas as pd

def load_m5_subset() -> pd.DataFrame:
    """
    Load a teaching-ready subset of the M5 dataset (FOODS_3, CA/TX, 2016–2019).
    Returns:
        pd.DataFrame with columns:
        [id, item_id, dept_id, cat_id, store_id, state_id,
         date, sales, wm_yr_wk, sell_price,
         event_name_1, event_type_1, event_name_2, event_type_2]
    """
    with pkg_resources.path("tsforge.datasets", "m5_subset.csv") as p:
        return pd.read_csv(p, parse_dates=["date"])
