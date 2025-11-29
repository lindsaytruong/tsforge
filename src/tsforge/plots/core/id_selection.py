# tsforge/plots/core/id_selection.py

from typing import Optional, Union, List
import pandas as pd

def select_ids(df: pd.DataFrame,
               id_col: str,
               ids: Optional[Union[str, int, List[str]]] = None,
               max_ids: int = 6):
    """
    Standardized ID selection used across all tsforge plots.
    Supports:
      - ids=None → pick first N series
      - ids='A' → single series
      - ids=['A','B'] → subset
      - caps results at max_ids
    """
    unique_ids = df[id_col].unique().tolist()

    if ids is None:
        return unique_ids[:max_ids]

    if isinstance(ids, (str, int)):
        return [ids]

    if isinstance(ids, list):
        # preserve order but filter only those present
        result = [i for i in ids if i in unique_ids]
        return result[:max_ids]

    raise ValueError("ids must be None, a string, int, or list.")
