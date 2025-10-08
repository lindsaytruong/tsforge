
"""
fill_gaps_plus: Factory pattern wrapper around Nixtla's fill_gaps

This module shows how to wrap an external library function (Nixtla's fill_gaps)
in a factory pattern for use with Recipe, while keeping the underlying
implementation unchanged.
"""

import pandas as pd
from typing import Callable

# ============================================================================
# Import Nixtla's fill_gaps at module level (standard practice)
# ============================================================================
from utilsforecast.preprocessing import fill_gaps as nixtla_fill_gaps

# Define Step type
Step = Callable[[pd.DataFrame], pd.DataFrame]


# ============================================================================
# YOUR FACTORY WRAPPER - Calls Nixtla underneath
# ============================================================================

def fill_gaps_plus(
    id_col: str,
    time_col: str,
    target_col: str,
    static_cols: list[str] | None = None,
    dynamic_fill: dict[str, str] | None = None,
    target_fill: str = "zero",
    freq: str | None = None,
    verbose: bool = True,
) -> Step:
    """
    Factory function that creates an enhanced gap-filling transformation.
    
    This wraps Nixtla's fill_gaps and adds handling for static/dynamic variables.
    
    ⭐ KEY POINT: Nixtla's fill_gaps is called INSIDE the returned function,
       so it executes at the same time as before - just with a cleaner API.

    Parameters
    ----------
    id_col : str
        Column with unique series identifier.
    time_col : str
        Datetime column.
    target_col : str
        Target column (e.g., sales).
    static_cols : list[str], optional
        Columns that don't change over time (e.g., category, store).
    dynamic_fill : dict[str, str], optional
        Dict mapping dynamic variable names to fill strategy:
          - "zero" → fill missing with 0
          - "ffill" → forward fill within id
          - "interpolate" → numeric interpolate within id
          - "nan" (default) → leave missing
    target_fill : str, default "zero"
        Fill strategy for the target: "zero" or "nan"
    freq : str, optional
        Expected pandas frequency string ("D", "W", "M").
    verbose : bool, default True
        If True, prints input/output shape.

    Returns
    -------
    Step
        A transformation function that uses Nixtla's fill_gaps internally.
    """
    
    # Validate inputs at configuration time
    if not freq:
        raise ValueError("freq must be specified (e.g., 'D', 'W', 'M')")
    
    # ========================================================================
    # Inner function - THIS is where Nixtla's fill_gaps gets called
    # ========================================================================
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        """
        The actual transformation that runs at execution time.
        
        This function:
        1. Calls Nixtla's fill_gaps (unchanged!)
        2. Adds your custom logic for static/dynamic variables
        """
        
        if verbose:
            print(f"[fill_gaps_plus] Input shape: {df.shape}")

        # ====================================================================
        # STEP 1: Call Nixtla's fill_gaps (EXACTLY as before!)
        # ====================================================================
        
        # Prepare base dataframe with only the columns Nixtla needs
        df_base = df[[id_col, time_col, target_col]].copy()
        
        # 🎯 THIS IS THE KEY: Nixtla's fill_gaps is called here
        # The factory pattern doesn't change this at all!
        filled = nixtla_fill_gaps(
            df_base, 
            freq=freq,           # ← These params come from closure
            id_col=id_col,       # ← Captured from outer function
            time_col=time_col    # ← Available automatically
        )
        
        # ====================================================================
        # STEP 2: Your custom enhancements (unchanged logic)
        # ====================================================================
        
        # Fill target based on strategy
        if target_fill == "zero":
            filled[target_col] = filled[target_col].fillna(0)
        elif target_fill == "nan":
            pass
        else:
            raise ValueError("target_fill must be 'zero' or 'nan'")

        # Add static columns if they exist
        if static_cols:
            static_cols_existing = [c for c in static_cols if c in df.columns]
            if static_cols_existing:
                static_df = df[[id_col] + static_cols_existing].drop_duplicates(id_col)
                filled = filled.merge(static_df, on=id_col, how="left")

        # Add dynamic columns if they exist
        ignore_cols = [id_col, time_col, target_col] + (static_cols or [])
        dyn_cols = [c for c in df.columns if c not in ignore_cols]
        
        if dyn_cols:
            dyn_df = df[[id_col, time_col] + dyn_cols].copy()
            filled = filled.merge(dyn_df, on=[id_col, time_col], how="left")

            # Apply per-column fill strategies
            if dynamic_fill:
                for col, strategy in dynamic_fill.items():
                    if col not in filled.columns:
                        continue
                    if strategy == "zero":
                        filled[col] = filled[col].fillna(0)
                    elif strategy == "ffill":
                        filled[col] = filled.groupby(id_col)[col].ffill()
                    elif strategy == "interpolate":
                        filled[col] = filled.groupby(id_col)[col].transform(
                            lambda x: x.interpolate()
                        )
                    elif strategy == "nan":
                        pass
                    else:
                        raise ValueError(
                            f"Unknown fill strategy {strategy} for column {col}"
                        )

        if verbose:
            added = filled.shape[0] - df.shape[0]
            print(f"[fill_gaps_plus] Output shape: {filled.shape}")
            print(f"[fill_gaps_plus] Added {added} rows (Nixtla) + enhanced")

        return filled
    
    # Return the configured function
    return _fn

