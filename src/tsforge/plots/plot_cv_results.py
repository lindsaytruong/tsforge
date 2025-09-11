import matplotlib.pyplot as plt
import pandas as pd

def plot_cv_results(folds_df: pd.DataFrame, metric: str = "wape"):
    """
    Plot CV results as mean ± std per model (clean teaching style).

    Parameters
    ----------
    folds_df : DataFrame
        Output from manager.cross_validation(..., mode="folds").
    metric : str
        Metric column to plot (e.g. "wape", "rmse").
    """

    if metric not in folds_df.columns:
        raise ValueError(f"Metric '{metric}' not found in folds_df")

    df = folds_df.copy()
    df["label"] = df["workflow"] + " | " + df["model"].astype(str)

    # Aggregate mean & std
    summary = (
        df.groupby("label")[metric]
        .agg(["mean", "std"])
        .sort_values("mean")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(summary) + 2))

    ax.errorbar(
        summary["mean"],
        summary["label"],
        xerr=summary["std"],
        fmt="o",
        color="C0",
        ecolor="lightgray",
        elinewidth=3,
        capsize=4,
    )

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Model")
    ax.set_title(f"Cross-validation Results ({metric.upper()})")
    plt.tight_layout()
    plt.show()
