import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def plot_cv_plan(obj,
                 df: Optional[pd.DataFrame] = None,
                 series_id: Optional[str] = None):
    """
    Plot the actual CV plan used in manager.cross_validation.
    Shows one subplot per fold with clearer styling.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    from tsforge.workflows.manager import WorkflowManager

    if not isinstance(obj, WorkflowManager):
        raise ValueError("First argument must be a WorkflowManager")

    manager = obj
    if manager.last_cv_params is None or manager.last_cv_preds is None:
        raise RuntimeError("No CV results stored. Run cross_validation() first.")

    h = manager.last_cv_params["h"]
    cutoffs = sorted(manager.last_cv_preds["cutoff"].unique())
    id_col, time_col, target_col = manager.id_col, manager.time_col, manager.target_col

    # Prep series
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    all_ids = df[id_col].unique()
    if series_id is None:
        if len(all_ids) == 1:
            series_id = all_ids[0]
        else:
            series_id = all_ids[0]
            print(f"[Info] Multiple series detected ({len(all_ids)}). Showing first: {series_id}")
    series = df[df[id_col] == series_id].sort_values(time_col)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(len(cutoffs), 1,
                             figsize=(6, 3 * len(cutoffs)),
                             sharex=True)

    if len(cutoffs) == 1:
        axes = [axes]

    for i, (ax, cutoff) in enumerate(zip(axes, cutoffs), 1):
        # Train = all rows <= cutoff
        train = series[series[time_col] <= cutoff]
        # Test = the next h rows after cutoff
        test = series[(series[time_col] > cutoff)].head(h)

        ax.plot(train[time_col], train[target_col],
                color="#1f77b4", lw=2, alpha=0.8, label="Train")
        ax.plot(test[time_col], test[target_col],
                color="#ff7f0e", lw=2.5, marker="o", label="Test")

        ax.axvline(cutoff, color="black", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"Fold {i}: cutoff = {cutoff.date()}  (series={series_id})",
                     fontsize=12, pad=10)

        ax.legend(loc="upper left", frameon=False)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    fig.suptitle("Cross-Validation Plan", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
