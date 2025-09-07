import pandas as pd
from tsforge.workflows.workflows import make_custom_workflow
from tsforge.workflows.manager import WorkflowManager

def dummy_forecaster(df, h, id_col="unique_id", time_col="date", target_col="sales"):
    last = df.groupby(id_col)[target_col].last()
    out = []
    for uid in df[id_col].unique():
        fut_dates = pd.date_range(df[time_col].max(), periods=h+1, freq="D")[1:]
        out.append(pd.DataFrame({id_col: uid, time_col: fut_dates, "yhat": last[uid], "model": "dummy"}))
    return pd.concat(out)

def test_custom_workflow_runs():
    df = pd.DataFrame({
        "unique_id": ["A"]*5,
        "date": pd.date_range("2020-01-01", periods=5),
        "sales": [1,2,3,4,5],
    })
    wf = make_custom_workflow("dummy", dummy_forecaster)
    manager = WorkflowManager([wf])
    leaderboard = manager.backtest_all(df, h=1)
    assert "mae" in leaderboard.columns
    assert leaderboard["workflow"].iloc[0] == "dummy"
