import pandas as pd
from tsforge.eda.check_panel import check_panel

def test_check_panel_flags_missing_dates():
    df = pd.DataFrame({
        "unique_id": ["A"]*3,
        "date": pd.to_datetime(["2020-01-01","2020-01-03","2020-01-04"]),
        "sales": [10, 20, 30]
    })
    report = check_panel(df, id_col="unique_id", date_col="date", freq="D")
    assert report["continuity_gaps"] == 1
    assert "warnings" in report
