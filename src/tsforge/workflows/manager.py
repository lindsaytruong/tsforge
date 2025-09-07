import pandas as pd
from statsforecast import StatsForecast
from mlforecast import MLForecast

from tsforge.evaluation import *

def _melt_mlforecast_preds(df_preds: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in df_preds.columns if c in ("unique_id","ds","date")]
    melted = df_preds.melt(id_vars=id_cols, var_name="model", value_name="yhat")
    if "ds" in melted.columns:
        melted = melted.rename(columns={"ds":"date"})
    return melted

class WorkflowManager:
    def __init__(self, workflows, id_col="unique_id", time_col="date", target_col="sales"):
        self.workflows = workflows
        self.id_col, self.time_col, self.target_col = id_col, time_col, target_col

    def backtest_all(self, df, h, metrics=(mae,wape,smape)):
        df = df.sort_values([self.id_col,self.time_col]).copy()
        is_test = df.groupby(self.id_col).cumcount(ascending=False) < h
        df_train, df_test = df[~is_test], df[is_test]

        rows=[]
        for wf in self.workflows:
            name, engine, params = wf["name"], wf["engine"], wf["params"]
            recipe = wf.get("recipe")
            Xtr, Xte = recipe.bake(df_train) if recipe else df_train, recipe.bake(df_test) if recipe else df_test

            if engine=="mlforecast":
                mlf = MLForecast(**params)
                mlf.fit(Xtr, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                        static_features=wf.get("static_features"))
                preds = _melt_mlforecast_preds(mlf.predict(h))
                preds["workflow"], preds["engine"]=name,"mlforecast"

            elif engine=="statsforecast":
                sf = StatsForecast(**params, n_jobs=-1)
                tr=Xtr.rename(columns={self.time_col:"ds",self.target_col:"y"})
                sf.fit(tr[[self.id_col,"ds","y"]])
                fc=sf.predict(h).reset_index()
                preds=fc.melt(id_vars=[self.id_col,"ds"], var_name="model", value_name="yhat")
                preds=preds.rename(columns={"ds":self.time_col})
                preds["workflow"],preds["engine"]=name,"statsforecast"

            elif engine=="custom":
                func=params["func"]
                preds=func(Xtr,h,self.id_col,self.time_col,self.target_col)
                preds["workflow"],preds["engine"]=name,"custom"

            else:
                raise ValueError(f"Unknown engine {engine}")

            truth=Xte[[self.id_col,self.time_col,self.target_col]]
            joined=preds.merge(truth,on=[self.id_col,self.time_col],how="inner")

            for (wf_name, mdl),g in joined.groupby(["workflow","model"]):
                row={"workflow":wf_name,"model":mdl}
                for m in metrics:
                    row[m.__name__]=m(g[self.target_col],g["yhat"])
                rows.append(row)

        return pd.DataFrame(rows).sort_values("wape")
