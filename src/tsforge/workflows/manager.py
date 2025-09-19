import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from statsforecast import StatsForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast

import warnings
from joblib import Parallel, delayed
from functools import reduce


class WorkflowManager:
    def __init__(self, workflows, id_col="unique_id", time_col="date", target_col="sales"):
        self.workflows = workflows
        self.id_col, self.time_col, self.target_col = id_col, time_col, target_col
        self.last_cv_params = None
        self.last_cv_preds = None
        self._pred_cache: dict = {}

    # --------------------------
    # Helpers
    # --------------------------
    def _ensure_types(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce")
        out[self.target_col] = pd.to_numeric(out[self.target_col], errors="coerce")
        return out.dropna(subset=[self.id_col, self.time_col, self.target_col])

    def _core_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[self.id_col, self.time_col, self.target_col]].copy()

    def _to_sf(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={self.time_col: "ds", self.target_col: "y"})

    def _to_nf(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={self.time_col: "ds", self.target_col: "y"})

    def _reset_index_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex) or df.index.name in [self.id_col, self.time_col, "cutoff"]:
            return df.reset_index()
        return df

    def _standardize_preds(self, preds: pd.DataFrame, name: str, engine: str) -> pd.DataFrame:
        df = self._reset_index_if_needed(preds.copy())
        if "ds" in df.columns: df = df.rename(columns={"ds": self.time_col})
        if "y"  in df.columns: df = df.rename(columns={"y": self.target_col})
        if "y_hat" in df.columns: df = df.rename(columns={"y_hat": "yhat"})
        if "mean"  in df.columns: df = df.rename(columns={"mean": "yhat"})

        keys = [c for c in [self.id_col, self.time_col, "cutoff"] if c in df.columns]
        reserved = set(keys + [self.target_col, "yhat", "workflow", "engine", "model"])
        meta_cols = [c for c in df.columns if c in ("pi_status","pi_source")]

        def _is_interval_col(c: str) -> bool:
            c = str(c); return ("-lo-" in c) or ("-hi-" in c) or c.startswith("lo-") or c.startswith("hi-")

        point_cols = [c for c in df.columns if c not in reserved and not _is_interval_col(c)]
        rows, to_num = [], pd.to_numeric

        if len(point_cols) == 0 and "yhat" in df.columns:
            base = df[keys].copy()
            if self.target_col in df.columns: base[self.target_col] = to_num(df[self.target_col], errors="coerce")
            for mcol in meta_cols: base[mcol] = df[mcol]
            base["yhat"] = to_num(df["yhat"], errors="coerce")
            for c in df.columns:
                if c.startswith("lo-") or c.startswith("hi-"):
                    base[c] = to_num(df[c], errors="coerce")
            base["model"] = name
            rows.append(base)
        else:
            for m in point_cols:
                out = df[keys].copy()
                if self.target_col in df.columns: out[self.target_col] = to_num(df[self.target_col], errors="coerce")
                for mcol in meta_cols: out[mcol] = df[mcol]
                out["yhat"] = to_num(df[m], errors="coerce")
                for c in df.columns:
                    if c.startswith(f"{m}-lo-"):
                        lvl = c.split("-")[-1]; out[f"lo-{lvl}"] = to_num(df[c], errors="coerce")
                    if c.startswith(f"{m}-hi-"):
                        lvl = c.split("-")[-1]; out[f"hi-{lvl}"] = to_num(df[c], errors="coerce")
                out["model"] = m
                rows.append(out)

        if not rows:
            # no predictions for this workflow → return empty standardized frame
            return pd.DataFrame(columns=[*keys, self.target_col, "yhat", "workflow", "engine", "model"]).assign(
                workflow=name, engine=engine
            )

        std = pd.concat(rows, ignore_index=True)
        std["workflow"] = name
        std["engine"]  = engine
        return std

    def _ensure_nan_intervals_raw(self, preds: pd.DataFrame, level: Optional[List[int]]) -> pd.DataFrame:
        """Ensure lo/hi columns exist with NaN, instead of filling with yhat (raw wide MLForecast)."""
        if not level:
            return preds
        out = preds.copy()
        base_cols = {"unique_id", "ds", "date", "y", "sales", "cutoff"}
        model_cols = [c for c in out.columns if c not in base_cols and not str(c).startswith(("lo-", "hi-"))]
        for m in model_cols:
            for lv in level:
                lo, hi = f"{m}-lo-{lv}", f"{m}-hi-{lv}"
                if lo not in out.columns: out[lo] = np.nan
                if hi not in out.columns: out[hi] = np.nan
        return out

    def _fingerprint(self, X: pd.DataFrame, h: int, mode: str,
                     n_windows: int | None, step_size: int | None, level: Optional[List[int]] = None) -> tuple:
        level_key = tuple(sorted(level)) if level else None
        return (
            X[self.time_col].min(),
            X[self.time_col].max(),
            int(X[self.id_col].nunique()),
            int(len(X)),
            mode, int(h), int(n_windows or 0), int(step_size or 0),
            level_key,
        )

    def _get_member_preds(self, member, X, h, mode, n_windows=None, step_size=None, level=None):
        if isinstance(member, dict):
            wf_dict, filter_model = member, None
        else:
            wf_dict, filter_model = member
        wf_name, wf_engine = wf_dict["name"], wf_dict["engine"]

        fp = ("std:"+wf_name, self._fingerprint(X, h, mode, n_windows, step_size, level))
        if fp in self._pred_cache:
            std = self._pred_cache[fp].copy()
        else:
            raw = self._run_workflow_raw(wf_dict, X, h, mode, n_windows, step_size, level)
            std = self._standardize_preds(raw, wf_name, wf_engine)
            self._pred_cache[fp] = std.copy()

        if filter_model:
            base = str(filter_model).strip()
            paren = base.find("(")
            if paren != -1: base = base[:paren].strip()
            if base.endswith("()"): base = base[:-2]
            m = std["model"].astype(str)
            mask = (m == base) | m.str.startswith(base)
            if mask.sum() == 0:
                raise ValueError(f"Model '{filter_model}' not found in workflow '{wf_name}'. "
                                 f"Available: {sorted(std['model'].unique().tolist())}")
            if mask.sum() > 1 and (m == base).sum() == 0:
                raise ValueError(f"Ambiguous model '{filter_model}' in workflow '{wf_name}'. "
                                 f"Candidates: {sorted(std.loc[mask,'model'].unique().tolist())}")
            std = std.loc[mask].copy()
        return std

    # --------------------------
    # Engine runner
    # --------------------------
    def _run_workflow_raw(self, wf, X, h, mode, n_windows=None, step_size=None, level=None):
        name, engine, params = wf["name"], wf["engine"], wf["params"]
        recipe, static_feats = wf.get("recipe"), wf.get("static_features")
        X = self._ensure_types(recipe.bake(X) if recipe else X)

        # MLForecast
        if engine == "mlforecast":
            mlf  = MLForecast(**params)
            pi   = wf.get("prediction_intervals")
            lags = wf.get("lags") or []

            def _counts_len(df):
                return df.groupby(self.id_col, observed=True)[self.time_col].count()

            if mode == "forecast":
                if pi is not None and lags:
                    need   = max(lags) + pi.n_windows * pi.h
                    counts = _counts_len(X)
                    long_ids  = counts[counts >= need].index
                    short_ids = counts[counts <  need].index
                else:
                    long_ids  = X[self.id_col].unique()
                    short_ids = []

                preds_list = []

                if len(long_ids) > 0 and pi is not None:
                    X_long = X[X[self.id_col].isin(long_ids)]
                    mlf_long = MLForecast(**params)
                    mlf_long.fit(
                        X_long, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                        static_features=static_feats, prediction_intervals=pi, fitted=True
                    )
                    preds_long = mlf_long.predict(h=h, level=level)
                    preds_long["pi_status"] = "conformal"
                    preds_list.append(preds_long)

                if len(long_ids) > 0 and pi is None:
                    X_long = X[X[self.id_col].isin(long_ids)]
                    mlf_long = MLForecast(**params)
                    mlf_long.fit(
                        X_long, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                        static_features=static_feats, fitted=True
                    )
                    preds_long = mlf_long.predict(h=h, level=None)
                    preds_long = self._ensure_nan_intervals_raw(preds_long, level)
                    preds_long["pi_status"] = "no_pi_config"
                    preds_list.append(preds_long)

                if len(short_ids) > 0:
                    warnings.warn(f"{len(short_ids)} series too short for PI (need≈{need}). Using NaN intervals.")
                    X_short = X[X[self.id_col].isin(short_ids)]
                    mlf_short = MLForecast(**params)
                    mlf_short.fit(
                        X_short, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                        static_features=static_feats, fitted=True
                    )
                    preds_short = mlf_short.predict(h=h, level=None)
                    preds_short = self._ensure_nan_intervals_raw(preds_short, level)
                    preds_short["pi_status"] = "short_series"
                    preds_list.append(preds_short)

                return pd.concat(preds_list, ignore_index=True) if preds_list else pd.DataFrame()

            # CV
            elif mode == "cv":
                if pi is not None and lags:
                    need   = max(lags) + pi.n_windows * pi.h
                    counts = _counts_len(X)
                    long_ids  = counts[counts >= need].index
                    short_ids = counts[counts <  need].index
                else:
                    long_ids  = pd.Index([])
                    short_ids = X[self.id_col].unique()

                preds_list = []

                if len(long_ids) > 0 and pi is not None:
                    try:
                        preds_long = mlf.cross_validation(
                            X[X[self.id_col].isin(long_ids)],
                            id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                            h=h, n_windows=n_windows, step_size=step_size,
                            static_features=static_feats, level=level, prediction_intervals=pi
                        )
                        preds_long["pi_status"] = "conformal"
                        preds_list.append(preds_long)
                    except ValueError as e:
                        import re
                        m = re.search(r"Minimum required samples.*?:\s*(\d+),\s*shortest serie has:\s*(\d+)", str(e))
                        if m:
                            required = int(m.group(1))
                            counts   = _counts_len(X)
                            long2  = counts[counts >= required].index
                            short2 = counts[counts <  required].index

                            if len(long2):
                                try:
                                    preds_long2 = mlf.cross_validation(
                                        X[X[self.id_col].isin(long2)],
                                        id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                                        h=h, n_windows=n_windows, step_size=step_size,
                                        static_features=static_feats, level=level, prediction_intervals=pi
                                    )
                                    preds_long2["pi_status"] = "conformal"
                                    preds_list.append(preds_long2)
                                except ValueError:
                                    preds_long2 = mlf.cross_validation(
                                        X[X[self.id_col].isin(long2)],
                                        id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                                        h=h, n_windows=n_windows, step_size=step_size,
                                        static_features=static_feats, level=None
                                    )
                                    preds_long2 = self._ensure_nan_intervals_raw(preds_long2, level)
                                    preds_long2["pi_status"] = "pi_failed"
                                    preds_list.append(preds_long2)

                            if len(short2):
                                mlf_short = MLForecast(**params)
                                preds_short2 = mlf_short.cross_validation(
                                    X[X[self.id_col].isin(short2)],
                                    id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                                    h=h, n_windows=n_windows, step_size=step_size,
                                    static_features=static_feats, level=None
                                )
                                preds_short2 = self._ensure_nan_intervals_raw(preds_short2, level)
                                preds_short2["pi_status"] = "short_series"
                                preds_list.append(preds_short2)
                        else:
                            preds_long = mlf.cross_validation(
                                X[X[self.id_col].isin(long_ids)],
                                id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                                h=h, n_windows=n_windows, step_size=step_size,
                                static_features=static_feats, level=None
                            )
                            preds_long = self._ensure_nan_intervals_raw(preds_long, level)
                            preds_long["pi_status"] = "pi_failed"
                            preds_list.append(preds_long)

                if len(short_ids) > 0:
                    mlf_short = MLForecast(**params)
                    preds_short = mlf_short.cross_validation(
                        X[X[self.id_col].isin(short_ids)],
                        id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                        h=h, n_windows=n_windows, step_size=step_size,
                        static_features=static_feats, level=None
                    )
                    preds_short = self._ensure_nan_intervals_raw(preds_short, level)
                    preds_short["pi_status"] = "short_series"
                    preds_list.append(preds_short)

                if preds_list:
                    return pd.concat(preds_list, ignore_index=True)

                preds = mlf.cross_validation(
                    X, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col,
                    h=h, n_windows=n_windows, step_size=step_size,
                    static_features=static_feats, level=None
                )
                preds = self._ensure_nan_intervals_raw(preds, level)
                preds["pi_status"] = "no_pi_config"
                return preds

        # StatsForecast
        elif engine == "statsforecast":
            sf = StatsForecast(**params)
            sf_df = self._to_sf(self._core_cols(X))
            sf.fit(sf_df[[self.id_col, "ds", "y"]])
            if mode == "forecast":
                return sf.forecast(h=h, level=level).reset_index()
            else:
                return sf.cross_validation(df=sf_df, h=h, n_windows=n_windows, step_size=step_size, level=level)

        # NeuralForecast
        elif engine == "neuralforecast":
            nf_df = self._to_nf(self._core_cols(X))
            nf = NeuralForecast(models=params["models"], freq=params["freq"])
            nf.fit(nf_df)
            if mode == "forecast":
                return nf.predict(h=h, level=level).reset_index()
            else:
                return nf.cross_validation(df=nf_df, n_windows=n_windows, step_size=step_size, level=level, verbose=0)

        # Ensemble
        elif engine == "ensemble":
            return self._run_ensemble(X, h, mode, wf["params"], name, n_windows, step_size, level)

        else:
            raise ValueError(f"Unknown engine {engine}")

    # --------------------------
    # Public APIs
    # --------------------------
    def forecast(self, df: pd.DataFrame, h: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        def _run_one(wf):
            preds = self._run_workflow_raw(wf, df, h, mode="forecast", level=level)
            return self._standardize_preds(preds, wf["name"], wf["engine"])

        if len(self.workflows) == 1:
            rows = [_run_one(self.workflows[0])]
        else:
            rows = Parallel(n_jobs=-1, backend="threading")(delayed(_run_one)(wf) for wf in self.workflows)
        return pd.concat(rows, ignore_index=True)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: Optional[int] = None,
        level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        step = step_size or h
        self.last_cv_params = dict(h=h, n_windows=n_windows, step_size=step, level=level)

        def _run_one(wf):
            preds = self._run_workflow_raw(wf, df, h, mode="cv", n_windows=n_windows, step_size=step, level=level)
            return self._standardize_preds(preds, wf["name"], wf["engine"])

        if len(self.workflows) == 1:
            rows = [_run_one(self.workflows[0])]
        else:
            rows = Parallel(n_jobs=-1, backend="threading")(delayed(_run_one)(wf) for wf in self.workflows)
        preds_df = pd.concat(rows, ignore_index=True)
        self.last_cv_preds = preds_df
        return preds_df

    # --------------------------
    # Ensemble logic
    # --------------------------
    def _run_ensemble(self, X, h, mode, params, name,
                      n_windows=None, step_size=None, level: Optional[List[int]] = None):
        """Combine STANDARDIZED member predictions (yhat/lo/hi/sales/cutoff), dropping meta."""
        member_frames = []
        tags: List[str] = []

        for member in params["members"]:
            preds_m = self._get_member_preds(member, X, h, mode,
                                             n_windows=n_windows,
                                             step_size=step_size,
                                             level=level)
            if isinstance(member, dict):
                tag = member["name"]
            else:
                tag = f"{member[0]['name']}|{member[1]}"
            tags.append(tag)

            # drop meta to keep matrices numeric
            preds_m = preds_m.drop(columns=[c for c in preds_m.columns if c in ("pi_status","pi_source")],
                                   errors="ignore")

            # tag numeric cols
            rename_map = {}
            for c in preds_m.columns:
                if c in ("yhat",) or c.startswith("lo-") or c.startswith("hi-") or c == self.target_col:
                    rename_map[c] = f"{tag}:{c}"
            preds_m = preds_m.rename(columns=rename_map)
            member_frames.append(preds_m)

        keys = [self.id_col, self.time_col]
        if mode == "cv" and all("cutoff" in f.columns for f in member_frames):
            keys.append("cutoff")

        merged = reduce(lambda l, r: pd.merge(l, r, on=keys, how="inner"), member_frames)

        def ordered_cols(suffix: str) -> List[str]:
            return [f"{t}:{suffix}" for t in tags if f"{t}:{suffix}" in merged.columns]

        yhat_cols = ordered_cols("yhat")
        if not yhat_cols:
            raise ValueError("No member yhat columns to ensemble.")
        Yhat_df = merged[yhat_cols].apply(pd.to_numeric, errors="coerce")
        M_yhat  = Yhat_df.to_numpy(dtype=float)

        method = params.get("method", "mean")
        weights = params.get("weights")

        def combine(M: np.ndarray, cols: List[str]) -> np.ndarray:
            if method == "mean":
                return np.nanmean(M, axis=1)
            elif method == "median":
                return np.nanmedian(M, axis=1)
            elif method == "weighted":
                if weights is None or len(weights) != len(tags):
                    raise ValueError(f"weights must be length {len(tags)}")
                w_map = {t: w for t, w in zip(tags, np.array(weights, dtype=float))}
                w_sub = np.array([w_map[c.split(":")[0]] for c in cols], dtype=float)
                w_sub = w_sub / w_sub.sum()
                return np.nansum(M * w_sub[None, :], axis=1)
            else:
                raise ValueError("method must be one of {'mean','median','weighted'}")

        out = merged[keys].copy()

        if mode == "cv":
            sales_cols = [c for c in merged.columns if c.endswith(f":{self.target_col}") or c == self.target_col]
            if sales_cols:
                out[self.target_col] = merged[sales_cols].bfill(axis=1).iloc[:, 0]

        out["yhat"] = combine(M_yhat, yhat_cols)

        levels = sorted({c.split("-")[-1] for c in merged.columns if ":lo-" in c or ":hi-" in c})
        req_levels = set(map(str, level)) if level else set()
        for lvl in sorted(set(levels).union(req_levels)):
            lo_cols = ordered_cols(f"lo-{lvl}")
            hi_cols = ordered_cols(f"hi-{lvl}")
            if lo_cols:
                M_lo = merged[lo_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                out[f"lo-{lvl}"] = combine(M_lo, lo_cols)
            else:
                out[f"lo-{lvl}"] = out["yhat"]
            if hi_cols:
                M_hi = merged[hi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                out[f"hi-{lvl}"] = combine(M_hi, hi_cols)
            else:
                out[f"hi-{lvl}"] = out["yhat"]

        out["workflow"] = name
        out["engine"] = "ensemble"
        out["model"] = f"Ensemble-{method}"
        return out
