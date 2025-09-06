import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import re




def plot_series(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col,                    # str (col name) or callable transform
    ids: list[str] | None = None,
    max_ids: int = 6,
    engine: str = "plotly",       # 'plotly' or 'matplotlib'
    facet: bool | None = None,    # None = auto; True = facet grid; False = overlay
    facet_wrap: int = 3,          # facet columns
    smooth_window: int | None = None,
    dropdown: bool = False,       # Plotly overlay dropdown
    freq: str | None = None       # e.g. 'D','W','M','Q' for resampling
):
    """
    Plot multiple time series with optional smoothing, resampling, and interactivity.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with at least [id_col, date_col, value_col].
    id_col : str
        Series identifier column (e.g., 'unique_id').
    date_col : str
        Date column.
    value_col : str or callable
        Column name (str) or a function that takes df and returns a Series
        (e.g., lambda d: np.log1p(d['sales'])).
    ids : list of str, optional
        Specific series IDs to plot. If None, a sample is drawn.
    max_ids : int, default=6
        Maximum series to overlay before switching to facet mode.
    engine : {'plotly','matplotlib'}, default='plotly'
        Plotting backend.
    facet : bool, optional
        Whether to facet (small multiples) or overlay. If None, chosen automatically.
    facet_wrap : int, default=3
        Number of facet columns if faceting.
    smooth_window : int, optional
        Rolling mean window for smoother.
    dropdown : bool, default False
        If True (overlay+plotly), adds a series selector dropdown.
    freq : str, optional
        Pandas offset alias ('D','W','M','Q', etc.). If given, resamples data
        to that frequency using sum aggregation.
    """

    # Copy + prep
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Handle value_col: string or callable
    if callable(value_col):
        df["_value"] = value_col(df)
        value_col_plot = "_value"
    else:
        value_col_plot = value_col

    # Optional resample
    if freq is not None:
        df = (
            df.set_index(date_col)
              .groupby(id_col)[value_col_plot]
              .resample(freq).sum()
              .reset_index()
        )

    # Pick IDs
    unique_ids = df[id_col].unique()
    if ids is None:
        n = min(max_ids, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n, random_state=42).tolist()
    df_sub = df[df[id_col].isin(ids)].copy()

    # Optional smoother
    if smooth_window and smooth_window > 1:
        df_sub["_smooth"] = (
            df_sub.groupby(id_col)[value_col_plot]
                  .transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
        )

    # Decide faceting
    if facet is None:
        facet = len(ids) > max_ids

    # =========================
    # Plotly backend
    # =========================
    if engine == "plotly":
        if facet:
            fig = px.line(
                df_sub, x=date_col, y=value_col_plot,
                facet_col=id_col, facet_col_wrap=facet_wrap,
                title="Time Series Faceted Grid"
            )
            for tr in fig.data:
                tr.line.color, tr.line.width, tr.opacity = "dodgerblue", 1, 0.6

            if smooth_window and "_smooth" in df_sub.columns:
                fig_smooth = px.line(
                    df_sub, x=date_col, y="_smooth",
                    facet_col=id_col, facet_col_wrap=facet_wrap
                )
                for tr in fig_smooth.data:
                    tr.line.color, tr.line.width, tr.opacity = "indianred", 2, 0.9
                    fig.add_trace(tr)

            rows = int(np.ceil(len(ids) / facet_wrap))
            fig.update_layout(
                height=300 * rows,
                plot_bgcolor="whitesmoke",
                paper_bgcolor="white",
                showlegend=False
            )
            fig.update_yaxes(matches=None)
            fig.for_each_yaxis(lambda a: a.update(showticklabels=True, showline=True, showgrid=True))
            return fig

        else:
            fig = go.Figure()
            for uid in ids:
                sub = df_sub[df_sub[id_col] == uid]
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub[value_col_plot],
                    mode="lines", name=f"{uid} raw",
                    line=dict(color="dodgerblue", width=1), opacity=0.6
                ))
                if smooth_window and "_smooth" in sub.columns:
                    fig.add_trace(go.Scatter(
                        x=sub[date_col], y=sub["_smooth"],
                        mode="lines", name=f"{uid} smooth",
                        line=dict(color="indianred", width=2), opacity=0.9
                    ))

            fig.update_layout(
                title="Time Series Overlay",
                plot_bgcolor="whitesmoke",
                paper_bgcolor="white"
            )

            if dropdown:
                buttons = []
                for uid in ids:
                    visible = [uid in tr.name for tr in fig.data]
                    buttons.append(dict(
                        label=str(uid),
                        method="update",
                        args=[{"visible": visible}]
                    ))
                if buttons:
                    fig.update_layout(
                        updatemenus=[{
                            "buttons": buttons,
                            "direction": "down",
                            "x": 1.05, "y": 1.15,
                            "xanchor": "left", "yanchor": "top"
                        }]
                    )
            return fig

    # =========================
    # Matplotlib backend
    # =========================
    elif engine == "matplotlib":
        if facet:
            n = len(ids)
            rows = int(np.ceil(n / facet_wrap))
            fig, axes = plt.subplots(rows, facet_wrap,
                                     figsize=(5 * facet_wrap, 3 * rows),
                                     sharex=True, sharey=False)
            if rows * facet_wrap == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            for ax, uid in zip(axes, ids):
                sub = df_sub[df_sub[id_col] == uid]
                ax.set_facecolor("whitesmoke")
                ax.plot(sub[date_col], sub[value_col_plot],
                        color="dodgerblue", alpha=0.6, lw=1)
                if smooth_window and "_smooth" in sub.columns:
                    ax.plot(sub[date_col], sub["_smooth"],
                            color="indianred", alpha=0.9, lw=2)
                ax.set_title(f"{id_col}={uid}")
            for ax in axes[len(ids):]:
                ax.axis("off")
            plt.tight_layout()
            return fig

        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_facecolor("whitesmoke")
            for uid in ids:
                sub = df_sub[df_sub[id_col] == uid]
                ax.plot(sub[date_col], sub[value_col_plot],
                        color="dodgerblue", alpha=0.6, lw=1, label=f"{uid} raw")
                if smooth_window and "_smooth" in sub.columns:
                    ax.plot(sub[date_col], sub["_smooth"],
                            color="indianred", alpha=0.9, lw=2, label=f"{uid} smooth")
            ax.legend()
            plt.tight_layout()
            return fig

    else:
        raise ValueError("engine must be 'plotly' or 'matplotlib'")


def _normalize_lags(lags):
    if isinstance(lags, int):
        return np.arange(lags+1, dtype=int)
    if isinstance(lags, range):
        return np.array(list(lags), dtype=int)
    if isinstance(lags, slice):
        start = 0 if lags.start is None else int(lags.start)
        step = 1 if lags.step is None else int(lags.step)
        return np.arange(start, int(lags.stop), step, dtype=int)
    if isinstance(lags, (list, tuple, np.ndarray, pd.Series)):
        return np.array(lags, dtype=int)
    if isinstance(lags, str) and re.fullmatch(r"\[\d+:\d+\]", lags.strip()):
        a, b = map(int, lags.strip()[1:-1].split(":"))
        return np.arange(a, b+1, dtype=int)
    raise ValueError("lags must be int, list, tuple, range, slice, or '[a:b]' string")

def _seasonal_lags_for_freq(freq: str):
    f = (freq or "").upper()
    return {"D":[7,30,365], "W":[52], "M":[12], "Q":[4]}.get(f, [])

def _band_traces(x, lo, hi, color_rgba, sid, kind):
    """Return lower & upper traces that form a confidence band."""
    lower = go.Scatter(x=x, y=lo, mode="lines",
                       line=dict(color=color_rgba, width=0),
                       name=f"{sid} {kind} band", showlegend=False,
                       hoverinfo="skip")
    upper = go.Scatter(x=x, y=hi, mode="lines",
                       line=dict(color=color_rgba, width=0),
                       fill="tonexty", fillcolor=color_rgba,
                       name=f"{sid} {kind} band", showlegend=False,
                       hoverinfo="skip")
    return lower, upper

def plot_act_pacf(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    series_ids: str | list[str] | int,
    lags: int | list[int] | range | slice | str = 60,
    freq: str = "D",
    facet: bool = False,
    dropdown: bool = False,
    alpha: float | None = 0.05
):
    """Plot ACF & PACF with Plotly, side-by-side, with optional confidence bands and dropdown."""

    # series_ids
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    elif isinstance(series_ids, int):
        series_ids = df[id_col].dropna().unique().tolist()[:series_ids]
    else:
        series_ids = list(series_ids)

    lag_idx = np.unique(_normalize_lags(lags))
    seasonal_lags = set(_seasonal_lags_for_freq(freq))

    results = []
    for sid in series_ids:
        sub = df.loc[df[id_col]==sid, [date_col, value_col]].dropna().sort_values(date_col)
        vals = sub[value_col].to_numpy()
        if vals.size < 3: 
            continue

        max_lag = min(vals.size-1, int(lag_idx.max()))
        if alpha is not None:
            a_full, a_conf = acf(vals, nlags=max_lag, fft=True, alpha=alpha)
            p_full, p_conf = pacf(vals, nlags=max_lag, method="ywmle", alpha=alpha)
            a_vals, p_vals = a_full[lag_idx], p_full[lag_idx]
            a_lo, a_hi = a_conf[lag_idx,0], a_conf[lag_idx,1]
            p_lo, p_hi = p_conf[lag_idx,0], p_conf[lag_idx,1]
            results.append((sid, lag_idx, a_vals, p_vals, (a_lo,a_hi), (p_lo,p_hi)))
        else:
            a_full = acf(vals, nlags=max_lag, fft=True)
            p_full = pacf(vals, nlags=max_lag, method="ywmle")
            results.append((sid, lag_idx, a_full[lag_idx], p_full[lag_idx], None, None))

    # -------- FACET --------
    if facet:
        n = len(results)
        titles = []
        for sid, *_ in results:
            titles += [f"{sid} - ACF", f"{sid} - PACF"]
        fig = make_subplots(rows=n, cols=2, subplot_titles=titles, horizontal_spacing=0.12)

        for i,(sid,idx,a_vals,p_vals,a_ci,p_ci) in enumerate(results, start=1):
            if a_ci is not None:
                lo,hi = a_ci
                lo_tr,hi_tr = _band_traces(idx,lo,hi,"rgba(30,144,255,0.15)",sid,"ACF")
                fig.add_trace(lo_tr,row=i,col=1)
                fig.add_trace(hi_tr,row=i,col=1)
            if p_ci is not None:
                lo,hi = p_ci
                lo_tr,hi_tr = _band_traces(idx,lo,hi,"rgba(205,92,92,0.15)",sid,"PACF")
                fig.add_trace(lo_tr,row=i,col=2)
                fig.add_trace(hi_tr,row=i,col=2)

            fig.add_trace(go.Bar(x=idx,y=a_vals,name=f"{sid} ACF",marker_color="dodgerblue",opacity=0.8),row=i,col=1)
            fig.add_trace(go.Bar(x=idx,y=p_vals,name=f"{sid} PACF",marker_color="indianred",opacity=0.8),row=i,col=2)

            for s in (seasonal_lags & set(idx)):
                fig.add_vline(x=s,line_dash="dash",line_color="black",row=i,col=1)
                fig.add_vline(x=s,line_dash="dash",line_color="black",row=i,col=2)

        fig.update_layout(height=300*n,width=950,
                          plot_bgcolor="whitesmoke",paper_bgcolor="white",
                          title="ACF | PACF (Faceted)")
        return fig

    # -------- OVERLAY --------
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF","PACF"], horizontal_spacing=0.12)
    for sid,idx,a_vals,p_vals,a_ci,p_ci in results:
        if a_ci is not None:
            lo,hi=a_ci
            lo_tr,hi_tr = _band_traces(idx,lo,hi,"rgba(30,144,255,0.15)",sid,"ACF")
            fig.add_trace(lo_tr,row=1,col=1)
            fig.add_trace(hi_tr,row=1,col=1)
        if p_ci is not None:
            lo,hi=p_ci
            lo_tr,hi_tr = _band_traces(idx,lo,hi,"rgba(205,92,92,0.15)",sid,"PACF")
            fig.add_trace(lo_tr,row=1,col=2)
            fig.add_trace(hi_tr,row=1,col=2)

        fig.add_trace(go.Bar(x=idx,y=a_vals,name=f"{sid} ACF",marker_color="dodgerblue",opacity=0.8),row=1,col=1)
        fig.add_trace(go.Bar(x=idx,y=p_vals,name=f"{sid} PACF",marker_color="indianred",opacity=0.8),row=1,col=2)

    for s in seasonal_lags:
        if s in lag_idx:
            fig.add_vline(x=s,line_dash="dash",line_color="black",row=1,col=1)
            fig.add_vline(x=s,line_dash="dash",line_color="black",row=1,col=2)

    fig.update_layout(height=450,width=950,
                      plot_bgcolor="whitesmoke",paper_bgcolor="white",
                      title="ACF | PACF (Overlay)")

    # Dropdown logic
    if dropdown and len(series_ids) > 1:
        names = [tr.name for tr in fig.data]
        buttons = []
        # All button
        buttons.append(dict(label="All", method="update",
                            args=[{"visible": [True]*len(names)}]))
        for sid in series_ids:
            visible = [(nm.startswith(f"{sid} ")) for nm in names]
            buttons.append(dict(label=str(sid),
                                method="update",
                                args=[{"visible": visible}]))
        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 1.05, "y": 1.15,
                "xanchor": "left", "yanchor": "top"
            }]
        )

    return fig
