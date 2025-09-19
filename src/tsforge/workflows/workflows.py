from statsforecast.models import Naive, SeasonalNaive, AutoARIMA
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.utils import PredictionIntervals
from statsforecast.utils import ConformalIntervals

import catboost as cb



def workflow(name, engine, params, static_features=None, recipe=None):
    return {
        "name":name,
        "engine":engine,
        "params":params,
        "static_features":static_features,
        "recipe":recipe
    }
def make_sf_workflow(name, models, freq="D",
                     with_intervals=False,
                     intervals_method="conformal_distribution",
                     **params):
    """
    StatsForecast workflow.
    
    - Ensures unique, descriptive aliases for duplicate models.
    - Optionally attaches conformal prediction intervals.
    - Levels are always passed at forecast/cross_validation call-time.
    """
    # --- alias handling ---
    seen = {}
    for i, m in enumerate(models):
        # If no alias provided, auto-generate one
        if not getattr(m, "alias", None):
            base = m.__class__.__name__
            if base in seen:
                alias = f"{base}_{seen[base]}"
                seen[base] += 1
            else:
                alias = base
                seen[base] = 1
            m.alias = alias

    # --- attach conformal intervals if requested ---
    if with_intervals:
        for m in models:
            m.prediction_intervals = ConformalIntervals(
                n_windows=2, h=1, method=intervals_method
            )

    # --- build workflow dict ---
    return {
        "name": name,
        "engine": "statsforecast",
        "models": models,
        "params": {
            "models": models,
            "freq": freq,
            **params,
        },
    }


def make_mlf_workflow(name, models, lags, 
                      date_features=None,
                      static_features=None,
                      recipe=None,
                      freq="D",
                      with_intervals=False,
                      intervals_method="conformal_distribution",  # ← match toy
                      intervals_h=None,
                      intervals_windows=5,                         # ← new, match toy
                      **params):
    pi = None
    if with_intervals:
        from mlforecast.utils import PredictionIntervals
        if intervals_h is None:
            raise ValueError("You must provide intervals_h when with_intervals=True")
        pi = PredictionIntervals(
            n_windows=intervals_windows,
            h=intervals_h,
            method=intervals_method
        )

    return {
        "name": name,
        "engine": "mlforecast",
        "models": models,
        "lags": lags,
        "date_features": date_features,
        "static_features": static_features,
        "recipe": recipe,
        "prediction_intervals": pi,
        "params": {
            "models": models,
            "freq": freq,
            "lags": lags,
            "date_features": date_features,
            **params,
        },
    }



def make_nf_workflow(name, models, freq="D", **params):
    """
    NeuralForecast workflow. Levels are passed at call-time.
    """
    return {
        "name": name,
        "engine": "neuralforecast",
        "models": models,
        "params": {
            "models": models,
            "freq": freq,
            **params,
        },
    }

def make_custom_workflow(name, func):
    return workflow(name,"custom",dict(func=func))

def make_ensemble(name: str,
                  members: list,
                  method: str = "mean",
                  weights: list[float] | None = None):
    """
    Create an ensemble workflow.

    Parameters
    ----------
    name : str
        Name of the ensemble (e.g., "Average Ensemble").
    members : list
        - Workflow dicts: [wf1, wf2] → combine whole workflows
        - Tuples: [(wf1, "ModelName"), (wf2, "OtherModel")] → combine specific models
    method : {"mean","median","weighted"}
        How to combine predictions.
    weights : list[float], optional
        Only used if method="weighted". Must match len(members).
    """
    return {
        "name": name,
        "engine": "ensemble",
        "params": dict(
            members=members,
            method=method,
            weights=weights,
        ),
    }
