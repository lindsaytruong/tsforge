from statsforecast.models import Naive, SeasonalNaive, AutoARIMA
from mlforecast.lag_transforms import ExpandingMean
import catboost as cb

def workflow(name, engine, params, static_features=None, recipe=None):
    return {
        "name":name,
        "engine":engine,
        "params":params,
        "static_features":static_features,
        "recipe":recipe
    }

def make_mlf_workflow(name, models, lags, date_features, static_features, recipe):
    fixed_models = []
    for m in models:
        if isinstance(m, cb.CatBoostRegressor) and not m.get_params().get("cat_features"):
            m = cb.CatBoostRegressor(
                **{**m.get_params(),
                   "cat_features": static_features,
                   "verbose": m.get_params().get("verbose", 0)}
            )
        fixed_models.append(m)

    return workflow(
        name, "mlforecast",
        dict(models=fixed_models, freq="D", lags=lags, lag_transforms={1:[ExpandingMean()]}, date_features=date_features),
        static_features, recipe
    )

def make_sf_workflow(name, models):
    return workflow(name,"statsforecast",dict(models=models,freq="D"))

def make_custom_workflow(name, func):
    return workflow(name,"custom",dict(func=func))
