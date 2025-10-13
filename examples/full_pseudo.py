# ===============================================
# COMBINED WORKFLOW UX: STATSFORECAST + MLFORECAST + NEURALFORECAST
# ===============================================

# ------------------------------------------------
# CREATE RECIPES
# ------------------------------------------------

 # StatsForecast basic (no cleaning, no exog)
sf_recipe_basic = recipe()   # plain passthrough

# --- StatsForecast enriched (cleansing + transforms + exog)
sf_recipe_enhanced = (
    recipe()
    .fill_gaps_plus(strategy="linear")
    .detect_outliers(method="zscore", threshold=3)
    .correct_outliers(strategy="linear_interp")
    .stabilize_variance(method="log1p")
    .decompose(signal_col="y", method="stl", seasonal_period=7, store_only=True)
    .add_dynamic_features(exog_df=prices, join_on=["unique_id","ds"], forward_fill=True)
    .add_exog_lags(cols=["price"], lags=[7,28])
)
#endregion

# --- MLForecast base recipe (explicit feature engineering)
ml_recipe_base = (
    recipe()
    .add_timeseries_signature("ds")
    .add_dummy(["store_id","dept_id","cat_id"], one_hot=True)
    .add_mean_encoder(["item_id"])
)

# --- MLForecast variant with static + dynamic exog
ml_recipe_exog = (
    recipe()
    .add_recipe(ml_recipe_base)
    .add_static_features(static_df=static_meta)
    .add_dynamic_features(exog_df=prices, join_on=["unique_id","ds"], forward_fill=True)
)

# --- NeuralForecast simple univariate
nf_recipe_uni = recipe()  # minimal, assumes continuous y

# --- NeuralForecast with static + dynamic vars
nf_recipe_multi = (
    recipe()
    .add_static_features(static_df=static_meta)
    .add_dynamic_features(exog_df=prices, join_on=["unique_id","ds"], forward_fill=True)
    .fill_gaps_plus(strategy="ffill")
)


# ------------------------------------------------
#  CREATE WORKFLOWS
# ------------------------------------------------

# --- STATSFORECAST: BASIC
wf_sf_basic = (
    workflow(engine="statsforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(sf_recipe_basic)
    .add_model_spec(
        StatsForecast(
            models=[Naive(), SeasonalNaive(season_length=7), AutoARIMA(season_length=7)],
            freq="D",
            n_jobs=-1
        )
    )
)

# --- STATSFORECAST: ENHANCED (CLEANING + EXOG)
wf_sf_enh = (
    workflow(engine="statsforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(sf_recipe_enhanced)
    .add_model_spec(
        StatsForecast(
            models=[AutoETS(season_length=7), Theta(), TBATS(seasonal_periods=[7,28])],
            freq="D",
            n_jobs=-1
        )
    )
)

# --- MLFORECAST: BASELINE FEATURED
wf_mlf_base = (
    workflow(engine="mlforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(ml_recipe_base)
    .add_model_spec(
        MLForecast(
            models=[LightGBM(), CatBoost()],
            freq="D",
            lags=[1,7,28],
            lag_transforms={
                1: [ExpandingMean(), ExpandingStd()],
                7: [RollingMean(window_size=14), RollingStd(window_size=14)],
            },
            date_features=["day","month","year","is_month_end","is_year_end"],
            target_transforms=[Differences([24])],
            n_jobs=-1
        )
    )
)

# --- MLFORECAST: STATIC + DYNAMIC EXOG
wf_mlf_exog = (
    workflow(engine="mlforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(ml_recipe_exog)
    .add_model_spec(
        MLForecast(
            models=[LightGBM(), RandomForestRegressor()],
            freq="D",
            lags=[1,7,28],
            static_features=["item_id","dept_id","cat_id","store_id","state_id"],
            date_features=["day","month","is_month_end"],
            n_jobs=-1
        )
    )
)

# --- NEURALFORECAST: BASIC UNIVARIATE
wf_nf_basic = (
    workflow(engine="neuralforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(nf_recipe_uni)
    .add_model_spec(
        NeuralForecast(
            models=[NHITS(), NBEATS()],
            freq="D",
            batch_size=64,
            max_epochs=50
        )
    )
)

# --- NEURALFORECAST: WITH STATIC + DYNAMIC EXOG
wf_nf_exog = (
    workflow(engine="neuralforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(nf_recipe_multi)
    .add_model_spec(
        NeuralForecast(
            models=[NHITS(), TFT()],
            freq="D",
            max_epochs=50,
            static_features=["store_id","dept_id"],
            hist_exog_list=["price"],
            futr_exog_list=["price"],
            batch_size=64
        )
    )
)

# ------------------------------------------------
# TUNE WORKFLOWS
# ------------------------------------------------
cv_plan = CVPlan.from_nixtla(
    df=train_df,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
    h=28,
    n_windows=3,
    step_size=28,
    freq="D"
)
# Use its min cutoff to define historical tuning window
tune_cutoff = cv_plan.min_cutoff
train_tune = train_df.query("ds <= @tune_cutoff")


wf_mlf_exog.tune(
    df=train_tune,
    search_space={
        "LightGBM": {
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [10, 20, 40],
        },
        "RandomForestRegressor": {
            "n_estimators": [200, 400, 800],
            "max_depth": [10, 20, None],
        },
    },
    metric="mae",
    backend="ray",
    n_trials=25,
)

wf_nf_exog.tune(
    df=train_tune,
    search_space={
        "NHITS": {"max_epochs": [50, 100], "batch_size": [32, 64]},
        "TFT": {"hidden_size": [64, 128], "dropout": [0.1, 0.3]},
    },
    metric="mae",
    backend="ray",
    n_trials=25,
)

# Save tuning results
wf_mlf_exog.save_tuning_trials("configs/mlf_exog_trials.parquet")
wf_mlf_exog.save_best_params("configs/mlf_exog_best.yaml")
wf_nf_exog.save_tuning_trials("configs/nf_exog_trials.parquet")
wf_nf_exog.save_best_params("configs/nf_exog_best.yaml")

# ------------------------------------------------
# Load tuned parameters before cross-validation
# ------------------------------------------------
wf_mlf_exog.load_best_params("configs/mlf_exog_best.yaml")
wf_nf_exog.load_best_params("configs/nf_exog_best.yaml")



# ------------------------------------------------
# COMBINE ALL WORKFLOWS
# ------------------------------------------------
all_workflows = WorkflowCollection([
    wf_sf_basic, wf_sf_enh,
    wf_mlf_base, wf_mlf_exog,
    wf_nf_basic, wf_nf_exog
])


# ------------------------------------------------
# CROSS-VALIDATION
# ------------------------------------------------
cv_results = all_workflows.cross_validation(
    train_df,
    h=28,
    n_windows=3,
    step_size=28,
    level=[80,95]
)
# standardized schema: unique_id | ds | cutoff | y | yhat | lo_80 | hi_80 | workflow | engine | model


# ------------------------------------------------
# ADD ENSEMBLES (POST-HOC)
# ------------------------------------------------
cv_tbl = ResultsTable(cv_results)
cv_tbl = (
    cv_tbl
    .add_ensembles(
        name="Mean Ensemble (All Engines)",
        members=[
            "wf_sf_basic","wf_sf_enh","wf_mlf_base","wf_mlf_exog","wf_nf_basic","wf_nf_exog"
        ],
        method="mean"
    )
    .add_ensembles(
        name="Weighted Ensemble (Core Models)",
        members=[
            ("wf_sf_enh","AutoETS"),
            ("wf_mlf_exog","LightGBM"),
            ("wf_nf_exog","NHITS")
        ],
        method="weighted",
        weights=[0.7, 0.2, 0.1],
    )
)
cv_final = cv_tbl.dataframe()


# ------------------------------------------------
# SCOREBOARD
# ------------------------------------------------
leaderboard = ScoreboardTable(cv_final, metrics=["mae","rmse","mape","bias"]).dataframe()


# ------------------------------------------------
# oops - missed a workflow
# ------------------------------------------------

# Define new workflow
wf_new = (
    workflow(engine="mlforecast", id_col="unique_id", date_col="ds", target_col="y")
    .add_recipe(ml_recipe_base)
    .add_model_spec(
        MLForecast(models=[XGBoost()], freq="D", lags=[1,7,28])
    )
)

# cross-validate only that workflow
cv_new = wf_new.cross_validation(train_df, h=28, n_windows=3)

# append new results to existing CV table
cv_tbl.add(cv_new)

# get updated combined CV dataframe
cv_final = cv_tbl.dataframe()

# refresh leaderboard
leaderboard = ScoreboardTable(cv_final, metrics=["mae","rmse","mape","bias"]).dataframe()

# register the workflow for future fits/forecasts
all_workflows.add(wf_new)

# later, fit all workflows (including the new one)
all_fitted = all_workflows.fit(train_df)

# forecast together seamlessly
fcst = all_fitted.forecast(train_df, h=28, level=[80,95])


# ------------------------------------------------
# FIT FULL SERIES
# ------------------------------------------------
all_fitted = all_workflows.fit(train_df)


# ------------------------------------------------
# FORECAST FUTURE
# ------------------------------------------------
future_df = make_future_frame(train_df, h=28, freq="D")
pred_df = all_fitted.forecast(df=train_df, h=28, level=[80,95])


# ------------------------------------------------
# FINAL ENSEMBLES ON FORECAST
# ------------------------------------------------
fcst_tbl = ResultsTable(pred_df)
fcst_tbl = (
    fcst_tbl
    .add_ensembles(
        name="Mean Ensemble (Final)",
        members=[
            "wf_sf_basic","wf_sf_enh","wf_mlf_base","wf_mlf_exog","wf_nf_basic","wf_nf_exog"
        ],
        method="mean"
    )
    .add_ensembles(
        name="Weighted Ensemble (Final)",
        members=[
            ("wf_sf_enh","AutoETS"),
            ("wf_mlf_exog","LightGBM"),
            ("wf_nf_exog","NHITS")
        ],
        method="weighted",
        weights=[0.4,0.35,0.25]
    )
)
fcst_final = fcst_tbl.dataframe()


# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------
selector = ModelSelection(
    cv_df=cv_final,
    metric="mae + abs(bias)",
    group_by="unique_id",
    top_n=1
)
best_models = selector.select()
summary     = selector.summary()

fcst_best = fcst_final.merge(
    best_models[["unique_id","workflow","engine","model"]],
    on=["unique_id","workflow","engine","model"],
    how="inner"
)
