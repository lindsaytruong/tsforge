TO DO 
MLFORECAST ACTIONS
1. Add in future frame to recipe workflow - whats that look like 
2. How does it look like adding static and dynic exogenous regressors to workflow
3. What would prediction intervals look like (lessor priority)
4. Model Tuning (lessor priority)
5. Feature Importance (lessor priority)
6. Feature Selection (lessor priority)
7. SHAP (lessor priority)

TO DO 
STATSFORECAST ACTIONS
1. outlier cleansing
2. Decomposition
3. Prediction Intervals
4. exogenous regressors for ARIMA 





################################
## CREATE RECIPES             ##
################################


base_recipe = (recipe()
    .update_role(rowid, new_role = "indicator")
    .add_timeseries_signature('ds')
    .add_dummy(['store_id','dept_id','cat_id'], one_hot = True)
    .add_mean_encoder(['item_id'])
)

recipe_2 = (recipe()
    .add_recipe(base_recipe)
    .add_exogenous_regressors(['price'])
    .add_lagged_regressors(
        transform_exog(prices, #<-- Nixtla
                        lags=[7], 
                        lag_transforms={1: [ExpandingMean()}
                        )
                    )
            )

################################
## CREATE Workflows           ##
################################
 workflow1 = workflow(  engine = 'mlforecast',
                        id_col ='unique_id',
                        date_col = 'ds',
                        target_col = 'y'
                            )
    .add_recipe(base_recipe)
    .add_model_spec(
        MLForecast( #<- Expose nixtla class
            models=[lightgbm(),catboost()] ,
            freq = 'D',
            lags = [1,24],
            lag_transforms = {
                1: [ExpandingMean(),ExpandingStd()],
                2: [ExponentiallyWeightedMean(alpha=0.3)],
                7: [RollingMean(window_size=14), 
                    RollingStd(window_size=14),
                    RollingQuantile(p=0.9,window_size=28)],
                14: [RollingMin(window_size = 28), RollingMax(window_size= 28)],
                28: [SeasonalRollingMean(seasonal_length=7,window_size=4),
                     SeasonalRollingStd(seasonal_length = 7, window_size = 4),
                     SeasonalRollingQuantile(p=0.1, season_length = 7, window_size = 4)]
                24: [RollingMean(window_size = 48),rolling_mean_48]
            }
            date_features = ['day', 'month', 'year', 'is_month_end', 'is_year_end' , CalendarFourier(period=7, order=3)]
            target_transforms = [Differences([24]),
            n_jobs = -1,            
                )
            )

 workflow2 = workflow(engine = 'mlforecast',
                        id_col ='unique_id',
                        date_col = 'ds',
                        target_col = 'y')
    .add_recipe(recipe_2)
    .add_model_spec(
        MLForecast(
            models=[lightgbm(),catboost()] ,
            freq = 'D',
            lags = [1,24],
            lag_transforms = {
                1: [ExpandingMean(),ExpandingStd()],
                2: [ExponentiallyWeightedMean(alpha=0.3)],
                7: [RollingMean(window_size=14), 
                    RollingStd(window_size=14),
                    RollingQuantile(p=0.9,window_size=28)],
                14: [RollingMin(window_size = 28), RollingMax(window_size= 28)],
                28: [SeasonalRollingMean(season_length=7,window_size=4),
                     SeasonalRollingStd(season_length = 7, window+size = 4),
                     SeasonalRollingQuantile(p=0.1, season_length = 7, window_size = 4)]
                24: [RollingMean(window_size = 48),rolling_mean_48]
            }
            date_features = ['day', 'month', 'year', 'is_month_end', 'is_year_end' , CalendarFourier(period=7, order=3)]
            target_transforms = [Differences([24]),
            n_jobs = -1,            
                )
            )

#####################################
## Combine Workflows               ##
#####################################

all_workflows = WorkflowCollection([workflow1, workflow2])

#####################################
## Pass workflow to CV             ##
#####################################
cv_results = all_workflows.cross_validation(train_df,
                                            h=h, 
                                            n_windows=3, 
                                            )

#####################################
## Add CV results to ResultsTale   ##
#####################################
cv_tbl = ResultsTable(cv_results)


#####################################
## Create Ensembles               ##
#####################################
cv_tbl = cv_tbl
    .add_ensembles(
        name="Mean Ensemble",
        members=['workflow1', 'workflow2'],
        method="mean")
    .add_ensembles(
        name="Weighted Ensemble (MLF internal)",
        members=[('workflow1', "LightGBM"), ('workflow1', "CatBoost")],
        method="weighted",
        weights=[0.7, 0.3]
)
cv_tbl.remove_models(['Mean Ensemble'])

# Extract CV results dataframe
cv_final = cv_tbl.dataframe()

#####################################
## Score Cross Val                 ##
#####################################

leaderboard = ScoreboardTable(cv_final, metrics=["mae","rmse","mape","bias"])


#####################################
## Pass workflow to fit            ##
#####################################
all_workflows_fitted = all_workflows.fit(train_df)



#####################################
## Predict Future                  ##
#####################################

# Build the future date grid
future_df = make_future_frame(train_df, h=28, freq="D")

# Apply *only deterministic* recipe steps for future
future_prepped = recipe.transform_future(future_df)


all_workflows_pred = all_workflows_fitted.predict(h=h, X_df = exog_df) 

final_forecasts = ResultsTable(all_workflows_pred)
        .add_ensembles(
            name="Mean Ensemble",
            members=[workflow1, workflow2],
            method="mean")
        .add_ensembles(
            name="Weighted Ensemble (MLF internal)",
            members=[(workflow1, "LightGBM"), (workflow1, "CatBoost")],
            method="weighted",
            weights=[0.7, 0.3]
    )
final_forecasts_df = final_forecasts.dataframe()

#####################################
## Add Model Selection             ##
#####################################

selector = ModelSelection(
    cv_df=cv_final,
    metric="mae + abs(bias)",
    group_by="unique_id",      # or "model", "workflow", etc.
    top_n=1
)

best_models = selector.select()     # DataFrame of chosen models
summary = selector.summary()   # Summary of selection results from cross validation


fcst_best = final_forecasts_df.merge(
    best_models[["unique_id","workflow","engine","model"]],
    on=["unique_id","workflow","engine","model"],
    how="inner"
)

