from prophet import Prophet
import itertools
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import optuna
import numpy as np

from prophet import Prophet
import matplotlib.pyplot as plt
import datetime

import logging
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)


def tune_gridsearch(df, print_tuning_results=False, regressors=(), monthly_seasonality=False,
                    initial='730 days', period='182 days', horizon='365 days'):
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.95]
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []
    maes = []
    mapes = []
    for params in all_params:
        m = Prophet(**params).fit(df)
        if monthly_seasonality:
            m.add_seasonality(name='monthly', period=30.4, fourier_order=5)
        for regressor in regressors:
            m.add_regressor(regressor)
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        logging.getLogger('prophet').setLevel(logging.CRITICAL)
        logging.getLogger('fbprophet').setLevel(logging.CRITICAL)
        df_cv = cross_validation(m, horizon, initial=initial, period=period, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].mean())
        maes.append(df_p['mae'].mean())
        mapes.append(df_p['mape'].mean())

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mae'] = maes
    tuning_results['mape'] = mapes
    tuning_results = tuning_results.sort_values('mae')
    if print_tuning_results:
        print(tuning_results.head(5))
    return tuning_results


def tune_optuna(df, save_study_name=None, trials_number=256, metric="mae", regressors=(), monthly_seasonality=False,
                initial='730 days', period='182 days', horizon='365 days'):

    def objective(trial) -> float:
        changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 1.0)
        seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 100.0)
        holidays_prior_scale = trial.suggest_loguniform('holidays_prior_scale', 0.01, 100.0)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_range = trial.suggest_uniform('changepoint_range', 0.65, 0.95)

        m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale, seasonality_mode=seasonality_mode,
                    changepoint_range=changepoint_range)
        for regressor in regressors:
            m.add_regressor(regressor)
        if monthly_seasonality:
            m.add_seasonality(name='monthly', period=30.4, fourier_order=5)
        m.fit(df)  # Fit model with given params
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        logging.getLogger('prophet').setLevel(logging.CRITICAL)
        logging.getLogger('fbprophet').setLevel(logging.CRITICAL)
        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)

        score = df_p[metric].mean()
        return score

    if save_study_name is None:
        study = optuna.create_study(direction="minimize", load_if_exists=True)
    else:
        storage_name = "sqlite:///{}.db".format(save_study_name)
        study = optuna.create_study(study_name=save_study_name, direction="minimize", storage=storage_name,
                                    load_if_exists=True)

    study.enqueue_trial({
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "seasonality_mode": "additive",
        "changepoint_range": 0.8
    })
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials_number)
    print("Best trial:")
    best_trial = study.best_trial

    print("  ", metric, ": ", str(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

def tune_optuna_logistic(df, cap, save_study_name=None, trials_number=256, metric="mae", regressors=(), monthly_seasonality=False,
                         initial='730 days', period='182 days', horizon='365 days'):

    def objective(trial) -> float:
        changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 1.0)
        seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 100.0)
        holidays_prior_scale = trial.suggest_loguniform('holidays_prior_scale', 0.01, 100.0)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_range = trial.suggest_uniform('changepoint_range', 0.65, 0.95)

        m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale, seasonality_mode=seasonality_mode,
                    changepoint_range=changepoint_range, growth='logistic')
        for regressor in regressors:
            m.add_regressor(regressor)
        if monthly_seasonality:
            m.add_seasonality(name='monthly', period=30.4, fourier_order=5)
        df['cap'] = cap
        m.fit(df)  # Fit model with given params
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        logging.getLogger('prophet').setLevel(logging.CRITICAL)
        logging.getLogger('fbprophet').setLevel(logging.CRITICAL)
        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)

        score = df_p[metric].mean()
        return score

    if save_study_name is None:
        study = optuna.create_study(direction="minimize", load_if_exists=True)
    else:
        storage_name = "sqlite:///{}.db".format(save_study_name)
        study = optuna.create_study(study_name=save_study_name, direction="minimize", storage=storage_name,
                                    load_if_exists=True)

    study.enqueue_trial({
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "seasonality_mode": "additive",
        "changepoint_range": 0.8
    })
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials_number)
    print("Best trial:")
    best_trial = study.best_trial

    print("  ", metric, ": ", str(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))