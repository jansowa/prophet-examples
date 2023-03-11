from prophet import Prophet
import itertools
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import optuna

import logging
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)


def tune_gridsearch(df):
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
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        df_cv = cross_validation(m, '365 days', initial='730 days', period='180 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        maes.append(df_p['mae'].values[0])
        mapes.append(df_p['mape'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mae'] = maes
    tuning_results['mape'] = mapes
    print(tuning_results.sort_values('mae').head(5))
    return tuning_results


def tune_optuna(df, save_study_name=None, trials_number=256, metric="mae"):

    def objective(trial) -> float:
        changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 1.0)
        seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 100.0)
        holidays_prior_scale = trial.suggest_loguniform('holidays_prior_scale', 0.01, 100.0)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_range = trial.suggest_uniform('changepoint_range', 0.65, 0.95)

        m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale, seasonality_mode=seasonality_mode,
                    changepoint_range=changepoint_range).fit(df)  # Fit model with given params
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)

        score = df_p[metric].values[0]
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
