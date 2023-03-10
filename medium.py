import pandas as pd
from prophet import Prophet
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


df = pd.read_csv('medium_posts.csv')
df.head()

from prophet.diagnostics import cross_validation

pd.DataFrame().columns.drop()

m = Prophet(changepoint_prior_scale=1, seasonality_prior_scale=2
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
holidays_prior_scale = holidays_prior_scale, seasonality_mode = seasonality_mode,
changepoint_range = changepoint_range)

m.add_seasonality()