#Data was taken from freshly updated kaggle dataset: https://www.kaggle.com/datasets/sandippalit009/covid19-india-state-wise-dataset-cleaned
import os

import pandas as pd
from plotnine import *
from mizani.breaks import date_breaks
import datetime
from ngboost import NGBRegressor
from ngboost.distns import Exponential
from ngboost.scores import CRPScore
from scipy.stats import expon, norm
import numpy as np


os.chdir('/Users/bekzatajan/Projects/MLprojects/CovidInIndia/')

filesindir=os.listdir('data/')

meanss=[]
stdd=[]
df_raw=pd.DataFrame()
good=[x for x in filesindir if 'csv' in x]

ColumnsDf = ['date', 'Confirmed', 'Deceased', 'Recovered']
for goody in good:
    df_csv=pd.read_csv(('data/'+goody))
    df_csv.columns = ColumnsDf
    df_csv['City'] = goody.split('.')[0]
    df_raw=df_raw.append(df_csv)


#There are some minus values, let's get rid of them 
df_raw[ColumnsDf[1:4]] = np.abs(df_raw[ColumnsDf[1:4]])

df_raw.sort_values(['date'], inplace = True)
df_raw.sort_values(['Confirmed'], inplace = True)

#Each day has data from all 37 cities
print('For each day the confirmed cases are in: {}'.format(df_raw.groupby('date')['Confirmed'].count().unique()))

df_raw.groupby('City')['Confirmed'].sum().sort_values()

# df_raw=df_raw[df_raw['City'].isin(['Maharashtra', 'Kerala', 'Karnataka'])]
#Then the data is clean and ready 

df_raw = (df_raw
    .assign(ds=lambda df: pd.to_datetime(df.date))
    .assign(year=lambda df: df.ds.dt.year)
    .assign(day=lambda df: df.ds.dt.day_name())
    )



train_max_date = df_raw.ds.max() - datetime.timedelta(weeks=3)
df_train = df_raw.query('ds <= @train_max_date')
df_test_zeroes = df_raw.query('ds > @train_max_date').assign(Recovered=lambda df: df.Recovered * 0)
df = df_train.append(df_test_zeroes)

horizon = 21  # 7 * 3
seasonality = 7  # 7 days

df_roll = (df
        .assign(lag_Confirmed=lambda df: df.groupby('City')['Confirmed'].transform(lambda x: x.shift(horizon)))
        .assign(lag_Confirmed2=lambda df: df.groupby('City')['lag_Confirmed'].transform(lambda x: x.shift(seasonality)))
        .assign(lag_Confirmed3=lambda df: df.groupby('City')['lag_Confirmed'].transform(lambda x: x.shift(seasonality * 2)))
        .assign(ma1=lambda df: df.groupby(['City', 'day'])['lag_Confirmed'].transform(lambda x: x.rolling(window=seasonality).mean()))
        .assign(ewm1=lambda df: df.groupby(['City', 'day'])['lag_Confirmed'].transform(lambda x: x.ewm(span=seasonality).mean()))
        )

df_plotting = df_roll.query('date >= "2020-03-14"')
theme_set(theme_538)
palette = ["#ee1d52", "#f2d803", "#69c9d0", "#000000"]
p = (
    ggplot(df_plotting, aes(x="ds", y="Confirmed", color="City"))
    + geom_line(aes(y = 'lag_Confirmed'), color = 'gray')
    + geom_line(aes(y = 'lag_Confirmed2'), color = 'gray')
    + geom_line(aes(y = 'lag_Confirmed3'), color = 'gray')
    + geom_line(aes(y = 'ma1'), color = 'gray')
    + geom_line(aes(y = 'ewm1'), color = 'gray')
    + geom_line()
    + geom_point()
    + scale_x_datetime(breaks=date_breaks("1 month"))
    + theme(axis_text_x=element_text(angle=45))
    + xlab("")
    + ggtitle("Confirmed cases vs Time by City")
    + scale_color_manual(palette)
    + facet_wrap(facets="City", ncol=1)
)
p.save(filename="plots/forecast_m5_state_ts_tail.jpg", width=14, height=10)


df_prep_boost = df_roll.dropna()

X_train = df_prep_boost.query('ds <= @train_max_date').loc[:, ('lag_Confirmed', 'lag_Confirmed2', 'lag_Confirmed3', 'ma1', 'ewm1')]
Y_train = df_prep_boost.query('ds <= @train_max_date').loc[:, ('Confirmed')]
X_test = df_prep_boost.query('ds > @train_max_date').loc[:, ('lag_Confirmed', 'lag_Confirmed2', 'lag_Confirmed3', 'ma1', 'ewm1')]
Y_test = df_raw.query('ds > @train_max_date').loc[:, ('Confirmed')]

ngb_exp = NGBRegressor(Dist=Exponential, verbose=True, Score=CRPScore).fit(X_train, Y_train)
Y_dists = ngb_exp.pred_dist(X_test)
array_crps = np.empty([1, 0])

# ngboost crps
def crps(y, vec_forecast):
    x = np.sort(vec_forecast)
    m = len(vec_forecast)
    return (2 / m) * np.mean((x - y) * (m * np.where(y < x, 1, 0) - np.arange(start=0, stop=m, step=1) + 1 / 2))



np.random.seed(42)
for i in range(len(X_test)):
    vec_forecast_ngboost = expon.rvs(loc=0, scale=Y_dists[i].params.get('scale'), size=5000)
    y = Y_test.iat[i]
    score = crps(y, vec_forecast_ngboost)
    array_crps = np.append(array_crps, score)


np.mean(array_crps)


# snaive crps
last_feature_date = train_max_date - datetime.timedelta(weeks=3)

X_train_last = (df_prep_boost
        .query('ds <= @train_max_date')
        .groupby(["City", "day"], as_index=False)
        .last())

X_train_sd = (df_prep_boost
        .query('ds <= @last_feature_date')
        .groupby(["City", "day"], as_index=False)
        .std()
        .loc[:, ('City', 'day', 'Confirmed')]
        .rename(columns={"Confirmed": "sd"}))

df_last = X_train_last.merge(X_train_sd, on=['City', 'day']).loc[:, ('ds', 'City', 'day', 'lag_Confirmed', 'sd')].drop(columns=['ds'])

start_forecast = df_prep_boost.query('ds > @train_max_date').ds.min()
end_forecast = df_prep_boost.query('ds > @train_max_date').ds.max()
df_dr = pd.DataFrame({"ds": pd.date_range(start=start_forecast, end=end_forecast)}).assign(day=lambda df: df.ds.dt.day_name())
df_states = pd.DataFrame({"City": X_train_last.City.unique()})

df_snaive = (df_dr
        .merge(df_states, how="cross")
        .merge(df_last, on=['City', 'day'])
        .assign(k=lambda df: df.groupby(['City', 'day'], as_index=False).cumcount() + 1)
        .assign(sd_hat=lambda df: df.sd * np.sqrt(df.k + 1)))


array_crps_snaive = np.empty([1, 0])

for i in range(len(X_test)):
    vec_forecast_snaive = norm.rvs(loc=df_snaive.lag_Confirmed.iat[i], scale=df_snaive.sd_hat.iat[i], size=5000)
    y = Y_test.iat[i]
    score = crps(y, vec_forecast_snaive)
    array_crps_snaive = np.append(array_crps_snaive, score)


#ngboost - tuning
np.mean(array_crps)
#snaive
np.mean(array_crps_snaive)
