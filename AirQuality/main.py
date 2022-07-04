import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest

os.chdir(os.getcwd()+'/AirQuality')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.info()
test.info()

train.date_time = pd.to_datetime(train.date_time)
test.date_time = pd.to_datetime(test.date_time)

target_cols = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']
feature_cols = train.columns.difference(target_cols+['date_time'])
feature_cols

fig, axs = plt.subplots(figsize = (18, 16), ncols = 1, nrows = 3)
colors = ["palevioletred", "deepskyblue", "teal"]

for i in range(3):
    axs[i].plot(train.date_time, train[target_cols[i]], color = colors[i])
    axs[i].set_title(f'{target_cols[i]} over time')
    axs[i].set_xlabel('dates')
    axs[i].set_ylabel('value')
    


plt.savefig('plots/TargetColsPlot.jpg')


hours = [5, 10, 15, 20]

ht = train.loc[train.date_time.dt.hour.isin(hours)].copy()

fig, axs = plt.subplots(figsize=(16, 18), ncols=2, nrows=3, sharex=False,
                        gridspec_kw={'width_ratios': [1, 1.5]})

fig.suptitle("Target values distribution per month and day of week at given hours", fontsize=20)

plt.subplots_adjust(hspace = 0.3)
width=0.15
x = np.sort(ht.date_time.dt.day_of_week.unique()) + 1

for i in range(3):
    bar1 = axs[i, 0].bar(x-2*width, ht[ht.date_time.dt.hour==5].groupby(ht.date_time.dt.day_of_week)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="05:00", color="salmon")
    bar2 = axs[i, 0].bar(x-width, ht[ht.date_time.dt.hour==10].groupby(ht.date_time.dt.day_of_week)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="10:00", color="skyblue")
    bar3 = axs[i, 0].bar(x, ht[ht.date_time.dt.hour==15].groupby(ht.date_time.dt.day_of_week)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="15:00", color="teal")
    bar4 = axs[i, 0].bar(x+width, ht[ht.date_time.dt.hour==20].groupby(ht.date_time.dt.day_of_week)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="20:00", color="palevioletred")
    
    axs[i, 0].set_title(f'{target_cols[i]} distribution during the day')
    axs[i, 0].set_xlabel('Day of week')
    axs[i, 0].set_ylabel('Target value')
    axs[i, 0].set_xticks(x)
    axs[i, 0].legend(fontsize = 10)
    
x = np.sort(ht.date_time.dt.month.unique())
for i in range(3):
    bar1 = axs[i, 1].bar(x-2*width, ht[ht.date_time.dt.hour == 5].groupby(ht.date_time.dt.month)[target_cols[i]].mean(),width = width,
                edgecolor = 'black', label = '05:00', color = 'salmon')
    bar2 = axs[i, 1].bar(x-width, ht[ht.date_time.dt.hour==10].groupby(ht.date_time.dt.month)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="10:00", color="skyblue")
    bar3 = axs[i, 1].bar(x, ht[ht.date_time.dt.hour==15].groupby(ht.date_time.dt.month)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="15:00", color="teal")
    bar4 = axs[i, 1].bar(x+width, ht[ht.date_time.dt.hour==20].groupby(ht.date_time.dt.month)[target_cols[i]].mean(), width=width, 
               edgecolor="black", label="20:00", color="palevioletred")
    axs[i, 1].set_title(f'{target_cols[i]} distribution over month')
    axs[i, 1].set_xlabel = ('Month')
    axs[i, 1].set_ylabel = ('Target value')
    axs[i, 1].set_xticks(x)
    axs[i, 1].legend(fontsize = 10)


plt.savefig('plots/hourly.jpg')


df = pd.concat([train, test]).reset_index(drop = True)

df['month'] = df.date_time.dt.month
df['summer'] = df['month'].isin([6,7,8]).astype(int)
df['autumn'] = df['month'].isin([9,10,11]).astype(int)
df['winter'] = df['month'].isin([12,1,2]).astype(int)
df['spring'] = df['month'].isin([3,4,5]).astype(int)
df['weekend'] = (df.date_time.dt.day_of_week>=5).astype(int)
df['morning'] = (df.date_time.dt.hour.isin(np.arange(5,12,1))).astype(int)
df['daytime'] = (df.date_time.dt.hour.isin(np.arange(12,18,1))).astype(int)
df['night'] = (df.date_time.dt.hour.isin(np.concatenate((np.arange(18,24,1), np.arange(0, 5,1 )), axis = 0))).astype(int)


#Correlation between features
plt.figure(figsize = (16,6))

mask = np.triu(np.ones_like(train[feature_cols].corr(), dtype=bool), +1)
heatmap = sns.heatmap(train[feature_cols].corr(), vmin = -1, vmax = 1, mask = mask, annot = True, cmap = 'BrBG')

heatmap.set_title('Correlation between features', fontsize = 18, pad = 12)

plt.savefig('plots/correlationFeatures.jpg')

#Correlation between all variables
plt.figure(figsize = (25,7))

mask = np.triu(np.ones_like(train.corr(), dtype=bool), +1)
heatmap = sns.heatmap(train.corr(), vmin = -1, vmax = 1, mask = mask, annot = True, cmap = 'BrBG')

heatmap.set_title('Correlation between all variables', fontsize = 18, pad = 12)
plt.savefig('plots/correlationAllVars.jpg')



corr_list_max = []
corr_list_min = []
train_copy = train.drop('date_time', axis = 1).copy()
for i in range(train_copy.shape[1]):
    t = train_copy.copy()
    t.iloc[:,i] = t.iloc[:,i].shift(fill_value = 0)
    cr = t.corr()
    corr_list_max.append((cr.iloc[i, :].drop(labels = train_copy.columns[i]).idxmax(), train_copy.columns[i], 
                      cr.iloc[i, :].drop(labels = train_copy.columns[i]).max()))
    corr_list_min.append((cr.iloc[i, :].drop(labels = train_copy.columns[i]).idxmin(), train_copy.columns[i], 
                      cr.iloc[i, :].drop(labels = train_copy.columns[i]).min()))


for elem in corr_list_max:
    el1, el2, el3 = elem
    print('{:<40}{:<30}{}'.format(el1, el2, el3))


for elem in corr_list_min:
    el1, el2, el3 = elem
    print('{:<40}{:<30}{}'.format(el1, el2, el3))


lagged = []

for feat in feature_cols:
    new_feat = feat+'_lag'
    df[new_feat] = df[feat].shift(fill_value = 0)
    lagged.append(new_feat)


train_y = df.loc[:train.shape[0]-1, target_cols].copy()
train_X = df.loc[:train.shape[0]-1, df.columns.difference(target_cols)].copy()
test_X = df.loc[train.shape[0]:, df.columns.difference(target_cols)].reset_index(drop = True).copy()


train_y = np.log(train_y)
cb_params = [
                {'learning_rate': 0.04094650317955774,
                 'l2_leaf_reg': 8.555213318408395,
                 'bagging_temperature': 4.188124681571345,
                 'random_strength': 1.444399265342111,
                 'depth': 8,
                 'grow_policy': 'Lossguide',
                 'leaf_estimation_method': 'Gradient'},
                {'learning_rate': 0.010499552543881853,
                 'l2_leaf_reg': 2.630654006362146,
                 'bagging_temperature': 4.824439111895089,
                 'random_strength': 1.3480005087465852,
                 'depth': 4,
                 'grow_policy': 'Lossguide',
                 'leaf_estimation_method': 'Newton'},
               {'learning_rate': 0.010202325317933652,
                'l2_leaf_reg': 0.9134009064920859,
                'bagging_temperature': 8.535456442729302,
                'random_strength': 1.353469950151128,
                'depth': 10,
                'grow_policy': 'Lossguide',
                'leaf_estimation_method': 'Newton'},
            ]


all_fi = []

splits = 10

# Initializing and filling predictions dataframe with datetime values
preds = pd.DataFrame()
preds["date_time"] = test_X["date_time"].copy()

# The months will be used for folds split
months = train_X.drop(7110, axis=0)["date_time"].dt.month

total_mean_rmsle = 0

train_X.drop([7110], axis = 0, inplace = True)
train_y.drop([7110], axis = 0, inplace = True)

for i, target in enumerate(target_cols):
    print(f"\nTraining for {target}...")

    y = train_y.iloc[:,i]
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((train_X.shape[0],))
    model_preds = 0
    model_fi = 0
    for num, (train_idx, valid_idx) in enumerate(skf.split(train_X, months)):
        X_train, X_valid = train_X.loc[train_idx], train_X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        model = CatBoostRegressor(random_state=42,
                                 thread_count=4,
                                 verbose=False,
                                 loss_function='RMSE',
                                 eval_metric='RMSE',
                                 od_type="Iter",
                                 early_stopping_rounds=500,
                                 use_best_model=True,
                                 iterations=10000,
                                 **cb_params[i])
        model.fit(X_train, y_train,
                  eval_set=(X_valid, y_valid),
                  verbose=False,
                  cat_features=["night", "weekend", "morning"])
        model_preds += np.exp(model.predict(test_X)) / splits
        model_fi += model.feature_importances_
        oof_preds[valid_idx] = np.exp(model.predict(X_valid))
        oof_preds[oof_preds < 0] = 0
        print(f"Fold {num} RMSLE: {np.sqrt(mean_squared_log_error(np.exp(y_valid), oof_preds[valid_idx]))}")
#         print(f"Trees: {model.tree_count_}")
    target_rmsle = np.sqrt(mean_squared_log_error(np.exp(y), oof_preds))
    total_mean_rmsle += target_rmsle / len(target_cols)
    print(f"\nOverall {target} RMSLE: {target_rmsle}")    
    preds[target] = model_preds
    all_fi.append(dict(zip(test_X.columns, model_fi)))
print(f"\n\nTotal RMSLE is {total_mean_rmsle}\n")



feature_list = set()
for i in np.arange(len(all_fi)):
    feature_list = set.union(feature_list, set(all_fi[i].keys()))


df = pd.DataFrame(columns=["Feature"])
df["Feature"] = list(feature_list)

for i in range(len(all_fi)):
    for key in all_fi[i]:
        df.loc[df['Feature']==key, 'Importance_'+str(i+1)] = all_fi[i][key]/1000
        
df.fillna(0, inplace= True)
df.sort_values('Importance_1', axis = 0, ascending = False, inplace=True)



x = np.arange(len(df.Feature))
height = .3

fig, ax = plt.subplots(figsize = (16,7))

bar1 = ax.barh(x-height, df['Importance_1'], height = height, color = 'teal', label = target_cols[0], edgecolor = 'black')

bar2 = ax.barh(x, df['Importance_2'], height = height, color = 'skyblue', label = target_cols[1], edgecolor = 'black')    

bar3 = ax.barh(x+height, df['Importance_3'], height = height, color = 'salmon', label = target_cols[2], edgecolor = 'black')

ax.set_title('Variable Importances')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature names')
ax.set_yticks(x)
ax.set_yticklabels(df.Feature, fontsize = 12)

ax.legend(fontsize = 14, loc = 'upper right')

plt.savefig('plots/VariableImportance.jpg')
