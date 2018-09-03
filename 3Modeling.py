# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:25:20 2018

@author: crhuffer
"""

# %% Libraries

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid

# %%

mpl.rcParams['figure.figsize'] = [15, 7]

# %% Path and file declarations.

path_RawData = 'D:/CodeDDrive/BlogData/2018-08-25-Insurance/'

filename_InsuranceProcessed_train = path_RawData + 'insuranceProcessed_train.csv'
filename_InsuranceProcessed_val = path_RawData + 'insuranceProcessed_val.csv'
filename_InsuranceProcessed_test = path_RawData + 'insuranceProcessed_test.csv'

# %% Load the data

df_InsuranceProcessed_train = pd.read_csv(filename_InsuranceProcessed_train)
df_InsuranceProcessed_val = pd.read_csv(filename_InsuranceProcessed_val)
df_InsuranceProcessed_test = pd.read_csv(filename_InsuranceProcessed_test)

# %% selecting columns for modeling.

X_columns = ['age', 'bmi', 'children', 'IsSmoker', 'IsMale', 'northwest', 'southeast', 'southwest']
y_columns = 'charges'

# %% Setup data

y_train = df_InsuranceProcessed_train.loc[:, y_columns]
X_train = df_InsuranceProcessed_train.loc[:, X_columns]
y_val = df_InsuranceProcessed_val.loc[:, y_columns]
X_val = df_InsuranceProcessed_val.loc[:, X_columns]
y_test = df_InsuranceProcessed_test.loc[:, y_columns]
X_test = df_InsuranceProcessed_test.loc[:, X_columns]

# %%

def plot(x, y, z, data, colormap, x_panel, xlim=None, ylim=None, x_panel_alias=None):
    iffirstpassmakelegend=True
    fig, ax = plt.subplots(1, df_grid[x_panel].nunique(), sharex=True, sharey=True)
    for x_value in pd.Series(df_grid[x]).unique():
        for z_value in colormap.keys():
            for x_panel_index, x_panel_value in enumerate(pd.Series(df_grid[x_panel]).unique()):
                plt.sca(ax[x_panel_index])
                plt.grid()
                indexer = ((data[z]==z_value) & (data[x_panel]==x_panel_value))
                if iffirstpassmakelegend:
                    plt.plot(x, y, data=df_grid.loc[indexer, :], color=colormap[z_value], label=z+'={}'.format(z_value), marker='.')
                    
                else:
                    plt.plot(x, y, data=df_grid.loc[indexer, :], color=colormap[z_value], marker='.')
                if x_panel_alias:
                    plt.title(x_panel_alias+'\n={}'.format(x_panel_value))
                else:
                    plt.title(x_panel+'\n={}'.format(x_panel_value))
                ax[x_panel_index].set_xlabel(x)
        # After the first set of colors don't add labels to the legend.
        if iffirstpassmakelegend:
            iffirstpassmakelegend=False
            plt.legend()
    ax[0].set_xlabel(x)
    ax[0].set_ylabel(y)
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)

# %% ParamSet1

paramset = 1
params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6],
          'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.9],
#          'gamma': [0, 1, 10, 100]}
          'min_child_weight': [1, 10, 25, 50]}

# %% ParamSet2.0
paramset = 2.0
params = {'n_estimators': [10, 20, 24, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6]
          }

# %% ParamSet3.0: learning_rate
paramset = 3.0
params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6],
          'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.9]}

# %% ParamSet3.1: learning_rate higher resolution
paramset = 3.1
params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6],
          'learning_rate': [0.01, 0.04, 0.07, 0.1, 0.13, 0.16, 0.19]}

# %% ParamSet4.0: gamma
paramset = 4.0
params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6],
          'gamma': [1e4, 1e5, 1e6, 1e7, 1e8]}

# %% ParamSet4.1: gamma
paramset = 4.1
params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80 , 90, 100, 110, 120, 130, 140, 150, 200, 250],
          'max_depth': [2, 3, 4, 5, 6],
          'gamma': [3e5, 6e5, 1e6, 3e6, 6e6, 1e7, 3e7, 6e7]}

# %% -------- run XGB grid---------------------

now = pd.datetime.now()
print('Start model buildling now: {}'.format(now))
model = xgb.XGBRegressor()
grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=8, return_train_score=True)
grid.fit(X_train, y_train)

print('Ending model buildling now: {}'.format(pd.datetime.now()))
print('Modeling duration: {}'.format(pd.datetime.now()-now))

y_predict_val = pd.Series(grid.best_estimator_.predict(X_val))
y_predict_train = pd.Series(grid.best_estimator_.predict(X_train))

# %% Convert the grid to a dataframe so we can explore it

df_grid = pd.DataFrame.from_dict(grid.cv_results_)

# the parameter columns are not treated as numeric values
# grab them with the list comprehension then convert them to numeric.
list_columns = [col for col in df_grid if col.startswith('param_')]
for col in list_columns:
    df_grid[col] = pd.to_numeric(df_grid[col])

# %%

sns.boxplot(x='param_n_estimators', y='mean_test_score', data=df_grid, hue='param_max_depth')

# %%

sns.boxplot(x='param_max_depth', y='mean_test_score', data=df_grid, hue='param_n_estimators')

# %% ParamSet3.0 Plots: Larger scale
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_learning_rate', ylim=(-3e7, -2e7))

# %% ParamSet3.0 Plots: Smaller scale
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_learning_rate', ylim=(-2.4e7, -2.1e7))

# %% ParamSet3.1 Plots: Larger scale
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_learning_rate', ylim=(-3e7, -2e7), x_panel_alias='lr')

# %% ParamSet3.1 Plots: Smaller scale
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_learning_rate', ylim=(-2.25e7, -2.1e7), x_panel_alias='lr')

# %% ParamSet4.0 Plots: 
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_gamma', ylim=(-3e7, -2e7), x_panel_alias='gamma')

# %% ParamSet4.1 Plots: 
colormap = {2: 'r', 3: 'y', 4: 'g', 5: 'b', 6: 'c'}
plot('param_n_estimators', 'mean_test_score', 'param_max_depth', df_grid, colormap, 'param_gamma', ylim=(-3e7, -2e7), x_panel_alias='gamma')

# %% Running the model

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_predict_val = pd.Series(model.predict(X_val))
y_predict_train = pd.Series(model.predict(X_train))

# %% Define predicitons for train and validation

y_predict_val = pd.Series(grid.best_estimator_.predict(X_val))
y_predict_train = pd.Series(grid.best_estimator_.predict(X_train))

# %% Calculating the loss metrics

Val_RMSE = mean_squared_error(y_val, y_predict_val)
Train_RMSE = mean_squared_error(y_train, y_predict_train)
print('Val_RMSE: {}'.format(Val_RMSE))
print('Train_RMSE: {}'.format(Train_RMSE))

# %% Storing the prediction and residuals

df_InsuranceProcessed_val['Model{}Pred'.format(paramset)] = y_predict_val
df_InsuranceProcessed_val['Model{}Res'.format(paramset)] = y_predict_val - y_val
df_InsuranceProcessed_val['Model{}FracErr'.format(paramset)] = (y_predict_val - y_val)/y_val

# %% Error Analysis: Plotting actual and predicted

fig, ax = plt.subplots()
df_InsuranceProcessed_val.loc[:, ['charges', 'Model{}Pred'.format(paramset)]].plot(marker = '.', linestyle='None', ax=ax)
ax.set_ylabel('charge')
ax.set_xlabel('index')

# %% Error Analysis: Plot of residual

fig, ax = plt.subplots()
df_InsuranceProcessed_val.plot(y='Model{}Res'.format(paramset), ax=ax)
ax.set_ylabel('Residual')
ax.set_xlabel('index')

# %% Error Analysis: Hist of residual
bins = np.linspace(-25000, 7500, 100)
fig, ax = plt.subplots()
df_InsuranceProcessed_val['Model{}Res'.format(paramset)].hist(ax=ax, bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('Residual')

# %% Error Analysis: Plot of fractional error

fig, ax = plt.subplots()
df_InsuranceProcessed_val.plot(y='Model{}FracErr'.format(paramset), ax=ax)
ax.set_ylabel('Fractional Error')
ax.set_xlabel('index')

# %% Error Analysis: Hist of fractional error

bins = np.linspace(-1, 2.5, 100)
fig, ax = plt.subplots()
df_InsuranceProcessed_val['Model{}FracErr'.format(paramset)].hist(ax=ax, bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('Fractional Error')
