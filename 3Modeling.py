# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:25:20 2018

@author: crhuffer
"""

# %% Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import xgboost as xgb

from sklearn.metrics import mean_squared_error

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
# %% Running the model


model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_predict_val = pd.Series(model.predict(X_val))

# %% Calculating the loss metrics

Val_RMSE = mean_squared_error(y_val, y_predict_val)
print('Val_RMSE: {}'.format(Val_RMSE))

# %% Storing the prediction and residuals

df_InsuranceProcessed_val['Model1Pred'] = y_predict_val
df_InsuranceProcessed_val['Model1Res'] = y_predict_val - y_val
df_InsuranceProcessed_val['Model1FracErr'] = (y_predict_val - y_val)/y_val

# %% Error Analysis: Plotting actual and predicted

fig, ax = plt.subplots()
df_InsuranceProcessed_val.loc[:, ['charges', 'Model1Pred']].plot(marker = '.', linestyle='None', ax=ax)
ax.set_ylabel('charge')
ax.set_xlabel('index')

# %% Error Analysis: Plot of residual

fig, ax = plt.subplots()
df_InsuranceProcessed_val.plot(y='Model1Res', ax=ax)
ax.set_ylabel('Residual')
ax.set_xlabel('index')


# %% Error Analysis: Hist of residual
bins = np.linspace(-25000, 7500, 100)
fig, ax = plt.subplots()
df_InsuranceProcessed_val['Model1Res'].hist(ax=ax, bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('Residual')

# %% Error Analysis: Plot of fractional error

fig, ax = plt.subplots()
df_InsuranceProcessed_val.plot(y='Model1FracErr', ax=ax)
ax.set_ylabel('Fractional Error')
ax.set_xlabel('index')

# %% Error Analysis: Hist of fractional error

bins = np.linspace(-1, 2.5, 100)
fig, ax = plt.subplots()
df_InsuranceProcessed_val['Model1FracErr'].hist(ax=ax, bins=bins)
ax.set_ylabel('Counts')
ax.set_xlabel('Fractional Error')