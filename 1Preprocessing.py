# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:25:20 2018

@author: crhuffer
"""

#%% Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#%% Path and file declarations.

# downloaded from https://www.kaggle.com/mirichoi0218/insurance/data#
path_Data = 'C:/Data/BlogData/2018-08-25-Insurance/'

filename_Insurance = path_Data + 'insurance.csv'
filename_InsuranceProcessed = path_Data + 'insuranceProcessed.csv'
filename_InsuranceProcessed_train = path_Data + 'insuranceProcessed_train.csv'
filename_InsuranceProcessed_val = path_Data + 'insuranceProcessed_val.csv'
filename_InsuranceProcessed_test = path_Data + 'insuranceProcessed_test.csv'

#%% Load the data

df_Insurance = pd.read_csv(filename_Insurance)

#%% Make the processed dataframe

df_InsuranceProcessed = df_Insurance.copy()

#%% Dummifying variables

# smoker
df_InsuranceProcessed['IsSmoker'] = df_InsuranceProcessed['smoker'].replace({'yes': 1, 'no': 0})

# sex
df_InsuranceProcessed['IsMale'] = df_InsuranceProcessed['sex'].replace({'male': 1, 'female': 0})

# region
df_InsuranceProcessed = df_InsuranceProcessed.join(pd.get_dummies(df_InsuranceProcessed['region']).iloc[:, 1:], how='left')


#%% Saving the processed data

df_InsuranceProcessed.to_csv(filename_InsuranceProcessed)

#%% Train, validation, test split 70/20/10

df_InsuranceProcessed_train, df_InsuranceProcessed_valandtest = train_test_split(df_InsuranceProcessed, train_size=.7)
df_InsuranceProcessed_val, df_InsuranceProcessed_test = train_test_split(df_InsuranceProcessed_valandtest, train_size=.66)

#%%

df_InsuranceProcessed_train.to_csv(filename_InsuranceProcessed_train)
df_InsuranceProcessed_val.to_csv(filename_InsuranceProcessed_val)
df_InsuranceProcessed_test.to_csv(filename_InsuranceProcessed_test)
#%%

