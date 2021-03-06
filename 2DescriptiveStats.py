# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:25:20 2018

@author: crhuffer
"""

#%% Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Path and file declarations.

path_Data = 'C:/Data/BlogData/2018-08-25-Insurance/'

filename_InsuranceProcessed_train = path_Data + 'insuranceProcessed_train.csv'
filename_InsuranceProcessed_val = path_Data + 'insuranceProcessed_val.csv'
filename_InsuranceProcessed_test = path_Data + 'insuranceProcessed_test.csv'

#%% Load the data

df_InsuranceProcessed_train = pd.read_csv(filename_InsuranceProcessed_train)
df_InsuranceProcessed_val = pd.read_csv(filename_InsuranceProcessed_val)
df_InsuranceProcessed_test = pd.read_csv(filename_InsuranceProcessed_test)

#%% Charges: plot

fig, ax = plt.subplots()
df_InsuranceProcessed_train['charges'].plot(marker='.', linestyle='none')
fig.show()

#%% Charges: Histogram

fig, ax = plt.subplots()
df_InsuranceProcessed_train['charges'].hist(bins=100)
ax.set_ylabel('Counts')
ax.set_xlabel('Charges')

#%% Insurance: Exploring age vs sex: boxplot

fig, ax = plt.subplots()
sns.boxplot(y='age', x='sex', data=df_InsuranceProcessed_train)

#%% Insurance: Exploring age vs sex: histograms

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
indexer = df_InsuranceProcessed_train['sex'] == 'male'
df_InsuranceProcessed_train.loc[indexer, 'age'].hist(label='male', color='b', ax=ax[0])
indexer = df_InsuranceProcessed_train['sex'] == 'female'
df_InsuranceProcessed_train.loc[indexer, 'age'].hist(label='female', color='r', ax=ax[1])
for axes in ax:
    axes.set_ylabel('Counts')
    
ax[1].set_xlabel('Age [yrs]')
plt.legend()

#%% Insurance: Exploring age vs sex: histograms one plot

fig, ax = plt.subplots()
indexer = df_InsuranceProcessed_train['sex'] == 'male'
df_InsuranceProcessed_train.loc[indexer, 'age'].hist(label='male', color='b', alpha=0.5)
indexer = df_InsuranceProcessed_train['sex'] == 'female'
df_InsuranceProcessed_train.loc[indexer, 'age'].hist(label='female', color='r', alpha=0.5)
ax.set_ylabel('Counts')
    
ax.set_xlabel('Age [yrs]')
plt.legend()

#%% Insurance: Exploring age vs sex: describes

print('All rows: \n{}'.format(df_InsuranceProcessed_train['age'].describe()))
indexer = df_InsuranceProcessed_train['sex'] == 'male'
print('\nMales: \n{}'.format(df_InsuranceProcessed_train.loc[indexer, 'age'].describe()))
indexer = df_InsuranceProcessed_train['sex'] == 'female'
print('\nFemales: \n{}'.format(df_InsuranceProcessed_train.loc[indexer, 'age'].describe()))

#%% Insurance: Exploring age vs sex: Boxplot of target

fig, ax = plt.subplots()
sns.boxplot(y='charges', x='age', hue='sex', data=df_InsuranceProcessed_train)

#%% Insurance: Exploring age vs sex: Boxplot of target

fig, ax = plt.subplots()
indexer = df_InsuranceProcessed_train['sex'] == 'male'
plt.scatter(y='charges', x='age', data=df_InsuranceProcessed_train.loc[indexer], marker='.', color='b', label='male')
indexer = df_InsuranceProcessed_train['sex'] == 'female'
plt.scatter(y='charges', x='age', data=df_InsuranceProcessed_train.loc[indexer], marker='.', color='r', label='female')
plt.legend()
ax.set_ylabel('charges (USD)')
ax.set_xlabel('age (yrs)')

#%% Insurance: Exploring age vs sex: Boxplot of target

fig, ax = plt.subplots()
sns.boxplot(y='charges', x='sex', hue='age', data=df_InsuranceProcessed_train)

#%% Insurance: Regions: value_counts

df_InsuranceProcessed_train['region'].value_counts()

#%% 

#fig, ax = plt.subplots()
#df_InsuranceProcessed_train['']

#%%

