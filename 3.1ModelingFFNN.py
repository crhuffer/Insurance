# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:25:20 2018

@author: crhuffer
"""

#%% Libraries

import pandas as pd
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid

#%%

mpl.rcParams['figure.figsize'] = [15, 7]

#%% Path and file declarations.

path_Data = 'C:/Data/BlogData/2018-08-25-Insurance/'

filename_InsuranceProcessed_train = path_Data + 'insuranceProcessed_train.csv'
filename_InsuranceProcessed_val = path_Data + 'insuranceProcessed_val.csv'
filename_InsuranceProcessed_test = path_Data + 'insuranceProcessed_test.csv'

#%% Load the data

df_InsuranceProcessed_train = pd.read_csv(filename_InsuranceProcessed_train)
df_InsuranceProcessed_val = pd.read_csv(filename_InsuranceProcessed_val)
df_InsuranceProcessed_test = pd.read_csv(filename_InsuranceProcessed_test)

#%% selecting columns for modeling.

X_columns = ['age', 'bmi', 'children', 'IsSmoker', 'IsMale', 'northwest', 'southeast', 'southwest']
y_columns = 'charges'

#%% Setup data

y_train = df_InsuranceProcessed_train.loc[:, y_columns]
X_train = df_InsuranceProcessed_train.loc[:, X_columns]
y_val = df_InsuranceProcessed_val.loc[:, y_columns]
X_val = df_InsuranceProcessed_val.loc[:, X_columns]
X_test = df_InsuranceProcessed_test.loc[:, X_columns]
y_test = df_InsuranceProcessed_test.loc[:, y_columns]

#%%

def build_model(nodes_layer1, nodes_layer2, learning_rate):
  model = keras.Sequential([
    keras.layers.Dense(nodes_layer1, activation='relu', input_shape=[len(X_train.columns)]),
    keras.layers.Dense(nodes_layer2, activation='relu'),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

#%%

model = build_model()

#%%

model.summary()

#%%

EPOCHS = 1000
NODES_LAYER1 = 64
NODES_LAYER2 = 64
LEARNING_RATE = 0.001

#%%

# # Train the estimator
# EPOCHS = 1000
#
# history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split = 0.2, verbose=1,
#           callbacks=[tfdocs.modeling.EpochDots()],
#           validation_data=(X_val, y_val))

#%%



EPOCHS = 1000
NODES_LAYER1 = 64
NODES_LAYER2 = 64
LEARNING_RATE = 0.001

model = build_model(learning_rate = LEARNING_RATE, nodes_layer1 = NODES_LAYER1, nodes_layer2 = NODES_LAYER2)

log_dir = "train/linreg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
  np.array(X_train), np.array(y_train),
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1,
  callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback])

#%%

EPOCHS = 1000
NODES_LAYER1 = 32
NODES_LAYER2 = 32
LEARNING_RATE = 0.001

model = build_model(learning_rate = LEARNING_RATE, nodes_layer1 = NODES_LAYER1, nodes_layer2 = NODES_LAYER2)

log_dir = "train/linreg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
  np.array(X_train), np.array(y_train),
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1,
  callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback])

#%%

EPOCHS = 1000
NODES_LAYER1 = 32
NODES_LAYER2 = 32
LEARNING_RATE = 0.01

model = build_model(learning_rate = LEARNING_RATE, nodes_layer1 = NODES_LAYER1, nodes_layer2 = NODES_LAYER2)

log_dir = "train/linreg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
  np.array(X_train), np.array(y_train),
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1,
  callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback])

#%%

EPOCHS = 1000
NODES_LAYER1 = 2
NODES_LAYER2 = 2
LEARNING_RATE = 0.001

model = build_model(learning_rate = LEARNING_RATE, nodes_layer1 = NODES_LAYER1, nodes_layer2 = NODES_LAYER2)

log_dir = "train/linreg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
  np.array(X_train), np.array(y_train),
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1,
  callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback])

#%%


EPOCHS = 2000
NODES_LAYER1 = 4
NODES_LAYER2 = 4
LEARNING_RATE = 0.001

for NODES_LAYER1 in [4, 8, 16, 32, 64]:
    for NODES_LAYER2 in [4, 8, 16, 32, 64]:

        model = build_model(learning_rate = LEARNING_RATE, nodes_layer1 = NODES_LAYER1, nodes_layer2 = NODES_LAYER2)

        log_dir = "train/nodesearch2000epochs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(
          np.array(X_train), np.array(y_train),
          epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1,
          callbacks=[tfdocs.modeling.EpochDots(), tensorboard_callback])


#%%

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#%%

plotter = tfdocs.plots.HistoryPlotter()

#%%

plotter.plot({'Basic': history}, metric = "mae")
# plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

#%%

paramset = 1
params = {'nodes_layer1': [2, 4, 8, 16, 32, 64],
          'nodes_layer2': [2, 4, 8, 16, 32, 64],
          'learning_rate': [0.0001, 0.001, 0.01, 0.1],
          'epochs': [1000]}

#%%

history.history.keys()

#%%

