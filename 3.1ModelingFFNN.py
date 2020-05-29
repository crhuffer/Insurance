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

from tensorboard.plugins.hparams import api as hp


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
# history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split = 0.2, verbose=0,
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
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
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
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
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
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
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
  epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
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
          epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
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

HP_NUM_UNITS_LAYER1 = hp.HParam('num_units_layer1', hp.Discrete([2, 4, 8]))
HP_NUM_UNITS_LAYER2 = hp.HParam('num_units_layer2', hp.Discrete([2, 4, 8, 16, 32, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.15, 0.2, 0.3]))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.01, 0.001]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([2000]))

log_dir = "logs/hparam_tuningv11withdropout/"

# tensorboard --logdir logs/hparam_tuningv3
#%%

METRIC_MSE = 'mse'

with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS_LAYER1, HP_NUM_UNITS_LAYER2, HP_LEARNING_RATE, HP_EPOCHS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_MSE, display_name='mse')]
  )

#%%

def train_model(hparams, logdir):
  model = keras.Sequential([
    keras.layers.Dense(hparams[HP_NUM_UNITS_LAYER1], activation='relu', input_shape=[len(X_train.columns)]),
    keras.layers.Dropout(hparams[HP_DROPOUT]),
    keras.layers.Dense(hparams[HP_NUM_UNITS_LAYER2], activation='relu'),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=hparams[HP_LEARNING_RATE])

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse', 'mae'])

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  history = model.fit(
                      np.array(X_train), np.array(y_train),
                      epochs=hparams[HP_EPOCHS], validation_data=(X_val, y_val), verbose=0,
                      callbacks=[tf.keras.callbacks.TensorBoard(logdir)])#,  # log metrics
                                 # hp.KerasCallback(logdir, hparams)])
  # model.fit(X_train, y_train, epochs=hparams[HP_EPOCHS], verbose=0)  # Run with 1 epoch to speed things up for demo purposes
  _, mse, _ = model.evaluate(X_val, y_val)
  return mse

#%%


def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    mse = train_model(hparams, run_dir)
    try:
        tf.summary.scalar(METRIC_MSE, mse, step=1)
    except:
        tf.summary.scalar(METRIC_MSE, mse, step=1)

#%%

session_num = 0

for learning_rate in HP_LEARNING_RATE.domain.values:
    for dropoutrate in HP_DROPOUT.domain.values:
        for num_units_layer1 in HP_NUM_UNITS_LAYER1.domain.values:
            for num_units_layer2 in HP_NUM_UNITS_LAYER2.domain.values:
              for epochs in HP_EPOCHS.domain.values:


                  hparams = {
                      HP_NUM_UNITS_LAYER1: num_units_layer1,
                      HP_NUM_UNITS_LAYER2: num_units_layer2,
                      HP_LEARNING_RATE: learning_rate,
                      HP_EPOCHS: epochs,
                      HP_DROPOUT: dropoutrate
                  }
                  run_name = "run-%d" % session_num
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparams[h] for h in hparams})
                  run(log_dir + run_name, hparams)
                  session_num += 1

#%%

