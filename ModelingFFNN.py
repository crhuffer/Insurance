import pandas as pd
import argparse
import mlflow.sklearn
from mlflow import log_metric


from tensorflow import keras

from time import time

#%%

#
#build a  a network
#
def build_model(in_dim=20, drate=0.5, nodes=64):

    model = keras.Sequential()
    model.add(keras.layers.Dense(int(nodes), input_dim=in_dim, activation='relu'))
    if drate:
        model.add(keras.layers.Dropout(drate))
    model.add(keras.layers.Dense(int(nodes), activation='relu'))
    if drate:
        model.add(keras.layers.Dropout(drate))
    model.add(keras.layers.Dense(1, activation='relu'))

    return model
#
# compile a network
#
def compile_and_run_model(model, X_train, y_train, X_validation, y_validation, epochs=20,
                          train_batch_size=128, evaluation_batch_size = 1000,
                          learning_rate=0.1):
    #
    # compile the model
    #

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    # train the model
    #
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=train_batch_size,
              verbose=0)
    #
    # evaluate the network
    #
    score = model.evaluate(X_validation, y_validation, batch_size=evaluation_batch_size)
    print('Test mse:', score[0])
    print('Test mae:', score[1])

    print("First few predictions for Y:")
    for index in range(5):
        print('Prediction: {}, Actual: {}'.format(model.predict(X_validation)[index], y_validation.iloc[index]))
    model.summary()

    return ([score[0], score[1]])

def run(nodes=4, drop_rate=0.5, learning_rate=0.1,
        evaluation_batch_size=128, train_batch_size=128, epochs=1000,
        mlflow_run_name='unnamed'):

    print("drop_rate", drop_rate)
    print('learning_rate', learning_rate)
    print("size", evaluation_batch_size)
    print("nodes", nodes)
    print("train_batch_size", train_batch_size)
    print("epochs", epochs)

    # %% Path and file declarations.

    path_Data = 'C:/Data/BlogData/2018-08-25-Insurance/'

    filename_InsuranceProcessed_train = path_Data + 'insuranceProcessed_train.csv'
    filename_InsuranceProcessed_val = path_Data + 'insuranceProcessed_val.csv'
    filename_InsuranceProcessed_test = path_Data + 'insuranceProcessed_test.csv'

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
    X_test = df_InsuranceProcessed_test.loc[:, X_columns]
    y_test = df_InsuranceProcessed_test.loc[:, y_columns]

    model = build_model(in_dim=len(X_train.columns), drate=drop_rate, nodes=nodes)

    start_time = time()
    with mlflow.start_run(run_name=mlflow_run_name):
        results = compile_and_run_model(model, X_train, y_train, X_val, y_val, epochs=epochs,
                                        train_batch_size=train_batch_size, evaluation_batch_size=evaluation_batch_size,
                                        learning_rate=learning_rate)
        mlflow.log_param("drop_rate", drop_rate)

        mlflow.log_param("evaluation_batch_size", evaluation_batch_size)
        mlflow.log_param("nodes", nodes)
        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        timed = time() - start_time

        print("This model took", timed, "seconds to train and test.")
        log_metric("Time to run", timed)
        log_metric('mse', float(results[0]))
        log_metric('mae', float(results[1]))

    mlflow.end_run()
    return results[0]


if __name__ == '__main__':
    run()