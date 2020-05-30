import pandas as pd
import mlflow.sklearn
from mlflow import log_metric

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from time import time

#%%

def build_model(n_estimators=100,
                max_depth=5,
                min_samples_leaf=1,
                max_features=1.0):

    model = RandomForestRegressor(n_estimators=int(n_estimators),
                                  max_depth=int(max_depth),
                                  min_samples_leaf=int(min_samples_leaf),
                                  max_features=max_features)

    return model


def compile_and_run_model(model, X_train, y_train, X_validation, y_validation):

    model.fit(X_train, y_train)

    y_validation_predict = model.predict(X_validation)

    score = mean_squared_error(y_validation, y_validation_predict),\
            mean_absolute_error(y_validation, y_validation_predict)

    print('Test mse:', score[0])
    print('Test mae:', score[1])

    print("First few predictions for Y:")
    for index in range(5):
        print('Prediction: {}, Actual: {}'.format(model.predict(X_validation)[index], y_validation.iloc[index]))

    return ([score[0], score[1]])


#%%


def run(n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        max_features=1.0,
        mlflow_run_name = 'unnamed'):

    print("n_estimators", n_estimators)
    print('max_depth', max_depth)
    print("min_samples_leaf", min_samples_leaf)
    print("max_features", max_features)

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

    model = build_model(n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features)

    start_time = time()
    mlflow.set_experiment('randomforestv1')
    with mlflow.start_run(run_name=mlflow_run_name):
        results = compile_and_run_model(model, X_train, y_train, X_val, y_val)

        print("n_estimators", n_estimators)
        print('max_depth', max_depth)
        print("min_samples_leaf", min_samples_leaf)
        print("max_features", max_features)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)

        timed = time() - start_time

        print("This model took", timed, "seconds to train and test.")
        log_metric("Time to run", timed)
        log_metric('mse', float(results[0]))
        log_metric('mae', float(results[1]))

    mlflow.end_run()
    return results[0]


if __name__ == '__main__':
    run()