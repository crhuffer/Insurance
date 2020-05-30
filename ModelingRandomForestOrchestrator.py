#Import Modules

from GPyOpt.methods import BayesianOptimization
import ModelingRandomForest

#%%

import importlib
importlib.reload(ModelingRandomForest)

#%%

# MLFLOW_RUN_NAME = 'testing'
MLFLOW_RUN_NAME = 'InsuranceRandomForest'


#%%

def run(args):
    print(args[0])
    n_estimators, max_depth, min_samples_leaf, max_features = args[0]

    # nodes=4
    # subprocess.call(" python ModelingFFNN.py --nodes {}".format(nodes), shell=True)
    #
    mse = ModelingRandomForest.run(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   mlflow_run_name = MLFLOW_RUN_NAME)
    return mse


#%%

# Example of a discrete and continuous space.
# {'name': 'nodes', 'type': 'discrete', 'domain': (2, 4, 8, 16, 32, 64)},
# {'name': 'drop_rate', 'type': 'continuous', 'domain': (1e-3, 0.5)},
domain = [

        {'name': 'n_estimators', 'type': 'discrete', 'domain': tuple([int(2**x) for x in range(1, 10)])},
        {'name': 'max_depth', 'type': 'discrete', 'domain': tuple([int(x) for x in range(1, 10)])},
        {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': tuple([int(2**x) for x in range(1, 8)])},
        {'name': 'max_features', 'type': 'continuous', 'domain': (0.1, 1.0)},
    ]

#%%

myBopt_1d = BayesianOptimization(f=run, domain=domain)

#%%

myBopt_1d.run_optimization(max_iter=10000, max_time=3600)

#%%

myBopt_1d.plot_acquisition()

#%%

myBopt_1d.plot_convergence()

#%%

