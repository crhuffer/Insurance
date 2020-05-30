#Import Modules

from GPyOpt.methods import BayesianOptimization

import ModelingFFNN

#%%

# import importlib
# importlib.reload(ModelingFFNNmlflow_example)

#%%

# MLFLOW_RUN_NAME = 'testing'
MLFLOW_RUN_NAME = 'Insurance2LayerFFNNgpyopt'

EPOCHS = 1000

#%%

def run(args):
    print(args[0])
    nodes, drop_rate, learning_rate = args[0]
    print( nodes, drop_rate, learning_rate)
    # nodes=4
    # subprocess.call(" python ModelingFFNN.py --nodes {}".format(nodes), shell=True)
    #
    mse = ModelingFFNN.run(nodes=nodes,
                           drop_rate=drop_rate,
                           learning_rate=learning_rate,
                           mlflow_run_name = MLFLOW_RUN_NAME,
                           epochs=EPOCHS)
    return mse


#%%

domain = [
        # {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
        {'name': 'nodes', 'type': 'discrete', 'domain': tuple([int(2**x) for x in range(1, 6)])},
        # {'name': 'nodes', 'type': 'discrete', 'domain': (2, 4, 8, 16, 32, 64)},
        {'name': 'drop_rate', 'type': 'continuous', 'domain': (1e-3, 0.5)},
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 0.5)},
        # {'name': 'nodes_layer2', 'type': 'discrete', 'domain': tuple([2**x for x in range(1, 6)])}
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

