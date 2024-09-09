import random
from math import log10
from itertools import product

import torch.utils
import torch.utils.data
from exercise_code.solver import Solver
import gc
import torch
import os

i2dl_exercises_path=os.path.dirname(os.path.abspath(os.getcwd()))
temp_dir=os.path.join(i2dl_exercises_path,"temp")
os.makedirs(temp_dir, exist_ok=True)
weight_paths=os.path.join(temp_dir, "best_model_weights.pth")

ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def grid_search(train_dataset, val_dataset,
                grid_search_spaces, model, max_epochs=20, patience=5):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Required arguments:
        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

    Optional arguments:
        - grid_search_spaces: a dictionary where every key corresponds to a
        to-tune-hyperparameter and every value contains a list of possible
        values. Our function will test all value combinations which can take
        quite a long time. If we don't specify a value here, we will use the
        default values of both our chosen model as well as our solver
        - model: our selected model for this exercise
        - epochs: number of epochs we are training each model
        - patience: if we should stop early in our solver

    Returns:
        - The best performing model
        - A list of all configurations and results
    """
    configs = []

    """
    # Simple implementation with nested loops
    for lr in grid_search_spaces["learning_rate"]:
        for reg in grid_search_spaces["reg"]:
            configs.append({"learning_rate": lr, "reg": reg})
    """

    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(train_dataset, val_dataset, configs, max_epochs, patience,
                          model)



def random_search(train_dataset, val_dataset,
                  random_search_spaces, model, num_search=20):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.

    See the grid search documentation above.

    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """
    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(train_dataset, val_dataset, configs, model)


def findBestConfig(train_dataset, val_dataset, configs, model_type):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """
    
    best_val = None
    best_config = None
    best_model = None
    results = []
    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i+1), len(configs)),configs[i])
        batch_size=configs[i].get('batch_size', 25)
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = model_type(configs[i]).to(configs[i]["device"])
        solver = Solver(model, train_loader, val_loader, configs[i])
        solver.train(**configs[i])
        results.append(solver.best_model_stats)

        if not best_val or solver.best_model_stats["val_loss"] < best_val:
            best_params = {keys: torch.clone(values) for keys,values in model.state_dict().items()}
            gc.collect()  # Force garbage collection, to free memory
            torch.save(best_params, weight_paths)
            best_val, best_config = solver.best_model_stats["val_loss"], configs[i]
    best_model=model_type(best_config).to(best_config["device"])
    best_model.load_state_dict(best_params)

    print("\nSearch done. Best Val Loss = {}".format(best_val))
    print("Best Config:", best_config)

    
    return best_model, best_config, list(zip(configs, results))
        

def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """
    
    config = {}

    for key, (rng, mode)  in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <=0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10**(sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config


def calculate_validation_thing(model, val_loader, config):
    import torch
    from exercise_code.loss import Loss
    model.eval()
    with torch.no_grad():
        val_loss=0
        for X,y in iter(val_loader):
            X=X.view(X.shape[0],-1).to("mps")
            y=y.to("mps")
            y_pred=model(X)
            loss=Loss(**config).to("mps")
            val_loss+=loss.compute_loss_without_regularization(y_pred,y).item()
        val_loss/=len(val_loader)
    return val_loss