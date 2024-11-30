import os
import pickle
from itertools import product
import pandas as pd


def open_dataset(filepath):
    """
    Utility function to open a data set given a filepath.

    Inputs:
        filepath : (string) path to allen institute packaged data

    Outputs:
        data : (xarray.DataArray) data structure
    """
    if os.path.isfile(filepath):
        if pd.__version__ >= "1.0":
            data = pd.read_pickle(open(filepath, "rb"))
        else:
            data = pickle.load(open(filepath, "rb"))
        return data
    else:
        raise ValueError(f"{filepath} is not a file or does not exist.")


def dict_to_str(adict):
    """Converts a dictionary (e.g. hyperparameter configuration) into a string"""
    return "".join("{}{}".format(key, val) for key, val in sorted(adict.items()))


def iterate_dicts(inp):
    """Computes cartesian product of parameters
    From: https://stackoverflow.com/questions/10472907/how-to-convert-dictionary-into-string"""
    return list((dict(zip(inp.keys(), values)) for values in product(*inp.values())))


def get_params_from_workernum(worker_num, param_lookup, group_idx=None):
    if group_idx is None:
        return param_lookup[worker_num]
    else:
        # group_idx should only be used when there are over 1000 jobs
        # 1000 jobs is the sherlock cluster job submission limit
        worker_num = int(worker_num) + (1000 * group_idx)
        worker_num = str(worker_num)
        return param_lookup[worker_num]


def get_base_arch_name(model_name):
    """
    This function returns the base architecture name given a model name. This
    will be useful when untrained models are provided by the user and we need
    to obtain its base architecture name to obtain, e.g., image transforms.
    Untrained models will have the following arch name syntax: "untrained_{arch_name}"

    Inputs:
        model_name : (string) model identifier

    Outputs:
        arch_name  : (string) base architecture name
        trained    : (boolean) whether or not the base architecture is trained
    """

    if "untrained" == model_name.split("_")[0]:
        arch_name = model_name[len("untrained") + 1 :]
        trained = False
    else:
        arch_name = model_name
        trained = True

    return arch_name, trained
