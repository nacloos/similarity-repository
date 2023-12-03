#!/usr/local/linux/anaconda3.8/bin/python

import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import sys
import os

# use the distances in dists/feather and the taks accuracies in task_accs/feather to build the full dataframes
# given a reference representation, the corresponding full dataframe is the dataframe that contains distances and functional differences for all the relevant pairs

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

path_prefix = pathlib.Path("/scratch/users/repsim/sim_metric_resources")
scores_path = resources_path / pathlib.Path("scores/pca_deletion/scores.pkl")
dists_path = resources_path / pathlib.Path("dists/pca_deletion/dists.csv")
full_df_path = resources_path / pathlib.Path("full_dfs/pca_deletion/full_df.csv")
# dists_path = resources_path / pathlib.Path("dists/pca_deletion/dists_self_computed.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/pca_deletion/full_df_self_computed.csv")

# only consider these layers and these models - filter them before saving full_df
REF_SEEDS = [1, 2, 3, 4, 5, 6, 7]
LAYERS = [7, 8, 9, 10, 11]

# various info, some of it is probably useless
dissim_dataset = "mnli_matched_100"  # TODO: useless right
probe_task = "SST-2"


data_dict = pkl.load(open(scores_path, "rb"))
# data_dict[probe_task][pretrain_seed][layer][dims_deleted] is a list of the accuracy results over 3 probing runs
data = data_dict[probe_task]


def get_acc(data_dict, task, seed, layer, dims, run="average"):
    if run == "average":
        return np.mean(data_dict[task][seed][layer + 1][dims])
    elif run == "std":
        return np.std(data_dict[task][seed][layer + 1][dims])
    else:
        return data_dict[task][seed][layer + 1][dims][run]


def get_acc_diff(data_dict, row):
    acc1 = get_acc(
        data_dict,
        task=probe_task,
        seed=row["seed1"],
        layer=row["layer1"],
        dims=0,
        run="average",
    )
    acc2 = get_acc(
        data_dict,
        task=probe_task,
        seed=row["seed2"],
        layer=row["layer2"],
        dims=row["dims_deleted"],
        run="average",
    )
    return np.abs(acc1 - acc2)


def get_full_df(scores_path, dists_path, full_df_path):
    # get pairwise distances df
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")

    # TODO: do we prefilter layers and seeds here?
    full_df = pd.DataFrame(
        dists_df[
            (dists_df["seed1"].isin(REF_SEEDS))
            & (dists_df["seed2"].isin(REF_SEEDS))
            & (dists_df["layer1"].isin(LAYERS))
            & (dists_df["layer2"].isin(LAYERS))
        ]
    )
    print("filtered full_df layers and seeds")

    # add probing accuracy differences
    print("adding probing scores to get full_df")
    data_dict = pkl.load(open(scores_path, "rb"))
    f = lambda row: get_acc_diff(data_dict, row)
    full_df[f"{probe_task}_diff"] = full_df.apply(f, axis=1)

    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")

    return full_df


# TODO: to process this in the notebook still need to subset reference layers and seeds - we'll see about this later