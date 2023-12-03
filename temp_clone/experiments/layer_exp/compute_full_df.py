#!/usr/local/linux/anaconda3.8/bin/python

import pickle as pkl
import numpy as np
import pathlib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

# paths
scores_path = resources_path / pathlib.Path("scores/layer_exp/scores.pkl")
dists_path = resources_path / pathlib.Path("dists/layer_exp/dists.csv")
full_df_path = resources_path / pathlib.Path("full_dfs/layer_exp/full_df.csv")
# dists_path = resources_path / pathlib.Path("dists/layer_exp/dists_self_computed.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/layer_exp/full_df_self_computed.csv")

# list of probing tasks
task_list = ["QNLI", "SST-2"]


def get_probing_accuracy(data_dict, task, seed, depth):
    """
    average accuracy of model finetuned with finetuning seed seed on mnli
    when probing layer layer on task
    """
    # delete 0 neurons with deletion seed 0
    return np.mean(data_dict[task][seed][depth + 1][0][0])


# TODO: this is different from but better than what we had originally
def get_full_df(scores_path, dists_path, full_df_path):
    # get pairwise distances df
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")

    # add probing accuracy differences to dists_df to get full_df
    print("adding probing scores to get full_df")
    full_df = dists_df
    data_dict = pkl.load(open(scores_path, "rb"))
    for task in task_list:
        task_diff_list = []
        for _, row in dists_df.iterrows():
            acc1 = get_probing_accuracy(data_dict, task, row["seed1"], row["layer1"])
            acc2 = get_probing_accuracy(data_dict, task, row["seed2"], row["layer2"])
            task_diff_list.append(np.abs(acc1 - acc2))

        full_df[f"{task}_diff"] = np.array(task_diff_list)

    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df
