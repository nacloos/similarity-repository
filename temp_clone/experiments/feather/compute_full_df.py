#!/usr/local/linux/anaconda3.8/bin/python

import pandas as pd
import pathlib
import os
import sys

# We consider the following tasks
# - accuracy on mnli dev set
# - overall accuracy on hans samples
# - accuracy on samples from hans subcategories

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

# paths
scores_path = resources_path / pathlib.Path("scores/feather/scores.csv")
dists_path = resources_path / pathlib.Path("dists/feather/dists.csv")
full_df_path = resources_path / pathlib.Path("full_dfs/feather/full_df.csv")
# dists_path = resources_path / pathlib.Path("dists/feather/dists_self_computed.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/feather/full_df_self_computed.csv")


# Function to compute the accuracy difference between two models
def get_acc_diff(row, scores_df, task_list):
    score_row1 = scores_df.iloc[row["seed1"]]
    score_row2 = scores_df.iloc[row["seed2"]]
    for task in task_list:
        acc1 = score_row1[task]
        acc2 = score_row2[task]
        row[f"{task}_diff"] = abs(acc1 - acc2)
    return row


def rename_scores(scores_df):
    scores_df = scores_df.rename(
        columns={
            "MNLI dev acc.": "mnli_dev_acc",
            "Lexical (entailed)": "lex_ent",
            "Subseq (entailed)": "sub_ent",
            "Constituent (entailed)": "const_ent",
            "Lexical (nonent)": "lex_nonent",
            "Subseq (nonent)": "sub_nonent",
            "Constituent (nonent)": "const_nonent",
            "Overall accuracy": "overall_accuracy",
        }
    )
    return scores_df


# function to compute and save full_df
def get_full_df(scores_path, dists_path, full_df_path):
    # accuracy of 100 models on tasks from the HANS dataset
    scores_df = pd.read_csv(scores_path)[0:100]
    scores_df = rename_scores(scores_df)

    task_list = list(scores_df.columns[1:9])
    print("got scores_df")

    # distance between all pairs of all layers from the 100 models
    # (ie 12 * 12 * 100 * 100 = 1,440,000 rows)
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")

    # merge dists_df and scores_df:
    # add accuracy differences columns to the data frame
    # inefficient, takes a long time
    print("getting full_df, will take a while")
    full_df = dists_df.apply(
        lambda row: get_acc_diff(row, scores_df, task_list), axis=1
    )
    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df


# note that full_df is much larger than we need (but it's good to have it in case we want to use a different choice of reference seed)
# the notebook takes care of selecting the rows we care about (those that contain a layer from the reference seed)
# therefore the notebook also needs access to the scores_df in order to find out what the reference seed is