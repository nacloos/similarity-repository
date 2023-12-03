#!/usr/local/linux/anaconda3.8/bin/python

import numpy as np
import pickle as pkl
import pathlib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

# new paths
scores_path = resources_path / pathlib.Path("scores/pretrain_finetune/scores.pkl")
dists_path = resources_path / pathlib.Path("dists/pretrain_finetune/dists.csv")
full_df_path = resources_path / pathlib.Path("full_dfs/pretrain_finetune/full_df.csv")
# dists_path = resources_path / pathlib.Path("dists/pretrain_finetune/dists_self_computed.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/pretrain_finetune/full_df_self_computed.csv")

# constants
num_layers = 8
num_ft_seeds = 10


def collect_scores(scores_path):
    model2correctness_tensor, data_dict = pkl.load(open(scores_path, "rb"))

    guid_set = set()
    for datapoint in data_dict:
        guid_set.add(datapoint["guid"].split("-")[0])

    acc_dict = {}
    for test_set in guid_set:
        test_set_idxes = [
            idx
            for idx, d in enumerate(data_dict)
            if d["guid"].split("-")[0] == test_set
        ]
        acc_dict[test_set] = []
        for pretraining_seed in range(1, 11):
            these_seed_accs = [
                np.mean(
                    model2correctness_tensor[pretraining_seed][finetuning_seed][
                        test_set_idxes
                    ]
                )
                for finetuning_seed in range(1, 11)
            ]
            acc_dict[test_set].append(these_seed_accs)
        acc_dict[test_set] = np.array(acc_dict[test_set])

    # Add inputs corresponding to lex_nonent to acc_dict
    lex_nonent_idxes = [
        idx
        for idx, d in enumerate(data_dict)
        if ("HANS" in d["guid"])
        and (d["heuristic"] == "lexical_overlap")
        and (d["label"] == "non-entailment")
    ]
    acc_dict["lex_nonent"] = []

    for pretraining_seed in range(1, 11):
        these_seed_accs = [
            np.mean(
                model2correctness_tensor[pretraining_seed][finetuning_seed][
                    lex_nonent_idxes
                ]
            )
            for finetuning_seed in range(1, 11)
        ]
        acc_dict["lex_nonent"].append(these_seed_accs)
    acc_dict["lex_nonent"] = np.array(acc_dict["lex_nonent"])
    guid_set.add("lex_nonent")

    return guid_set, acc_dict


def get_accuracy(acc_dict, stress_test, pretraining_seed, finetuning_seed):
    return acc_dict[stress_test][pretraining_seed][finetuning_seed]


def get_acc_diff(acc_dict, stress_test, pre_seed1, pre_seed2, fine_seed1, fine_seed2):
    avg_acc1 = get_accuracy(acc_dict, stress_test, pre_seed1, fine_seed1)
    avg_acc2 = get_accuracy(acc_dict, stress_test, pre_seed2, fine_seed2)
    return np.abs(avg_acc2 - avg_acc1)


def add_acc_diff_cols(dists_df, acc_dict, guid_set):
    for stress_test in guid_set:
        new_column = []
        # checked manually that this ordering of the seeds is compatible with dists_df
        for pre_seed1 in range(1, 11):
            for fine_seed1 in range(1, 11):
                for pre_seed2 in range(pre_seed1, 11):
                    for fine_seed2 in range(1, 11):
                        if pre_seed2 == pre_seed1 and fine_seed2 < fine_seed1:
                            continue
                        else:
                            new_column += num_layers * [
                                get_acc_diff(
                                    acc_dict,
                                    stress_test,
                                    pre_seed1 - 1,
                                    pre_seed2 - 1,
                                    fine_seed1 - 1,
                                    fine_seed2 - 1,
                                )
                            ]
        dists_df[f"{stress_test}_diff"] = np.array(new_column)
    return dists_df


def get_full_df(scores_path, dists_path, full_df_path):
    dists_df = pd.read_csv(dists_path)
    dists_df = dists_df.rename(
        columns={
            "step1": "fine_seed1",
            "step2": "fine_seed2",
            "seed1": "pre_seed1",
            "seed2": "pre_seed2",
        }
    )
    print("got dists_df")

    print("adding probing scores to get full_df")
    guid_set, acc_dict = collect_scores(scores_path)
    full_df = add_acc_diff_cols(dists_df, acc_dict, guid_set)

    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")

    return full_df