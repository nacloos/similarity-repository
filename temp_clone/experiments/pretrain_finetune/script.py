import pathlib
import pandas as pd
import numpy as np
import sys
import os
from icecream import ic

from compute_full_df import collect_scores

sys.path.append(os.path.abspath("../"))
from utils import aggregate_rank_corrs, plot_rank_corrs

sys.path.append(os.path.abspath("../../"))
from paths import resources_path

scores_path = resources_path / pathlib.Path("scores/pretrain_finetune/scores.pkl")
full_df_path = resources_path / pathlib.Path("full_dfs/pretrain_finetune/full_df.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/pretrain_finetune/full_df_self_computed.csv")

results_path = pathlib.Path("results.txt")

# Load `full_df`
full_df = pd.read_csv(full_df_path)

# Dataframe subsetting
def best_seed_pair(task):
    _, acc_dict = collect_scores(scores_path)
    acc_array = acc_dict[task].flatten()
    idxs = acc_array.argsort()[-1:][::-1]
    ref_seeds = []
    for idx in idxs:
        ref_seeds.append((int(idx / 10), idx % 10))
    return ref_seeds[0]


def ftvft_sub_df(df, task, ref_depth):
    # find best seed for the task
    best_pre_seed, best_fine_seed = best_seed_pair(task)

    # select rows of full_df corresponding to the reference layer (layer depth and both seeds)
    sub_df = df[
        (df.layer1 == ref_depth)
        & (df.layer2 == ref_depth)
        & (
            ((df.pre_seed1 == best_pre_seed) & (df.fine_seed1 == best_fine_seed))
            | ((df.pre_seed2 == best_pre_seed) & (df.fine_seed2 == best_fine_seed))
        )
    ]

    return sub_df


# Rank correlation results

METRICS = ["Procrustes", "CKA", "PWCCA"]
num_layers = 8  # medium

## First task: `antonymy`
task = "STRESS_ANTONYMY"
rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
    full_df, task, num_layers, METRICS, ftvft_sub_df
)

# average all of these over the different reference layers
with open(results_path, "a") as f:
    f.write(f"task: {task}\n")
    for metric in METRICS:
        avg_rho = round(np.mean(rho[metric]), 3)
        avg_rho_p = format(np.mean(rho_p[metric]), ".1e")
        avg_tau = round(np.mean(tau[metric]), 3)
        avg_tau_p = format(np.mean(tau_p[metric]), ".1e")

        ic(metric, avg_rho, avg_rho_p, avg_tau, avg_tau_p)
        f.write(f"metric: {metric}\n")
        f.write(f"avg_rho: {avg_rho}\n")
        f.write(f"avg_rho_p: {avg_rho_p}\n")
        f.write(f"avg_tau: {avg_tau}\n")
        f.write(f"avg_tau_p: {avg_tau_p}\n")
        f.write("\n")


plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, title=task)

## Second task: `numerical`

task = "STRESS_NUMERICAL"
rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
    full_df, task, num_layers, METRICS, ftvft_sub_df
)


# average all of these over the different reference layers
with open(results_path, "a") as f:
    f.write(f"task: {task}\n")
    for metric in METRICS:
        avg_rho = round(np.mean(rho[metric]), 3)
        avg_rho_p = format(np.mean(rho_p[metric]), ".1e")
        avg_tau = round(np.mean(tau[metric]), 3)
        avg_tau_p = format(np.mean(tau_p[metric]), ".1e")

        ic(metric, avg_rho, avg_rho_p, avg_tau, avg_tau_p)
        f.write(f"metric: {metric}\n")
        f.write(f"avg_rho: {avg_rho}\n")
        f.write(f"avg_rho_p: {avg_rho_p}\n")
        f.write(f"avg_tau: {avg_tau}\n")
        f.write(f"avg_tau_p: {avg_tau_p}\n")
        f.write("\n")

plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, title=task)
