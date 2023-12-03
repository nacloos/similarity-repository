import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import sys
import os
from icecream import ic


sys.path.append(os.path.abspath("../"))
from utils import plot_rank_corrs, get_rank_corrs

sys.path.append(os.path.abspath("../../"))
from paths import resources_path

scores_path = resources_path / pathlib.Path("scores/layer_exp/scores.pkl")
full_df_path = resources_path / pathlib.Path("full_dfs/layer_exp/full_df.csv")
# full_df_path = resources_path / pathlib.Path("full_dfs/layer_exp/full_df_self_computed.csv")

results_path = pathlib.Path("results.txt")

# Load full_df
full_df = pd.read_csv(full_df_path)

# Dataframe subsetting
def best_probing_seed(task, ref_depth, list_ref_seeds):
    data_dict = pkl.load(open(scores_path, "rb"))
    list_to_max = [
        np.mean(data_dict[task][seed][ref_depth + 1][0][0]) for seed in list_ref_seeds
    ]
    idx, _ = max(enumerate(list_to_max), key=lambda x: x[1])
    return list_ref_seeds[idx]


def layer_sub_df(df, ref_depth, ref_seed):
    sub_df = df.loc[
        ((df["seed1"] == ref_seed) & (df["layer1"] == ref_depth))
        | ((df["seed2"] == ref_seed) & (df["layer2"] == ref_depth))
    ].reset_index()

    # TODO: maybe get rid of this assert
    num_layers = 12
    assert len(sub_df) == num_layers * 10
    return sub_df


# Rank correlation results

# the function to aggregate rank correlations is a bit different from the other experiments since in this case we sometimes have to aggregate both over depths and over seeds
def aggregate_rank_corrs(df, task, layer_depths, list_ref_seeds, METRICS, sub_df_fn):
    rho = {metric: [] for metric in METRICS}
    rho_p = {metric: [] for metric in METRICS}
    tau = {metric: [] for metric in METRICS}
    tau_p = {metric: [] for metric in METRICS}
    bad_fracs = {metric: [] for metric in METRICS}

    for ref_depth in layer_depths:
        ref_seed = best_probing_seed(task, ref_depth, list_ref_seeds)
        sub_df = sub_df_fn(df, ref_depth, ref_seed)
        for metric in METRICS:
            rho_corr, rho_os_p, tau_corr, tau_os_p, bad_frac = get_rank_corrs(
                sub_df, metric, task
            )

            rho[metric].append(rho_corr)
            rho_p[metric].append(rho_os_p)
            tau[metric].append(tau_corr)
            tau_p[metric].append(tau_os_p)
            bad_fracs[metric].append(bad_frac)
    return rho, rho_p, tau, tau_p, bad_fracs


task = "QNLI"
layer_depths = [11]
list_ref_seeds = list(range(1, 11))
METRICS = ["Procrustes", "CKA", "PWCCA"]


rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
    full_df, task, layer_depths, list_ref_seeds, METRICS, layer_sub_df
)


plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, title=task)


avg_rho = {metric: round(np.mean(rho[metric]), 3) for metric in METRICS}
avg_rho_p = {metric: round(np.mean(rho_p[metric]), 3) for metric in METRICS}
avg_tau = {metric: round(np.mean(tau[metric]), 3) for metric in METRICS}
avg_tau_p = {metric: round(np.mean(tau_p[metric]), 3) for metric in METRICS}

with open(results_path, "a") as f:
    for metric in METRICS:
        ic("\n")
        ic(metric)
        ic(avg_rho[metric])
        ic(avg_rho_p[metric])
        ic(avg_tau[metric])
        ic(avg_tau_p[metric])

        f.write(f"metric: {metric}\n")
        f.write(f"avg_rho: {avg_rho[metric]}\n")
        f.write(f"avg_rho_p: {avg_rho_p[metric]}\n")
        f.write(f"avg_tau: {avg_tau[metric]}\n")
        f.write(f"avg_tau_p: {avg_tau_p[metric]}\n")
        f.write("\n")
