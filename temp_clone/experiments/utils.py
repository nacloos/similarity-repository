import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, percentileofscore

pc = percentileofscore
import numpy as np


def qs(xs):
    return np.array(list(map(lambda x: pc(xs, x, "rank") / 100, xs)))


def plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, scatter=False, title=""):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    fig.suptitle(title)

    if scatter:
        x, y = [], []
        for i, metric in enumerate(METRICS):
            x += len(rho[metric]) * [i]
            y += rho[metric]
        ax[0, 0].scatter(x, y)
        ax[0, 0].scatter(
            list(range(len(METRICS))), [np.mean(rho[metric]) for metric in METRICS]
        )
    else:
        ax[0, 0].bar(
            x=list(range(len(METRICS))),
            height=[np.mean(rho[metric]) for metric in METRICS],
        )
    ax[0, 0].set_title("Spearman's rho")
    ax[0, 0].set_xticks(list(range(len(METRICS))))
    ax[0, 0].set_xticklabels(METRICS)

    if scatter:
        x, y = [], []
        for i, metric in enumerate(METRICS):
            x += len(rho_p[metric]) * [i]
            y += rho_p[metric]
        ax[0, 1].scatter(x, y)
        ax[0, 1].scatter(
            list(range(len(METRICS))), [np.mean(rho_p[metric]) for metric in METRICS]
        )
    else:
        ax[0, 1].bar(
            x=list(range(len(METRICS))),
            height=[np.mean(rho_p[metric]) for metric in METRICS],
        )
    ax[0, 1].set_title("Spearman's rho: p-values")
    ax[0, 1].set_xticks(list(range(len(METRICS))))
    ax[0, 1].set_xticklabels(METRICS)
    ax[0, 1].set_yscale("log")

    if scatter:
        x, y = [], []
        for i, metric in enumerate(METRICS):
            x += len(tau[metric]) * [i]
            y += tau[metric]
        ax[1, 0].scatter(x, y)
        ax[1, 0].scatter(
            list(range(len(METRICS))), [np.mean(tau[metric]) for metric in METRICS]
        )
    else:
        ax[1, 0].bar(
            x=list(range(len(METRICS))),
            height=[np.mean(tau[metric]) for metric in METRICS],
        )
    ax[1, 0].set_title("Kendall's tau")
    ax[1, 0].set_xticks(list(range(len(METRICS))))
    ax[1, 0].set_xticklabels(METRICS)

    if scatter:
        x, y = [], []
        for i, metric in enumerate(METRICS):
            x += len(tau_p[metric]) * [i]
            y += tau_p[metric]
        ax[1, 1].scatter(x, y)
        ax[1, 1].scatter(
            list(range(len(METRICS))), [np.mean(tau_p[metric]) for metric in METRICS]
        )
    else:
        ax[1, 1].bar(
            x=list(range(len(METRICS))),
            height=[np.mean(tau_p[metric]) for metric in METRICS],
        )
    ax[1, 1].set_title("Kendall's tau: p-values")
    ax[1, 1].set_xticks(list(range(len(METRICS))))
    ax[1, 1].set_xticklabels(METRICS)
    ax[1, 1].set_yscale("log")

    plt.show()


def get_rank_corrs(sub_df, metric, task):
    plot_x = sub_df[metric]
    plot_y = sub_df[f"{task}_diff"]

    # spearman's rho and p-value
    rho = spearmanr(plot_x, plot_y)
    rho_corr = rho.correlation
    rho_os_p = (
        (rho.pvalue / 2) if rho_corr > 0 else (1 - rho.pvalue / 2)
    )  # one-sided p-value

    # kendall's tau and p-value
    tau = kendalltau(plot_x, plot_y)
    tau_corr = tau.correlation
    tau_os_p = (
        (tau.pvalue / 2) if tau_corr > 0 else (1 - tau.pvalue / 2)
    )  # one-sided p-value

    # bad_frac
    q_x = qs(plot_x)
    q_y = qs(plot_y)
    bad_frac = np.mean((q_x < 0.2) * (q_y > 0.8))

    return rho_corr, rho_os_p, tau_corr, tau_os_p, bad_frac


def aggregate_rank_corrs(
    full_df, task, num_layers, METRICS, sub_df_fn, list_layers=None
):
    if list_layers == None:
        list_layers = list(range(num_layers))

    rho = {metric: [] for metric in METRICS}
    rho_p = {metric: [] for metric in METRICS}
    tau = {metric: [] for metric in METRICS}
    tau_p = {metric: [] for metric in METRICS}
    bad_fracs = {metric: [] for metric in METRICS}

    for ref_depth in list_layers:
        sub_df = sub_df_fn(full_df, task, ref_depth)

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