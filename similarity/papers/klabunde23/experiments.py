"""
Reference: https://github.com/mklabunde/survey_measures/blob/8a413cc1bb7de23b41527460999e665a40604604/appendix_procrustes.ipynb
Modifications:
* encapsulate the code in a function
* add an argument to specify where to save the plot
* add measure argument and move the procrustes implementation to a separate function
"""
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from tqdm import tqdm


def procrustes(a, b):    
    r, scale = orthogonal_procrustes(a, b)
    total_norm = (
        -2 * scale
        + np.linalg.norm(a, ord="fro") ** 2
        + np.linalg.norm(b, ord="fro") ** 2
    )
    return total_norm


def benchmark_dimensionality(measure, dimensionalities, noise_levels, N, D, n_repetitions, n_permutations, save_path):
    rng = np.random.default_rng(1234)

    results = {"D": [], "dist": [], "value": [], "noise": []}

    for i in dimensionalities:
        increased_D = i * D
        for noise_level in noise_levels:
            for repetition in tqdm(range(n_repetitions)):
                # One random matrix and one rotated version with added noise
                Q = ortho_group.rvs(increased_D, size=1)
                a = rng.standard_normal((N, increased_D))
                b = a @ Q + noise_level * rng.standard_normal((N, increased_D))

                # r, scale = orthogonal_procrustes(a, b)
                # total_norm = (
                #     -2 * scale
                #     + np.linalg.norm(a, ord="fro") ** 2
                #     + np.linalg.norm(b, ord="fro") ** 2
                # )
                total_norm = measure(a, b)
                
                results["D"].append(increased_D)
                results["dist"].append("ortho_proc")
                results["value"].append(np.sqrt(total_norm))
                results["noise"].append(noise_level)
                
                results["D"].append(increased_D)
                results["dist"].append("squared_ortho_proc")
                results["value"].append(total_norm)
                results["noise"].append(noise_level)
                
                # baseline value for unrelated matrices
                for _ in range(n_permutations):
                    a_shuffled = rng.permutation(a, axis=0)
                    shuffled_norm = (
                        -2 * orthogonal_procrustes(a_shuffled, b)[1]
                        + np.linalg.norm(a_shuffled, ord="fro") ** 2
                        + np.linalg.norm(b, ord="fro") ** 2
                    )
                    results["D"].append(increased_D)
                    results["dist"].append("shuffled")
                    results["value"].append(np.sqrt(shuffled_norm))
                    results["noise"].append(noise_level)
                    
                    results["D"].append(increased_D)
                    results["dist"].append("squared_shuffled")
                    results["value"].append(shuffled_norm)
                    results["noise"].append(noise_level)
        
    df = pd.DataFrame.from_dict(results)


    df["Representations"] = df["dist"]
    df["Noise SD"] = df["noise"]
    df.loc[df["Representations"] == "ortho_proc", "Representations"] = "Rotated + Noise"
    df.loc[df["Representations"] == "shuffled", "Representations"] = "Shuffled Baseline"


    g = sns.relplot(
        data=df[~df["dist"].isin(["squared_shuffled", "squared_ortho_proc"])],
        # data=df[~df["dist"].isin(["squared_shuffled", "squared_ortho_proc", "shuffled"])],
        # data=df,
        x="D",
        y="value",
        hue="Representations",
        style="Noise SD",
        kind="line",
        markers=True,
        height=4,
        aspect=1.3,
        errorbar="sd",
    )
    g.set(yscale="log")
    g.set(xscale="log")
    g.set(ylabel="Orthogonal Procrustes Distance")
    g.set(xlabel="Dimensionality D")

    # g.savefig("orthoproc_dim.pdf")
    from pathlib import Path
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path)
