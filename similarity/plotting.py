from functools import partial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd

import similarity

repo_paper_names = {
    "platonic": "(Huh et al., 2024)",
    "repsim": "(Lange et al., 2023)",
    "netrep": "(Williams et al., 2021)",
    "sim_metric": "(Ding et al., 2021)",
    "nn_similarity_index": "(Tang et al., 2020)",
    "representation_similarity": "(Kornblith et al., 2019)",
    "neuroaimetrics": "(Soni et al., 2024)",
    "rsatoolbox": "(Kriegeskorte et al., 2008)",
}


def plot_measures(measures: dict[str, callable], derived_measures: dict[str, callable] = None, save_dir: str = None):
    derived_measures = derived_measures or {}
    
    measure_names = []
    repo_names = []
    for k in measures.keys():
        repo_name, measure_name = k.split("/")
        if measure_name not in measure_names:
            measure_names.append(measure_name)
        if repo_name not in repo_names:
            repo_names.append(repo_name)

    measure_names = sorted(np.unique(measure_names))
    
    # sort repo names by year of paper
    repo_names = sorted(repo_names, key=lambda x: int(repo_paper_names[x].split(",")[-1].strip().strip(")")))
    repo_names = list(reversed(repo_names))

    indices = []
    derived_indices = []
    for k in measures.keys():
        repo_name, measure_name = k.split("/")
        assert measure_name in measure_names, f"Measure name {measure_name} not in {measure_names}"
        assert repo_name in repo_names, f"Repo name {repo_name} not in {repo_names}"
        x_idx = measure_names.index(measure_name)
        y_idx = repo_names.index(repo_name)
        indices.append((x_idx, y_idx))

    for k in derived_measures.keys():
        repo_name, measure_name = k.split("/")
        assert measure_name in measure_names, f"Measure name {measure_name} not in {measure_names}"
        assert repo_name in repo_names, f"Repo name {repo_name} not in {repo_names}"
        x_idx = measure_names.index(measure_name)
        y_idx = repo_names.index(repo_name)
        derived_indices.append((x_idx, y_idx))
    
    # Modify ylabels to make the paper name bold using LaTeX syntax
    ylabels = [f"{k.split('/')[0]} {repo_paper_names[k.split('/')[0]]}" for k in repo_names]

    color_registered = "#669bbc"
    color_derived = "#B7CFDE"

    # scatter plot
    plt.figure(figsize=(20, 7), dpi=100)
    for i, (x, y) in enumerate(indices):
        plt.scatter(x, y, c=color_registered, marker="s", s=100, edgecolors='white', linewidth=0.5)
    for i, (x, y) in enumerate(derived_indices):
        plt.scatter(x, y, c=color_derived, marker="s", s=100, edgecolors='white', linewidth=0.5)
    
    plt.xticks(range(len(measure_names)), measure_names, rotation=45, ha='left', fontsize=8)
    plt.yticks(range(len(repo_names)), ylabels)
    plt.xlabel('Measures', fontsize=12, fontweight='bold')
    plt.ylabel('Repositories', fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # x ticks at the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # Set x-axis limits to start at the first index and end at the last index
    ax.set_xlim(-0.5, len(measure_names) - 0.5)
    ax.set_ylim(-0.5, len(repo_names) - 0.5)
    # aspect
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(save_dir / "measures.pdf", bbox_inches='tight')
        plt.savefig(save_dir / "measures.png", bbox_inches='tight')
    else:
        plt.show()



def plot_scores(measures, X=None, Y=None, data_shape=(50, 30), figsize=(30, 8), save_dir=None):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    if X is None or Y is None:
        # Sample data
        X, Y = np.random.randn(*data_shape), np.random.randn(*data_shape)


    scores = {}
    for k, measure in measures.items():
        print(k)
        scores[k] = measure(X, Y)
        print(k, scores[k])

    if '/' in list(measures.keys())[0]:
        repo_names = np.unique([k.split("/")[1] for k in measures.keys()])
        measure_names = np.unique([k.split("/")[2] for k in measures.keys()])
    else:
        repo_names = np.unique([k.split(".")[1] for k in measures.keys()])
        measure_names = np.unique([k.split(".")[2] for k in measures.keys()])

    # Sort alphabetically
    repo_names = sorted(repo_names)
    measure_names = sorted(measure_names)

    # Create a DataFrame for the scores
    df = pd.DataFrame(index=repo_names, columns=measure_names, dtype=float)
    for k, score in scores.items():
        if '/' in k:
            _, repo, measure = k.split("/")
        else:
            repo = k.split(".")[1]
            measure = k.split(".")[2]

        df.loc[repo, measure] = float(score)

    # print(df)
    # breakpoint()

    # save df to csv
    df.to_csv(save_dir / "metric_vs_repo.csv")

    # add paper names "{repo} ({paper_name})" to the index
    df.index = [f"{repo} {repo_paper_names.get(repo, '')}" for repo in df.index]

    # Plot heatmap
    plt.figure(figsize=figsize, dpi=300)
    sns.heatmap(df, vmin=-1000, vmax=1, annot_kws={"fontsize": 5, "color": "black"}, fmt='.2g', annot=True, cmap="viridis", cbar=False, linewidths=1, linecolor='white')
    plt.ylabel("Repositories", fontsize=10)
    plt.xlabel("Measures", fontsize=10)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0, va='center')
    plt.axis('scaled')
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "metric_vs_repo_heatmap.png")
        plt.savefig(save_dir / "metric_vs_repo_heatmap.pdf")

    plt.close('all')


if __name__ == "__main__":
    measures = similarity.all_measures()

    save_dir = Path(__file__).parent.parent / "figures" / Path(__file__).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    # plot_measures(measures, derived_measures=measures, save_dir=save_dir)
    plot_scores(measures, save_dir=save_dir)
