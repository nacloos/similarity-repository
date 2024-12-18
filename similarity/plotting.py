from functools import partial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd

import similarity


def plot_measures(measures: dict[str, callable], derived_measures: dict[str, callable] = None, save_dir: str = None):
    derived_measures = derived_measures or {}

    # keep only valid measures
    measures = {k: v for k, v in measures.items() if len(k.split("/")) == 3 and k.split("/")[0] == "measure"}
    derived_measures = {k: v for k, v in derived_measures.items() if len(k.split("/")) == 3 and k.split("/")[0] == "measure"}

    all_measures = {**measures, **derived_measures}

    measure_names = []
    repo_names = []
    # for k in measures.keys():
    for k in all_measures.keys():
        obj_type, repo_name, measure_name = k.split("/")
        assert obj_type == "measure", f"Expected measure, got: {obj_type}"
        if measure_name not in measure_names:
            measure_names.append(measure_name)
        if repo_name not in repo_names:
            repo_names.append(repo_name)

    measure_names = sorted(np.unique(measure_names))
    
    # TODO?: keep only derived measures that are in the measures dict
    # derived_measures = {k: v for k, v in derived_measures.items() if len(k.split("/")) == 3 and k.split("/")[0] == "measure" and k.split("/")[2] in measure_names}
    papers = similarity.all_papers()
    repo_paper_names = {k.split("/")[1]: v["shortcite"] for k, v in papers.items()}

    # sort repo names by year of paper
    repo_years = [int(repo_paper_names.get(repo, repo).split(",")[-1].strip().strip(")")) for repo in repo_names]
    repo_names = [repo for _, repo in sorted(zip(repo_years, repo_names), key=lambda x: x[0])]
    # repo_names = sorted(repo_names, key=lambda x: int(repo_paper_names.get(x, x).split(",")[-1].strip().strip(")")))
    repo_names = list(reversed(repo_names))

    indices = []
    derived_indices = []
    for k in measures.keys():
        obj_type, repo_name, measure_name = k.split("/")
        assert obj_type == "measure", f"Expected measure, got: {obj_type}"
        assert measure_name in measure_names, f"Measure name {measure_name} not in {measure_names}"
        assert repo_name in repo_names, f"Repo name {repo_name} not in {repo_names}"
        x_idx = measure_names.index(measure_name)
        y_idx = repo_names.index(repo_name)
        indices.append((x_idx, y_idx))

    for k in derived_measures.keys():
        obj_type, repo_name, measure_name = k.split("/")
        assert obj_type == "measure", f"Expected measure, got: {obj_type}"
        assert measure_name in measure_names, f"Measure name {measure_name} not in {measure_names}"
        assert repo_name in repo_names, f"Repo name {repo_name} not in {repo_names}"
        x_idx = measure_names.index(measure_name)
        y_idx = repo_names.index(repo_name)
        derived_indices.append((x_idx, y_idx))
    
    # Modify ylabels to make the paper name bold using LaTeX syntax
    # ylabels = [f"{k.split('/')[1]} {repo_paper_names[k.split('/')[1]]}" for k in repo_names]
    
    # add paper names "{repo} ({paper_name})" to the index
    ylabels = [f"{repo} ({repo_paper_names.get(repo, repo)})" for repo in repo_names]

    color_registered = "#669bbc"
    color_derived = "#B7CFDE"

    # color_registered = "#ff7f0e"  # orange
    # color_derived = "#ffbb78"  # light orange


    # scatter plot
    # plt.figure(figsize=(20, 7), dpi=100)
    # plt.figure(figsize=(25, 13), dpi=100)
    plt.figure(figsize=(35, 20), dpi=100)
    for i, (x, y) in enumerate(indices):
        plt.scatter(x, y, c=color_registered, marker="s", s=100, edgecolors='white', linewidth=0.5)
    for i, (x, y) in enumerate(derived_indices):
        plt.scatter(x, y, c=color_derived, marker="s", s=100, edgecolors='white', linewidth=0.5)
    
    plt.xticks(range(len(measure_names)), measure_names, rotation=45, ha='left', fontsize=8)
    plt.yticks(range(len(repo_names)), ylabels)
    plt.xlabel(f'Measures ({len(measure_names)})', fontsize=15, fontweight='bold')
    plt.ylabel(f'Repositories ({len(repo_names)})', fontsize=15, fontweight='bold')
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



def plot_scores(measures, X=None, Y=None, data_shape=(30, 25), figsize=(30, 8), save_dir=None):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    if X is None or Y is None:
        # Sample data
        # X, Y = np.random.randn(*data_shape), np.random.randn(*data_shape)
        X, Y = np.random.uniform(size=data_shape), np.random.uniform(size=data_shape)

    scores = {}
    for k, measure in measures.items():
        print(k)
        score = measure(X, Y)
        assert isinstance(score, float), f"Expected type float, got: {type(score)}"
        scores[k] = score
        print(k, scores[k])

    if '/' in list(measures.keys())[0]:
        repo_names = np.unique([k.split("/")[1] for k in measures.keys()])
        measure_names = np.unique([k.split("/")[2] for k in measures.keys()])
    else:
        repo_names = np.unique([k.split(".")[1] for k in measures.keys()])
        measure_names = np.unique([k.split(".")[2] for k in measures.keys()])

    # Sort alphabetically
    # repo_names = sorted(repo_names)
    measure_names = sorted(measure_names)

    # sort repo names by year of paper
    papers = similarity.all_papers()
    repo_paper_names = {k.split("/")[1]: v["shortcite"] for k, v in papers.items()}
    repo_names = sorted(repo_names, key=lambda x: int(repo_paper_names.get(x, x).split(",")[-1].strip().strip(")")))
    # repo_names = list(reversed(repo_names))


    # Create a DataFrame for the scores
    df = pd.DataFrame(index=repo_names, columns=measure_names, dtype=float)
    for k, score in scores.items():
        print(k)
        if '/' in k:
            _, repo, measure = k.split("/")
        else:
            repo = k.split(".")[1]
            measure = k.split(".")[2]

        df.loc[repo, measure] = float(score)

    if save_dir is not None:
        # save df to csv
        df.to_csv(save_dir / "metric_vs_repo.csv")

    # add paper names "{repo} ({paper_name})" to the index
    df.index = [f"{repo} ({repo_paper_names.get(repo, repo)})" for repo in df.index]

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
    else:
        plt.show()



if __name__ == "__main__":
    from similarity.plotting import plot_scores, plot_measures

    np.random.seed(0)
    repos_to_plot = None
    save_dir = Path(__file__).parent.parent / "figures"


    measures = similarity.all_measures()

    if repos_to_plot is not None:
        measures = {k: v for k, v in measures.items() if any(repo in k for repo in repos_to_plot)}

    # original = measures - derived
    original_measures = {k: v for k, v in measures.items() if k not in similarity.registration.DERIVED_MEASURES}
    derived_measures = similarity.registration.DERIVED_MEASURES

    if repos_to_plot is not None:
        original_measures = {k: v for k, v in original_measures.items() if any(repo in k for repo in repos_to_plot)}
        derived_measures = {k: v for k, v in derived_measures.items() if any(repo in k for repo in repos_to_plot)}


    plot_measures(original_measures, derived_measures=derived_measures, save_dir=save_dir)


    # for all measures with parameter 'sigma={sigma}', create a new measure with 'sigma=1.0'
    for k, v in list(measures.items()):
        if 'sigma={sigma}' in k:
            new_k = k.replace('sigma={sigma}', 'sigma=1.0')
            measures[new_k] = partial(v, sigma=1.0)

    # filter measures that don't have parameters
    measures = {k: v for k, v in measures.items() if '{' not in k}

    plot_scores(measures, save_dir=save_dir)
