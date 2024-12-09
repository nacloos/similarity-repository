from functools import partial
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

import similarity


def eval_naming_convention(
    measures: list,
    names: list,
    save_dir,
    color=None,
    fill_style=None,
    shape=(40, 20),
    num_samples=10,
    num_datasets=10,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    Xs, Ys = make_datasets(shape, num_samples, num_datasets)

    # compute scores
    scores = []
    for i, name in enumerate(names):
        _scores = []
        for X, Y in zip(Xs, Ys):
            score = measures[i](X, Y)
            if np.isnan(score):
                raise ValueError(f"nan score for {name}")
                # TODO: solve nan issue
                print(name, score)
                _scores.append(0)
            else:
                _scores.append(score)
        scores.append(np.array(_scores))
    score_arr = np.array(scores)
    print("len of scores", len(scores))

    # dist matrix ordered by metrics
    scores_by_id = defaultdict(list)
    labels_by_id = defaultdict(list)
    for i, k in enumerate(names):
        _, repo_name, measure_id = k.split("/")
        scores_by_id[measure_id].append(score_arr[i])
        labels_by_id[measure_id].append(k)
    scores_by_id_arr = np.concatenate([l for l in scores_by_id.values()])
    labels_by_id_arr = np.concatenate([l for l in labels_by_id.values()])
    dist_matrix_by_id = euclidean_distances(scores_by_id_arr)

    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(dist_matrix_by_id, cmap="viridis")
    plt.xticks(np.arange(len(labels_by_id_arr)), labels=labels_by_id_arr, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels_by_id_arr)), labels=labels_by_id_arr)
    plt.colorbar(label="Distance")
    plt.tight_layout()
    plt.savefig(save_dir / f"dissimilarity_matrix_by_id.png")

    # plot average distance within each metric name
    avg_dists = []
    for k, v in scores_by_id.items():
        _scores = np.array(v)
        _dist = euclidean_distances(_scores)
        avg_dists.append(np.mean(_dist))
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(list(scores_by_id.keys()), avg_dists, color=color, lw=2, marker=".", markerfacecolor=fill_style)
    ymax = None if np.max(avg_dists) > 0.1 else 0.1
    plt.ylim(-0.1, ymax)
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.xlabel("Measure")
    plt.ylabel("Average error")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.title(pretty_names[convention])
    plt.tight_layout()
    plt.savefig(save_dir / f"avg_dist_by_id.png")
    plt.savefig(save_dir / f"avg_dist_by_id.pdf")

    # plot number of implementations for each name
    impl_counts = [len(v) for v in scores_by_id.values()]
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(list(scores_by_id.keys()), impl_counts, color=color, lw=2, marker=".", markerfacecolor=fill_style)
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.xlabel("Measure")
    plt.ylabel("Number of Implementations")
    # plt.title(pretty_names[convention])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / f"impl_count_by_id.png")
    plt.savefig(save_dir / f"impl_count_by_id.pdf")

    # plot raw scores for one pair of datasets
    # dataset_to_plot = 4
    # plt.figure(figsize=(4, 3), dpi=300)
    # for k, scores in scores_by_id.items():
    #     _scores = [v[dataset_to_plot] for v in scores]
    #     print(k, len(_scores))
    #     plt.scatter([k]*len(_scores), _scores, color=color, marker=".")

    # plt.xticks(rotation=45, ha='right', fontsize=5)
    # plt.xlabel("Measure")
    # plt.ylabel("Output")
    # ax = plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig(save_dir / f"raw_scores_dataset_{dataset_to_plot}.png")

    # # plot raw scores for the first 10 datasets for all the measures corresponds to one name
    # def _plot_raw_scores(name_idx_to_plot, save_path):
    #     save_path.parent.mkdir(parents=True, exist_ok=True)

    #     name_scores = list(scores_by_id.values())[name_idx_to_plot]
    #     datasets_to_plot = slice(0, 10)

    #     plt.figure(figsize=(4, 3), dpi=300)
    #     scores_to_plot = []
    #     for measure_scores in name_scores:
    #         _scores = measure_scores[datasets_to_plot]
    #         plt.plot(_scores, color=color)
    #         scores_to_plot.append(_scores)
    #     plt.xlabel("Datasets")
    #     plt.ylabel(list(scores_by_id.keys())[name_idx_to_plot], fontsize=6)
    #     ax = plt.gca()
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     plt.tight_layout()
    #     plt.savefig(save_path)

    #     plt.figure(figsize=(4, 3), dpi=300)
    #     # plot mean and std
    #     mean_scores = np.mean(scores_to_plot, axis=0)
    #     std_scores = np.std(scores_to_plot, axis=0)
    #     plt.plot(mean_scores, color=color, lw=2)
    #     plt.fill_between(np.arange(len(mean_scores)), mean_scores-std_scores, mean_scores+std_scores, color=color, alpha=0.2)

    #     plt.xlabel("Datasets")
    #     plt.ylabel(list(scores_by_id.keys())[name_idx_to_plot], fontsize=8)
    #     ax = plt.gca()
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     plt.tight_layout()
    #     plt.savefig(save_path.parent / f"{save_path.stem}_mean.png")

    # for name_idx_to_plot in range(len(scores_by_id)):
    #     _plot_raw_scores(name_idx_to_plot, save_dir / "raw_scores" / f"name_{name_idx_to_plot}")

    plt.close("all")
    return impl_counts, avg_dists, scores_by_id



def make_datasets(shape, num_samples, num_ref_datasets=1):
    def make_gaussian_exp_dataset(shape):
        eigenvalues = np.logspace(np.log10(0.1), np.log10(10), shape[1])
        cov_matrix = np.diag(eigenvalues)
        X = np.random.multivariate_normal(mean=np.zeros(shape[1]), cov=cov_matrix, size=shape[0])
        return X

    Xs = []
    Ys = []
    for _ in range(num_ref_datasets):
        eta_values = np.linspace(0, 10, num_samples)
        noise = np.random.randn(*shape)
        X = make_gaussian_exp_dataset(shape)
        Xs.extend([X for _ in range(num_samples)])
        Ys.extend([X + eta * noise for eta in eta_values])
    
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    return Xs, Ys


if __name__ == "__main__":
    from similarity.plotting import plot_scores

    save_dir = Path(__file__).parent.parent / "figures" / Path(__file__).stem
    measures_dict = similarity.all_measures()


    repos_to_plot = [
        # "rsatoolbox",
        "representation_similarity",
        "nn_similarity_index",
        "netrep",
        # "sim_metric",
        "repsim",
        "platonic",
        # "neuroaimetrics",
        # "resi"
        # "diffscore"
    ]
    measures = similarity.all_measures()
    measures = {k: v for k, v in measures.items() if any(repo in k for repo in repos_to_plot)}

    # for all measures with parameter 'sigma={sigma}', create a new measure with 'sigma=1.0'
    params = {
        "sigma": [0.1, 1.0, 10.0, 100.0],
    }
    for k, v in list(measures.items()):
        for param, values in params.items():
            if f'{param}={{{param}}}' in k:
                for value in values:
                    new_k = k.replace(f'{param}={{{param}}}', f'{param}={value}')
                    measures[new_k] = partial(v, **{param: value})

    # filter measures that don't have parameters
    measures = {k: v for k, v in measures.items() if '{' not in k}

    # only cka-kernel=linear/rbf-score/distance=angular measures
    measures = {k: v for k, v in measures.items() if 'cka' in k}
    measures = {k: v for k, v in measures.items() if 'euclidean' not in k}
    measures = {k: v for k, v in measures.items() if 'laplace' not in k}


    plot_scores(measures, save_dir=save_dir)

    measure_functions = list(measures.values())
    names = list(measures.keys())
    impl_counts, avg_dists, scores_by_id = eval_naming_convention(measure_functions, names, save_dir)

    print("impl_counts", impl_counts)
    print("avg_dists", np.mean(avg_dists))

    measure_names = set([name.split("/")[-1] for name in names])
    print("number of measure names", len(measure_names))
