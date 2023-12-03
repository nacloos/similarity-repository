import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import similarity


np.random.seed(5)


def make_backend_df():
    # tODO: use api.json
    backend_cards = similarity.make(package="backend:backends", key="cards", use_cache=True)
    default_backend = similarity.make(package="backend:backends", key="default_backend", use_cache=True)

    measure_cards = similarity.make(package="measure:card", key="cards", use_cache=True)
    measure_names = list(measure_cards.keys())
    print("loaded")
    # backend_cards = similarity.make("backend.card")
    # measure_names = list(similarity.make("measure", return_config=True).keys())

    for name in measure_names:
        if name not in default_backend:
            default_backend[name] = "No implementation"

    # measure_names = list(default_backend.keys())
    # measure_names = similarity.make(package="backend:backends", key="measure_names", use_cache=False)

    backend_measures = {}
    for k, backend in backend_cards.items():
        backend_measures[k] = {}
        for measure in measure_names:
            if k == default_backend[measure]:
                backend_measures[k][measure] = 2
            elif measure in backend["measures"]:
                backend_measures[k][measure] = 1.5
            else:
                backend_measures[k][measure] = 0

    # backend_measures = {
    #     k: [measure in backend["measures"] for measure in all_measures]
    #     for k, backend in backend_cards.items()
    # }

    backend_df = pd.DataFrame.from_dict(backend_measures, orient="index", columns=measure_names)
    # print(backend_df)
    backend_df = backend_df[sorted(backend_df.columns)]
    return backend_df, backend_cards, measure_names


def plot_backend_measures(backend_df, backend_cards, measure_names, save_path=None):
    # plt.figure(figsize=(4, 3), dpi=100)
    plt.figure(figsize=(6+0.8*len(measure_names), 1+0.2*len(backend_cards)), dpi=100)
    ax = sns.heatmap(backend_df, annot=False, cmap="viridis", cbar=False, linewidths=0, linecolor='white')
    plt.ylabel("backend")
    # plt.xlabel("measures")
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top') 
    # plt.tick_params(length=0)  # Removing the small tick bars
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0, va='center')
    plt.axis('scaled')
    # plt.show()

    # Adding a categorical legend
    cmap = sns.color_palette("viridis", as_cmap=True)
    normalized_values = backend_df.values / backend_df.values.max()  # Normalize the values to the range [0, 1]
    unique_values = np.unique(normalized_values)  # Unique normalized values

    # Getting the corresponding colors from the colormap
    legend_colors = [cmap(value) for value in unique_values]

    # Creating legend elements with the correct colors
    labels = ['Not implemented', 'Implemented', 'Default implementation']
    legend_elements = [
        Patch(facecolor=legend_colors[i], label=labels[i])
        for i in range(len(unique_values))
    ]
    # Display the legend
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3,
        handlelength=1, handleheight=1, frameon=False)

    # TODO: make figure in github action on push
    # TODO: figures for dark and light modes (transparent=True)
    if save_path:
        plt.savefig(save_path, transparent=False, bbox_inches='tight', dpi=300)


def plot_backend_consistency(discard_variants=False, save_path=None):
    backend_by_measure = {
        k: v["backends"]
        for k, v in similarity.make("measure", return_config=True).items()
    }
    backend_names = similarity.make("backend", return_config=True).keys()
    measure_names = similarity.make("measure", return_config=True).keys()
    print(list(backend_names))

    X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

    backend_measures = {}
    for backend_name in backend_names:
        backend_measures[backend_name] = {}

        for measure_name in measure_names:
            print(backend_name, measure_name)
            if backend_name in backend_by_measure[measure_name]:
                measure = similarity.make(f"backend.{backend_name}.measure.{measure_name}")
                res = measure.fit_score(X, Y)
            else:
                res = None

            if discard_variants:
                measure_name = measure_name.split("-")[0]

            # if res is not None:
            #     print("new", measure_name)
            #     print("res", res)
            #     print("----------------------")
            if measure_name not in backend_measures[backend_name] or \
               backend_measures[backend_name][measure_name] is None:
                backend_measures[backend_name][measure_name] = res

    # print line by line
    print(json.dumps(backend_measures, indent=4))
    measure_names = list(backend_measures[backend_name].keys())

    backend_df = pd.DataFrame.from_dict(backend_measures, orient="index", columns=measure_names)
    backend_df = backend_df[sorted(backend_df.columns)]
    print(backend_df)

    # remove columns with None (no implementation)
    backend_df = backend_df.dropna(axis=1, how='all')

    plt.figure(figsize=(6+0.8*len(measure_names), 1+0.2*len(backend_names)), dpi=100)
    sns.heatmap(backend_df, vmin=0, vmax=0, annot_kws={"fontsize": 5}, annot=True, cmap="viridis", cbar=False, linewidths=1, linecolor='white')
    plt.ylabel("backend")
    # plt.xlabel("measures")
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0, va='center')
    plt.axis('scaled')

    if save_path:
        plt.savefig(save_path, transparent=False, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    # backend_df, backend_cards, measure_names = make_backend_df()
    # # order cols by alphabetical order
    # plot_backend_measures(backend_df, backend_cards, measure_names, save_path="./figures/backend_measures.png")

    plot_backend_consistency(save_path="./figures/backend_consistency.png")
    # plot_backend_consistency(save_path="./figures/backend_consistency_no_variants.png", discard_variants=True)
