import numpy as np
from matplotlib.patches import Patch

import similarity
from similarity import make


def test_measures():
    measures = make("measure.*.*")
    for measure_id, measure in measures.items():
        print(measure_id)
        X = np.random.randn(15, 20, 30)
        Y = np.random.randn(15, 20, 30)

        score = measure(X, Y)
        print("score: {score}")
        assert isinstance(score, float), f"Expected float, but got {type(score)}"


def backend_consistency(plot_paper_id=True, plot_values=True, save_path=None):
    from collections import defaultdict
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    measures = make("measure.*.*")

    X = np.random.randn(15, 10, 40)
    Y = np.random.randn(15, 10, 40)

    results = defaultdict(list)
    for measure_id, measure in measures.items():
        print(measure_id)
        backend = measure_id.split(".")[1]
        measure_name = measure_id.split(".")[2]

        score = measure(X, Y)

        results["backend"].append(backend)
        results["measure"].append(measure_name)
        results["score"].append(score)

    # convert to 2d dataframe (backend x measure)
    backend_df = pd.DataFrame(results).pivot(index="backend", columns="measure", values="score")
    print(backend_df)
    measure_names = list(backend_df.columns)
    backend_names = list(backend_df.index)

    if plot_paper_id:
        y_labels = []
        for backend in backend_names:
            if similarity.is_registered(f"measure.{backend}"):
                cfg = make(f"measure.{backend}")
                assert isinstance(cfg, dict), f"Expected dict for id 'measure.{backend}', but got {type(cfg)}"
                paper_id = cfg.get("paper_id", backend)
            else:
                paper_id = backend

            y_labels.append(paper_id)

        # rename backend_df
        backend_df.index = y_labels
        # extract date from paper_id (e.g. "williams2021" -> "2021")
        date = [int(paper_id[-4:]) if paper_id[-4:].isdigit() else 0 for paper_id in y_labels]
        # sort by date
        y_labels = [y for _, y in sorted(zip(date, y_labels), reverse=True)]
        # sort backend_df by date
        backend_df = backend_df.loc[y_labels]
    else:
        y_labels = backend_names

    plt.figure(figsize=(6+0.8*len(measure_names), 1+0.2*len(backend_names)), dpi=100)
    if plot_values:
        sns.heatmap(backend_df, vmin=-1000, vmax=1, annot_kws={"fontsize": 5, "color": "black"}, fmt='.2g', annot=True, cmap="viridis", cbar=False, linewidths=1, linecolor='white')
    else:
        # 1 if implemented, 0 if nan
        implemented = ~backend_df.isna()
        ax = sns.heatmap(implemented, vmin=0, vmax=1, cmap="viridis", cbar=False, linewidths=0, linecolor='white')

        # plot legend
        cmap = ax.collections[0].cmap
        legend_colors = [cmap(0.), cmap(1.)]
        # labels = ['Not implemented', 'Implemented', 'Default implementation']
        labels = ['Not implemented', 'Implemented']
        legend_elements = [
            Patch(facecolor=legend_colors[i], label=labels[i])
            for i in range(len(legend_colors))
        ]
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_elements),
                   handlelength=1, handleheight=1, frameon=False, fontsize=12)

    plt.ylabel("Backends", fontsize=15)
    plt.xlabel("Measures", fontsize=15)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0, va='center')
    plt.axis('scaled')
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, transparent=False, bbox_inches='tight', dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # test_measures()
    backend_consistency(plot_paper_id=False, plot_values=False, save_path="implemented_measures.png")
    # test_transforms()
