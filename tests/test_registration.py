from pathlib import Path
import numpy as np
from matplotlib.patches import Patch

import similarity
from similarity import make
from similarity.transforms import DERIVED_MEASURES

print(len(DERIVED_MEASURES))
print(DERIVED_MEASURES)

def test_measures():
    measures = make("measure.*.*")
    for measure_id, measure in measures.items():
        print(measure_id)
        X = np.random.randn(15, 20, 30)
        Y = np.random.randn(15, 20, 30)

        score = measure(X, Y)
        print(f"score: {score}")


def backend_consistency(plot_paper_id=True, plot_values=True, save_path=None):
    from collections import defaultdict
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    measures = make("measure.*.*")

    # TODO: evaluate on multiple datasets and cluster measures with similar scores together (color by cluster)
    # X = np.random.randn(15, 10, 40)
    # Y = np.random.randn(15, 10, 40)
    # test with non-zero mean data
    X = np.random.rand(15, 10, 40)
    Y = np.random.rand(15, 10, 40)

    results = defaultdict(list)
    for measure_id, measure in measures.items():
        print(measure_id)
        backend = measure_id.split(".")[1]
        measure_name = measure_id.split(".")[2]

        score = measure(X, Y)

        results["backend"].append(backend)
        results["measure"].append(measure_name)
        results["score"].append(score)
        results["derived"].append(measure_id in DERIVED_MEASURES)

    # TODO: different color for derived

    # convert to 2d dataframe (backend x measure)
    backend_df = pd.DataFrame(results).pivot(index="backend", columns="measure", values="score")
    # convert torch tensor to float
    import torch
    backend_df = backend_df.applymap(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
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
            
            print(paper_id)
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


def test_cards():
    scoring_measures = {
        k: v for k, v in make("measure.*.*").items() if "score" in make(f"card.{k.split('.')[-1]}")["props"]
    }
    metrics = {
        k: v for k, v in make("measure.*.*").items() if "metric" in make(f"card.{k.split('.')[-1]}")["props"]
    }


if __name__ == "__main__":
    save_dir = Path(__file__).parent / ".." / "figures"
    backend_consistency(plot_paper_id=False, plot_values=True, save_path=save_dir / "backend_consistency.png")
    backend_consistency(plot_paper_id=False, plot_values=False, save_path=save_dir / "implemented_measures.png")

    # TODO
    # backend_consistency(plot_paper_id=True, plot_values=True, save_path=save_dir / "backend_consistency_by_paper.png")

    test_measures()
    test_cards()
