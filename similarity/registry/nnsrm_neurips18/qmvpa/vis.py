import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.3)


def show_heatmap(heatmap, k):
    # compute a mask
    mask = np.zeros(np.shape(heatmap))
    mask[np.triu_indices_from(mask, k=k)] = True
    # manage tick frequency
    tick_frequency = 10
    n_egs = np.shape(heatmap)[0]
    n_ticks = n_egs//tick_frequency
    # plot it
    my_heatmap = sns.heatmap(heatmap,
                             cmap="viridis",
                             xticklabels=n_ticks,
                             yticklabels=n_ticks,
                             mask=mask,
                             square=True,
                             cbar=True)
    return my_heatmap
