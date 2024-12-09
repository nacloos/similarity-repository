import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

import pdb

BLACK = "k"
GREEN = "#59d98e"
RED = "#e74c3c"
BLUE = "#3498db"
PURPLE = "#9b59b6"
ORANGE = "#f39c12"
color = [BLACK, GREEN, RED, BLUE, PURPLE,ORANGE]

def tensor_dictionary_to_numpy(dict, batch, num, epoch):
    sim_matrix, distance_matrix = torch.empty(0).to('cuda'), torch.empty(0).to('cuda')
    for key, values in dict.items():
        sim_matrix = torch.cat((sim_matrix, values['sims'].unsqueeze(0)), dim=-1)
        distance_matrix = torch.cat((distance_matrix, values['dist'].unsqueeze(0)), dim=-1)
    sim_matrix = sim_matrix.view(int(epoch), -1)
    distance_matrix = distance_matrix.view(int(epoch), -1)

    sim_matrix = sim_matrix.detach().cpu().numpy()
    distance_matrix = distance_matrix.detach().cpu().numpy()
    # print(len(sim_matrix[0]))
    nomi = len(sim_matrix[0]) // batch // num
    numbers = []
    for i in range(nomi):
        if i == 0:
            continue
        else:
            number = list(range(i * batch * num, i * batch * num + batch))
            numbers.extend(number)
    sim_matrix = sim_matrix[:, numbers]
    distance_matrix = distance_matrix[:, numbers]
    return sim_matrix, distance_matrix

def save_CosineSimilarity_Distance(cosine, dist, path):
    mean_sim, mean_dist = np.mean(cosine, axis=-1), np.mean(dist, axis=-1)
    #normalize cos & dist for scaling
    #cosine, dist = normalize(cosine, axis=1), normalize(dist, axis=1)
    var_sim, var_dist = np.var(normalize(cosine, axis=1), axis=-1), np.var(normalize(dist, axis=1), axis=-1)
    #set x,y axis range
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 25.0)
    for i in range(len(cosine)):
        label = "Var_sim: %0.8f, Var_dis: %0.8f" % (var_sim[i], var_dist[i])
        plt.plot([cosine[i]], [dist[i]], marker="o", color=color[i])
        plt.plot([mean_sim[i]], [mean_dist[i]], marker="x", markersize=10, color=color[i], label=label)
    plt.legend()
    plt.savefig(path)