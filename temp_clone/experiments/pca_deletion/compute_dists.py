#!/usr/local/linux/anaconda3.8/bin/python

# for every layer depth `layer`
# for every number of deleted dimensions `dims_deleted`
# for every pair of seeds `seed1` and `seed2`
# let `rep1` be the representation of the layer of depth `layer` in the model with seed `seed1`
# let `rep2` be the representation of the layer of depth `layer` in the model with seed `seed2`
# delete `dims_deleted` principal components from `rep2` to get `deleted_rep`
# compute the distances between `rep1` and `deleted_rep`

import numpy as np
import os
import pathlib
import sys

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import *

# result file
result_filename = resources_path / pathlib.Path(
    "dists/pca_deletion/dists_self_computed.csv"
)

if not os.path.exists(result_filename):
    os.mknod(result_filename)
else:
    os.remove(result_filename)
    os.mknod(result_filename)

# constants
metrics = ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"]
dataset = "mnli_matched_100"
architecture = "base"
step = 2000000
num_layers = 12
num_seeds = 10

dims_deleted = np.array(
    [0, 100, 200, 300, 400, 500, 600, 650, 700, 725, 750, 758, 763, 767]
)
TOTAL_DIM = 768
dims_kept = TOTAL_DIM - dims_deleted


# this is new
for seed1 in range(1, num_seeds + 1):
    for seed2 in range(1, num_seeds + 1):
        for layer in range(num_layers):
            rep1 = load_embedding(dataset, architecture, seed1, step, layer)
            rep1 = rep1 - rep1.mean(axis=1, keepdims=True)

            rep2 = load_embedding(dataset, architecture, seed2, step, layer)
            rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
            u, s, vh = np.linalg.svd(rep2, full_matrices=False)  # SVD of rep2, not rep1

            for dim in dims_deleted:
                dims_kept = TOTAL_DIM - dim
                deleted_rep = u[:, :dims_kept].T @ rep2
                deleted_rep = deleted_rep * 768 / dims_kept

                results = {
                    "dataset1": dataset,
                    "architecture1": architecture,
                    "seed1": seed1,
                    "step1": step,
                    "layer1": layer,
                    "dataset2": dataset,
                    "architecture2": architecture,
                    "seed2": seed2,
                    "step2": step,
                    "layer2": layer,
                    "dims_deleted": dim,
                    "dims_kept": TOTAL_DIM - dim,
                }

                score_local_pair(
                    rep1=rep1,
                    rep2=deleted_rep,
                    metadata=results,
                    metrics=metrics,
                    filename=result_filename,
                )
