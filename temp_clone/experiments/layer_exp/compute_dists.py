#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys

sys.path.append(os.path.abspath("../../"))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import *
from utils import *

# result file
result_filename = resources_path / pathlib.Path(
    "dists/layer_exp/dists_self_computed.csv"
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
seeds_list = range(1, 11)


if not os.path.exists(result_filename):
    os.mknod(result_filename)
else:
    os.remove(result_filename)
    os.mknod(result_filename)

for idx, seed1 in enumerate(seeds_list):
    for seed2 in seeds_list[idx:]:
        for layer1 in range(num_layers):
            for layer2 in range(num_layers):
                if seed1 == seed2 and layer1 > layer2:
                    pass
                else:
                    ic(seed1, seed2, layer1, layer2)
                    # define dictionaries containing info on each representation
                    rep1_dict = {}
                    rep1_dict["dataset"] = dataset
                    rep1_dict["architecture"] = "base"
                    rep1_dict["seed"] = seed1
                    rep1_dict["step"] = step
                    rep1_dict["layer"] = layer1

                    rep2_dict = {}
                    rep2_dict["dataset"] = dataset
                    rep2_dict["architecture"] = "base"
                    rep2_dict["seed"] = seed2
                    rep2_dict["step"] = step
                    rep2_dict["layer"] = layer2

                    score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)
