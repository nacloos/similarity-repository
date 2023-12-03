#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import *
from utils import *

# result file
result_filename = resources_path / pathlib.Path("dists/feather/dists_self_computed.csv")

if not os.path.exists(result_filename):
    os.mknod(result_filename)
else:
    os.remove(result_filename)
    os.mknod(result_filename)

# constants
metrics = ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"]
dataset = "mnli_matched_100"
architecture = "feather"
step = 0
num_layers = 12
num_seeds = 100


for seed1 in range(num_seeds):
    for seed2 in range(seed1, num_seeds):
        for layer in range(num_layers):
            ic(seed1, seed2, layer)
            # define dictionaries containing info on each representation
            rep1_dict = {}
            rep1_dict["dataset"] = dataset
            rep1_dict["architecture"] = architecture
            rep1_dict["seed"] = seed1
            rep1_dict["step"] = step
            rep1_dict["layer"] = layer

            rep2_dict = {}
            rep2_dict["dataset"] = dataset
            rep2_dict["architecture"] = architecture
            rep2_dict["seed"] = seed2
            rep2_dict["step"] = step
            rep2_dict["layer"] = layer

            score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)
