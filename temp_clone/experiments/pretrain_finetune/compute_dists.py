#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys
import itertools

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import *
from utils import *

# result file
result_filename = resources_path / pathlib.Path(
    "dists/pretrain_finetune/dists_self_computed.csv"
)

if not os.path.exists(result_filename):
    os.mknod(result_filename)
else:
    os.remove(result_filename)
    os.mknod(result_filename)

# constants
metrics = ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"]
dataset = "mnli_matched_100"
architecture = "medium_finetuned"
num_layers = 8
num_seeds = 10
num_fseeds = 10

all_model_seeds = list(
    itertools.product(range(1, 1 + num_seeds), range(1, 1 + num_fseeds))
)

for idx, (pseed1, fseed1) in enumerate(all_model_seeds):
    for (pseed2, fseed2) in all_model_seeds[idx:]:
        for layer in range(num_layers):
            ic(pseed1, fseed1, pseed2, fseed2, layer)
            # define dictionaries containing info on each representation
            # here we use the step attribute to refer to the fine tuning seed
            rep1_dict = {}
            rep1_dict["dataset"] = dataset
            rep1_dict["architecture"] = architecture
            rep1_dict["seed"] = pseed1
            rep1_dict["step"] = fseed1
            rep1_dict["layer"] = layer

            rep2_dict = {}
            rep2_dict["dataset"] = dataset
            rep2_dict["architecture"] = architecture
            rep2_dict["seed"] = pseed2
            rep2_dict["step"] = fseed2
            rep2_dict["layer"] = layer

            score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)
