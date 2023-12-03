# Grounding Representation Similarity with Statistical Testing
This repo contains code to replicate the results in [our paper](https://arxiv.org/abs/2108.01661), which evaluates representation similarity measures with a series of benchmark tasks. The experiments in the paper require first computing neural network embeddings of a dataset and computing accuracy scores of that neural network, which we provide pre-computed. This repo contains the code that implements our benchmark evaluation, given these embeddings and performance scores.

## File descriptions

### This repo: `sim_metric`
This repo is organized as follows:
* `experiments/` contains code to run the experiments in part 4 of the paper:
    - `layer_exp` is the first experiment in part 4, with different random seeds and layer depths
    - `pca_deletion` is the second experiment in part 4, with different numbers of principal components deleted
    - `feather` is the first experiment in part 4.1, with different finetuning seeds
    - `pretrain_finetune` is the second experiment in part 4.2, with different pretraining and finetuning seeds
* `dists/` contains functions to compute dissimilarities between representations.

### Pre-computed resources: `sim_metric_resources`
The pre-computed embeddings and scores available at <https://zenodo.org/record/5117844> can be downloaded and unzipped into a folder titled `sim_metric_resources`, which is organized as follows:
* `embeddings` contains the embeddings between which we are computing dissimilarities
* `dists` contains, for every experiment, the dissimilarities between the corresponding embeddings, for every metric:
    - `dists.csv` contains the precomputed dissimilarities
    - `dists_self_computed.csv` contains the dissimilarities computed by running `compute_dists.py` (see below)
* `scores` contains, for every experiment, the accuracy scores of the embeddings
* `full_dfs` contains, for every experiment, a csv file aggregating the dissimilarities and accuracy differences between the embeddings


## Instructions

* clone this repository
* go to <https://zenodo.org/record/5117844> and download `sim_metric_resources.tar`
* untar it with `tar -xvf sim_metric_resources sim_metric_resources.tar`
* in `sim_metric/paths.py`, modify the path to `sim_metric_resources`

### Replicating the results

For every experiment (eg `feather`, `pretrain_finetune`, `layer_exp`, or `pca_deletion`):
* the relevant dissimilarities and accuracies differences have already been precomputed and aggregated in a dataframe `full_df`
* make sure that `dists_path` and `full_df_path` in `compute_full_df.py`, `script.py` and `notebook.ipynb` are set to `dists.csv` and `full_df.csv`, and not `dists_self_computed.csv` and `full_df_self_computed.csv`.
* to get the results, you can:
    - run the notebook `notebook.ipynb`, or
    - run `script.py` in the experiment's folder, and find the results in `results.txt`, in the same folder
To run the scripts for all four experiments, run `experiments/script.py`.

### Recomputing dissimilarities

For every experiment, you can:
* recompute the dissimilarities between embeddings by running `compute_dists.py` in this experiment's folder
* use these and the accuracy scores to recompute the aggregate dataframe by running `compute_full_df.py` in this experiment's folder
* change `dists_path` and `full_df_path` in `compute_full_df.py`, `script.py` and `notebook.ipynb` from `dists.csv` and `full_df.csv` to `dists_self_computed.csv` and `full_df_self_computed.csv`
* run the experiments with `script.py` or `notebook.ipynb` as above.

## Adding a new metric
This repo also allows you to test a new representational similarity metric and see how it compares according to our benchmark. To add a new metric:
* add the corresponding function at the end of `dists/scoring.py`
* add a condition in `dists/score_pair.py`, around line 160
* for every experiment in `experiments`, add the name of the metric to the `metrics` list in `compute_dists.py`
