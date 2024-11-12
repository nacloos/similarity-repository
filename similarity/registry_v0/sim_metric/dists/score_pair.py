import numpy as np
import sys
import os
import csv
import pathlib
import logging

from scoring import *
from compute_embeddings import compute_embeddings

sys.path.append(os.path.abspath(".."))
from paths import resources_path


def get_embedding_folder(dataset, architecture, seed, step, layer):
    suffix = pathlib.Path(f"embeddings/{dataset}/{architecture}/{seed}/{step}/{layer}")
    return resources_path / suffix


def load_embedding(
    dataset: str,
    architecture: str,  # in practice this can be feather, base, or medium_finetuned
    seed: int,
    step: int,
    layer: int,
) -> np.ndarray:

    # path to look for embedding
    folder_path = get_embedding_folder(dataset, architecture, seed, step, layer)

    # check if representations have already been computed; otherwise compute them:
    if not os.path.exists(folder_path):
        print("Computing representations for model")
        os.makedirs(folder_path)
        rep = compute_embeddings(dataset, architecture, seed, step, layer)

    else:
        print("Representation already exists...loading...")
        rep = np.load(folder_path / pathlib.Path("rep.npy"))
    return rep


def score_pair_to_csv(
    rep1_dict: dict,
    rep2_dict: dict,
    filename: str,
    metrics: list,
) -> None:
    """
    Compute metric distance between two representations and save it to a csv file

    Args:
        rep1_dict (dict): dictionary specifying configuration of representation 1, to load its representation from disk
        rep2_dict (dict): dictionary specifying configuration of representation 2, to load its representation from disk
        filename (str): output filename to save results to
        metrics (list, optional): list of metrics to apply, eg CCA and/or CKA and/or GLD (by default all)
    """

    rep1 = load_embedding(
        rep1_dict["dataset"],
        rep1_dict["architecture"],
        rep1_dict["seed"],
        rep1_dict["step"],
        rep1_dict["layer"],
    )
    rep2 = load_embedding(
        rep2_dict["dataset"],
        rep2_dict["architecture"],
        rep2_dict["seed"],
        rep2_dict["step"],
        rep2_dict["layer"],
    )

    logging.info(f"representation 1 shape: {rep1.shape}")
    logging.info(f"representation 2 shape: {rep2.shape}")

    results = {
        "dataset1": rep1_dict["dataset"],
        "architecture1": rep1_dict["architecture"],
        "seed1": rep1_dict["seed"],
        "step1": rep1_dict["step"],
        "layer1": rep1_dict["layer"],
        "dataset2": rep2_dict["dataset"],
        "architecture2": rep2_dict["architecture"],
        "seed2": rep2_dict["seed"],
        "step2": rep2_dict["step"],
        "layer2": rep2_dict["layer"],
    }

    score_local_pair(
        rep1=rep1, rep2=rep2, metrics=metrics, filename=filename, metadata=results
    )


def score_local_pair(
    rep1: np.ndarray,
    rep2: np.ndarray,
    filename: str,
    metrics: list,
    metadata: dict = {},
) -> None:
    """
    Compute metric distances between two representations (in numpy array format) and
    save results to a csv file

    Args:
        rep1 (np.ndarray): representation 1 to compare
        rep2 (np.ndarray): representation 2 to compare
        filename (str): file name for output csv
        metrics (list, optional): list of metrics to apply (by default all)
        metadata (dict, optional): metadata for the representations to print to the csv (by default empty)
    """

    # center each row
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)

    # normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)

    results = metadata

    ## CCA like: first decompose, then compute metrics themselves
    if (
        "PWCCA" in metrics
        or "mean_sq_cca_corr" in metrics
        or "mean_cca_corr" in metrics
    ):
        logging.info("Computing CCA decomposition...")
        cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = cca_decomp(
            rep1, rep2
        )

        if "PWCCA" in metrics:
            logging.info("Computing PWCCA distance...")
            results["PWCCA"] = pwcca_dist(rep1, cca_rho, transformed_rep1)
        if "mean_sq_cca_corr" in metrics:
            logging.info("Computing mean square CCA corelation...")
            results["mean_sq_cca_corr"] = mean_sq_cca_corr(cca_rho)
        if "mean_cca_corr" in metrics:
            logging.info("Computing mean CCA corelation...")
            results["mean_cca_corr"] = mean_cca_corr(cca_rho)

    ## CKA like
    if "CKA" in metrics:
        logging.info("Computing Linear CKA dist...")
        lin_cka_sim = lin_cka_dist(rep1, rep2)
        results["CKA"] = lin_cka_sim

    if "CKA'" in metrics:
        logging.info("Computing Linear CKA' dist...")
        lin_cka_sim = lin_cka_prime_dist(rep1, rep2)
        results["CKA'"] = lin_cka_sim

    ## Procrustes
    if "Procrustes" in metrics:
        logging.info("Computing GLD dist...")
        results["Procrustes"] = procrustes(rep1, rep2)

    ## your metric here
    # function: my_metric_fn(rep1, rep2)
    # name: "my_new_metric"
    # if "my_new_metric" in metrics:
    #     logging.info("Computing my_new_metric dist...")
    #     results["my_new_metric"] = my_metric_fn(rep1, rep2)

    # Save results to file
    with open(filename, mode="a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)