import os
import json
import numpy as np

from embedding_utils import *

# TODO: this does not work yet, fix all the paths
EMBEDDING_PATH = "/scratch/users/repsim/embeddings"  # deprecated
# TODO: update these paths
DATA_PATH = "/data/js_denain/"
SCRATCH_PATH = "/scratch/users/repsim/"

# Language paths
BERT_PATH = "/accounts/campus/frances/Documents/bert/"
CODE_PATH = "/accounts/projects/jsteinhardt/repsim/similarity_metric/"
BERT_CHECKPOINT_PATH = "/scratch/users/repsim/checkpoints/bert/"

PTB_PATH = "/accounts/campus/frances/Documents/probing/example/data/raw.ptb3.dev.txt"
MNLI_MISMATCHED_PATH = (
    "{}/data/mnli/multinli_1.0_dev_mismatched_fd_raw_tenth.txt".format(CODE_PATH)
)
MNLI_MATCHED_PATH = "{}/data/mnli/multinli_1.0_dev_matched_fd_raw_tenth.txt".format(
    CODE_PATH
)
MNLI_MATCHED_100_PATH = (
    "{}/data/mnli/multinli_1.0_dev_matched_fd_raw_hundredth.txt".format(CODE_PATH)
)
HANS_PATH = "{}/data/hans/heuristics_evaluation_set_fd_raw_tenth.txt".format(CODE_PATH)
HANS_100_PATH = "{}/data/hans/heuristics_evaluation_set_fd_raw_hundredth.txt".format(
    CODE_PATH
)
BERT_BASE_DIR = "/data/frances/ruiqi-bert/feather/uncased_L-12_H-768_A-12/"


def compute_embeddings(
    dataset: str,
    architecture: str,
    seed: int,
    step: int,
    layer: int,
) -> np.ndarray:
    """
    Compute the representations of a layer specified by the arguments and save to a npy file

    :param dataset: Dataset to compute embeddings for
    :param architecture: Model weights to load
    :param seed: Random seed used in the model pretraining
    :param step: Checkpoint during pretraining to use
    :param layer: Layer of the model to load
    :return: embedding (i.e. representation) just computed
    """

    assert dataset in [
        "ptb_dev",
        "mnli_matched",
        "mnli_matched_100",
        "mnli_mismatched",
        "hans_evaluation",
        "hans_evaluation_100",
    ]
    if dataset == "ptb_dev":
        datapath = PTB_PATH
    elif dataset == "mnli_matched":
        datapath = MNLI_MATCHED_PATH
    elif dataset == "mnli_matched_100":
        datapath = MNLI_MATCHED_100_PATH
    elif dataset == "mnli_mismatched":
        datapath = MNLI_MISMATCHED_PATH
    elif dataset == "hans_evaluation":
        datapath = HANS_PATH
    elif dataset == "hans_evaluation_100":
        datapath = HANS_100_PATH
    else:
        datapath = None

    if architecture == "feather":  # BERTS of a feather models need a different config
        bertnumber = str(seed)
        if seed < 10:
            bertnumber = "0" + bertnumber
        model_path = "{head}/feather/bert_{number}".format(
            head=BERT_CHECKPOINT_PATH, number=bertnumber
        )

        output_path = get_embedding_folder(dataset, architecture, seed, step, layer)
        json_output = output_path / pathlib.Path("rep.json")
        npy_output = output_path / pathlib.Path("rep.npy")

        command_outline = (
            "python extract_features.py --input_file={data}"
            " --output_file={output}"
            " --vocab_file={bertbase}/vocab.txt"
            " --bert_config_file={bertbase}/bert_config.json"
            " --init_checkpoint={model}/model.ckpt-36815"
            " --layers={layer}"
            " --max_seq_length=128"
            " --batch_size=8"
        )
        command = command_outline.format(
            data=datapath,
            output=str(json_output),
            bertbase=BERT_BASE_DIR,
            model=model_path,
            layer=layer,
        )
    else:
        # First get json file
        model_path = "{head}/{architecture}/pretrain_seed{seed}step{step}".format(
            head=EMBEDDING_PATH, architecture=architecture, seed=seed, step=step
        )

        output_path = get_embedding_folder(dataset, architecture, seed, step, layer)
        json_output = output_path / pathlib.Path("rep.json")
        npy_output = output_path / pathlib.Path("rep.npy")

        command_outline = (
            "python extract_features.py --input_file={data}"
            " --output_file={output}"
            " --vocab_file={model}/vocab.txt"
            " --bert_config_file={model}/bert_config.json"
            " --init_checkpoint={model}/bert_model.ckpt"
            " --layers={layer}"
            " --max_seq_length=128"
            " --batch_size=8"
        )
        command = command_outline.format(
            data=datapath, output=str(json_output), model=model_path, layer=layer
        )

    os.system("echo {}".format(command))
    os.system("cd {}".format(BERT_PATH))
    os.system(command)

    # Then convert json file to npy file
    representation = []
    with open(json_output) as f:
        for line in f:
            data = json.loads(line)
            for token in data["features"]:
                representation.append(token["layers"][0]["values"])
    representation = np.array(representation).T
    print("Saving representations at {}".format(npy_output))
    np.save(npy_output, representation)

    # Delete json file
    os.system("rm {}".format(str(json_output)))
    return representation
