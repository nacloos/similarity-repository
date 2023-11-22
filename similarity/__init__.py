import os
from typing import Literal
import config_utils

from similarity.metric import Metric

# CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../configs')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), './')

# TODO: generate automatically from configs
MetricId = Literal[
    "cca",
    "cka",
    "procrustes",
    "svcca"
]


def make(id: MetricId, **kwargs) -> Metric:
    """
    Instantiate a python object from a config file.
    Args:
        id: path to the config file and key to instantiate
        kwargs: keyword arguments passed to the object constructor
    """
    return config_utils.make(id, config_dir=CONFIG_DIR, **kwargs)
