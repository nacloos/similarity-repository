import os
from typing import Literal
import config_utils

from similarity.metric import Metric

# CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../configs')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), './')

# TODO: generate automatically from configs
# MetricId = Literal[
#     "cca",
#     "cka",
#     "procrustes",
#     "svcca"
# ]
# TODO: add type hint for auto-complete
PackageId = Literal[
    "metric"
]

cached_configs = {}


# TODO: remove id arg (use separate package and key args instead)?
# but if use a single id arg, then can have autocomplete 
# e.g. automatically generate a Literal type for all the possible ids 
# or use a separate make function for each package (e.g. make_metric)
def make(id=None, package: PackageId = None, key=None, **kwargs) -> Metric:
    """
    Instantiate a python object from a config file.
    Args:
        id: path to the config file and key to instantiate
        kwargs: keyword arguments passed to the object constructor
    """
    if package is not None:
        if package in cached_configs:
            cached_config = cached_configs[package]
        else:
            # compile the package config
            cached_config = config_utils.make(package=package,
                                              config_dir=CONFIG_DIR,
                                              return_config=True)
            cached_configs[package] = cached_config
    else:
        cached_config = None

    # use cached config
    return config_utils.make(
        id=id,
        package=package,
        key=key,
        config_dir=CONFIG_DIR,
        cached_config=cached_config,
        **kwargs
    )


# TODO: build all the packages beforehand?
def build():
    # TODO: compile cue configs and save them as json
    config = config_utils.make(package="metric", config_dir=CONFIG_DIR, return_config=True)
    print(config)
