import os
from typing import Literal
from omegaconf import OmegaConf
import json

import config_utils
from similarity.metric import Metric
from similarity.api import KeyId

# CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../configs')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), './')
BUILD_DIR = os.path.join(os.path.dirname(__file__), "./api")


# TODO: add type hint for auto-complete
PackageId = Literal[
    "metric"
]

cached_configs = {}


# TODO: remove id arg (use separate package and key args instead)?
# but if use a single id arg, then can have autocomplete 
# e.g. automatically generate a Literal type for all the possible ids 
# or use a separate make function for each package (e.g. make_metric)
def make(
        key: KeyId,
        package: PackageId = "api",
        defaults_only=False,
        variants_only=False,
        use_cache=True,
        **kwargs) -> Metric:
    """
    Instantiate a python object from a config file.
    Args:
        id: path to the config file and key to instantiate
        kwargs: keyword arguments passed to the object constructor
    """
    # TODO: problem with caching is that it doesn't update the package config when modifying a cue config
    if package is not None and use_cache:
        if package in cached_configs:
            cached_config = cached_configs[package]
        elif package == "api" and os.path.exists(BUILD_DIR + "/api.json"):
            # load api.json
            with open(BUILD_DIR + "/api.json", "r") as f:
                cached_config = json.load(f)
        else:
            # compile the package config
            cached_config = config_utils.make(package=package,
                                              config_dir=CONFIG_DIR,
                                              return_config=True)
        cached_configs[package] = cached_config
    else:
        cached_config = None

    # TODO: preprocessing

    # use cached config
    return config_utils.make(
        id=None,
        package=package,
        key=key,
        config_dir=CONFIG_DIR,
        cached_config=cached_config,
        **kwargs
    )


def build(build_dir=BUILD_DIR):
    print("Building API...")
    api = config_utils.make(package="api", config_dir=CONFIG_DIR, return_config=True)
    with open(build_dir + "/api.json", "w") as f:
        json.dump(api, f, indent=4)

    # save all the keys in a literal type (allow autocomplete for similarity.make)
    with open(build_dir + "/api.json", "r") as f:
        api = json.load(f)

    def extract_dot_paths(config, dot_paths, prefix="", stop_at=[]):
        """
        Extract dot paths from a config dict. Stop recursion at keys in stop_at. 
        """
        def _stop(config, stop_at):
            for stop in stop_at:
                if stop in config:
                    return True
            return False

        for k, v in config.items():
            if isinstance(v, dict):
                # store intermediate paths
                dot_paths.append(prefix + k)

                if _stop(v, stop_at):
                    # stop recursion
                    continue

                extract_dot_paths(v, dot_paths, prefix + k + ".", stop_at=stop_at)
            else:
                dot_paths.append(prefix + k)
        return dot_paths

    dot_paths = extract_dot_paths(api, [], stop_at=["_target_", "_out_"])

    # save literal type
    code = "# Automatically generated code. Do not modify.\n"
    code += "from typing import Literal\n"
    code += "\n\n"
    literal_type = ",\n\t".join([f'"{p}"' for p in dot_paths])
    literal_type = "Literal[\n\t" + literal_type + "\n]"

    code += f"KeyId = {literal_type}\n"

    with open(build_dir + "/__init__.py", "w") as f:
        f.write(code)


if __name__ == "__main__":
    build()
