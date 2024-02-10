from collections import defaultdict
from functools import partial
import inspect
import os
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import json

import config_utils
from config_utils import DictModule, dict_set, dict_in, dict_get, dict_update
from similarity.measure import Measure
from similarity.api import KeyId


# TODO: probably don't need caching now that configs are compiled

# path to config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), './')
# path to build directory
BUILD_DIR = os.path.join(os.path.dirname(__file__), "./api")

# cue package where the configs to instantiate are stored
API_PKG = "api"

# cache for package configs
cached_configs = {}

# default config for user registered configs
default_measure_config = {
    "_target_": "similarity.Measure",
    "measure": None,
    "fit_score": None,
    "fit": None,
    "score": None,
    # "call_key": None
}

# store for user defined configs
registry = {}

# register backends
if (Path(BUILD_DIR) / "api.json").exists():
    with open(Path(BUILD_DIR) / "api.json", "r") as f:
        configs = json.load(f)

    # TODO: assume to levels
    for k, v in configs.items():
        for k2, v2 in v.items():
            registry[k + "." + k2] = v2


def make(id, return_config=False, **kwargs):
    """
    Instantiate a config into a python object.
    Args:
        id: id of the config to instantiate.
        return_config: if True, return the config instead of the instantiated object.
        **kwargs: keyword arguments to pass to the python target.
    Returns:
        Instantiated python object.
    """
    if id not in registry:
        raise ValueError(f"Config '{id}' not found. Use similarity.register to register a new config.")

    if return_config is True:
        return {"id": id, **kwargs}

    obj = registry[id]
    if isinstance(obj, dict):
        # instantiate config
        return config_utils.make(
            id=None,
            package=None,
            key=None,
            config_dir=CONFIG_DIR,
            cached_config=obj,
            return_config=False,
            **kwargs
        )
    else:
        return registry[id](**kwargs)


def register(id, obj=None, override=False):
    def _register(id, obj):
        if not override:
            assert id not in registry, f"{id} already registered. Use override=True to force override."
        registry[id] = obj

    # if obj is None, register can be used as a decorator
    if obj is None:
        def decorator(obj):
            _register(id, obj)
            return obj
        return decorator
    else:
        _register(id, obj)


def make_old(
        key: KeyId,
        package: str = API_PKG,
        variants: bool = True,
        use_cache: bool = True,
        cached_package=None,
        return_config=False,
        _convert_="all",
        **kwargs) -> Measure:
    """
    Instantiate a python object from a config.
    When the result is a dict, by default don't output keys for variants. A key indicates a variant by specifying
    the variations from the default in the name, separated by a hypen '-'.
    Args:
        key: dot path to config
        package: package to use
        variants: if true, output all the keys, even keys containing a hypen '-' char (indicating a variant of a base method)
        use_cache: whether to use cached config
        cached_package: cached config
        _convert_: by default convert all omegaconf configs to python objects
        **kwargs: additional kwargs to pass to config
    """
    if cached_package is None:
        cached_package = {}

    if package is not None and use_cache:
        if package in cached_configs:
            cached_package = cached_configs[package]
        elif package == "api" and os.path.exists(BUILD_DIR + "/api.json"):
            # load api.json
            with open(BUILD_DIR + "/api.json", "r") as f:
                cached_package = json.load(f)
        else:
            # compile the package config (don't use key)
            cached_package = config_utils.make(package=package,
                                              config_dir=CONFIG_DIR,
                                              return_config=True)
        # store in cache
        cached_configs[package] = cached_package

    if package == "api":
        dict_update(cached_package, registry)

    # TODO: keep this?
    # potentially don't return variants
    # assume keys have the format: '{category}.{name}-{variation}'
    if variants is False and "." not in key:
        # the result is a dict of all the keys in the category
        # get all the keys without instantiating objects
        res = config_utils.make(
            id=None,
            package=package,
            key=key,
            config_dir=CONFIG_DIR,
            cached_config=cached_package,
            _convert_=_convert_,
            return_config=True,
            **kwargs
        )
        # filter out keys with a hypen
        non_variant_keys = {k: v for k, v in res.items() if "-" not in k}
        # instantiate objects
        res = {
            k: config_utils.make(
                id=None,
                package=package,
                key=key + "." + k,
                config_dir=CONFIG_DIR,
                cached_config=cached_package,
                return_config=return_config,
                _convert_=_convert_,
                **kwargs
            ) for k in non_variant_keys
        }
        return res

    # use cached config
    res = config_utils.make(
        id=None,
        package=package,
        key=key,
        config_dir=CONFIG_DIR,
        cached_config=cached_package,
        return_config=return_config,
        _convert_=_convert_,
        **kwargs
    )
    return res


def register_old(obj: object, id: str, **kwargs):
    global registry

    if isinstance(obj, dict):
        cfg = obj

    elif inspect.isfunction(obj):
        sig = inspect.signature(obj)
        params = sig.parameters

        # measure function should have 2 arguments (no if allow kwargs...)
        # assert len(params) == 2
        input_names = list(params.keys())
        # assume first arg is X, second is Y
        in_keys = [["X", input_names[0]], ["Y", input_names[1]]]

        if len(kwargs) > 0:
            obj = partial(obj, **kwargs)

        fun = DictModule(
            module=obj,
            in_keys=in_keys,
            # out_keys=["score"]  # TODO: with that, the output is {"score": score} instead of just score
            out_keys=None
        )
        cfg = fun

    elif inspect.isclass(obj):
        def _method_to_module(obj, method_name, outputs):
            if not hasattr(obj, method_name):
                return None

            method = getattr(obj, method_name)
            method_inputs = list(inspect.signature(method).parameters.keys())

            module = DictModule(
                module=method,
                in_keys=[
                    # assume first arg is self and refers to the measure object
                    ["measure", method_inputs[0]],
                    ["X", method_inputs[1]],
                    ["Y", method_inputs[2]]
                ],
                out_keys=outputs
            )
            return module

        # assume outputs are ["score"] for  score and fit_score
        fit = _method_to_module(obj, "fit", outputs=[])
        score = _method_to_module(obj, "score", outputs=["score"])
        fit_score = _method_to_module(obj, "fit_score", outputs=["score"])

        cfg = {
            "_target_": "similarity.Measure",
            "measure": {
                # need to create the measure object when the config is being instantiated
                "_target_": "similarity.create_object",
                "obj": obj,
                "kwargs": kwargs
            },
            "fit_score": fit_score,
            "fit": fit,
            "score": score
        }

    else:
        raise TypeError(f"Expected type function or class, got {type(obj)}")

    # default measure config
    if id.split(".")[0] == "measure":
        if "measure" not in registry:
            registry["measure"] = {}
        registry["measure"][id.split(".")[1]] = default_measure_config

    dict_set(registry, id, cfg, mkidx=True)
    # TODO: user registered measures don't have "properties" field


def create_object(obj, kwargs):
    """Just a utility function to create an object with kwargs (for use in config))"""
    return obj(**kwargs)


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


class Measure:
    def __new__(cls, id, **kwargs):
        return make(f"measure.{id}", **kwargs)


class Metric:
    def __new__(cls, id, **kwargs):
        return make(f"metric.{id}", **kwargs)


if __name__ == "__main__":
    build()
