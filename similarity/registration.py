from __future__ import annotations
import fnmatch
from functools import partial

from similarity.types import IdType


# store registered objects
registry = {}

# keep track of derived measures
DERIVED_MEASURES = {}


def match(id: str) -> list[str]:
    """
    Find all the keys that match a pattern.

    Args:
        id: pattern to match.

    Returns:
        List of keys that match the pattern.
    """
    assert isinstance(id, str), f"Expected type str, got {type(id)}"
    return [k for k in registry.keys() if fnmatch.fnmatch(k, id)]


def _preprocess(X, Y, preprocessing):
    """
    Apply preprocessing to data before passing to measure.
    """
    for p in preprocessing:
        if isinstance(p, str):
            X = make(f"preprocessing/{p}")(X)
            Y = make(f"preprocessing/{p}")(Y)
        elif isinstance(p, dict):
            # if dict, check for inputs key to pass data to the preprocessing function
            assert "id" in p, f"Expected 'id' in preprocessing dict, got {p}"
            if "inputs" in p:
                data = {"X": X, "Y": Y}
                args = [data[i] for i in p["inputs"]]
                X, Y = make(f"preprocessing/{p['id']}")(*args)
            else:
                X = make(p["id"])(X)
                Y = make(p["id"])(Y)
        else:
            # assume p is a function
            X = p(X)
            Y = p(Y)
    return X, Y

def _postprocess(X, Y, score, postprocessing):
    """
    Apply postprocessing to score after it is returned from measure object.
    """
    for p in postprocessing:
        if isinstance(p, str):
            score = make(f"postprocessing/{p}")(score)
        elif isinstance(p, dict):
            # if dict, check for inputs key to pass data to the postprocessing function
            assert "id" in p, f"Expected 'id' in postprocessing dict, got {p}"
            if "inputs" in p:
                data = {"X": X, "Y": Y, "score": score}
                args = [data[k] for k in p["inputs"]]
                score = make(f"postprocessing/{p['id']}")(*args)
        else:
            # assume p is a function
            score = p(score)
    return score

def apply_measure(X, Y, measure, preprocessing=None, postprocessing=None, **kwargs):
    X, Y = _preprocess(X, Y, preprocessing=preprocessing)
    score = measure(X, Y, **kwargs)
    score = _postprocess(X, Y, score, postprocessing=postprocessing)
    return score


def wrap_measure(measure, preprocessing=None, postprocessing=None):
    preprocessing = [] if preprocessing is None else preprocessing
    postprocessing = [] if postprocessing is None else postprocessing
    return partial(apply_measure, measure=measure, preprocessing=preprocessing, postprocessing=postprocessing)


def register(id, obj=None, preprocessing=None, postprocessing=None):
    """
    Register a new object in the registry.

    Args:
        id: id of the object to register.
        obj: object to register.
        preprocessing: preprocessing to apply to the object.
        postprocessing: postprocessing to apply to the object.
    """
    def _register(id, obj):
        if preprocessing is not None or postprocessing is not None:
            registry[id] = wrap_measure(obj, preprocessing, postprocessing)
        else:
            registry[id] = obj

    if obj is None:
        def decorator(obj):
            _register(id, obj)
            return obj
        return decorator
    else:
        _register(id, obj)


def make(id: IdType, *args, **kwargs):
    """
    Instantiate a config into a python object.

    Args:
        id: id of the config to instantiate.
        *args: positional arguments to pass to the python target.
        **kwargs: keyword arguments to pass to the python target.

    Returns:
        Instantiated python object.
    """
    if id not in registry:
        matches = match(id)
        if len(matches) > 0:
            return {k: make(k, *args, **kwargs) for k in matches}

        # no matches found, try suggesting closest match
        import difflib
        suggestion = difflib.get_close_matches(id, registry.keys(), n=1)

        if len(suggestion) == 0:
            raise ValueError(f"`{id}` not found in registry. Use `similarity.register` to register a new entry.")
        else:
            raise ValueError(f"`{id}` not found in registry. Did you mean: `{suggestion[0]}`? Use `similarity.register` to register a new entry.")

    return registry[id]


def all_measures():
    return make("measure/*/*")


def all_papers():
    return make("paper/*")
