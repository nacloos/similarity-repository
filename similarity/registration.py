from __future__ import annotations
from functools import partial
import fnmatch
import inspect

from similarity.types import IdType, MeasureIdType


def _register_imports():
    # important to import transforms after backends since it uses measures defined in backends
    import similarity.transforms


# store for user defined objects
registry = {}


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
    _register_imports()

    if id not in registry:
        matches = match(id)
        if len(matches) > 0:
            return {k: make(k, *args, **kwargs) for k in matches}

        # no matches found, suggest closest match
        import difflib
        suggestion = difflib.get_close_matches(id, registry.keys(), n=1)
    
        if len(suggestion) == 0:
            raise ValueError(f"`{id}` not found in registry. Use `similarity.register` to register a new entry.")
        else:
            raise ValueError(f"`{id}` not found in registry. Did you mean: `{suggestion[0]}`? Use `similarity.register` to register a new entry.")

    obj = registry[id]

    if isinstance(obj, dict):
        return obj
    else:
        return registry[id](*args, **kwargs)


def is_registered(id: str) -> bool:
    """
    Check if a key is registered.

    Args:
        id: key to check.

    Returns:
        True if the key is registered, False otherwise.
    """
    assert isinstance(id, str), f"Expected type str, got {type(id)}"
    return id in registry


def match(id: str) -> list[str]:
    """
    Find all the keys that match a pattern.

    Args:
        id: pattern to match.

    Returns:
        List of keys that match the pattern.
    """
    _register_imports()
    assert isinstance(id, str), f"Expected type str, got {type(id)}"
    return [k for k in registry.keys() if fnmatch.fnmatch(k, id)]


class Measure:
    """
    Factory class for creating MeasureInterface object from measure id.
    """
    def __new__(cls, measure_id: MeasureIdType, *args, **kwargs) -> "MeasureInterface":
        return make(f"measure.{measure_id}", *args, **kwargs)


class MeasureInterface:
    """
    Thin wrapper around a measure object that allows for preprocessing and postprocessing of data.
    """
    def __init__(
            self,
            measure,
            interface: dict = None,
            preprocessing: list = None,
            postprocessing: list = None
    ):
        assert not isinstance(measure, MeasureInterface), f"Measure is already a MeasureInterface object."
        interface = {} if interface is None else interface
        preprocessing = [] if preprocessing is None else preprocessing
        postprocessing = [] if postprocessing is None else postprocessing

        assert isinstance(interface, dict), f"Expected type dict, got {type(interface)}"
        assert isinstance(preprocessing, list), f"Expected type list, got {type(preprocessing)}"
        assert isinstance(postprocessing, list), f"Expected type list, got {type(postprocessing)}"

        self.measure = measure
        self.interface = interface
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def _preprocess(self, X, Y):
        """
        Apply preprocessing to data before passing to measure object.
        """
        for p in self.preprocessing:
            if isinstance(p, str):
                X = make(f"preprocessing.{p}", X)
                Y = make(f"preprocessing.{p}", Y)
            elif isinstance(p, dict):
                assert "id" in p, f"Expected 'id' in preprocessing dict, got {p}"
                if "inputs" in p:
                    data = {"X": X, "Y": Y}
                    args = [data[i] for i in p["inputs"]]
                    X, Y = make(f"preprocessing.{p['id']}", *args)
                else:
                    X = make(p["id"], X)
                    Y = make(p["id"], Y)
            else:
                X = p(X)
                Y = p(Y)
        return X, Y

    def _postprocess(self, X, Y, score):
        """
        Apply postprocessing to score after it is returned from measure object.
        """
        for p in self.postprocessing:
            if isinstance(p, str):
                score = make(f"postprocessing.{p}", score)
            elif isinstance(p, dict):
                assert "id" in p, f"Expected 'id' in postprocessing dict, got {p}"
                if "inputs" in p:
                    data = {"X": X, "Y": Y, "score": score}
                    args = [data[k] for k in p["inputs"]]
                    score = make(f"postprocessing.{p['id']}", *args)
            else:
                score = p(score)
        return score

    def fit(self, X, Y):
        """
        Fit the measure object to the data.

        Args:
            X: data.
            Y: data.
        """
        X, Y = self._preprocess(X, Y)
        getattr(self.measure, self.interface.get("fit", "fit"))(X, Y)

    def fit_score(self, X, Y):
        """
        Fit the measure object to the data and return the score.

        Args:
            X: data.
            Y: data.

        Returns:
            score: similarity score.
        """
        X, Y = self._preprocess(X, Y)
        score = getattr(
            self.measure,
            self.interface.get("fit_score", "fit_score")
        )(X, Y)
        score = self._postprocess(X, Y, score)
        return score

    def score(self, X, Y):
        """
        Return the score without fitting the measure object.

        Args:
            X: data.
            Y: data.

        Returns:
            score: similarity score.
        """
        X, Y = self._preprocess(X, Y)
        score = getattr(self.measure, self.interface.get("score", "score"))(X, Y)
        score = self._postprocess(X, Y, score)
        return score

    def __call__(self, X, Y):
        """
        Call the measure object directly.

        Args:
            X: data.
            Y: data.

        Returns:
            score: similarity score.
        """
        X, Y = self._preprocess(X, Y)
        score = getattr(self.measure, self.interface.get("__call__", "__call__"))(X, Y)
        score = self._postprocess(X, Y, score)
        return score


def register(id, obj=None, function=False, interface=None, preprocessing=None, postprocessing=None, override=True):
    """
    Register a function or class in the registry. Can be used as a decorator if obj argument is None.

    Args:
        id: id to register the object under.
        obj: object to register.
        function: if True, obj is a function that returns the object.
        interface: interface to use for the measure object.
        preprocessing: preprocessing to apply to the data before passing to the measure object.
        postprocessing: postprocessing to apply to the score after it is returned from the measure object.
        override: if True, override existing registration.

    Returns:
        obj: object that was registered.
    """
    def _register(id, obj):
        if not override:
            assert id not in registry, f"{id} already registered. Use override=True to force override."

        # if obj is a dict, register it directly (no need to add interface class)
        if isinstance(obj, dict):
            registry[id] = obj
            return

        if isinstance(obj, partial):
            base_obj = obj.func
        else:
            base_obj = obj
        make_obj = obj

        category = id.split(".")[0]
        if category == "measure":

            # TODO: clean implementation
            if function:
                assert inspect.isfunction(base_obj) or inspect.ismethod(base_obj), f"Expected type function or method for {obj}, got {type(obj)}"
                # encapsulate in a function so that make(id) returns the function itself without calling it
                def _obj():
                    return obj
            else:
                # assert inspect.isclass(base_obj), f"Expected type class for {obj}, got {type(obj)}"
                _obj = obj

            # wrap measure in a MeasureInterface
            def make_obj():
                measure = _obj()
                if isinstance(measure, MeasureInterface):
                    assert interface is None, f"Expected interface to be None, got {interface}"
                    assert preprocessing is None, f"Expected preprocessing to be None, got {preprocessing}"
                    assert postprocessing is None, f"Expected postprocessing to be None, got {postprocessing}"
                    return measure

                measure_interface = MeasureInterface(
                    measure=measure,
                    interface=interface,
                    preprocessing=preprocessing,
                    postprocessing=postprocessing
                )
                return measure_interface

        registry[id] = make_obj

    # if obj is None, register can be used as a decorator
    if obj is None:
        def decorator(obj):
            _register(id, obj)
            return obj
        return decorator
    else:
        _register(id, obj)
