from __future__ import annotations
import fnmatch

from similarity.types import BackendIdType, IdType, MeasureIdType


def _register_imports():
    # important to import transforms after backends since it uses measures defined in backends
    import similarity.transforms


# store registered objects
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

        # no matches found, try suggesting closest match
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
        return obj(*args, **kwargs)


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
    Factory class for creating MeasureInterface objects from measure id.
    """
    # def __new__(cls, measure_id: MeasureIdType, *args, **kwargs) -> "MeasureInterface":
    #     return make(f"measure.{measure_id}", *args, **kwargs)

    def __new__(cls, measure: MeasureIdType, backend: BackendIdType = "default", *args, **kwargs) -> "MeasureInterface":
        return make(f"measure.{backend}.{measure}", *args, **kwargs)


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
        """
        Args:
            measure: measure object, either instantiated from a class or a function. If the `interface` argument is None,
                the measure object is expected to have the following methods:
                * fit: fit the measure object to the data.
                * fit_score: fit the measure object to the data and return the score.
                * score: return the score without fitting the measure object.
                * __call__: call the measure object directly.
                If `measure` is a function, only __call__ is used. Calling fit, fit_score, or score will raise an error.
            interface: interface to use for the measure object. If None, uses default interface.
                The interface is specified as a dict where the keys are the method names of the MeasureInterface
                class and the values are the method names of the wrapped measure object to which they are mapped.
            preprocessing: preprocessing to apply to the data before passing to the measure object.
                The preprocessing can be a string, dict, or function. If a string, the string is used to
                look up the preprocessing function in the registry. If a dict, the dict should have the following
                keys:
                * id: id of the preprocessing function to use.
                * inputs: potential list of inputs to pass to the preprocessing function.
                If a function, the function is applied directly to each dataset.
            postprocessing: postprocessing to apply to the score after it is returned from the measure object.
                The postprocessing can be a string, dict, or function. If a string, the string is used to
                look up the postprocessing function in the registry. If a dict, the dict should have the following
                keys:
                * id: id of the postprocessing function to use.
                * inputs: potential list of inputs to pass to the postprocessing function.
                If a function, the function is applied directly to the score.
        """
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
                X = make(f"preprocessing/{p}", X)
                Y = make(f"preprocessing/{p}", Y)
            elif isinstance(p, dict):
                # if dict, check for inputs key to pass data to the preprocessing function
                assert "id" in p, f"Expected 'id' in preprocessing dict, got {p}"
                if "inputs" in p:
                    data = {"X": X, "Y": Y}
                    args = [data[i] for i in p["inputs"]]
                    X, Y = make(f"preprocessing/{p['id']}", *args)
                else:
                    X = make(p["id"], X)
                    Y = make(p["id"], Y)
            else:
                # assume p is a function
                X = p(X)
                Y = p(Y)
        return X, Y

    def _postprocess(self, X, Y, score):
        """
        Apply postprocessing to score after it is returned from measure object.
        """
        for p in self.postprocessing:
            if isinstance(p, str):
                score = make(f"postprocessing/{p}", score)
            elif isinstance(p, dict):
                # if dict, check for inputs key to pass data to the postprocessing function
                assert "id" in p, f"Expected 'id' in postprocessing dict, got {p}"
                if "inputs" in p:
                    data = {"X": X, "Y": Y, "score": score}
                    args = [data[k] for k in p["inputs"]]
                    score = make(f"postprocessing/{p['id']}", *args)
            else:
                # assume p is a function
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
        Measure the similarity between two datasets.
        
        Args:
            X: data array, 2d or 3d with last dimension `neuron`.
            Y: data array, 2d or 3d with last dimension `neuron`.

        Returns:
            score: similarity score.
        """
        X, Y = self._preprocess(X, Y)
        score = getattr(self.measure, self.interface.get("__call__", "__call__"))(X, Y)
        score = self._postprocess(X, Y, score)
        return score


def register(id, obj=None, function=False, interface=None, preprocessing=None, postprocessing=None, override=True, separator='/'):
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
        if obj is None, returns a decorator function.
    """
    def _register(id, obj):
        if not override:
            assert id not in registry, f"{id} already registered. Use override=True to force override."

        # if obj is a dict, register it directly (no need to add interface class)
        if isinstance(obj, dict):
            registry[id] = obj
            return

        # if id starts with 'measure', wrap obj in MeasureInterface
        category = id.split(separator)[0]
        if category == "measure":
            if function:
                # encapsulate in a function so that make(id) returns the function itself without calling it
                def _obj():
                    return obj
            else:
                _obj = obj

            # wrap measure in a MeasureInterface
            def wrap_measure():
                measure = _obj()
                if isinstance(measure, MeasureInterface):
                    # if measure is already a MeasureInterface object, return it directly
                    # currently don't support overriding interface, preprocessing, or postprocessing
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

            registry[id] = wrap_measure

        else:
            registry[id] = obj

    # if obj is None, register can be used as a decorator
    if obj is None:
        def decorator(obj):
            _register(id, obj)
            return obj
        return decorator
    else:
        _register(id, obj)


def all_measures():
    # TODO: don't work if don't call make
    # return {k: v for k, v in registry.items() if "measure" in k and len(k.split(".")) == 3}
    # return make("measure.*.*")
    return make("measure/*/*")


# def register_measure(id, obj=None, **kwargs):
#     id = f"measure/{id}"
#     register(id, obj=obj, **kwargs)


# def make_measure(id):
#     return make(f"measure/{id}")
