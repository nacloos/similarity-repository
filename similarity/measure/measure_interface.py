from typing import Callable
from config_utils import DictModule, DictSequential


class Measure:
    default_interface = {
        "fit": "fit",
        "score": "score",
        "fit_score": "fit_score"
    }
    default_preprocessing_inputs = ["X", "Y"]
    default_preprocessing_outputs = ["X", "Y"]
    default_postprocessing_inputs = ["score"]
    default_postprocessing_outputs = ["score"]

    def __init__(
            self,
            measure,
            fit_score,
            score=None,
            fit=None,
            interface=None,
            preprocessing=None,
            postprocessing=None):

        self._measure = measure
        self._score = score
        self._fit = fit
        self._fit_score = fit_score

        if interface is None:
            interface = {
                "fit": "fit",
                "score": "score",
                "fit_score": "fit_score"
            }
        else:
            assert isinstance(interface, dict), type(interface)
        self.interface = interface

        _preprocess = create_module_seq(preprocessing, self.default_preprocessing_inputs, self.default_preprocessing_outputs)
        _postprocess = create_module_seq(postprocessing, self.default_postprocessing_inputs, self.default_postprocessing_outputs)

        # compose modules
        self._fit = DictModule(
            module=_preprocess + fit,
            in_keys=["measure", "X", "Y"],
            out_keys=[]
        )
        self._score = DictModule(
            module=_preprocess + score + _postprocess,
            in_keys=["measure", "X", "Y"],
            out_keys=[["score", None]]  # return the score value as a number (not a dict)
        )
        self._fit_score = DictModule(
            module=_preprocess + fit_score + _postprocess,
            in_keys=["measure", "X", "Y"],
            out_keys=[["score", None]]  # return the score value as a number (not a dict)
        )

        impls = {
            "fit": self._fit_impl,
            "score": self._score_impl,
            "fit_score": self._fit_score_impl
        }

        self.impls = {}
        # use interface to rename impls keys
        for k, v in impls.items():
            new_k = interface.get(k, k)
            self.impls[new_k] = v

    def _fit_impl(self, X, Y):
        if self._fit is not None:
            self._fit(measure=self._measure, X=X, Y=Y)

    def _score_impl(self, X, Y):
        return self._score(measure=self._measure, X=X, Y=Y)

    def _fit_score_impl(self, X, Y):
        if self._fit_score is not None:
            return self._fit_score(measure=self._measure, X=X, Y=Y)
        else:
            self._fit_impl(X, Y)
            return self._score_impl(X, Y)

    def fit(self, X, Y):
        if self._fit is not None:
            self._fit(measure=self._measure, X=X, Y=Y)

    def score(self, X, Y):
        return self._score(measure=self._measure, X=X, Y=Y)

    def fit_score(self, X, Y):
        if self._fit_score is not None:
            return self._fit_score(measure=self._measure, X=X, Y=Y)
        else:
            self.fit(X, Y)
            return self.score(X, Y)

    def __getattr__(self, name):
        # prevent accessing private attributes
        if name.startswith("_"):
            raise AttributeError(name)

        # call the implementation
        if name in self.impls:
            return self.impls[name]

        # if attr is not found in this class, try the measure
        return getattr(self._measure, name)

    def __call__(self, X, Y):
        # have to handle __call__ separately
        if "__call__" in self.impls:
            return self.impls["__call__"](X, Y)
        else:
            raise TypeError(f"'{self.__class__.__name__}' object is not callable")

    def __repr__(self):

        s = ""
        # s += "Metric:\n"
        s += "Metric["
        for k, v in self.interface.items():
            # s += f"  {v}(X, Y)\n"
            s += f"{v}(X, Y), "
        s = s[:-2] + "]"
        # TODO: Remove wrap DictModule around DictSequential to simplify it
        return s


def create_module_seq(modules, in_keys, out_keys):
    def _parse_modules(modules, in_keys, out_keys):
        if modules is None:
            modules = []
        elif isinstance(modules, DictModule):
            modules = [modules]
        elif isinstance(modules, Callable):
            modules = [
                DictModule(
                    module=modules,
                    in_keys=in_keys,
                    out_keys=out_keys
                )
            ]
        elif isinstance(modules, list):
            parsed_modules = []
            for m in modules:
                parsed_modules.extend(
                    _parse_modules(m, in_keys, out_keys)
                )
            modules = parsed_modules
        else:
            raise TypeError(f"Expected Callable, DictModule or list, found {type(modules)}")
        return modules

    modules = _parse_modules(modules, in_keys, out_keys)
    return DictSequential(*modules)
