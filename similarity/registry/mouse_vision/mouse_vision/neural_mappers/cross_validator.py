import numpy as np
from collections import OrderedDict
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import mouse_vision.neural_mappers.neural_map_base as nm
from mouse_vision.reliability.metrics import noise_estimation
from mouse_vision.neural_mappers.pipeline_neural_map import PipelineNeuralMap
from mouse_vision.neural_mappers.utils import map_from_str
from mouse_vision.core.utils import iterate_dicts, dict_to_str

__all__ = ["CrossValidator"]

class CrossValidator():
    """
    Inputs:
        neural_map_str : (string) one of ["corr", "pls", "factored", "identity"]
        cv_params      : (list) of neural mapper to hyperparameter-value pairs. For example:
                         [{"map_type": "pls", "map_kwargs": {"n_components": 10, "scale": False}},
                          {"map_type": "pls", "map_kwargs": {"n_components": 20, "scale": False}}].
                         See neural_mapppers/map_param_utils.py for more details.
        n_cv_splits    : (int) number of train/val splits for cross-validation
    """
    def __init__(self, neural_map_str, cv_params, n_cv_splits,
                 center_trials=True, mode="spearman_brown_split_half"):
        assert neural_map_str in ["corr", "pls", "factored", "identity"]
        self.neural_map_str = neural_map_str
        self.neural_mapper = None
        self.scores = None
        self.center_trials = center_trials
        self.mode = mode
        self.Y_sample_dim = 0 if self.center_trials else 1
        self.n_targets = None
        self.n_source = None

        # Cross-validation parameters
        self._cv_params = cv_params
        self._n_cv_splits = n_cv_splits

        if len(self._cv_params) == 0:
            raise ValueError("Need to specify hyperparameter values, even if only one dictionary.")

    def _check_dims(self, X, Y):
        # If target is a one-dimensional vector, make it into a column vector
        if (Y.ndim == 1) and self.center_trials:
            Y = Y[:,np.newaxis]
        assert X.ndim == 2
        if self.center_trials:
            assert Y.ndim == 2
        else:
            assert Y.ndim == 3
        assert X.shape[0] == Y.shape[self.Y_sample_dim], \
            f"Features and targets sample dimension does not match: {X.shape} and {Y.shape}"
        return Y

    def _Y_index(self, Y, idx):
        if self.center_trials:
            return Y[idx, :]
        else:
            return Y[:, idx, :]

    def fit(self, X, Y):
        """
        Performs the cross-validated fitting procedure

        Inputs:
            X : (numpy.ndarray) of dimensions (num_stimuli, num_features)
            Y : (numpy.ndarray) of dimensions (num_stimuli, num_targets) if not separating by animal
              : otherwise a dict of these numpy arrays per animal
        """
        if isinstance(Y, dict):
            self.n_targets = OrderedDict()
            for spec_name, spec_resp in Y.items():
                Y[spec_name] = self._check_dims(X, spec_resp)
                self.n_targets[spec_name] = Y[spec_name].shape[-1]
        else:
            Y = self._check_dims(X, Y)
            self.n_targets = Y.shape[-1]

        self.n_source = X.shape[1]

        num_samples = X.shape[0]
        scores = []
        for param_set in self._cv_params:
            assert "map_type" in param_set.keys() and "map_kwargs" in param_set.keys()
            assert param_set["map_type"] == self.neural_map_str
            param = param_set["map_kwargs"]

            print(f"Current parameters: {param}")
            param_scores = list()

            # If X has no samples, it means we aren't training / doing cross-validation
            # so we skip the k-fold cross-validation loop.
            if X.shape[0] == 0:
                break

            # Perform KFold cross-validation for _n_cv_splits splits
            kf = KFold(n_splits=self._n_cv_splits)
            for train_idx, val_idx  in kf.split(X):
                if self.center_trials:
                    X_train = X[train_idx, :]
                    X_val = X[val_idx, :]
                    if isinstance(Y, dict):
                        Y_train = OrderedDict()
                        Y_val = OrderedDict()
                        for spec_name, spec_resp in Y.items():
                            Y_train[spec_name] = self._Y_index(spec_resp, train_idx)
                            Y_val[spec_name] = self._Y_index(spec_resp, val_idx)
                    else:
                        Y_train = self._Y_index(Y, train_idx)
                        Y_val = self._Y_index(Y, val_idx)

                    curr_mapper = PipelineNeuralMap(map_type=self.neural_map_str,
                                                    map_kwargs=param)
                    curr_mapper.fit(X_train, Y_train)
                    Y_pred = curr_mapper.predict(X_val)

                    if curr_mapper._fitted == False:
                        curr_score = np.NaN
                    else:
                        curr_score = curr_mapper.score(Y_val, Y_pred)
                else:
                    if isinstance(Y, dict):
                        score_dict = OrderedDict()
                        for spec_name, spec_resp in Y.items():
                            curr_results = noise_estimation(target_N=spec_resp,
                                                            source_N=X,
                                                            source_map_kwargs=param_set,
                                                            train_img_idx=train_idx, test_img_idx=val_idx,
                                                            num_source_units=None,
                                                            parallelize_per_target_unit=False,
                                                            metric="pearsonr" if self.neural_map_str != "identity" else "rsa",
                                                            mode=self.mode,
                                                            center=np.nanmean,
                                                            summary_center="raw",
                                                            sync=True,
                                                            n_jobs=5
                                                            )
                            # average across bootstrap trials
                            score_dict[spec_name] = np.mean(curr_results, axis=0)

                        # average across animals since we want a single score
                        curr_score = np.mean([s for s in score_dict.values()])
                    else:
                        curr_results = noise_estimation(target_N=Y,
                                                        source_N=X,
                                                        source_map_kwargs=param_set,
                                                        train_img_idx=train_idx, test_img_idx=val_idx,
                                                        num_source_units=None,
                                                        parallelize_per_target_unit=False,
                                                        metric="pearsonr" if self.neural_map_str != "identity" else "rsa",
                                                        mode=self.mode,
                                                        center=np.nanmean,
                                                        summary_center="raw",
                                                        sync=True,
                                                        n_jobs=5
                                                        )

                        curr_score = np.mean(curr_results, axis=0)
                param_scores.append(curr_score)
            scores.append(np.nanmean(param_scores))

        # This is the case when self._cv_params is empty: [{}] when the identity neural mapper is used
        if len(scores) != len(self._cv_params):
            assert len(scores)+1 == len(self._cv_params)
            scores.append(0) # hack so that nanargmax(scores) doesn't fail

        cv_params_str = [dict_to_str(p) for p in self._cv_params]
        self.scores = dict(zip(cv_params_str, scores))

        # If not cross-validation params were used
        best_params_idx = np.nanargmax(scores)
        assert self._cv_params[best_params_idx]["map_type"] == self.neural_map_str
        best_params = self._cv_params[best_params_idx]["map_kwargs"]
        print(f"Best parameters: {best_params}")

        # Once we selected the best params, then fit to the entire set
        # but only if we center_trials, otherwise
        # we run noise_estimation on the train/test split (test indices are outside fit method
        if self.center_trials:
            self.neural_mapper = PipelineNeuralMap(map_type=self.neural_map_str,
                                                   map_kwargs=best_params)
            self.neural_mapper.fit(X, Y)

        self.best_params = {"map_type": self.neural_map_str, "map_kwargs": best_params}

    def predict(self, X):
        """
        Performs response prediction with features from val/test set.

        Inputs:
            X      : (numpy.ndarray) of dimensions (num_stimuli, num_features)

        Returns:
            Y_pred : (numpy.ndarray) of dimensions (num_stimuli, num_targets)
        """
        # we only call predict outside if we have fit the neural mapper in the fit method, so center_trials=True
        assert self.center_trials is True
        assert self.n_targets is not None
        assert self.n_source is not None
        assert X.ndim == 2
        assert X.shape[1] == self.n_source, \
            f"{X.shape[1]} does not match number of sources that were fit {self.n_source}."

        if self.neural_mapper is None:
            print("[WARNING] Fitting was not performed or failed, returning NaN predictions.")
            n_samples = X.shape[0]
            if isinstance(self.n_targets, dict):
                Y_pred = OrderedDict()
                for spec_name, spec_n_targets in self.n_targets.items():
                    Y_pred[spec_name] = np.zeros((n_samples, spec_n_targets)) + np.NaN
            else:
                Y_pred = np.zeros((n_samples, self.n_targets)) + np.NaN
        else:
            Y_pred = self.neural_mapper.predict(X)
        return Y_pred

if __name__ == "__main__":
    # Initialize variables
    mapper = "pls"
    cv_params = {"n_components": [6,8,10], "scale": [True, False]}
    n_splits = 5

    np.random.seed(0)

    # Setup
    source = 30
    target = 20
    n_train = 200
    n_test = 100
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    X_test = np.random.randn(n_test,source)
    Y_test = np.random.randn(n_test,target)

    c = CrossValidator(mapper, cv_params, n_splits)
    c.fit(X, Y)
    Y_pred = c.predict(X_test)
    assert Y_pred.shape == Y_test.shape
    print(f"Scores {c.scores}")


