import os

import numpy as np
import torchvision.transforms as transforms

from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from mouse_vision.core.utils import get_base_arch_name
from mouse_vision.core.default_dirs import HVM_DATA_PATH
from mouse_vision.core.constants import SVM_CV_C, SVM_CV_C_LONG, RIDGE_CV_ALPHA, RIDGE_CV_ALPHA_LONG
from mouse_vision.sklearn_transfer_datasets import HvmDataset, BaseFitter
from mouse_vision.model_training.trainer_transforms import TRAINER_TRANSFORMS


__all__ = ["HvmFitter"]


class HvmFitter(BaseFitter):
    def __init__(self, model_name, train_test_splits_file, hvm_data_path=HVM_DATA_PATH, name="hvm_dataset"):

        # Get HVM dataset object and use generic image transforms
        arch_name, _ = get_base_arch_name(model_name)
        img_transforms = [transforms.ToPILImage()] + TRAINER_TRANSFORMS[
            "SupervisedImageNetTrainer_64x64"
        ]["val"]
        image_transforms = transforms.Compose(img_transforms)
        dataset = HvmDataset(image_transforms, datapath=hvm_data_path, name=name)

        super(HvmFitter, self).__init__(dataset, model_name)

        self.stim_meta = dataset.get_stim_metadata()
        self.train_test_splits_file = train_test_splits_file

    def train_test_split(self, num_stimuli):
        """
        Reads and returns the list of train-test splits from a file.

        Outputs:
            splits : (list of dict) list of length number of splits, where each entry
                     is a dictionary with keys "train" and "test". The values are the
                     train indicies and test indicies respectively.
        """
        assert os.path.isfile(self.train_test_splits_file)

        train_test_splits = np.load(self.train_test_splits_file, allow_pickle=True)
        assert "arr_0" in train_test_splits.keys()
        splits = train_test_splits["arr_0"]
        return splits

    def fit_task(self, task_type, split_data, cls_kwargs=None):
        self.train_features = split_data["train_features"]
        self.train_idx = split_data["train_idx"]
        self.test_features = split_data["test_features"]
        self.test_idx = split_data["test_idx"]
        self.train_targets = None
        self.test_targets = None

        if task_type == "categorization":
            return self._fit_categorization(instance_cat=False, cls_kwargs=cls_kwargs)
        elif task_type == "instance_categorization":
            return self._fit_categorization(instance_cat=True, cls_kwargs=cls_kwargs)
        elif task_type == "cat1":
            return self._fit_categorization(cat=1, cls_kwargs=cls_kwargs)
        elif task_type == "cat2":
            return self._fit_categorization(cat=2, cls_kwargs=cls_kwargs)
        elif task_type == "cat3":
            return self._fit_categorization(cat=3, cls_kwargs=cls_kwargs)
        elif task_type == "cat4":
            return self._fit_categorization(cat=4, cls_kwargs=cls_kwargs)
        elif task_type == "cat5":
            return self._fit_categorization(cat=5, cls_kwargs=cls_kwargs)
        elif task_type == "cat6":
            return self._fit_categorization(cat=6, cls_kwargs=cls_kwargs)
        elif task_type == "cat7":
            return self._fit_categorization(cat=7, cls_kwargs=cls_kwargs)
        elif task_type == "cat8":
            return self._fit_categorization(cat=8, cls_kwargs=cls_kwargs)
        elif task_type == "pose":
            return self._fit_pose(cls_kwargs=cls_kwargs)
        elif task_type == "position":
            return self._fit_position(cls_kwargs=cls_kwargs)
        elif task_type == "size":
            return self._fit_size(cls_kwargs=cls_kwargs)
        else:
            raise ValueError(f"{task_type} is not supported.")

    def compute_metrics(self, clf, task_type):
        metrics = dict()

        # Obtain either R-squared metric or classification accuracy
        metrics["train_acc"] = clf.score(self.train_features, self.train_targets)
        metrics["test_acc"] = clf.score(self.test_features, self.test_targets)

        if (task_type == "categorization") or (task_type == "instance_categorization") or (task_type in ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8"]):
            # Categorization does not have Pearson's R metric.
            return metrics

        preds_train = clf.predict(self.train_features)
        preds_test = clf.predict(self.test_features)
        assert preds_train.shape == self.train_targets.shape
        assert preds_test.shape == self.test_targets.shape

        n_target_features = self.train_targets.shape[1]
        assert n_target_features == self.test_targets.shape[1]

        # Obtain a Pearson's R metric for each target feature
        metrics["train_r"]= list()
        metrics["test_r"] = list()
        for i in range(n_target_features):
            train_r = pearsonr(preds_train[:,i], self.train_targets[:,i])[0]
            test_r = pearsonr(preds_test[:,i], self.test_targets[:,i])[0]
            metrics["train_r"].append(train_r)
            metrics["test_r"].append(test_r)

        return metrics

    def _check_dims(self, features, targets):
        assert features.shape[0] == targets.shape[0]
        assert features.ndim == 2

    def _fit_categorization(self, instance_cat=False, cat=None, cls_kwargs=None):
        if cat is not None:
            cat_idx = np.where(self.stim_meta["category_index"] == cat)[0]
            new_train_features_idx = [i for i, e in enumerate(self.train_idx) if e in set(cat_idx)]
            new_train_idx = [e for e in self.train_idx if e in set(cat_idx)]
            # assert(len(new_train_idx) == 400)
            # assert(len(new_train_features_idx) == 400)
            self.train_features = self.train_features[new_train_features_idx]
            self.train_idx = new_train_idx
            new_test_features_idx = [i for i, e in enumerate(self.test_idx) if e in set(cat_idx)]
            new_test_idx = [e for e in self.test_idx if e in set(cat_idx)]
            # assert(len(new_test_idx) == 160)
            # assert(len(new_test_features_idx) == 160)
            self.test_features = self.test_features[new_test_features_idx]
            self.test_idx = new_test_idx
            targets = self.stim_meta["instance_index"]
            assert len(np.unique(targets)) == 64
        else:
            if instance_cat:
                targets = self.stim_meta["instance_index"]
                assert len(np.unique(targets)) == 64
            else:
                targets = self.stim_meta["category_index"]
                assert len(np.unique(targets)) == 8

        self.train_targets = targets[self.train_idx]
        self.test_targets = targets[self.test_idx]
        assert self.train_targets.ndim == 1 and self.test_targets.ndim == 1
        self._check_dims(self.train_features, self.train_targets)
        self._check_dims(self.test_features, self.test_targets)

        if cls_kwargs is None:
            parameters = {'C': SVM_CV_C_LONG}
            svc = LinearSVC()
            clf = GridSearchCV(svc, parameters)
        else:
            clf = LinearSVC(**cls_kwargs)

        clf.fit(self.train_features, self.train_targets)
        metrics = self.compute_metrics(clf, "categorization")

        if cls_kwargs is None:
            metrics["cls_kwargs"] = clf.best_params_
        else:
            metrics["cls_kwargs"] = cls_kwargs

        print(f"Metrics: {metrics}")
        return metrics

    def _fit_pose(self, cls_kwargs=None):
        targets_1 = np.expand_dims(self.stim_meta["rotation_xy"], axis=-1)
        targets_2 = np.expand_dims(self.stim_meta["rotation_xz"], axis=-1)
        targets_3 = np.expand_dims(self.stim_meta["rotation_yz"], axis=-1)
        assert targets_1.ndim == 2 and targets_2.ndim == 2 and targets_3.ndim == 2
        targets = np.concatenate((targets_1, targets_2, targets_3), axis=1)
        self.train_targets = targets[self.train_idx,:]
        self.test_targets = targets[self.test_idx,:]
        self._check_dims(self.train_features, self.train_targets)
        self._check_dims(self.test_features, self.test_targets)

        if cls_kwargs is None:
            ridge = Ridge()
            parameters = {"alpha": RIDGE_CV_ALPHA_LONG}
            clf = GridSearchCV(ridge, parameters)
        else:
            clf = Ridge(**cls_kwargs)

        clf.fit(self.train_features, self.train_targets)
        metrics = self.compute_metrics(clf, "pose")

        if cls_kwargs is None:
            metrics["cls_kwargs"] = clf.best_params_
        else:
            metrics["cls_kwargs"] = cls_kwargs

        print(f"Metrics: {metrics}")
        return metrics

    def _fit_position(self, cls_kwargs=None):
        targets_1 = np.expand_dims(self.stim_meta["translation_y"], axis=-1)
        targets_2 = np.expand_dims(self.stim_meta["translation_z"], axis=-1)
        assert targets_1.ndim == 2 and targets_2.ndim == 2
        targets = np.concatenate((targets_1, targets_2), axis=1)
        self.train_targets = targets[self.train_idx,:]
        self.test_targets = targets[self.test_idx,:]
        self._check_dims(self.train_features, self.train_targets)
        self._check_dims(self.test_features, self.test_targets)

        if cls_kwargs is None:
            ridge = Ridge()
            parameters = {"alpha": RIDGE_CV_ALPHA_LONG}
            clf = GridSearchCV(ridge, parameters)
        else:
            clf = Ridge(**cls_kwargs)

        clf.fit(self.train_features, self.train_targets)
        metrics = self.compute_metrics(clf, "position")

        if cls_kwargs is None:
            metrics["cls_kwargs"] = clf.best_params_
        else:
            metrics["cls_kwargs"] = cls_kwargs

        print(f"Metrics: {metrics}")
        return metrics

    def _fit_size(self, cls_kwargs=None):
        targets = self.stim_meta["size"]
        self.train_targets = np.expand_dims(targets[self.train_idx], axis=-1)
        self.test_targets = np.expand_dims(targets[self.test_idx], axis=-1)
        self._check_dims(self.train_features, self.train_targets)
        self._check_dims(self.test_features, self.test_targets)

        if cls_kwargs is None:
            ridge = Ridge()
            parameters = {"alpha": RIDGE_CV_ALPHA_LONG}
            clf = GridSearchCV(ridge, parameters)
        else:
            clf = Ridge(**cls_kwargs)

        clf.fit(self.train_features, self.train_targets)
        metrics = self.compute_metrics(clf, "size")

        if cls_kwargs is None:
            metrics["cls_kwargs"] = clf.best_params_
        else:
            metrics["cls_kwargs"] = cls_kwargs

        print(f"Metrics: {metrics}")
        return metrics

