import os
import pickle

import numpy as np

from collections import defaultdict
from sklearn.decomposition import PCA

from mouse_vision.models.model_paths import MODEL_PATHS
from mouse_vision.sklearn_transfer_datasets import BaseDataset
from mouse_vision.core.default_dirs import SKLEARN_TRANSFER_RESULTS_DIR
from mouse_vision.core.utils import get_base_arch_name
from mouse_vision.core.model_loader_utils import load_model
from mouse_vision.core.feature_extractor import (
    FeatureExtractor,
    CustomFeatureExtractor,
    get_layer_features,
)


__all__ = ["BaseFitter"]


def _grab_model(model_name):
    # model_name: string of model. could be untrained_{arch_name} or arch_name
    # For example: untrained_resnet18_64x64_input or resnet18_64x64_input

    model_family = "imagenet"
    if "cifar10" in model_name:
        model_family = "cifar10"

    arch_name, trained = get_base_arch_name(model_name)

    # Grab custom model path if applicable. Note that if the model_name is
    # "untrained_{arch_name}" it won't exist in MODEL_PATHS and therefore,
    # model_path will remain None, which is the desired behaviour (we don't
    # want to load weights for untrained models anyway).
    model_path = None
    if model_name in MODEL_PATHS.keys():
        assert trained
        model_path = MODEL_PATHS[model_name]

    # Load model
    try:
        print("Trying 'state_dict' key to load parameters")
        model, _ = load_model(
            arch_name,
            trained=trained,
            model_path=model_path,
            model_family=model_family,
            state_dict_key="state_dict",
        )
    except:
        print("'state_dict' key failed, using 'model_state_dict'")
        model, _ = load_model(
            arch_name,
            trained=trained,
            model_path=model_path,
            model_family=model_family,
            state_dict_key="model_state_dict",
        )

    return model, arch_name


class BaseFitter:
    def __init__(self, dataset, model_name, train_frac=None, num_train_test_splits=10):
        if not isinstance(dataset, list):
           assert isinstance(dataset, BaseDataset)
        else:
            for curr_d in dataset:
                assert isinstance(curr_d, BaseDataset)
        self.train_frac = train_frac
        self.num_train_test_splits = num_train_test_splits
        self.dataset = dataset

        # self.model_name: e.g., untrained_resnet18_64x64_input
        # self.arch_name: e.g., resnet18_64x64_input
        self.model_name = model_name
        self.model, self.arch_name = _grab_model(model_name)

    def get_features(self, layer_name):
        """
        This function can be reimplemented by a derived class if the user
        has multiple dataloaders for extracting image features (e.g.,
        dataloaders for train/test/val set images
        """
        print("Getting dataloader and feature extractor...")
        dataloader = self.dataset.get_dataloader()
        feature_extractor = self.get_feature_extractor(dataloader)

        print("Getting image features...")
        features = get_layer_features(
            feature_extractor, layer_name, self.model, self.arch_name
        )
        return features

    def get_feature_extractor(self, dataloader):
        # Dictionary based models
        if (
            "mousenet" in self.model_name
            or "alexnet_64x64_input_dict" in self.model_name
        ):
            feature_extractor = CustomFeatureExtractor(
                dataloader=dataloader, vectorize=True, debug=False
            )
        else:
            feature_extractor = FeatureExtractor(
                dataloader=dataloader, vectorize=True, debug=False
            )

        return feature_extractor

    def train_test_split(self, num_stimuli):
        """
        The implementation should take in a single argument called num_stimuli
        and return a list of dictionaries of length self.num_train_test_splits.
        Each dictionary is of the form {"train": train_indices, "test": test_indices}
        """
        raise NotImplementedError

    def do_pca(self, train_set, test_set):
        print("Performing PCA...")
        assert train_set.shape[1] == test_set.shape[1]
        n_components = 1000

        # If number of features is less than n_components, then do not do PCA.
        if n_components >= train_set.shape[1]:
            return train_set, test_set

        # Do PCA to project features in to n_components dimensions.
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_set)
        train_proj = pca.transform(train_set)
        test_proj = pca.transform(test_set)

        return train_proj, test_proj

    def _checker(self, features, labels, layer_name):
        if isinstance(features, list):
            for feat in features:
                print(f"Image features of {layer_name} are of dimensions {feat.shape}.")
                assert feat.ndim == 2
        else:
            print(f"Image features of {layer_name} are of dimensions {features.shape}.")
            assert features.ndim == 2

        # First, obtain the train and test set features,
        if isinstance(features, list):
            num_stimuli = features[0].shape[0]
            for feat in features:
                assert(num_stimuli == feat.shape[0])
            if labels is not None:
                assert isinstance(labels, list)
                for label in labels:
                    assert(label.shape[0] == num_stimuli)
        else:
            num_stimuli = features.shape[0]
            if labels is not None:
                assert(labels.shape[0] == num_stimuli)
        return num_stimuli

    def fit(self, task_type, layer_name):
        """
        Returns a metrics dict where the keys are the metrics and the values are
        a list of length self.num_train_test_splits.
        """
        features = self.get_features(layer_name)
        labels = None
        if isinstance(features, tuple):
            assert len(features) == 2 # features and labels
            features, labels = features

        num_stimuli = self._checker(features=features, labels=labels, layer_name=layer_name)

        splits = self.train_test_split(num_stimuli)

        # Check if the train indicies are identical across all splits.
        duplicate_train = all(
            set(sp["train"]) == set(splits[0]["train"]) for sp in splits
        )

        cls_kwargs = None
        metrics = defaultdict(list)
        for sp_idx, sp in enumerate(splits):
            print(f"Split {sp_idx+1}/{len(splits)}")

            train_idx = sp["train"]
            test_idx = sp["test"]
            if isinstance(features, list):
                train_set = features[sp_idx][train_idx, :]
                test_set = features[sp_idx][test_idx, :]
            else:
                train_set = features[train_idx, :]
                test_set = features[test_idx, :]

            # Second, perform PCA on the train set features and project both
            # sets of features into lower-dimensional space.
            train_set, test_set = self.do_pca(train_set, test_set)
            print(f"Train set projected features are of dimensions {train_set.shape}.")
            print(f"Test set projected features are of dimensions {test_set.shape}.")

            split_data = {
                "train_features": train_set,
                "train_idx": train_idx,
                "test_features": test_set,
                "test_idx": test_idx,
            }
            if labels is not None:
                split_data["train_labels"] = labels[sp_idx][train_idx] if isinstance(labels, list) else labels[train_idx]
                split_data["test_labels"] = labels[sp_idx][test_idx] if isinstance(labels, list) else labels[test_idx]

            # Finally, do fitting on the specific task type.
            if duplicate_train and sp_idx > 0:
                assert cls_kwargs is not None
                # If all the train indices are identical, as indicated by duplicate_train,
                # then avoid rerunning CV on the same train indices by supplying the
                # previous classifier/regression keyword arguments.
                metrics_sp = self.fit_task(task_type, split_data, cls_kwargs=cls_kwargs)
            else:
                metrics_sp = self.fit_task(task_type, split_data)
                cls_kwargs = metrics_sp["cls_kwargs"]

            # Record all the metrics
            for k in metrics_sp.keys():
                metrics[k].append(metrics_sp[k])

        self.save_metrics(task_type, metrics, layer_name)

    def fit_task(self, task_type, split_data, cls_kwargs=None):
        raise NotImplementedError

    def save_metrics(self, task_type, metrics, layer_name):
        if isinstance(self.dataset, list):
            dataset_name = self.dataset[0].get_name()
            for curr_d in self.dataset:
                assert(dataset_name == curr_d.get_name())
        else:
            dataset_name = self.dataset.get_name()

        layer_name = layer_name.replace('/', '_') # handle MouseNet layer names
        save_dir = os.path.join(
            SKLEARN_TRANSFER_RESULTS_DIR, f"{dataset_name}/{self.model_name}/{layer_name}"
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fname = os.path.join(save_dir, f"{task_type}.pkl")
        pickle.dump(metrics, open(fname, "wb"))
        print(f"Saved results to {fname}.")

