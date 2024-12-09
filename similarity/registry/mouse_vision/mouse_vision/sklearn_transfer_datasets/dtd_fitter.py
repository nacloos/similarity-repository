import numpy as np
from torchvision import transforms
from mouse_vision.core.constants import SVM_CV_C, SVM_CV_C_LONG
from mouse_vision.core.utils import get_base_arch_name
from mouse_vision.sklearn_transfer_datasets.base_fitter import BaseFitter
from mouse_vision.sklearn_transfer_datasets.dtd_dataset import DTDataloader
from mouse_vision.model_training.trainer_transforms import TRAINER_TRANSFORMS
from mouse_vision.core.feature_extractor import get_layer_features
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

__all__ = ["DTDFitter"]

class DTDFitter(BaseFitter):
    def __init__(self, model_name, split_idxs=range(1, 11), C_vals=SVM_CV_C_LONG):
        arch_name, _ = get_base_arch_name(model_name)
        img_transforms = TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"]["val"]
        image_transforms = transforms.Compose(img_transforms)
        dataset = [DTDataloader(split_num=split_idx, image_transforms=image_transforms) for split_idx in split_idxs]
        super(DTDFitter, self).__init__(dataset=dataset, model_name=model_name)
        self.split_idxs = split_idxs
        self.num_train_test_splits = len(self.split_idxs)
        self.C_vals = C_vals
        self.splits = None

    def get_features(self, layer_name):
        """
        This function loops through the the dataloader splits and extracts features for each split
        """
        print("Getting dataloader and feature extractor...")
        self.splits = list()
        features_list = list()
        labels_list = list()
        for curr_d in self.dataset:
            curr_dataloader = curr_d.get_dataloader()
            assert(set(list(curr_dataloader.keys())) == set(["train", "val", "test"]))
            train_idx_end = 0
            test_idx_start = 0
            test_idx_end = 0
            features_split = list()
            labels_split = list()
            for mode in ["train", "val", "test"]: # train, val, test
                feature_extractor = self.get_feature_extractor(curr_dataloader[mode])
                features, labels = get_layer_features(
                    feature_extractor=feature_extractor,
                    layer_name=layer_name,
                    model=self.model,
                    model_name=self.arch_name,
                    return_labels=True
                )
                assert(features.shape[0] == labels.shape[0])
                if mode in ["train", "val"]:
                    train_idx_end += labels.shape[0]
                else:
                    assert mode == "test"
                    test_idx_start = train_idx_end
                    test_idx_end = test_idx_start + labels.shape[0]
                features_split.append(features)
                labels_split.append(labels)
            curr_sp = {"train": np.arange(0, train_idx_end), "test": np.arange(test_idx_start, test_idx_end)}
            self.splits.append(curr_sp)
            features_split = np.concatenate(features_split, axis=0)
            features_list.append(features_split)
            labels_split = np.concatenate(labels_split, axis=0)
            labels_list.append(labels_split)
        return features_list, labels_list

    def train_test_split(self, num_stimuli):
        """
        This function returns a list of dictionaries of length self.num_train_test_splits.
        Each dictionary is of the form {"train": train_indices, "test": test_indices}
        """
        assert self.splits is not None
        assert(len(self.splits) == self.num_train_test_splits)
        for curr_sp in self.splits:
            assert(curr_sp["test"][-1] + 1 == num_stimuli)
        return self.splits


    def fit_task(self, task_type, split_data, cls_kwargs=None):
        assert(task_type == "categorization")
        metrics = {}
        if cls_kwargs is None:
            parameters = {'C': self.C_vals}
            svc = LinearSVC()
            clf = GridSearchCV(svc, parameters)
        else:
            clf = LinearSVC(**cls_kwargs)
            metrics["cls_kwargs"] = cls_kwargs

        clf.fit(X=split_data["train_features"], y=split_data["train_labels"])
        metrics["test_acc"] = clf.score(X=split_data["test_features"], y=split_data["test_labels"])
        metrics["train_acc"] = clf.score(X=split_data["train_features"], y=split_data["train_labels"])
        if cls_kwargs is None:
            metrics["cls_kwargs"] = clf.best_params_
        return metrics
