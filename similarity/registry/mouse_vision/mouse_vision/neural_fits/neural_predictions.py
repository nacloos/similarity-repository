import numpy as np
import torchvision.transforms as transforms

from scipy.stats import pearsonr
from collections import OrderedDict
from mouse_vision.models.model_layers import MODEL_LAYERS
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS

from mouse_vision.neural_data.model_data import generate_model_comparison_data
from mouse_vision.reliability.metrics import noise_estimation
from mouse_vision.neural_mappers.utils import generate_train_test_img_splits
from mouse_vision.neural_mappers.cross_validator import CrossValidator

from mouse_vision.core.default_dirs import NEUROPIX_DATA_PATH_WITH_RELS, CALCIUM_DATA_PATH_WITH_RELS
from mouse_vision.core.dataloader_utils import get_image_array_dataloader
from mouse_vision.core.feature_extractor import FeatureExtractor, CustomFeatureExtractor, get_layer_features
from mouse_vision.core.model_loader_utils import load_model
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.utils import open_dataset

class NeuralPredictions():
    """
    Arguments:
        arch_name         : (string) model architecture name
        visual_area       : (string) one of the mouse visual areas
        dataset_name      : (string) one of {neuropixels, calcium}
        map_type          : (string) one of {corr, pls, factored}
        map_cv_params     : (dict) contains hyperparameter values for cross-validation
        map_n_cv_splits   : (int) number of cross-validation splits
        num_splits        : (int) number of train/test splits
        train_frac        : (float) proportion of train samples to use
        trained           : (boolean) trained or untrained model
        model_params_path : (string or None) path to load model parameters from
    """
    def __init__(
        self,
        arch_name,
        visual_area,
        dataset_name,
        map_type="pls",
        map_cv_params=[{"map_type":"pls", "map_kwargs":{"n_components": 20, "scale": False}}],
        map_n_cv_splits=5,
        num_splits=10,
        train_frac=0.5,
        trained=True,
        model_params_path=None,
        model_family=None,
        state_dict_key="state_dict",
        center_trials=True,
        separate_by_animal=False,
        mode="spearman_brown_split_half"
    ):
        assert map_type in ["corr", "pls", "factored", "identity"]
        assert dataset_name in ["neuropixels", "calcium"]
        assert visual_area in map_cv_params.keys()

        assert arch_name in MODEL_LAYERS.keys(), f"{arch_name} not in {MODEL_LAYERS.keys()}."
        self.arch_name = arch_name
        self.model, self.model_layers = load_model(
            arch_name,
            trained=trained,
            model_path=model_params_path,
            model_family=model_family,
            state_dict_key=state_dict_key
        )

        self.visual_area = visual_area
        self.dataset_name = dataset_name
        self.center_trials = center_trials
        self.mode = mode
        self.separate_by_animal = separate_by_animal
        print(f"Cross-validation params: {map_cv_params[visual_area]}")
        self.cross_validator = CrossValidator(neural_map_str=map_type,
                                              cv_params=map_cv_params[visual_area],
                                              n_cv_splits=map_n_cv_splits,
                                              center_trials=center_trials,
                                              mode=mode)
        self.num_splits = num_splits
        self.train_frac = train_frac

        self.data, self.stimuli = generate_model_comparison_data(
            center_trials=self.center_trials,
            return_stimuli=True,
            dataset_name=self.dataset_name,
            separate_by_animal=self.separate_by_animal
        )

        assert self.visual_area in VISUAL_AREAS
        assert self.visual_area in self.data.keys(), f"{self.visual_area} not in {self.data.keys()}"

        self.stim_dataloader = self._get_stim_dataloader()
        # dictionary based models
        if "mousenet" in self.arch_name or "alexnet_64x64_input_dict" in self.arch_name:
            self.feature_extractor = CustomFeatureExtractor(
                dataloader=self.stim_dataloader,
                vectorize=True,
                debug=False
            )
        else:
            self.feature_extractor = FeatureExtractor(
                dataloader=self.stim_dataloader,
                vectorize=True,
                debug=False
            )

    def _get_stim_dataloader(self):
        assert self.stimuli.ndim == 4
        assert self.stimuli.shape[3] == 3 # RGB channels
        assert self.arch_name in MODEL_TRANSFORMS.keys(), f"{self.arch_name} does not exist."

        # Use the val transforms of the architecture/model, but prepend with ToPILImage() since
        # neural images are tensors instead of jpegs
        img_transforms_arr = [transforms.ToPILImage()] + MODEL_TRANSFORMS[self.arch_name]["val"]
        img_transforms = transforms.Compose(img_transforms_arr)
        n_stim = self.stimuli.shape[0]
        labels = np.arange(n_stim).astype(int)

        # Get stimuli dataloader
        stim_dataloader = get_image_array_dataloader(
            image_array=self.stimuli,
            labels=labels,
            img_transform=img_transforms,
            batch_size=256,
            num_workers=8,
            shuffle=False,
            pin_memory=True
        )

        return stim_dataloader

    def _dim_checker(self, area_resp):
        if self.center_trials:
            assert(area_resp.ndim == 2)
            assert(area_resp.dims[0] == "frame_id")
            assert(area_resp.dims[1] == "units")
        else:
            assert(area_resp.ndim == 3)
            assert(area_resp.dims[0] == "trials")
            assert(area_resp.dims[1] == "frame_id")
            assert(area_resp.dims[2] == "units")

    def fit(self, layer_names):
        if not isinstance(layer_names, list):
            layer_names = [layer_names]

        for layer_name in layer_names:
            assert layer_name in self.model_layers, f"{layer_name} not in {self.model_layers}"

        # Get model activations
        layer_features = []
        for layer_name in layer_names:
            curr_layer_features = get_layer_features(
                feature_extractor=self.feature_extractor,
                layer_name=layer_name,
                model=self.model,
                model_name=self.arch_name
            )
            print(f"Current layer: {layer_name}, Activations of shape: {curr_layer_features.shape}")
            layer_features.append(curr_layer_features)
        layer_features = np.concatenate(layer_features, axis=-1)
        print(f"Final concatenated activations of shape: {layer_features.shape}")

        # Get neural responses
        area_resp = self.data[self.visual_area]
        if self.separate_by_animal:
            assert(isinstance(area_resp, dict))
            for spec_name, spec_area_resp in area_resp.items():
                self._dim_checker(area_resp=spec_area_resp)
        else:
            self._dim_checker(area_resp=area_resp)

        train_test_splits = generate_train_test_img_splits(
            num_splits=self.num_splits,
            train_frac=self.train_frac
        )

        # Do fitting for each train / test split
        if self.separate_by_animal:
            n_neurons = OrderedDict()
            for spec_name, spec_area_resp in area_resp.items():
                if self.center_trials:
                    assert spec_area_resp.values.ndim == 2
                else:
                    assert spec_area_resp.values.ndim == 3
                n_neurons[spec_name] = spec_area_resp.values.shape[-1]
        else:
            if self.center_trials:
                assert area_resp.values.ndim == 2
            else:
                assert area_resp.values.ndim == 3
            n_neurons = area_resp.values.shape[-1]
        scores_per_split = list()
        best_params_per_split = list()
        for i, curr_split in enumerate(train_test_splits):
            layer_features_train = layer_features[curr_split["train"]]
            layer_features_test = layer_features[curr_split["test"]]

            # Average across trials
            if self.separate_by_animal:
                area_resp_train = OrderedDict()
                area_resp_test = OrderedDict()
                for spec_name, spec_area_resp in area_resp.items():
                    area_resp_train[spec_name] = spec_area_resp.isel(frame_id=curr_split["train"])
                    area_resp_test[spec_name] = spec_area_resp.isel(frame_id=curr_split["test"])
                    print(f"* Train: {layer_features_train.shape} -> Specimen {spec_name}: {area_resp_train[spec_name].shape}")
                    print(f"* Test: {layer_features_test.shape} -> Specimen {spec_name}: {area_resp_test[spec_name].shape}")
            else:
                area_resp_train = area_resp.isel(frame_id=curr_split["train"])
                area_resp_test = area_resp.isel(frame_id=curr_split["test"])

                print(f"* Train: {layer_features_train.shape} -> {area_resp_train.shape}")
                print(f"* Test: {layer_features_test.shape} -> {area_resp_test.shape}")

            self.cross_validator.fit(layer_features_train, area_resp_train)
            if self.center_trials:
                preds = self.cross_validator.predict(layer_features_test)

                if self.cross_validator.neural_mapper is not None:
                    # Fitting successful
                    # preds.shape[1] could be different than area_resp_test.shape[1] in the case
                    # where we wanted to do an identity map for the RSA metric.
                    scores = self.cross_validator.neural_mapper.score(preds, area_resp_test,
                                                                      spec_score_agg_func=None)
                else:
                    # Sanity check since it doesn't make sense to have scores for each
                    # target neuron for an identity neural mapper
                    assert self.cross_validator.neural_map_str != "identity"
                    if self.separate_by_animal:
                        scores = OrderedDict()
                        for spec_name, spec_area_resp in area_resp_test.items():
                            assert preds[spec_name].shape[-1] == spec_area_resp.shape[-1]
                            scores[spec_name] = np.zeros((n_neurons[spec_name],)) + np.NaN
                    else:
                        assert preds.shape[-1] == area_resp_test.shape[-1]
                        scores = np.zeros((n_neurons,)) + np.NaN
            else:
                if self.separate_by_animal:
                    scores = OrderedDict()
                    for spec_name, spec_area_resp in area_resp.items():
                        scores[spec_name] = noise_estimation(target_N=spec_area_resp,
                                                            source_N=layer_features,
                                                            source_map_kwargs=self.cross_validator.best_params,
                                                            train_img_idx=curr_split["train"], test_img_idx=curr_split["test"],
                                                            num_source_units=None,
                                                            parallelize_per_target_unit=False,
                                                            metric="pearsonr" if self.cross_validator.neural_map_str != "identity" else "rsa",
                                                            mode=self.mode,
                                                            center=np.nanmean,
                                                            summary_center="raw",
                                                            sync=True,
                                                            n_jobs=5
                                                            )
                else:
                    scores = noise_estimation(target_N=area_resp,
                                                source_N=layer_features,
                                                source_map_kwargs=self.cross_validator.best_params,
                                                train_img_idx=curr_split["train"], test_img_idx=curr_split["test"],
                                                num_source_units=None,
                                                parallelize_per_target_unit=False,
                                                metric="pearsonr" if self.cross_validator.neural_map_str != "identity" else "rsa",
                                                mode=self.mode,
                                                center=np.nanmean,
                                                summary_center="raw",
                                                sync=True,
                                                n_jobs=5
                                                )

            scores_per_split.append(scores)
            best_params_per_split.append(self.cross_validator.best_params)

        scores_per_split = np.array(scores_per_split)
        return scores_per_split, best_params_per_split


