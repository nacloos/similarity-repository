import os

from mouse_vision.core.utils import open_dataset
from mouse_vision.neural_fits.perform_neural_fits import perform_neural_fits
from mouse_vision.core.default_dirs import MODEL_FEATURES_SAVE_DIR
from mouse_vision.reliability.spec_map_utils import generate_spec_map

#=================================
print("Reading data...")
visual_area_spec_map = generate_spec_map(
    dataset_name="calcium", equalize_source_units=False
)
visual_area_spec_map = visual_area_spec_map["VISam"]

print(visual_area_spec_map.keys())

#=================================
print("Retrieving model features...")
model_name = "alexnet_64x64_input_pool_6"
model_features_path = os.path.join(
    MODEL_FEATURES_SAVE_DIR, "calcium", f"{model_name}.pkl"
)

model_features = open_dataset(model_features_path)
model_features = model_features["features.7"]

#=================================
print("Doing neural fits...")
map_kwargs = {"map_type": "pls", "map_kwargs": {"n_components": 25}}
metric = "pearsonr"

results = perform_neural_fits(
    "features.7",
    model_features,
    visual_area_spec_map,
    map_kwargs,
    num_train_test_splits=1,
    train_frac=0.5,
    metric=metric,
    correction="spearman_brown_split_half_denominator",
    results_dir="./"
)

for spec in results.keys():
    print(f"Specimen {spec}: {results[spec].shape}")

