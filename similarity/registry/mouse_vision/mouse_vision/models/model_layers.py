"""
PyTorch modules for each model architecture. These are the layers that we will
extract activations from for downstream analyses.
"""

MODEL_LAYERS = {
    "alexnet_bn": ["features.3"]
    + ["features.7"]
    + ["features.10"]
    + ["features.13"]
    + ["features.17"]
    + ["classifier.3"]
    + ["classifier.7"],
    "shi_mousenet": [
        "LGNv",
        "VISp4",
        "VISp2/3",
        "VISp5",
        "VISal4",
        "VISal2/3",
        "VISal5",
        "VISpl4",
        "VISpl2/3",
        "VISpl5",
        "VISli4",
        "VISli2/3",
        "VISli5",
        "VISrl4",
        "VISrl2/3",
        "VISrl5",
        "VISl4",
        "VISl2/3",
        "VISl5",
        "VISpor4",
        "VISpor2/3",
        "VISpor5",
    ],
    "simplified_mousenet_six_stream": [
        "VISp",
        "VISal",
        "VISli",
        "VISl",
        "VISrl",
        "VISpl",
        "VISpm",
        "VISpor",
        "VISam",
    ],
    "simplified_mousenet_dual_stream": ["VISp", "ventral", "dorsal", "VISpor", "VISam"],
}

MODEL_LAYERS["alexnet_bn_ir_64x64_input_pool_6"] = MODEL_LAYERS["alexnet_bn"]

MODEL_LAYERS["shi_mousenet_vispor5_ir"] = MODEL_LAYERS["shi_mousenet"]
MODEL_LAYERS["shi_mousenet_ir"] = MODEL_LAYERS["shi_mousenet"]

MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_simclr"] = MODEL_LAYERS[
    "simplified_mousenet_six_stream"
]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir"] = MODEL_LAYERS[
    "simplified_mousenet_dual_stream"
]

if __name__ == "__main__":
    import mouse_vision.models.imagenet_models as im

    def assert_module_exists(model, layer_name):
        module = model
        for p in layer_name.split("."):
            module = module._modules.get(p)
            assert (
                module is not None
            ), f"No submodule found for layer {layer_name}, at part {p}."

    for model_name in MODEL_LAYERS.keys():
        if "mousenet" in model_name or model_name == "alexnet_64x64_input_dict":
            continue

        print(model_name)
        model = im.__dict__[model_name](pretrained=False)
        layer_names = MODEL_LAYERS[model_name]
        if layer_names == []:
            continue
        else:
            for layer_name in layer_names:
                assert_module_exists(model, layer_name)
