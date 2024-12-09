import os

from mouse_vision.core.default_dirs import MODEL_SAVE_DIR
from mouse_vision.models.model_layers import MODEL_LAYERS

MODEL_PATHS = {
    "shi_mousenet_ir": os.path.join(MODEL_SAVE_DIR, "shi_mousenet_ir.pt"),
    "shi_mousenet_vispor5_ir": os.path.join(
        MODEL_SAVE_DIR,
        "shi_mousenet_vispor5_ir.pt",
    ),
    "alexnet_bn_ir_64x64_input_pool_6": os.path.join(
        MODEL_SAVE_DIR,
        "alexnet_bn_ir.pt",
    ),
    "simplified_mousenet_dual_stream_visp_3x3_ir": os.path.join(
        MODEL_SAVE_DIR,
        "dual_stream_ir.pt",
    ),
    "simplified_mousenet_six_stream_visp_3x3_simclr": os.path.join(
        MODEL_SAVE_DIR,
        "six_stream_simclr.pt",
    ),
}

for model in MODEL_PATHS.keys():
    assert model in MODEL_LAYERS.keys(), f"{model} not in model_layers.py"
