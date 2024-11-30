import torch.nn as nn
import torchvision.transforms as transforms

from mouse_vision.sklearn_transfer_datasets import HvmFitter
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS

HVM_SPLIT_FILE = "/mnt/fs5/nclkong/allen_inst/hvm_data/test_hvm_splits.npz"


# Model: "simplified_mousenet_depth_hour_glass_six_stream"
model_name = "simplified_mousenet_depth_hour_glass_six_stream"
bf = HvmFitter(model_name, HVM_SPLIT_FILE)
#data = bf.fit("categorization", "VISpor")
#data = bf.fit("size", "VISpor")
#data = bf.fit("pose", "VISpor")
data = bf.fit("position", "VISrl")


# Model: "simplified_mousenet_single_stream"
model_name = "simplified_mousenet_single_stream"
bf = HvmFitter(model_name, HVM_SPLIT_FILE)
#data = bf.fit("categorization", "VISpor")
#data = bf.fit("size", "VISpor")
#data = bf.fit("pose", "VISpor")
data = bf.fit("position", "VISpor")


# Model: "untrained_alexnet_64x64_input_pool_6"
model_name = "untrained_alexnet_64x64_input_pool_6"
bf = HvmFitter(model_name, HVM_SPLIT_FILE)
data = bf.fit("categorization", "classifier.5")


# Model: "alexnet_64x64_input_pool_6"
model_name = "alexnet_64x64_input_pool_6"
bf = HvmFitter(model_name, HVM_SPLIT_FILE)
data = bf.fit("categorization", "classifier.5")

