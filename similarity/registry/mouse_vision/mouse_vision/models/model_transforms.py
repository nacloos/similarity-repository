"""
Contains information on image mean and standard deviation for RGB channels
that models were trained on. Will also contain necessary padding for doing
neural prediction. For example, inception_v3 needs to be padded up to
299x299 input.

The format is as follows: each model name is associated with a dictionary
that has "mean", "std", "input_size" as keys.  "input_size" is the image size
the model was trained on.  Padding comes into play when doing neural fits
(e.g., a stimulus size of 40x40 may need to be padded up to 224x224).
"""
from mouse_vision.model_training.trainer_transforms import TRAINER_TRANSFORMS

MODEL_TRANSFORMS = {
    "alexnet": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "alexnet_64x64_rl_scratch_truncated": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "alexnet_two_64x64": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_three_64x64": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_four_64x64": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_five_64x64": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_six_64x64": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_rotnet": TRAINER_TRANSFORMS["RotNetTrainer"],
    "alexnet_rotnet_transfer": TRAINER_TRANSFORMS["RotNetTrainer"],
    "alexnet_64x64_input_pool_6": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "alexnet_64x64_input_pool_6_cifar10": TRAINER_TRANSFORMS[
        "SupervisedCIFAR10Trainer_64x64"
    ],
    "alexnet_bn_simsiam_64x64": TRAINER_TRANSFORMS["SimSiamTrainer_64x64"],
    "alexnet_bn_simclr_64x64": TRAINER_TRANSFORMS["SimCLRTrainer_64x64"],
    "alexnet_bn_mocov2_64x64": TRAINER_TRANSFORMS["MoCov2Trainer_64x64"],
    "alexnet_ir_dmlocomotion": TRAINER_TRANSFORMS[
        "DMLocomotionInstanceDiscriminationTrainer"
    ],
    "alexnet_ir_64x64_input_pool_6": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_64x64"
    ],
    "alexnet_ir_84x84": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_84x84"],
    "alexnet_ir_104x104": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_104x104"],
    "alexnet_ir_124x124": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_124x124"],
    "alexnet_ir_144x144": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_144x144"],
    "alexnet_ir_164x164": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_164x164"],
    "alexnet_ir_184x184": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_184x184"],
    "alexnet_ir_204x204": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_204x204"],
    "alexnet_ir_224x224": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer"],
    "alexnet_bn_barlow_twins_64x64": TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"],
    "alexnet_bn_barlow_twins": TRAINER_TRANSFORMS["BarlowTwinsTrainer"],
    "alexnet_bn_vicreg_64x64": TRAINER_TRANSFORMS["VICRegTrainer_64x64"],
    "vgg11": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "vgg13": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "vgg16": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "vgg16_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "vgg16_64x64_input_cifar10": TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer_64x64"],
    "vgg16_ir_64x64": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"],
    "vgg19": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet18_cifar10": TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer"],
    "resnet18": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet18_ir": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer"],
    "resnet18_ir_64x64": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"],
    "resnet18_simclr": TRAINER_TRANSFORMS["SimCLRTrainer"],
    "resnet18_simsiam": TRAINER_TRANSFORMS["SimSiamTrainer"],
    "resnet18_relative_location": TRAINER_TRANSFORMS["RelativeLocationTrainer"],
    "resnet18_mocov2": TRAINER_TRANSFORMS["MoCov2Trainer"],
    "resnet18_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "resnet18_64x64_input_cifar10": TRAINER_TRANSFORMS[
        "SupervisedCIFAR10Trainer_64x64"
    ],
    "resnet34": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet34_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "resnet50": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet50_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "resnet50_relative_location": TRAINER_TRANSFORMS["RelativeLocationTrainer"],
    "resnet50_mocov2": TRAINER_TRANSFORMS["MoCov2Trainer"],
    "resnet50_barlow_twins": TRAINER_TRANSFORMS["BarlowTwinsTrainer"],
    "resnet101": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet101_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "resnet152": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "resnet152_64x64_input": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "wide_resnet50_2": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "wide_resnet101_2": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "squeezenet1_0": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "squeezenet1_1": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "densenet121": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "densenet161": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "densenet169": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "densenet201": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "googlenet": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],  # Inception v1
    "inception_v3": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_inception"],
    "shufflenet_v2_x0_5": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "shufflenet_v2_x1_0": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "mobilenet_v2": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "mnasnet0_5": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "mnasnet1_0": TRAINER_TRANSFORMS["SupervisedImageNetTrainer"],
    "xception": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_xception"],
    "nasnetamobile": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_nasnetamobile"],
    "shi_mousenet": TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"],
    "shi_mousenet_cifar10": TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer_64x64"],
    "shi_mousenet_ir": TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"],
    "simplified_mousenet_six_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_six_stream_cifar10": TRAINER_TRANSFORMS[
        "SupervisedCIFAR10Trainer_64x64"
    ],
    "simplified_mousenet_dual_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_dual_stream_cifar10": TRAINER_TRANSFORMS[
        "SupervisedCIFAR10Trainer_64x64"
    ],
    "simplified_mousenet_single_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_single_stream_cifar10": TRAINER_TRANSFORMS[
        "SupervisedCIFAR10Trainer_64x64"
    ],
    "simplified_mousenet_single_stream_ir": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_64x64"
    ],
    "simplified_mousenet_single_stream_ir_224x224": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer"
    ],
    "simplified_mousenet_single_stream_simclr": TRAINER_TRANSFORMS[
        "SimCLRTrainer_64x64"
    ],
    "simplified_mousenet_single_stream_rotnet": TRAINER_TRANSFORMS[
        "RotNetTrainer_64x64"
    ],
    "simplified_mousenet_single_stream_mocov2": TRAINER_TRANSFORMS[
        "MoCov2Trainer_64x64"
    ],
    "simplified_mousenet_single_stream_simsiam": TRAINER_TRANSFORMS[
        "SimSiamTrainer_64x64"
    ],
    "simplified_mousenet_ae_single_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_depth_hour_glass_single_stream": TRAINER_TRANSFORMS[
        "DepthPredictionTrainer_64x64"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_32x32": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_32x32"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_44x44": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_44x44"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_64x64"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_84x84": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_84x84"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_104x104": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_104x104"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_124x124": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_124x124"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_144x144": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_144x144"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_164x164": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_164x164"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_184x184": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_184x184"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_204x204": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_204x204"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_ir_224x224": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_simclr": TRAINER_TRANSFORMS[
        "SimCLRTrainer_64x64"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_rotnet": TRAINER_TRANSFORMS[
        "RotNetTrainer_64x64"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_mocov2": TRAINER_TRANSFORMS[
        "MoCov2Trainer_64x64"
    ],
    "simplified_mousenet_dual_stream_visp_3x3_simsiam": TRAINER_TRANSFORMS[
        "SimSiamTrainer_64x64"
    ],
    "simplified_mousenet_ae_dual_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_depth_hour_glass_dual_stream": TRAINER_TRANSFORMS[
        "DepthPredictionTrainer_64x64"
    ],
    "simplified_mousenet_six_stream_visp_3x3_ir": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer_64x64"
    ],
    "simplified_mousenet_six_stream_visp_3x3_ir_224x224": TRAINER_TRANSFORMS[
        "InstanceDiscriminationTrainer"
    ],
    "simplified_mousenet_six_stream_visp_3x3_simclr": TRAINER_TRANSFORMS[
        "SimCLRTrainer_64x64"
    ],
    "simplified_mousenet_six_stream_visp_3x3_rotnet": TRAINER_TRANSFORMS[
        "RotNetTrainer_64x64"
    ],
    "simplified_mousenet_six_stream_visp_3x3_mocov2": TRAINER_TRANSFORMS[
        "MoCov2Trainer_64x64"
    ],
    "simplified_mousenet_six_stream_visp_3x3_simsiam": TRAINER_TRANSFORMS[
        "SimSiamTrainer_64x64"
    ],
    "simplified_mousenet_ae_six_stream": TRAINER_TRANSFORMS[
        "SupervisedImageNetTrainer_64x64"
    ],
    "simplified_mousenet_depth_hour_glass_six_stream": TRAINER_TRANSFORMS[
        "DepthPredictionTrainer_64x64"
    ],
}

# AlexNet variants
MODEL_TRANSFORMS["alexnet_64x64_input_pool_1"] = MODEL_TRANSFORMS[
    "alexnet_64x64_input_pool_6"
]
MODEL_TRANSFORMS["alexnet_64x64_input_dict"] = MODEL_TRANSFORMS[
    "alexnet_64x64_input_pool_6"
]
MODEL_TRANSFORMS["alexnet_64x64_input_pool_1_cifar10"] = MODEL_TRANSFORMS[
    "alexnet_64x64_input_pool_6_cifar10"
]
MODEL_TRANSFORMS["alexnet_64x64_input_pool_6_with_ir_transforms"] = MODEL_TRANSFORMS[
    "alexnet_ir_64x64_input_pool_6"
]
MODEL_TRANSFORMS["alexnet_bn_64x64_input_pool_6_with_ir_transforms"] = MODEL_TRANSFORMS[
    "alexnet_ir_64x64_input_pool_6"
]
MODEL_TRANSFORMS["alexnet_bn_ir_64x64_input_pool_6"] = MODEL_TRANSFORMS[
    "alexnet_ir_64x64_input_pool_6"
]

# deeper resnets
MODEL_TRANSFORMS["resnet34_ir_64x64"] = MODEL_TRANSFORMS["resnet18_ir_64x64"]
MODEL_TRANSFORMS["resnet50_ir_64x64"] = MODEL_TRANSFORMS["resnet18_ir_64x64"]
MODEL_TRANSFORMS["resnet101_ir_64x64"] = MODEL_TRANSFORMS["resnet18_ir_64x64"]
MODEL_TRANSFORMS["resnet152_ir_64x64"] = MODEL_TRANSFORMS["resnet18_ir_64x64"]
# variants
MODEL_TRANSFORMS["resnet18_ir_preavgpool_64x64"] = MODEL_TRANSFORMS["resnet18_ir_64x64"]
MODEL_TRANSFORMS["resnet34_ir_preavgpool_64x64"] = MODEL_TRANSFORMS["resnet34_ir_64x64"]
MODEL_TRANSFORMS["resnet50_ir_preavgpool_64x64"] = MODEL_TRANSFORMS["resnet50_ir_64x64"]
MODEL_TRANSFORMS["resnet101_ir_preavgpool_64x64"] = MODEL_TRANSFORMS[
    "resnet101_ir_64x64"
]
MODEL_TRANSFORMS["resnet152_ir_preavgpool_64x64"] = MODEL_TRANSFORMS[
    "resnet152_ir_64x64"
]

# Shi MouseNet variants
MODEL_TRANSFORMS["shi_mousenet_vispor5"] = MODEL_TRANSFORMS["shi_mousenet"]
MODEL_TRANSFORMS["shi_mousenet_vispor5_pool_4"] = MODEL_TRANSFORMS["shi_mousenet"]

MODEL_TRANSFORMS["shi_mousenet_vispor5_cifar10"] = MODEL_TRANSFORMS[
    "shi_mousenet_cifar10"
]
MODEL_TRANSFORMS["shi_mousenet_vispor5_ir"] = MODEL_TRANSFORMS["shi_mousenet_ir"]

# Parallel Stream MouseNet variants
MODEL_TRANSFORMS["simplified_mousenet_single_stream_rand"] = MODEL_TRANSFORMS[
    "simplified_mousenet_single_stream"
]

MODEL_TRANSFORMS["simplified_mousenet_six_stream_visp_3x3"] = MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_six_stream_visp_3x3_bn"] = MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_six_stream_visp_3x3_bn_rand"] = MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_six_stream_vispor_only"] = MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream"
]
MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream_vispor_only_visp_3x3"
] = MODEL_TRANSFORMS["simplified_mousenet_six_stream"]

MODEL_TRANSFORMS["simplified_mousenet_dual_stream_visp_3x3"] = MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_dual_stream_visp_3x3_bn"] = MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_dual_stream_visp_3x3_bn_rand"] = MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream"
]
MODEL_TRANSFORMS["simplified_mousenet_dual_stream_vispor_only"] = MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream"
]
MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream_vispor_only_visp_3x3"
] = MODEL_TRANSFORMS["simplified_mousenet_dual_stream"]

MODEL_TRANSFORMS["simplified_mousenet_six_stream_visp_3x3_cifar10"] = MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream_cifar10"
]
MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream_visp_3x3_bn_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_six_stream_cifar10"]
MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream_vispor_only_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_six_stream_cifar10"]
MODEL_TRANSFORMS[
    "simplified_mousenet_six_stream_vispor_only_visp_3x3_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_six_stream_cifar10"]

MODEL_TRANSFORMS["simplified_mousenet_dual_stream_visp_3x3_cifar10"] = MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream_cifar10"
]
MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream_visp_3x3_bn_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_dual_stream_cifar10"]
MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream_vispor_only_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_dual_stream_cifar10"]
MODEL_TRANSFORMS[
    "simplified_mousenet_dual_stream_vispor_only_visp_3x3_cifar10"
] = MODEL_TRANSFORMS["simplified_mousenet_dual_stream_cifar10"]

# SimCLR variants
MODEL_TRANSFORMS["resnet18_simclr_nosyncbn"] = MODEL_TRANSFORMS["resnet18_simclr"]
