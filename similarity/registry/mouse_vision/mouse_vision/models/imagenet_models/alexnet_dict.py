import torch.nn as nn

__all__ = ["AlexNetDict", "alexnet_64x64_input_dict"]


class AlexNetDict(nn.Module):
    """
    Implementation of AlexNet using ModuleDict
    """

    def __init__(self, num_classes=1000, pool_size=6, drop_final_fc=False):
        super(AlexNetDict, self).__init__()

        self.alexnet_params = {
            "input_pool1": {
                "kernel_size": 11,
                "num_channels": 64,
                "stride": 4,
                "padding": 2,
            },
            "pool1_pool2": {
                "kernel_size": 5,
                "num_channels": 192,
                "stride": 1,
                "padding": 2,
            },
            "pool2_conv3": {
                "kernel_size": 3,
                "num_channels": 384,
                "stride": 1,
                "padding": 1,
            },
            "conv3_conv4": {
                "kernel_size": 3,
                "num_channels": 256,
                "stride": 1,
                "padding": 1,
            },
            "conv4_pool5": {
                "kernel_size": 3,
                "num_channels": 256,
                "stride": 1,
                "padding": 1,
            },
        }
        if drop_final_fc:
            self.alexnet_layers = [
                "input_pool1",
                "pool1_pool2",
                "pool2_conv3",
                "conv3_conv4",
                "conv4_pool5",
                "pool5_fc1",
                "fc1_fc2",
            ]
            self.output_layer = "fc2"
        else:
            self.alexnet_layers = [
                "input_pool1",
                "pool1_pool2",
                "pool2_conv3",
                "conv3_conv4",
                "conv4_pool5",
                "pool5_fc1",
                "fc1_fc2",
                "fc2_fc3",
            ]
            self.output_layer = "fc3"

        # self.layers will be a dictionary populated with activations mouse_vision/
        # core/feature_extractor.py: CustomFeatureExtractor() uses this convention
        # self.layer_names will contain a list of modules (i.e., representative layers)
        # of alexnet: ["pool1", "pool2", "conv3", "conv4", "pool5", "fc1", "fc2", "fc3"]
        self.layers = None
        self.layer_names = None

        # Set up model layers dictionary
        self.model_layers = nn.ModuleDict()
        in_channels = 3
        for layer in self.alexnet_layers:
            if "fc" not in layer:
                kernel_size = self.alexnet_params[layer]["kernel_size"]
                out_channels = self.alexnet_params[layer]["num_channels"]
                stride = self.alexnet_params[layer]["stride"]
                padding = self.alexnet_params[layer]["padding"]

            if "pool" in layer.split("_")[-1]:
                self.model_layers[layer] = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(inplace=True),
                    # All pooling in alexnet has kernel_size=3 and stride=2
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
            elif "conv" in layer.split("_")[-1]:
                self.model_layers[layer] = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(inplace=True),
                )
            elif layer == "pool5_fc1":
                self.model_layers[layer] = nn.Sequential(
                    nn.AdaptiveAvgPool2d((pool_size, pool_size)),
                    nn.Flatten(start_dim=1),
                    nn.Dropout(),
                    nn.Linear(256 * pool_size * pool_size, 4096),
                    nn.ReLU(inplace=True),
                )
            elif layer == "fc1_fc2":
                self.model_layers[layer] = nn.Sequential(
                    nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True)
                )
            elif layer == "fc2_fc3":
                self.model_layers[layer] = nn.Sequential(nn.Linear(4096, num_classes))
            else:
                raise ValueError(f"{layer} undefined.")

            in_channels = out_channels

    # Similarly, CustomFeatureExtractor requires a _layer_features() function which
    # computes the outputs from each desired module.
    def _layer_features(self, x):
        self.layer_names = list()
        self.layers = dict()
        for layer in self.alexnet_layers:
            source_target_pair = layer.split("_")
            assert len(source_target_pair) == 2
            source = source_target_pair[0]
            target = source_target_pair[1]
            if source == "input":
                self.layers[target] = self.model_layers[layer](x)
            else:
                assert source in self.layers.keys()
                self.layers[target] = self.model_layers[layer](self.layers[source])

            self.layer_names.append(target)

        assert self.output_layer in self.layers.keys()
        return self.layers[self.output_layer]

    def forward(self, x):
        x = self._layer_features(x)
        return x


def alexnet_64x64_input_dict(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Implemented using module dictionaries and stores features using a dictionary
    which is an attribute of the model. Takes in 64x64 inputs.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetDict(**kwargs)
    return model


if __name__ == "__main__":
    import numpy as np
    import torch

    from alexnet import alexnet

    from mouse_vision.models.model_layers import MODEL_LAYERS
    from mouse_vision.core.feature_extractor import FeatureExtractor, get_layer_features
    from mouse_vision.core.dataloader_utils import get_image_array_dataloader

    # Define inputs
    torch.manual_seed(0)
    x = torch.rand(8, 3, 224, 224)

    # Define original alexnet
    torch.manual_seed(0)
    orig_alexnet = alexnet()
    orig_alexnet.eval()

    # Define my alexnet
    torch.manual_seed(0)
    my_alexnet = alexnet_64x64_input_dict()
    my_alexnet.eval()

    # Obtain outputs from original and my alexnet
    with torch.no_grad():
        orig_output = orig_alexnet(x)
        my_output = my_alexnet(x)

    # Are the outputs from both models the same?
    print(f"Same outputs: {torch.equal(orig_output, my_output)}")

    # Are the outputs from representative layers from both models the same?
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    dataloader = get_image_array_dataloader(x, torch.ones(x.shape[0]), Identity())
    fe = FeatureExtractor(
        dataloader=dataloader, n_batches=None, vectorize=False, debug=True
    )
    for i, layer_name in enumerate(MODEL_LAYERS["alexnet"]):
        features = get_layer_features(
            feature_extractor=fe,
            layer_name=layer_name,
            model=orig_alexnet,
            model_name="alexnet",
        )
        my_layer_output = my_alexnet.layers[my_alexnet.layer_names[i]]

        print(
            f"Layer {my_alexnet.layer_names[i]}, {layer_name}, features "
            f"identical: {np.allclose(features, my_layer_output.numpy(), rtol=1e-2, atol=1e-3)}"
        )
        print(features.shape, my_layer_output.cpu().numpy().shape)
        print(f"Max difference: {np.max(my_layer_output.numpy() - features)}")
        print(f"Min difference: {np.min(my_layer_output.numpy() - features)}")
        print(f"Average difference: {np.mean(my_layer_output.numpy() - features)}")
