import torch
import torch.nn as nn
from mouse_vision.loss_functions.loss_utils import build_norm_layer
import numpy as np
import copy

__all__ = [
    "simplified_mousenet_six_stream",
    "simplified_mousenet_six_stream_visp_3x3",
    "simplified_mousenet_six_stream_vispor_only",
    "simplified_mousenet_six_stream_vispor_only_visp_3x3",
    "simplified_mousenet_dual_stream",
    "simplified_mousenet_dual_stream_visp_3x3",
    "simplified_mousenet_dual_stream_vispor_only",
    "simplified_mousenet_dual_stream_vispor_only_visp_3x3",
    "simplified_mousenet_single_stream",
    "simplified_mousenet_single_stream_ir",
    "simplified_mousenet_single_stream_rotnet",
    "simplified_mousenet_single_stream_mocov2",
    "simplified_mousenet_single_stream_simclr",
    "simplified_mousenet_single_stream_simsiam",
    "simplified_mousenet_dual_stream_visp_3x3_bn",
    "simplified_mousenet_dual_stream_visp_3x3_ir",
    "simplified_mousenet_dual_stream_visp_3x3_rotnet",
    "simplified_mousenet_dual_stream_visp_3x3_mocov2",
    "simplified_mousenet_dual_stream_visp_3x3_simclr",
    "simplified_mousenet_dual_stream_visp_3x3_simsiam",
    "simplified_mousenet_six_stream_visp_3x3_bn",
    "simplified_mousenet_six_stream_visp_3x3_ir",
    "simplified_mousenet_single_stream_ir_224x224",
    "simplified_mousenet_dual_stream_visp_3x3_ir_224x224",
    "simplified_mousenet_six_stream_visp_3x3_ir_224x224",
    "simplified_mousenet_dual_stream_visp_3x3_ir_32x32",
    "simplified_mousenet_dual_stream_visp_3x3_ir_44x44",
    "simplified_mousenet_dual_stream_visp_3x3_ir_84x84",
    "simplified_mousenet_dual_stream_visp_3x3_ir_104x104",
    "simplified_mousenet_dual_stream_visp_3x3_ir_124x124",
    "simplified_mousenet_dual_stream_visp_3x3_ir_144x144",
    "simplified_mousenet_dual_stream_visp_3x3_ir_164x164",
    "simplified_mousenet_dual_stream_visp_3x3_ir_184x184",
    "simplified_mousenet_dual_stream_visp_3x3_ir_204x204",
    "simplified_mousenet_six_stream_visp_3x3_rotnet",
    "simplified_mousenet_six_stream_visp_3x3_mocov2",
    "simplified_mousenet_six_stream_visp_3x3_simclr",
    "simplified_mousenet_six_stream_visp_3x3_simsiam",
    "simplified_mousenet_ae_single_stream",
    "simplified_mousenet_ae_dual_stream",
    "simplified_mousenet_ae_six_stream",
    "simplified_mousenet_depth_hour_glass_single_stream",
    "simplified_mousenet_depth_hour_glass_dual_stream",
    "simplified_mousenet_depth_hour_glass_six_stream",
    "simplified_mousenet_single_stream_rand",
    "simplified_mousenet_dual_stream_visp_3x3_bn_rand",
    "simplified_mousenet_six_stream_visp_3x3_bn_rand",
]


class MultiStreamMouseNet(nn.Module):
    """
    Implementation of simplified MouseNet (for example, we can simplify by removing
    cortical layer nodes in the computation graph, compare to ShiMouseNet).
    Arguments:
        parallel_modules : (list of string) defines the intermediate nodes of the
                           computation graph. (e.g., ["dorsal", "ventral"]).
                           Default: ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
        output_areas     : (list of string) defines which modules are outputs for the
                           task. Concatenates the outputs of those modules prior to
                           feeding into a full-connected layer for image classification.
                           Default: ["VISpor", "VISam"]
        visp_output_pool : (boolean) If set to True, then the connections from VISp to
                           VISam and VISpor will be 1x1 convolution followed by a max-
                           pool. If set to False, then the connections from VISp to VISam
                           and VISpor will be a 3x3 convolution with stride of 2.
                           Default: True
        pool_size     : (int) size for adaptive average pooling prior to the fully-
                           connected layer
        num_classes      : (int) number of output classes for the task.
                           Default: 1000 (ImageNet)
    """

    def __init__(
        self,
        parallel_modules=["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"],
        output_areas=["VISpor", "VISam"],
        visp_output_pool=True,
        pool_size=6,
        num_classes=1000,
        norm_layer=None,
        single_stream=False,
        drop_final_fc=False,
        return_maxpool_indices=False,
    ):
        super(MultiStreamMouseNet, self).__init__()

        self.return_maxpool_indices = return_maxpool_indices

        assert pool_size > 0, f"Pooling size must be greater than 0. Given {pool_size}."
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.output_areas = output_areas
        self.visp_output_pool = visp_output_pool
        self.single_stream = single_stream
        print(f"Single stream set to {self.single_stream}")
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            print(f"Using {self.norm_layer} normalization")

        # prevent repeats
        if (
            len(parallel_modules) > 0
        ):  # 0 parallel modules still has connection from VISp->*
            assert len(list(set(parallel_modules))) == len(parallel_modules)
        # prevent repeats
        assert len(output_areas) > 0
        assert len(list(set(output_areas))) == len(output_areas)
        # check that the parallel and output modules are non-overlapping
        assert set(output_areas).isdisjoint(set(parallel_modules))
        # output areas is always at least one of VISpor or VISam
        assert set(output_areas).issubset(set(["VISpor", "VISam"]))
        if self.single_stream:
            """In this case, even if we have one parallel module, we never create a VISam layer (as we normally would)
            that is left untrained. Partly for memory purposes and partly because DDP does not allow an untrained module."""

            # maintains skip connection from VISp->* in the event of 0 parallel modules
            assert len(parallel_modules) <= 1
            # checks that length of outputs is list of 1 and we want it to be VISpor for convenience
            assert output_areas == ["VISpor"]

        # self.layers will be a dictionary populated with activations mouse_vision/
        # core/feature_extractor.py: CustomFeatureExtractor() uses this convention
        # self.layer_names will contain a list of modules (i.e., representative layers)
        self.layers = None
        self.layer_names = None

        # Set up source-target connections
        self.model_conns = ["input_VISp"]  # input -> VISp
        # VISp -> intermediate
        for parallel_mod in parallel_modules:
            connection = f"VISp_{parallel_mod}"
            self.model_conns.append(connection)
        # VISp -> VISpor / VISam
        self.model_conns.append("VISp_VISpor")
        if not self.single_stream:
            self.model_conns.append("VISp_VISam")
        # intermediate -> VISpor / VISam
        for parallel_mod in parallel_modules:
            if not self.single_stream:
                connection = f"{parallel_mod}_VISam"
                self.model_conns.append(connection)
            connection = f"{parallel_mod}_VISpor"
            self.model_conns.append(connection)

        # Set up connection parameters
        self.model_layers = nn.ModuleDict()
        for conn in self.model_conns:
            if conn == "input_VISp":
                if self.norm_layer is not None:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                        build_norm_layer(self.norm_layer, 64)[1],
                        nn.ReLU(inplace=False),
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            return_indices=self.return_maxpool_indices,
                        ),
                    )
                else:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                        nn.ReLU(inplace=False),
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            return_indices=self.return_maxpool_indices,
                        ),
                    )

            elif (conn == "VISp_VISpor") or (conn == "VISp_VISam"):
                if visp_output_pool:
                    if self.norm_layer is not None:
                        self.model_layers[conn] = nn.Sequential(
                            nn.Conv2d(
                                64, 256, kernel_size=1, stride=1, padding=0
                            ),  # 1x1 convolution to make spatial sizes match with max pooling
                            build_norm_layer(self.norm_layer, 256)[1],
                            nn.ReLU(inplace=False),
                            nn.MaxPool2d(
                                kernel_size=3,
                                stride=2,
                                return_indices=self.return_maxpool_indices,
                            ),
                        )
                    else:
                        self.model_layers[conn] = nn.Sequential(
                            nn.Conv2d(
                                64, 256, kernel_size=1, stride=1, padding=0
                            ),  # 1x1 convolution to make spatial sizes match with max pooling
                            nn.ReLU(inplace=False),
                            nn.MaxPool2d(
                                kernel_size=3,
                                stride=2,
                                return_indices=self.return_maxpool_indices,
                            ),
                        )
                else:
                    if self.norm_layer is not None:
                        self.model_layers[conn] = nn.Sequential(
                            nn.Conv2d(
                                64, 256, kernel_size=3, stride=2, padding=0
                            ),  # 3x3 convolution to make spatial sizes match with stride 2
                            build_norm_layer(self.norm_layer, 256)[1],
                            nn.ReLU(inplace=False),
                        )
                    else:
                        self.model_layers[conn] = nn.Sequential(
                            nn.Conv2d(
                                64, 256, kernel_size=3, stride=2, padding=0
                            ),  # 3x3 convolution to make spatial sizes match with stride 2
                            nn.ReLU(inplace=False),
                        )
            # Connections to VISpor and VISam
            elif ("VISam" in conn) or ("VISpor" in conn):
                if self.norm_layer is not None:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(384, 256, kernel_size=3, padding=1),
                        build_norm_layer(self.norm_layer, 256)[1],
                        nn.ReLU(inplace=False),
                    )
                else:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(384, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=False),
                    )
            # Intermediate connections
            else:
                if self.norm_layer is not None:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(64, 192, kernel_size=5, padding=2),
                        build_norm_layer(self.norm_layer, 192)[1],
                        nn.ReLU(inplace=False),
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            return_indices=self.return_maxpool_indices,
                        ),
                        nn.Conv2d(192, 384, kernel_size=3, padding=1),
                        build_norm_layer(self.norm_layer, 384)[1],
                        nn.ReLU(inplace=False),
                    )
                else:
                    self.model_layers[conn] = nn.Sequential(
                        nn.Conv2d(64, 192, kernel_size=5, padding=2),
                        nn.ReLU(inplace=False),
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            return_indices=self.return_maxpool_indices,
                        ),
                        nn.Conv2d(192, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=False),
                    )

        # Fully-connected layer for classification output: average pooling,
        # flatten activations, dropout during training, then FC layer. The
        # input size for the linear layer is the concatenation of the outputs
        # from each of the desired output modules (defined by user input for
        # output_areas).
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        if drop_final_fc:
            self.classifier = nn.Sequential(nn.Flatten(start_dim=1),)
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(),
                nn.Linear(len(output_areas) * 256 * pool_size * pool_size, num_classes),
            )

    def _get_source_target(self, conn):
        source_target_pair = conn.split("_")
        assert len(source_target_pair) == 2
        source = source_target_pair[0]
        target = source_target_pair[1]
        return source, target

    def _propagate_layer_features(
        self,
        x,
        set_layer_attrs=True,
        input_names=None,
        model_conns=None,
        model_layers=None,
        decoder_data=None,
        ret_decoder_data=False,
    ):

        layers = dict()
        layer_names = list()
        if ret_decoder_data:
            decoder_data = dict()

        if input_names is None:
            input_names = ["input"]
        if model_conns is None:
            model_conns = self.model_conns
        if model_layers is None:
            model_layers = self.model_layers

        if not isinstance(input_names, list):
            input_names = [input_names]

        assert set(list(x.keys())) == set(input_names)

        for conn in model_conns:

            # Get source-target pair
            source, target = self._get_source_target(conn)

            # Populate activations dictionary
            curr_layers = model_layers[conn]
            if source in input_names:
                curr_inp = x[source]
            else:
                assert source in layers.keys()
                curr_inp = layers[source]

            unpool_idxs = []
            decoder_output_sizes = []
            curr_out = curr_inp
            for m_idx, m in enumerate(curr_layers):
                # sizes of input here will be the output sizes in the decoder
                decoder_output_sizes.append(curr_out.size())

                curr_unpool_idxs = None
                if m.__class__.__name__ == "MaxPool2d":
                    if self.return_maxpool_indices:
                        curr_out, curr_unpool_idxs = m(curr_out)
                    else:
                        curr_out = m(curr_out)
                elif m.__class__.__name__ == "MaxUnpool2d":
                    cached_unpool_indices = decoder_data[conn]["unpool_idxs"][m_idx]
                    assert cached_unpool_indices is not None
                    curr_out = m(
                        curr_out,
                        indices=cached_unpool_indices,
                        output_size=decoder_data[conn]["output_sizes"][m_idx],
                    )
                elif m.__class__.__name__ == "ConvTranspose2d":
                    curr_out = m(
                        curr_out, output_size=decoder_data[conn]["output_sizes"][m_idx]
                    )
                else:
                    curr_out = m(curr_out)

                unpool_idxs.append(curr_unpool_idxs)

            if ret_decoder_data:
                # applied in reverse order in autoencoder
                decoder_data[target + "_" + source] = {
                    "unpool_idxs": unpool_idxs[::-1],
                    "output_sizes": decoder_output_sizes[::-1],
                }

            if target not in layers.keys():
                layers[target] = curr_out
            else:  # Add inputs from each source module of the target module
                layers[target] += curr_out

            if target not in layer_names:
                layer_names.append(target)

        if set_layer_attrs:
            self.layers = layers
            self.layer_names = layer_names
        else:
            if ret_decoder_data:
                return layers, layer_names, decoder_data
            else:
                return layers, layer_names

    def _concat_output_areas(self):
        concat_module_output = None
        for out_area in self.output_areas:
            if concat_module_output is None:
                concat_module_output = self.layers[out_area]
            else:
                assert self.layers[out_area].shape[0] == concat_module_output.shape[0]
                assert self.layers[out_area].shape[2] == concat_module_output.shape[2]
                assert self.layers[out_area].shape[3] == concat_module_output.shape[3]

                concat_module_output = torch.cat(
                    [concat_module_output, self.layers[out_area]],
                    axis=1,  # channels is the first axis (N, C, H, W)
                )

        return concat_module_output

    def forward(self, x):
        # First compute the activations for each module
        self._propagate_layer_features({"input": x})

        # Concatenate the outputs for the desired output areas
        concat_module_output = self._concat_output_areas()

        # Use the concatenated outputs as input to average pooling and classifier
        output = self.avgpool(concat_module_output)
        self.layers["avgpool"] = output
        self.layer_names.append("avgpool")

        output = self.classifier(output)
        return output


class MultiStreamMouseNetAE(MultiStreamMouseNet):
    """
    Implementation of the AutoEncoder version of the MultiStreamMouseNet.
    Arguments same as above, except for embedding dim.
    """

    def __init__(self, embedding_dim=128, compress=True, **kwargs):
        super(MultiStreamMouseNetAE, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.encoder_feats = (
            len(self.output_areas) * 256 * self.pool_size * self.pool_size
        )

        if compress:
            self.compress_fc = nn.Linear(self.encoder_feats, self.embedding_dim)
            self.decompress_fc = nn.Linear(self.embedding_dim, self.encoder_feats)
        else:
            self.compress_fc = nn.Identity()
            self.decompress_fc = nn.Identity()

        # we now invert the model
        self.decoder_layers = nn.ModuleDict()

        self.decoder_conns = []
        for conn in self.model_conns[::-1]:
            curr_block = self.model_layers[conn]
            source, target = self._get_source_target(conn)
            decoder_conn = target + "_" + source
            self.decoder_conns.append(decoder_conn)

            decoder_modules = []
            # traverse the operations in reverse order
            for m in reversed(curr_block):
                if m.__class__.__name__ == "Conv2d":
                    curr_op = nn.ConvTranspose2d(
                        in_channels=m.out_channels,
                        out_channels=m.in_channels,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                    )
                elif m.__class__.__name__ == "MaxPool2d":
                    curr_op = nn.MaxUnpool2d(
                        kernel_size=m.kernel_size, stride=m.stride, padding=m.padding
                    )
                else:
                    curr_op = copy.deepcopy(m)

                decoder_modules.append(curr_op)

            self.decoder_layers[decoder_conn] = nn.Sequential(*decoder_modules)

    def forward(self, x):
        # First compute the activations for each module
        self.layers, self.layer_names, decoder_data = self._propagate_layer_features(
            {"input": x}, set_layer_attrs=False, ret_decoder_data=True
        )

        # Concatenate the outputs for the desired output areas
        concat_module_output = self._concat_output_areas()

        # Use the concatenated outputs as input to average pooling and classifier
        encoder_output = self.avgpool(concat_module_output)
        self.layers["avgpool"] = encoder_output
        self.layer_names.append("avgpool")

        encoder_output = self.classifier(encoder_output)

        # Embed to embedding dimension
        compress_output = self.compress_fc(encoder_output)

        # Convert from embedding dimension back to what it was before
        output = self.decompress_fc(compress_output)

        # sanity check
        assert output.shape[-1] == np.prod(encoder_output.shape[1:])

        output = output.reshape(
            output.shape[0],
            len(self.output_areas) * 256,
            self.pool_size,
            self.pool_size,
        )

        # avg pool does nearest neighbor resize when input < pool size and avgpool when input > pool size
        # so we use the same op for consistency
        avgunpool_op = nn.AdaptiveAvgPool2d(
            (concat_module_output.shape[2], concat_module_output.shape[3])
        )
        output = avgunpool_op(output)

        decoder_x = {}
        for output_area_idx, output_area in enumerate(self.output_areas):
            decoder_x[output_area] = output[
                :, output_area_idx * 256 : (output_area_idx + 1) * 256, :, :
            ]

        self.decoder_outputs, _ = self._propagate_layer_features(
            decoder_x,
            set_layer_attrs=False,
            input_names=self.output_areas,
            model_conns=self.decoder_conns,
            model_layers=self.decoder_layers,
            decoder_data=decoder_data,
        )

        return {
            "output": self.decoder_outputs["input"],
            "hidden_vec": compress_output,
            "encoder_output": encoder_output,
        }


class MultiStreamMouseNetDepthHourGlass(MultiStreamMouseNetAE):
    """
    Implementation of the hour-glass version of the MultiStreamMouseNet.
    """

    def __init__(self, **kwargs):
        super(MultiStreamMouseNetDepthHourGlass, self).__init__(
            compress=False, **kwargs
        )


def simplified_mousenet_six_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=True,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_six_stream_visp_3x3(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_six_stream_vispor_only(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are taken only from VISpor prior to the fully-
    connected classifier.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=True,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_six_stream_vispor_only_visp_3x3(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are taken only from VISpor prior to the fully-
    connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=False,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_dual_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are concatenated from VISpor and
    VISam prior to the fully-connected classifier.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=True,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_dual_stream_visp_3x3(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_dual_stream_vispor_only(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are taken only from VISpor prior to
    the fully-connected classifier.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=True,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_dual_stream_vispor_only_visp_3x3(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are taken only from VISpor prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=False,
        pool_size=6,
        **kwargs,
    )

    return model


def simplified_mousenet_single_stream_base(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with one stream: used for trying out different tasks.
    Outputs from VISp are processed with a 3x3 convolution (stride 2) before VISpor layer.
    as this gave slightly higher ImageNet performance/neural fits in some cases.
    """
    parallel_modules = ["ventral"]
    model = MultiStreamMouseNet(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=False,
        pool_size=6,
        single_stream=True,
        **kwargs,
    )

    return model


def simplified_mousenet_single_stream(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), **kwargs
    )


def simplified_mousenet_single_stream_rand(pretrained=False, **kwargs):

    """We set drop_final_fc=True to train finetune linear layer off of it later."""
    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_single_stream_ir(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_single_stream_ir_224x224(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_single_stream_rotnet(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_single_stream_mocov2(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_single_stream_simclr(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_single_stream_simsiam(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_dual_stream_visp_3x3_bn(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_bn_rand(pretrained=False, **kwargs):
    """We set drop_final_fc=True to train finetune linear layer off of it later."""
    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_32x32(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_44x44(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_84x84(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_104x104(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_124x124(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_144x144(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_164x164(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_184x184(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_204x204(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_ir_224x224(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_rotnet(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_mocov2(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_simclr(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_dual_stream_visp_3x3_simsiam(pretrained=False, **kwargs):

    return simplified_mousenet_dual_stream_visp_3x3(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_six_stream_visp_3x3_bn(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_bn_rand(pretrained=False, **kwargs):
    """We set drop_final_fc=True to train finetune linear layer off of it later."""
    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_ir(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_ir_224x224(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_rotnet(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_mocov2(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained, norm_layer=dict(type="BN"), drop_final_fc=True, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_simclr(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_six_stream_visp_3x3_simsiam(pretrained=False, **kwargs):

    return simplified_mousenet_six_stream_visp_3x3(
        pretrained=pretrained,
        norm_layer=dict(type="SyncBN"),
        drop_final_fc=True,
        **kwargs,
    )


def simplified_mousenet_ae_single_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet AutoEncoder with one stream.
    """
    parallel_modules = ["ventral"]
    model = MultiStreamMouseNetAE(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=False,
        single_stream=True,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


def simplified_mousenet_ae_dual_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet AutoEncoder architecture with two parallel streams.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNetAE(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


def simplified_mousenet_ae_six_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet AutoEncoder architecture with six parallel streams.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNetAE(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


def simplified_mousenet_depth_hour_glass_single_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet Depth Prediction with one stream.
    """
    parallel_modules = ["ventral"]
    model = MultiStreamMouseNetDepthHourGlass(
        parallel_modules=parallel_modules,
        output_areas=["VISpor"],
        visp_output_pool=False,
        single_stream=True,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


def simplified_mousenet_depth_hour_glass_dual_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet Depth Prediction with two parallel streams.
    """
    parallel_modules = ["ventral", "dorsal"]
    model = MultiStreamMouseNetDepthHourGlass(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


def simplified_mousenet_depth_hour_glass_six_stream(pretrained=False, **kwargs):
    """
    Simplified MouseNet Depth Prediction with six parallel streams.
    """
    parallel_modules = ["VISl", "VISrl", "VISal", "VISli", "VISpm", "VISpl"]
    model = MultiStreamMouseNetDepthHourGlass(
        parallel_modules=parallel_modules,
        output_areas=["VISpor", "VISam"],
        visp_output_pool=False,
        norm_layer=dict(type="BN"),
        drop_final_fc=True,
        return_maxpool_indices=True,
        **kwargs,
    )

    return model


if __name__ == "__main__":

    x = torch.rand(8, 3, 64, 64)

    # Six intermediate modules with VISpor and VISam output
    w = simplified_mousenet_six_stream(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)
    for layer in w.layer_names:
        print(f"{layer}: {w.layers[layer].shape}")

    # Six intermediate modules with VISpor and VISam output, VISp to VISam and VISpor
    # are 3x3 convolutions (instead of 1x1 and maxpool)
    w = simplified_mousenet_six_stream_visp_3x3(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)

    # Six intermediate modules with VISpor output only
    w = simplified_mousenet_six_stream_vispor_only(pretrained=False)
    o = w(x)
    print(o.shape)

    # Two intermediate modules with VISpor and VISam output
    w = simplified_mousenet_dual_stream(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)
    for layer in w.layer_names:
        print(f"{layer}: {w.layers[layer].shape}")

    # Two intermediate modules with VISpor output only
    w = simplified_mousenet_dual_stream_vispor_only(pretrained=False)
    o = w(x)
    print(o.shape)
    for layer in w.layer_names:
        print(f"{layer}: {w.layers[layer].shape}")

    # Single intermediate modules AE
    w = simplified_mousenet_ae_single_stream(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)
    for layer in w.decoder_outputs.keys():
        print(f"{layer}: {w.decoder_outputs[layer].shape}")

    # Dual intermediate modules AE
    w = simplified_mousenet_ae_dual_stream(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)
    for layer in w.decoder_outputs.keys():
        print(f"{layer}: {w.decoder_outputs[layer].shape}")

    # Six intermediate modules AE
    w = simplified_mousenet_ae_six_stream(pretrained=False)
    o = w(x)
    print(o.shape)
    print(w)
    for layer in w.decoder_outputs.keys():
        print(f"{layer}: {w.decoder_outputs[layer].shape}")
