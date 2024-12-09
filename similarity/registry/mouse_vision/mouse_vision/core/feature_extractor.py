import numpy as np
import torch


class FeatureExtractor:
    """
    Extracts activations from a layer of a model.

    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
        debug      : (boolean) whether or not to test with two batches
    """

    def __init__(self, dataloader, n_batches=None, vectorize=False, debug=False):
        self.dataloader = dataloader
        if n_batches is None:
            self.n_batches = len(self.dataloader)
        else:
            self.n_batches = n_batches
        self.vectorize = vectorize
        self.debug = debug

    def _store_features(self, layer, inp, out):
        out = out.cpu().numpy()

        if self.vectorize:
            self.layer_feats.append(np.reshape(out, (len(out), -1)))
        else:
            self.layer_feats.append(out)

    def extract_features(self, model, model_layer, return_labels=False):
        if torch.cuda.is_available():
            model.cuda().eval()
        else:
            model.cpu().eval()
        self.layer_feats = list()
        self.labels = list()
        # Set up forward hook to extract features
        handle = model_layer.register_forward_hook(self._store_features)

        with torch.no_grad():
            for i, (x, label_x) in enumerate(self.dataloader):
                if i == self.n_batches:
                    break

                # Break when you go through 2 batches for faster testing
                if self.debug and i == 2:
                    break

                print(f"Step {i+1}/{self.n_batches}")
                if torch.cuda.is_available():
                    x = x.cuda()
                    label_x = label_x.cuda()

                model(x)
                label_x = label_x.cpu().numpy()
                self.labels.append(label_x)

        self.layer_feats = np.concatenate(self.layer_feats)
        self.labels = np.concatenate(self.labels)
        # Reset forward hook so next time function runs, previous hooks
        # are removed
        handle.remove()

        if return_labels:
            return self.layer_feats, self.labels

        return self.layer_feats


class CustomFeatureExtractor(FeatureExtractor):
    """
    Extracts activations from a layer of a custom model, e.g. MouseNet, where we do not use nn.Sequential.

    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
        debug      : (boolean) whether or not to test with two batches
    """

    def __init__(self, dataloader, **kwargs):
        super(CustomFeatureExtractor, self).__init__(dataloader, **kwargs)

    def extract_features(self, model, model_layer, return_labels=False):
        assert isinstance(model_layer, str)
        if torch.cuda.is_available():
            model.cuda().eval()
        else:
            model.cpu().eval()
        self.layer_feats = list()
        self.labels = list()
        with torch.no_grad():
            for i, (x, label_x) in enumerate(self.dataloader):
                if i == self.n_batches:
                    break

                # Break when you go through 2 batches for faster testing
                if self.debug and i == 2:
                    break

                print(f"Step {i+1}/{self.n_batches}")
                if torch.cuda.is_available():
                    x = x.cuda()
                    label_x = label_x.cuda()

                model(x)
                out = model.layers[model_layer]
                out = out.cpu().numpy()
                label_x = label_x.cpu().numpy()

                if self.vectorize:
                    self.layer_feats.append(np.reshape(out, (len(out), -1)))
                else:
                    self.layer_feats.append(out)

                self.labels.append(label_x)

        self.layer_feats = np.concatenate(self.layer_feats)
        self.labels = np.concatenate(self.labels)

        if return_labels:
            return self.layer_feats, self.labels

        return self.layer_feats


def get_layer_features(
    feature_extractor, layer_name, model, model_name, return_labels=False
):
    """
    Helper function to extract stimuli features from a layer in a model.

    Inputs:
        feature_extractor : (FeatureExtractor) object used to extract features from a layer
        layer_name        : (string) name of layer from which to extract features
        model             : (torch.nn.Module) object

    Outputs:
        features          : (numpy.ndarray) of dimensions (num_images, num_features)
    """
    if isinstance(model, torch.nn.DataParallel):
        layer_module = model.module
    else:
        layer_module = model

    for part in layer_name.split("."):
        if "mousenet" in model_name or "alexnet_64x64_input_dict" in model_name:
            layer_module = part
        else:
            layer_module = layer_module._modules.get(part)
        assert (
            layer_module is not None
        ), f"No submodule found for layer {layer_name}, at part {part}."

    features = feature_extractor.extract_features(
        model=model, model_layer=layer_module, return_labels=return_labels
    )
    return features


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from mouse_vision.core.dataloader_utils import get_image_array_dataloader
    from mouse_vision.core.model_loader_utils import load_model

    model, layers = load_model("alexnet", trained=False, model_path=None)

    N = 10
    array = torch.rand(N, 224, 224, 3).numpy()
    img_transforms = transforms.Compose([transforms.ToTensor()])
    dataloader = get_image_array_dataloader(array, torch.ones(N), img_transforms)

    layer_name = "features.0"
    fe = FeatureExtractor(
        dataloader=dataloader, n_batches=3, vectorize=False, debug=True
    )
    features = get_layer_features(
        feature_extractor=fe, layer_name=layer_name, model=model, model_name="alexnet"
    )
    print(features.shape)
