import torch

from mouse_vision.model_training.imagenet_datasets.imagenet_base import ImageNetBase

__all__ = ["ImageNetContrastive"]


class ImageNetContrastive(ImageNetBase):
    """
    ImageNet data set class that also returns two "views" of the same image.

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        super(ImageNetContrastive, self).__init__(
            is_train, imagenet_dir, image_transforms
        )

    def __getitem__(self, index):
        # Get two views of the SAME images
        image_label_view1 = list(self.dataset.__getitem__(index))
        image_label_view2 = list(self.dataset.__getitem__(index))
        image_view1 = image_label_view1[0]  # Assuming (C, H, W)

        assert image_view1.shape[0] == 3
        image_view2 = image_label_view2[0]  # Assuming (C, H, W)
        assert image_view2.shape[0] == 3

        # (2, C, H, W)
        image_cat = torch.cat(
            (image_view1.unsqueeze(0), image_view2.unsqueeze(0)), dim=0
        )

        # Return the concatenated image, discarding labels
        return image_cat


if __name__ == "__main__":
    from torchvision import transforms
    from mouse_vision.core.default_dirs import IMAGENET_DATA_DIR

    my_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    d = ImageNetContrastive(True, IMAGENET_DATA_DIR, my_transforms)
    image = d[2000]
    print(image.shape)

