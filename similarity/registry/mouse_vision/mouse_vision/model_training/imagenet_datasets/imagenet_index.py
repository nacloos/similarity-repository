from mouse_vision.model_training.imagenet_datasets.imagenet_base import ImageNetBase

__all__ = ["ImageNetIndex"]


class ImageNetIndex(ImageNetBase):
    """
    ImageNet data set class that also returns the index of the image in the data set.

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        super(ImageNetIndex, self).__init__(is_train, imagenet_dir, image_transforms)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # Return the index
        data = [index] + image_data
        return tuple(data)


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
    d = ImageNetIndex(True, IMAGENET_DATA_DIR, my_transforms)
    index, image, label = d[2000]
    print(index)
    print(image.shape)
    print(label)

