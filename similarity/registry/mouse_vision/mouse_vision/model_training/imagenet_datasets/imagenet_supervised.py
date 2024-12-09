from mouse_vision.model_training.imagenet_datasets.imagenet_base import ImageNetBase

__all__ = ["ImageNetSupervised"]


class ImageNetSupervised(ImageNetBase):
    """
    ImageNet data set class that returns an image and its label.

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        super(ImageNetSupervised, self).__init__(is_train, imagenet_dir, image_transforms)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


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
    d = ImageNetSupervised(True, IMAGENET_DATA_DIR, my_transforms)
    image, label = d[1000000]
    print(image.shape)
    print(label)

    import numpy as np
    import matplotlib.pyplot as plt
    base = 10
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(base, base))
    ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    ax.set_title(label)
    plt.savefig("sample_image.pdf", format="pdf")


