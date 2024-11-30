import torch

from mouse_vision.model_training.imagenet_datasets.imagenet_base import ImageNetBase

__all__ = ["ImageNetRotation"]


class ImageNetRotation(ImageNetBase):
    """
    ImageNet data set class that also returns the four rotations of an image and their
    associated "labels".

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        super(ImageNetRotation, self).__init__(is_train, imagenet_dir, image_transforms)

    def __getitem__(self, index):
        img0, _ = self.dataset.__getitem__(index)
        assert img0.ndim == 3 and img0.shape[0] == 3  # C,H,W
        assert img0.shape[1] == img0.shape[2]
        img90 = torch.flip(torch.transpose(img0, 1, 2), (1,))
        img180 = torch.flip(torch.flip(img0, (2,)), (1,))
        img270 = torch.flip(torch.transpose(img0, 1, 2), (2,))

        imgs = [img0, img90, img180, img270]
        imgs = torch.stack(imgs, dim=0)

        labels = torch.LongTensor([0, 1, 2, 3])

        return imgs, labels


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
    d = ImageNetRotation(True, IMAGENET_DATA_DIR, my_transforms)
    images, labels = d[3000]
    assert images.shape[0] == labels.shape[0] == 4
    print(labels)
    print(images.shape)

    import numpy as np
    import matplotlib.pyplot as plt

    base = 10
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(base * 4, base))
    for i, (im, _ax) in enumerate(zip(images, ax.ravel())):
        _ax.imshow(np.transpose(im.numpy(), (1, 2, 0)))
        _ax.set_title(labels[i])
    plt.savefig("images.pdf", format="pdf")

