import numpy as np
from torch.utils import data
from torchvision import datasets
from mouse_vision.core.constants import DMLOCOMOTION_NUM_IMGS

__all__ = ["DMLocomotionBase", "DMLocomotionIndex"]

class CustomSubset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.imgs = [self.dataset.imgs[i] for i in self.indices]
        self.samples = [self.dataset.samples[i] for i in self.indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class DMLocomotionBase(data.Dataset):
    """
    Base class for obtaining DMLocomotion data set.

    Arguments:
        base_dir     : (string) base directory for DMLocomotion images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, base_dir, image_transforms, num_imgs=None, img_sampler_seed=0):
        # Assumes base_dir organization is:
        # /PATH/TO/DMLocomotion/rec_*/strrec_*/*.JPEG

        super(DMLocomotionBase, self).__init__()
        full_dataset = datasets.ImageFolder(
            base_dir, transform=image_transforms
        )
        assert(len(full_dataset) == DMLOCOMOTION_NUM_IMGS)
        full_idxs = np.arange(DMLOCOMOTION_NUM_IMGS)
        if num_imgs is None:
            self.dataset = full_dataset
            self.sel_idxs = full_idxs
        else:
            assert(num_imgs <= len(full_dataset))
            print(f"Selecting {num_imgs} images")
            self.sel_idxs = np.random.RandomState(img_sampler_seed).permutation(full_idxs)[:num_imgs]
            self.dataset = CustomSubset(dataset=full_dataset, indices=self.sel_idxs)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

class DMLocomotionIndex(DMLocomotionBase):
    """
    DMLocomotion data set class that also returns the index of the image in the data set.

    Arguments:
        base_dir     : (string) base directory fo DMLocomotion images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, base_dir, image_transforms, num_imgs=None, img_sampler_seed=0):
        super(DMLocomotionIndex, self).__init__(base_dir=base_dir,
                                                image_transforms=image_transforms,
                                                num_imgs=num_imgs,
                                                img_sampler_seed=img_sampler_seed)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # Return the index
        data = [index] + image_data
        return tuple(data)


if __name__ == "__main__":
    from torchvision import transforms
    from mouse_vision.core.default_dirs import DMLOCOMOTION_DATA_DIR

    my_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    d = DMLocomotionIndex(DMLOCOMOTION_DATA_DIR, my_transforms)
    index, image, label = d[2000]
    print(index)
    print(image.shape)
    print(label)

