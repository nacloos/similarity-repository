import numpy as np
from torch.utils.data import Dataset

class ClassSubsampler(Dataset):
    """
    Samples just one class from a Dataset object (eg: torchvision.datasets.CIFAR10)
    """
    def __init__(self, root, train, download, transform, class_to_sample, base_dataset, class_label_offset=0):
        """
        root, train, download, transform: placeholders, only so that this works with loaders.py
        transform and train would've been applied to base_dataset, so they have no purpose here
        base_dataset: instance of torch.utils.data.Dataset 
                      should have `targets` and `classes` 
        class_to_sample: list[int, int ...]; indices of classes to be sampled from the larger dataset
        class_label_offset: this is subtracted from each class label
        """
        self.base_dataset = base_dataset
        self.class_to_sample = class_to_sample
        self.base_targets = np.array(self.base_dataset.targets)

        self.name = f'{self.base_dataset.__class__.__name__}'
        for c in self.class_to_sample:
            self.name += f'_{self.base_dataset.classes[c]}'
        self.indices_to_sample_from = np.concatenate([np.where(self.base_targets == c)[0] for c in self.class_to_sample])
        self.data = base_dataset.data[self.indices_to_sample_from] # required by the loader
        self.class_label_offset = class_label_offset

    def __getitem__(self, index):
        x, y = self.base_dataset.__getitem__(self.indices_to_sample_from[index])
        return x, y - self.class_label_offset

    def __len__(self):
        return len(self.indices_to_sample_from)
