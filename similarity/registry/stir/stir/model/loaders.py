import argparse

from .tools import folder

import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import numpy as np
import torch as ch
import torch.utils.data
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, shuffle=False):
        self.labels_list = dataset.targets
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = sorted(list(set(self.labels.numpy())))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        if shuffle:
            np.random.seed(1)
            np.random.shuffle(self.labels_set)
            for l in self.labels_set:
                np.random.shuffle(self.label_to_indices[l])
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        while self.count + self.batch_size < len(self.dataset):
            classes = self.labels_set[:self.n_classes]
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


def make_loaders(workers, batch_size, transforms, data_path, data_aug=True,
                custom_class=None, dataset="", label_mapping=None, subset=None,
                subset_type='rand', subset_start=0, val_batch_size=None,
                only_val=False, shuffle_train=True, shuffle_val=True, seed=1,
                custom_class_args=None,
                n_classes=None, n_samples=None, pin_memory=True, return_datasets=False):
    '''
    **INTERNAL FUNCTION**

    This is an internal function that makes a loader for any dataset. You
    probably want to call dataset.make_loaders for a specific dataset,
    which only requires workers and batch_size. For example:

    >>> cifar_dataset = CIFAR10('/path/to/cifar')
    >>> train_loader, val_loader = cifar_dataset.make_loaders(workers=10, batch_size=128)
    >>> # train_loader and val_loader are just PyTorch dataloaders
    '''
    print(f"==> Preparing dataset {dataset}..")
    transform_train, transform_test = transforms
    if not data_aug:
        transform_train = transform_test

    if not val_batch_size:
        val_batch_size = batch_size

    if not custom_class:
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, 'test')

        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        if not only_val:
            train_set = folder.ImageFolder(root=train_path, transform=transform_train,
                                           label_mapping=label_mapping)
        test_set = folder.ImageFolder(root=test_path, transform=transform_test,
                                      label_mapping=label_mapping)
    else:
        if custom_class_args is None: custom_class_args = {}
        if not only_val:
            train_set = custom_class(root=data_path, train=True, download=True, 
                transform=transform_train, **custom_class_args)
        test_set = custom_class(root=data_path, train=False, download=True, 
                                transform=transform_test, **custom_class_args)

    if not only_val:
        attrs = ["samples", "train_data", "data"]
        vals = {attr: hasattr(train_set, attr) for attr in attrs}
        assert any(vals.values()), f"dataset must expose one of {attrs}"
        train_sample_count = len(getattr(train_set,[k for k in vals if vals[k]][0]))

    if (not only_val) and (subset is not None) and (subset <= train_sample_count):
        assert not only_val
        if subset_type == 'rand':
            rng = np.random.RandomState(seed)
            subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
            subset = subset[subset_start:]
        elif subset_type == 'first':
            subset = np.arange(subset_start, subset_start + subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)
        train_set = Subset(train_set, subset)

    if not only_val:
        if n_classes:
            balanced_batch_sampler = BalancedBatchSampler(train_set, n_classes, n_samples, shuffle_train)
            train_loader = DataLoader(train_set, shuffle=shuffle_train, 
                num_workers=workers, pin_memory=pin_memory, 
                batch_sampler=balanced_batch_sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, 
                shuffle=shuffle_train, 
                num_workers=workers, pin_memory=pin_memory)

    if n_classes:
        balanced_batch_sampler = BalancedBatchSampler(test_set, n_classes, n_samples, shuffle_val)
        test_loader = DataLoader(test_set, num_workers=workers, pin_memory=pin_memory, shuffle=shuffle_val,
            batch_sampler=balanced_batch_sampler)
    else:
        test_loader = DataLoader(test_set, batch_size=val_batch_size, 
            shuffle=shuffle_val, num_workers=workers, pin_memory=pin_memory)

    if only_val:
        return (None, test_loader, None, test_set) if return_datasets else (None, test_loader)

    return (train_loader, test_loader, train_set, test_set) if return_datasets else (train_loader, test_loader)

## loader wrapper (for adding custom functions to dataloader)
class PerEpochLoader:
    '''
    A blend between TransformedLoader and LambdaLoader: stores the whole loader
    in memory, but recomputes it from scratch every epoch, instead of just once
    at initialization.
    '''
    def __init__(self, loader, func, do_tqdm=True):
        self.orig_loader = loader
        self.func = func
        self.do_tqdm = do_tqdm
        self.data_loader = self.compute_loader()
        self.loader = iter(self.data_loader)

    def compute_loader(self):
        return TransformedLoader(self.orig_loader, self.func, None,
                    self.orig_loader.num_workers, self.orig_loader.batch_size,
                    do_tqdm=self.do_tqdm)

    def __len__(self):
        return len(self.orig_loader)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration as e:
            self.data_loader = self.compute_loader()
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)

class LambdaLoader:
    '''
    This is a class that allows one to apply any given (fixed) 
    transformation to the output from the loader in *real-time*.

    For instance, you could use for applications such as custom 
    data augmentation and adding image/label noise.

    Note that the LambdaLoader is the final transformation that
    is applied to image-label pairs from the dataset as part of the
    loading process---i.e., other (standard) transformations such
    as data augmentation can only be applied *before* passing the
    data through the LambdaLoader.

    For more information see :ref:`our detailed walkthrough <using-custom-loaders>`

    '''

    def __init__(self, loader, func):
        '''
        Args:
            loader (PyTorch dataloader) : loader for dataset (*required*).
            func (function) : fixed transformation to be applied to 
                every batch in real-time (*required*). It takes in 
                (images, labels) and returns (images, labels) of the 
                same shape.
        '''
        self.data_loader = loader
        self.loader = iter(self.data_loader)
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __next__(self):
        try:
            im, targ = next(self.loader)
        except StopIteration as e:
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

def TransformedLoader(loader, func, transforms, workers=None, 
        batch_size=None, do_tqdm=False, augment=False, fraction=1.0,
        shuffle=True):
    '''
    This is a function that allows one to apply any given (fixed) 
    transformation to the output from the loader *once*. 

    For instance, you could use for applications such as assigning
    random labels to all the images (before training).

    The TransformedLoader also supports the application of addiotional
    transformations (such as standard data augmentation) after the fixed
    function.

    For more information see :ref:`our detailed walkthrough <using-custom-loaders>`

    Args:
        loader (PyTorch dataloader) : loader for dataset
        func (function) : fixed transformation to be applied once. It takes 
        in (images, labels) and returns (images, labels) with the same shape 
        in every dimension except for the first, i.e., batch dimension 
        (which can be any length).
        transforms (torchvision.transforms) : transforms to apply 
            to the training images from the dataset (after func) (*required*).
        workers (int) : number of workers for data fetching (*required*).
        batch_size (int) : batch size for the data loaders (*required*).
        do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
        augment (bool) : if True,  the output loader contains both the original
            (untransformed), and new transformed image-label pairs.
        fraction (float): fraction of image-label pairs in the output loader 
            which are transformed. The remainder is just original image-label 
            pairs from loader. 
        shuffle (bool) : whether or not the resulting loader should shuffle every 
            epoch (defaults to True)

    Returns:
        A loader and validation loader according to the
        parameters given. These are standard PyTorch data loaders, and
        thus can just be used via:

        >>> output_loader = ds.make_loaders(loader,
                                            assign_random_labels,
                                            workers=8, 
                                            batch_size=128) 
        >>> for im, lab in output_loader:
        >>>     # Do stuff...
    '''

    new_ims = []
    new_targs = []
    total_len = len(loader)
    enum_loader = enumerate(loader)

    it = enum_loader if not do_tqdm else tqdm(enum_loader, total=total_len)
    for i, (im, targ) in it:
        new_im, new_targ = func(im, targ)
        if augment or (i / float(total_len) > fraction):
              new_ims.append(im.cpu())
              new_targs.append(targ.cpu())
        if i / float(total_len) <= fraction:
            new_ims.append(new_im.cpu())
            new_targs.append(new_targ.cpu())

    dataset = folder.TensorDataset(ch.cat(new_ims, 0), ch.cat(new_targs, 0), transform=transforms)
    return ch.utils.data.DataLoader(dataset, num_workers=workers, 
                        batch_size=batch_size, shuffle=shuffle)
