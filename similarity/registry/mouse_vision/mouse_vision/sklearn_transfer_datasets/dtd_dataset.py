import os
from mouse_vision.sklearn_transfer_datasets import BaseDataset
from mouse_vision.core.default_dirs import DT_DATA_DIR
from torch.utils import data
from torchvision import datasets
from mouse_vision.core.dataloader_utils import _acquire_data_loader

__all__ = ["DTD", "DTDataloader"]


class DTD(data.Dataset):
    """
    Base class for obtaining Describable Textures Data set.

    Arguments:
        is_train         : (boolean) if training or validation set
        dtd_dir     : (string) base directory for DTD images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, mode, image_transforms, dtd_dir=DT_DATA_DIR):
        # Assumes dtd_dir organization is:
        # /PATH/TO/DTD/{train1, val1, test1}/{synsets}/*.JPEG

        super(DTD, self).__init__()
        self.dataset = datasets.ImageFolder(
            os.path.join(dtd_dir, mode), transform=image_transforms
        )

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

class DTDataloader(BaseDataset):
    def __init__(self, split_num, image_transforms, batch_size=256):
        self.batch_size = batch_size
        # these should be the val image transforms
        self.dtd_train = DTD(mode=f"train{split_num}", image_transforms=image_transforms)
        self.dtd_val = DTD(mode=f"val{split_num}", image_transforms=image_transforms)
        self.dtd_test = DTD(mode=f"test{split_num}", image_transforms=image_transforms)
        self.dataloader_kwargs = {"batch_size": self.batch_size, "shuffle": False, "num_workers": 8, "pin_memory": True}
        self.name = "DTDataloader"
        self.dataloader = None

    def get_dataloader(self):
        if self.dataloader is None:
            train_loader = _acquire_data_loader(
                dataset=self.dtd_train,
                **self.dataloader_kwargs
            )

            val_loader = _acquire_data_loader(
                dataset=self.dtd_val,
                **self.dataloader_kwargs
            )

            test_loader = _acquire_data_loader(
                dataset=self.dtd_test,
                **self.dataloader_kwargs
            )

            self.dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

        return self.dataloader
