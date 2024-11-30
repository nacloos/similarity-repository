__all__ = ["BaseDataset"]

class BaseDataset:
    def __init__(self):
        pass

    def get_name(self):
        return self.name

    def get_dataloader(self):
        raise NotImplementedError

