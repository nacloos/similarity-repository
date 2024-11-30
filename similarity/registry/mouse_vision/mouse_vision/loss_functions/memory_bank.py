import torch

import numpy as np
import torch.distributed as dist

from mouse_vision.loss_functions.loss_utils import l2_normalize


class MemoryBank:
    def __init__(self, num_samples, dimension, device):
        # Device: could be TPU/GPU device
        self.device = device

        # Memory bank will be of shape: (num_samples, dimension)
        self.dimension = dimension
        self.num_samples = num_samples

        # Initialize memory bank and then broadcast it to the other processes
        # from the source process (whose rank is 0)
        if self.device.type == "xla":  # TPU
            import torch_xla.core.xla_model as xm

            # If rank is 0, then get a random initialization, otherwise
            # initialize with zeros since we will do an all_reduce sum to
            # make sure that all cores get the same initialization.
            if xm.get_ordinal() == 0:
                self.memory_bank = self._initialize_bank()
            else:
                self.memory_bank = torch.zeros(
                    self.num_samples, self.dimension, device=self.device
                )
        else:
            self.memory_bank = self._initialize_bank()

        # This will synchronize the memory banks across all processes
        self.set_memory_bank(self.memory_bank)

    def _initialize_bank(self):
        memory_bank = torch.rand(
            self.num_samples, self.dimension, device=self.device, requires_grad=False
        )
        std_dev = 1.0 / np.sqrt(self.dimension / 3)
        memory_bank = memory_bank * (2 * std_dev) - std_dev
        memory_bank = l2_normalize(memory_bank, dim=1)

        return memory_bank

    def as_tensor(self):
        return self.memory_bank.cpu()

    def get_all_inner_products(self, embeddings):
        assert embeddings.ndim == 2
        assert isinstance(self.memory_bank, torch.Tensor)

        assert self.memory_bank.shape[1] == embeddings.shape[1]
        inner_prods = torch.matmul(embeddings, torch.transpose(self.memory_bank, 1, 0))
        return inner_prods

    def set_memory_bank(self, memory_bank_tensor):
        # memory_bank_tensor : (torch.Tensor) memory bank as a tensor
        assert hasattr(self, "memory_bank")
        assert isinstance(self.memory_bank, torch.Tensor)
        assert isinstance(memory_bank_tensor, torch.Tensor)

        assert memory_bank_tensor.shape == (self.num_samples, self.dimension)
        if self.device.type == "xla":  # TPU
            import torch_xla.core.xla_model as xm

            # If the memory bank is loaded, we need to make sure that the "non
            # -source" processes have an "all-zeros" memory bank. Otherwise,
            # all_reduce will result in the wrong update.
            if xm.get_ordinal() != 0:
                memory_bank_tensor = torch.zeros_like(memory_bank_tensor)

            # This will update memory_bank_tensor
            memory_bank_tensor = xm.all_reduce(xm.REDUCE_SUM, memory_bank_tensor)
            self.memory_bank = memory_bank_tensor
        else:
            dist.broadcast(memory_bank_tensor, 0)
            self.memory_bank = memory_bank_tensor

    def get_memory_bank(self):
        assert hasattr(self, "memory_bank")
        assert isinstance(self.memory_bank, torch.Tensor)
        return self.memory_bank

    def update(self, indices, entries_to_update):
        """
        This function updates the memory bank with the new embeddings.

        Inputs:
            indices           : (torch.Tensor) indices of the entries into the
                                memory bank
            entries_to_update : (torch.Tensor) updated memory embeddings
        """
        assert entries_to_update.ndim == 2
        assert indices.shape[0] == entries_to_update.shape[0]
        assert entries_to_update.shape[1] == self.dimension

        indices = indices.unsqueeze(1).repeat(1, self.dimension)
        assert indices.shape == entries_to_update.shape
        curr_device = self.memory_bank.device
        indices = indices.to(curr_device)
        entries_to_update = entries_to_update.to(curr_device)
        self.memory_bank.scatter_(0, indices, entries_to_update)

        # TODO: check that all memory banks for all processes are synced
