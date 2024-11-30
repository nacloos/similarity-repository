import torch
import torch.nn as nn
import numpy as np

from mouse_vision.loss_functions.loss_utils import l2_normalize
from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["InstanceDiscriminationLoss"]


class InstanceDiscriminationLoss(LossFunctionBase):
    """
    Loss function module for instance discrimination. Wu et al. (2018):
    https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf

    Arguments:
        memory_bank      : (torch.Tensor) memory bank tensor
        model_output_dim : (int) output dimension of model without FC layer
        m                : (int) number of "noise" samples. Default: 4096.
        gamma            : (float) momentum coefficient for updating memory bank
                           Default: 0.5.
        tau              : (float) "temperature" for computing probabilities
                           Default: 0.07.
        embedding_dim    : (int) dimension of embedding for memory bank.
                           Default: 128.
    """

    def __init__(
        self,
        memory_bank,
        model_output_dim,
        m=4096,
        gamma=0.5,
        tau=0.07,
        embedding_dim=128,
    ):
        super(InstanceDiscriminationLoss, self).__init__()

        self.memory_bank = memory_bank
        self.m, self.gamma, self.tau = m, gamma, tau

        self._total_samples = self.memory_bank.shape[0]
        self._embedding_func = nn.Linear(model_output_dim, embedding_dim)

    def trainable_parameters(self):
        return self._embedding_func.parameters()

    def update_memory_bank(self, memory_bank):
        assert memory_bank.shape == self.memory_bank.shape
        self.memory_bank = memory_bank

    def _softmax(self, inner_products):
        Z = 2876934.2 / 1281167 * self._total_samples
        probs = torch.exp(inner_products / self.tau) / Z

        return probs

    def _get_new_memory_bank_entries(self, embeddings, indices):
        assert self.memory_bank is not None
        curr_bank_entries = torch.index_select(self.memory_bank, 0, indices)

        assert embeddings.shape == curr_bank_entries.shape
        updated_entries = self.gamma * curr_bank_entries + (1 - self.gamma) * embeddings

        # Remember to normalize L2-norm of entries
        updated_entries = l2_normalize(updated_entries)
        return updated_entries

    def _compute_inner_products(self, embeddings, idxs):
        assert self.memory_bank is not None
        assert idxs.ndim in [1, 2]

        if idxs.ndim == 1:
            with torch.no_grad():
                memory_embeddings = torch.index_select(self.memory_bank, 0, idxs)
                assert memory_embeddings.shape == embeddings.shape
                assert memory_embeddings.ndim == 2
        else:  # idxs.ndim == 2
            with torch.no_grad():
                assert idxs.ndim == 2
                batch_size, m = idxs.shape
                _idxs = idxs.view(-1)
                memory_embeddings = torch.index_select(self.memory_bank, 0, _idxs)
                memory_embeddings = memory_embeddings.view(
                    batch_size, m, self.memory_bank.size(1)
                )

            # Important! Outside of torch.no_grad() space
            embeddings = embeddings.view(batch_size, 1, -1)
            assert embeddings.shape[-1] == memory_embeddings.shape[-1]

        inner_products = torch.mul(embeddings, memory_embeddings)
        assert inner_products.shape == memory_embeddings.shape
        inner_products = torch.sum(inner_products, dim=-1)
        return inner_products

    def _compute_data_probability(self, embeddings, idxs):
        inner_products = self._compute_inner_products(embeddings, idxs)
        data_probs = self._softmax(inner_products)

        return data_probs

    def _compute_noise_probability(self, embeddings, batch_size):
        assert self.memory_bank is not None
        random_idxs = torch.randint(
            0, self._total_samples, (batch_size, self.m), device=embeddings.device
        )
        random_idxs = random_idxs.long()
        inner_products = self._compute_inner_products(embeddings, random_idxs)
        noise_probs = self._softmax(inner_products)

        return noise_probs

    def forward(self, outputs, indices):
        """
        Main entry point for computing the loss value. Computes the instance
        discrimination loss value and also returns the memory bank entries
        that should be updated for the current batch.

        Inputs:
            outputs : (torch.Tensor) outputs of the model
            indices : (torch.Tensor) indices into memory bank of the batch samples

        Outputs:
            loss              : (torch.Tensor) loss value
            entries_to_update : (torch.Tensor) new memory bank entries for current batch
            data_loss         : (torch.Tensor) data part of the loss value
            noise_loss        : (torch.Tensor) noise part of the loss value
        """
        batch_size = outputs.shape[0]
        assert batch_size == indices.shape[0]

        indices = indices.detach()

        # Compute embedding of the current batch
        outputs = self._embedding_func(outputs)

        # Normalize L2-norm of embeddings
        outputs = l2_normalize(outputs)

        # Compute data sample probabilities
        data_probs = self._compute_data_probability(outputs, indices)
        assert data_probs.shape == (batch_size,)

        # Compute noise sample probabilities
        noise_probs = self._compute_noise_probability(outputs, batch_size)
        assert noise_probs.shape == (batch_size, self.m)

        # P_n(i) = 1 / n; Section 3.2 of the paper.
        noise_unif_distr = 1.0 / self._total_samples
        eps = 1e-7  # to avoid numerical issues

        # \log h(i,v) = \log(P(i|v)) - \log[P(i|v) + mP_n(i)]
        data_denom = data_probs + (self.m * noise_unif_distr) + eps
        log_data_prob = torch.log(data_probs) - torch.log(data_denom)

        # \log(1 - h(i,v')) = \log(mP_n(i)) - \log[P(i|v') + mP_n(i)]
        noise_denom = noise_probs + (self.m * noise_unif_distr) + eps
        log_noise_prob = np.log(self.m * noise_unif_distr) - torch.log(noise_denom)

        # J_{NCE}(\theta)
        loss = -(torch.sum(log_data_prob) + torch.sum(log_noise_prob))
        loss = loss / batch_size

        # Now, get the updated memory bank entries
        entries_to_update = self._get_new_memory_bank_entries(outputs, indices)
        assert entries_to_update.shape == outputs.shape

        data_loss = -(torch.sum(log_data_prob)) / batch_size
        noise_loss = -(torch.sum(log_noise_prob)) / batch_size

        return (
            loss.unsqueeze(0),
            entries_to_update,
            data_loss.unsqueeze(0),
            noise_loss.unsqueeze(0),
        )


if __name__ == "__main__":
    from mouse_vision.models.imagenet_models import resnet18

    model = resnet18(drop_final_fc=True)
    mem_bank = (torch.rand(50, 20), torch.rand(50, 20))
    embed_dim = 20

    loss_func = InstanceDiscriminationLoss(mem_bank, 512, m=10, embedding_dim=embed_dim)

    inputs = torch.rand(20, 3, 224, 224)
    loss, update = loss_func(model, inputs, torch.arange(20), torch.arange(1))

    assert update.shape == (20, embed_dim)
    print(update.shape)

    print(list(loss_func.parameters())[0].shape)
    print(list(loss_func.parameters())[0].requires_grad)
