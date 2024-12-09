import torch
import torch.nn as nn

from mouse_vision.loss_functions.necks import NonLinearNeckMoCov2
from mouse_vision.loss_functions.heads import ContrastiveHead
from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["MoCov2Loss"]


class MoCov2Loss(LossFunctionBase):
    """MoCov2.
    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Adapted from:
    https://github.com/open-mmlab/OpenSelfSup/blob/5e67129743ef093ffe87999f7953532602917379/openselfsup/models/moco.py#L11-L218
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(
        self,
        encoder_q_backbone,
        encoder_k_backbone,
        model_output_dim,
        hidden_dim=2048,
        pretrained=None,
        queue_len=65536,
        feat_dim=128,
        momentum=0.999,
        temperature=0.2,
        tpu=False,
        **kwargs
    ):
        super(MoCov2Loss, self).__init__()

        self._hidden_dim = hidden_dim
        self.encoder_q_neck = NonLinearNeckMoCov2(
            in_channels=model_output_dim,
            hid_channels=self._hidden_dim,
            out_channels=feat_dim,
        )
        self.encoder_k_neck = NonLinearNeckMoCov2(
            in_channels=model_output_dim,
            hid_channels=self._hidden_dim,
            out_channels=feat_dim,
        )
        self.encoder_q = nn.Sequential(encoder_q_backbone, self.encoder_q_neck)
        self.encoder_k = nn.Sequential(encoder_k_backbone, self.encoder_k_neck)

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = ContrastiveHead(temperature=temperature, return_accs=True)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.tpu = tpu

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print("load model from: {}".format(pretrained))
        self.encoder_q[1].init_weights(init_linear="kaiming")
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys, tpu=self.tpu)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # if self.queue_len % batch_size != 0:
        #    print('QUEUE LEN', self.queue_len, 'BATCH SIZE', batch_size)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, tpu=self.tpu)
        batch_size_all = x_gather.shape[0]

        num_devices = batch_size_all // batch_size_this
        if self.tpu:
            import torch_xla.core.xla_model as xm

            assert num_devices == xm.xrt_world_size()
        else:
            assert num_devices == torch.distributed.get_world_size()

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        if self.tpu:
            import torch_xla.core.xla_model as xm

            if xm.get_ordinal() != 0:
                idx_shuffle = torch.zeros_like(idx_shuffle)

            # broadcast from source 0 the same random value to all tpu cores
            idx_shuffle = xm.all_reduce(xm.REDUCE_SUM, idx_shuffle)
        else:
            torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        if self.tpu:
            import torch_xla.core.xla_model as xm

            rank_idx = xm.get_ordinal()
        else:
            rank_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_devices, -1)[rank_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, tpu=self.tpu)
        batch_size_all = x_gather.shape[0]

        num_devices = batch_size_all // batch_size_this
        if self.tpu:
            import torch_xla.core.xla_model as xm

            assert num_devices == xm.xrt_world_size()
        else:
            assert num_devices == torch.distributed.get_world_size()
        # restored index for this gpu
        if self.tpu:
            rank_idx = xm.get_ordinal()
        else:
            rank_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_devices, -1)[rank_idx]

        return x_gather[idx_this]

    def forward(self, img, mode="train", **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if mode == "train":
                self._momentum_update_key_encoder()  # update the key encoder
            else:
                print("Key encoder momentum update turned off during validation")

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)
        if mode == "train":
            # we don't want the last batch to be from validation set
            # when we start training again after validating
            self._dequeue_and_enqueue(k)
        else:
            print("Queue updating turned off during validation")

        return losses


# utils
@torch.no_grad()
def concat_all_gather(tensor, tpu=False):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if tpu:
        import torch_xla.core.xla_model as xm

        output = xm.all_gather(tensor, dim=0)
    else:
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
    return output
