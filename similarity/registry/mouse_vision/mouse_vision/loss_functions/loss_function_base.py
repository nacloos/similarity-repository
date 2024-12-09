import torch.nn as nn

__all__ = ["LossFunctionBase"]


class LossFunctionBase(nn.Module):
    def __init__(self):
        super(LossFunctionBase, self).__init__()

    def trainable_parameters(self):
        """
        Should return an iterable for parameters to be trained in the
        loss function. For example, could return self.model.parameters().
        """
        raise NotImplementedError

    def forward(self, model, inp, target, **kwargs):
        """
        Computes the loss function given a model, inputs and labels.

        Inputs:
            model  : (torch.nn.Module) the model being trained
            inp    : (torch.FloatTensor) (N, C, H, W); input images
            target : (torch.LongTensor) (N,); image labels

        Outputs:
            loss   : (torch.Tensor) scalar; loss value
            preds  : (torch.Tensor) (N, K); model predictions for K classes
        """
        raise NotImplementedError
