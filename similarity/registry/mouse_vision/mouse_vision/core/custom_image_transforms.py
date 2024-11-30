import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GaussianBlur"]


def _get_gaussian_kernel1d(kernel_size, sigma):
    """
    Computes the one-dimensional Gaussian.

    Inputs:
        kernel_size : (int) height or width of kernel
        sigma       : (float) sigma for the height or width of kernel

    Outputs:
        kernel1d    : (torch.Tensor) for the one-dimensional Gaussian kernel
    """
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
    """
    Computes outer-product of two one-dimensional Gaussians to get the two-dimensional
    Gaussian kernel

    Inputs:
        kernel_size : (list of int) kernel size of the Gaussian kernel (kx, ky)
        sigma       : (list of float) standard deviation of the Gaussian kernel (sx, sy)

    Outputs:
        kernel2d    : (torch.Tensor) for the two-dimensional Gaussian kernel
    """
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(
        device, dtype=dtype
    )
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(
        device, dtype=dtype
    )
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])  # outer product

    return kernel2d


def _gaussian_blur(img, kernel_size, sigma):
    """
    Inputs:
        img         : (torch.Tensor) image to be blurred
        kernel_size : (list of int) kernel size of the Gaussian kernel (kx, ky)
        sigma       : (list of float) standard deviation of the Gaussian kernel (sx, sy)

    Outputs:
        img         : (torch.Tensor) Gaussian blurred image
    """
    n_channel = img.shape[0]
    orig_height = img.shape[1]
    orig_width = img.shape[2]
    img = img.unsqueeze(
        0
    )  # unsqueeze first dimension since reflection pad only works for 4D tensors

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    img = F.pad(img, padding, mode="reflect")

    new_height = orig_height + (2 * (kernel_size[1] // 2))
    new_width = orig_width + (2 * (kernel_size[0] // 2))
    assert img.shape == (1, n_channel, new_height, new_width)
    img = F.conv2d(img, kernel, groups=img.shape[-3])

    img = img.squeeze(0)
    return img


class GaussianBlur(nn.Module):
    """
    Class for performing Gaussian blur.

    Arguments:
        kernel_size : (int) size of kernel, assumes square kernel
        sigma       : (float) size of standard deviation for Gaussian
    """

    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        assert isinstance(kernel_size, numbers.Number)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"Kernel size value should be an odd and "
                f"positive number. Given {kernel_size}."
            )
        self.kernel_size = (kernel_size, kernel_size)

        assert isinstance(sigma, numbers.Number)
        if sigma <= 0:
            raise ValueError("If sigma is a single number, it must be positive.")
        self.sigma = (sigma, sigma)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        blur_x = _gaussian_blur(x, self.kernel_size, self.sigma)
        return blur_x
