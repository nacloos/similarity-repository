import numpy as np


def ps_inv(x1, x2):
    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = np.ones(shape)
    x1_ones[:, :-1] = x1
    A_ones = np.matmul(np.linalg.pinv(x1_ones), x2).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    return {'w': w, 'b': b}


def rearrange_activations(activations):
    is_convolution = len(activations.shape) == 4
    if is_convolution:
        activations = np.transpose(activations, axes=[0, 2, 3, 1])
        n_channels = activations.shape[-1]
        new_shape = (-1, n_channels)
    else:
        new_shape = (activations.shape[0], -1)

    reshaped_activations = activations.reshape(*new_shape)
    return reshaped_activations
