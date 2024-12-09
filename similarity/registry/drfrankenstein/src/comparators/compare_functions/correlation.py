import numpy as np


def correlation(x, y):
    x = rearrange_activations(x)
    y = rearrange_activations(y)

    # Subtruct means
    x_norm = x - x.mean(axis=1)[:, None]
    y_norm = y - y.mean(axis=1)[:, None]

    # Sum of squares
    ss_x = (x_norm ** 2).sum(axis=1)
    ss_y = (y_norm ** 2).sum(axis=1)

    # Final form
    # (x-E[x]) * (y-E[y])
    # -------------------
    #  sqrt(E[x] * E[y])
    numerator = np.dot(x_norm, y_norm.T)
    denominator = np.sqrt(np.dot(ss_x[:, None], ss_y[None]))
    correlation = numerator / denominator

    return correlation.T


def rearrange_activations(activations):
    is_convolution = len(activations.shape) == 4
    permutation = [1, 2, 3, 0] if is_convolution else [1, 0]
    activations = np.transpose(activations, axes=permutation)
    new_shape = (activations.shape[0], -1)
    reshaped_activations = activations.reshape(*new_shape)
    return reshaped_activations


if __name__ == '__main__':
    # Shape
    batch_size = 100
    n_channels = 64
    spatial_size = (5, 7)
    shape = (batch_size, n_channels, *spatial_size)

    a = np.random.random(shape)

    # Corr 1
    b = a * 3 - 10
    corr = correlation(a, b)
    print(f"1 corr: {corr.shape} {np.diag(corr).mean()}")

    # Corr 0
    b = np.random.random(shape)
    corr = correlation(a, b)
    print(f"0 corr: {corr.shape} {np.diag(corr).mean()}")

    # Corr -1
    b = -20 * a + 103
    corr = correlation(a, b)
    print(f"-1 corr: {corr.shape} {np.diag(corr).mean()}")
