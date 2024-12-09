import numpy as np


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.reshape(batch_size, -1)
    return flat_activations


def lr(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)
    q2, r2 = np.linalg.qr(x2_flat)
    return (np.linalg.norm(q2.T @ x1_flat)) ** 2 / (np.linalg.norm(x1_flat)) ** 2
