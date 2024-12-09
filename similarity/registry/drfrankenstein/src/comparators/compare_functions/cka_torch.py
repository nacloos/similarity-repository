import torch


def cka(x1, x2):
    x1 = gram_linear(rearrange_activations(x1))
    x2 = gram_linear(rearrange_activations(x2))
    similarity = _cka(x1, x2)
    return similarity


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations


def gram_linear(x):
    return torch.mm(x, x.T)


def center_gram(gram, unbiased=False):
    if not torch.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')

    if unbiased:
        pass
        # TODO
    else:
        means = torch.mean(gram, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram -= torch.unsqueeze(means, len(means.shape))
        gram -= torch.unsqueeze(means, 0)

    return gram


def _cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()
    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)

    return scaled_hsic / (normalization_x * normalization_y)
