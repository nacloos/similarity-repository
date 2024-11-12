"""
Code taken from https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
"""
from functools import partial
import numpy as np

import similarity

similarity.register(
    "measure.kornblith19",
    {
        "paper_id": "kornblith2019",
        "github": "https://github.com/google-research/google-research/blob/master/representation_similarity"
    }
)

register = partial(
  similarity.register,
  function=True,
  preprocessing=[
    "reshape2d",
    "center_columns"
  ]
)


def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)


def cca(features_x, features_y):
  """Compute the mean squared CCA correlation (R^2_{CCA}).

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

  Returns:
    The mean squared CCA correlations between X and Y.
  """
  qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
  qy, _ = np.linalg.qr(features_y)
  return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
      features_x.shape[1], features_y.shape[1])


# register measures (not part of the original code)
# TODO: different ids for feature space and gram matrix cka? (one is faster if num_features < num_examples)
register("measure.kornblith19.cka-hsic_gretton", feature_space_linear_cka)
register("measure.kornblith19.cka-hsic_song", partial(feature_space_linear_cka, debiased=True))
register("measure.kornblith19.cca", cca)
# TODO: nonlinear cka
# register("measure.kornblith19.cka-hsic_gretton-rbf")


if __name__ == "__main__":
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10) + X

    cka_from_examples = cka(gram_linear(X), gram_linear(Y))
    cka_from_features = feature_space_linear_cka(X, Y)

    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)

    rbf_cka = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))
    print('RBF CKA: {:.5f}'.format(rbf_cka))

    cka_from_examples_debiased = cka(gram_linear(X), gram_linear(Y), debiased=True)
    cka_from_features_debiased = feature_space_linear_cka(X, Y, debiased=True)

    print('Linear CKA from Examples (Debiased): {:.5f}'.format(
        cka_from_examples_debiased))
    print('Linear CKA from Features (Debiased): {:.5f}'.format(
        cka_from_features_debiased))

    np.testing.assert_almost_equal(cka_from_examples_debiased,
                                cka_from_features_debiased)

    print('Mean Squared CCA Correlation: {:.5f}'.format(cca(X, Y)))


    transform = np.random.randn(10, 10)
    _, orthogonal_transform = np.linalg.eigh(transform.T.dot(transform))

    # CKA is invariant only to orthogonal transformations.
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y),
        feature_space_linear_cka(X.dot(orthogonal_transform), Y))
    np.testing.assert_(not np.isclose(
        feature_space_linear_cka(X, Y),
        feature_space_linear_cka(X.dot(transform), Y)))

    # CCA is invariant to any invertible linear transform.
    np.testing.assert_almost_equal(cca(X, Y), cca(X.dot(orthogonal_transform), Y))
    np.testing.assert_almost_equal(cca(X, Y), cca(X.dot(transform), Y))

    # Both CCA and CKA are invariant to isotropic scaling.
    np.testing.assert_almost_equal(cca(X, Y), cca(X * 1.337, Y))
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y),
        feature_space_linear_cka(X * 1.337, Y))