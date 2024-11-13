"""
https://github.com/google-research/google-research/blob/master/representation_similarity
"""
from functools import partial
import numpy as np

from .demo import *

import similarity

similarity.register(
    "measure/kornblith19",
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


# register measures (not part of the original code)
# TODO: different ids for feature space and gram matrix cka? (one is faster if num_features < num_examples)
register("measure/kornblith19/cka-hsic_gretton", feature_space_linear_cka)
register("measure/kornblith19/cka-hsic_song", partial(feature_space_linear_cka, debiased=True))
register("measure/kornblith19/cca", cca)
# TODO: nonlinear cka
# register("measure/kornblith19/cka-hsic_gretton-rbf")


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