import numpy as np

from mouse_vision.neural_mappers import PLSNeuralMap
from mouse_vision.neural_mappers import CorrelationNeuralMap
from mouse_vision.neural_mappers import FactoredNeuralMap
from mouse_vision.neural_mappers import PipelineNeuralMap
from mouse_vision.neural_mappers import IdentityNeuralMap

# TODO: In the future, use unittest package

def test_factored():
    f = FactoredNeuralMap()
    assert f._score_func == "corr"

def test_correlation():
    np.random.seed(0)

    # Setup
    source = 30
    target = 20
    n_train = 100
    n_test = 50
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    test_X = np.random.randn(n_test,source)
    test_Y = np.random.randn(n_test,target)
    bad_test_X = np.random.randn(n_test, source+5)

    # Correlation mapper
    corr_map = CorrelationNeuralMap(percentile=100, identity=False)
    corr_map.fit(X, Y)
    preds = corr_map.predict(test_X)
    assert corr_map._score_func == "corr"
    assert preds.shape == test_Y.shape

    # Correlation identity mapper
    corr_map = CorrelationNeuralMap(percentile=100, identity=True)
    corr_map.fit(X, Y)
    preds = corr_map.predict(test_X)
    assert corr_map._score_func == "corr"
    assert preds.shape == test_Y.shape
    assert len(corr_map._mappers) == target
    for i in range(target):
        assert np.array_equal(preds[:,i], test_X[:,corr_map._mappers[i]])

    # This should fail and give an error message
    #corr_map.predict(bad_test_X)

def test_pls():
    np.random.seed(0)

    # Setup
    source = 30
    target = 20
    n_train = 100
    n_test = 50
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    test_X = np.random.randn(n_test,source)
    test_Y = np.random.randn(n_test,target)
    bad_test_X = np.random.randn(n_test, source+5)

    # PLS mapper
    pls_map = PLSNeuralMap(n_components=5, scale=False)
    pls_map.fit(X, Y)
    preds = pls_map.predict(test_X)
    assert preds.shape == test_Y.shape
    assert pls_map._score_func == "corr"

def test_pipeline():
    np.random.seed(0)

    # Setup
    source = 30
    target = 20
    n_train = 100
    n_test = 50
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    test_X = np.random.randn(n_test,source)
    test_Y = np.random.randn(n_test,target)
    bad_test_X = np.random.randn(n_test, source+5)

    # Correlation mapper
    corr_map = CorrelationNeuralMap(percentile=100, identity=False)
    corr_map.fit(X, Y)
    preds = corr_map.predict(test_X)
    assert preds.shape == test_Y.shape
    corr_map_pipeline = PipelineNeuralMap(map_type="corr", map_kwargs={'percentile':100, 'identity':False})
    corr_map_pipeline.fit(X, Y)
    preds_pipeline = corr_map_pipeline.predict(test_X)
    assert np.array_equal(preds_pipeline, preds) is True
    assert corr_map_pipeline._map_func._score_func == "corr"

    # Correlation identity mapper
    corr_map = CorrelationNeuralMap(percentile=100, identity=True)
    corr_map.fit(X, Y)
    preds = corr_map.predict(test_X)
    assert preds.shape == test_Y.shape
    assert len(corr_map._mappers) == target
    for i in range(target):
        assert np.array_equal(preds[:,i], test_X[:,corr_map._mappers[i]])

    corr_map_pipeline = PipelineNeuralMap(map_type="corr", map_kwargs={'percentile':100, 'identity':True})
    corr_map_pipeline.fit(X, Y)
    preds_pipeline = corr_map_pipeline.predict(test_X)
    assert np.array_equal(preds_pipeline, preds) is True
    assert corr_map_pipeline._map_func._score_func == "corr"

    # This should fail and give an error message
    #corr_map.predict(bad_test_X)

    # Setup
    source = 30
    target = 20
    n_train = 100
    n_test = 50
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    test_X = np.random.randn(n_test,source)
    test_Y = np.random.randn(n_test,target)
    bad_test_X = np.random.randn(n_test, source+5)

    # PLS mapper
    pls_map = PLSNeuralMap(n_components=5, scale=False)
    pls_map.fit(X, Y)
    preds = pls_map.predict(test_X)
    assert preds.shape == test_Y.shape
    pls_map_pipeline = PipelineNeuralMap(map_type="pls", map_kwargs={'n_components': 5, 'scale': False})
    pls_map_pipeline.fit(X, Y)
    preds_pipeline = pls_map_pipeline.predict(test_X)
    assert np.array_equal(preds_pipeline, preds) is True
    assert pls_map_pipeline._map_func._score_func == "corr"

    # Identity mapper
    identity_map_pipeline = PipelineNeuralMap(map_type="identity")
    identity_map_pipeline.fit(X, Y)
    preds = identity_map_pipeline.predict(test_X)
    assert np.array_equal(preds, test_X)
    assert identity_map_pipeline._map_func._score_func == "rsa"

def test_identity():
    # Setup
    source = 30
    target = 20
    n_train = 100
    n_test = 50
    X = np.random.randn(n_train,source)
    Y = np.random.randn(n_train,target)
    test_X = np.random.randn(n_test,source)
    test_Y = np.random.randn(n_test,target)

    identity_map = IdentityNeuralMap()
    identity_map.fit(X, Y)
    preds = identity_map.predict(test_X)
    score = identity_map.score(preds, test_X)
    assert score == 1.0
    assert np.array_equal(preds, test_X)
    assert identity_map._score_func == "rsa"


if __name__ == "__main__":
    print("Testing factored")
    test_factored()

    print("Testing correlation")
    test_correlation()

    print("Testing pls")
    test_pls()

    print("Testing pipeline")
    test_pipeline()

    print("Testing identity")
    test_identity()

