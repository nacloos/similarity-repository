from . import neural_map_base as nm

__all__ = ["FactoredNeuralMap"]

class FactoredNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing a factored readout fitting procedure
    between source and target. Klindt et al. 2017.
    """
    def __init__(self):
        super(FactoredNeuralMap, self).__init__()

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


