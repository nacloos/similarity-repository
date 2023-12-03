import numpy as np

from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.rdm import RDMCrossValidated
from brainio.assemblies import NeuroidAssembly



# rdm_measure = RDMCrossValidated()  # TODO: not working


def numpy_to_brainio(X, Y):
    """
    BrainScore measure takes NeuroidAssembly objects
    """
    def _convert(X):
        assert len(X.shape) == 2, X.shape
        return NeuroidAssembly(
            X, 
            # filled with dummy values
            coords={
                'stimulus_id': ('presentation', np.arange(X.shape[0])),
                'object_name': ('presentation', ['']*X.shape[0]),
                'neuroid_id': ('neuroid', np.arange(X.shape[1])),
                'region': ('neuroid', [0] * X.shape[1])
            },
            dims=['presentation', 'neuroid']
        )
    return _convert(X), _convert(Y)


def aggregate_score(score):
    if "aggregation" in score.coords:
        return score.sel(aggregation='center').values.item()
    else:
        return score.values.item()


if __name__ == "__main__":
    import similarity
    from omegaconf import OmegaConf
    import json
    cards = similarity.make("", package="backend:backends", key="cards")
    # print(json.dumps(cards, indent=2))
    print(OmegaConf.to_yaml(cards))

    # X = np.random.randn(100, 50)
    # Y = np.random.randn(100, 50)
    # X, Y = numpy_to_brainio(X, Y)

    # measure = pls_measure()
    # score = measure(source=X, target=Y)
    # score = score.sel(aggregation='center')
    # print(score)


    # measure = similarity.make("backend/brainscore/measure.cka", return_config=True)
    # print(json.dumps(measure, indent=2))

    measure = similarity.make("backend/brainscore/measure.pls")
    print(measure)

    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    print(measure.fit_score(X, Y))



# assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.randn(30 * 25)).reshape((30, 25)),
#                            coords={'stimulus_id': ('presentation', np.arange(30)),
#                                    'object_name': ('presentation', ['a', 'b', 'c'] * 10),
#                                    'neuroid_id': ('neuroid', np.arange(25)),
#                                    'region': ('neuroid', [0] * 25)
#                                 },
#                            dims=['presentation', 'neuroid'])
# # assembly = numpy_to_brainio((np.arange(30 * 25) + np.random.randn(30 * 25)).reshape((30, 25)))

# score = rdm_measure(assembly, assembly)  # TODO: not working

# prediction, target = assembly, assembly  # we're testing how well the measure can predict the dataset itself
# score = measure(source=prediction, target=target)
# print(score)
# print(score.raw)


class BrainScoremeasure:
    def __init__(self, symmeasure=False, arccos=True):
        super().__init__()
        self.symmeasure = symmeasure
        self.arccos = arccos

    def fit(self, act1, act2):
        X1 = act1.reshape(act1.shape[0]*act1.shape[1], -1)
        X2 = act2.reshape(act2.shape[0]*act2.shape[1], -1)

        X1 = numpy_to_brainio(X1)
        X2 = numpy_to_brainio(X2)
        # predict X1 (neura data) from X2 (model activity)
        score = measure(source=X2, target=X1)
        print(score)
        print(np.mean(score.raw))
        # self.r2 = np.mean(score.raw)
        self.r2 = score.sel(aggregation='center')

    def eval(self, act1, act2):
        self.fit(act1, act2)
        if self.symmeasure:
            first_r2 = self.r2
            self.fit(act2, act1)
            assert first_r2 != self.r2
            self.r2 = (first_r2 + self.r2) / 2

        return self.r2 if not self.arccos else np.arccos(self.r2)

    def __str__(self):
        return "BrainScore distance"


class BrainScoreRDMmeasure:
    def __init__(self, symmeasure=False, arccos=True):
        super().__init__()
        self.symmeasure = symmeasure
        self.arccos = arccos

    def fit(self, act1, act2):
        X1 = act1.reshape(act1.shape[0]*act1.shape[1], -1)
        X2 = act2.reshape(act2.shape[0]*act2.shape[1], -1)

        X1 = numpy_to_brainio(X1)
        X2 = numpy_to_brainio(X2)
        score = rdm_measure(X1, X2)
        print(score)
        print(np.mean(score.raw))
        # self.r2 = np.mean(score.raw)
        self.r2 = score.sel(aggregation='center')

    def eval(self, act1, act2):
        self.fit(act1, act2)
        if self.symmeasure:
            first_r2 = self.r2
            self.fit(act2, act1)
            assert first_r2 != self.r2
            self.r2 = (first_r2 + self.r2) / 2

        return self.r2 if not self.arccos else np.arccos(self.r2)



