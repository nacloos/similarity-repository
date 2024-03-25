from functools import partial
import similarity


transforms = [
    {"inp": "cka", "out": "cka-angular", "postprocessing": ["arccos"]},
    {"inp": "cca", "out": "cca-angular", "postprocessing": ["arccos"]},
    {"inp": "nbs", "out": "procrustes-angular", "postprocessing": ["arccos"]},
    {"inp": "procrustes-euclidean", "out": "procrustes-sq-euclidean", "postprocessing": ["square"]},
    {
        "inp": "procrustes-euclidean",
        "out": "procrustes-angular",
        "postprocessing": [
            {"id": "euclidean_to_angular_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    },
    {
        "inp": "procrustes-angular",
        "out": "procrustes-euclidean",
        "postprocessing": [
            {"id": "angular_to_euclidean_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    },
    # aliases
    {"inp": "bures_distance", "out": "procrustes-euclidean", "postprocessing": []},
    {"inp": "procrustes-euclidean", "out": "bures_distance", "postprocessing": []},
    # cka usually refers to the Gretton2007 estimate of HSIC
    {"inp": "cka", "out": "cka-hsic_gretton", "postprocessing": []},
    {"inp": "cka-angular", "out": "cka-hsic_gretton-angular", "postprocessing": []},
    {"inp": "cka-angular-score", "out": "cka-hsic_gretton-angular-score", "postprocessing": []},

]


def add_inverse_transforms(transforms, inverse_functions):
    """
    Add inverse transforms to the list of transforms.
    For example, if there is a transform from "a" to "b" with postprocessing "cos", then add a transform from "b" to "a" with postprocessing "arccos".
    """
    new_transforms = []
    for transform in transforms:
        if not len(transform["postprocessing"]) == 1:
            continue

        postprocessing = transform["postprocessing"][0]
        if isinstance(postprocessing, dict):
            continue

        if postprocessing in inverse_functions:
            new_transform = {
                "inp": transform["out"],
                "out": transform["inp"],
                "postprocessing": [
                    inverse_functions[postprocessing]
                ]
            }
            new_transforms.append(new_transform)
    return transforms + new_transforms


inverse_functions = {
    "cos": "arccos",
    "arccos": "cos",
    "one_minus": "one_minus",
    "square": "sqrt",
    "sqrt": "square",
}
transforms = add_inverse_transforms(transforms, inverse_functions)


def register_derived_measures(transform):
    matches = similarity.match(f"measure.*.{transform['inp']}")

    # all the derived measures that are not already registered
    base_measure_ids = []  # match measure id (with transform["inp"])
    derived_measure_ids = []  # replace transforms["inp"] by transforms["out"]
    for match in matches:
        assert len(match.split(".")) == 3, f"Expected 3 parts in id: '{{category}}.{{backend}}.{{measure}}', but got {match}"

        backend = match.split(".")[1]
        derived_measure_id = f"measure.{backend}.{transform['out']}"

        if similarity.is_registered(derived_measure_id):
            continue
        print("found match:", match, derived_measure_id)
        derived_measure_ids.append(derived_measure_id)
        base_measure_ids.append(match)

    # register the derived measures
    for base_measure_id, derived_measure_id in zip(base_measure_ids, derived_measure_ids):
        def derived_measure(measure_id, postprocessing):
            measure = similarity.make(measure_id)
            # similarity.make returns a MeasureInterface object
            assert isinstance(measure, similarity.MeasureInterface), f"Expected type MeasureInterface, but got {type(measure)}"
            # here recreate a new MeasureInterface object with the same underlying measure, but with the new postprocessing
            _measure = similarity.MeasureInterface(
                measure=measure.measure,
                interface=measure.interface,
                preprocessing=measure.preprocessing,
                postprocessing=measure.postprocessing + postprocessing,
            )
            return _measure

        # not working as expected if directly use register as decorator and don't pass the match and transform as arguments
        similarity.register(
            derived_measure_id,
            partial(derived_measure, base_measure_id, transform["postprocessing"])
        )
        if transform["inp"] == "procrustes-euclidean" and transform["out"] == "procrustes-angular":
            print("Registered derived measure:", derived_measure_id, transform)

    return derived_measure_ids

# recursively register derived measures until no more derived measures are registered
done = False
while not done:
    for transform in transforms:
        if transform["inp"] == "procrustes-euclidean" and transform["out"] == "procrustes-angular":
            print(f"Registering derived measures for transform: {transform}")
        registered_ids = register_derived_measures(transform)
        done = len(registered_ids) == 0



import numpy as np
X = np.random.rand(15, 20, 30)
Y = np.random.rand(15, 20, 30)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
X = X - X.mean(axis=0)
Y = Y - Y.mean(axis=0)

measures = similarity.make("measure.mklabunde.*")
proc = similarity.make("measure.netrep.procrustes-angular")

print("ref", proc(X, Y))
from similarity.backend.mklabunde import procrustes
from similarity.processing import euclidean_to_angular_shape_metric

score = procrustes(X, Y)
print("Score", score)
print("Sqrt:", np.sqrt(score))
print("angular", euclidean_to_angular_shape_metric(X, Y, np.sqrt(score)))


print(measures.keys())
for k, v in measures.items():
    print(k)
    print(v.measure)
    print(v.interface)
    print(v.preprocessing)
    print(v.postprocessing)
    print(v(X, Y))
    print("=====")
