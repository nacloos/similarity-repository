"""
Automatic derivation of new similarity measures based on existing ones. For
example, if there is a measure "procrustes-euclidean" that computes the
Procrustes distance between two matrices, then we can automatically derive a
new measure "procrustes-sq-euclidean" that computes the square of the
Procrustes distance by adding a new transform to the list of transforms,
"""
from functools import partial
import similarity


# list of transforms used to automatically derive new measures
transforms = [
    {"inp": "procrustes-euclidean", "out": "procrustes-sq-euclidean", "postprocessing": ["square"]},

    # Generalized Shape Metrics on Neural Representations (Williams et al., 2021)
    # take the arccosine to get angular distance
    {"inp": "cka", "out": "cka-angular", "postprocessing": ["arccos"]},
    {"inp": "cca", "out": "cca-angular", "postprocessing": ["arccos"]},

    # svcca = pca + cca
    {"inp": "cca", "out": "svcca-var95", "preprocessing": ["reshape2d", "pca-var95"]},
    {"inp": "cca", "out": "svcca-var99", "preprocessing": ["reshape2d", "pca-var99"]},

    # transformation between angular and euclidean shape metrics
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

    # Duality of Bures and Shape Distances with Implications for Comparing Neural Representations (Harvey et al., 2023)
    {"inp": "nbs", "out": "procrustes-angular", "postprocessing": ["arccos"]},
    {"inp": "bures_distance", "out": "procrustes-euclidean", "postprocessing": []},
    {"inp": "procrustes-euclidean", "out": "bures_distance", "postprocessing": []},

    # CKA by default refers to the Gretton2007 estimate of HSIC
    {"inp": "cka", "out": "cka-hsic_gretton", "postprocessing": []},
    {"inp": "cka-angular", "out": "cka-hsic_gretton-angular", "postprocessing": []},
    {"inp": "cka-angular-score", "out": "cka-hsic_gretton-angular-score", "postprocessing": []},

    # rescale angular distances to scores in [0, 1]
    {"inp": "procrustes-angular", "out": "procrustes-angular-score", "postprocessing": ["normalize_pi_half", "one_minus"]},
    {"inp": "cka-angular", "out": "cka-angular-score", "postprocessing": ["normalize_pi_half", "one_minus"]},
    {"inp": "cca-angular", "out": "cca-angular-score", "postprocessing": ["normalize_pi_half", "one_minus"]},
    {"inp": "nbs-angular", "out": "nbs-angular-score", "postprocessing": ["normalize_pi_half", "one_minus"]},
]

# inverse functions used to automatically add inverse transforms to the list of transforms
inverse_functions = {
    "cos": "arccos",
    "arccos": "cos",
    "one_minus": "one_minus",
    "square": "sqrt",
    "sqrt": "square",
}


def register_derived_measures(transform):
    """
    Register derived measures based on a transform.

    Args:
        transform: dictionary with keys "inp", "out", and "postprocessing".
            "inp": input measure id.
            "out": output measure id.
            "postprocessing": list of postprocessing functions.

    Returns:
        derived_measure_ids: list of derived measure ids that were registered.
    """
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
        derived_measure_ids.append(derived_measure_id)
        base_measure_ids.append(match)

    # register the derived measures
    for base_measure_id, derived_measure_id in zip(base_measure_ids, derived_measure_ids):
        def derived_measure(measure_id, postprocessing, preprocessing):
            measure = similarity.make(measure_id)
            # similarity.make returns a MeasureInterface object
            assert isinstance(measure, similarity.MeasureInterface), f"Expected type MeasureInterface, but got {type(measure)}"
            # here recreate a new MeasureInterface object with the same underlying measure, but with the new postprocessing
            _measure = similarity.MeasureInterface(
                measure=measure.measure,
                interface=measure.interface,
                preprocessing=preprocessing + measure.preprocessing,
                postprocessing=measure.postprocessing + postprocessing,
            )
            return _measure

        # not working as expected if directly use register as decorator and don't pass the match and transform as arguments
        similarity.register(
            derived_measure_id,
            partial(
                derived_measure,
                base_measure_id,
                transform.get("postprocessing", []),
                transform.get("preprocessing", [])
            )
        )

    return derived_measure_ids


def add_inverse_transforms(transforms, inverse_functions):
    """
    Add inverse transforms to the list of transforms. For example, if there is a transform from "a" to "b" with postprocessing "cos", then add a transform from "b" to "a" with postprocessing "arccos".

    Args:
        transforms: list of transforms.
        inverse_functions: dictionary mapping postprocessing functions to their inverses.

    Returns:
        new_transforms: list of transforms with added inverse transforms.
    """
    new_transforms = []
    for transform in transforms:
        # only consider transforms with exactly one postprocessing function
        if "postprocessing" not in transform or len(transform["postprocessing"]) != 1:
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


# add inverse transforms to the list of transforms
transforms = add_inverse_transforms(transforms, inverse_functions)

# recursively register derived measures until no more derived measures are registered
done = False
while not done:
    for transform in transforms:
        registered_ids = register_derived_measures(transform)
        done = len(registered_ids) == 0
