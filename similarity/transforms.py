from functools import partial
import similarity


transforms = [
    {"inp": "cka", "out": "cka-angular", "postprocessing": ["arccos"]},
    {"inp": "cca", "out": "cca-angular", "postprocessing": ["arccos"]},
    {"inp": "nbs", "out": "procrustes-angular", "postprocessing": ["arccos"]},
    # TODO: require X and Y as inputs (not just the score)
    # {"inp": "procrustes-euclidean", "out": "procrustes-angular", "postprocessing": ["euclidean_to_angular_shape_metric"]},
    # {"inp": "procrustes-euclidean", "out": "procrustes-sq-euclidean", "postprocessing": ["square"]},
    # aliases
    {"inp": "bures_distance", "out": "procrustes-euclidean", "postprocessing": []},
    {"inp": "procrustes-euclidean", "out": "bures_distance", "postprocessing": []},
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
    "euclidean_to_angular_shape_metric": "angular_to_euclidean_shape_metric",
    "angular_to_euclidean_shape_metric": "euclidean_to_angular_shape_metric",
}
transforms = add_inverse_transforms(transforms, inverse_functions)


def register_derived_measures(transform):
    matches = similarity.match(f"measure.*.{transform['inp']}")

    # all the derived measures that are not already registered
    derived_measure_ids = []
    for match in matches:
        assert len(match.split(".")) == 3, f"Expected 3 parts in id: '{{category}}.{{backend}}.{{measure}}', but got {match}"
        backend = match.split(".")[1]
        derived_measure_id = f"measure.{backend}.{transform['out']}"

        if similarity.is_registered(derived_measure_id):
            continue

        derived_measure_ids.append(derived_measure_id)

    # register the derived measures
    for derived_measure_id in derived_measure_ids:
        def derived_measure(measure_id, postprocessing):
            measure = similarity.make(measure_id)
            # similarity.make returns a MeasureInterface object
            assert isinstance(measure, similarity.MeasureInteface), f"Expected type MeasureInterface, but got {type(measure)}"
            # here recreate a new MeasureInterface object with the same underlying measure, but with the new postprocessing
            _measure = similarity.MeasureInteface(
                measure=measure.measure,
                interface=measure.interface,
                preprocessing=measure.preprocessing,
                postprocessing=measure.postprocessing + postprocessing,
            )
            return _measure

        # not working as expected if directly use register as decorator and don't pass the match and transform as arguments
        similarity.register(
            derived_measure_id,
            partial(derived_measure, match, transform["postprocessing"])
        )

    return derived_measure_ids


# recursively register derived measures until no more derived measures are registered
done = False
while not done:
    for transform in transforms:
        registered_ids = register_derived_measures(transform)
        done = len(registered_ids) == 0
