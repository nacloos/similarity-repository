from .diffscore.analysis.similarity_measures import measures

import similarity


for k, v in measures.items():
    # TODO
    if not ("measure/" in k):
        continue

    # k: "measure/{name}"
    similarity.register(
        f"diffscore/{k.split('/')[1]}",
        v,
        preprocessing=["array_to_tensor"],
        postprocessing=["tensor_to_float"]
    )
